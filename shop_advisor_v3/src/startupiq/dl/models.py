"""Deep forecasters with three quantile outputs (P10, P50, P90) per horizon step (Task 7).

All three share the decoder: for each step h, [encoding of the past, known future features
of step h, static features (+ optional series embedding)] -> P50 and two positive offsets,
so P10 <= P50 <= P90 always (no quantile crossing).

    MLP    the basic neural model: the flattened past window, no notion of order
    LSTM   recurrent encoder; optional attention over the encoder steps, whose weights are
           reported for interpretability (Task 8)
    TCN    1D convolutional encoder: causal dilated convolutions with residual connections
           (dilations 1, 2, 4, ...; 6 blocks give a 127-step receptive field); its feature
           maps are inspected in Task 8

Defaults are the weekly panel's input sizes; the Rossmann daily experiment passes its own.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

PAST_DIM, FUTURE_DIM, STATIC_DIM = 3, 4, 1  # weekly panel (dl/windows.py)
QUANTILES = (0.1, 0.5, 0.9)


class Decoder(nn.Module):
    def __init__(self, enc_dim: int, hidden: int, dropout: float, future_dim: int, static_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(enc_dim + future_dim + static_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, 3))

    def forward(self, enc: torch.Tensor, future: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        # enc [B, E] or [B, H, E]; future [B, H, F]; static [B, S] -> [B, H, 3] quantiles
        h = future.shape[1]
        if enc.dim() == 2:
            enc = enc.unsqueeze(1).expand(-1, h, -1)
        out = self.net(torch.cat([enc, future, static.unsqueeze(1).expand(-1, h, -1)], dim=-1))
        mid = out[..., 1]
        return torch.stack([mid - F.softplus(out[..., 0]), mid, mid + F.softplus(out[..., 2])], dim=-1)


class Base(nn.Module):
    """Shared input handling: optional embedding of the series id appended to the static features."""

    def __init__(self, past_dim, future_dim, static_dim, n_ids, emb):
        super().__init__()
        self.past_dim, self.future_dim = past_dim, future_dim
        self.embed = nn.Embedding(n_ids, emb) if n_ids else None
        self.static_dim = static_dim + (emb if n_ids else 0)

    def _static(self, static, ids):
        return torch.cat([static, self.embed(ids)], dim=-1) if self.embed is not None else static


class MLPForecaster(Base):
    kind = "mlp"

    def __init__(self, lookback: int, hidden: int = 64, dropout: float = 0.1, past_dim=PAST_DIM, future_dim=FUTURE_DIM,
                 static_dim=STATIC_DIM, n_ids=0, emb=8, **_):
        super().__init__(past_dim, future_dim, static_dim, n_ids, emb)
        self.enc = nn.Sequential(nn.Linear(lookback * past_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ReLU())
        self.dec = Decoder(hidden, hidden, dropout, future_dim, self.static_dim)

    def forward(self, past, future, static, ids=None):
        return self.dec(self.enc(past.flatten(1)), future, self._static(static, ids))


class LSTMForecaster(Base):
    kind = "lstm"

    def __init__(self, lookback: int, hidden: int = 64, layers: int = 1, dropout: float = 0.1, attention: bool = True,
                 past_dim=PAST_DIM, future_dim=FUTURE_DIM, static_dim=STATIC_DIM, n_ids=0, emb=8, **_):
        super().__init__(past_dim, future_dim, static_dim, n_ids, emb)
        self.lstm = nn.LSTM(past_dim, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.attention = attention
        if attention:
            self.query = nn.Linear(hidden + future_dim, hidden)
        self.drop = nn.Dropout(dropout)
        self.dec = Decoder(hidden * (2 if attention else 1), hidden, dropout, future_dim, self.static_dim)
        self.last_attention: torch.Tensor | None = None

    def forward(self, past, future, static, ids=None):
        static = self._static(static, ids)
        outs, (h_n, _) = self.lstm(past)              # outs [B, L, D]
        last = self.drop(h_n[-1])                      # [B, D]
        if not self.attention:
            return self.dec(last, future, static)
        h = future.shape[1]
        q = self.query(torch.cat([last.unsqueeze(1).expand(-1, h, -1), future], dim=-1))     # [B, H, D]
        scores = torch.einsum("bhd,bld->bhl", q, outs) / outs.shape[-1] ** 0.5
        weights = torch.softmax(scores, dim=-1)        # [B, H, L]: which past steps each forecast step looks at
        self.last_attention = weights.detach()
        context = torch.einsum("bhl,bld->bhd", weights, outs)
        enc = torch.cat([last.unsqueeze(1).expand(-1, h, -1), context], dim=-1)
        return self.dec(enc, future, static)


class CausalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float, kernel: int = 3):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel, dilation=dilation)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.drop(F.relu(self.conv1(F.pad(x, (self.pad, 0)))))
        y = self.drop(F.relu(self.conv2(F.pad(y, (self.pad, 0)))))
        return F.relu(x + y)


class TCNForecaster(Base):
    kind = "tcn"

    def __init__(self, lookback: int, hidden: int = 32, layers: int = 6, dropout: float = 0.1, past_dim=PAST_DIM,
                 future_dim=FUTURE_DIM, static_dim=STATIC_DIM, n_ids=0, emb=8, **_):
        super().__init__(past_dim, future_dim, static_dim, n_ids, emb)
        self.inp = nn.Conv1d(past_dim, hidden, 1)
        self.blocks = nn.ModuleList([CausalBlock(hidden, 2 ** i, dropout) for i in range(layers)])
        self.dec = Decoder(2 * hidden, hidden, dropout, future_dim, self.static_dim)
        self.feature_maps: list[torch.Tensor] = []

    def forward(self, past, future, static, ids=None, keep_maps: bool = False):
        x = self.inp(past.transpose(1, 2))             # [B, C, L]
        maps = [x]
        for b in self.blocks:
            x = b(x)
            maps.append(x)
        if keep_maps:
            self.feature_maps = [m.detach() for m in maps]
        enc = torch.cat([x[..., -1], x.mean(-1)], dim=-1)   # last step + average over the window
        return self.dec(enc, future, self._static(static, ids))


ARCHS = {"mlp": MLPForecaster, "lstm": LSTMForecaster, "tcn": TCNForecaster}


def pinball(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean quantile (pinball) loss over P10, P50, P90 and the horizon steps where mask is true."""
    q = torch.tensor(QUANTILES, device=pred.device)
    diff = target.unsqueeze(-1) - pred
    loss = torch.maximum(q * diff, (q - 1) * diff).mean(-1)   # [B, H]
    if mask is None:
        return loss.mean()
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
