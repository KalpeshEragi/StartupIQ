"""Three neural forecasters with the same inputs and outputs, so the comparison is fair.

Each takes the 52 past weeks (6 features each), the 12 future weeks' calendar (4 features each) and the sector id,
and returns P10 / P50 / P90 for each of the 12 weeks. P10 <= P50 <= P90 is guaranteed by predicting P50 and two
positive gaps.

  - MLP:  flattens everything into one vector. Simplest; a strong baseline for short histories.
  - CNN:  dilated 1D convolutions over the past weeks (a small temporal convolutional network). Finds repeating
          local shapes such as a festive ramp-up.
  - LSTM: reads the past weeks in order with a memory, plus attention over all past weeks. The attention weights
          show which past weeks drove a forecast.
"""
from __future__ import annotations

import torch
from torch import nn

from market.dataset import H, L

P_FEATS, F_FEATS = 6, 4
QUANTILES = (0.1, 0.5, 0.9)


class Head(nn.Module):
    """Turns a feature vector into monotone quantiles, shape (batch, H, 3)."""

    def __init__(self, d_in: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, H * 3))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        o = self.net(z).view(-1, H, 3)
        mid = o[..., 1]
        lo = mid - nn.functional.softplus(o[..., 0])
        hi = mid + nn.functional.softplus(o[..., 2])
        return torch.stack([lo, mid, hi], dim=-1)


class MLP(nn.Module):
    def __init__(self, n_sectors: int, hidden: int = 256, dropout: float = 0.2, emb: int = 8):
        super().__init__()
        self.emb = nn.Embedding(n_sectors, emb)
        d = L * P_FEATS + H * F_FEATS + emb
        self.body = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout))
        self.head = Head(hidden, hidden, dropout)

    def forward(self, xp, xf, sid):
        z = torch.cat([xp.flatten(1), xf.flatten(1), self.emb(sid)], dim=1)
        return self.head(self.body(z))


class CNN(nn.Module):
    def __init__(self, n_sectors: int, hidden: int = 64, dropout: float = 0.2, emb: int = 8):
        super().__init__()
        self.emb = nn.Embedding(n_sectors, emb)
        layers, c = [], P_FEATS
        for dil in (1, 2, 4, 8, 16):
            layers += [nn.Conv1d(c, hidden, kernel_size=3, dilation=dil, padding=dil), nn.GELU(), nn.Dropout(dropout)]
            c = hidden
        self.conv = nn.Sequential(*layers)
        self.head = Head(hidden * 2 + H * F_FEATS + emb, hidden * 2, dropout)

    def forward(self, xp, xf, sid):
        h = self.conv(xp.transpose(1, 2))                      # (batch, hidden, L)
        z = torch.cat([h[..., -1], h.mean(-1), xf.flatten(1), self.emb(sid)], dim=1)
        return self.head(z)


class LSTM(nn.Module):
    def __init__(self, n_sectors: int, hidden: int = 64, dropout: float = 0.2, emb: int = 8):
        super().__init__()
        self.emb = nn.Embedding(n_sectors, emb)
        self.rnn = nn.LSTM(P_FEATS, hidden, batch_first=True)
        self.att = nn.Linear(hidden, 1)
        self.head = Head(hidden * 2 + H * F_FEATS + emb, hidden * 2, dropout)
        self.last_attention: torch.Tensor | None = None

    def forward(self, xp, xf, sid):
        out, (h, _) = self.rnn(xp)                              # out: (batch, L, hidden)
        w = torch.softmax(self.att(torch.tanh(out)).squeeze(-1), dim=1)
        self.last_attention = w.detach()
        ctx = (out * w.unsqueeze(-1)).sum(1)
        z = torch.cat([h[-1], ctx, xf.flatten(1), self.emb(sid)], dim=1)
        return self.head(z)


ARCHS = {"mlp": MLP, "cnn": CNN, "lstm": LSTM}


def pinball(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Quantile loss averaged over the three quantiles. pred (batch, H, 3), y (batch, H)."""
    loss = 0.0
    for k, q in enumerate(QUANTILES):
        e = y - pred[..., k]
        loss = loss + torch.maximum(q * e, (q - 1) * e).mean()
    return loss / len(QUANTILES)
