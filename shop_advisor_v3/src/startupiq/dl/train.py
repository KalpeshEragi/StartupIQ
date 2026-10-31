"""Training loop: AdamW, gradient clipping, learning-rate decay on plateau, early stopping (Task 7).

Early stopping uses a holdout of whole series (10%) drawn from the training windows, so it
never looks at the evaluation origin.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from startupiq.dl.models import ARCHS, pinball
from startupiq.dl.windows import Windows

torch.set_num_threads(max(1, torch.get_num_threads()))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    arch: str
    hidden: int = 64
    layers: int = 1
    dropout: float = 0.1
    lr: float = 2e-3
    weight_decay: float = 1e-4
    attention: bool = True
    batch: int = 512
    max_epochs: int = 30
    patience: int = 4

    def label(self) -> str:
        att = "+attention" if self.arch == "lstm" and self.attention else ""
        return f"{self.arch}{att} h{self.hidden} l{self.layers} do{self.dropout} lr{self.lr} wd{self.weight_decay}"


@dataclass
class Trained:
    model: torch.nn.Module
    config: Config
    seed: int
    epochs: int
    seconds: float
    history: list[tuple[float, float]] = field(default_factory=list)  # (train loss, holdout loss) per epoch


def _tensors(w: Windows):
    mask = torch.from_numpy(w.mask.astype(np.float32)) if w.mask is not None else None
    ids = torch.from_numpy(w.ids.astype(np.int64)) if w.ids is not None else None
    return (torch.from_numpy(w.past), torch.from_numpy(w.future), torch.from_numpy(w.static), torch.from_numpy(np.nan_to_num(w.target)), mask, ids)


def _forward(model, p, f, s, ids, idx):
    return model(p[idx].to(DEVICE), f[idx].to(DEVICE), s[idx].to(DEVICE), None if ids is None else ids[idx].to(DEVICE))


def _loss(model, w: Windows, batch: int = 512) -> float:  # small batches: the 4 GB GPU is shared with the desktop
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        p, f, s, t, m, ids = _tensors(w)
        for i in range(0, len(w), batch):
            idx = slice(i, i + batch)
            out = _forward(model, p, f, s, ids, idx)
            total += float(pinball(out, t[idx].to(DEVICE), None if m is None else m[idx].to(DEVICE))) * len(out)
            n += len(out)
    return total / n


def build_model(cfg: Config, w: Windows, lookback: int, n_ids: int | None = None) -> torch.nn.Module:
    """Input sizes come from the windows, so the same code serves the weekly panel and Rossmann daily."""
    n = n_ids if n_ids is not None else (int(w.ids.max()) + 1 if w.ids is not None else 0)
    return ARCHS[cfg.arch](lookback=lookback, hidden=cfg.hidden, layers=cfg.layers, dropout=cfg.dropout, attention=cfg.attention,
                           past_dim=w.past.shape[-1], future_dim=w.future.shape[-1], static_dim=w.static.shape[-1], n_ids=n).to(DEVICE)


def fit(train: Windows, cfg: Config, seed: int = 0, lookback: int = 52, n_ids: int | None = None) -> Trained:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ids = np.unique(train.series)
    hold_ids = set(rng.choice(ids, max(1, len(ids) // 10), replace=False))
    hold = np.isin(train.series, list(hold_ids))
    tr, ho = train.subset(~hold), train.subset(hold)
    model = build_model(cfg, train, lookback, n_ids)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    p, f, s, t, m, ids = _tensors(tr)
    if DEVICE == "cuda":  # the whole training set is tens of MB: copy it to the GPU once instead of every batch
        p, f, s, t = p.to(DEVICE), f.to(DEVICE), s.to(DEVICE), t.to(DEVICE)
        m = None if m is None else m.to(DEVICE)
        ids = None if ids is None else ids.to(DEVICE)
    best, best_state, bad, history = np.inf, None, 0, []
    t0 = time.time()
    for epoch in range(cfg.max_epochs):
        model.train()
        order = torch.from_numpy(rng.permutation(len(tr))).to(p.device)
        run, n = 0.0, 0
        for i in range(0, len(tr), cfg.batch):
            b = order[i:i + cfg.batch]
            out = _forward(model, p, f, s, ids, b)
            loss = pinball(out, t[b].to(DEVICE), None if m is None else m[b].to(DEVICE))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += loss.item() * len(b); n += len(b)
        val = _loss(model, ho)
        history.append((run / n, val))
        sched.step(val)
        if val < best - 1e-5:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    del p, f, s, t, m, ids
    if DEVICE == "cuda":
        torch.cuda.empty_cache()  # give memory back before the next model
    return Trained(model, cfg, seed, len(history), time.time() - t0, history)


def predict(model, w: Windows, batch: int = 512) -> np.ndarray:
    """Quantiles in sales units: [N, H, 3] for P10, P50, P90."""
    model.eval()
    outs = []
    p, f, s, _, _, ids = _tensors(w)
    with torch.no_grad():
        for i in range(0, len(w), batch):
            outs.append(_forward(model, p, f, s, ids, slice(i, i + batch)).cpu().numpy())
    q = np.concatenate(outs)
    return np.clip(np.expm1(q), 0, None) * w.scale[:, None, None]
