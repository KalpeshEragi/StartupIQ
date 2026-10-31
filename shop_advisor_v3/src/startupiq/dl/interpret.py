"""Task 8: interpretability of the weekly deep forecasters. python -m startupiq.dl.interpret

Uses the models saved by the validation run (data/models/dl_weekly_<arch>.pt) on the
validation origin. Four questions:

1. Saliency (gradient x input): which past weeks move the P50 forecast, per model?
2. Faithfulness: do the LSTM's attention weights agree with occlusion (blank one past week,
   measure how much the forecast changes)? Attention is often read as an explanation; this
   checks whether it is one.
3. Feature maps: what patterns do the TCN's first convolution filters respond to?
4. How do accuracy, calibration, size, speed and explainability trade off across the basic
   neural model (MLP), LSTM, TCN and the Step 5 models?
Writes docs/results/dl_interpret.md and figures 18-21.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from startupiq.dl import windows as W  # noqa: E402
from startupiq.dl.models import n_params  # noqa: E402
from startupiq.dl.train import Config, build_model, predict  # noqa: E402
from startupiq.evaluate import report  # noqa: E402
from startupiq.features import weekly as fw  # noqa: E402
from startupiq.features.splits import WEEK_HORIZON as H  # noqa: E402

FIG = Path("docs/results/figures")
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.titlelocation": "left",
                     "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "lines.linewidth": 2, "font.size": 9, "legend.frameon": False, "axes.axisbelow": True})
SAMPLE = 600  # windows used for the gradient and occlusion studies (stratified by dataset)


def load(arch: str, ev: W.Windows):
    chosen = json.loads(Path("docs/results/dl_weekly.json").read_text(encoding="utf-8"))["chosen"][arch]
    cfg = Config(**chosen["config"])
    model = build_model(cfg, ev, W.L)
    model.load_state_dict(torch.load(Path("data/models") / f"dl_weekly_{arch}.pt", weights_only=True, map_location=next(model.parameters()).device))
    model.eval()
    return model, cfg


def _t(model, a: np.ndarray, grad: bool = False) -> torch.Tensor:
    """numpy -> tensor on the model's device (GPU when available)."""
    return torch.tensor(a, device=next(model.parameters()).device, requires_grad=grad)


def saliency(model, ev: W.Windows) -> tuple[np.ndarray, np.ndarray]:
    """|gradient x input| of the summed P50 (log space) w.r.t. each past week, normalised per window;
    plus the same for the 'same week last year' future feature, per horizon step."""
    past, future = _t(model, ev.past, True), _t(model, ev.future, True)
    if isinstance(model, torch.nn.LSTM) or getattr(model, "kind", "") == "lstm":
        torch.backends.cudnn.enabled = False  # cuDNN's LSTM cannot run backward in eval mode
    out = model(past, future, _t(model, ev.static))
    out[..., 1].sum().backward()
    torch.backends.cudnn.enabled = True
    sp = (past.grad * past).abs().sum(-1).detach().cpu().numpy()             # [N, L]
    sf = (future.grad[..., 0] * future[..., 0]).abs().detach().cpu().numpy()  # [N, H] seasonal channel
    total = sp.sum(1, keepdims=True) + sf.sum(1, keepdims=True) + 1e-12
    return sp / total, sf / total


def occlusion(model, ev: W.Windows) -> np.ndarray:
    """Change in summed P50 when one past week's sales value is replaced by the window's mean."""
    with torch.no_grad():
        base = model(_t(model, ev.past), _t(model, ev.future), _t(model, ev.static))[..., 1].sum(1).cpu().numpy()
        out = np.zeros((len(ev), W.L))
        for lag in range(W.L):
            p = ev.past.copy()
            p[:, lag, 0] = ev.past[:, :, 0].mean(1)
            out[:, lag] = np.abs(model(_t(model, p), _t(model, ev.future), _t(model, ev.static))[..., 1].sum(1).cpu().numpy() - base)
    return out / (out.sum(1, keepdims=True) + 1e-12)


def attention(model, ev: W.Windows) -> np.ndarray:
    with torch.no_grad():
        model(_t(model, ev.past), _t(model, ev.future), _t(model, ev.static))
    return model.last_attention.mean(1).cpu().numpy()                         # average over horizon steps -> [N, L]


def tcn_filters(model, ev: W.Windows, top: int = 6) -> tuple[list[np.ndarray], list[int]]:
    """For the first block's most active channels: the average input (sales channel, standardised) over the
    5 weeks ending where the channel fires hardest. Receptive field of block 1 = 5 weeks."""
    with torch.no_grad():
        model(_t(model, ev.past), _t(model, ev.future), _t(model, ev.static), keep_maps=True)
    maps = model.feature_maps[1].cpu().numpy()                                 # after block 1: [N, C, L]
    x = ev.past[:, :, 0]
    z = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)
    channels = np.argsort(-maps.mean((0, 2)))[:top]
    patterns = []
    for c in channels:
        pos = maps[:, c, 4:].argmax(1) + 4                               # need 4 earlier weeks for the 5-week segment
        patterns.append(np.mean([z[i, p - 4:p + 1] for i, p in enumerate(pos)], axis=0))
    return patterns, [int(c) for c in channels]


def _fig_saliency(curves: dict, seasonal: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [3, 1]})
    lags = np.arange(W.L, 0, -1)                                         # weeks before the origin
    for i, (name, c) in enumerate(curves.items()):
        axes[0].plot(lags, c, color=SERIES[i], label=name)
    axes[0].invert_xaxis(); axes[0].set_xlabel("Weeks before the forecast starts"); axes[0].set_ylabel("Share of saliency")
    axes[0].legend(loc="upper left"); axes[0].set_title("Which past weeks drive the forecast (gradient x input)")
    names = list(seasonal)
    axes[1].bar(range(len(names)), [seasonal[n] for n in names], color=[SERIES[i] for i in range(len(names))], width=0.6)
    axes[1].set_xticks(range(len(names)), names, rotation=20); axes[1].set_title("Share on 'same week last year'")
    fig.tight_layout(); fig.savefig(FIG / "18_dl_saliency.png", dpi=110); plt.close(fig)


def _fig_attention(att: np.ndarray, occ: np.ndarray, rho: float) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.4))
    lags = np.arange(W.L, 0, -1)
    ax.plot(lags, att.mean(0), color=SERIES[0], label="Attention weight (average)")
    ax.plot(lags, occ.mean(0), color=SERIES[1], label="Occlusion importance (average)")
    ax.invert_xaxis(); ax.set_xlabel("Weeks before the forecast starts"); ax.set_ylabel("Share")
    ax.legend(loc="upper left"); ax.set_title(f"LSTM: attention vs occlusion (median per-series Spearman rho = {rho:.2f})")
    fig.tight_layout(); fig.savefig(FIG / "19_dl_attention_vs_occlusion.png", dpi=110); plt.close(fig)


def _fig_filters(patterns: list[np.ndarray], channels: list[int]) -> None:
    fig, axes = plt.subplots(1, len(patterns), figsize=(11, 2.4), sharey=True)
    for ax, p, c in zip(axes, patterns, channels):
        ax.plot(range(-4, 1), p, color=SERIES[0], marker="o", markersize=4)
        ax.axhline(0, color=INK2, linewidth=0.8); ax.set_title(f"filter {c}", fontsize=9)
        ax.set_xticks(range(-4, 1), ["-4", "-3", "-2", "-1", "0"])
    axes[0].set_ylabel("Sales (standardised)")
    fig.suptitle("TCN first-layer filters: average 5-week input where each filter fires hardest", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / "20_dl_cnn_filters.png", dpi=110); plt.close(fig)


def _fig_case(series, origin, ev, q, sal, picks) -> None:
    fig, axes = plt.subplots(1, len(picks), figsize=(11, 3.4))
    for ax, sid in zip(axes, picks):
        i = int(np.where(ev.series == sid)[0][0])
        y, o = series[sid], origin[sid]
        hist_t = np.arange(o - W.L, o)
        ax.plot(hist_t, y[o - W.L:o], color=INK2, linewidth=1.4, label="History")
        ax.plot(np.arange(o, o + H), y[o:o + H], color=INK, linewidth=1.4, linestyle="--", label="What happened")
        ax.fill_between(np.arange(o, o + H), q[i, :, 0], q[i, :, 2], color=SERIES[0], alpha=0.2, label="P10-P90")
        ax.plot(np.arange(o, o + H), q[i, :, 1], color=SERIES[0], label="P50 (LSTM)")
        top = np.argsort(-sal[i])[:3]
        ax.scatter(hist_t[top], y[o - W.L:o][top], s=60, color=SERIES[1], zorder=3, label="3 most salient weeks")
        ax.set_title(sid.replace("_", " "), fontsize=10)
    axes[0].legend(loc="upper left", fontsize=7.5)
    fig.tight_layout(); fig.savefig(FIG / "21_dl_case_study.png", dpi=110); plt.close(fig)


def run() -> dict:
    t0 = time.time()
    series, weeks = fw.panel()
    origin = {s: len(y) - 2 * H for s, y in series.items()}           # validation origin, as in the experiment
    ev_all = W.build(series, weeks, {s: [o] for s, o in origin.items()})
    rng = np.random.default_rng(0)
    is_ross = np.char.startswith(ev_all.series.astype(str), "ross_")
    pick = np.concatenate([rng.choice(np.where(is_ross)[0], SAMPLE // 2, replace=False), rng.choice(np.where(~is_ross)[0], SAMPLE // 2, replace=False)])
    ev = ev_all.subset(np.sort(pick))
    ev_ross = np.char.startswith(ev.series.astype(str), "ross_")
    exp = json.loads(Path("docs/results/dl_weekly.json").read_text(encoding="utf-8"))
    FIG.mkdir(parents=True, exist_ok=True)

    models, curves, seasonal, sal_by, findings = {}, {}, {}, {}, {}
    for arch, name in (("mlp", "MLP (basic)"), ("lstm", "LSTM + attention"), ("tcn", "TCN (1D-CNN)")):
        m, cfg = load(arch, ev)
        models[arch] = m
        sp, sf = saliency(m, ev)
        sal_by[arch] = sp
        curves[name] = sp.mean(0)
        seasonal[name] = float(sf.sum(1).mean())
        findings[arch] = {"share_last_4_weeks": float(sp[:, -4:].sum(1).mean()), "share_seasonal_feature": seasonal[name],
                          "share_52_weeks_back_window": float(sp[:, :4].sum(1).mean()),
                          "rossmann_last_4": float(sp[ev_ross][:, -4:].sum(1).mean()), "uci_last_4": float(sp[~ev_ross][:, -4:].sum(1).mean())}
    _fig_saliency(curves, seasonal)

    att = attention(models["lstm"], ev)
    occ = occlusion(models["lstm"], ev)
    rhos = np.array([spearmanr(a, o).statistic for a, o in zip(att, occ)])
    rho_mean_curve = float(spearmanr(att.mean(0), occ.mean(0)).statistic)
    sal_occ = np.array([spearmanr(s, o).statistic for s, o in zip(sal_by["lstm"], occ)])
    top_overlap = float(np.mean([len(set(np.argsort(-a)[:5]) & set(np.argsort(-o)[:5])) / 5 for a, o in zip(att, occ)]))
    _fig_attention(att, occ, float(np.nanmedian(rhos)))

    patterns, channels = tcn_filters(models["tcn"], ev)
    _fig_filters(patterns, channels)
    shapes = []
    for p in patterns:
        if p[-1] > 1 and p[:-1].mean() < 0.3:
            shapes.append("fires on a sudden spike in the latest week")
        elif np.all(np.diff(p) > 0):
            shapes.append("fires on a steady rise")
        elif np.all(np.diff(p) < 0):
            shapes.append("fires on a steady fall")
        elif p[-1] < -1:
            shapes.append("fires on a sudden drop")
        else:
            shapes.append("mixed / level pattern")

    q_lstm = predict(models["lstm"], ev)
    picks = [ev.series[ev_ross][0], ev.series[~ev_ross][0]]
    _fig_case(series, origin, ev, q_lstm, sal_by["lstm"], picks)

    # Speed: forecast all 2,539 series once.
    speed = {}
    for arch, m in models.items():
        t = time.time(); predict(m, ev_all); speed[arch] = (time.time() - t) / len(ev_all) * 1000
    seeds = {r["arch"]: r for r in exp["seeds"]}
    bar = exp["step5_bar"]
    comp = pd.DataFrame([
        {"model": "Moving average (4 weeks)", "pooled MASE": f"{bar['Moving average (4 weeks)']:.3f}", "coverage": "n/a", "params": 0, "ms per series": "<0.1",
         "how it explains itself": "fully transparent: the average of the last 4 weeks"},
        {"model": "Auto (Step 5, serves today)", "pooled MASE": f"{bar['Auto']:.3f}", "coverage": "0.946 / 0.587", "params": 0, "ms per series": "about 20 (fits ETS)",
         "how it explains itself": "names the method it picked and its backtest error"},
        {"model": "LightGBM global (Step 5)", "pooled MASE": f"{bar['LightGBM global']:.3f}", "coverage": "0.339 / 0.719", "params": "600 trees", "ms per series": "<1",
         "how it explains itself": "feature importance (global), no per-week attribution"},
        *[{"model": seeds[a]["model"], "pooled MASE": seeds[a]["pooled MASE"], "coverage": seeds[a]["P10-P90 coverage"], "params": n_params(models[a]),
           "ms per series": f"{speed[a]:.2f}", "how it explains itself": how} for a, how in
          (("mlp", "gradient saliency only; no notion of order"),
           ("lstm", f"attention weights (agree with occlusion: median rho {np.nanmedian(rhos):.2f}) + saliency"),
           ("tcn", "feature maps (filters with readable shapes) + saliency"))],
    ])
    fx = pd.DataFrame([{"model": a, **{k: round(v, 3) for k, v in f.items()}} for a, f in findings.items()])
    out = {"findings": findings, "attention_vs_occlusion": {"median_spearman": float(np.nanmedian(rhos)), "mean_curve_spearman": rho_mean_curve,
                                                            "top5_overlap": top_overlap, "saliency_vs_occlusion_median": float(np.nanmedian(sal_occ))},
           "filters": dict(zip(map(str, channels), shapes)), "speed_ms": speed, "seconds": round(time.time() - t0)}
    report.write("dl_interpret", "Interpretability of the deep forecasters (Task 8)", [
        ("Setup", f"{SAMPLE} validation windows (half Rossmann stores, half UCI products), models from the validation run."),
        ("1. Which past weeks drive the forecast", "![saliency](figures/18_dl_saliency.png)\n\n" + report.table(fx, "{:.3f}")
         + "\n\nShares are of total gradient x input; 'last 4 weeks' and '52 weeks back window' are the newest and oldest 4 lookback weeks."),
        ("2. Is the LSTM's attention a faithful explanation?", "![attention](figures/19_dl_attention_vs_occlusion.png)\n\n"
         f"Per series, Spearman correlation between attention weights and occlusion importance: median {np.nanmedian(rhos):.2f} "
         f"(interquartile {np.nanpercentile(rhos, 25):.2f} to {np.nanpercentile(rhos, 75):.2f}); overlap of the top-5 weeks {top_overlap:.0%}; "
         f"correlation of the average curves {rho_mean_curve:.2f}. Gradient saliency vs occlusion: median {np.nanmedian(sal_occ):.2f}."),
        ("3. What the TCN's filters detect", "![filters](figures/20_dl_cnn_filters.png)\n\n" + "\n".join(f"- filter {c}: {s}" for c, s in zip(channels, shapes))),
        ("4. Accuracy, calibration, cost and explainability", report.table(comp)
         + "\n\nCoverage for Step 5 models is Rossmann / UCI on validation; deep models show pooled coverage mean ± sd over 3 seeds."),
        ("Case study", "![case](figures/21_dl_case_study.png)"),
    ], out, command="python -m startupiq.dl.interpret")
    return out


if __name__ == "__main__":
    print(run())
