"""Output images for Tasks 7, 8 and 10: python -m startupiq.task_outputs

Task 7: comparison of all deep learning models (from docs/results/dl_*.json).
Task 8: feature maps (CNN), activation visualization (saliency), attention mechanism (LSTM),
        computed live from the saved models in data/models/.
Task 10: explainable AI with SHAP and LIME on the competitor-move random forest and the
         market-share LightGBM, on Olist test rows.
Writes docs/results/figures/task7_dl_*.png, task8_*.png, task10_*.png.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from startupiq import task_figures as tf  # noqa: E402  (palette, rcParams, helpers)

warnings.filterwarnings("ignore")
FIG, RES = tf.FIG, tf.RES
S = tf.SERIES
NAMES = {"mlp": "MLP (basic)", "lstm": "LSTM + attention", "tcn": "CNN (TCN)"}


def first(s) -> float:
    return float(str(s).split(" ")[0])


def sd(s) -> float:
    parts = str(s).split(" ")
    return float(parts[-1]) if len(parts) > 1 else 0.0


# ---------------- Task 7: all deep learning models ----------------

def dl_comparison() -> None:
    dl, dlt, dr = tf.load("dl_weekly"), tf.load("dl_weekly_test"), tf.load("dl_rossmann")
    seeds = {r["arch"]: r for r in dl["seeds"]}
    test = dict(zip(("mlp", "lstm", "tcn"), dlt["test"]))
    daily = dict(zip(("mlp", "lstm", "tcn"), dr["rows"]))
    no_att = dl["attention_ablation"][1]["pooled MASE"]
    arch = ["mlp", "lstm", "tcn"]
    cols = [S[3], S[0], S[2]]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), gridspec_kw={"width_ratios": [1.6, 1, 1, 1]})

    ax = axes[0]
    x = np.arange(4)
    labels = ["MLP\n(basic)", "LSTM\n(no attention)", "LSTM +\nattention", "CNN\n(TCN)"]
    val = [first(seeds["mlp"]["pooled MASE"]), no_att, first(seeds["lstm"]["pooled MASE"]), first(seeds["tcn"]["pooled MASE"])]
    tst = [first(test["mlp"]["pooled MASE"]), np.nan, first(test["lstm"]["pooled MASE"]), first(test["tcn"]["pooled MASE"])]
    b1 = ax.bar(x - 0.2, val, 0.38, color=S[0], label="Validation")
    b2 = ax.bar(x + 0.2, tst, 0.38, color=S[1], label="Test")
    tf.label_bars(ax, b1); tf.label_bars(ax, [b for b in b2 if np.isfinite(b.get_height())])
    ax.set_xticks(x, labels, fontsize=8); ax.set_ylim(0.5, 1.0); ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Weekly sales error (MASE)")

    def panel(ax, vals, title, fmt, ylim=None):
        b = ax.bar(range(3), vals, color=cols, width=0.6)
        tf.label_bars(ax, b, fmt)
        ax.set_xticks(range(3), [NAMES[a].replace(" + ", "\n+ ").replace(" (", "\n(") for a in arch], fontsize=8)
        ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)

    panel(axes[1], [first(daily[a]["RMSPE"]) for a in arch], "Daily sales error (RMSPE)", "{:.3f}", (0.1, 0.17))
    panel(axes[2], [first(seeds[a]["UCI products"]) for a in arch], "Small, irregular products (MASE)", "{:.3f}", (0.55, 0.63))
    panel(axes[3], [seeds[a]["params"] / 1000 for a in arch], "Model size (thousand weights)", "{:.1f}k")
    fig.suptitle("Task 7: comparison of all deep learning models (lower error is better; CNN (TCN) is the best)", x=0.01, ha="left", fontsize=11)
    tf.save(fig, "task7_dl_comparison")


# ---------------- Task 8: feature maps, activation, attention ----------------

def _setup():
    from startupiq.dl import interpret as I
    from startupiq.dl import windows as W
    from startupiq.features import weekly as fw
    from startupiq.features.splits import WEEK_HORIZON as H
    series, weeks = fw.panel()
    origin = {s: len(y) - 2 * H for s, y in series.items()}
    ev_all = W.build(series, weeks, {s: [o] for s, o in origin.items()})
    rng = np.random.default_rng(0)
    is_ross = np.char.startswith(ev_all.series.astype(str), "ross_")
    pick = np.concatenate([rng.choice(np.where(is_ross)[0], 300, replace=False), rng.choice(np.where(~is_ross)[0], 300, replace=False)])
    ev = ev_all.subset(np.sort(pick))
    models = {a: I.load(a, ev)[0] for a in ("mlp", "lstm", "tcn")}
    # Example: the Rossmann store in the sample with the clearest December peak.
    x = ev.past[:, :, 0]
    ross = np.where(np.char.startswith(ev.series.astype(str), "ross_"))[0]
    ex = int(ross[np.argmax((x[ross].max(1) - x[ross].mean(1)) / (x[ross].std(1) + 1e-6))])
    return I, W, ev, models, ex, series, origin


def feature_maps(I, W, ev, models, ex) -> None:
    import torch
    m = models["tcn"]
    one = ev.subset(np.array([ex]))
    with torch.no_grad():
        m(I._t(m, one.past), I._t(m, one.future), I._t(m, one.static), keep_maps=True)
    maps = [f[0].cpu().numpy() for f in m.feature_maps]
    picks = [1, len(maps) // 2, len(maps) - 1]
    patterns, channels = I.tcn_filters(m, ev)
    fig = plt.figure(figsize=(14, 7.6))
    gs = fig.add_gridspec(4, 6, height_ratios=[1, 1.3, 1.3, 1.1], hspace=0.75, wspace=0.35)
    weeks = np.arange(-W.L + 1, 1)
    ax = fig.add_subplot(gs[0, :])
    ax.plot(weeks, one.past[0, :, 0], color=tf.INK, linewidth=1.6)
    ax.set_title(f"Input: 52 weeks of sales ({ev.series[ex].replace('_', ' ')})"); ax.set_xlim(weeks[0], 0)
    for r, k in enumerate(picks[:2]):
        ax = fig.add_subplot(gs[1 + r, :])
        fm = maps[k]
        im = ax.imshow(fm, aspect="auto", cmap="Blues", extent=[weeks[0] - 0.5, 0.5, fm.shape[0] - 0.5, -0.5])
        ax.grid(False); ax.set_ylabel("filter")
        ax.set_title(f"Feature map after convolution block {k} ({fm.shape[0]} filters x 52 weeks; dark = filter is active)")
        fig.colorbar(im, ax=ax, pad=0.01, fraction=0.02)
    ax.set_xlabel("Weeks before the forecast starts")
    for j, (p, c) in enumerate(zip(patterns, channels)):
        ax = fig.add_subplot(gs[3, j])
        ax.plot(range(-4, 1), p, color=S[0], marker="o", markersize=4)
        ax.axhline(0, color=tf.INK2, linewidth=0.8); ax.set_xticks(range(-4, 1))
        ax.set_title(f"filter {c} detects", fontsize=8.5)
    fig.suptitle("Task 8: feature maps of the CNN (TCN). Bottom row: the 5-week sales pattern that each top filter responds to",
                 x=0.01, ha="left", fontsize=11)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "task8_1_feature_maps.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    print("wrote task8_1_feature_maps")


def activation(I, W, ev, models, ex) -> None:
    sal = {a: I.saliency(m, ev)[0] for a, m in models.items()}
    weeks = np.arange(-W.L + 1, 1)
    fig = plt.figure(figsize=(14, 6.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1.1, 1.2], hspace=0.6)
    ax = fig.add_subplot(gs[0])
    ax.plot(weeks, ev.past[ex, :, 0], color=tf.INK, linewidth=1.6); ax.set_xlim(weeks[0] - 0.5, 0.5)
    ax.set_title(f"Input: 52 weeks of sales ({ev.series[ex].replace('_', ' ')})")
    ax = fig.add_subplot(gs[1])
    grid = np.stack([sal[a][ex] / sal[a][ex].max() for a in models])
    ax.imshow(grid, aspect="auto", cmap="Oranges", extent=[weeks[0] - 0.5, 0.5, 2.5, -0.5]); ax.grid(False)
    ax.set_yticks(range(3), [NAMES[a] for a in models])
    ax.set_title("Activation (saliency) heatmap for this store: how strongly each past week pushes the forecast (dark = strong)")
    ax = fig.add_subplot(gs[2])
    for i, a in enumerate(models):
        ax.plot(weeks, sal[a].mean(0) * 100, color=[S[3], S[0], S[2]][i], label=NAMES[a])
    ax.set_xlim(weeks[0] - 0.5, 0.5); ax.set_xlabel("Weeks before the forecast starts"); ax.set_ylabel("% of influence")
    ax.legend(loc="upper left", fontsize=8); ax.set_title("Average over 600 series: all models focus most on the latest weeks")
    fig.suptitle("Task 8: activation visualization (gradient x input saliency)", x=0.01, ha="left", fontsize=11)
    fig.savefig(FIG / "task8_2_activation_visualization.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    print("wrote task8_2_activation_visualization")


def attention_fig(I, W, ev, models, ex) -> None:
    import torch
    m = models["lstm"]
    with torch.no_grad():
        m(I._t(m, ev.past), I._t(m, ev.future), I._t(m, ev.static))
    att = m.last_attention.cpu().numpy()                           # [N, H, L]
    weeks = np.arange(-W.L + 1, 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
    a = att[ex]
    im = axes[0].imshow(a, aspect="auto", cmap="Purples", extent=[weeks[0] - 0.5, 0.5, a.shape[0] + 0.5, 0.5]); axes[0].grid(False)
    axes[0].set_xlabel("Past week the model looks at"); axes[0].set_ylabel("Forecast week (1-12)")
    axes[0].set_title(f"Attention weights for one store ({ev.series[ex].replace('_', ' ')})")
    fig.colorbar(im, ax=axes[0], pad=0.01, fraction=0.03)
    mean = att.mean((0, 1)) * 100
    axes[1].bar(weeks, mean, color=S[5], width=0.8)
    axes[1].set_xlabel("Weeks before the forecast starts"); axes[1].set_ylabel("Average attention (%)")
    axes[1].set_title("Average attention over 600 series")
    fig.suptitle("Task 8: attention mechanism of the LSTM (which past weeks each forecast week focuses on)", x=0.01, ha="left", fontsize=11)
    tf.save(fig, "task8_3_attention_mechanism")


# ---------------- Task 10: SHAP and LIME ----------------

def _pretty(name: str) -> str:
    name = re.sub(r"^(num|cat|missingindicator)__", "", name)
    return name.replace("missingindicator_", "missing: ").replace("_", " ")


def shap_lime() -> None:
    import shap
    from lime.lime_tabular import LimeTabularExplainer

    from startupiq.features import olist as fo
    from startupiq.models import serve
    rows = serve.olist_rows()
    test = rows[rows.split == "test"].sample(400, random_state=0)

    # SHAP on the market-share LightGBM: beeswarm over 400 test rows.
    share = serve.share_model()
    X = test[fo.FEATURES]
    sv = shap.TreeExplainer(share)(X)
    sv.feature_names = [_pretty(f) for f in fo.FEATURES]
    plt.figure()
    shap.plots.beeswarm(sv, max_display=12, show=False)
    fig = plt.gcf(); fig.set_size_inches(10, 5.6)
    plt.title("Task 10: SHAP for the market-share model (LightGBM), 400 unseen test sellers\n"
              "red = high feature value; right of 0 = pushes predicted share up", loc="left", fontsize=10.5)
    fig.savefig(FIG / "task10_3_shap_market_share.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    print("wrote task10_3_shap_market_share")

    # SHAP on the competitor-move random forest: mean |SHAP| per class.
    comp = serve.competitor_model()
    ct, rf = comp.steps[0][1], comp.steps[-1][1]
    Xt = ct.transform(X.iloc[:200])
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    names = [_pretty(n) for n in ct.get_feature_names_out()]
    vals = shap.TreeExplainer(rf).shap_values(Xt, check_additivity=False)
    vals = np.stack(vals, -1) if isinstance(vals, list) else vals          # [n, features, classes]
    imp = np.abs(vals).mean(0)                                              # [features, classes]
    order = np.argsort(-imp.sum(1))[:12][::-1]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    left = np.zeros(len(order))
    for k, c in enumerate(rf.classes_):
        ax.barh(range(len(order)), imp[order, k], left=left, color=S[k], label=str(c).replace("_", " "))
        left += imp[order, k]
    ax.set_yticks(range(len(order)), [names[i] for i in order]); ax.set_xlabel("mean |SHAP value| (effect on predicted probability)")
    ax.legend(loc="lower right", title="competitor move"); ax.grid(axis="y", visible=False)
    ax.set_title("Task 10: SHAP for the competitor-move model (random forest): what drives each predicted move")
    tf.save(fig, "task10_4_shap_competitor")

    # LIME on one competitor: why the model predicts its most likely move.
    num = fo.NUMERIC
    proba = comp.predict_proba(X)
    classes = list(comp.classes_)
    i = int(np.argmax(proba[:, classes.index("exit")])) if "exit" in classes else 0
    row = X.iloc[[i]]

    def predict_fn(arr):
        import pandas as pd
        df = pd.DataFrame(arr, columns=num)
        for c in fo.CATEGORICAL:
            df[c] = pd.Categorical([row[c].iloc[0]] * len(df), categories=X[c].cat.categories)
        return comp.predict_proba(df[fo.FEATURES])

    train = rows[rows.split.isin(["train", "val"])][num].fillna(rows[num].median())
    lime = LimeTabularExplainer(train.values, feature_names=[_pretty(n) for n in num], class_names=classes,
                                mode="classification", discretize_continuous=True, random_state=0)
    k = int(np.argmax(proba[i]))
    exp = lime.explain_instance(row[num].fillna(train.median()).values[0], predict_fn, num_features=10, labels=[k], num_samples=3000)
    lst = exp.as_list(label=k)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 2.4]})
    axes[0].barh(range(len(classes)), proba[i] * 100, color=[S[j] for j in range(len(classes))])
    axes[0].set_yticks(range(len(classes)), [c.replace("_", " ") for c in classes]); axes[0].set_xlabel("probability (%)")
    for j, p in enumerate(proba[i]):
        axes[0].text(p * 100 + 1, j, f"{p:.0%}", va="center", fontsize=8, color=tf.INK2)
    axes[0].set_title("Model prediction for one competitor"); axes[0].grid(axis="y", visible=False)
    w = [v for _, v in lst]
    axes[1].barh(range(len(lst)), w, color=[tf.GOOD if v > 0 else tf.BAD for v in w])
    axes[1].set_yticks(range(len(lst)), [n for n, _ in lst], fontsize=8.5); axes[1].axvline(0, color=tf.INK2, linewidth=0.8)
    axes[1].set_xlabel(f"effect on P({classes[k].replace('_', ' ')})"); axes[1].grid(axis="y", visible=False)
    axes[1].set_title(f"LIME: why '{classes[k].replace('_', ' ')}'? (green = supports, red = against)")
    fig.suptitle("Task 10: LIME local explanation of a single competitor prediction (unseen test seller)", x=0.01, ha="left", fontsize=11)
    tf.save(fig, "task10_5_lime_competitor")


def main() -> None:
    dl_comparison()
    I, W, ev, models, ex, *_ = _setup()
    feature_maps(I, W, ev, models, ex)
    activation(I, W, ev, models, ex)
    attention_fig(I, W, ev, models, ex)
    shap_lime()


if __name__ == "__main__":
    main()
