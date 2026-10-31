"""Output images for the Task 7-10 write-up: python -m startupiq.task_figures [--retrain]

Every number is read from docs/results/*.json (or demo/DEMO_GUIDE.md); nothing is typed in.
--retrain trains the chosen MLP, LSTM and CNN once (seed 0, GPU) to record their loss per
epoch for the training-curve figure; the curves are cached in docs/results/dl_training_curves.json.
Writes docs/results/figures/task7_*.png ... task10_*.png.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

RES = Path("docs/results")
FIG = RES / "figures"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
GOOD, BAD = "#0ca30c", "#d03b3b"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.titlelocation": "left",
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 2, "font.size": 9.5, "legend.frameon": False})


def load(name):
    return json.loads((RES / f"{name}.json").read_text(encoding="utf-8"))


def first(s) -> float:
    return float(str(s).split(" ")[0])


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=130, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", name)


def label_bars(ax, bars, fmt="{:.3f}"):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(b.get_height()), ha="center", va="bottom", fontsize=7.5, color=INK2)


# ---------------- Task 7 ----------------

def training_curves(retrain: bool) -> None:
    path = RES / "dl_training_curves.json"
    if retrain or not path.exists():
        from startupiq.dl import windows as W
        from startupiq.dl.train import Config, fit
        from startupiq.features import weekly as fw
        series, weeks = fw.panel()
        val_o = {s: len(y) - 24 for s, y in series.items()}
        train = W.build(series, weeks, W.training_origins(series, val_o))
        chosen = load("dl_weekly")["chosen"]
        curves = {a: fit(train, Config(**chosen[a]["config"]), 0).history for a in ("mlp", "lstm", "tcn")}
        path.write_text(json.dumps(curves), encoding="utf-8")
    curves = json.loads(path.read_text(encoding="utf-8"))
    names = {"mlp": "MLP (basic)", "lstm": "LSTM + attention", "tcn": "CNN (TCN)"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, (a, h) in zip(axes, curves.items()):
        h = np.array(h)
        ep = np.arange(1, len(h) + 1)
        ax.plot(ep, h[:, 0], color=SERIES[0], label="Training loss")
        ax.plot(ep, h[:, 1], color=SERIES[1], label="Held-out loss (10% of series)")
        best = int(np.argmin(h[:, 1]))
        ax.scatter([best + 1], [h[best, 1]], s=40, color=SERIES[1], zorder=3)
        ax.annotate("best epoch kept\n(early stopping)", (best + 1, h[best, 1]), textcoords="offset points", xytext=(-10, 18), fontsize=7.5, color=INK2, ha="right")
        ax.set_title(names[a]); ax.set_xlabel("Epoch")
    axes[0].set_ylabel("Pinball (quantile) loss"); axes[0].legend(loc="upper right")
    fig.suptitle("Task 7: training curves on the GPU (no overfitting: held-out loss stays at or below training loss)", x=0.01, ha="left", fontsize=11)
    save(fig, "task7_1_training_curves")


def model_comparison() -> None:
    dl, dlt = load("dl_weekly"), load("dl_weekly_test")
    seeds = {r["arch"]: r for r in dl["seeds"]}
    test = {r["model"].split(" ")[0]: r for r in dlt["test"]}
    hybrid = next(r for r in load("dl_hybrid")["summary"] if r["dataset"] == "Pooled")
    fw = {r["dataset"]: r for r in load("forecast_weekly")["validation"]["summary"]}
    fwt = {r["dataset"]: r for r in load("forecast_weekly_test")["test"]["summary"]}
    auto_key = next(k for k in fw["Pooled"] if k.startswith("Auto"))
    models = ["Auto\n(classical)", "LightGBM\n(classical)", "MLP\n(basic NN)", "LSTM +\nattention", "CNN\n(TCN)", "Hybrid\nAuto + CNN"]
    val_all = [float(fw["Pooled"][auto_key]), first(fw["Pooled"]["LightGBM global"]), first(seeds["mlp"]["pooled MASE"]),
               first(seeds["lstm"]["pooled MASE"]), first(seeds["tcn"]["pooled MASE"]), first(hybrid["Hybrid"])]
    test_all = [float(fwt["Pooled"][auto_key]), first(fwt["Pooled"]["LightGBM global"]), first(test["mlp"]["pooled MASE"]),
                first(test["lstm+attention"]["pooled MASE"]), first(test["tcn"]["pooled MASE"]), np.nan]
    val_p = [float(fw["UCI products"][auto_key]), first(fw["UCI products"]["LightGBM global"]), first(seeds["mlp"]["UCI products"]),
             first(seeds["lstm"]["UCI products"]), first(seeds["tcn"]["UCI products"]), first(next(r for r in load("dl_hybrid")["summary"] if r["dataset"] == "UCI products")["Hybrid"])]
    test_p = [float(fwt["UCI products"][auto_key]), first(fwt["UCI products"]["LightGBM global"]), first(test["mlp"]["UCI products"]),
              first(test["lstm+attention"]["UCI products"]), first(test["tcn"]["UCI products"]), np.nan]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    x = np.arange(len(models))
    for ax, (v, t, title) in zip(axes, ((val_all, test_all, "All 2,539 series"), (val_p, test_p, "Small, irregular products (1,604 UCI series)"))):
        b1 = ax.bar(x - 0.2, v, 0.38, color=SERIES[0], label="Validation")
        b2 = ax.bar(x + 0.2, t, 0.38, color=SERIES[1], label="Test (scored once)")
        label_bars(ax, b1); label_bars(ax, [b for b in b2 if np.isfinite(b.get_height())])
        ax.set_xticks(x, models, fontsize=8); ax.set_title(title)
    axes[0].set_ylabel("Weekly forecast error, MASE (lower is better)"); axes[0].legend(loc="upper left")
    fig.suptitle("Task 7: deep models vs classical models (weekly forecast error)", x=0.01, ha="left", fontsize=11)
    save(fig, "task7_2_model_comparison")


def regularisation() -> None:
    dl = load("dl_weekly")
    reg = dl["regularisation"]
    names = {(0, 0): "none", (1, 0): "dropout 0.3", (0, 1): "weight decay", (1, 1): "both"}
    labels = [names[(int(r["dropout"] > 0), int(r["weight decay"] > 0))] for r in reg]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), gridspec_kw={"width_ratios": [2, 2, 1.3]})
    b = axes[0].bar(range(4), [r["pooled MASE"] for r in reg], color=SERIES[0], width=0.6)
    label_bars(axes[0], b); axes[0].set_xticks(range(4), labels, fontsize=7.5); axes[0].set_title("Error (MASE)"); axes[0].set_ylim(0.6, 0.73)
    b = axes[1].bar(range(4), [r["coverage"] * 100 for r in reg], color=SERIES[2], width=0.6)
    label_bars(axes[1], b, "{:.0f}%"); axes[1].axhline(80, color=INK2, linestyle="--", linewidth=1)
    axes[1].text(-0.4, 80.6, "target 80%", ha="left", fontsize=7.5, color=INK2)
    axes[1].set_xticks(range(4), labels, fontsize=7.5); axes[1].set_title("Range contains the truth"); axes[1].set_ylim(60, 90)
    att = dl["attention_ablation"]
    b = axes[2].bar([0, 1], [att[1]["pooled MASE"], att[0]["pooled MASE"]], color=[SERIES[3], SERIES[0]], width=0.6)
    label_bars(axes[2], b); axes[2].set_xticks([0, 1], ["LSTM\nwithout attention", "LSTM\nwith attention"], fontsize=8)
    axes[2].set_title("Attention (MASE)"); axes[2].set_ylim(0.6, 0.8)
    fig.suptitle("Task 7: regularization and attention (LSTM, validation): dropout makes ranges honest, attention lowers error", x=0.01, ha="left", fontsize=11)
    save(fig, "task7_3_regularization_attention")


def daily_comparison() -> None:
    dr = load("dl_rossmann")
    step5 = dr["step5"]
    rows = [("Same day\nlast year", step5["Same day last year (rows with history only)"]), ("Ridge", next(v for k, v in step5.items() if k.startswith("Ridge"))),
            ("LightGBM", next(v for k, v in step5.items() if k.startswith("LightGBM"))), ("Random\nforest", next(v for k, v in step5.items() if k.startswith("Random")))]
    rows += [(n, first(r["RMSPE"])) for n, r in zip(["MLP", "LSTM +\nattention", "CNN\n(TCN)"], dr["rows"])]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    colors = [GRID if i < 4 else SERIES[0] for i in range(len(rows))]
    colors = ["#b9b8b3"] * 4 + [SERIES[0]] * 3
    b = ax.bar(range(len(rows)), [r[1] for r in rows], color=colors, width=0.6)
    label_bars(ax, b); ax.set_xticks(range(len(rows)), [r[0] for r in rows], fontsize=8.5)
    ax.set_ylabel("RMSPE (lower is better)")
    ax.set_title("Task 7: daily sales, 1,115 Rossmann stores (grey = classical, blue = deep)")
    save(fig, "task7_4_daily_rossmann")


# ---------------- Task 8 ----------------

def interpret_comparison() -> None:
    i = load("dl_interpret")
    dl = load("dl_weekly")
    seeds = {r["arch"]: r for r in dl["seeds"]}
    names = {"mlp": "MLP (basic)", "lstm": "LSTM + attention", "tcn": "CNN (TCN)"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    x = np.arange(3)
    b = axes[0].bar(x, [first(seeds[a]["pooled MASE"]) for a in names], color=SERIES[0], width=0.6)
    label_bars(axes[0], b); axes[0].set_xticks(x, names.values(), fontsize=8); axes[0].set_title("Performance: error (MASE)"); axes[0].set_ylim(0.6, 0.75)
    f = i["findings"]
    w = 0.38
    b1 = axes[1].bar(x - w / 2, [f[a]["share_last_4_weeks"] * 100 for a in names], w, color=SERIES[0], label="last 4 weeks")
    b2 = axes[1].bar(x + w / 2, [f[a]["share_seasonal_feature"] * 100 for a in names], w, color=SERIES[3], label="same week last year")
    label_bars(axes[1], b1, "{:.0f}%"); label_bars(axes[1], b2, "{:.0f}%")
    axes[1].set_xticks(x, names.values(), fontsize=8); axes[1].set_title("What drives the forecast (saliency)"); axes[1].legend(loc="upper left", fontsize=8)
    a = i["attention_vs_occlusion"]
    vals = [a["median_spearman"], a["saliency_vs_occlusion_median"]]
    b = axes[2].bar([0, 1], vals, color=[BAD, GOOD], width=0.6)
    for bb in b:
        axes[2].text(bb.get_x() + bb.get_width() / 2, bb.get_height() + (0.02 if bb.get_height() >= 0 else -0.07), f"{bb.get_height():+.2f}", ha="center", fontsize=8, color=INK2)
    axes[2].axhline(0, color=INK2, linewidth=0.8)
    axes[2].set_xticks([0, 1], ["LSTM attention\nvs occlusion", "Saliency\nvs occlusion"], fontsize=8)
    axes[2].set_title("Is the explanation faithful? (correlation)"); axes[2].set_ylim(-0.3, 0.8)
    fig.suptitle("Task 8: performance and interpretability compared with the basic model (MLP)", x=0.01, ha="left", fontsize=11)
    save(fig, "task8_1_comparison_with_basic_model")


# ---------------- Task 9 ----------------

def architecture() -> None:
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.2); ax.axis("off")

    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=color, ec="none", alpha=0.16))
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc="none", ec=color, lw=1.4))
        ax.text(x + 0.12, y + h - 0.2, title, fontsize=9.5, fontweight="bold", va="top", color=INK)
        ax.text(x + 0.12, y + h - 0.55, body, fontsize=8, va="top", color=INK2, linespacing=1.35)

    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color=INK2, lw=1.2))

    box(0.1, 3.6, 2.2, 2.2, "Founder's input", "sales file (CSV/Excel)\ncash, fixed costs, margin\nstock, supplier time\ncompetitors", "#888780")
    box(2.7, 3.6, 2.3, 2.2, "Intake (rules)", "finds date / amount columns\nreads ₹, Indian/US dates,\nHindi headers, wide sheets\nrefuses thin data", "#888780")
    box(5.4, 3.6, 2.6, 2.2, "Deep Learning", "feature representation\nCNN (TCN) on 52 weeks\nhybrid: picked per shop\nwhen it backtests better", SERIES[0])
    box(8.4, 3.6, 3.5, 2.2, "Machine Learning", "prediction\nforecast choice (Auto)\ncompetitor moves: random forest\nmarket share: LightGBM", SERIES[2])
    box(5.4, 0.4, 6.5, 2.6, "Artificial Intelligence: decision making", "knowledge model (frames + semantic network + constraints)\n47 IF-THEN rules: forward chaining -> warnings, diagnosis, advice\n"
        "backward chaining -> 'why not this move?'\nsafety gate: illegal / unaffordable moves blocked\nA* search: cheapest plan of moves", SERIES[1])
    box(0.1, 0.4, 4.9, 2.6, "Website: 7 cards", "your data · next 3 months · your money\nwhat will go wrong · your market\nyour competitors · what to do next\n"
        "every card says measured / predicted / inferred\nfull reasoning trace shown", "#888780")
    arrow((2.3, 4.7), (2.7, 4.7)); arrow((5.0, 4.7), (5.4, 4.7)); arrow((8.0, 4.7), (8.4, 4.7))
    arrow((10.15, 3.6), (10.15, 3.0)); ax.text(10.3, 3.3, "facts with a confidence", fontsize=8, color=INK2)
    arrow((5.4, 1.7), (5.0, 1.7))
    ax.set_title("Task 9: end-to-end pipeline and data flow (one function, analyze(), behind a FastAPI website)", loc="left")
    save(fig, "task9_1_architecture")


def demo_scenarios() -> None:
    g = Path("demo/DEMO_GUIDE.md").read_text(encoding="utf-8")
    secs = re.split(r"\n## \d+\. ", g)[1:]
    rows = []
    for s in secs:
        title = s.split("\n")[0]
        m = re.search(r"\*\*What to do next:\*\* (.+)", s) or re.search(r"\*\*What the system says:\*\* (.+?)(?: Tip:|$)", s, re.M)
        decision = m.group(1).strip() if m else ""
        if "What the system says" in s:
            decision = "Refused: " + decision.split(".")[0]
        warn = re.search(r"\*\*Warnings:\*\*\n- (.+)", s)
        asks = re.search(r"\*\*It asks the founder:\*\*\n- (.+)", s)
        fixed = re.search(r"\*\*What it fixed by itself:\*\*\n- (.+)", s)
        note = (warn or asks or fixed)
        n = note.group(1).split(":")[0].split(" (")[0].split(". ")[0] if note else ""
        rows.append((title, decision, n if len(n) <= 52 else n[:50].rsplit(" ", 1)[0] + "..."))
    table = re.findall(r"\| (\d+) \| `[^`]+` \| [^|]+ \| (\d+)/(\d+) \|", g)
    ok = {int(i): a == b for i, a, b in table}
    fig, ax = plt.subplots(figsize=(13, 0.255 * len(rows) + 0.5))
    ax.axis("off")
    cells = [[str(k + 1), t[:52], d[:44], n, "yes" if ok.get(k + 1) else "no"] for k, (t, d, n) in enumerate(rows)]
    tb = ax.table(cellText=cells, colLabels=["#", "Situation (demo/files)", "System's decision", "Key warning / question / fix", "As intended"],
                  colWidths=[0.03, 0.29, 0.27, 0.33, 0.08], loc="upper left", cellLoc="left")
    tb.auto_set_font_size(False); tb.set_fontsize(7.8); tb.scale(1, 1.25)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_text_props(fontweight="bold"); cell.set_facecolor("#efeeea")
        if c == 4 and r > 0:
            cell.get_text().set_color(GOOD if cells[r - 1][4] == "yes" else BAD)
    ax.set_title(f"Task 9: unified pipeline tested on {len(rows)} representative inputs ({sum(ok.values())}/{len(ok)} as intended)", loc="left")
    save(fig, "task9_2_representative_inputs")


# ---------------- Task 10 ----------------

def unseen_validation() -> None:
    v = load("validation")
    by = v["by_size"]
    ov = v["overall"][0]
    groups = ["All 300"] + [f"{r['size'].title()} sellers" for r in by]
    sysv = [ov["product forecaster"]] + [r["product"] for r in by]
    naive = [ov["naive"]] + [r["naive"] for r in by]
    ma = [ov["moving average"]] + [r["moving_average"] for r in by]
    cov = [ov["P10-P90 coverage"]] + [r["coverage"] for r in by]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(len(groups)); w = 0.26
    for k, (vals, name, col) in enumerate(((sysv, "Startup IQ", SERIES[0]), (naive, "Repeat last week", "#b9b8b3"), (ma, "4-week average", SERIES[3]))):
        b = ax.bar(x + (k - 1) * w, vals, w, color=col, label=name)
        label_bars(ax, b)
    ax.set_xticks(x, groups); ax.set_ylabel("Error, MASE (lower is better)"); ax.set_ylim(0, 1.32)
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    ax.set_title("300 unseen Olist sellers, split by seller size (fairness)")
    fig.suptitle(f"Task 10: validation on unseen inputs · {v['latency_seconds']['median']:.2f} s per analysis · same answer every time ({v['deterministic']})",
                 x=0.01, ha="left", fontsize=11)
    save(fig, "task10_1_unseen_validation")


def ethics_checks() -> None:
    comp = load("competitor")
    acc = comp["errors"]["accuracy_by_size_band"]
    rows = [
        ("Safety", "Illegal move (agreeing prices with a rival)", "Always refused: rule R90 + 'legal' constraint + planner excludes it", "pass"),
        ("Safety", "Too little or broken data", "Refused with the reason (4/4 refusal files)", "pass"),
        ("Safety", "No cash figures given", "Asks for finances instead of advising (rule R84)", "pass"),
        ("Safety", "Unaffordable move", "Blocked by the knowledge-model constraints, reason shown", "pass"),
        ("Honesty", "Competitor outputs", "Labelled 'estimates, not observations'", "pass"),
        ("Honesty", "Rival price cuts", "Stated as unpredictable (2% of months)", "pass"),
        ("Honesty", "Forecast range", "Shown as a 'likely range' with the method used", "pass"),
        ("Privacy", "Uploaded files", "Processed in memory, never stored", "pass"),
        ("Fairness", "Forecast by seller size", "Best for small sellers (0.917), weakest for large (1.105)", "noted"),
        ("Fairness", "Competitor model by size", f"Accuracy {acc['small']:.0%} small vs {acc['large']:.0%} large sellers", "noted"),
        ("Bias", "Where the data comes from", "Germany, UK, Brazil; no Indian shops yet; survivors over-represented", "noted"),
        ("Explainability", "Every decision", "Shows the rules fired, the numbers, and 'why not' via backward chaining", "pass"),
        ("Legal", "Data licences", "Olist is non-commercial: academic use only", "noted"),
    ]
    fig, ax = plt.subplots(figsize=(13, 0.29 * len(rows) + 0.4))
    ax.axis("off")
    tb = ax.table(cellText=[list(r) for r in rows], colLabels=["Area", "Check", "What the system does", "Result"],
                  colWidths=[0.1, 0.25, 0.55, 0.08], loc="upper left", cellLoc="left")
    tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.3)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_text_props(fontweight="bold"); cell.set_facecolor("#efeeea")
        elif c == 3:
            cell.get_text().set_color(GOOD if rows[r - 1][3] == "pass" else SERIES[0])
    ax.set_title("Task 10: ethical, fairness and bias-related checks", loc="left")
    save(fig, "task10_2_ethics_fairness")


def main(argv):
    retrain = "--retrain" in argv
    training_curves(retrain); model_comparison(); regularisation(); daily_comparison()
    interpret_comparison(); architecture(); demo_scenarios(); unseen_validation(); ethics_checks()


if __name__ == "__main__":
    main(sys.argv[1:])
