"""Plain matplotlib images for Tasks 7 and 8 into docs/output/task7|task8. Run: PYTHONPATH=src python -m startupiq.pipelines.output_figures_78"""
import json, os, shutil
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

r7 = json.load(open("docs/results/task7/results_task7.json")); r8 = json.load(open("docs/results/task8/results_task8.json"))
O7, O8 = "docs/output/task7", "docs/output/task8"; os.makedirs(O7, exist_ok=True); os.makedirs(O8, exist_ok=True)
ms = ["LR", "CNN", "BiLSTM"]; col = ["#2a78d6", "#888888", "#bbbbbb"]
def save(fig, p): fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)

# ---- Task 7 ----
for key, ci, name, title in (("macro_f1", "macro_f1_ci", "1_test_macro_f1.png", "Test macro-F1 (95% CI)"), ("accuracy", "acc_ci", "2_test_accuracy.png", "Test accuracy (95% CI)")):
    v = [r7["test"][m][key] for m in ms]; lo = [v[i] - r7["test"][m][ci][0] for i, m in enumerate(ms)]; hi = [r7["test"][m][ci][1] - v[i] for i, m in enumerate(ms)]
    fig, ax = plt.subplots(figsize=(5, 3.6)); ax.bar(ms, v, yerr=[lo, hi], color=col, capsize=4)
    for i, x in enumerate(v): ax.text(i, 0.02, f"{x:.3f}", ha="center", color="white")
    ax.set_ylabel(key.replace("_", "-")); ax.set_title(title); ax.set_ylim(0, 1); save(fig, f"{O7}/{name}")
tr = r7["cnn_random_search"]; fig, ax = plt.subplots(figsize=(5, 3.6))
ax.bar([f"trial {i+1}" for i in range(len(tr))], [t["val_f1"] for t in tr], color="#888888")
ax.axhline(r7["decision"]["val_macro_f1"]["LR"], color="#2a78d6", ls="--", label="LR validation")
ax.set_ylabel("validation macro-F1"); ax.set_title("CNN random search vs LR"); ax.legend(); save(fig, f"{O7}/3_cnn_random_search.png")
fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.4))
axs[0].bar(ms, [r7["test"][m]["params"] / 1e6 for m in ms], color=col); axs[0].set_title("Parameters (millions)")
axs[1].bar(ms, [r7["test"][m]["fit_s"] for m in ms], color=col); axs[1].set_title("Fit time (seconds)"); save(fig, f"{O7}/4_cost.png")

# ---- Task 8 ----
src = "docs/results/task8/figures"
shutil.copy(f"{src}/t8_perturbation.png", f"{O8}/1_perturbation.png"); shutil.copy(f"{src}/t8_calibration.png", f"{O8}/2_calibration.png")
al = r8["accuracy_by_length"]; fig, ax = plt.subplots(figsize=(5.2, 3.6)); w = 0.25
for j, m in enumerate(ms): ax.bar(np.arange(3) + (j - 1) * w, [al[m][k] for k in ("short", "medium", "long")], w, label=m, color=col[j])
ax.set_xticks(range(3)); ax.set_xticklabels(["short", "medium", "long"]); ax.set_ylabel("accuracy"); ax.set_title("Accuracy by description length"); ax.legend(); ax.set_ylim(0, 1)
save(fig, f"{O8}/3_accuracy_by_length.png")
fig, ax = plt.subplots(figsize=(5, 3.6)); a = r8["attention_vs_erasure_spearman"]["mean"]; s = r8["cnn_saliency_vs_erasure_spearman"]["mean"]
ax.bar(["BiLSTM attention", "CNN saliency"], [a, s], color=["#bbbbbb", "#888888"]); ax.set_ylim(0, 1); ax.set_ylabel("Spearman vs token erasure")
ax.set_title("Do highlighted words matter? (150 texts)"); 
for i, x in enumerate([a, s]): ax.text(i, x + 0.02, f"{x:.2f}", ha="center")
save(fig, f"{O8}/4_explanation_agreement.png")
