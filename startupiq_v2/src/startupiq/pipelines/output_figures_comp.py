"""Plain images for Tasks 7 and 8 (competitor finder) into docs/output/task7|task8."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
a = json.load(open("docs/results/task7_competitors/results_task7_competitors.json")); b = json.load(open("docs/results/task8_competitors/results_task8_competitors.json"))
O7, O8 = "docs/output/task7", "docs/output/task8"
def save(f, p): f.tight_layout(); f.savefig(p, dpi=130); plt.close(f)
names = ["TFIDF", "LSA", "CNN", "HYBRID"]; T = a["test"]
# T7-1 precision@10 with CI
v = [a["test"]["random_p10"]] + [T[n]["p10"] for n in names]; lab = ["random"] + names
err = [[0] + [T[n]["p10"] - T[n]["p10_ci"][0] for n in names], [0] + [T[n]["p10_ci"][1] - T[n]["p10"] for n in names]]
f, ax = plt.subplots(figsize=(5.6, 3.6)); ax.bar(lab, v, yerr=err, capsize=4, color=["#cccccc", "#888", "#2a78d6", "#888", "#888"])
for i, x in enumerate(v): ax.text(i, 0.01, f"{x:.2f}", ha="center", color="white" if i else "black")
ax.set_ylabel("precision@10"); ax.set_title("Test: share of 10 competitors that share a tag"); save(f, f"{O7}/1_competitor_precision.png")
# T7-2 tag prediction
t = a["tags_test"]; f, axs = plt.subplots(1, 2, figsize=(6.2, 3.4))
for ax, k, ttl in ((axs[0], "micro_AP", "Micro average precision"), (axs[1], "macro_AUC", "Macro AUC")):
    ax.bar(["TF-IDF\nLR", "CNN"], [t["TFIDF_LR"][k], t["CNN"][k]], color=["#888", "#2a78d6"]); ax.set_title(ttl, fontsize=9)
    if k == "micro_AP": ax.axhline(t["random_micro_AP"], color="gray", ls="--", label="random"); ax.legend(fontsize=7)
    for i, x in enumerate([t["TFIDF_LR"][k], t["CNN"][k]]): ax.text(i, x + 0.01, f"{x:.2f}", ha="center")
save(f, f"{O7}/2_tag_prediction.png")
# T7-3 validation decision
d = a["decision"]; f, ax = plt.subplots(figsize=(5.4, 3.6)); ax.bar(names, [d["val_p10"][n] for n in names], color=["#888", "#2a78d6", "#888", "#888"])
ax.axhline(d["lower_bound"], color="red", ls="--", label=f"lower bound of best = {d['lower_bound']}"); ax.axhline(d["random_p10"], color="gray", ls=":", label="random")
ax.set_ylabel("validation precision@10"); ax.set_title("Decision rule: simplest model above the line ships"); ax.legend(fontsize=7); save(f, f"{O7}/3_decision_validation.png")
# T7-4 same industry
f, ax = plt.subplots(figsize=(5.2, 3.6)); ax.bar(names, [T[n]["same_industry_at_10"] for n in names], color="#888")
for i, n in enumerate(names): ax.text(i, T[n]["same_industry_at_10"] + 0.01, f"{T[n]['same_industry_at_10']:.2f}", ha="center")
ax.set_ylabel("same industry @10"); ax.set_title("Sanity check: competitors in the same industry"); ax.set_ylim(0, 1); save(f, f"{O7}/4_same_industry.png")
# T8-1 faithfulness
fa = b["faithfulness"]; f, ax = plt.subplots(figsize=(5.4, 3.6)); w = 0.35; x = np.arange(2)
ax.bar(x - w / 2, [fa[k]["drop_top3_why_words"] for k in ("LSA", "CNN")], w, label="delete the 3 'why' words", color="#2a78d6")
ax.bar(x + w / 2, [fa[k]["drop_3_random_words"] for k in ("LSA", "CNN")], w, label="delete 3 random words", color="#bbb")
ax.set_xticks(x); ax.set_xticklabels(["LSA", "CNN"]); ax.set_ylabel("drop in similarity to top competitor"); ax.set_title("Do the explanation words really matter?"); ax.legend(fontsize=7); save(f, f"{O8}/1_explanation_faithfulness.png")
# T8-2 robustness
r = b["robustness"]; ks = ["typos (10% of words)", "random 30% of words removed", "first sentence only"]
f, ax = plt.subplots(figsize=(5.8, 3.6)); ax.bar([k.replace(" of words", "\nof words").replace("random ", "random\n") for k in ks], [r[k]["top10_overlap_with_original"] for k in ks], color="#888")
for i, k in enumerate(ks): ax.text(i, r[k]["top10_overlap_with_original"] + 0.01, f"{r[k]['top10_overlap_with_original']:.2f}", ha="center")
ax.set_ylim(0, 1); ax.set_ylabel("share of top-10 list that stays"); ax.set_title("Competitor list after changing the description"); ax.tick_params(axis="x", labelsize=7); save(f, f"{O8}/2_robustness.png")
# T8-3 similarity vs precision
q = b["p10_by_top1_similarity_quintile"]; f, ax = plt.subplots(figsize=(5.2, 3.6)); ax.bar(list(q), list(q.values()), color="#2a78d6")
ax.axhline(a["test"]["random_p10"], color="gray", ls=":", label="random"); ax.set_xlabel("quintile of best-match similarity (Q1 lowest)"); ax.set_ylabel("precision@10")
ax.set_title("Low similarity = less reliable list"); ax.legend(fontsize=7); save(f, f"{O8}/3_confidence_vs_precision.png")
# T8-4 by industry
bi = b["p10_by_industry"]; f, ax = plt.subplots(figsize=(6.4, 3.8)); ks = list(bi); ax.bar([f"{k[:12]}\n(n={bi[k]['n']})" for k in ks], [bi[k]["p10"] for k in ks], color="#888")
ax.set_ylabel("precision@10"); ax.set_title("Finder quality by industry"); ax.tick_params(axis="x", labelsize=7); save(f, f"{O8}/4_quality_by_industry.png")
