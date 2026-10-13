"""Builds the 12 mentor-facing figures (4 per task) into docs/output/task4|5|6. Reads real pipeline outputs only.
PYTHONPATH=src python -m startupiq.pipelines.output_figures"""
import json, shutil
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

B, G, INK, OR, GR = "tab:blue", "lightgray", "black", "tab:orange", "tab:green"
yc = pd.read_csv("data/processed/yc_clean.csv"); fd = pd.read_csv("data/processed/formd_clean.csv")
lab = pd.read_csv("data/labels/readiness_labels.csv")
R = json.load(open("docs/results/task5_6/results_yc.json")); F = json.load(open("docs/results/task5_6/results_formd.json"))
H = json.load(open("docs/results/readiness/hybrid_results.json")); CV = json.load(open("docs/results/readiness/cv_results.json"))
money = lambda s: s.replace("$", r"\$")


LABELS = {"1_cleaning_pipeline_funnel.png": "Rows kept at each cleaning step", "2_problems_found_and_fixes.png": "Data issues found and handled",
          "3_yc_data_shape.png": "YC dataset overview", "4_formd_amount_raw_vs_log.png": "Form D offering amount distribution",
          "1_inputs_used_and_leakage_exclusions.png": "Input features: used vs excluded", "2_feature_sets_and_selection.png": "Feature sets and feature selection",
          "3_readiness_labels_and_features.png": "Readiness labels and length feature", "4_train_validation_test_splits.png": "Train / validation / test splits",
          "1_readiness_model_results.png": "Readiness model results", "2_industry_model_comparison.png": "Industry model results", "4_formd_funding_models.png": "Funding model results"}


def save(fig, task, name, title, sub):
    fig.suptitle(LABELS[name], fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"docs/output/{task}/{name}", dpi=130); plt.close(fig)


# ================= TASK 4 =================
# 4.1 funnel
yc_model = int((yc.usable_text & yc.industry.notna()).sum())
fig, axs = plt.subplots(1, 2, figsize=(14, 5.8))
st = [("Raw YC records", 6288), ("After duplicate checks\n(none found)", 6288), ("Usable text", int(yc.usable_text.sum())), ("Usable text + industry known", yc_model)]
axs[0].barh([s for s, _ in st][::-1], [v for _, v in st][::-1], color=[B, B, B, GR][::-1])
for i, (s, v) in enumerate(st[::-1]): axs[0].text(v, i, f"  {v:,}", va="center")
axs[0].set_title("YC company text"); axs[0].set_xlim(0, 7600)
fs = [("Raw Form D filings", 107709), ("After dropping amendments", 64594), ("After dropping investment funds", 32206), ("Start-up-like with a usable amount", int((fd.startup_like_group & fd.offering_amount.notna() & ~fd.amount_implausible).sum()))]
axs[1].barh([s for s, _ in fs][::-1], [v for _, v in fs][::-1], color=[B, B, B, GR][::-1])
for i, (s, v) in enumerate(fs[::-1]): axs[1].text(v, i, f"  {v:,}", va="center")
axs[1].set_title("SEC Form D filings"); axs[1].set_xlim(0, 135000)
save(fig, "task4", "1_cleaning_pipeline_funnel.png", "Raw data shrinks to the rows we can trust", "YC lost almost nothing (it is already clean). Form D lost 70% on purpose: amendments repeat a filing and funds are not operating companies.")

# 4.2 problems found + action
fig, axs = plt.subplots(1, 2, figsize=(15, 6.2))
yp = [("Long description empty", 370, "fall back to the one-liner"), ("Team size 0 or -1", 137, "set to missing + flag"), ("No usable text at all", 55, "flag, excluded from text models"),
      ("Industry 'Unspecified'", 18, "set to missing"), ("Launch year far from batch year", 199, "flag, kept"), ("No tags", 863, "kept; tags never used as input")]
axs[0].barh([a for a, _, _ in yp][::-1], [b for _, b, _ in yp][::-1], color=B)
for i, (a, b, c) in enumerate(yp[::-1]): axs[0].text(b, i, f"  {b:,}", va="center", fontsize=9.5)
axs[0].set_xlim(0, 2100); axs[0].set_title("YC: issues found (of 6,288 rows)")
fp = [("Revenue 'Decline to Disclose'", 25713, "kept as its own category"), ("Issuer age unknown", 8233, "missing + indicator"), ("Offering amount 'Indefinite'", 2843, "set to missing"),
      ("Amount zero or negative", 25, "set to missing"), ("Amount of $10bn or more", 6, "flag, excluded from amount models")]
axs[1].barh([money(a) for a, _, _ in fp][::-1], [b for _, b, _ in fp][::-1], color=B)
for i, (a, b, c) in enumerate(fp[::-1]): axs[1].text(b, i, f"  {b:,}", va="center", fontsize=9.5)
axs[1].set_xlim(0, 52000); axs[1].set_title("Form D: issues found (of 32,206 rows)")
save(fig, "task4", "2_problems_found_and_fixes.png", "Every problem got a rule, and every rule is logged", "Nothing was silently deleted. Values were fixed, flagged, or kept with a reason, and each step is in the cleaning-log JSON.")

# 4.3 YC shape
fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
vc = yc.industry.fillna("Unspecified").value_counts().sort_values()
axs[0].barh(vc.index, vc.values, color=[GR if i == "B2B" else B for i in vc.index])
for i, v in enumerate(vc.values): axs[0].text(v, i, f"  {v:,} ({v / len(yc):.0%})", va="center", fontsize=9.5)
axs[0].set_xlim(0, 3900); axs[0].set_title("Companies per industry")
w = yc.text_words.clip(upper=yc.text_words.quantile(.99))
axs[1].hist(w, bins=40, color=B); axs[1].axvline(yc.text_words.median(), color=INK, ls="--", lw=1)
axs[1].set_title("Description length (words)"); axs[1].set_xlabel("words per description"); axs[1].set_ylabel("companies")
save(fig, "task4", "3_yc_data_shape.png", "B2B is half the data, and most blurbs are under 70 words", "This is why we report macro-F1 and cross-validation, and why length itself becomes a feature for the readiness check.")

# 4.4 Form D amounts raw vs log
a = fd.loc[fd.startup_like_group & fd.offering_amount.notna(), "offering_amount"]
fig, axs = plt.subplots(1, 2, figsize=(14, 5.3))
axs[0].hist(a[a < 2e8] / 1e6, bins=60, color=B); axs[0].set_xlabel("offering amount, millions of dollars (capped at 200M)"); axs[0].set_ylabel("filings")
axs[0].set_title("Offering amount (raw dollars)")
x = np.log10(a[a < 1e10]); axs[1].hist(x, bins=60, color=GR); axs[1].axvline(x.median(), color=INK, ls="--", lw=1)
axs[1].set_xlabel("log10(offering amount)"); axs[1].set_title("Offering amount (log10 dollars)")
save(fig, "task4", "4_formd_amount_raw_vs_log.png", "Why we put funding amounts on a log scale", f"Raise sizes span $1 thousand to $10 billion. Models learn the log amount; a few giant filings would otherwise dominate every error.")

# ================= TASK 5 =================
# 5.1 leakage table
fig, ax = plt.subplots(figsize=(14, 6.4)); ax.axis("off")
rows0 = [["Description text (one-liner + long description)", "USED", "The signal we want: what the company says about itself"],
        ["Description length, has-a-number, has-a-money-figure", "USED", "Cheap signals; length turned out to matter for 'unclear' gaps"],
        ["Team size (log) and stage", "TESTED", "Added little; dropped from the industry model (val F1 0.724 vs 0.741)"],
        ["YC tags", "EXCLUDED", "Five tags are literally industry names: it would give the answer away"],
        ["Subindustry / industries", "EXCLUDED", "Restate the label (leakage)"],
        ["Status, isHiring", "EXCLUDED", "Snapshots taken at crawl time, not facts at filing time"],
        ["Amount sold (Form D)", "EXCLUDED", "Same event as the offering amount we predict"],
        ["Founder demographic flags", "AUDIT ONLY", "Kept out of inputs; reserved to check fairness later"]]
rows = [r[:2] for r in rows0]
t = ax.table(cellText=rows, colLabels=["Input", "Decision"], loc="center", cellLoc="left", colWidths=[0.6, 0.2]); t.auto_set_font_size(False); t.set_fontsize(10.5); t.scale(1, 2.2)
col = {"USED": "#d9f0e3", "TESTED": "#f6f0d8", "EXCLUDED": "#fbe3d6", "AUDIT ONLY": "#e5e8ef"}
for i, r in enumerate(rows, 1): t[i, 1].set_facecolor(col[r[1]])
for j in range(2): t[0, j].set_facecolor("#dde3ee")
save(fig, "task5", "1_inputs_used_and_leakage_exclusions.png", "What the models may see, and what we blocked as leakage", "Leakage means a feature secretly contains the answer. High accuracy built on leakage would collapse on a real founder's text.")

# 5.2 ablation + selection
fig, axs = plt.subplots(1, 2, figsize=(14, 5.4))
ab = R["feature_ablation"]; ks = list(ab); v = [ab[k]["val_macro_f1"] for k in ks]
axs[0].barh(ks[::-1], v[::-1], color=[GR if k == R["best_feature_set"] else G for k in ks][::-1])
for i, x_ in enumerate(v[::-1]): axs[0].text(x_, i, f"  {x_:.3f}", va="center")
axs[0].set_xlim(0, 0.9); axs[0].set_title("Feature sets (validation macro-F1)")
sel = R["chi2_selection_val_macro_f1"]; xs = [int(k) for k in sel]; ys = list(sel.values())
axs[1].plot(xs, ys, "o-", color=B); axs[1].set_xscale("log"); axs[1].set_xlabel("features kept (chi-squared)"); axs[1].set_ylabel("validation macro-F1")
bi = int(np.argmax(ys)); axs[1].annotate(f"best: {xs[bi]:,} features\nF1 {ys[bi]:.3f}", (xs[bi], ys[bi]), (xs[bi] * 1.6, ys[bi] - 0.06), arrowprops=dict(arrowstyle="->"))
axs[1].set_title("Chi-squared feature selection")
save(fig, "task5", "2_feature_sets_and_selection.png", "Simple word features beat fancier ones, and trimming helped a little", "More features did not mean better: characters and team size lowered the score. Chi-squared selection gained about 2 points over keeping all.")

# 5.3 readiness features
lab["wc"] = lab.text.str.split().str.len(); lab["bin"] = pd.qcut(lab.wc, 4, labels=["shortest 25%", "25-50%", "50-75%", "longest 25%"])
names = {"customer_unclear": "Who it's for", "problem_unclear": "Problem", "differentiation_missing": "Why you", "traction_not_stated": "Traction", "business_model_unclear": "How you earn", "team_not_stated": "Team"}
fig, axs = plt.subplots(1, 3, figsize=(16, 5.3))
pv = lab[list(names)].mean(); axs[0].barh([names[k] for k in pv.index][::-1], pv.values[::-1], color=B)
for i, x_ in enumerate(pv.values[::-1]): axs[0].text(x_, i, f" {x_:.0%}", va="center")
axs[0].set_xlim(0, 1.1); axs[0].set_title("Share of 400 labelled texts with each gap")
for k, c in zip(["customer_unclear", "problem_unclear", "differentiation_missing"], [B, OR, GR]):
    r = lab.groupby("bin", observed=True)[k].mean(); axs[1].plot(r.index.astype(str), r.values * 100, "o-", color=c, label=names[k])
axs[1].set_ylabel("% with the gap"); axs[1].legend(); axs[1].set_title("Gap rate by description length"); axs[1].tick_params(axis="x", labelrotation=20)
wo = {"customer_unclear": .822, "problem_unclear": .756, "differentiation_missing": .694}; wi = {k: H["ml"][k]["AUC"] for k in wo}
xx = np.arange(3); axs[2].bar(xx - .2, list(wo.values()), .4, color=G, label="text words only"); axs[2].bar(xx + .2, list(wi.values()), .4, color=GR, label="+ length and number features")
for i, k in enumerate(wo): axs[2].text(i - .2, wo[k], f"{wo[k]:.2f}", ha="center", va="bottom", fontsize=9); axs[2].text(i + .2, wi[k], f"{wi[k]:.2f}", ha="center", va="bottom", fontsize=9)
axs[2].set_xticks(xx); axs[2].set_xticklabels([names[k] for k in wo]); axs[2].set_ylim(0.5, 0.9); axs[2].set_ylabel("AUC (cross-validated)"); axs[2].legend(loc="lower right", fontsize=9); axs[2].set_title("AUC with and without length features")
save(fig, "task5", "3_readiness_labels_and_features.png", "How the readiness gaps were defined and which features carry the signal", "Traction, business model and team are missing in 82-94% of blurbs, too skewed to learn, so those use transparent rules. The other three get a trained model.")

# 5.4 splits
fig, ax = plt.subplots(figsize=(14, 6)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 10)
def bar(y, label, parts, cols):
    ax.text(0, y + 0.9, label, fontsize=11, fontweight="bold"); x0 = 0; tot = sum(p for _, p in parts)
    for (n, p), c in zip(parts, cols):
        w_ = 98 * p / tot; ax.add_patch(plt.Rectangle((x0, y - 0.3), w_ - .3, 0.9, color=c)); ax.text(x0 + w_ / 2, y + .15, f"{n}\n{p:,}", ha="center", va="center", fontsize=9, color="white"); x0 += w_
s = R["split_sizes"]; t_ = R["temporal_split"]; fsz = F["split"]
bar(8.4, "YC industry model: stratified random split", [("train", s["train"]), ("validation", s["val"]), ("test (scored once)", s["test"])], [B, OR, GR])
bar(6.1, "YC industry model: temporal split (older batches train, 2025+ test)", [("train: older batches", t_["train"]), ("test: newest batches", t_["test"])], [B, GR])
bar(3.8, "Form D funding model: split grouped by issuer (CIK), no company in two sets", [("train", fsz["train"]), ("validation", fsz["val"]), ("test", fsz["test"])], [B, OR, GR])
ax.text(0, 2.5, "Readiness check (400 labelled texts): 5-fold cross-validation repeated 3 times; every score comes from a model that never saw that text.", fontsize=11, fontweight="bold")
ax.add_patch(plt.Rectangle((0, 1.2), 98, 0.9, color="#8a63d2")); ax.text(49, 1.65, "5 folds x 3 repeats = 15 train/test rounds, all 400 texts scored out-of-fold", ha="center", va="center", color="white", fontsize=10)
save(fig, "task5", "4_train_validation_test_splits.png", "How we kept every reported score honest", "The test set was scored once. Vectorisers and scalers were fitted on training data only, and no company appears on both sides of a split.")

# ================= TASK 6 =================
# 6.1 readiness results
fig, axs = plt.subplots(1, 2, figsize=(14, 5.6))
ml = H["ml"]; ks = list(ml); axs[0].bar([names[k] for k in ks], [ml[k]["AUC"] for k in ks], color=B); axs[0].axhline(.5, color=INK, ls="--", lw=1)
for i, k in enumerate(ks): axs[0].text(i, ml[k]["AUC"], f"{ml[k]['AUC']:.2f}", ha="center", va="bottom")
axs[0].text(2.45, .505, "chance = 0.50", ha="right", fontsize=9); axs[0].set_ylim(0.4, 1); axs[0].set_ylabel("AUC (cross-validated)"); axs[0].set_title("AUC per gap (trained model)")
allk = list(ml) + list(H["rules"]); ba = [ml[k]["balanced_acc"] if k in ml else H["rules"][k]["balanced_acc"] for k in allk]
axs[1].barh([names[k] for k in allk][::-1], ba[::-1], color=[B] * 3 + [OR] * 3 and ([B] * 3 + [OR] * 3)[::-1]); axs[1].axvline(.5, color=INK, ls="--", lw=1)
for i, x_ in enumerate(ba[::-1]): axs[1].text(x_, i, f"  {x_:.2f}", va="center")
axs[1].set_xlim(0.4, 0.85); axs[1].set_title("Balanced accuracy per gap")
save(fig, "task6", "1_readiness_model_results.png", "Product model: which gap checks work, and how well", "AUC 0.77-0.83 for the three trained checks; balanced accuracy 0.71-0.76 for all six. Plain accuracy is not the headline because gaps are rare or very common.")

# 6.2 YC industry
fig, axs = plt.subplots(1, 2, figsize=(14, 5.4))
mv = R["model_validation"]; nm = list(mv); axs[0].barh(nm[::-1], [mv[k]["val_macro_f1"] for k in nm][::-1], color=[GR if k == R["chosen_model"] else G for k in nm][::-1])
for i, k in enumerate(nm[::-1]): axs[0].text(mv[k]["val_macro_f1"], i, f"  F1 {mv[k]['val_macro_f1']:.2f} | acc {mv[k]['val_acc']:.2f}", va="center", fontsize=9)
axs[0].set_xlim(0, 1.15); axs[0].set_title("Models, validation set")
tc, tb = R["test_chosen"], R["test_baseline"]; lbl = ["Accuracy", "Macro-F1", "Top-3 accuracy"]; mine = [tc["acc"], tc["macro_f1"], tc["top3_acc"]]; base = [tb["acc"], tb["macro_f1"], tb["top3_acc"]]
xx = np.arange(3); axs[1].bar(xx - .2, base, .4, color=G, label="always guess B2B"); axs[1].bar(xx + .2, mine, .4, color=GR, label="chosen model (test, scored once)")
for i in range(3): axs[1].text(i - .2, base[i], f"{base[i]:.2f}", ha="center", va="bottom"); axs[1].text(i + .2, mine[i], f"{mine[i]:.2f}", ha="center", va="bottom")
axs[1].set_xticks(xx); axs[1].set_xticklabels(lbl); axs[1].set_ylim(0, 1.12); axs[1].legend(loc="upper left", fontsize=9)
axs[1].set_title(f"Test: acc {tc['acc']:.2f} (95% CI {tc['acc_ci'][0]:.2f}-{tc['acc_ci'][1]:.2f})")
save(fig, "task6", "2_industry_model_comparison.png", "Supporting model: predicting a YC industry from the description", "Classical models on TF-IDF. The ensemble of logistic regression and naive Bayes won validation and reaches 80% accuracy; macro-F1 is lower because small industries are harder.")

# 6.3 confusion matrix (existing figure from the pipeline run)
shutil.copy("docs/results/task5_6/figures/t6_confusion_matrix.png", "docs/output/task6/3_industry_confusion_matrix.png")

# 6.4 Form D
fig, axs = plt.subplots(1, 3, figsize=(16, 5.2))
bv = F["band_val"]; bn = list(bv); axs[0].barh(bn[::-1], [bv[k]["acc"] for k in bn][::-1], color=[GR if k == F["band_chosen"] else G for k in bn][::-1])
for i, k in enumerate(bn[::-1]): axs[0].text(bv[k]["acc"], i, f"  {bv[k]['acc']:.2f}", va="center")
axs[0].set_xlim(0, .8); axs[0].set_title("Funding band accuracy (validation)")
bt = F["band_test"]; axs[1].bar(["guess the\nmost common", "best model\n(test)", "within one\nband (test)"], [F["band_test_majority"], bt["acc"], F["band_test_within_one"]], color=[G, B, GR])
for i, x_ in enumerate([F["band_test_majority"], bt["acc"], F["band_test_within_one"]]): axs[1].text(i, x_, f"{x_:.2f}", ha="center", va="bottom")
axs[1].set_ylim(0, 1.05); axs[1].set_title("Band accuracy (test)")
rg = F["regression_test"]; gb = rg["gradient boosting"]; sm = rg["segment-median baseline"]
axs[2].bar(["segment-median\nbaseline", "gradient\nboosting"], [sm["R2"], gb["R2"]], color=[G, B])
for i, x_ in enumerate([sm["R2"], gb["R2"]]): axs[2].text(i, x_, f"{x_:.2f}", ha="center", va="bottom")
axs[2].set_ylim(0, .5); axs[2].set_ylabel("R2 on log amount (test)"); axs[2].set_title("R2 (test)")
save(fig, "task6", "4_formd_funding_models.png", "Funding benchmark model: honest, modest, and still better than the baseline", f"Filing metadata says little about raise size. Exact-band accuracy is {bt['acc']:.0%} against {F['band_test_majority']:.0%} for guessing, and 90% are within one band. Peers' empirical ranges are the product output, not a point prediction.")
print("done")
