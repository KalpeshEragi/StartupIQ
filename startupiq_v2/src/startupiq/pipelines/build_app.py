"""Inlines the trained model and its cross-validated stats into app/template.html -> app/startupiq_readiness.html"""
import json
m = json.load(open("app/readiness_model.json"))
m["vocab_idf"] = {str(i): idf for _, (i, idf) in m["vocab"].items()}
m["vocab"] = {t: [i] for t, (i, _) in m["vocab"].items()}
h = json.load(open("docs/results/readiness/hybrid_results.json"))
stats = {}
for g, s in h["ml"].items(): stats[g] = {"prevalence": s["prevalence"], "auc": s["AUC"], "balanced_acc": s["balanced_acc"], "recall": s["recall_of_gaps"]}
for g, s in h["rules"].items(): stats[g] = {"prevalence": s["prevalence"], "auc": None, "balanced_acc": s["balanced_acc"], "recall": s["recall_of_gaps"]}
t = open("app/template.html").read().replace("__MODEL__", json.dumps(m, separators=(",", ":"))).replace("__STATS__", json.dumps(stats))
open("app/startupiq_readiness.html", "w").write(t); print(len(t) // 1024, "KB")
