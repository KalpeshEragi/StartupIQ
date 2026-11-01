"""Train and compare the expense categorisers on the IPSA data.

    python -m categoriser.train            (from ml/)

Protocol
- Splits by financial year (ipsa.py): fit on train, choose the model and its confidence threshold on validation,
  score the test years once.
- Every model sees the same features: TF-IDF of word 1-2 grams and character 3-5 grams of the cleaned text, plus the
  amount in log bins. Vectorisers are fitted on training text only.
- Training rows are capped per category so the large classes (stationery, phones) do not drown the small ones.
- Reported: accuracy, macro-F1, and the share of transactions that can be filed automatically while keeping
  precision >= 95% (threshold chosen on validation). Also on the "new wording" subset: texts never seen in training.

Writes ../docs/results/categoriser.md and .json, and models/categoriser.joblib (the chosen model, refit on all years).
"""
from __future__ import annotations

import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

from categoriser import ipsa
from categoriser.data import ROOT
from categoriser.keywords import keyword_category
from categoriser.taxonomy import EXPENSE_CATEGORIES

RESULTS = ROOT.parent / "docs" / "results"
MODELS = ROOT / "models"
RUNS = MODELS / "runs"
PER_CLASS = 25_000
TARGET_PRECISION = 0.95
SEED = 0
AMOUNT_BINS = [0, 5, 10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, np.inf]


class Features:
    """Text TF-IDF (words + characters) and amount bins, fitted on training rows only."""

    def fit(self, df: pd.DataFrame) -> Features:
        self.words = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=60_000, sublinear_tf=True)
        self.chars = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=120_000, sublinear_tf=True)
        self.amount = OneHotEncoder(handle_unknown="ignore")
        self.words.fit(df.text)
        self.chars.fit(df.text)
        self.amount.fit(self._bins(df.amount))
        return self

    @staticmethod
    def _bins(amount: pd.Series) -> np.ndarray:
        return np.digitize(np.abs(amount.to_numpy()), AMOUNT_BINS).reshape(-1, 1)

    def transform(self, df: pd.DataFrame) -> sparse.csr_matrix:
        return sparse.hstack([self.words.transform(df.text), self.chars.transform(df.text),
                              self.amount.transform(self._bins(df.amount))]).tocsr()


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Model:
    """A classifier plus how to get class scores out of it; reduce=True adds chi-squared feature selection."""

    def __init__(self, name: str, estimator, scores: str = "proba", reduce: int | None = None):
        self.name, self.estimator, self.scores, self.reduce = name, estimator, scores, reduce

    def fit(self, X, y):
        if self.reduce:
            self.selector = SelectKBest(chi2, k=self.reduce).fit(X, y)
            X = self.selector.transform(X)
        self.estimator.fit(X, y)
        return self

    def predict_scores(self, X) -> np.ndarray:
        if self.reduce:
            X = self.selector.transform(X)
        if self.scores == "proba":
            return self.estimator.predict_proba(X)
        return _softmax(self.estimator.decision_function(X))  # SVM margins -> comparable confidences


def candidates() -> list[Model]:
    from lightgbm import LGBMClassifier
    return [
        Model("Naive Bayes (complement)", ComplementNB(alpha=0.3)),
        Model("Logistic regression", LogisticRegression(C=8.0, max_iter=300, solver="saga", random_state=SEED)),
        Model("Linear SVM", LinearSVC(C=0.5, random_state=SEED), scores="margin"),
        # Lighter than the linear models' feature set: with 14 classes LightGBM grows 14 trees per round, and 400 rounds on
        # 20,000 features ran for over an hour without finishing on this machine.
        Model("LightGBM", LGBMClassifier(n_estimators=150, learning_rate=0.15, num_leaves=31, min_child_samples=20,
                                         colsample_bytree=0.5, random_state=SEED, verbose=-1, n_jobs=-1), reduce=5_000),
        # Kept small so it trains in minutes on a CPU (scikit-learn has no GPU support).
        Model("Neural network (MLP)", MLPClassifier(hidden_layer_sizes=(128,), alpha=1e-4, batch_size=1024, max_iter=8,
                                                    early_stopping=True, n_iter_no_change=2, random_state=SEED), reduce=5_000),
    ]


def coverage_at_precision(conf: np.ndarray, correct: np.ndarray, target: float) -> tuple[float, float]:
    """Lowest confidence threshold whose auto-filed share keeps precision >= target; returns (threshold, coverage)."""
    order = np.argsort(-conf)
    precision = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    ok = np.where(precision >= target)[0]
    if len(ok) == 0:
        return 1.01, 0.0
    k = ok.max()
    return float(conf[order][k]), (k + 1) / len(order)


def evaluate(y: np.ndarray, pred: np.ndarray, conf: np.ndarray | None, threshold: float | None, new: np.ndarray) -> dict:
    out = {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
           "accuracy_new_wording": accuracy_score(y[new], pred[new]),
           "macro_f1_new_wording": f1_score(y[new], pred[new], average="macro")}
    if conf is not None and threshold is not None:
        filed = conf >= threshold
        out["auto_filed"] = filed.mean()
        out["precision_when_filed"] = (pred[filed] == y[filed]).mean() if filed.any() else float("nan")
    return out


def table(rows: list[dict], cols: list[str]) -> str:
    fmt = lambda v: f"{v:.1%}" if isinstance(v, float) and v <= 1.0 else (f"{v:.3f}" if isinstance(v, float) else str(v))  # noqa: E731
    head = "| " + " | ".join(cols) + " |\n| " + " | ".join("---" for _ in cols) + " |"
    return head + "".join("\n| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |" for r in rows)


def cap_per_class(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return pd.concat([g.sample(min(len(g), n), random_state=SEED) for _, g in df.groupby("category")])


def run() -> dict:
    t0 = time.time()
    data = ipsa.load()
    train_all = data[data.split == "train"]
    train = cap_per_class(train_all, PER_CLASS)
    val, test = data[data.split == "val"], data[data.split == "test"]
    seen = set(train_all.text)
    new_val, new_test = ~val.text.isin(seen).to_numpy(), ~test.text.isin(seen).to_numpy()
    classes = np.array(sorted(EXPENSE_CATEGORIES))
    print(f"train {len(train):,} (of {len(train_all):,}), val {len(val):,}, test {len(test):,}", flush=True)

    RUNS.mkdir(parents=True, exist_ok=True)
    feats_path = RUNS / f"features-{PER_CLASS}.joblib"  # fitted once; every cached model below depends on exactly these features
    feats = joblib.load(feats_path) if feats_path.exists() else Features().fit(train)
    joblib.dump(feats, feats_path)
    Xtr, Xva, Xte = feats.transform(train), feats.transform(val), feats.transform(test)
    ytr, yva, yte = train.category.to_numpy(), val.category.to_numpy(), test.category.to_numpy()

    val_rows, test_rows, fitted = [], [], {}
    kw_val = np.array([keyword_category(t) or "none" for t in val.text])
    kw_test = np.array([keyword_category(t) or "none" for t in test.text])
    for name, pred, y, new, rows in (("Keyword rules (baseline)", kw_val, yva, new_val, val_rows),
                                     ("Keyword rules (baseline)", kw_test, yte, new_test, test_rows)):
        r = evaluate(y, pred, None, None, new)
        answered = pred != "none"
        r.update(model=name, auto_filed=answered.mean(), precision_when_filed=(pred[answered] == y[answered]).mean())
        rows.append(r)

    for m in candidates():
        # Each fitted model is saved as soon as it finishes, so a stopped run never has to retrain finished models.
        key = hashlib.sha256(f"{PER_CLASS}|{m.name}|{m.reduce}|{m.estimator!r}".encode()).hexdigest()[:10]
        cached = RUNS / f"{m.name.split()[0].lower()}-{key}.joblib"
        if cached.exists():
            m, fit_s = joblib.load(cached)
            print(f"{m.name}: loaded from cache", flush=True)
        else:
            t = time.time()
            m.fit(Xtr, ytr)
            fit_s = time.time() - t
            joblib.dump((m, fit_s), cached)
        sv = m.predict_scores(Xva)
        cls = m.estimator.classes_
        pv, cv = cls[sv.argmax(1)], sv.max(1)
        threshold, _ = coverage_at_precision(cv, pv == yva, TARGET_PRECISION)
        rv = evaluate(yva, pv, cv, threshold, new_val)
        rv.update(model=m.name, threshold=threshold, fit_seconds=round(fit_s))
        val_rows.append(rv)
        fitted[m.name] = (m, threshold)
        print(f"{m.name}: val macro-F1 {rv['macro_f1']:.3f}, auto-filed {rv['auto_filed']:.1%} ({fit_s:.0f}s)", flush=True)

    chosen = max((r for r in val_rows if r["model"] in fitted), key=lambda r: (round(r["macro_f1"], 3), r["auto_filed"]))["model"]
    for name, (m, threshold) in fitted.items():  # test scored once, every model, chosen settings untouched
        st = m.predict_scores(Xte)
        cls = m.estimator.classes_
        pt, ct = cls[st.argmax(1)], st.max(1)
        r = evaluate(yte, pt, ct, threshold, new_test)
        r.update(model=name)
        test_rows.append(r)
        if name == chosen:
            chosen_pred = pt
    per_class = pd.DataFrame({"category": [EXPENSE_CATEGORIES[c] for c in classes],
                              "test F1": f1_score(yte, chosen_pred, labels=classes, average=None),
                              "test rows": [int((yte == c).sum()) for c in classes]})
    cm = pd.DataFrame(confusion_matrix(yte, chosen_pred, labels=classes), index=classes, columns=classes)
    confused = (cm.where(~np.eye(len(classes), dtype=bool)).stack().sort_values(ascending=False).head(8)
                .rename_axis(["true", "predicted"]).reset_index(name="rows").astype({"rows": int}))

    # production model: same settings, refit on every year so it knows the newest vendors and phrasing
    prod_rows = cap_per_class(data, PER_CLASS)
    prod_feats = Features().fit(prod_rows)
    prod_model, prod_threshold = next(c for c in candidates() if c.name == chosen), fitted[chosen][1]
    prod_model.fit(prod_feats.transform(prod_rows), prod_rows.category.to_numpy())
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"features": prod_feats, "model": prod_model, "threshold": prod_threshold, "name": chosen,
                 "categories": EXPENSE_CATEGORIES, "trained_rows": len(prod_rows)}, MODELS / "categoriser.joblib")

    out = {"chosen": chosen, "validation": val_rows, "test": test_rows, "per_class": per_class.to_dict("records"),
           "most_confused": confused.to_dict("records"),
           "data": {"train_used": len(train), "train_available": len(train_all), "val": len(val), "test": len(test),
                    "val_new_wording": float(new_val.mean()), "test_new_wording": float(new_test.mean())},
           "seconds": round(time.time() - t0)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "categoriser.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    cols = ["model", "accuracy", "macro_f1", "auto_filed", "precision_when_filed", "accuracy_new_wording", "macro_f1_new_wording"]
    md = [
        "# Expense categoriser: model comparison", "",
        "Generated by `python -m categoriser.train` (in `ml/`). Do not edit by hand.", "",
        "## Data", "",
        "Real UK MP office expense claims published by IPSA, mapped onto 14 startup expense categories "
        "(`ml/categoriser/taxonomy.py`). Split by financial year: train 2010-11 to 2021-22, validation 2022-23 to 2023-24, "
        f"test 2024-25 to 2026-27. Training rows used: {len(train):,} (capped at {PER_CLASS:,} per category, of "
        f"{len(train_all):,}). Validation {len(val):,} rows, test {len(test):,} rows; {new_test.mean():.0%} of test rows have "
        "wording never seen in training.", "",
        "**Auto-filed** is the share of transactions the model files without asking, at a confidence threshold chosen on "
        f"validation so that precision stays at or above {TARGET_PRECISION:.0%}. For keyword rules it is the share with any "
        "keyword match.", "",
        "## Validation (choose here)", "", table(val_rows, cols + ["fit_seconds"]), "",
        f"## Chosen: {chosen}", "", "Highest validation macro-F1 (ties broken by auto-filed share).", "",
        "## Test (scored once)", "", table(test_rows, cols), "",
        "## Chosen model, per category (test)", "", table(per_class.to_dict("records"), ["category", "test F1", "test rows"]), "",
        "## Most frequent confusions (test)", "", table(confused.to_dict("records"), ["true", "predicted", "rows"]), "",
    ]
    (RESULTS / "categoriser.md").write_text("\n".join(md), encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s; chosen {chosen}")
    return out


if __name__ == "__main__":
    run()
