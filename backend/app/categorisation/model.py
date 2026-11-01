"""The learned expense categoriser at serving time.

The artefact is a plain dict of scikit-learn parts written by ml/categoriser (export_for_backend), so the product does
not import any training code:
    version, name, classes, threshold, scores ("proba" | "margin"),
    words, chars (fitted TfidfVectorizers), amount_bins (list), amount_encoder (OneHotEncoder),
    selector (SelectKBest or None), estimator
Features are built exactly as in training: cleaned text -> word and character TF-IDF, plus the amount in log bins.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.categorisation.text import clean
from app.core.config import get_settings


@dataclass
class Prediction:
    category: str
    confidence: float
    confident: bool                   # at or above the threshold chosen for >= 95% precision
    alternatives: list[tuple[str, float]]


class ExpenseModel:
    def __init__(self, artefact: dict):
        self.a = artefact
        self.version: str = artefact["version"]
        self.threshold: float = artefact["threshold"]

    def _features(self, texts: list[str], amounts: list[float]):
        from scipy import sparse
        a = self.a
        bins = np.digitize(np.abs(np.asarray(amounts, float)), a["amount_bins"]).reshape(-1, 1)
        X = sparse.hstack([a["words"].transform(texts), a["chars"].transform(texts), a["amount_encoder"].transform(bins)]).tocsr()
        return a["selector"].transform(X) if a.get("selector") is not None else X

    def predict(self, descriptions: list[str], amounts: list[float]) -> list[Prediction]:
        if not descriptions:
            return []
        X = self._features([clean(d) for d in descriptions], amounts)
        est = self.a["estimator"]
        if self.a["scores"] == "proba":
            scores = est.predict_proba(X)
        else:  # SVM margins turned into comparable confidences, as in training
            z = est.decision_function(X)
            z = z - z.max(axis=1, keepdims=True)
            scores = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        classes = est.classes_
        out = []
        for row in scores:
            order = np.argsort(-row)[:3]
            best = float(row[order[0]])
            out.append(Prediction(str(classes[order[0]]), best, best >= self.threshold,
                                  [(str(classes[i]), float(row[i])) for i in order]))
        return out


@lru_cache(maxsize=1)
def load_model() -> ExpenseModel | None:
    """The installed model, or None when no model file is present (lines then go to review)."""
    path = Path(get_settings().categoriser_path)
    if not path.exists():
        return None
    import joblib
    return ExpenseModel(joblib.load(path))
