"""Task 6 model definitions. Preprocessing lives inside the pipeline so that imputation,
scaling and encoding are re-fitted on the training part of every CV fold."""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.select import make_preprocessor

SEED = 42

# name -> (estimator, param grid, purpose)
MODELS = {
    "logistic_regression": (
        LogisticRegression(max_iter=2000, class_weight="balanced"),
        {"model__C": [0.1, 1.0]},
        "interpretable linear baseline",
    ),
    "random_forest": (
        RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=SEED),
        {"model__min_samples_leaf": [5, 20], "model__max_depth": [8, None]},
        "non-linear model; captures interactions and thresholds",
    ),
    "hist_gradient_boosting": (
        HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED, early_stopping=True),
        {"model__learning_rate": [0.05], "model__max_depth": [3, None], "model__min_samples_leaf": [20, 50]},
        "boosted trees; strongest common tabular baseline",
    ),
}


def make_pipeline(model_name: str, features) -> tuple[Pipeline, dict, str]:
    est, grid, purpose = MODELS[model_name]
    from sklearn.base import clone
    return Pipeline([("prep", make_preprocessor(features)), ("model", clone(est))]), grid, purpose
