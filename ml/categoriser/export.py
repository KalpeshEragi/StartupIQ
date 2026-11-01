"""Turn the trained categoriser into the plain artefact the backend serves (backend/app/categorisation/model.py).

    python -m categoriser.export        (from ml/, after categoriser.train)

The training file holds training-code classes; the product must not import training code, so the parts are copied
into a dict of scikit-learn objects with a version stamp.
"""
from __future__ import annotations

import hashlib
from datetime import date

import joblib

from categoriser.data import ROOT
from categoriser.train import AMOUNT_BINS, MODELS

BACKEND_MODEL = ROOT.parent / "backend" / "models" / "categoriser.joblib"


def main() -> None:
    import __main__
    from categoriser.train import Features, Model
    # train.py runs as a script, so its pickles name these classes under __main__; point that name at them.
    __main__.Features, __main__.Model = Features, Model
    trained = joblib.load(MODELS / "categoriser.joblib")
    feats, model = trained["features"], trained["model"]
    stamp = hashlib.sha256((MODELS / "categoriser.joblib").read_bytes()).hexdigest()[:8]
    artefact = {
        "version": f"{date.today():%Y%m%d}-{stamp}", "name": trained["name"], "threshold": float(trained["threshold"]),
        "classes": [str(c) for c in model.estimator.classes_], "scores": model.scores,
        "words": feats.words, "chars": feats.chars, "amount_bins": list(AMOUNT_BINS), "amount_encoder": feats.amount,
        "selector": getattr(model, "selector", None) if model.reduce else None, "estimator": model.estimator,
        "trained_rows": trained["trained_rows"],
    }
    BACKEND_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefact, BACKEND_MODEL, compress=3)
    print(f"wrote {BACKEND_MODEL} ({artefact['name']}, version {artefact['version']}, threshold {artefact['threshold']:.3f})")


if __name__ == "__main__":
    main()
