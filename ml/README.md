# ml/ — training the expense categoriser

Offline work: download data, build the labelled dataset, compare models, export the winner to the backend.
The code the product runs (text cleaning, Indian bank-format rules, categories) lives in `backend/app/categorisation`
and is imported from there, so training and the product can never clean text differently.

## Run (from `ml/`, with the backend installed in the same environment)

```bash
pip install -e ../backend -r requirements.txt
python -m categoriser.download     # IPSA expenses, AgamiAI Indian statements, US benchmark -> data/raw/ (not committed)
python -m categoriser.ipsa         # labelled dataset + split sizes -> data/interim/
python -m categoriser.train        # compares 6 approaches, writes ../docs/results/categoriser.md
python -m categoriser.export       # installs the chosen model at ../backend/models/categoriser.joblib
pytest
```

Trained models are cached in `models/runs/`, so a stopped run resumes without retraining finished models.

## Data

| Dataset | Use | Licence |
| --- | --- | --- |
| IPSA UK MP office expenses (1.33M claims, via `kieraisverybored/uk-mp-expenses`) | training and evaluation | Open Government Licence (per IPSA) |
| `AgamiAI/Indian-Bank-Statements` | checking the Indian bank-format rules and the model on Indian narrations | Apache-2.0 |
| `DoDataThings/us-bank-transaction-categories-v2` | downloaded as an optional external benchmark | MIT |

## Results

- Model comparison: [`docs/results/categoriser.md`](../docs/results/categoriser.md)
- Check on Indian narrations: [`docs/results/categoriser_indian_check.md`](../docs/results/categoriser_indian_check.md)
