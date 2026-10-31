"""Run Task 5/6 experiments: python -m startupiq.models.run <name> [--final]

Without --final only the validation split is used. --final also scores the test
split; run it once per experiment, after the choice on validation is recorded.
"""

from __future__ import annotations

import argparse
import importlib

EXPERIMENTS = {
    "forecast_rossmann": "startupiq.models.forecast_rossmann",
    "forecast_weekly": "startupiq.models.forecast_weekly",
    "competitor": "startupiq.models.competitor",
    "share": "startupiq.models.share",
    "clustering": "startupiq.models.clustering",
    "regime": "startupiq.models.regime",
    "dl_weekly": "startupiq.dl.experiment_weekly",
    "dl_rossmann": "startupiq.dl.experiment_rossmann",
    "dl_hybrid": "startupiq.dl.hybrid",
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="+", choices=list(EXPERIMENTS) + ["all"])
    p.add_argument("--final", action="store_true", help="also score the test split (once)")
    args = p.parse_args(argv)
    names = list(EXPERIMENTS) if "all" in args.names else args.names
    for n in names:
        out = importlib.import_module(EXPERIMENTS[n]).run(final=args.final)
        print(f"{n}: done in {out.get('seconds', '?')}s -> docs/results/{n}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
