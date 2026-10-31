"""Task 10 system validation: python -m startupiq.evaluate.validation

1. Unseen business type: Olist marketplace sellers were never used to train or choose any
   forecaster. Each seller's weekly sales go through intake like a founder's upload; the last
   12 weeks are hidden and forecast by the product forecaster (hybrid when the serving TCN is
   present, else Auto). MASE vs naive baselines, and P10-P90 coverage.
2. Fairness: forecast error by seller size (small / medium / large by revenue).
3. End to end: every seller's full upload through analyze(); latency and determinism (same
   upload twice -> identical report).
Writes docs/results/validation.md and .json.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from startupiq.contracts import BusinessContext, SalesSeries
from startupiq.evaluate import report
from startupiq.evaluate.metrics import mase
from startupiq.intake import read_table
from startupiq.models import forecast_baseline
from startupiq.models.forecast_methods import moving_average, naive
from startupiq.pipeline import analyze

H = 12
MIN_ACTIVE_WEEKS = 30
MAX_SELLERS = 300


def sellers() -> dict[str, pd.DataFrame]:
    items = pd.read_parquet("data/processed/olist_items.parquet", columns=["seller_id", "order_purchase_timestamp", "price"])
    items["week"] = items.order_purchase_timestamp.dt.to_period("W")
    active = items.groupby("seller_id").week.nunique()
    keep = active[active >= MIN_ACTIVE_WEEKS].sort_values(ascending=False).index[:MAX_SELLERS]
    return {s: g[["order_purchase_timestamp", "price"]].rename(columns={"order_purchase_timestamp": "date", "price": "sales"})
            for s, g in items[items.seller_id.isin(keep)].groupby("seller_id")}


def run() -> dict:
    t0 = time.time()
    rows, latencies, deterministic = [], [], 0
    ups = sellers()
    for i, (sid, up) in enumerate(ups.items()):
        got = read_table(up)
        if got.tier.value == "refused" or not got.series:
            continue
        s = got.series[0]
        y = np.asarray(s.values, float)
        if len(y) < 2 * H + 8:
            continue
        hist, actual = y[:-H], y[-H:]
        scale = float(np.mean(np.abs(np.diff(hist)))) or 1.0  # non-seasonal MASE scale: Olist series are under two years
        fc = forecast_baseline.forecast(SalesSeries(frequency="W", dates=s.dates[:-H], values=hist.tolist()))
        p50, p10, p90 = map(np.asarray, (fc.p50, fc.p10, fc.p90))
        rows.append({"seller": sid[:8], "revenue": float(y.sum()), "weeks": len(y), "method": fc.model,
                     "product forecaster": mase(actual, p50, scale), "naive": mase(actual, naive(hist), scale),
                     "moving average": mase(actual, moving_average(hist), scale),
                     "covered": float(np.mean((actual >= p10) & (actual <= p90)))})
        if i < 40:  # end-to-end latency and determinism on a sample
            ctx = BusinessContext(cash_on_hand=float(y[-8:].sum()), monthly_fixed_costs=float(y[-8:].mean() * 2), gross_margin=0.3,
                                  competitors=["rival"])
            t = time.perf_counter(); r1 = analyze(up, ctx); latencies.append(time.perf_counter() - t)
            r2 = analyze(up, ctx)
            deterministic += int(r1.model_dump_json() == r2.model_dump_json())
    d = pd.DataFrame(rows)
    d["size"] = pd.qcut(d.revenue, 3, labels=["small", "medium", "large"])
    overall = pd.DataFrame([{"sellers": len(d), "product forecaster": d["product forecaster"].mean(), "naive": d.naive.mean(),
                             "moving average": d["moving average"].mean(), "P10-P90 coverage": d.covered.mean(),
                             "share where product beats naive": (d["product forecaster"] < d.naive).mean()}]).round(3)
    by_size = d.groupby("size", observed=True).agg(sellers=("seller", "size"), product=("product forecaster", "mean"), naive=("naive", "mean"),
                                                   moving_average=("moving average", "mean"), coverage=("covered", "mean")).round(3).reset_index()
    methods = d.method.str.replace(r" \(.*\)", "", regex=True).value_counts().rename_axis("method").reset_index()
    out = {"overall": overall.to_dict("records"), "by_size": by_size.to_dict("records"),
           "latency_seconds": {"median": float(np.median(latencies)), "p95": float(np.quantile(latencies, 0.95))} if latencies else {},
           "deterministic": f"{deterministic} of {len(latencies)}", "seconds": round(time.time() - t0)}
    report.write("validation", "System validation on unseen businesses (Task 10)", [
        ("Setup", f"{len(d)} Olist sellers with {MIN_ACTIVE_WEEKS}+ active weeks, never used to train or choose a forecaster. Each upload goes "
                  "through intake; the last 12 weeks are hidden. MASE uses the mean absolute weekly change of the history as its scale "
                  "(the series are under two years, so the seasonal scale is unavailable)."),
        ("Forecast accuracy (MASE, lower is better)", report.table(overall, "{:.3f}")),
        ("Fairness: by seller size", report.table(by_size, "{:.3f}")),
        ("Which method the forecaster picked", report.table(methods)),
        ("End to end", f"Full analysis (intake, forecast, rules, planner): median {out['latency_seconds'].get('median', float('nan')):.2f} s, "
                       f"95th percentile {out['latency_seconds'].get('p95', float('nan')):.2f} s per upload. Same upload twice gave an identical "
                       f"report in {out['deterministic']} cases."),
    ], out, command="python -m startupiq.evaluate.validation")
    return out


if __name__ == "__main__":
    print(run())
