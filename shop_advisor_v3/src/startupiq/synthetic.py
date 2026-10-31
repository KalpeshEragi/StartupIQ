"""SYNTHETIC Indian small-retail businesses for the business-type classifier and the category-aware forecaster.

python -m startupiq.synthetic        writes data/synthetic/businesses.parquet and data/synthetic/daily.parquet

Every business is one of six categories. A category fixes the *typical* values of what a shop owner can
see in a sales register (price per item, items per bill, returns, which weekdays are busy, which months and
festivals lift sales); each business draws its own values around them, so categories overlap, and about
10% of businesses are atypical (their seasons are mostly flattened). The numbers are plausible
assumptions, not measurements: models trained here show the method works, not how accurate it is on
real shops.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "data" / "synthetic"
DIWALI = pd.to_datetime(["2021-11-04", "2022-10-24", "2023-11-12", "2024-11-01", "2025-10-21", "2026-11-08", "2027-10-29"])

# month lifts are additive (+0.2 = 20% above a normal month); weekday factors run Monday..Sunday
CATEGORIES: dict[str, dict] = {
    "Clothing & fashion": dict(level=25000, item_price=900, price_sd=0.45, items_per_bill=2.2, return_rate=0.08, noise=0.20,
                               weekday=[0.85, 0.85, 0.9, 0.9, 1.0, 1.25, 1.4], diwali=1.4, month_start=0.0,
                               months={1: 0.25, 2: 0.1, 4: 0.1, 5: 0.1, 7: 0.1, 8: -0.15, 9: -0.1, 11: 0.2, 12: 0.2}),
    "Electronics & mobiles": dict(level=60000, item_price=4500, price_sd=0.7, items_per_bill=1.2, return_rate=0.035, noise=0.25,
                                  weekday=[0.9, 0.9, 0.9, 0.95, 1.0, 1.15, 1.25], diwali=1.1, month_start=0.05,
                                  months={3: -0.05, 4: 0.15, 5: 0.2, 6: 0.05, 9: 0.1, 2: -0.1}),
    "Grocery & kirana": dict(level=20000, item_price=70, price_sd=0.4, items_per_bill=8.0, return_rate=0.005, noise=0.10,
                             weekday=[1.0, 0.95, 0.95, 0.95, 1.0, 1.1, 1.05], diwali=0.3, month_start=0.15,
                             months={3: 0.03, 9: 0.03, 10: 0.05}),
    "Restaurant & cafe": dict(level=30000, item_price=220, price_sd=0.4, items_per_bill=3.0, return_rate=0.002, noise=0.18,
                              weekday=[0.75, 0.75, 0.8, 0.85, 1.1, 1.45, 1.5], diwali=-0.1, month_start=0.03,
                              months={12: 0.25, 1: 0.1, 7: -0.12, 8: -0.12, 5: 0.05, 2: 0.08}),
    "Pharmacy": dict(level=18000, item_price=150, price_sd=0.4, items_per_bill=2.5, return_rate=0.012, noise=0.10,
                     weekday=[1.05, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8], diwali=0.0, month_start=0.08,
                     months={7: 0.2, 8: 0.25, 9: 0.2, 12: 0.1, 1: 0.12, 4: -0.05, 5: -0.08}),
    "Stationery & books": dict(level=10000, item_price=120, price_sd=0.5, items_per_bill=4.0, return_rate=0.02, noise=0.22,
                               weekday=[1.05, 1.05, 1.05, 1.05, 1.05, 1.0, 0.6], diwali=0.15, month_start=0.02,
                               months={6: 1.0, 7: 0.5, 3: 0.2, 4: 0.15, 5: -0.3, 10: -0.1, 12: -0.2}),
}
NAMES = list(CATEGORIES)


def month_curve(months: dict[int, float], doy: np.ndarray) -> np.ndarray:
    """Smooth day-of-year curve through each month's lift, placed at mid-month (periodic interpolation)."""
    mids = np.array([15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349], float)
    vals = np.array([months.get(m, 0.0) for m in range(1, 13)])
    x = np.concatenate([mids - 365, mids, mids + 365])
    return np.interp(doy, x, np.tile(vals, 3))


def diwali_curve(days: pd.DatetimeIndex) -> np.ndarray:
    """Shopping before Diwali: a bump peaking about five days ahead of it, gone two days after."""
    out = np.zeros(len(days))
    for d in DIWALI:
        delta = (days - d).days.to_numpy()
        out += np.where((delta >= -21) & (delta <= 2), np.exp(-((delta + 5) ** 2) / (2 * 6.0 ** 2)), 0.0)
    return out


def business(rng: np.random.Generator, category: str, end: pd.Timestamp, weeks: int, atypical: bool = False) -> tuple[pd.DataFrame, dict]:
    c = CATEGORIES[category]
    days = pd.date_range(end - pd.Timedelta(days=7 * weeks - 1), end, freq="D")
    strength = rng.uniform(0.0, 0.3) if atypical else rng.uniform(0.6, 1.3)  # how strongly this shop follows its category's seasons
    months = {m: v * strength * rng.lognormal(0, 0.3) for m, v in c["months"].items()}
    weekday = np.array(c["weekday"]) ** (strength if not atypical else 0.3) * rng.lognormal(0, 0.05, 7)
    weekday /= weekday.mean()
    doy = days.dayofyear.to_numpy()
    t = (days - days[0]).days.to_numpy() / 365
    growth = rng.normal(0.08, 0.2)
    level = c["level"] * rng.lognormal(0, 0.6)
    shock = np.zeros(weeks)
    for i in range(1, weeks):  # week-to-week swings that persist (AR(1))
        shock[i] = 0.6 * shock[i - 1] + rng.normal(0, 0.06)
    season = (1 + month_curve(months, doy)) * (1 + c["diwali"] * strength * rng.lognormal(0, 0.3) * diwali_curve(days))
    season *= 1 + c["month_start"] * strength * (days.day <= 7)
    sales = level * (1 + growth) ** t * season * weekday[days.dayofweek] * np.exp(shock[np.arange(len(days)) // 7])
    sales *= rng.lognormal(0, c["noise"] * rng.uniform(0.7, 1.3), len(days))
    price = c["item_price"] * rng.lognormal(0, c["price_sd"])
    per_bill = max(1.0, c["items_per_bill"] * rng.lognormal(0, 0.25))
    returns_rate = c["return_rate"] * rng.lognormal(0, 0.5)
    orders = np.maximum(1, np.round(sales / (price * per_bill) * rng.lognormal(0, 0.08, len(days))))
    items = np.maximum(orders, np.round(orders * per_bill * rng.lognormal(0, 0.05, len(days))))
    returns = np.round(sales * returns_rate * rng.lognormal(0, 0.6, len(days)) * (rng.random(len(days)) < 0.7))
    df = pd.DataFrame({"date": days, "sales": np.round(sales), "orders": orders, "items": items, "returns": returns})
    meta = {"category": category, "start": days[0], "end": end, "weeks": weeks, "atypical": atypical, "growth": growth,
            "item_price": price, "items_per_bill": per_bill, "return_rate": returns_rate, "strength": strength}
    return df, meta


def upload_view(df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, str]:
    """What a real upload of this business might contain: some shops only keep date and sales, some only weekly totals."""
    r = rng.random()
    cols = ["date", "sales"] if r < 0.35 else (["date", "sales", "orders", "items"] if r < 0.5 else list(df.columns))
    view, columns = df[cols], ("sales only" if len(cols) == 2 else "with orders")
    if rng.random() < 0.25:
        return view.set_index("date").resample("W-SUN").sum().reset_index(), f"weekly, {columns}"
    return view, f"daily, {columns}"


def generate(per_category: int = 700, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    metas, frames = [], []
    first_end, last_end = pd.Timestamp("2025-03-02"), pd.Timestamp("2026-10-25")  # Sundays
    n_end = (last_end - first_end).days // 7
    i = 0
    for cat in NAMES:
        for _ in range(per_category):
            end = first_end + pd.Timedelta(weeks=int(rng.integers(0, n_end + 1)))
            weeks = int(rng.integers(12, 105))
            df, meta = business(rng, cat, end, weeks, atypical=rng.random() < 0.1)
            df.insert(0, "business", i)
            metas.append({"business": i, **meta})
            frames.append(df)
            i += 1
    meta = pd.DataFrame(metas)
    split = np.empty(len(meta), dtype=object)
    for cat in NAMES:  # stratified 60/20/20 split by business: no shop appears in two splits
        idx = rng.permutation(meta.index[meta.category == cat].to_numpy())
        a, b = int(0.6 * len(idx)), int(0.8 * len(idx))
        split[idx[:a]], split[idx[a:b]], split[idx[b:]] = "train", "val", "test"
    meta["split"] = split
    return meta, pd.concat(frames, ignore_index=True)


def main() -> None:
    meta, daily = generate()
    OUT.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(OUT / "businesses.parquet")
    daily.to_parquet(OUT / "daily.parquet")
    print(meta.groupby(["category", "split"]).size().unstack(), f"\n{len(daily):,} daily rows")


if __name__ == "__main__":
    main()
