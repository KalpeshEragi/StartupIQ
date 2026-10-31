"""From a role-mapped table to regular series, with checks the user should see.

Thresholds: devdocs/decisions/0004-data-tiers.md and 0011-intake-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from startupiq.contracts import SalesSeries, Tier
from startupiq.intake.parsing import robust_z, to_dates, to_number

MIN_ROWS_TRANSACTIONS = 20
MIN_PERIODS = 4
WEEKS_SUMMARY_ONLY = 8
WEEKS_BASIC = 26
WEEKS_STANDARD = 52
WEEKS_PER_MONTH = 52 / 12
OUTLIER_Z = 4.0
GAP_PERIODS = 3  # this many zero periods in a row inside the history is a question for the user
SPARSE_SHARE = 0.5
TOP_PRODUCTS = 5
MIN_PRODUCT_WEEKS = 8


def tier_for(weeks: float) -> Tier:
    if weeks < WEEKS_SUMMARY_ONLY:
        return Tier.SUMMARY_ONLY
    if weeks < WEEKS_BASIC:
        return Tier.BASIC
    if weeks < WEEKS_STANDARD:
        return Tier.STANDARD
    return Tier.FULL


@dataclass
class Built:
    tier: Tier
    series: list[SalesSeries] = field(default_factory=list)
    granularity: str | None = None
    log: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    confirm: list[str] = field(default_factory=list)


def granularity_of(dates: pd.Series, products: pd.Series | None = None) -> str:
    """Spacing of distinct dates decides M/W/D; many rows per day (per product, if known) means a transaction log."""
    d = dates.dropna()
    keys = d if products is None else pd.Series(list(zip(d, products.loc[d.index])), index=d.index)
    distinct = d.drop_duplicates().sort_values()
    gap = distinct.diff().dt.days.median() if len(distinct) >= 3 else 1
    if gap >= 25:
        return "M"
    if gap >= 6:
        return "W"
    if len(d) > 1.5 * keys.nunique():
        return "transactions"
    return "D"


def _complete_weeks(s: pd.Series, first_day: pd.Timestamp, last_day: pd.Timestamp, log: list[str]) -> pd.Series:
    """Drop the first and last week when the data does not cover all 7 days of them."""
    dropped = []
    if len(s) and first_day.dayofweek != 0:
        dropped.append(s.index[0]); s = s.iloc[1:]
    if len(s) and last_day.dayofweek != 6:
        dropped.append(s.index[-1]); s = s.iloc[:-1]
    if dropped:
        log.append(f"Dropped {len(dropped)} partial week(s) at the start/end so totals are not understated.")
    return s


def build(df: pd.DataFrame, cmap: dict[str, str], today: date | None = None) -> Built:
    out = Built(tier=Tier.REFUSED)
    log, msgs = out.log, out.messages
    today = today or date.today()

    if "date" not in cmap:
        msgs.append("Could not find a date column. Add a column with the date of each sale, or use the template.")
        return out
    if not {"revenue", "units", "price"} & cmap.keys():
        msgs.append("Could not find a sales amount or quantity column. Add one, or use the template.")
        return out

    dates, how = to_dates(df[cmap["date"]])
    if how.endswith("_ambiguous"):
        out.confirm.append("date_order")
        msgs.append("Dates like 03/04 could be 3 April or 4 March. We read them day-first (3 April). Please confirm.")
    if "revenue" in cmap:
        amount = to_number(df[cmap["revenue"]])
    elif "units" in cmap and "price" in cmap:
        amount = to_number(df[cmap["units"]]) * to_number(df[cmap["price"]])
        log.append("No total column: revenue computed as quantity x price.")
    elif "units" in cmap:
        amount = to_number(df[cmap["units"]])
        msgs.append("Only quantities found, so forecasts are in units. Add a price or amount column to see revenue.")
    else:  # only a price column: marketplace exports list one item per row
        amount = to_number(df[cmap["price"]])
        out.confirm.append("revenue")
        msgs.append(f"No amount or quantity column, so each row is counted as one item sold at '{cmap['price']}'. Correct?")
    data = pd.DataFrame({"date": dates, "amount": amount, "_dup": df.duplicated().values})
    if "product" in cmap:
        data["product"] = df[cmap["product"]].astype("string").str.strip()

    n0 = len(data)
    data = data.dropna(subset=["date", "amount"])
    if n0 - len(data):
        log.append(f"Dropped {n0 - len(data)} of {n0} rows with an unreadable date or amount.")
        if (n0 - len(data)) / max(n0, 1) > 0.2:
            msgs.append(f"{n0 - len(data)} of {n0} rows could not be read. Check that the date and amount columns are right.")
    has_time = bool((data.date != data.date.dt.normalize()).any())
    granularity = granularity_of(data.date.dt.normalize(), data.get("product")) if len(data) else None
    # Duplicates are judged on the whole uploaded row: rows that differ in any column (an item number,
    # an invoice) are separate sales even if date, product and amount match.
    dupes = int(data._dup.sum())
    if dupes and granularity == "transactions" and not has_time:
        # Two identical sales on one day are normal in a shop register, so they are kept.
        log.append(f"{dupes} identical rows kept: in a sales log they are usually repeat purchases, not double entries.")
    elif dupes:
        data = data[~data._dup]
        why = "same timestamp and every other column" if has_time else "one row per period expected"
        log.append(f"Removed {dupes} exact duplicate rows ({why}, so these are double entries).")
    data = data.drop(columns="_dup")
    future = data.date.dt.date > today
    ancient = data.date.dt.year < 1990
    if future.any() or ancient.any():
        data = data[~future & ~ancient]
        log.append(f"Removed {int(future.sum())} future-dated and {int(ancient.sum())} pre-1990 rows.")
    negatives = int((data.amount < 0).sum())
    if negatives:
        log.append(f"{negatives} negative rows (returns or refunds) netted against sales in the same period.")

    out.granularity = granularity
    if out.granularity in ("transactions", "D") and len(data) < MIN_ROWS_TRANSACTIONS:
        msgs.append(f"Only {len(data)} usable rows. Need at least {MIN_ROWS_TRANSACTIONS} to say anything useful.")
        return out
    if len(data) == 0 or data.amount.sum() <= 0:
        msgs.append("All sales are zero or negative, so there is nothing to analyse.")
        return out

    freq = "M" if out.granularity == "M" else "W"
    if freq == "M":
        total = data.set_index("date").amount.resample("MS").sum()
        step = "month"
    else:
        total = data.set_index("date").amount.resample("W-SUN").sum()
        if out.granularity in ("transactions", "D"):
            total = _complete_weeks(total, data.date.min(), data.date.max(), log)
        step = "week"
    nonzero = int((total > 0).sum())
    if nonzero < MIN_PERIODS:
        msgs.append(f"Sales in only {nonzero} {step}(s). Need at least {MIN_PERIODS} to say anything useful.")
        return out

    weeks = len(total) * (WEEKS_PER_MONTH if freq == "M" else 1)
    tier = tier_for(weeks)
    log.append(f"{len(total)} {step}s from {total.index.min().date()} to {total.index.max().date()} ({out.granularity} upload).")

    zero = (total.values == 0).astype(int)
    runs, start = [], None
    for i, z in enumerate(zero):
        if z and start is None:
            start = i
        if (not z or i == len(zero) - 1) and start is not None:
            end = i if z else i - 1
            if end - start + 1 >= GAP_PERIODS and start > 0 and end < len(zero) - 1:
                runs.append((total.index[start].date(), total.index[end].date()))
            start = None
    if runs:
        out.confirm.append("gaps")
        a, b = runs[0]
        more = f" and {len(runs) - 1} more stretch(es)" if len(runs) > 1 else ""
        msgs.append(f"No sales from {a} to {b}{more}. Was the business closed, or were sales not recorded? "
                    "We treat it as closed (zero sales) unless you say otherwise.")

    if zero.mean() > SPARSE_SHARE and tier in (Tier.STANDARD, Tier.FULL):
        tier = Tier.BASIC
        msgs.append(f"Sales in only {nonzero} of {len(total)} {step}s: too sparse for a detailed forecast, so results are basic.")

    positive = total[total > 0]
    if len(positive) >= 8:
        z = robust_z(np.log(positive.values))
        spikes = positive[z > OUTLIER_Z]
        if len(spikes):
            out.confirm.append("outliers")
            top = spikes.idxmax()
            ratio = spikes.max() / positive.median()
            msgs.append(f"The {step} of {top.date()} is {ratio:.1f}x a typical {step}. One-off bulk order? "
                        "It is included; exclude it if it will not repeat.")
            log.append(f"{len(spikes)} unusual {step}(s) flagged (robust z > {OUTLIER_Z}); kept.")

    out.series.append(SalesSeries(frequency=freq, dates=[d.date() for d in total.index], values=total.astype(float).tolist()))
    if "product" in data and freq == "W":
        by = data.groupby("product").amount.sum().sort_values(ascending=False)
        for p in by.index[:TOP_PRODUCTS]:
            ps = data[data["product"] == p].set_index("date").amount.resample("W-SUN").sum().reindex(total.index, fill_value=0)
            if (ps > 0).sum() >= MIN_PRODUCT_WEEKS:
                out.series.append(SalesSeries(frequency="W", dates=[d.date() for d in ps.index], values=ps.astype(float).tolist(), product=str(p)))
        if len(out.series) > 1:
            log.append(f"Built weekly series for {len(out.series) - 1} top product(s).")
    out.tier = tier
    return out
