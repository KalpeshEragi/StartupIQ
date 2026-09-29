"""Task 5 feature engineering + leakage registry.

Feature tiers (see FEATURE_REGISTRY):
  profile   : static company description, knowable at evaluation time
  early     : early-stage funding signals (seed/angel/grant/convertible/crowdfunding)
  full      : full funding history - contains FUTURE information relative to an early evaluation
  calendar  : calendar-year features; carry an observation-window effect, not startup quality

Feature sets used downstream:
  early_safe   = profile + early           (prediction-safe, primary)
  full_history = profile + early + full    (leaky upper bound, for comparison only)
  ablations    = profile only; early_safe + calendar
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RESOLVED = {"closed": 0, "acquired": 1}  # target: 1 = acquired


def _has(s: pd.Series) -> pd.Series:
    """1/0 presence flag that stays missing when the amount is unknown."""
    return (s > 0).astype(float).where(s.notna())


def _years(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a - b).dt.days / 365.25


# name -> (tier, kind, source columns, calculation, reason)
FEATURE_REGISTRY = {
    "market": ("profile", "cat", "market", "as is (rare values grouped at encoding)", "industry"),
    "country_code": ("profile", "cat", "country_code", "as is (rare values grouped)", "geography"),
    "region": ("profile", "cat", "region", "as is (rare values grouped)", "startup hub / local ecosystem; city is too high-cardinality"),
    "n_categories": ("profile", "num", "category_list", "number of tags in category_list", "breadth of business description"),
    "age_at_first_funding": ("profile", "num", "founded_at, first_funding_at",
                             "(first_funding - founded)/365.25; < -1y -> missing (invalid), -1..0 -> 0; capped at 40",
                             "how quickly the company raised its first round"),
    "has_seed": ("early", "num", "seed", "seed > 0", "raised a seed round"),
    "has_angel": ("early", "num", "angel", "angel > 0", "angel investor backing"),
    "has_convertible_note": ("early", "num", "convertible_note", "> 0", "early instrument"),
    "has_grant": ("early", "num", "grant", "> 0", "non-dilutive early funding"),
    "has_crowdfunding": ("early", "num", "equity_crowdfunding, product_crowdfunding", "either > 0", "crowd validation"),
    "log_seed": ("early", "num", "seed", "log1p(seed)", "size of seed funding (skewed)"),
    "log_angel": ("early", "num", "angel", "log1p(angel)", "size of angel funding (skewed)"),
    "log_funding_total": ("full", "num", "funding_total_usd", "log1p(total)", "cumulative funding incl. all later rounds"),
    "funding_rounds": ("full", "num", "funding_rounds", "as is", "count includes rounds after the outcome-relevant period"),
    "has_venture": ("full", "num", "venture", "venture > 0", "venture rounds are usually later than seed"),
    "log_venture": ("full", "num", "venture", "log1p(venture)", "later-stage funding size"),
    "has_private_equity": ("full", "num", "private_equity", "> 0", "late-stage funding"),
    "has_debt_financing": ("full", "num", "debt_financing", "> 0", "later financing"),
    "has_post_ipo": ("full", "num", "post_ipo_equity, post_ipo_debt", "either > 0", "exists only after an IPO"),
    "n_named_rounds": ("full", "num", "round_A..round_H", "number of named rounds with money", "progression through A-H"),
    "max_named_round": ("full", "num", "round_A..round_H", "highest letter round reached (0-8)", "stage reached"),
    "avg_funding_per_round_log": ("full", "num", "funding_total_usd, funding_rounds", "log1p(total / rounds)", "round size"),
    "venture_share": ("full", "num", "venture, funding_total_usd", "venture / total", "funding mix"),
    "funding_span_years": ("full", "num", "first_funding_at, last_funding_at", "(last - first)/365.25", "uses the last funding date"),
    "rounds_per_year": ("full", "num", "funding_rounds, funding_span", "rounds / max(span, 1)", "funding velocity"),
    "years_since_last_funding": ("full", "num", "last_funding_at", "(2015-01-01 - last)/365.25",
                                 "closed/acquired companies stop raising, so this encodes the outcome"),
    "founded_year": ("calendar", "num", "founded_year", "as is", "cohort effect: acquired share falls from ~85% (1990s) to ~45% (2010s)"),
    "first_funding_year": ("calendar", "num", "first_funding_at", "year", "cohort / observation-window effect"),
}
SNAPSHOT = pd.Timestamp("2015-01-01")

TIERS = {t: [f for f, v in FEATURE_REGISTRY.items() if v[0] == t] for t in ("profile", "early", "full", "calendar")}
FEATURE_SETS = {
    "profile_only": TIERS["profile"],
    "early_safe": TIERS["profile"] + TIERS["early"],
    "early_safe_plus_calendar": TIERS["profile"] + TIERS["early"] + TIERS["calendar"],
    "full_history": TIERS["profile"] + TIERS["early"] + TIERS["full"],
}
META_COLS = ["permalink", "name", "status", "target", "founded_year_meta"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise, non-learned transformations only (no statistics fitted from the data)."""
    o = pd.DataFrame(index=df.index)
    o["permalink"], o["name"], o["status"] = df.permalink, df.name, df.status
    o["target"] = df.status.map(RESOLVED)

    o["market"] = df.market
    o["country_code"] = df.country_code
    o["region"] = df.region
    o["n_categories"] = df.category_list.str.strip("|").str.split("|").str.len()
    age = _years(df.first_funding_at, df.founded_at)
    o["age_at_first_funding"] = age.where(age >= -1).clip(lower=0, upper=40)

    o["has_seed"], o["has_angel"] = _has(df.seed), _has(df.angel)
    o["has_convertible_note"], o["has_grant"] = _has(df.convertible_note), _has(df.grant)
    o["has_crowdfunding"] = _has(df.equity_crowdfunding.fillna(0) + df.product_crowdfunding.fillna(0)).where(df.seed.notna())
    o["log_seed"], o["log_angel"] = np.log1p(df.seed), np.log1p(df.angel)

    o["log_funding_total"] = np.log1p(df.funding_total_usd)
    o["funding_rounds"] = df.funding_rounds.astype(float)
    o["has_venture"], o["log_venture"] = _has(df.venture), np.log1p(df.venture)
    o["has_private_equity"], o["has_debt_financing"] = _has(df.private_equity), _has(df.debt_financing)
    o["has_post_ipo"] = _has(df.post_ipo_equity.fillna(0) + df.post_ipo_debt.fillna(0)).where(df.seed.notna())
    rounds = df[[f"round_{c}" for c in "ABCDEFGH"]].fillna(0)
    o["n_named_rounds"] = (rounds > 0).sum(axis=1).astype(float)
    o["max_named_round"] = (rounds > 0).mul(np.arange(1, 9)).max(axis=1).astype(float)
    o["avg_funding_per_round_log"] = np.log1p(df.funding_total_usd / df.funding_rounds.astype(float))
    o["venture_share"] = (df.venture / df.funding_total_usd).where(df.funding_total_usd > 0)
    span = _years(df.last_funding_at, df.first_funding_at)
    o["funding_span_years"] = span
    o["rounds_per_year"] = df.funding_rounds.astype(float) / span.clip(lower=1)
    o["years_since_last_funding"] = _years(SNAPSHOT, df.last_funding_at)

    o["founded_year"] = df.founded_year.astype(float)
    o["first_funding_year"] = df.first_funding_at.dt.year.astype(float)
    o["founded_year_meta"] = o["founded_year"]  # kept for time-based checks / error analysis
    return o


def registry_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [(k, *v) for k, v in FEATURE_REGISTRY.items()],
        columns=["feature", "tier", "kind", "source_columns", "calculation", "reason"],
    ).assign(available_at_early_evaluation=lambda d: d.tier.isin(["profile", "early"]))
