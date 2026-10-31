"""Every train / validation / test boundary in one place (devdocs/evaluation_protocol.md).

Validation picks models and settings. Test is scored only by `--final` runs, once,
after the choice is written down. Changing a date here needs a decision record.
"""

from __future__ import annotations

import pandas as pd

# Rossmann, daily. Horizon 42 days, matching the competition's test length.
ROSS_HORIZON_DAYS = 42
ROSS_TEST_START = pd.Timestamp("2015-06-20")
ROSS_TEST_END = pd.Timestamp("2015-07-31")
ROSS_VAL_START = ROSS_TEST_START - pd.Timedelta(days=ROSS_HORIZON_DAYS)  # 2015-05-09

# Weekly benchmark (Rossmann store-weeks and UCI product-weeks). Horizon 12 weeks.
WEEK_HORIZON = 12

# Olist, monthly. A row at month t has features up to t and a label looking at t+1..t+3.
OLIST_TRAIN_LAST = pd.Period("2017-11", "M")  # labels reach 2018-02
OLIST_VAL = (pd.Period("2017-12", "M"), pd.Period("2018-02", "M"))  # labels reach 2018-05
OLIST_TEST = (pd.Period("2018-03", "M"), pd.Period("2018-05", "M"))  # labels reach 2018-08 (end of window)
OLIST_FIRST_FEATURE_MONTH = pd.Period("2017-03", "M")  # two months of lags available

SEEDS = (0, 1, 2)


def olist_split(month: pd.Series) -> pd.Series:
    """'train' / 'val' / 'test' / None for a Series of monthly Periods."""
    out = pd.Series(None, index=month.index, dtype="object")
    out[(month >= OLIST_FIRST_FEATURE_MONTH) & (month <= OLIST_TRAIN_LAST)] = "train"
    out[(month >= OLIST_VAL[0]) & (month <= OLIST_VAL[1])] = "val"
    out[(month >= OLIST_TEST[0]) & (month <= OLIST_TEST[1])] = "test"
    return out
