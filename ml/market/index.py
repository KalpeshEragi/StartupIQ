"""Turn raw page views into one daily and weekly market interest index per sector.

    python -m market.index                (from ml/, after market.pageviews)

How the index is built, so it can be explained in one breath:
0. Each article's daily views are divided by all human views on that Wikipedia that day (views per million), so
   the index measures a sector's share of attention. Wikipedia lost readers from 2023 to 2026 (AI answers, bot
   reclassification), which would otherwise look like every market shrinking at once.
1. Each article's daily share are divided by that article's median daily views over the whole window, so a huge
   article (Zomato) and a small one (Rebel Foods) count equally. 1.0 = "a normal day for this article". Days before
   an article existed (Google Gemini started in 2024) are left empty, not counted as zero.
2. Extreme one-day spikes (a viral news story) are capped at the article's 99.5th percentile.
3. The sector index for a day is the median of its articles' scaled values, times 100. The median means one
   article going viral cannot move the whole sector.
4. Relative index = the sector's index divided by the median of all sectors that day, times 100 (100 = a typical
   market). Topic articles lost readers faster than Wikipedia overall (people ask chatbots for factual lookups), so
   even the share of attention drifts down for nearly every sector. Dividing by the median of all sectors removes
   that shared drift and keeps what is specific to each market. This is what the models forecast.
5. Weekly = mean of the 7 daily values (weeks end on Sunday).

Outputs (not committed): ml/data/processed/market/articles.parquet (one row per article per day) and
sectors_daily.parquet / sectors_weekly.parquet (one row per sector per day / week).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from market import pageviews
from market.sectors import START, articles

OUT = Path(__file__).resolve().parents[1] / "data" / "processed" / "market"
MIN_BASE = 20     # articles whose normal day has fewer views than this are too noisy to use
CAP_Q = 0.995


def share(views: pd.Series, totals: pd.Series) -> pd.Series:
    """Views per million human views on the whole project that day."""
    return views / totals.reindex(views.index) * 1_000_000


def scale(views: pd.Series, totals: pd.Series | None = None) -> pd.Series | None:
    """Scale one article's daily share of attention to its median, with spikes capped. None if too small to use.
    Without totals (tests), raw views are scaled."""
    if not views.median() or views.median() < MIN_BASE:  # too few readers, whatever the share
        return None
    s = share(views, totals) if totals is not None else views
    ratio = s / s.median()
    return ratio.clip(upper=ratio.quantile(CAP_Q))


def sector_index(scaled: pd.DataFrame) -> pd.Series:
    """Median across a sector's articles (columns), times 100."""
    return scaled.median(axis=1) * 100


def relative(wide: pd.DataFrame) -> pd.DataFrame:
    """Each sector (column) divided by the median across sectors on the same day, times 100."""
    return wide.div(wide.median(axis=1), axis=0) * 100


def weekly(daily: pd.Series) -> pd.Series:
    """Mean per week ending Sunday; partial weeks at either end are dropped."""
    counts = daily.resample("W-SUN").count()
    return daily.resample("W-SUN").mean()[counts == 7]


def build(end: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    rows, skipped = [], []
    totals = {}
    for project in ("en.wikipedia", "hi.wikipedia"):
        t = pd.Series(pageviews.read_totals(project), dtype="float64")
        if t.empty:
            raise SystemExit(f"No totals for {project}. Run: python -m market.pageviews")
        t.index = pd.to_datetime(t.index)
        totals[project] = t
    for sector, project, title in articles():
        have = pageviews.read(project, title)
        if not have:
            continue
        s = pd.Series(have, dtype="float64")
        s.index = pd.to_datetime(s.index)
        days = pd.date_range(START, end or s.index.max(), freq="D")
        s = s.reindex(days)
        s[s.index >= s.first_valid_index()] = s[s.index >= s.first_valid_index()].fillna(0.0)  # API skips 0-view days
        scaled = scale(s, totals[project])
        if scaled is None:
            skipped.append(f"{sector}: {project} / {title}")
            continue
        rows.append(pd.DataFrame({"date": days, "sector": sector, "project": project, "title": title,
                                  "views": s.values, "scaled": scaled.values}))
    if not rows:
        raise SystemExit("No page views found. Run: python -m market.pageviews")
    arts = pd.concat(rows, ignore_index=True)

    absolute = {sector: sector_index(g.pivot_table(index="date", columns=["project", "title"], values="scaled"))
                for sector, g in arts.groupby("sector")}
    counts = arts.groupby("sector")[["project", "title"]].apply(lambda g: len(g.drop_duplicates()))
    abs_wide = pd.DataFrame(absolute)
    rel_wide = relative(abs_wide)
    daily_parts, weekly_parts = [], []
    for sector in abs_wide.columns:
        daily_parts.append(pd.DataFrame({"date": abs_wide.index, "sector": sector, "share_index": abs_wide[sector].values,
                                         "index": rel_wide[sector].values, "articles": int(counts[sector])}))
        wa, wr = weekly(abs_wide[sector]), weekly(rel_wide[sector])
        weekly_parts.append(pd.DataFrame({"week_end": wr.index, "sector": sector, "share_index": wa.reindex(wr.index).values,
                                          "index": wr.values}))
    return arts, pd.concat(daily_parts, ignore_index=True), pd.concat(weekly_parts, ignore_index=True), skipped


def main() -> None:
    arts, daily, week, skipped = build()
    OUT.mkdir(parents=True, exist_ok=True)
    arts.to_parquet(OUT / "articles.parquet", index=False)
    daily.to_parquet(OUT / "sectors_daily.parquet", index=False)
    week.to_parquet(OUT / "sectors_weekly.parquet", index=False)
    print(f"{arts[['project', 'title']].drop_duplicates().shape[0]} articles, {daily['sector'].nunique()} sectors, "
          f"{daily['date'].min():%Y-%m-%d} to {daily['date'].max():%Y-%m-%d}, {week['week_end'].nunique()} weeks")
    if skipped:
        print(f"Too few views to use ({len(skipped)}):")
        print("\n".join(f"  {s}" for s in skipped))


if __name__ == "__main__":
    main()
