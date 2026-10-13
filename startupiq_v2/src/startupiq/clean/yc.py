"""Cleaning for the Y Combinator company directory (Stream A, text).

Rules follow docs/data_cards/yc_directory_intake.md. Nothing is invented: unknown
values stay missing. Rows are flagged rather than silently removed wherever a later
phase may want to decide differently.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from startupiq.clean.log import CleaningLog
from startupiq.clean.text import normalize_text, website_domain, word_count

# Columns that carry no usable information for this project (see intake report).
DROP_COLUMNS = [
    "small_logo_thumb_url", "tags_highlighted", "app_answers", "question_answers",
    "app_video_public", "demo_day_video_public", "api", "url", "slug",
]
VALID_STATUS = {"Active", "Inactive", "Acquired", "Public"}
VALID_STAGE = {"Early", "Growth"}
# Hosts shared by many unrelated companies: never used to decide two rows are the same company.
PLATFORM_DOMAINS = {
    "github.com", "linkedin.com", "twitter.com", "x.com", "facebook.com", "instagram.com",
    "medium.com", "notion.site", "notion.so", "vercel.app", "netlify.app", "herokuapp.com",
    "apple.com", "apps.apple.com", "play.google.com", "google.com", "youtube.com",
    "linktr.ee", "bit.ly", "t.me", "producthunt.com", "ycombinator.com",
}
_BATCH = re.compile(r"^(Winter|Spring|Summer|Fall)\s+(\d{4})$")
SEASON_ORDER = {"Winter": 0, "Spring": 1, "Summer": 2, "Fall": 3}
NEAR_DUP_THRESHOLD = 0.8
SHINGLE = 3


def load_raw(path: Path) -> pd.DataFrame:
    with Path(path).open(encoding="utf-8") as handle:
        return pd.DataFrame(json.load(handle))


def _shingles(text: str) -> set[str]:
    words = text.lower().split()
    if len(words) < SHINGLE:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)}


def near_duplicate_groups(texts: list[str], threshold: float = NEAR_DUP_THRESHOLD) -> list[int]:
    """Group indexes whose word-3-gram Jaccard similarity is at least `threshold`.

    Returns a group id per input (singletons get their own id). Uses an inverted
    index so only texts sharing at least one shingle are compared.
    """
    sets = [_shingles(t) for t in texts]
    parent = list(range(len(texts)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    index: dict[str, list[int]] = {}
    for i, s in enumerate(sets):
        for sh in s:
            index.setdefault(sh, []).append(i)
    seen: set[tuple[int, int]] = set()
    for ids in index.values():
        if len(ids) < 2 or len(ids) > 50:  # very common shingles carry no signal
            continue
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                i, j = ids[a], ids[b]
                if (i, j) in seen:
                    continue
                seen.add((i, j))
                if not sets[i] or not sets[j]:
                    continue
                if len(sets[i] & sets[j]) / len(sets[i] | sets[j]) >= threshold:
                    parent[find(i)] = find(j)
    return [find(i) for i in range(len(texts))]


def clean_yc(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningLog]:
    log = CleaningLog("yc_companies")
    df = raw.copy()
    n_raw = len(df)
    log.info("raw_records", n_raw)

    # 1. Drop columns with no usable content.
    present = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=present)
    log.info("dropped_columns", present, "uninformative or almost entirely empty (intake report)")

    # 2. Duplicate ids.
    before = len(df)
    df = df.drop_duplicates(subset="id", keep="first")
    log.rows("duplicate_id", before, len(df))

    # 3. Text normalisation.
    df["one_liner_clean"] = df["one_liner"].map(normalize_text)
    df["long_description_clean"] = df["long_description"].map(normalize_text)
    df["text"] = (df["one_liner_clean"] + ". " + df["long_description_clean"]).str.strip(" .")
    df.loc[(df["one_liner_clean"] != "") & (df["long_description_clean"] == ""), "text"] = df["one_liner_clean"]
    df.loc[(df["one_liner_clean"] == "") & (df["long_description_clean"] != ""), "text"] = df["long_description_clean"]
    df["usable_text"] = df["text"].str.len() > 0
    df["one_liner_words"] = df["one_liner_clean"].map(word_count)
    df["long_description_words"] = df["long_description_clean"].map(word_count)
    df["text_words"] = df["text"].map(word_count)
    log.flag("no_usable_text", (~df["usable_text"]).sum(), len(df),
             "both one_liner and long_description empty; flagged usable_text=False, excluded from text models later")
    only_one = ((df["long_description_clean"] == "") & (df["one_liner_clean"] != "")).sum()
    log.flag("long_description_missing_fallback_to_one_liner", only_one, len(df))

    # 4. Missing values and sentinels.
    df["team_size"] = pd.to_numeric(df["team_size"], errors="coerce")
    zero_team = (df["team_size"] <= 0).sum()
    df.loc[df["team_size"] <= 0, "team_size"] = np.nan
    log.flag("team_size_nonpositive_to_missing", zero_team, len(df),
             "0 and -1 are not real team sizes (one record holds -1)")
    df["team_size_missing"] = df["team_size"].isna()
    log.flag("team_size_missing_total", df["team_size_missing"].sum(), len(df))
    unspecified = (df["industry"] == "Unspecified").sum()
    df["industry"] = df["industry"].replace({"Unspecified": np.nan, "": np.nan})
    df["industry_known"] = df["industry"].notna()
    log.flag("industry_unspecified_to_missing", unspecified, len(df), "treated as unlabelled, not as a class")

    # 5. Tags: trim, remove empties, remove case-insensitive duplicates.
    def clean_tags(value) -> list[str]:
        out, seen = [], set()
        for tag in (value if isinstance(value, list) else []):
            t = normalize_text(tag)
            if t and t.lower() not in seen:
                seen.add(t.lower())
                out.append(t)
        return out

    df["tags"] = df["tags"].map(clean_tags)
    df["n_tags"] = df["tags"].map(len)
    log.flag("no_tags", (df["n_tags"] == 0).sum(), len(df))
    df["regions"] = df["regions"].map(lambda v: [normalize_text(x) for x in v if normalize_text(x)] if isinstance(v, list) else [])

    # 6. Inconsistencies.
    bad_status = ~df["status"].isin(VALID_STATUS)
    bad_stage = ~df["stage"].isin(VALID_STAGE)
    log.flag("status_not_in_expected_set", bad_status.sum(), len(df), f"expected {sorted(VALID_STATUS)}")
    log.flag("stage_not_in_expected_set", bad_stage.sum(), len(df), f"expected {sorted(VALID_STAGE)}")
    df.loc[bad_status, "status"] = np.nan
    df.loc[bad_stage, "stage"] = np.nan

    def level2(sub, ind):
        if not isinstance(sub, str) or " -> " not in sub:
            return np.nan
        return sub.split(" -> ", 1)[1].strip()

    df["subindustry_level2"] = [level2(s, i) for s, i in zip(df["subindustry"], df["industry"])]
    parent_of_sub = df["subindustry"].map(lambda s: s.split(" -> ", 1)[0].strip() if isinstance(s, str) else np.nan)
    mismatch = df["industry"].notna() & parent_of_sub.notna() & (parent_of_sub != df["industry"])
    log.flag("subindustry_parent_differs_from_industry", mismatch.sum(), len(df), "kept as-is, flagged for Phase 3")
    df["subindustry_inconsistent"] = mismatch

    parsed = df["batch"].map(lambda b: _BATCH.match(b) if isinstance(b, str) else None)
    df["batch_season"] = parsed.map(lambda m: m.group(1) if m else np.nan)
    df["batch_year"] = parsed.map(lambda m: int(m.group(2)) if m else np.nan)
    df["batch_index"] = df["batch_year"] * 4 + df["batch_season"].map(SEASON_ORDER)
    log.flag("batch_unparseable", df["batch_year"].isna().sum(), len(df), "batch name not 'Season YYYY'")

    launched = pd.to_datetime(df["launched_at"], unit="s", errors="coerce")
    df["launched_at"] = launched
    df["launched_year"] = launched.dt.year
    out_of_range = df["launched_year"].notna() & ~df["launched_year"].between(2005, 2026)
    log.flag("launched_year_out_of_range", out_of_range.sum(), len(df))
    df.loc[out_of_range, ["launched_at", "launched_year"]] = [pd.NaT, np.nan]
    far = df["launched_year"].notna() & df["batch_year"].notna() & ((df["launched_year"] - df["batch_year"]).abs() > 2)
    log.flag("launched_year_differs_from_batch_year_by_more_than_2", far.sum(), len(df), "reported only")

    # 7. Duplicates by website domain (same company relaunched), unless the host is a shared platform.
    df["domain"] = df["website"].map(website_domain)
    df["domain_is_platform"] = df["domain"].isin(PLATFORM_DOMAINS)
    dedupe_pool = df["domain"].notna() & ~df["domain_is_platform"]
    sort_key = df["batch_index"].fillna(-1)
    order = df.assign(_k=sort_key).sort_values("_k", ascending=False).index
    dup_mask = pd.Series(False, index=df.index)
    keep_seen: set[str] = set()
    for idx in order:
        if not dedupe_pool[idx]:
            continue
        d = df.at[idx, "domain"]
        if d in keep_seen:
            dup_mask[idx] = True
        else:
            keep_seen.add(d)
    before = len(df)
    df = df[~dup_mask].copy()
    log.rows("duplicate_website_domain_keep_most_recent_batch", before, len(df),
             "same non-platform domain means same company; the most recent batch row is kept")
    df["name_key"] = df["name"].astype(str).str.lower().str.strip()
    log.flag("repeated_company_names_kept", int(df["name_key"].duplicated(keep=False).sum()), len(df),
             "different companies often share a name; never deduplicated by name")

    ol = df["one_liner_clean"].fillna("").str.lower()
    ol_dup = ol.ne("") & ol.duplicated(keep=False)
    log.flag("identical_one_liner_different_company_kept", int(ol_dup.sum()), len(df),
             "short generic one-liners repeat across distinct companies; the full text differs, so rows are kept")

    # 8. Near-duplicate text: flagged and grouped, not removed (the group id is used for splits in Phase 3).
    texts = df["text"].tolist()
    groups = near_duplicate_groups([t if u else "" for t, u in zip(texts, df["usable_text"])])
    df["dup_group"] = groups
    sizes = df["dup_group"].map(df["dup_group"].value_counts())
    df["in_near_dup_group"] = (sizes > 1) & df["usable_text"]
    log.flag("near_duplicate_text_rows_grouped", df["in_near_dup_group"].sum(), len(df),
             f"word-{SHINGLE}-gram Jaccard >= {NEAR_DUP_THRESHOLD}; kept, share one dup_group id")

    # 9. Outliers: flagged, never deleted.
    df["team_size_log"] = np.log1p(df["team_size"])
    q1, q3 = df["team_size_log"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
    df["team_size_outlier"] = df["team_size_log"].notna() & ~df["team_size_log"].between(lo, hi)
    log.flag("team_size_extreme_outlier_log_iqr3", df["team_size_outlier"].sum(), len(df),
             f"log1p scale fences [{lo:.2f}, {hi:.2f}]")
    p99 = df.loc[df["usable_text"], "long_description_words"].quantile(0.99)
    df["long_text_outlier"] = df["long_description_words"] > p99
    log.flag("long_description_above_p99_words", df["long_text_outlier"].sum(), len(df), f"p99 = {p99:.0f} words")

    # 10. Scaling note: z-scoring is fitted inside training folds in Phase 3 to avoid leakage.
    log.info("scaling", "log1p(team_size) provided; z-scoring deferred to Phase 3 pipelines (fit on training data only)")

    keep = ["id", "name", "website", "domain", "batch", "batch_season", "batch_year", "batch_index", "launched_at",
            "launched_year", "status", "stage", "industry", "industry_known", "subindustry", "subindustry_level2",
            "subindustry_inconsistent", "tags", "n_tags", "regions", "all_locations", "team_size", "team_size_log",
            "team_size_missing", "team_size_outlier", "isHiring", "nonprofit", "top_company", "one_liner_clean",
            "long_description_clean", "text", "usable_text", "one_liner_words", "long_description_words", "text_words",
            "long_text_outlier", "dup_group", "in_near_dup_group", "domain_is_platform", "former_names"]
    df = df[[c for c in keep if c in df.columns]].reset_index(drop=True)
    log.info("final_records", len(df))
    return df, log


def to_csv(df: pd.DataFrame, path: Path) -> None:
    """Write with list columns joined by '|' (documented in the data card)."""
    out = df.copy()
    for col in ("tags", "regions", "former_names", "all_locations"):
        if col in out.columns:
            out[col] = out[col].map(lambda v: "|".join(map(str, v)) if isinstance(v, (list, tuple)) else v)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
