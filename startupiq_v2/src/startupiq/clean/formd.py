"""Task 4 cleaning for Stream B: SEC Form D quarterly data sets.

One row per filing (accession number). Related-person names are never loaded, only counted.
Every rule is logged with rows before and after.
"""
from __future__ import annotations

import glob
import io
import os
import zipfile

import numpy as np
import pandas as pd

from startupiq.clean.log import CleaningLog

SUB_COLS = ["ACCESSIONNUMBER", "FILING_DATE", "SUBMISSIONTYPE", "SIC_CODE", "TESTORLIVE"]
OFF_COLS = ["ACCESSIONNUMBER", "INDUSTRYGROUPTYPE", "REVENUERANGE", "ISPOOLEDINVESTMENTFUNDTYPE",
            "ISEQUITYTYPE", "ISDEBTTYPE", "ISOPTIONTOACQUIRETYPE", "ISAMENDMENT", "PREVIOUSACCESSIONNUMBER",
            "FEDERALEXEMPTIONS_ITEMS_LIST", "TOTALOFFERINGAMOUNT", "TOTALAMOUNTSOLD", "MINIMUMINVESTMENTACCEPTED",
            "TOTALNUMBERALREADYINVESTED"]
ISS_COLS = ["ACCESSIONNUMBER", "IS_PRIMARYISSUER_FLAG", "CIK", "ENTITYNAME", "STATEORCOUNTRY", "ENTITYTYPE",
            "YEAROFINC_TIMESPAN_CHOICE", "YEAROFINC_VALUE_ENTERED"]

# Industry groups that can contain operating start-ups. A coarse flag, not a claim that a filer is a start-up.
STARTUP_LIKE_GROUPS = {"Other Technology", "Computers", "Biotechnology", "Pharmaceuticals", "Other Health Care",
                       "Business Services"}
YES = {"true", "y", "yes", "1"}
# Amounts at or above this are treated as implausible for a single private raise.
AMOUNT_CAP = 10_000_000_000


def _read(zf: zipfile.ZipFile, name: str, cols: list[str]) -> pd.DataFrame:
    member = next(n for n in zf.namelist() if n.endswith("/" + name + ".tsv"))
    with zf.open(member) as f:
        return pd.read_csv(io.TextIOWrapper(f, encoding="utf-8", errors="replace"), sep="\t", dtype=str,
                           usecols=lambda c: c in cols, keep_default_na=False, quoting=3)


def load_raw(raw_dir: str) -> pd.DataFrame:
    """Join the tables of every quarterly zip into one frame, one row per accession number."""
    frames = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "*_d.zip"))):
        quarter = os.path.basename(path)[:6].upper()
        with zipfile.ZipFile(path) as zf:
            sub = _read(zf, "FORMDSUBMISSION", SUB_COLS)
            off = _read(zf, "OFFERING", OFF_COLS)
            iss = _read(zf, "ISSUERS", ISS_COLS)
            rel = _read(zf, "RELATEDPERSONS", ["ACCESSIONNUMBER"])["ACCESSIONNUMBER"].value_counts()
        iss = iss[iss["IS_PRIMARYISSUER_FLAG"].str.strip().str.upper().isin(["YES", "Y", "TRUE", "1"])]
        iss = iss.drop_duplicates("ACCESSIONNUMBER").drop(columns="IS_PRIMARYISSUER_FLAG")
        df = sub.merge(off, on="ACCESSIONNUMBER", how="inner").merge(iss, on="ACCESSIONNUMBER", how="left")
        df["n_related_persons"] = df["ACCESSIONNUMBER"].map(rel).fillna(0).astype(int)
        df["quarter"] = quarter
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _amount(s: pd.Series) -> pd.Series:
    """Numeric amount; the string 'Indefinite' and anything non-numeric become NaN."""
    return pd.to_numeric(s.str.strip(), errors="coerce")


def clean_formd(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningLog]:
    log = CleaningLog("sec_form_d")
    df = raw.copy()
    log.info("input_filings", len(df), "filings across all loaded quarters, one row per accession number")

    n = len(df)
    df = df[df["TESTORLIVE"].str.upper().ne("TEST")]
    log.rows("drop_test_filings", n, len(df))

    n = len(df)
    df = df.drop_duplicates("ACCESSIONNUMBER")
    log.rows("duplicate_accession_number", n, len(df))

    df["filing_date"] = pd.to_datetime(df["FILING_DATE"], format="%d-%b-%Y", errors="coerce")
    log.flag("filing_date_unparseable", df["filing_date"].isna().sum(), len(df))
    df["filing_year"] = df["filing_date"].dt.year
    df["filing_quarter"] = df["filing_date"].dt.quarter

    # Amendments are counted per issuer, then dropped: the original filing is the unit of analysis.
    df["is_amendment"] = df["SUBMISSIONTYPE"].eq("D/A")
    df["cik"] = df["CIK"].str.strip().replace("", np.nan)
    amend_per_cik = df[df["is_amendment"]].groupby("cik").size()
    df["issuer_n_amendments"] = df["cik"].map(amend_per_cik).fillna(0).astype(int)
    n = len(df)
    df = df[~df["is_amendment"]].copy()
    log.rows("drop_amendments_D/A", n, len(df), "amendments restate an earlier filing; counted per issuer, not analysed")

    pooled = df["ISPOOLEDINVESTMENTFUNDTYPE"].str.strip().str.lower().isin(YES) | \
        df["INDUSTRYGROUPTYPE"].eq("Pooled Investment Fund")
    n = len(df)
    df = df[~pooled].copy()
    log.rows("drop_pooled_investment_funds", n, len(df), "funds raise money to invest, not to operate")

    df["offering_amount"] = _amount(df["TOTALOFFERINGAMOUNT"])
    df["amount_sold"] = _amount(df["TOTALAMOUNTSOLD"])
    df["min_investment"] = _amount(df["MINIMUMINVESTMENTACCEPTED"])
    df["n_investors"] = _amount(df["TOTALNUMBERALREADYINVESTED"])
    df["offering_indefinite"] = df["TOTALOFFERINGAMOUNT"].str.strip().str.lower().eq("indefinite")
    log.flag("offering_amount_indefinite", df["offering_indefinite"].sum(), len(df), "the string 'Indefinite' becomes missing")
    zero = df["offering_amount"].le(0)
    log.flag("offering_amount_nonpositive_to_missing", zero.sum(), len(df))
    df.loc[zero, "offering_amount"] = np.nan
    df["amount_sold_zero"] = df["amount_sold"].eq(0)
    log.flag("amount_sold_zero", df["amount_sold_zero"].sum(), len(df), "kept; zero sold is real for a new offering")

    over = df["offering_amount"].ge(AMOUNT_CAP)
    log.flag("offering_amount_ge_10bn_implausible", over.sum(), len(df),
             "flagged and excluded from amount modelling; kept in the table")
    df["amount_implausible"] = over
    df["log10_offering"] = np.log10(df["offering_amount"].where(~over))
    lo, hi = df["log10_offering"].quantile([0.01, 0.99])
    df["log10_offering_winsor"] = df["log10_offering"].clip(lo, hi)
    log.info("winsorise_log10_offering", f"[{lo:.2f}, {hi:.2f}]",
             "1st and 99th percentile of this dataset, for plots only; Phase 3 recomputes from training data")

    df["sold_gt_offering"] = df["amount_sold"].gt(df["offering_amount"])
    log.flag("amount_sold_exceeds_offering", df["sold_gt_offering"].sum(), len(df), "flag only")

    yr = pd.to_numeric(df["YEAROFINC_VALUE_ENTERED"], errors="coerce")
    choice = df["YEAROFINC_TIMESPAN_CHOICE"].str.strip()
    df["inc_year"] = yr.where(yr.between(1800, 2030))
    log.flag("year_of_inc_out_of_range", (yr.notna() & df["inc_year"].isna()).sum(), len(df), "set to missing")
    df["issuer_age"] = (df["filing_year"] - df["inc_year"]).where(lambda a: a >= 0)
    df["incorporated_within_5y"] = choice.str.contains("Within", case=False, na=False)
    log.flag("issuer_age_unknown", df["issuer_age"].isna().sum(), len(df))

    df["industry_group"] = df["INDUSTRYGROUPTYPE"].replace("", np.nan)
    log.flag("industry_group_missing", df["industry_group"].isna().sum(), len(df))
    df["revenue_range"] = df["REVENUERANGE"].replace("", np.nan)
    df["revenue_declined"] = df["revenue_range"].eq("Decline to Disclose")
    df["revenue_not_applicable"] = df["revenue_range"].eq("Not Applicable")
    log.flag("revenue_decline_to_disclose", df["revenue_declined"].sum(), len(df), "kept as its own category")
    df["startup_like_group"] = df["industry_group"].isin(STARTUP_LIKE_GROUPS)
    df["entity_type"] = df["ENTITYTYPE"].replace("", np.nan)
    df["state_or_country"] = df["STATEORCOUNTRY"].replace("", np.nan)

    df["issuer_filing_count"] = df.groupby("cik")["ACCESSIONNUMBER"].transform("size")
    log.flag("cik_missing", df["cik"].isna().sum(), len(df))
    df["is_equity"] = df["ISEQUITYTYPE"].str.strip().str.lower().isin(YES)
    df["is_debt"] = df["ISDEBTTYPE"].str.strip().str.lower().isin(YES)
    df["exemptions"] = df["FEDERALEXEMPTIONS_ITEMS_LIST"].replace("", np.nan)

    log.info("usable_for_amount_models",
             int((df["startup_like_group"] & df["offering_amount"].notna() & ~df["amount_implausible"]).sum()),
             "start-up-like group with a finite, plausible offering amount")

    keep = ["ACCESSIONNUMBER", "cik", "ENTITYNAME", "quarter", "filing_date", "filing_year", "filing_quarter",
            "industry_group", "startup_like_group", "revenue_range", "revenue_declined", "revenue_not_applicable",
            "entity_type", "state_or_country", "inc_year", "issuer_age", "incorporated_within_5y", "is_equity",
            "is_debt", "exemptions", "offering_amount", "offering_indefinite", "amount_sold", "amount_sold_zero",
            "sold_gt_offering", "min_investment", "n_investors", "amount_implausible", "log10_offering",
            "log10_offering_winsor", "n_related_persons", "issuer_filing_count", "issuer_n_amendments"]
    out = df[keep].rename(columns={"ACCESSIONNUMBER": "accession", "ENTITYNAME": "issuer_name"})
    log.info("output_shape", f"{out.shape[0]}x{out.shape[1]}")
    return out.reset_index(drop=True), log
