"""Task 4 EDA: a small set of figures/tables that support cleaning and modelling decisions."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESOLVED = ["acquired", "closed"]


def _save(fig, out_dir, name):
    fig.tight_layout()
    fig.savefig(out_dir / name, dpi=110)
    plt.close(fig)


def _rate_by(df, col, top=12, min_n=100):
    r = df[df.status.isin(RESOLVED)].copy()
    r["acq"] = (r.status == "acquired").astype(int)
    g = r.groupby(col)["acq"].agg(["count", "mean"])
    return g[g["count"] >= min_n].sort_values("count", ascending=False).head(top)


def run_eda(df: pd.DataFrame, out_dir):
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    res = df[df.status.isin(RESOLVED)].copy()
    summary = {}

    # 1. target distribution
    fig, ax = plt.subplots(figsize=(5, 3.5))
    counts = df.status.fillna("unknown").value_counts()
    counts.plot.bar(ax=ax, color="steelblue")
    ax.set_title("Company status (all companies)")
    for i, v in enumerate(counts):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    _save(fig, fig_dir, "01_status_distribution.png")
    summary["status_counts"] = counts.to_dict()

    # 2. missingness (after cleaning)
    miss = (df.isna().mean() * 100).sort_values(ascending=False)
    miss = miss[miss > 0].head(20)
    fig, ax = plt.subplots(figsize=(6, 5))
    miss[::-1].plot.barh(ax=ax, color="indianred")
    ax.set_xlabel("% missing")
    ax.set_title("Missing values (cleaned data, top 20)")
    _save(fig, fig_dir, "02_missingness.png")

    # 3. funding distribution raw vs log
    f = df.funding_total_usd.dropna()
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
    axs[0].hist(f.clip(upper=f.quantile(.99)) / 1e6, bins=50, color="steelblue")
    axs[0].set_title("Total funding, USD M (clipped at p99 for display)")
    axs[1].hist(np.log10(f), bins=50, color="seagreen")
    axs[1].set_title("log10(total funding USD)")
    _save(fig, fig_dir, "03_funding_distribution.png")

    # 4. funding by outcome (resolved)
    fig, ax = plt.subplots(figsize=(5, 4))
    data = [np.log10(res[res.status == s].funding_total_usd.dropna()) for s in RESOLVED]
    ax.boxplot(data, labels=RESOLVED)
    ax.set_ylabel("log10(total funding USD)")
    ax.set_title("Funding by outcome")
    _save(fig, fig_dir, "04_funding_by_outcome.png")
    summary["median_funding_by_status"] = {
        s: float(res[res.status == s].funding_total_usd.median()) for s in RESOLVED}

    # 5. founding year and outcome
    r = res.dropna(subset=["founded_year"])
    r = r[r.founded_year >= 1990]
    yr = r.groupby("founded_year").status.value_counts().unstack(fill_value=0)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
    yr.plot.bar(stacked=True, ax=axs[0], width=.9)
    axs[0].set_title("Resolved companies by founding year")
    (yr["acquired"] / yr.sum(axis=1)).plot(ax=axs[1], marker="o")
    axs[1].set_title("Share acquired by founding year")
    axs[1].set_ylim(0, 1)
    _save(fig, fig_dir, "05_founding_year_outcome.png")
    summary["acquired_share_by_founding_year"] = (yr["acquired"] / yr.sum(axis=1)).round(2).to_dict()

    # 6. age at first funding vs outcome (raw dates, invalid ones excluded)
    age = (res.first_funding_at - res.founded_at).dt.days / 365.25
    age = age.where(age >= -1)
    res = res.assign(age_first_funding=age)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot([age[res.status == s].dropna().clip(upper=15) for s in RESOLVED], labels=RESOLVED)
    ax.set_title("Age at first funding (years, capped 15)")
    _save(fig, fig_dir, "06_age_first_funding.png")

    # 7. market and country: volume and acquired rate
    for col, name in [("market", "07_market"), ("country_code", "08_country")]:
        g = _rate_by(df, col, top=12)
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.barh(g.index[::-1], g["mean"][::-1], color="steelblue")
        ax.axvline((res.status == "acquired").mean(), color="k", ls="--", lw=1)
        ax.set_xlabel("share acquired among resolved (dashed = overall)")
        ax.set_title(f"Acquired rate by {col} (top 12 by size, n>=100)")
        _save(fig, fig_dir, f"{name}_acquired_rate.png")
        g.round(3).to_csv(out_dir / f"{name}_acquired_rate.csv")

    # 8. funding rounds and funding types vs outcome
    res["acq"] = (res.status == "acquired").astype(int)
    rr = res.groupby(res.funding_rounds.clip(upper=6)).acq.agg(["count", "mean"])
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(rr.index.astype(str), rr["mean"], color="seagreen")
    ax.set_title("Acquired share by number of funding rounds (6 = 6+)")
    _save(fig, fig_dir, "09_rounds_vs_outcome.png")
    summary["acquired_share_by_rounds"] = rr["mean"].round(2).to_dict()

    types = ["seed", "venture", "angel", "grant", "private_equity", "debt_financing", "convertible_note"]
    tab = pd.DataFrame({t: res.groupby("acq")[t].apply(lambda x: (x > 0).sum() / x.notna().sum()) for t in types}).T
    tab.columns = ["closed", "acquired"]
    tab.round(3).to_csv(out_dir / "funding_type_presence_by_outcome.csv")

    # 9. correlation of numeric features
    num = pd.DataFrame({
        "log_funding": res.funding_total_usd_log1p, "rounds": res.funding_rounds.astype(float),
        "log_seed": np.log1p(res.seed), "log_venture": np.log1p(res.venture),
        "founded_year": res.founded_year.astype(float), "age_first_funding": age, "acquired": res.acq})
    c = num.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(c, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(c)), c.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(c)), c.columns)
    for i in range(len(c)):
        for j in range(len(c)):
            ax.text(j, i, f"{c.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im)
    ax.set_title("Spearman correlation (resolved companies)")
    _save(fig, fig_dir, "10_correlation.png")
    c.round(3).to_csv(out_dir / "correlation_resolved.csv")

    # numeric summary
    df[["funding_total_usd", "funding_rounds", "founded_year", "seed", "venture"]].astype(float).describe(
        percentiles=[.5, .9, .99]).round(1).to_csv(out_dir / "numeric_summary.csv")
    pd.Series(summary).to_json(out_dir / "eda_findings.json", indent=2, default_handler=str)
    print(pd.Series(summary).to_string())
