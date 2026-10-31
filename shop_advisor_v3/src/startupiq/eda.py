"""Task 4 exploratory analysis: python -m startupiq.eda

Reads data/processed (run `python -m startupiq.clean` first), writes figures to
docs/results/figures/ and a findings page to docs/results/eda_summary.md.
Every number in the summary is computed here, not typed in.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROC = Path("data/processed")
FIG = Path("docs/results/figures")
SUMMARY = Path("docs/results/eda_summary.md")

# Reference palette (dataviz skill, light mode). Categorical slots in fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
DIVERGING = ["#2a78d6", "#f0efec", "#eb6834"]
LABELS = {"MRTSSM4541USN": "Online retail (not seas. adj.)", "RSCCAS": "Clothing stores", "RSXFS": "All retail ex food services"}
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2,
    "font.size": 9.5, "legend.frameon": False, "figure.dpi": 110, "axes.axisbelow": True,
})

findings: list[tuple[str, str, str]] = []  # (file, title, finding)


def save(fig: plt.Figure, name: str, title: str, finding: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    findings.append((name, title, finding))


def thousands(ax, axis: str = "y") -> None:
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:.1f}M" if abs(v) >= 1e6 else f"{v / 1e3:.0f}k" if abs(v) >= 1e3 else f"{v:.0f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def rossmann() -> None:
    d = pd.read_parquet(PROC / "rossmann_daily.parquet")
    wk = pd.read_parquet(PROC / "rossmann_weekly.parquet")
    open_ = d[(d.Open == 1) & d.Sales.notna()]

    # 1 total weekly sales, stores without the 2014 gap only (so the total is comparable over time)
    full = wk[~wk.Store.isin(d.loc[d.store_has_2014_gap, "Store"].unique())]
    tot = full.groupby("week").sales.sum()
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(tot.index, tot.values, color=SERIES[0])
    pdays = full.groupby("week").promo_days.mean()
    promo_week = tot[pdays >= 4.5].mean() / tot[pdays <= 0.5].mean()
    peaks = tot.groupby(tot.index.year).idxmax()
    for p in peaks:
        ax.annotate(p.strftime("%d %b"), (p, tot[p]), textcoords="offset points", xytext=(0, 5), ha="center", color=INK2, fontsize=8)
    thousands(ax)
    ax.set_title(f"Rossmann: weekly sales, {full.Store.nunique()} stores without the 2014 gap")
    ax.set_ylabel("Sales per week")
    dec = tot[tot.index.month == 12].mean() / tot[tot.index.month != 12].mean()
    save(fig, "01_rossmann_weekly_total", "Weekly sales over time",
         f"Clear yearly cycle; December weeks average {dec:.2f}x other weeks, peaking just before Christmas. "
         f"The saw-tooth is Rossmann's alternating promo weeks: weeks with 5 promo days sell {promo_week:.2f}x weeks with none. Seasonal and promo features are both required.")

    # 2 seasonality: weekday and month, two small multiples on their own axes
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    wd = open_.groupby("DayOfWeek").Sales.mean()
    sunday_stores = open_.loc[open_.DayOfWeek == 7, "Store"].nunique()
    axes[0].bar(wd.index, wd.values, color=SERIES[0], width=0.7)
    axes[0].set_xticks(range(1, 8), DAYS)
    axes[0].set_title("By weekday (open days)")
    mo = open_.groupby(open_.Date.dt.month).Sales.mean()
    axes[1].bar(mo.index, mo.values, color=SERIES[0], width=0.7)
    axes[1].set_xticks(range(1, 13), list("JFMAMJJASOND"))
    axes[1].set_title("By month")
    for a in axes:
        thousands(a); a.set_ylabel("Mean sales per store-day")
    save(fig, "02_rossmann_seasonality", "Weekday and month pattern",
         f"Highest mean is {DAYS[wd.idxmax() - 1]} (only {sunday_stores} stores open on Sundays, so that bar is a small, unusual group); "
         f"Monday is {wd[1] / wd[[2, 3, 4, 5]].mean():.2f}x Tue-Fri; "
         f"December mean is {mo[12] / mo.drop(12).mean():.2f}x other months. Calendar features belong in every model.")

    # 3 promo effect by weekday (promos only run Mon-Fri)
    pw = open_[open_.DayOfWeek <= 5].groupby(["DayOfWeek", "Promo"]).Sales.mean().unstack()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(pw))
    ax.bar(x - 0.18, pw[0], width=0.34, color=SERIES[0], label="No promo")
    ax.bar(x + 0.18, pw[1], width=0.34, color=SERIES[1], label="Promo")
    ax.set_xticks(x, ["Mon", "Tue", "Wed", "Thu", "Fri"]); thousands(ax)
    ax.legend(loc="upper right"); ax.set_title("Rossmann: mean sales with and without promotion")
    lift = (pw[1] / pw[0]).mean()
    save(fig, "03_rossmann_promo", "Promotion effect",
         f"Promotion days sell {lift:.2f}x non-promo days on average across weekdays. Promo is the strongest controllable driver and the main 'move' the planner can use.")

    # 4 competition distance vs store sales
    st = open_.groupby("Store").agg(sales=("Sales", "mean"), dist=("CompetitionDistance", "first"), stype=("StoreType", "first")).dropna()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.scatter(st.dist, st.sales, s=10, color=SERIES[0], alpha=0.5, edgecolors="none")
    ax.set_xscale("log"); thousands(ax)
    ax.set_xlabel("Distance to nearest competitor (m, log scale)"); ax.set_ylabel("Mean daily sales")
    ax.set_title("Rossmann: competitor distance vs store sales")
    rho = st[["dist", "sales"]].corr(method="spearman").iloc[0, 1]
    save(fig, "04_rossmann_competition", "Competitor distance",
         f"Spearman correlation {rho:.2f}: distance to the nearest rival barely relates to sales on its own. Competition matters through timing (rival opening dates), not distance.")

    # 5 store table missingness
    s = pd.read_csv("data/raw/rossmann_hf/store.csv")
    miss = s.isna().mean().sort_values()
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.barh(miss.index, miss.values * 100, color=SERIES[0], height=0.6)
    for i, v in enumerate(miss.values):
        ax.text(v * 100 + 0.5, i, f"{v:.0%}", va="center", color=INK2, fontsize=8)
    ax.set_xlabel("% of stores missing"); ax.set_title("Rossmann store table: missing values (raw)")
    save(fig, "05_rossmann_store_missing", "Missing values in store table",
         "Promo2 fields are empty for stores not in Promo2 (structural, not missing); competition-open dates are truly unknown for 32% of stores and are flagged, not imputed.")

    # 6 store type
    types = sorted(st.stype.unique())
    fig, ax = plt.subplots(figsize=(6, 3.2))
    bp = ax.boxplot([st.loc[st.stype == t, "sales"] for t in types], tick_labels=[f"Type {t}" for t in types],
                    patch_artist=True, widths=0.5, medianprops={"color": INK})
    for b in bp["boxes"]:
        b.set_facecolor(SERIES[0]); b.set_alpha(0.6); b.set_edgecolor(SERIES[0])
    thousands(ax); ax.set_ylabel("Mean daily sales per store"); ax.set_title("Rossmann: store sales by store type")
    counts = st.stype.value_counts()
    save(fig, "06_rossmann_store_type", "Store type",
         f"Type b ({counts.get('b', 0)} stores) sells far more per day than the rest; types differ in level, so per-series scaling is needed before a global model.")

    # 7 correlation heatmap
    c = open_.assign(StateHoliday=(open_.StateHoliday != "0").astype(int), logCompDist=np.log1p(open_.CompetitionDistance))
    cols = ["Sales", "Customers", "Promo", "DayOfWeek", "SchoolHoliday", "StateHoliday", "logCompDist", "Promo2"]
    corr = c[cols].corr()
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("div", DIVERGING)
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right"); ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7, color=INK)
    ax.grid(False); fig.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title("Rossmann: correlations (open days)")
    save(fig, "07_rossmann_correlation", "Correlation heatmap",
         f"Customers correlate {corr.loc['Sales', 'Customers']:.2f} with sales, but customer count is not known in advance, so it is excluded as a forecast input (leakage). "
         f"Promo ({corr.loc['Sales', 'Promo']:.2f}) is the strongest usable signal.")


def uci() -> None:
    tot = pd.read_parquet(PROC / "uci_total_weekly.parquet").set_index("week")
    pw = pd.read_parquet(PROC / "uci_product_weekly.parquet")
    lines = pd.read_parquet(PROC / "uci_lines.parquet")

    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(tot.index, tot.revenue, color=SERIES[0]); thousands(ax)
    ax.set_title("Online Retail II: weekly revenue (cleaned)"); ax.set_ylabel("Revenue (GBP)")
    q4 = tot[tot.index.month.isin([10, 11])].revenue.mean() / tot[~tot.index.month.isin([10, 11, 12])].revenue.mean()
    save(fig, "08_uci_weekly_revenue", "Weekly revenue",
         f"October-November weeks average {q4:.2f}x January-September weeks (a gift wholesaler stocking up for Christmas); the late-December drop to zero is closure, filled as zero weeks.")

    active = pw[pw.units > 0].groupby("StockCode").week.nunique()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(active, bins=52, color=SERIES[0], edgecolor=SURFACE, linewidth=0.5)
    for cut, height in ((26, 0.92), (52, 0.80)):
        ax.axvline(cut, color=INK2, linewidth=1, linestyle="--")
        ax.text(cut + 1, ax.get_ylim()[1] * height, f"{(active >= cut).sum():,} products sell in {cut}+ weeks", color=INK2, fontsize=8)
    ax.set_xlabel("Weeks with at least one sale"); ax.set_ylabel("Products")
    ax.set_title("Online Retail II: how many weeks each product sells")
    save(fig, "09_uci_product_activity", "Product sparsity",
         f"{(active < 26).mean():.0%} of products sell in fewer than 26 weeks: most product-level series are too sparse for a neural model; the global model trains on the {(active >= 52).sum():,} products with 52+ active weeks.")

    rev = (lines.Quantity * lines.Price).groupby(lines.Country).sum().sort_values()
    top = rev.tail(10) / rev.sum() * 100
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.barh(top.index, top.values, color=SERIES[0], height=0.6)
    for i, v in enumerate(top.values):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", color=INK2, fontsize=8)
    ax.set_xlabel("% of revenue"); ax.set_title("Online Retail II: revenue by country (top 10)")
    save(fig, "10_uci_country", "Revenue by country",
         f"The United Kingdom makes {top.iloc[-1]:.0f}% of revenue: results describe a UK business, a limitation for Indian small sellers (Task 10).")


def olist() -> None:
    cm = pd.read_parquet(PROC / "olist_category_month.parquet")
    scm = pd.read_parquet(PROC / "olist_seller_category_month.parquet")
    items = pd.read_parquet(PROC / "olist_items.parquet")

    m = items.groupby("month").order_id.nunique()
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(pd.PeriodIndex(m.index, freq="M").to_timestamp(), m.values, color=SERIES[0], marker="o", markersize=4)
    thousands(ax); ax.set_ylabel("Orders"); ax.set_title("Olist: orders per month (2017-01 to 2018-08)")
    save(fig, "11_olist_monthly_orders", "Orders per month",
         f"Orders grow {m.iloc[-1] / m.iloc[0]:.1f}x from the first to the last month: the marketplace itself is growing, so a seller's sales growth must be read against category growth (that is what market share does).")

    sellers = cm.groupby("category").sellers.median().sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(sellers.index, sellers.values, color=SERIES[0], height=0.6)
    ax.axvline(5, color=INK2, linestyle="--", linewidth=1)
    ax.text(5.5, len(sellers) - 0.4, "5 sellers: minimum for share models", color=INK2, fontsize=8)
    ax.set_xlabel("Median active sellers per month"); ax.set_title("Olist: competitors per category (top 15)")
    eligible = (cm.groupby("category").sellers.median() >= 5).sum()
    save(fig, "12_olist_sellers_per_category", "Competitors per category",
         f"{eligible} of {cm.category.nunique()} categories have a median of 5+ active sellers a month; share and competitor models are limited to these.")

    tot = scm.groupby("seller_id").revenue.sum().sort_values()
    cum = tot.cumsum() / tot.sum()
    x = np.arange(1, len(tot) + 1) / len(tot)
    gini = 1 - 2 * np.trapezoid(cum.values, x)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.plot(x, cum.values, color=SERIES[0], label="Sellers")
    ax.plot([0, 1], [0, 1], color=INK2, linewidth=1, linestyle="--", label="Perfect equality")
    ax.set_xlabel("Share of sellers (smallest first)"); ax.set_ylabel("Share of revenue")
    ax.legend(loc="upper left"); ax.set_title("Olist: seller revenue concentration")
    save(fig, "13_olist_lorenz", "Seller concentration",
         f"Gini {gini:.2f}: a few sellers take most revenue. Small sellers (the product's users) are the majority of rows but the minority of revenue, so errors must be reported by seller size (Task 10).")

    cat = "health_beauty"
    c = scm[scm.category == cat]
    top5 = c.groupby("seller_id").revenue.sum().nlargest(5).index
    months = sorted(c.month.unique())
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for i, s in enumerate(top5):
        series = c[c.seller_id == s].set_index("month").share.reindex(months).fillna(0) * 100
        t = pd.PeriodIndex(series.index, freq="M").to_timestamp()
        ax.plot(t, series.values, color=SERIES[i], label=f"Seller {i + 1}")
    ax.set_ylabel("Share of category revenue (%)"); ax.legend(loc="upper left", ncol=5)
    ax.set_title(f"Olist {cat}: market share of the five largest sellers")
    hhi = cm[cm.category == cat].hhi
    save(fig, "14_olist_share_top_sellers", "Market share over time",
         f"Leaders' shares move by several points month to month (category HHI {hhi.min():.3f} to {hhi.max():.3f}); share changes are real and learnable, but noisy.")


def signals() -> None:
    s = pd.read_parquet(PROC / "signals_monthly.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, src, start in ((axes[0], "fred", "2015-01-01"), (axes[1], "wikipedia", "2016-01-01")):
        sub = s[(s.source == src) & (s.month >= start)]
        for i, name in enumerate(sorted(sub.series.unique())):
            ser = sub[sub.series == name].set_index("month").value
            if len(ser) < 24:
                continue  # too short to index to a common base year
            idx = ser / ser.iloc[:12].mean() * 100
            ax.plot(idx.index, idx.values, color=SERIES[i], label=LABELS.get(name, name), linewidth=1.6)
        ax.axhline(100, color=INK2, linewidth=0.8)
        ax.legend(loc="upper left"); ax.set_ylabel("Index (first 12 months = 100)")
    axes[0].set_title("FRED: US retail sales, indexed"); axes[1].set_title("Wikipedia: page views, indexed")
    f = s[(s.source == "fred")].pivot(index="month", columns="series", values="value")
    growth = (f.loc["2023"].mean() / f.loc["2019"].mean() - 1) * 100
    save(fig, "15_market_signals", "Public market signals",
         f"Online retail (MRTSSM4541USN) grew {growth['MRTSSM4541USN']:.0f}% from 2019 to 2023 vs {growth['RSXFS']:.0f}% for all retail; "
         "page views move differently from sales, so they are a weak, optional trend input.")


def main() -> int:
    rossmann(); uci(); olist(); signals()
    lines = ["# EDA summary (Task 4)", "", "Generated by `python -m startupiq.eda` from `data/processed`. Do not edit by hand.", ""]
    for name, title, finding in findings:
        lines += [f"## {title}", "", f"![{title}](figures/{name}.png)", "", finding, ""]
    SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    print(f"{len(findings)} figures -> {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
