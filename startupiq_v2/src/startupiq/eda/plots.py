"""Task 4 figures (matplotlib, Agg). One function per figure; each saves a PNG and returns the path."""
from __future__ import annotations

import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, GRAY, INK = "#2a78d6", "#b8bcc2", "#222222"


def _style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", alpha=0.2)


def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    return p


def yc_overview(df: pd.DataFrame, out: str) -> str:
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.5))
    vc = df["industry"].fillna("Unspecified").value_counts().sort_values()
    axs[0, 0].barh(vc.index, vc.values, color=BLUE)
    for i, v in enumerate(vc.values):
        axs[0, 0].text(v, i, f" {v} ({v / len(df):.0%})", va="center", fontsize=8)
    _style(axs[0, 0], "Companies per industry (class balance)")
    tags = Counter(t for ts in df["tags"] for t in (ts if isinstance(ts, list) else str(ts).split("|")) if t)
    top = tags.most_common(15)[::-1]
    axs[0, 1].barh([t for t, _ in top], [c for _, c in top], color=BLUE)
    _style(axs[0, 1], "15 most frequent tags")
    axs[1, 0].hist(df["text_words"].clip(upper=df["text_words"].quantile(0.99)), bins=40, color=BLUE)
    _style(axs[1, 0], "Description length (words, capped at 99th pct)", "words", "companies")
    by = df.dropna(subset=["batch_year"]).groupby("batch_year").size()
    axs[1, 1].bar(by.index.astype(int), by.values, color=BLUE)
    _style(axs[1, 1], "Companies per batch year", "year")
    return _save(fig, out, "yc_overview.png")


def missingness(df: pd.DataFrame, out: str, name: str, title: str, cols=None) -> str:
    cols = cols or list(df.columns)
    miss = df[cols].isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.32 * len(miss) + 1.5)))
    if len(miss):
        ax.barh(miss.index[::-1], miss.values[::-1] * 100, color=BLUE)
        for i, v in enumerate(miss.values[::-1]):
            ax.text(v * 100, i, f" {v:.1%}", va="center", fontsize=8)
    _style(ax, title, "% missing")
    return _save(fig, out, name)


def yc_text_by_industry(df: pd.DataFrame, out: str) -> str:
    g = df.dropna(subset=["industry"])
    order = g.groupby("industry")["text_words"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot([g.loc[g["industry"] == i, "text_words"].clip(upper=400) for i in order], vert=False,
               labels=order, patch_artist=True, boxprops=dict(facecolor=BLUE, alpha=0.5),
               medianprops=dict(color=INK))
    _style(ax, "Description length by industry (words, capped at 400)", "words")
    return _save(fig, out, "yc_text_length_by_industry.png")


def tag_cooccurrence(df: pd.DataFrame, out: str, k: int = 20) -> str:
    rows = [[t for t in (ts if isinstance(ts, list) else str(ts).split("|")) if t] for ts in df["tags"]]
    top = [t for t, _ in Counter(t for r in rows for t in r).most_common(k)]
    idx = {t: i for i, t in enumerate(top)}
    m = np.zeros((k, k))
    for r in rows:
        ids = [idx[t] for t in set(r) if t in idx]
        for a in ids:
            for b in ids:
                m[a, b] += 1
    d = np.diag(m).copy(); d[d == 0] = 1
    jac = m / (d[:, None] + d[None, :] - m)
    np.fill_diagonal(jac, np.nan)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(jac, cmap="Blues")
    ax.set_xticks(range(k)); ax.set_xticklabels(top, rotation=90, fontsize=8)
    ax.set_yticks(range(k)); ax.set_yticklabels(top, fontsize=8)
    fig.colorbar(im, ax=ax, label="Jaccard overlap")
    ax.set_title("Tag co-occurrence (top 20 tags)", loc="left")
    return _save(fig, out, "yc_tag_cooccurrence.png")


def yc_team_size(df: pd.DataFrame, out: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(df["team_size_log"].dropna(), bins=40, color=BLUE)
    out_n = int(df["team_size_outlier"].sum())
    _style(ax, f"log(1+team size); {out_n} extreme outliers flagged", "log1p(team size)", "companies")
    return _save(fig, out, "yc_team_size_log.png")


def formd_overview(df: pd.DataFrame, out: str) -> str:
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.5))
    q = df.groupby("quarter").size()
    axs[0, 0].bar(q.index, q.values, color=BLUE); axs[0, 0].tick_params(axis="x", rotation=45)
    _style(axs[0, 0], "Operating-company initial filings per quarter")
    vc = df["industry_group"].value_counts().head(12).sort_values()
    colors = [BLUE if g in {"Other Technology", "Computers", "Biotechnology", "Pharmaceuticals",
                            "Other Health Care", "Business Services"} else GRAY for g in vc.index]
    axs[0, 1].barh(vc.index, vc.values, color=colors)
    _style(axs[0, 1], "Industry groups (blue = start-up-like)")
    x = df["log10_offering"].dropna()
    axs[1, 0].hist(x, bins=50, color=BLUE)
    axs[1, 0].axvline(x.median(), color=INK, ls="--", lw=1)
    _style(axs[1, 0], f"Offering amount, log10 dollars (median ${10 ** x.median():,.0f})", "log10(offering amount)", "filings")
    sub = df[df["startup_like_group"] & df["log10_offering"].notna()]
    order = sub.groupby("industry_group")["log10_offering"].median().sort_values().index
    axs[1, 1].boxplot([sub.loc[sub["industry_group"] == g, "log10_offering"] for g in order], vert=False,
                      labels=order, patch_artist=True, boxprops=dict(facecolor=BLUE, alpha=0.5),
                      medianprops=dict(color=INK))
    _style(axs[1, 1], "Amount by start-up-like group", "log10(offering amount)")
    return _save(fig, out, "formd_overview.png")


def formd_revenue_age(df: pd.DataFrame, out: str) -> str:
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    vc = df.loc[df["startup_like_group"], "revenue_range"].fillna("(blank)").value_counts().sort_values()
    axs[0].barh(vc.index, vc.values, color=BLUE)
    _style(axs[0], "Revenue range, start-up-like filings")
    axs[1].hist(df["issuer_age"].dropna().clip(upper=40), bins=40, color=BLUE)
    _style(axs[1], "Issuer age at filing (years, capped at 40)", "years", "filings")
    return _save(fig, out, "formd_revenue_age.png")


def cleaning_table(log, out: str, name: str, title: str) -> str:
    rows = []
    for s in log.to_dict()["steps"]:
        if s["kind"] == "rows":
            rows.append([s["rule"], f"{s['rows_before']:,}", f"{s['rows_after']:,}", f"-{s['rows_removed']:,}"])
        elif s["kind"] == "values":
            rows.append([s["rule"], f"{s['of_total']:,}", "", f"{s['affected']:,} flagged/fixed"])
    fig, ax = plt.subplots(figsize=(10, 0.34 * len(rows) + 1.2)); ax.axis("off")
    t = ax.table(cellText=rows, colLabels=["rule", "rows before / total", "rows after", "effect"], loc="center",
                 cellLoc="left", colWidths=[0.5, 0.17, 0.13, 0.2])
    t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.25)
    ax.set_title(title, loc="left")
    return _save(fig, out, name)
