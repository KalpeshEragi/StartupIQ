"""Read the forecast file written by `python -m market.forecast` and shape it for one startup.

The file holds, per sector: the last 52 weeks of the market interest index, the network ensemble's and exponential
smoothing's 12-week forecasts (P10 / P50 / P90), and each model's error on the 12 held-out test weeks. For each
sector the app serves whichever model was more accurate there, and says so.
"""
from __future__ import annotations

import json
import math
import re
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

from app.core.config import get_settings

# The 19 sectors the models cover (keys match ml/market/sectors.py), with words that suggest each one.
SECTORS: dict[str, tuple[str, tuple[str, ...]]] = {
    "food_beverage": ("Food and beverage brands (D2C)", ("food", "beverage", "drink", "snack", "tea", "chai", "coffee", "fmcg", "bakery")),  # noqa: E501
    "fashion_apparel": ("Fashion and apparel", ("fashion", "apparel", "clothing", "wear", "footwear", "textile", "jewel")),  # noqa: E501
    "beauty_personal_care": ("Beauty and personal care", ("beauty", "cosmetic", "skin", "personal care", "salon", "hair")),  # noqa: E501
    "quick_commerce_grocery": ("Grocery and quick commerce", ("grocery", "quick commerce", "q-commerce", "kirana", "supermarket")),  # noqa: E501
    "food_delivery_restaurants": ("Food delivery and restaurants", ("restaurant", "food delivery", "cloud kitchen", "cafe", "qsr")),  # noqa: E501
    "edtech": ("Education technology", ("edtech", "education", "learning", "school", "coaching", "course", "exam", "tutor")),  # noqa: E501
    "fintech_payments": ("Fintech and payments", ("fintech", "payment", "lending", "credit", "upi", "banking", "neobank", "insurtech")),  # noqa: E501
    "investing_wealth": ("Investing and wealth", ("wealth", "invest", "trading", "broking", "mutual fund", "crypto")),  # noqa: E501
    "saas_software": ("SaaS and business software", ("saas", "software", "b2b software", "crm", "erp", "devtool", "api")),  # noqa: E501
    "ai_tools": ("AI tools and products", ("ai", "artificial intelligence", "machine learning", "llm", "genai", "chatbot")),  # noqa: E501
    "health_wellness": ("Health and wellness", ("health", "wellness", "fitness", "medical", "pharma", "clinic", "diagnostic")),  # noqa: E501
    "travel_hospitality": ("Travel and hospitality", ("travel", "hospitality", "hotel", "tourism", "trip", "stay")),  # noqa: E501
    "mobility_ev": ("Mobility and electric vehicles", ("mobility", "ev", "electric vehicle", "auto", "ride", "scooter", "bike")),  # noqa: E501
    "gaming_esports": ("Gaming and esports", ("gaming", "game", "esports", "fantasy")),  # noqa: E501
    "media_entertainment": ("Media, OTT and creators", ("media", "ott", "content", "creator", "entertainment", "music", "podcast")),  # noqa: E501
    "real_estate_home": ("Real estate and home", ("real estate", "proptech", "home", "interior", "furniture", "coworking")),  # noqa: E501
    "jobs_hr": ("Jobs, HR and gig work", ("hr", "hiring", "recruit", "jobs", "talent", "payroll", "gig")),  # noqa: E501
    "logistics_supply": ("Logistics and B2B supply", ("logistics", "supply chain", "shipping", "delivery", "b2b", "warehouse")),  # noqa: E501
    "agritech": ("Agriculture and agritech", ("agri", "farm", "agriculture", "dairy", "crop")),  # noqa: E501
}


def suggest(sector_text: str) -> str | None:
    """Best-matching sector key for a free-text sector such as "D2C beverages", or None."""
    text = sector_text.lower()
    best, score = None, 0
    for key, (_, words) in SECTORS.items():
        hits = sum(1 for w in words if re.search(rf"\b{re.escape(w)}", text))
        if hits > score:
            best, score = key, hits
    return best


@lru_cache(maxsize=4)
def _load(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load() -> dict | None:
    """The forecast file, or None when it has not been generated yet. Re-read when the file changes."""
    path = Path(get_settings().market_forecast_path)
    if not path.exists():
        return None
    return _load(str(path), path.stat().st_mtime)


def outlook(data: dict, key: str) -> dict:
    """One sector's outlook, using the model that did better on the held-out weeks for this sector."""
    s = data["sectors"][key]
    bt = s["backtest"]
    use_network = bt["network_mase"] <= bt["smoothing_mase"]
    fc = s["forecast"] if use_network else s["smoothing"]
    recent = sum(h["index"] for h in s["history"][-4:]) / 4
    end = fc[-4:]

    def change(q: str, rows: list[dict]) -> float:
        return round(sum(r[q] for r in rows) / len(rows) / recent - 1, 4)

    peak = max(fc, key=lambda r: r["p50"])
    return {
        "key": key, "label": s["label"], "history": s["history"], "forecast": fc,
        "model": "network" if use_network else "smoothing",
        "model_label": "Deep learning ensemble (MLP, CNN, LSTM)" if use_network else "Exponential smoothing",
        "backtest": bt,
        "change_4w": change("p50", fc[:4]),
        "change_12w": change("p50", end), "change_12w_low": change("p10", end), "change_12w_high": change("p90", end),
        "peak_week": peak["week"],
        "monthly": s["monthly"],
    }


def _pct_change(a: float, b: float) -> float | None:
    return None if not a else b / a - 1


def you_vs_market(revenue_by_month: dict[str, Decimal], market_by_month: dict[str, float]) -> dict:
    """Compare the startup's monthly revenue with its market's monthly interest index over the months both have."""
    months = sorted(m for m in revenue_by_month if m in market_by_month)
    rows = [{"month": m, "revenue": float(revenue_by_month[m]), "market": market_by_month[m]} for m in months]
    out: dict = {"months": rows, "your_change": None, "market_change": None, "correlation": None, "summary": None}
    if len(rows) < 2:
        out["summary"] = "Upload at least two months of statements to compare your revenue with your market."
        return out
    a, b = rows[-2], rows[-1]
    yours, market = _pct_change(a["revenue"], b["revenue"]), _pct_change(a["market"], b["market"])
    out["your_change"], out["market_change"] = yours, market
    if len(rows) >= 6:  # month-on-month moves need a few months before a correlation means anything
        ry = [_pct_change(x["revenue"], y["revenue"]) for x, y in zip(rows, rows[1:], strict=False)]
        rm = [_pct_change(x["market"], y["market"]) for x, y in zip(rows, rows[1:], strict=False)]
        pairs = [(p, q) for p, q in zip(ry, rm, strict=True) if p is not None and q is not None]
        if len(pairs) >= 5:
            out["correlation"] = round(_corr([p for p, _ in pairs], [q for _, q in pairs]), 2)
    if yours is not None and market is not None:
        ahead = yours - market
        verb = "grew faster than" if ahead > 0.02 else "lagged" if ahead < -0.02 else "moved with"
        out["summary"] = (f"From {a['month']} to {b['month']} your revenue changed {yours:+.0%} while interest in your "
                          f"market changed {market:+.0%}: you {verb} your market.")
    return out


def _corr(x: list[float], y: list[float]) -> float:
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sx = math.sqrt(sum((v - mx) ** 2 for v in x))
    sy = math.sqrt(sum((v - my) ** 2 for v in y))
    return 0.0 if not sx or not sy else sum((p - mx) * (q - my) for p, q in zip(x, y, strict=True)) / (sx * sy)
