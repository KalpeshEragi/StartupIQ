"""Data contracts shared by every module.

Each module reads and writes these types only, so a stub can be swapped for a
real implementation without touching its neighbours. See devdocs/architecture.md.
"""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class Tier(str, Enum):
    """Data-quality tier decided by intake. Higher tiers unlock more cards."""

    REFUSED = "refused"
    SUMMARY_ONLY = "summary_only"  # under 8 weeks of history
    BASIC = "basic"  # 2 to 6 months
    STANDARD = "standard"  # 6 to 12 months
    FULL = "full"  # 12+ months


class Source(str, Enum):
    """How a number was produced. Shown next to every output in the UI."""

    MEASURED = "measured"  # computed directly from the user's data
    PREDICTED = "predicted"  # model output with a validated error
    INFERRED = "inferred"  # estimate from proxies; no ground truth for this user
    STUB = "stub"  # placeholder while a module is not built yet


class BusinessContext(BaseModel):
    """Facts the founder types in next to the upload. All optional."""

    category: str | None = None
    competitors: list[str] = Field(default_factory=list)
    cash_on_hand: float | None = None
    monthly_fixed_costs: float | None = None
    supplier_lead_time_days: int | None = None
    gross_margin: float | None = Field(default=None, ge=0, le=1)
    min_margin: float = Field(default=0.10, ge=0, le=1)  # the founder's floor for pricing moves
    stock_value: float | None = None  # inventory on hand, valued at cost
    market_size_monthly: float | None = None  # the category's monthly sales, if the founder knows it
    rival_price_cut_seen: bool | None = None  # founder reports a rival cut prices recently


class SalesSeries(BaseModel):
    """Cleaned, regular series produced by intake. Values are per period."""

    frequency: str  # "D", "W" or "M"
    dates: list[date]
    values: list[float]
    product: str | None = None


class IntakeResult(BaseModel):
    tier: Tier
    series: list[SalesSeries] = Field(default_factory=list)  # series[0] is the business total; then top products
    column_map: dict[str, str] = Field(default_factory=dict)  # role -> original column name
    confidence: dict[str, float] = Field(default_factory=dict)  # role -> 0..1 detection confidence
    needs_confirmation: list[str] = Field(default_factory=list)  # roles the user should confirm in the UI
    layout: str = "long"  # "long" (one row per sale/period) or "wide" (one column per period)
    granularity: str | None = None  # "transactions", "D", "W" or "M" as uploaded
    sheet: str | None = None
    header_row: int | None = None
    cleaning_log: list[str] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)  # shown to the user
    signals: dict[str, list] | None = None  # per date: sales and, if uploaded, orders / items / returns (intake/signals.py)


class Forecast(BaseModel):
    dates: list[date]
    p10: list[float]
    p50: list[float]
    p90: list[float]
    model: str
    source: Source = Source.PREDICTED
    backtest: dict[str, float] = Field(default_factory=dict)  # method -> error (% of sales) on the series' own last 12 periods
    chosen: str = ""  # the method used for p50


class Fact(BaseModel):
    """A statement the rule engine can reason over. ML outputs carry their confidence."""

    name: str
    value: float | str | bool
    certainty: float = Field(default=1.0, ge=0, le=1)
    source: Source = Source.MEASURED
    evidence: str = ""


class Recommendation(BaseModel):
    action: str
    reason: str
    rules_fired: list[str] = Field(default_factory=list)
    expected_effect: str = ""


class Card(BaseModel):
    """One block of the founder's briefing."""

    key: str
    title: str
    headline: str
    source: Source
    details: dict = Field(default_factory=dict)
    locked_reason: str | None = None  # set when the data tier does not allow this card


class Report(BaseModel):
    tier: Tier
    cards: list[Card]
    facts: list[Fact] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)  # step-by-step log of what ran
