"""The class frames of the Startup IQ domain and the moves catalogue (Task 1).

Hierarchy (is_a):
    Thing
      Agent -> Business -> SmallSeller, Competitor
      Product, Category, Metric, Constraint
      Move -> one class frame per entry in configs/moves.yaml
"""

from __future__ import annotations

from pathlib import Path

import yaml

from startupiq.knowledge.frames import Frame

MOVES_FILE = Path(__file__).resolve().parents[3] / "configs" / "moves.yaml"
WEEKS_PER_MONTH = 52 / 12


def _monthly_sales(f: Frame):
    w = f.get("weekly_sales")
    return None if w is None else w * WEEKS_PER_MONTH


def _runway(f: Frame):
    cash, fixed, margin, monthly = f.get("cash"), f.get("monthly_fixed_costs"), f.get("gross_margin"), f.get("monthly_sales")
    if None in (cash, fixed, margin, monthly):
        return None
    burn = fixed - monthly * margin
    return float("inf") if burn <= 0 else cash / burn


def _stock_cover(f: Frame):
    stock, weekly, margin = f.get("stock_value"), f.get("weekly_sales"), f.get("gross_margin")
    if None in (stock, weekly, margin) or weekly <= 0:
        return None
    return stock / (weekly * (1 - margin))  # stock is valued at cost; weekly sales at price


def build_classes(moves_file: Path = MOVES_FILE) -> dict[str, Frame]:
    c: dict[str, Frame] = {}
    c["Thing"] = Frame("Thing", is_class=True)
    c["Agent"] = Frame("Agent", c["Thing"], is_class=True)
    c["Business"] = (Frame("Business", c["Agent"], is_class=True)
                     .define("cash", range=(0, 1e12), doc="cash on hand")
                     .define("monthly_fixed_costs", range=(0, 1e12))
                     .define("gross_margin", default=0.30, range=(0.0, 1.0), doc="share of a sale left after the cost of goods")
                     .define("min_margin", default=0.10, range=(0.0, 1.0), doc="floor below which pricing moves are refused")
                     .define("weekly_sales", range=(0, 1e12))
                     .define("monthly_sales", if_needed=_monthly_sales)
                     .define("runway_months", if_needed=_runway)
                     .define("stock_value", range=(0, 1e12), doc="inventory valued at cost")
                     .define("stock_cover_weeks", if_needed=_stock_cover)
                     .define("supplier_lead_time_weeks", default=2, range=(0, 52))
                     .define("price_position", default="similar", range=("cheaper", "similar", "pricier")))
    c["SmallSeller"] = Frame("SmallSeller", c["Business"], is_class=True).define("channel", default="online").define("observed", default=True)
    c["Competitor"] = (Frame("Competitor", c["Business"], is_class=True)
                       .define("observed", default=False, doc="a founder cannot see a rival's books")
                       .define("source", default="inferred", range=("inferred", "measured"))
                       .define("p_exit", default=0.16).define("p_expansion", default=0.19).define("p_price_cut", default=0.02))
    c["Product"] = Frame("Product", c["Thing"], is_class=True).define("share_of_revenue", range=(0.0, 1.0)).define("weekly_sales", range=(0, 1e12))
    c["Category"] = (Frame("Category", c["Thing"], is_class=True).define("trend", default="unknown", range=("rising", "flat", "falling", "unknown"))
                     .define("sellers", range=(0, 1e7)).define("hhi", range=(0.0, 1.0)))
    c["Metric"] = Frame("Metric", c["Thing"], is_class=True)
    c["Constraint"] = Frame("Constraint", c["Thing"], is_class=True).define("statement")
    c["Move"] = (Frame("Move", c["Thing"], is_class=True).define("cost", default=0.0, range=(0.0, 10.0))
                 .define("lead_time_weeks", default=0).define("duration_weeks", default=1).define("demand_lift", default=1.0, range=(0.5, 3.0))
                 .define("margin_change", default=0.0).define("legal", default=True, range=(True, False)).define("requires", default=[])
                 .define("affects", default={}).define("label").define("source", default="assumption"))
    moves = yaml.safe_load(moves_file.read_text(encoding="utf-8"))
    for key, spec in moves.items():
        f = Frame(key, c["Move"], is_class=True)
        for slot, value in spec.items():
            f.set(slot, value)
        c[key] = f
    return c


def move_names(classes: dict[str, Frame]) -> list[str]:
    return [n for n, f in classes.items() if f.parent is not None and f.parent.name == "Move"]
