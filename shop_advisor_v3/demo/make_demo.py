"""Build the demo pack: python demo/make_demo.py

1. Writes one SYNTHETIC sales file per business situation into demo/files/.
2. Runs every file through the real pipeline (the same code the website calls), with the
   business details listed for it.
3. Writes demo/DEMO_GUIDE.md from what the system actually said: the story in the data, what to
   type on the website, the result, and why (facts and rule IDs). Nothing in the guide is typed
   by hand, so the demo matches the website exactly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from startupiq.contracts import BusinessContext  # noqa: E402
from startupiq.intake import read_file  # noqa: E402
from startupiq.pipeline import analyze_intake  # noqa: E402
from startupiq.synthetic import CATEGORIES  # noqa: E402

OUT = ROOT / "demo" / "files"
START, END = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-10-19")   # ends on a Sunday: the last week is complete
DAYS = pd.date_range(START, END, freq="D")
PRODUCTS = ["Phone cases", "Earbuds", "Chargers", "Screen guards", "Cables"]
# The kind of shop each scenario file is made as (synthetic.CATEGORIES); the classifier has to recover it.
TYPES = {"01_healthy_growing_shop": "Electronics & mobiles", "02_stockout_before_festival": "Electronics & mobiles",
         "03_growing_but_cash_crunch": "Restaurant & cafe", "04_slow_season_ahead": "Stationery & books", "05_losing_ground": "Clothing & fashion",
         "06_price_war_thin_margin": "Grocery & kirana", "07_one_product_dependence": "Electronics & mobiles",
         "08_overstocked_falling_demand": "Clothing & fashion", "09_very_volatile_sales": "Restaurant & cafe",
         "10_new_shop_ten_weeks": "Pharmacy", "11_no_finances_given": "Grocery & kirana", "12_margin_below_floor": "Pharmacy"}
SALES_ONLY = {"09_very_volatile_sales", "11_no_finances_given"}  # registers with only date and sales


def rng(seed):
    return np.random.default_rng(seed)


def bump(doy, centre, width, height):
    return height * np.exp(-((doy - centre) ** 2) / (2 * width ** 2))


def daily(base=6000.0, growth_per_year=0.0, season=None, noise=0.08, seed=0, days=DAYS, weekend=1.2):
    """Sales per day = base x trend x season(day of year) x weekend lift x noise."""
    r = rng(seed)
    t = np.arange(len(days)) / 365
    doy = days.dayofyear.to_numpy()
    s = np.ones(len(days)) if season is None else season(doy)
    wk = np.where(days.dayofweek >= 5, weekend, 1.0)
    return base * (1 + growth_per_year) ** t * s * wk * r.lognormal(0, noise, len(days))


def save_daily(name, values, days=DAYS, products=None, seed=0, indian=True):
    """The shop's type (TYPES) sets its weekday pattern in place of the generic weekend lift, and, unless the shop is
    in SALES_ONLY, adds the bill count, items sold and returns that a register of that type would hold."""
    r = rng(seed + 100)
    values = np.clip(values, 0, None)
    category = TYPES.get(name)
    df = pd.DataFrame({"Date": days.strftime("%d/%m/%Y") if indian else days.strftime("%Y-%m-%d")})
    if products is not None:
        names, probs = products
        df["Item"] = r.choice(names, len(days), p=probs)
    if category is not None:
        c = CATEGORIES[category]
        wd = np.array(c["weekday"]) / np.mean(c["weekday"])
        values = values / np.where(days.dayofweek >= 5, 1.2, 1.0) * wd[days.dayofweek]
    df["Sales (₹)"] = ["₹" + f"{v:,.0f}" for v in values]
    if category is not None and name not in SALES_ONLY:
        price = c["item_price"] * r.lognormal(0, 0.15)
        per_bill = max(1.0, c["items_per_bill"] * r.lognormal(0, 0.1))
        bills = np.maximum(1, np.round(values / (price * per_bill) * r.lognormal(0, 0.08, len(days))))
        df["No. of bills"] = bills.astype(int)
        df["Items sold"] = np.maximum(bills, np.round(bills * per_bill * r.lognormal(0, 0.05, len(days)))).astype(int)
        df["Returns (₹)"] = np.round(values * c["return_rate"] * r.lognormal(0, 0.6, len(days)) * (r.random(len(days)) < 0.7)).astype(int)
    path = OUT / f"{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def festive(doy):  # Diwali-style peak in late October / early November, quieter in early spring
    return 1 + bump(doy, 305, 10, 1.2) + bump(doy, 15, 12, 0.3) - bump(doy, 75, 25, 0.15)


def back_to_school(doy):  # busy June-September, slow from October to January
    return 1 + bump(doy, 200, 45, 0.9) - bump(doy, 345, 30, 0.35)


def build() -> list[dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    s = []

    def add(**kw):
        kw["category"] = TYPES.get(Path(kw["file"]).stem)
        s.append(kw)

    # ---------- business situations ----------
    v = daily(5000, growth_per_year=0.45, season=festive, noise=0.05, seed=1)
    add(file=save_daily("01_healthy_growing_shop", v, seed=1), title="Healthy, growing shop heading into its busy season",
        story="Sales grew about 45% a year, swings are small, and the festive peak is coming. Cash and stock are comfortable.",
        ctx=dict(cash_on_hand=600000, monthly_fixed_costs=50000, gross_margin=0.40, stock_value=150000, supplier_lead_time_days=7, competitors=["Mobile Mart"]),
        expect=["diagnose:healthy_growth", "recommend:channel_expansion"], say=["forecast_growth", "market_pulse", "runway_months", "stock_cover_weeks"])

    v = daily(6000, growth_per_year=0.10, season=festive, noise=0.06, seed=2)
    add(file=save_daily("02_stockout_before_festival", v, seed=2), title="Stock runs out just before the festival",
        story="Same kind of shop, but less than a week of stock on hand, the supplier needs three weeks, and the festive peak is two weeks away.",
        ctx=dict(cash_on_hand=300000, monthly_fixed_costs=60000, gross_margin=0.35, stock_value=20000, supplier_lead_time_days=21, competitors=["Gadget Zone"]),
        expect=["warn:stockout", "warn:peak_ahead", "recommend:restock", "risk:sellout_in_peak"], say=["stock_cover_weeks", "supplier_lead_time_weeks", "weeks_to_peak"])

    days = DAYS
    ramp = np.where(days >= END - pd.Timedelta(days=55), 1 + (days - (END - pd.Timedelta(days=55))).days / 55 * 0.6, 1.0)
    v = daily(7000, noise=0.05, seed=3) * ramp
    add(file=save_daily("03_growing_but_cash_crunch", v, seed=3), title="Growing fast but running out of cash",
        story="Sales jumped about 50% in the last two months, but fixed costs are high and cash covers less than a month.",
        ctx=dict(cash_on_hand=40000, monthly_fixed_costs=150000, gross_margin=0.30, stock_value=60000, supplier_lead_time_days=7),
        expect=["warn:runway_critical", "diagnose:growing_but_cash_starved", "recommend:cost_reduction", "forbid:new_product", "risk:cash_out"], say=["growth_recent", "runway_months"])

    v = daily(5000, season=back_to_school, noise=0.06, seed=4)
    add(file=save_daily("04_slow_season_ahead", v, seed=4), title="Slow season ahead with tight cash",
        story="A school-supplies seller: busy June to September, and last year the weeks after mid-October were much quieter. Cash covers about four months.",
        ctx=dict(cash_on_hand=140000, monthly_fixed_costs=80000, gross_margin=0.30, stock_value=150000, supplier_lead_time_days=10),
        expect=["warn:market_falling", "recommend:cost_reduction", "risk:slow_season"], say=["market_pulse", "runway_months", "forecast_growth"])

    decline = np.where(days >= END - pd.Timedelta(days=120), 1 - (days - (END - pd.Timedelta(days=120))).days / 120 * 0.55, 1.0)
    v = daily(6000, season=lambda d: 1 + bump(d, 345, 45, 0.5), noise=0.05, seed=5) * decline
    add(file=save_daily("05_losing_ground", v, seed=5), title="Losing ground while the market rises",
        story="Last year sales rose steadily from October into December, but this shop's sales have slid about 55% over the last four months.",
        ctx=dict(cash_on_hand=400000, monthly_fixed_costs=40000, gross_margin=0.40, stock_value=80000, supplier_lead_time_days=7),
        expect=["diagnose:losing_ground", "recommend:promotion"], say=["forecast_growth", "market_pulse"])

    v = daily(6000, noise=0.06, seed=6)
    add(file=save_daily("06_price_war_thin_margin", v, seed=6), title="A rival cut prices and our margin is thin",
        story="Stable sales, a 20% margin, and the founder reports that a competitor has just cut prices.",
        ctx=dict(cash_on_hand=300000, monthly_fixed_costs=30000, gross_margin=0.20, stock_value=40000, supplier_lead_time_days=7,
                 competitors=["Discount Dukaan", "Mobile Mart"], rival_price_cut_seen=True),
        expect=["forbid:price_cut", "recommend:loyalty_offer"], say=["gross_margin"])

    v = daily(6000, growth_per_year=0.05, noise=0.06, seed=7)
    add(file=save_daily("07_one_product_dependence", v, products=(PRODUCTS, [0.8, 0.08, 0.06, 0.04, 0.02]), seed=7),
        title="One product carries the whole business", story="80% of sales come from phone cases; cash is comfortable.",
        ctx=dict(cash_on_hand=800000, monthly_fixed_costs=40000, gross_margin=0.40, stock_value=60000, supplier_lead_time_days=7),
        expect=["warn:dependence", "recommend:new_product", "risk:main_product"], say=["top_product_share", "runway_months"])

    decline = np.where(days >= END - pd.Timedelta(days=90), 1 - (days - (END - pd.Timedelta(days=90))).days / 90 * 0.5, 1.0)
    v = daily(6000, noise=0.05, seed=8) * decline
    add(file=save_daily("08_overstocked_falling_demand", v, seed=8), title="Too much stock while demand falls",
        story="Sales have been sliding for three months, yet the shop holds about half a year of stock.",
        ctx=dict(cash_on_hand=200000, monthly_fixed_costs=30000, gross_margin=0.35, stock_value=600000, supplier_lead_time_days=7),
        expect=["warn:overstock", "warn:forecast_drop"], say=["stock_cover_weeks", "forecast_growth"])

    r = rng(9)
    week_factor = r.lognormal(0, 0.9, len(DAYS) // 7 + 1)[np.arange(len(DAYS)) // 7]  # whole weeks swing, not just days
    v = daily(6000, noise=0.3, seed=9) * week_factor
    add(file=save_daily("09_very_volatile_sales", v, seed=9), title="Very unpredictable sales",
        story="Some days are huge, some are almost empty: week-to-week sales swing wildly.",
        ctx=dict(cash_on_hand=500000, monthly_fixed_costs=30000, gross_margin=0.40, stock_value=200000, supplier_lead_time_days=7),
        expect=["warn:volatile", "forbid:channel_expansion"], say=["sales_volatility", "band_width"])

    short = pd.date_range("2025-08-11", END, freq="D")
    v = daily(4000, noise=0.08, seed=10, days=short)
    add(file=save_daily("10_new_shop_ten_weeks", v, days=short, seed=10), title="A new shop with only ten weeks of sales",
        story="Opened in August. Ten weeks is too little to see a season.",
        ctx=dict(cash_on_hand=200000, monthly_fixed_costs=40000, gross_margin=0.35, stock_value=30000, supplier_lead_time_days=7),
        expect=["warn:short_history", "forbid:channel_expansion"], say=["history_weeks"])

    v = daily(6000, season=festive, noise=0.06, seed=11) * np.where(days >= END - pd.Timedelta(days=100), 0.7, 1.0)
    add(file=save_daily("11_no_finances_given", v, seed=11), title="Founder gives no money figures",
        story="Sales dropped recently, but the founder leaves cash, costs and margin empty.",
        ctx=dict(), expect=["warn:need_finances", "warn:no_cash_inputs"], say=["forecast_growth"])

    v = daily(6000, noise=0.06, seed=12)
    add(file=save_daily("12_margin_below_floor", v, seed=12), title="Margin already below a safe floor",
        story="The shop keeps only 8% of each sale after the cost of goods (the floor is 10%).",
        ctx=dict(cash_on_hand=300000, monthly_fixed_costs=10000, gross_margin=0.08, stock_value=30000, supplier_lead_time_days=7),
        expect=["warn:thin_margin", "forbid:price_cut"], say=["gross_margin"])

    # ---------- reading messy files (intake) ----------
    pack = ROOT / "tests" / "fixtures" / "intake" / "files"
    for src, dst, title, story, expect in [
        ("04_title_rows.xlsx", "13_tea_stall_title_rows.xlsx", "Messy Excel register with title rows",
         "Shop name and date printed above the table; quantity and rate but no total column.", ["intake:skipped_title_rows", "intake:qty_x_price"]),
        ("16_hindi_headers.csv", "14_hindi_headers.csv", "Hindi column names", "Columns are तारीख (date) and बिक्री (sales).", ["intake:hindi"]),
        ("05_wide_months.xlsx", "15_one_column_per_month.xlsx", "One column per month",
         "Products down the side, months across the top, and a Total row.", ["intake:wide"]),
        ("13_us_dates_semicolon.csv", "16_us_dates_semicolons.csv", "US dates separated by semicolons",
         "Exported from a US tool: month/day/year and ';' between columns.", ["intake:us_dates"]),
        ("21_ambiguous_dates.csv", "17_ambiguous_dates.csv", "Dates that could be read two ways",
         "Every date fits both 3 April and 4 March.", ["intake:asks_date_order"]),
        ("18_one_huge_order.csv", "18_one_huge_bulk_order.csv", "One huge one-off order", "A single bulk order 60 times a normal day.", ["intake:asks_outlier"]),
        ("17_gap_in_middle.csv", "19_closed_for_six_weeks.csv", "Six weeks with no sales", "Shop closed (or nobody recorded sales) for six weeks.", ["intake:asks_gap"]),
        # ---------- files it must refuse ----------
        ("10_no_date.csv", "20_refused_no_dates.csv", "Refused: no dates", "A product list with quantities but no dates.", ["refused"]),
        ("09_all_zeros.csv", "21_refused_all_zeros.csv", "Refused: all zeros", "Every sale is zero.", ["refused"]),
        ("20_ten_rows.csv", "22_refused_ten_rows.csv", "Refused: too little data", "Only ten days of sales.", ["refused"]),
        ("11_notes_not_data.csv", "23_refused_notes.csv", "Refused: not a sales file", "A to-do list saved as CSV.", ["refused"]),
    ]:
        (OUT / dst).write_bytes((pack / src).read_bytes())
        add(file=OUT / dst, title=title, story=story, ctx=dict(cash_on_hand=200000, monthly_fixed_costs=40000, gross_margin=0.35), expect=expect)
    return s


def outcome(spec) -> dict:
    got = read_file(spec["file"])
    rep = analyze_intake(got, BusinessContext(**spec["ctx"]))
    cards = {c.key: c for c in rep.cards}
    found = set()
    if rep.tier.value == "refused":
        found.add("refused")
    w = cards.get("warnings")
    if w:
        for kind, key in (("warn", "warnings"), ("diagnose", "diagnoses"), ("risk", "premortem")):
            found |= {f"{kind}:{k}" for k in (w.details.get(key) or {})}
        found |= {f"warn:{k}" for k in (w.details.get("add_to_unlock") or {})}
    nm = cards.get("next_move")
    if nm:
        labels = {r.action: r for r in rep.recommendations}
        found |= {f"recommend:{f.name.split(':', 1)[1]}" for f in rep.facts if f.name.startswith("recommend:") and f.certainty >= 0.2
                  and f.name.split(":", 1)[1] not in nm.details.get("blocked", {})}
        found |= {f"forbid:{f.name.split(':', 1)[1]}" for f in rep.facts if f.name.startswith("forbid:")}
    log = " ".join(got.cleaning_log)
    if "title row" in log:
        found.add("intake:skipped_title_rows")
    if "quantity x price" in log:
        found.add("intake:qty_x_price")
    if any(0x900 <= ord(ch) <= 0x97F for ch in "".join(got.column_map.values())):  # Devanagari (not the ₹ sign)
        found.add("intake:hindi")
    if got.layout == "wide":
        found.add("intake:wide")
    if got.column_map and "';'" in log:
        found.add("intake:us_dates")
    for q, tag in (("date_order", "intake:asks_date_order"), ("outliers", "intake:asks_outlier"), ("gaps", "intake:asks_gap")):
        if q in got.needs_confirmation:
            found.add(tag)
    fired = [t.split(" ")[1] for t in rep.trace if t.startswith("[")]
    facts = {f.name: f.value for f in rep.facts if ":" not in f.name}
    return {"report": rep, "intake": got, "cards": cards, "found": found, "fired": fired, "facts": facts}


KEY_FACTS = ["history_weeks", "growth_recent", "forecast_growth", "runway_months", "stock_cover_weeks", "supplier_lead_time_weeks",
             "weeks_to_peak", "market_pulse", "sales_volatility", "top_product_share", "gross_margin", "band_width"]


PERCENT = {"growth_recent", "forecast_growth", "sales_volatility", "top_product_share", "gross_margin", "band_width"}
LABEL = {"history_weeks": "weeks of history", "growth_recent": "last 4 weeks vs 4 before", "forecast_growth": "forecast next 12 weeks vs last 12",
         "runway_months": "cash runway (months)", "stock_cover_weeks": "stock lasts (weeks)", "supplier_lead_time_weeks": "supplier needs (weeks)",
         "weeks_to_peak": "busiest week in (weeks)", "market_pulse": "same weeks last year", "sales_volatility": "week-to-week swing",
         "top_product_share": "top product's share", "gross_margin": "margin", "band_width": "forecast range width"}


def fmt(name, v):
    if isinstance(v, float):
        if name == "runway_months" and v >= 999:
            return "profitable"
        return f"{v:+.0%}" if name in ("growth_recent", "forecast_growth") else (f"{v:.0%}" if name in PERCENT else f"{v:,.1f}".rstrip("0").rstrip("."))
    return str(v)


def guide(specs) -> str:
    rows = ["# Demo guide", "", "Generated by `python demo/make_demo.py` from what the real system said. All files are SYNTHETIC.", "",
            "Start the website (`uvicorn startupiq.api.app:app`), upload a file from `demo/files/`, type the details listed, press Analyse.",
            "For each scenario: **what is in the data** -> **what the system measured or predicted** -> **what it concluded, and which rule said so**.",
            "The \"cheapest plan\" line answers a different question from the recommendations: the cheapest set of moves that raises 12-week profit by 3% of sales (A* search).", "",
            "| # | File | Situation | Matches the intended outcome |", "| --- | --- | --- | --- |"]
    sections = []
    for i, sp in enumerate(specs, 1):
        o = sp["outcome"]
        ok = [e for e in sp["expect"] if e in o["found"]]
        rows.append(f"| {i} | `{Path(sp['file']).name}` | {sp['title']} | {len(ok)}/{len(sp['expect'])} |")
        rep, cards = o["report"], o["cards"]
        typed = ", ".join(f"{k.replace('_', ' ')}: {', '.join(v) if isinstance(v, list) else ('Yes' if v is True else (f'{v:.0%}' if k == 'gross_margin' else f'{v:,}'))}"
                          for k, v in sp["ctx"].items()) or "nothing (leave every box empty)"
        typed = typed.replace("gross margin:", "margin %:").replace("rival price cut seen:", "rival cut prices:").replace("supplier lead time days", "lead time (days)")
        sec = [f"## {i}. {sp['title']}", "", f"**File:** `demo/files/{Path(sp['file']).name}`  ", f"**Type on the website:** {typed}", "",
               f"**What is in the data:** {sp['story']}", ""]
        if rep.tier.value == "refused":
            sec += [f"**What the system says:** {cards['intake'].headline}", "",
                    "**Why:** intake checks the file before any model runs and refuses rather than inventing a forecast (decision 0004, 0011).", ""]
        else:
            ic = cards["intake"].details
            if "category" in cards and sp["category"]:
                k = cards["category"].details
                sec += [f"**Business type:** made as {sp['category']}; the classifier says {k['best_guess']} ({k['confidence']:.0%}) "
                        f"from {', '.join(['sales'] + k['inputs'])}.", ""]
            sec += [f"**What the system read:** {cards['intake'].headline}; columns {', '.join(f'{k} = {v}' for k, v in ic['column_map'].items())}; data quality: {ic['tier']}."]
            if ic["messages"]:
                sec += ["", "**It asks the founder:**", *[f"- {m}" for m in ic["messages"]]]
            notes = [n for n in ic["cleaning_log"] if any(x in n for x in ("title", "quantity x price", "reshaped", "Total", "';'", "duplicate", "netted"))]
            if notes:
                sec += ["", "**What it fixed by itself:**", *[f"- {n}" for n in notes]]
            facts = o["facts"]
            shown = [f"{LABEL[k]} {fmt(k, facts[k])}" for k in KEY_FACTS if k in facts]
            sec += ["", f"**What it measured / predicted:** {'; '.join(shown)}."]
            if "forecast" in cards and not cards["forecast"].locked_reason:
                sec += [f"**Forecast:** {cards['forecast'].headline} (method: {cards['forecast'].details['model']})."]
            w = cards["warnings"].details
            if w["warnings"]:
                sec += ["", "**Warnings:**", *[f"- {t}" for t in w["warnings"].values()]]
            if w["diagnoses"]:
                sec += ["", "**Diagnosis:**", *[f"- {t}" for t in w["diagnoses"].values()]]
            if w["premortem"]:
                sec += ["", "**How this could fail in 90 days:**", *[f"- {t}" for t in w["premortem"].values()]]
            nm = cards["next_move"].details
            sec += ["", f"**What to do next:** {cards['next_move'].headline}"]
            sec += [f"- **{r['action']}**: {r['reason']} (rules {', '.join(r['rules_fired'])})" for r in nm["recommendations"]]
            if nm.get("blocked"):
                sec += [f"- Not suggested: {k} ({v})" for k, v in nm["blocked"].items()]
            if nm.get("plan") and nm["plan"]["found"]:
                sec += [f"- Cheapest plan to beat the 12-week target (A* search): {'; '.join(nm['plan']['steps'])}"]
            forb = [f.name.split(":", 1)[1] for f in rep.facts if f.name.startswith("forbid:") and f.name != "forbid:price_coordination"]
            if forb:
                sec += [f"- Ruled out by the rules: {', '.join(forb)}"]
            sec += ["", f"**Why (rules that fired, in order):** {', '.join(o['fired'])}. Open \"Full reasoning trace\" on the website to read each one."]
            say_facts = [f"{LABEL[k]} {fmt(k, facts[k])}" for k in sp.get("say", []) if k in facts]
            main = [t.split(" ")[1] for t in rep.trace if t.startswith("[") and any(x in t for x in ("warn:", "recommend:", "forbid:", "diagnose:")) and "price_coordination" not in t]
            sec += ["", f"> **Say in the demo:** \"The data shows: {sp['story'][0].lower() + sp['story'][1:].rstrip('.')}. The system measured "
                        f"{', '.join(say_facts) or 'the numbers above'}. That made rules {', '.join(dict.fromkeys(main[:4]))} fire, "
                        f"so it says: {cards['next_move'].headline.lower()}"
                        + (f", and warns: {next(iter(w['warnings'].values())).rstrip('.')}" if w["warnings"] else "") + ".\""]
        missing = [e for e in sp["expect"] if e not in o["found"]]
        if missing:
            sec += ["", f"_Intended but not produced: {', '.join(missing)}_"]
        sections += sec + [""]
    return "\n".join(rows + [""] + sections)


def main():
    specs = build()
    for sp in specs:
        sp["outcome"] = outcome(sp)
        miss = [e for e in sp["expect"] if e not in sp["outcome"]["found"]]
        k = sp["outcome"]["cards"].get("category")
        kind = ""
        if k and sp["category"]:
            kind = f"  type {k.details['best_guess']} ({k.details['confidence']:.0%})" + ("" if k.details["best_guess"] == sp["category"] else f"  WRONG, made as {sp['category']}")
        print(f"{Path(sp['file']).name:40s} {'OK ' if not miss else 'MISSING ' + str(miss)}{kind}")
    (ROOT / "demo" / "DEMO_GUIDE.md").write_text(guide(specs), encoding="utf-8")
    inputs = {Path(sp["file"]).name: sp["ctx"] for sp in specs}
    (ROOT / "demo" / "inputs.json").write_text(json.dumps(inputs, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
