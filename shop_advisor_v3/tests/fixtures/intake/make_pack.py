"""Generate the broken-file pack: python tests/fixtures/intake/make_pack.py

Each file imitates a real way small businesses keep sales. expected.yaml says what
intake must do with it (tier, detected roles, layout, questions for the user).
Fixed seed, so the files are identical on every run. All data is SYNTHETIC.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "files"
rng = np.random.default_rng(7)
PRODUCTS = ["Masala tea", "Samosa", "Cold coffee", "Vada pav", "Biscuits"]


def daily(days: int, start: str = "2024-01-01") -> pd.DataFrame:
    d = pd.date_range(start, periods=days, freq="D")
    season = 1 + 0.3 * np.sin(2 * np.pi * d.dayofyear / 365)
    return pd.DataFrame({"date": d, "amount": (rng.normal(4000, 600, days) * season).round(0)})


def transactions(days: int, start: str = "2024-01-01") -> pd.DataFrame:
    rows = []
    for d in pd.date_range(start, periods=days, freq="D"):
        for _ in range(rng.integers(3, 9)):
            p = rng.choice(PRODUCTS)
            q = int(rng.integers(1, 6))
            price = {"Masala tea": 20, "Samosa": 15, "Cold coffee": 60, "Vada pav": 25, "Biscuits": 10}[p]
            rows.append({"date": d, "product": p, "qty": q, "price": price})
    return pd.DataFrame(rows)


def write_csv(name: str, df: pd.DataFrame, **kw) -> None:
    df.to_csv(OUT / name, index=False, **kw)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    d = daily(420)
    write_csv("01_clean_daily.csv", d.assign(date=d.date.dt.strftime("%Y-%m-%d")).rename(columns={"amount": "Sales"}))

    d = daily(430)
    write_csv("02_indian_rupees.csv", pd.DataFrame({
        "Bill Date": d.date.dt.strftime("%d/%m/%Y"),
        "Amount (₹)": ["₹" + f"{v:,.2f}" for v in d.amount * 30],  # e.g. ₹1,20,000.00 style values
        "Bill No": range(1001, 1001 + len(d))}))

    d = daily(300)
    write_csv("03_no_header.csv", pd.DataFrame({0: d.date.dt.strftime("%d-%m-%Y"), 1: d.amount, 2: rng.integers(5, 30, len(d))}), header=False)

    t = transactions(380)
    with pd.ExcelWriter(OUT / "04_title_rows.xlsx") as w:
        pd.DataFrame([["Sharma Tea Stall - Sales Register"], ["Generated 01/05/2025"], [None]]).to_excel(w, index=False, header=False)
        t.rename(columns={"date": "Date", "product": "Item", "qty": "Qty", "price": "Rate"}).to_excel(w, index=False, startrow=3)

    months = pd.period_range("2023-07", periods=18, freq="M")
    wide = pd.DataFrame({"Product": PRODUCTS})
    for m in months:
        wide[m.strftime("%b-%Y")] = rng.integers(8000, 20000, len(PRODUCTS))
    wide.loc[len(wide)] = ["Total"] + list(wide.iloc[:, 1:].sum())
    wide.to_excel(OUT / "05_wide_months.xlsx", index=False)

    with pd.ExcelWriter(OUT / "06_two_sheets.xlsx") as w:
        pd.DataFrame({"Metric": ["Total sales", "Best item"], "Value": ["₹12,34,567", "Samosa"]}).to_excel(w, sheet_name="Summary", index=False)
        tx = transactions(400)
        tx.assign(total=tx.qty * tx.price).to_excel(w, sheet_name="Transactions", index=False)

    write_csv("07_five_weeks.csv", daily(35).rename(columns={"amount": "revenue"}))
    write_csv("08_three_months.csv", daily(95).rename(columns={"amount": "revenue"}))
    write_csv("09_all_zeros.csv", daily(200).assign(amount=0).rename(columns={"amount": "sales"}))
    write_csv("10_no_date.csv", pd.DataFrame({"product": rng.choice(PRODUCTS, 100), "qty": rng.integers(1, 9, 100)}))
    (OUT / "11_notes_not_data.csv").write_text("Remember to call supplier\nPay rent on 5th\nDiwali stock order pending\n", encoding="utf-8")

    t = transactions(400)
    write_csv("12_qty_and_price.csv", t.rename(columns={"date": "Order Date", "product": "Item Name", "qty": "Quantity", "price": "Unit Price"}))

    d = daily(400, start="2024-02-01")
    us = d.date.dt.strftime("%m/%d/%Y")
    write_csv("13_us_dates_semicolon.csv", pd.DataFrame({"Date": us, "Revenue": d.amount}), sep=";")

    d = daily(400)
    serial = (d.date - pd.Timestamp("1899-12-30")).dt.days
    pd.DataFrame({"Date": serial, "Sales": d.amount}).to_excel(OUT / "14_excel_serial_dates.xlsx", index=False)

    d = daily(400)
    extra = d.sample(25, random_state=1)
    returns = d.sample(15, random_state=2).assign(amount=lambda x: -x.amount * 0.2)
    write_csv("15_returns_and_duplicates.csv", pd.concat([d, extra, returns]).sort_values("date").rename(columns={"amount": "net_sales"}))

    d = daily(400)
    write_csv("16_hindi_headers.csv", pd.DataFrame({"तारीख": d.date.dt.strftime("%d/%m/%Y"), "बिक्री": d.amount}))

    d = daily(420)
    d.loc[(d.date >= "2024-06-01") & (d.date < "2024-07-15"), "amount"] = np.nan
    write_csv("17_gap_in_middle.csv", d.dropna().rename(columns={"amount": "sales"}))

    d = daily(420)
    d.loc[200, "amount"] = d.amount.median() * 60
    write_csv("18_one_huge_order.csv", d.rename(columns={"amount": "sales"}))

    d = daily(400)
    (OUT / "19_latin1_pounds.csv").write_bytes(
        pd.DataFrame({"Date": d.date.dt.strftime("%d/%m/%Y"), "Takings": ["£" + f"{v:.2f}" for v in d.amount]})
        .to_csv(index=False).encode("cp1252"))

    write_csv("20_ten_rows.csv", daily(10).rename(columns={"amount": "sales"}))

    d = daily(10 * 31)
    write_csv("21_ambiguous_dates.csv", pd.DataFrame({"Date": d.date.dt.strftime("%d/%m/%Y"), "Sales": d.amount})
              .pipe(lambda x: x[d.date.dt.day <= 12]))

    d = pd.date_range("2024-01-01", periods=420, freq="D")
    write_csv("22_products_as_columns.csv", pd.DataFrame({"Date": d.strftime("%Y-%m-%d"),
              **{p: rng.integers(0, 40, len(d)) for p in PRODUCTS[:4]}}))

    m = pd.period_range("2023-01", periods=20, freq="M")
    write_csv("23_monthly_totals.csv", pd.DataFrame({"Month": m.strftime("%b %Y"), "Total Sales": rng.integers(90000, 160000, len(m))}))

    (OUT / "24_empty.csv").write_text("", encoding="utf-8")
    (OUT / "25_wrong_type.pdf").write_bytes(b"%PDF-1.4 not a spreadsheet")
    print(f"wrote {len(list(OUT.iterdir()))} files to {OUT}")


if __name__ == "__main__":
    main()
