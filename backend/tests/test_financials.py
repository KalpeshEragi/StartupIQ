"""P&L, cash, runway and unit economics, checked against numbers worked out by hand."""
import pytest

from app.categorisation import model as model_module
from app.core.config import get_settings
from tests.conftest import STARTUP, signup

STATEMENT = """Date,Narration,Withdrawal Amt.,Deposit Amt.,Closing Balance
01/09/26,RAZORPAY SETTLEMENT 1111,,"1,00,000.00","6,00,000.00"
05/09/26,NEFT Dr-704015962232-GST PAYMENT-GSTIN 29ABCDE1234F1Z5,"9,000.00",,"5,91,000.00"
10/09/26,NEFT DR-HDFC0001234-SURESH PACKAGING SUPPLIES,"30,000.00",,"5,61,000.00"
28/09/26,NEFT DR-HDFC0001234-PRIYA SHARMA,"50,000.00",,"5,11,000.00"
30/09/26,Service Charges-SMS ALERT CHGS,59.00,,"5,10,941.00"
05/10/26,NEFT DR-HDFC0001234-SURESH PACKAGING SUPPLIES,"40,000.00",,"4,70,941.00"
28/10/26,NEFT DR-HDFC0001234-PRIYA SHARMA,"50,000.00",,"4,20,941.00"
"""


@pytest.fixture
def startup(client, monkeypatch, tmp_path):
    monkeypatch.setattr(get_settings(), "categoriser_path", str(tmp_path / "none.joblib"))  # rules only, no model
    model_module.load_model.cache_clear()
    h = signup(client)
    sid = client.post("/api/v1/startups", json=STARTUP, headers=h).json()["id"]  # a physical-product startup
    acc = client.post(f"/api/v1/startups/{sid}/bank-accounts", json={"name": "HDFC"}, headers=h).json()["id"]
    r = client.post(f"/api/v1/startups/{sid}/bank-accounts/{acc}/imports", headers=h,
                    files={"file": ("s.csv", STATEMENT.encode())})
    assert r.status_code == 201, r.text
    yield client, h, sid
    model_module.load_model.cache_clear()


def months(client, h, sid):
    return {m["month"]: m for m in client.get(f"/api/v1/startups/{sid}/financials/pnl", headers=h).json()["months"]}


def test_pnl_from_transactions(startup):
    client, h, sid = startup
    sep = months(client, h, sid)["2026-09"]
    assert sep["revenue"] == 100000 and sep["cogs"] == 0
    # supplier not categorised yet (no model), salary counted as people costs, bank charge; GST is not an expense
    assert {o["key"]: o["amount"] for o in sep["opex"]} == {"people": 50000, "uncategorised": 30000, "bank_charges": 59}
    assert sep["ebitda"] == 100000 - 80059
    assert sep["taxes_paid"] == 9000 and sep["closing_cash"] == 510941
    assert sep["cash_in"] == 100000 and sep["cash_out"] == 89059 and sep["burn"] == -10941


def test_marking_a_supplier_as_cost_of_goods_moves_it_to_gross_profit(startup):
    client, h, sid = startup
    items = client.get(f"/api/v1/startups/{sid}/transactions", params={"q": "packaging"}, headers=h).json()["items"]
    r = client.patch(f"/api/v1/startups/{sid}/transactions/{items[0]['id']}", json={"category": "cogs"}, headers=h)
    assert r.status_code == 200 and r.json()["also_updated"] == 1  # the October line follows via vendor memory
    m = months(client, h, sid)
    assert m["2026-09"]["cogs"] == 30000 and m["2026-09"]["gross_profit"] == 70000 and m["2026-09"]["gross_margin"] == 0.7
    assert m["2026-10"]["cogs"] == 40000


def test_runway_uses_the_average_burn_of_recent_months(startup):
    client, h, sid = startup
    r = client.get(f"/api/v1/startups/{sid}/financials/pnl", headers=h).json()["runway"]
    # September made cash (burn -10,941), October spent 90,000: average burn 39,529.5 on 4,20,941 cash
    assert r["status"] == "burning" and r["cash"] == 420941 and r["avg_burn"] == 39529.5
    assert r["months"] == round(420941 / 39529.5, 1)


def test_dashboard_and_unit_metrics(startup):
    client, h, sid = startup
    d = client.get(f"/api/v1/startups/{sid}/financials/dashboard", headers=h).json()
    assert d["has_data"] and d["month"] == "2026-10" and d["kpis"]["revenue"] == 0 and d["needs_review"] >= 2
    assert [s["month"] for s in d["series"]] == ["2026-09", "2026-10"] and not d["units_entered"]
    r = client.put(f"/api/v1/startups/{sid}/unit-metrics/2026-09", json={"units_sold": 500, "orders": 200}, headers=h)
    assert r.status_code == 200
    sep = next(u for u in client.get(f"/api/v1/startups/{sid}/unit-metrics", headers=h).json() if u["month"] == "2026-09")
    assert sep["computed"]["average_order_value"] == 500 and sep["computed"]["revenue_per_unit"] == 200
    assert client.put(f"/api/v1/startups/{sid}/unit-metrics/2999-01", json={}, headers=h).status_code == 422


def test_opening_balance_when_the_bank_gives_no_balance_column(client, monkeypatch, tmp_path):
    monkeypatch.setattr(get_settings(), "categoriser_path", str(tmp_path / "none.joblib"))
    model_module.load_model.cache_clear()
    h = signup(client)
    sid = client.post("/api/v1/startups", json=STARTUP, headers=h).json()["id"]
    acc = client.post(f"/api/v1/startups/{sid}/bank-accounts", json={"name": "SBI"}, headers=h).json()["id"]
    no_balance = "Date,Narration,Withdrawal Amt.,Deposit Amt.\n01/09/26,RAZORPAY SETTLEMENT 1,,1000\n02/09/26,SMS CHARGES,100,\n"
    client.post(f"/api/v1/startups/{sid}/bank-accounts/{acc}/imports", headers=h, files={"file": ("s.csv", no_balance.encode())})
    assert months(client, h, sid)["2026-09"]["closing_cash"] is None
    client.patch(f"/api/v1/startups/{sid}/bank-accounts/{acc}", json={"opening_balance": 5000}, headers=h)
    assert months(client, h, sid)["2026-09"]["closing_cash"] == 5900
    model_module.load_model.cache_clear()
