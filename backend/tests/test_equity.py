"""Cap table, vesting, SAFEs, the dilution simulator and the investor pipeline."""
from datetime import date

import pytest

from app.finance.equity import months_between, vested
from app.models import OptionGrant
from tests.conftest import STARTUP, signup


@pytest.fixture
def company(client):
    h = signup(client)
    sid = client.post("/api/v1/startups", json=STARTUP, headers=h).json()["id"]
    r = client.post(f"/api/v1/startups/{sid}/cap-table/setup", headers=h, json={
        "incorporated_on": "2025-01-01", "pool_shares": 1_000_000,
        "founders": [{"name": "Asha", "shares": 4_000_000}, {"name": "Ravi", "shares": 4_000_000}]})
    assert r.status_code == 201, r.text
    return client, h, sid


def url(sid, path):
    return f"/api/v1/startups/{sid}/{path}"


def test_setup_gives_fully_diluted_ownership(company):
    client, h, sid = company
    t = client.get(url(sid, "cap-table"), headers=h).json()
    assert t["total_fully_diluted"] == 9_000_000
    own = {r["name"]: r["ownership"] for r in t["holders"]}
    assert own == pytest.approx({"Asha": 4 / 9, "Ravi": 4 / 9, "ESOP pool": 1 / 9})
    assert client.post(url(sid, "cap-table/setup"), headers=h, json={
        "incorporated_on": "2025-01-01", "founders": [{"name": "X", "shares": 1}]}).status_code == 409


def test_vesting_with_a_cliff():
    g = OptionGrant(shares=48_000, granted_on=date(2025, 1, 15), vesting_months=48, cliff_months=12)
    assert vested(g, date(2025, 12, 31)) == 0            # before the cliff
    assert vested(g, date(2026, 1, 15)) == 12_000         # cliff: a year vests at once
    assert vested(g, date(2026, 7, 15)) == 18_000         # then monthly
    assert vested(g, date(2030, 1, 1)) == 48_000          # never more than granted
    assert months_between(date(2025, 1, 31), date(2025, 2, 28)) == 0


def test_grants_come_out_of_the_pool(company):
    client, h, sid = company
    r = client.post(url(sid, "option-grants"), headers=h,
                    json={"employee_name": "Neha", "shares": 100_000, "granted_on": "2025-06-01"})
    assert r.status_code == 201
    t = client.get(url(sid, "cap-table"), headers=h).json()
    fd = {x["name"]: x["fully_diluted"] for x in t["holders"]}
    assert fd["Neha"] == 100_000 and fd["ESOP pool"] == 900_000 and t["total_fully_diluted"] == 9_000_000
    too_many = client.post(url(sid, "option-grants"), headers=h, json={"employee_name": "Z", "shares": 2_000_000,
                                                                      "granted_on": "2025-06-01"})
    assert too_many.status_code == 422 and "900,000" in too_many.json()["detail"]


def test_simulated_round_follows_the_pre_money_method(company):
    client, h, sid = company
    client.post(url(sid, "safes"), headers=h, json={"investor_name": "Angel One", "amount": 2_000_000,
                                                   "valuation_cap": 36_000_000, "discount": 0.2, "signed_on": "2025-06-01"})
    s = client.post(url(sid, "cap-table/simulate"), headers=h, json={
        "new_money": 10_000_000, "pre_money": 90_000_000, "pool_target": 0.10, "investor_name": "Seed Fund"}).json()
    table = {r["name"]: r for r in s["table"]}
    total_after = sum(r["after"] for r in s["table"])
    # the new investor owns new money / post-money, because the top-up and SAFE come out of the pre-money
    assert table["Seed Fund"]["ownership_after"] == pytest.approx(10 / 100, abs=1e-4)
    # the unallocated pool after the round is the target share
    assert table["ESOP pool"]["after"] / total_after == pytest.approx(0.10, abs=1e-4)
    # the SAFE converts at the better of cap price (36M / 9M shares = 4.00) and round price less 20%
    safe_price = 2_000_000 / s["safe_conversions"][0]["shares"]
    assert safe_price == pytest.approx(min(4.0, s["price_per_share"] * 0.8), rel=1e-3)
    assert s["post_money"] == 100_000_000 and sum(r["ownership_after"] for r in s["table"]) == pytest.approx(1)


def test_recording_and_undoing_a_round(company):
    client, h, sid = company
    client.post(url(sid, "safes"), headers=h, json={"investor_name": "Angel One", "amount": 1_000_000,
                                                   "valuation_cap": 20_000_000, "signed_on": "2025-06-01"})
    r = client.post(url(sid, "rounds"), headers=h, json={"name": "Seed", "closed_on": "2026-03-01", "new_money": 5_000_000,
                                                        "pre_money": 45_000_000, "investor_name": "Seed Fund"})
    assert r.status_code == 201, r.text
    t = r.json()
    assert t["rounds"][0]["name"] == "Seed" and t["rounds"][0]["post_money"] == 50_000_000 and t["safes"][0]["converted"]
    own = {x["name"]: x for x in t["holders"]}
    assert own["Seed Fund"]["ownership"] == pytest.approx(0.1, abs=1e-4)
    assert own["Seed Fund"]["invested"] == pytest.approx(5e6, rel=1e-4)
    assert client.delete(url(sid, f"rounds/{t['rounds'][0]['id']}"), headers=h).status_code == 204
    after = client.get(url(sid, "cap-table"), headers=h).json()
    assert after["rounds"] == [] and not after["safes"][0]["converted"] and after["total_fully_diluted"] == 9_000_000


def test_pipeline(company):
    client, h, sid = company
    d = client.post(url(sid, "deals"), headers=h, json={"investor_name": "Meera Iyer", "firm": "Peak Capital",
                                                       "amount": 2_000_000}).json()
    assert d["stage"] == "contacted"
    moved = client.patch(url(sid, f"deals/{d['id']}"), headers=h, json={"stage": "term_sheet", "next_step": "Legal review"})
    assert moved.json()["stage"] == "term_sheet" and moved.json()["next_step"] == "Legal review"
    assert len(client.get(url(sid, "deals"), headers=h).json()) == 1
    assert client.delete(url(sid, f"deals/{d['id']}"), headers=h).status_code == 204


def test_other_founders_cannot_see_the_cap_table(company):
    client, h, sid = company
    other = signup(client, "other@example.com")
    for method, path in (("get", "cap-table"), ("get", "deals"), ("post", "cap-table/simulate")):
        kw = {"json": {"new_money": 1, "pre_money": 1}} if method == "post" else {}
        assert getattr(client, method)(url(sid, path), headers=other, **kw).status_code == 404
