"""Task 9: the web app end to end through HTTP."""

from pathlib import Path

from fastapi.testclient import TestClient

from startupiq.api.app import app

PACK = Path(__file__).parent / "fixtures" / "intake" / "files"
client = TestClient(app)


def test_page_and_health():
    assert client.get("/api/health").json() == {"ok": True}
    assert "Startup IQ" in client.get("/").text


def test_analyze_full_upload_returns_all_cards():
    with open(PACK / "04_title_rows.xlsx", "rb") as f:
        r = client.post("/api/analyze", files={"file": ("sales.xlsx", f)},
                        data={"cash": "40000", "fixed_costs": "12000", "margin_pct": "40", "stock_value": "4000",
                              "lead_time_days": "10", "competitors": "Rival A, Rival B", "rival_cut_prices": "yes"})
    rep = r.json()
    assert r.status_code == 200 and rep["tier"] == "full"
    assert [c["key"] for c in rep["cards"] if c["key"] not in ("segment", "category")] == ["intake", "forecast", "money", "warnings", "market", "rivals", "next_move"]
    assert "plan" in rep["cards"][-1]["details"]


def test_refused_upload_explains_why():
    with open(PACK / "10_no_date.csv", "rb") as f:
        rep = client.post("/api/analyze", files={"file": ("x.csv", f)}).json()
    assert rep["tier"] == "refused" and "date" in rep["cards"][0]["headline"]


def test_evidence_lists_the_three_deep_models():
    ev = client.get("/api/evidence").json()
    assert [d["arch"] for d in ev["deep"]] == ["MLP", "LSTM + attention", "1D CNN (TCN)"]
    assert min(ev["deep"], key=lambda d: d["test"])["arch"] == "1D CNN (TCN)"


def test_samples_include_the_real_product():
    assert any(s["file"].startswith("real_") for s in client.get("/api/samples").json())
