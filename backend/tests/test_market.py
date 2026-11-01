"""Market outlook: sector suggestion, model choice per sector, "you vs your market", and the API."""
import json
from decimal import Decimal

from app.core.config import get_settings
from app.market import service
from tests.conftest import STARTUP, signup


def _week(i: int) -> str:
    return f"2026-{(i // 4) % 12 + 1:02d}-{(i % 4) * 7 + 1:02d}"


def _sector(network: float, smoothing: float) -> dict:
    hist = [{"week": _week(i), "index": 100.0} for i in range(52)]
    band = lambda mid: [{"week": f"2027-01-{i + 1:02d}", "p10": mid - 10, "p50": mid, "p90": mid + 10} for i in range(12)]  # noqa: E731
    return {"label": "Food and beverage brands (D2C)", "history": hist, "forecast": band(112.0), "smoothing": band(101.0),
            "backtest": {"network_mase": network, "smoothing_mase": smoothing},
            "monthly": {"2026-09": 100.0, "2026-10": 110.0, "2026-11": 99.0}}


FILE = {"generated_at": "2026-11-01T00:00:00+00:00", "data_through": "2026-10-25", "model": "test",
        "test": {}, "sectors": {"food_beverage": _sector(0.76, 1.26), "ai_tools": _sector(2.6, 2.0)}}


def test_suggest_maps_free_text_sectors():
    assert service.suggest("D2C beverages") == "food_beverage"
    assert service.suggest("B2B SaaS for CAs") == "saas_software"
    assert service.suggest("Edtech for JEE coaching") == "edtech"
    assert service.suggest("Underwater basket weaving") is None


def test_outlook_uses_the_model_that_did_better_for_that_sector():
    food = service.outlook(FILE, "food_beverage")
    assert food["model"] == "network" and food["change_12w"] == 0.12
    assert (food["change_12w_low"], food["change_12w_high"]) == (0.02, 0.22)
    ai = service.outlook(FILE, "ai_tools")
    assert ai["model"] == "smoothing" and ai["change_12w"] == 0.01


def test_you_vs_market_compares_last_month_moves():
    out = service.you_vs_market({"2026-10": Decimal("100"), "2026-11": Decimal("120")}, {"2026-10": 110.0, "2026-11": 99.0})
    assert round(out["your_change"], 2) == 0.20 and round(out["market_change"], 2) == -0.10
    assert "grew faster than" in out["summary"]
    assert out["correlation"] is None  # too few months to mean anything
    assert "two months" in service.you_vs_market({"2026-11": Decimal("1")}, {"2026-11": 1.0})["summary"]


def test_api_serves_outlook_and_remembers_the_chosen_sector(client, monkeypatch, tmp_path):
    path = tmp_path / "market_forecast.json"
    path.write_text(json.dumps(FILE))
    monkeypatch.setattr(get_settings(), "market_forecast_path", str(path))
    h = signup(client)
    sid = client.post("/api/v1/startups", json=STARTUP, headers=h).json()["id"]

    r = client.get(f"/api/v1/startups/{sid}/market", headers=h).json()
    assert r["sector"] == "food_beverage" and r["suggested"] == "food_beverage" and r["chosen"] is None
    assert r["outlook"]["model"] == "network" and len(r["outlook"]["forecast"]) == 12

    assert client.patch(f"/api/v1/startups/{sid}", json={"market_sector": "nonsense"}, headers=h).status_code == 422
    assert client.patch(f"/api/v1/startups/{sid}", json={"market_sector": "ai_tools"}, headers=h).status_code == 200
    r = client.get(f"/api/v1/startups/{sid}/market", headers=h).json()
    assert r["chosen"] == "ai_tools" and r["outlook"]["model"] == "smoothing"
    assert len(client.get("/api/v1/market/sectors").json()) == 19


def test_api_without_forecast_file_says_how_to_make_it(client, monkeypatch, tmp_path):
    monkeypatch.setattr(get_settings(), "market_forecast_path", str(tmp_path / "missing.json"))
    h = signup(client)
    sid = client.post("/api/v1/startups", json=STARTUP, headers=h).json()["id"]
    r = client.get(f"/api/v1/startups/{sid}/market", headers=h).json()
    assert r["available"] is False and "market.forecast" in r["message"]


def test_market_is_private_to_members(client):
    h = signup(client)
    sid = client.post("/api/v1/startups", json=STARTUP, headers=h).json()["id"]
    other = signup(client, email="other@example.com", name="Other")
    assert client.get(f"/api/v1/startups/{sid}/market", headers=other).status_code == 404
