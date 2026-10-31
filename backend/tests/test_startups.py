import uuid

from tests.conftest import STARTUP, signup


def test_create_and_list_startup(client):
    h = signup(client)
    r = client.post("/api/v1/startups", json=STARTUP, headers=h)
    assert r.status_code == 201
    s = r.json()
    assert s["role"] == "founder" and s["currency"] == "INR" and s["fy_start_month"] == 4
    listed = client.get("/api/v1/startups", headers=h).json()
    assert [x["id"] for x in listed] == [s["id"]]


def test_founder_can_own_several_startups(client):
    h = signup(client)
    client.post("/api/v1/startups", json=STARTUP, headers=h)
    client.post("/api/v1/startups", json={**STARTUP, "name": "Second Co", "product_type": "digital"}, headers=h)
    assert len(client.get("/api/v1/startups", headers=h).json()) == 2


def test_update_changes_only_sent_fields_and_is_audited(client):
    h = signup(client)
    sid = client.post("/api/v1/startups", json=STARTUP, headers=h).json()["id"]
    r = client.patch(f"/api/v1/startups/{sid}", json={"stage": "series_a"}, headers=h)
    assert r.status_code == 200 and r.json()["stage"] == "series_a" and r.json()["name"] == "Chai Labs"
    log = client.get(f"/api/v1/startups/{sid}/activity", headers=h).json()
    assert log[0]["action"] == "startup.update" and log[0]["details"]["stage"] == {"from": "seed", "to": "series_a"}
    assert log[-1]["action"] == "startup.create"


def test_other_users_cannot_see_or_change_a_startup(client):
    owner = signup(client, "owner@example.com")
    sid = client.post("/api/v1/startups", json=STARTUP, headers=owner).json()["id"]
    intruder = signup(client, "intruder@example.com")
    for method, path, kw in (("get", f"/api/v1/startups/{sid}", {}),
                             ("patch", f"/api/v1/startups/{sid}", {"json": {"name": "x"}}),
                             ("get", f"/api/v1/startups/{sid}/activity", {})):
        assert getattr(client, method)(path, headers=intruder, **kw).status_code == 404  # 404: existence not revealed
    assert client.get("/api/v1/startups", headers=intruder).json() == []
    assert client.get(f"/api/v1/startups/{uuid.uuid4()}", headers=owner).status_code == 404


def test_invalid_startup_input_is_rejected(client):
    h = signup(client)
    for bad in ({**STARTUP, "stage": "unicorn"}, {**STARTUP, "fy_start_month": 13}, {**STARTUP, "currency": "rupees"},
                {**STARTUP, "founded_on": "2999-01-01"}, {k: v for k, v in STARTUP.items() if k != "name"}):
        assert client.post("/api/v1/startups", json=bad, headers=h).status_code == 422


def test_times_are_sent_with_a_timezone(client):
    h = signup(client)
    s = client.post("/api/v1/startups", json=STARTUP, headers=h).json()
    log = client.get(f"/api/v1/startups/{s['id']}/activity", headers=h).json()
    assert s["created_at"].endswith(("Z", "+00:00")) and log[0]["created_at"].endswith(("Z", "+00:00"))
