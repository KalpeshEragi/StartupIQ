# ruff: noqa: F811 (pytest fixtures shared from test_transactions arrive as arguments)
"""Odd inputs found by fuzzing a running app: each one used to crash, be stored wrongly, or would overflow PostgreSQL."""
from tests.test_transactions import by_desc, no_model, setup, upload  # noqa: F401 (fixtures)

H = "Date,Narration,Withdrawal Amt.,Deposit Amt.,Closing Balance\n"


def test_large_statement_is_imported_in_chunks(setup, no_model):
    client, h, sid, acc = setup
    rows = "".join(f"01/10/26,LINE {i},1.00,,{100_000 - i}.00\n" for i in range(1, 2_501))
    r = upload(client, h, sid, acc, H + rows)
    assert r.status_code == 201 and r.json()["rows_added"] == 2_500 and r.json()["balance_status"] == "verified"
    assert upload(client, h, sid, acc, H + rows).json()["rows_duplicate"] == 2_500  # the duplicate check also chunks


def test_impossible_lines_are_left_out_with_a_reason(setup, no_model):
    client, h, sid, acc = setup
    text = H + ("01/01/1900,ANCIENT,1.00,,999.00\n01/10/99,FUTURE,1.00,,998.00\n02/10/26,BOTH,5.00,7.00,1000.00\n"
                "03/10/26,HUGE,9999999999999.00,,1.00\n04/10/26,OK,10.00,,990.00\n")
    imp = upload(client, h, sid, acc, text).json()
    assert imp["rows_added"] == 1 and imp["rows_skipped"] == 4
    notes = " ".join(imp["notes"])
    assert "before 2000 or in the future" in notes and "both a withdrawal and a deposit" in notes and "impossibly large" in notes


def test_utf16_and_windows_1252_files_are_read(setup, no_model):
    client, h, sid, acc = setup
    assert upload(client, h, sid, acc, "x").status_code == 422  # sanity: the helper sends text
    r = client.post(f"/api/v1/startups/{sid}/bank-accounts/{acc}/imports", headers=h,
                    files={"file": ("s.csv", (H + "01/10/26,ZOHO,100.00,,900.00\n").encode("utf-16"))})
    assert r.status_code == 201
    r = client.post(f"/api/v1/startups/{sid}/bank-accounts/{acc}/imports", headers=h,
                    files={"file": ("s.csv", (H + "02/10/26,CAFÉ COFFEE DAY,50.00,,850.00\n").encode("cp1252"))})
    assert r.status_code == 201 and "CAFÉ COFFEE DAY" in by_desc(client, h, sid)


def test_uploaded_file_name_keeps_no_path(setup, no_model):
    client, h, sid, acc = setup
    r = client.post(f"/api/v1/startups/{sid}/bank-accounts/{acc}/imports", headers=h,
                    files={"file": ("..\\..\\etc/passwd.csv", (H + "01/10/26,ZOHO,100.00,,900.00\n").encode())})
    assert r.json()["filename"] == "passwd.csv"


def test_lines_without_a_balance_are_checked_through_the_next_one(setup, no_model):
    client, h, sid, acc = setup
    ok = H + "01/10/26,A,100.00,,900.00\n02/10/26,B,50.00,,\n03/10/26,C,25.00,,825.00\n"
    imp = upload(client, h, sid, acc, ok).json()
    assert imp["balance_status"] == "verified" and imp["balance_checked"] == 1 and "1 line(s) have no balance" in imp["issues"][0]
    other = client.post(f"/api/v1/startups/{sid}/bank-accounts", json={"name": "B"}, headers=h).json()["id"]
    edited = H + "01/10/26,A,100.00,,900.00\n02/10/26,B,5.00,,\n03/10/26,C,25.00,,825.00\n"  # B lowered from 50 to 5
    assert upload(client, h, sid, other, edited).json()["balance_status"] == "warnings"


def test_numbers_outside_what_the_database_holds_are_refused(setup):
    client, h, sid, acc = setup
    base = f"/api/v1/startups/{sid}"
    assert client.patch(f"{base}/bank-accounts/{acc}", json={"opening_balance": 1e20}, headers=h).status_code == 422
    assert client.patch(f"{base}/bank-accounts/{acc}", json={"opening_balance": "NaN"}, headers=h).status_code == 422
    assert client.patch(f"{base}/bank-accounts/{acc}", json={"opening_balance": -5000}, headers=h).status_code == 200
    assert client.put(f"{base}/unit-metrics/2026-10", json={"units_sold": 10**12}, headers=h).status_code == 422
    assert client.post(f"{base}/cap-table/setup", headers=h, json={
        "incorporated_on": "2025-01-01", "founders": [{"name": "A", "shares": 10**12}]}).status_code == 422
    assert client.post(f"{base}/cap-table/setup", headers=h, json={
        "incorporated_on": "2999-01-01", "founders": [{"name": "A", "shares": 100}]}).status_code == 422
    assert client.get(f"{base}/transactions", params={"month": "2026-13"}, headers=h).status_code == 422


def test_too_long_or_all_rejected_statements_say_why(setup, no_model, monkeypatch):
    from app.categorisation import statements
    client, h, sid, acc = setup
    monkeypatch.setattr(statements, "MAX_ROWS", 3)
    long = H + "".join(f"01/10/26,L{i},1.00,,{1000 - i}.00\n" for i in range(1, 61))  # 60 lines > 3 + header scan
    r = upload(client, h, sid, acc, long)
    assert r.status_code == 422 and "more than" in r.json()["detail"]
    monkeypatch.setattr(statements, "MAX_ROWS", 50_000)
    r = upload(client, h, sid, acc, H + "01/10/99,FUTURE,1.00,,9.00\n")
    assert r.status_code == 422 and "1 line(s) dated before 2000 or in the future" in r.json()["detail"]
