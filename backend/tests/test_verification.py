# ruff: noqa: F811 (pytest fixtures shared from test_transactions arrive as arguments)
from decimal import Decimal

from sqlalchemy import delete, select, update

from app.core.db import SessionLocal
from app.models import AuditLog, StatementImport, Transaction
from app.services.verification import inr
from tests.test_transactions import NOVEMBER, OCTOBER, by_desc, no_model, setup, upload  # noqa: F401 (fixtures)

NEWEST_FIRST = """Date,Narration,Withdrawal Amt.,Deposit Amt.,Closing Balance
03/09/26,OFFICE RENT SEP,"50,000.00",,"1,70,000.00"
02/09/26,RAZORPAY SETTLEMENT 9911,,"20,000.00","2,20,000.00"
01/09/26,ZOHO SUBSCRIPTION,"1,000.00",,"2,00,000.00"
"""
NO_BALANCE = """Date,Narration,Withdrawal Amt.,Deposit Amt.
01/10/26,ZOHO SUBSCRIPTION,"1,000.00",
"""


def report(client, h, sid):
    r = client.get(f"/api/v1/startups/{sid}/verification", headers=h)
    assert r.status_code == 200, r.text
    return r.json()


def test_inr_uses_indian_grouping():
    assert inr(Decimal("1234567.5")) == "₹12,34,567.50"
    assert inr(Decimal("-999")) == "-₹999.00"


def test_statement_that_adds_up_is_verified(setup, no_model):
    client, h, sid, acc = setup
    imp = upload(client, h, sid, acc, OCTOBER).json()
    assert imp["balance_status"] == "verified" and imp["balance_checked"] == 4 and imp["balance_breaks"] == 0
    assert len(imp["file_sha256"]) == 64 and imp["same_file_as"] is None
    assert by_desc(client, h, sid)["NEFT DR-HDFC0001234-AMAZON WEB SERVICES"]["source_row"] == 4  # line 4 of the file
    rep = report(client, h, sid)
    assert rep["status"] == "verified" and rep["audit"]["ok"]
    assert rep["imports"][0]["opening_balance"] == 540000 and rep["imports"][0]["closing_balance"] == 604350
    assert rep["lines"]["total"] == rep["lines"]["verified"] == 5
    assert rep["months"][0]["month"] == "2026-10"


def test_edited_balance_is_flagged_with_its_line(setup, no_model):
    client, h, sid, acc = setup
    imp = upload(client, h, sid, acc, OCTOBER.replace('"6,32,550.00"', '"6,30,550.00"')).json()
    assert imp["balance_status"] == "warnings" and imp["balance_breaks"] == 2  # the edited line and the one after it
    assert imp["issues"][0].startswith("Line 6 (03 Oct 2026") and "₹6,32,550.00" in imp["issues"][0]
    rep = report(client, h, sid)
    assert rep["status"] == "warnings" and rep["lines"]["verified"] == 0


def test_newest_first_export_and_missing_balance_column(setup, no_model):
    client, h, sid, acc = setup
    imp = upload(client, h, sid, acc, NEWEST_FIRST, "dec.csv").json()
    assert imp["balance_status"] == "verified" and imp["balance_checked"] == 2
    assert report(client, h, sid)["imports"][0]["opening_balance"] == 201000
    other = client.post(f"/api/v1/startups/{sid}/bank-accounts", json={"name": "Second"}, headers=h).json()["id"]
    imp = upload(client, h, sid, other, NO_BALANCE).json()
    assert imp["balance_status"] == "unverified" and "no readable balances" in imp["issues"][0]


def test_missing_statement_between_uploads_and_same_file_again(setup, no_model):
    client, h, sid, acc = setup
    upload(client, h, sid, acc, OCTOBER, "oct.csv")
    upload(client, h, sid, acc, NOVEMBER, "nov.csv")  # starts at 5,19,100; October ended at 6,04,350
    rep = report(client, h, sid)
    assert rep["status"] == "warnings" and len(rep["continuity"]) == 1
    assert "oct.csv ends at ₹6,04,350.00" in rep["continuity"][0]
    again = upload(client, h, sid, acc, OCTOBER, "oct-copy.csv").json()
    assert again["same_file_as"] == "oct.csv" and again["rows_added"] == 0


def test_history_keeps_first_decision_and_every_change(setup, no_model):
    client, h, sid, acc = setup
    upload(client, h, sid, acc, OCTOBER, "oct.csv")
    gst = by_desc(client, h, sid)["NEFT Dr-704015962232-GST PAYMENT-GSTIN 76C3280"]
    client.patch(f"/api/v1/startups/{sid}/transactions/{gst['id']}", headers=h,
                 json={"category": "bank_charges", "remember": False})
    hist = client.get(f"/api/v1/startups/{sid}/transactions/{gst['id']}/history", headers=h).json()
    assert hist["file"] == "oct.csv" and hist["source_row"] == 6 and hist["statement_status"] == "verified"
    assert hist["first_category"] == gst["category"] and hist["first_source"] == "rule"
    assert hist["transaction"]["category"] == "bank_charges"
    whats = [e["what"] for e in hist["events"]]
    assert whats[0] == "Imported from oct.csv, line 6" and whats[2].startswith("Changed from") and hist["events"][2]["by"]
    assert report(client, h, sid)["lines"]["changed"] == 1


def test_audit_chain_detects_edits_and_removals(setup, no_model):
    client, h, sid, acc = setup
    upload(client, h, sid, acc, OCTOBER)
    chain = report(client, h, sid)["audit"]
    assert chain["ok"] and chain["sealed"] == 3 and len(chain["head"]) == 64  # startup, bank account, import
    with SessionLocal() as db:
        row = db.scalar(select(AuditLog).where(AuditLog.action == "bank_account.create"))
        db.execute(update(AuditLog).where(AuditLog.id == row.id).values(details={"name": "Someone else's"}))
        db.commit()
    chain = report(client, h, sid)["audit"]
    assert not chain["ok"] and "Entry 2" in chain["problem"] and "changed" in chain["problem"]
    with SessionLocal() as db:
        db.execute(delete(AuditLog).where(AuditLog.action == "bank_account.create"))
        db.commit()
    chain = report(client, h, sid)["audit"]
    assert not chain["ok"] and "missing" in chain["problem"]


def test_report_is_private_to_the_startup(setup, client):
    _, _, sid, _ = setup
    from tests.conftest import signup
    stranger = signup(client, "stranger@example.com")
    assert client.get(f"/api/v1/startups/{sid}/verification", headers=stranger).status_code in (403, 404)


def test_lines_changed_in_the_database_are_caught(setup, no_model):
    client, h, sid, acc = setup
    upload(client, h, sid, acc, OCTOBER, "oct.csv")
    assert report(client, h, sid)["status"] == "verified"
    with SessionLocal() as db:  # someone with database access, not the app, lowers a payment
        db.execute(update(Transaction).where(Transaction.description.like("%AMAZON%")).values(amount=Decimal("1845.00")))
        db.commit()
    rep = report(client, h, sid)
    assert rep["status"] == "warnings" and rep["imports"][0]["status"] == "warnings"
    assert "changed or deleted after upload" in rep["imports"][0]["issues"][0] and rep["lines"]["verified"] == 0


def test_statement_from_before_checks_is_checked_by_uploading_it_again(setup, no_model):
    client, h, sid, acc = setup
    upload(client, h, sid, acc, OCTOBER, "oct.csv")
    with SessionLocal() as db:  # what the migration leaves for statements uploaded before this feature
        db.execute(update(StatementImport).values(file_sha256=None, rows_sha256=None, balance_status=None, balance_checked=0,
                                                  opening_balance=None, closing_balance=None, issues=[]))
        db.commit()
    rep = report(client, h, sid)
    assert rep["status"] == "unverified" and "before checks existed" in rep["headline"]
    upload(client, h, sid, acc, OCTOBER, "oct-again.csv")
    rep = report(client, h, sid)
    assert rep["status"] == "verified" and rep["lines"]["verified"] == 5
    assert rep["imports"][0]["issues"][0] == "Checked with oct-again.csv, uploaded later."
