import joblib
import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from app.categorisation import model as model_module
from app.core.config import get_settings
from tests.conftest import STARTUP, signup

OCTOBER = """HDFC BANK Ltd.,,,,,,
Statement of account,,,,,,
Date,Narration,Chq./Ref.No.,Value Dt,Withdrawal Amt.,Deposit Amt.,Closing Balance
01/10/26,NEFT DR-HDFC0001234-AMAZON WEB SERVICES,0000123,01/10/26,"18,450.00",,"5,21,550.00"
02/10/26,RAZORPAY SETTLEMENT 8822,0000124,02/10/26,,"1,20,000.00","6,41,550.00"
03/10/26,NEFT Dr-704015962232-GST PAYMENT-GSTIN 76C3280,0000125,03/10/26,"9,000.00",,"6,32,550.00"
04/10/26,UPI/DR/655539332400/VARUN MEHTA/UPI-TRANSFER TO 6791527638,0000126,04/10/26,"25,000.00",,"6,07,550.00"
05/10/26,ELECTRICITY BILL BESCOM,0000127,05/10/26,"3,200.00",,"6,04,350.00"
,,,,,,
Total,,,,"55,650.00","1,20,000.00",
"""
NOVEMBER = """Date,Narration,Withdrawal Amt.,Deposit Amt.,Closing Balance
01/11/26,NEFT DR-HDFC0001234-AMAZON WEB SERVICES,"19,100.00",,"5,00,000.00"
"""


@pytest.fixture
def setup(client):
    h = signup(client)
    sid = client.post("/api/v1/startups", json=STARTUP, headers=h).json()["id"]
    acc = client.post(f"/api/v1/startups/{sid}/bank-accounts", json={"name": "HDFC current", "last4": "1234"}, headers=h).json()
    return client, h, sid, acc["id"]


def upload(client, h, sid, acc, text, name="statement.csv"):
    return client.post(f"/api/v1/startups/{sid}/bank-accounts/{acc}/imports", headers=h, files={"file": (name, text.encode())})


def by_desc(client, h, sid):
    return {t["description"]: t for t in client.get(f"/api/v1/startups/{sid}/transactions", headers=h).json()["items"]}


@pytest.fixture
def no_model(monkeypatch, tmp_path):
    monkeypatch.setattr(get_settings(), "categoriser_path", str(tmp_path / "missing.joblib"))
    model_module.load_model.cache_clear()
    yield
    model_module.load_model.cache_clear()


@pytest.fixture
def tiny_model(monkeypatch, tmp_path):
    """A stand-in for the trained artefact with the same keys, so the serving path is tested without the big model."""
    texts = ["electricity bill", "electric power bill", "water bill", "software subscription", "cloud services hosting",
             "adobe software", "train ticket", "taxi ride", "hotel stay"]
    labels = ["utilities"] * 3 + ["software_it"] * 3 + ["travel"] * 3
    words, chars = TfidfVectorizer().fit(texts), TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4)).fit(texts)
    bins = [0, 100, 1_000, 10_000, np.inf]
    enc = OneHotEncoder(handle_unknown="ignore").fit(np.arange(1, 5).reshape(-1, 1))
    from scipy import sparse
    X = sparse.hstack([words.transform(texts), chars.transform(texts), enc.transform(np.ones((9, 1), int))]).tocsr()
    est = LogisticRegression(C=50, max_iter=500).fit(X, labels)
    path = tmp_path / "tiny.joblib"
    joblib.dump({"version": "test-1", "name": "tiny", "classes": sorted(set(labels)), "threshold": 0.5, "scores": "proba",
                 "words": words, "chars": chars, "amount_bins": bins, "amount_encoder": enc, "selector": None,
                 "estimator": est}, path)
    monkeypatch.setattr(get_settings(), "categoriser_path", str(path))
    model_module.load_model.cache_clear()
    yield
    model_module.load_model.cache_clear()


def test_import_reads_rows_and_applies_rules(setup, no_model):
    client, h, sid, acc = setup
    r = upload(client, h, sid, acc, OCTOBER)
    assert r.status_code == 201, r.text
    imp = r.json()
    assert imp["rows_added"] == 5 and imp["rows_duplicate"] == 0 and imp["period_start"] == "2026-10-01"
    t = by_desc(client, h, sid)
    assert t["RAZORPAY SETTLEMENT 8822"]["category"] == "revenue" and not t["RAZORPAY SETTLEMENT 8822"]["needs_review"]
    gst = t["NEFT Dr-704015962232-GST PAYMENT-GSTIN 76C3280"]
    assert gst["category"] == "taxes" and gst["category_source"] == "rule"
    person = next(v for k, v in t.items() if "VARUN MEHTA" in k)
    assert person["category"] == "person_payment" and person["needs_review"]
    assert person["suggestions"] == ["people", "professional_services"]
    aws = t["NEFT DR-HDFC0001234-AMAZON WEB SERVICES"]
    assert aws["category"] is None and aws["needs_review"] and "not installed" in aws["reason"]


def test_title_lines_shorter_than_the_table(setup, no_model):
    """Real exports put one-cell title lines above the table; they must not make the reader drop the real rows."""
    client, h, sid, acc = setup
    ragged = 'HDFC BANK Ltd.\n"Statement of account for: CHAI LABS"\n\n' + NOVEMBER
    r = upload(client, h, sid, acc, ragged)
    assert r.status_code == 201, r.text
    assert r.json()["rows_added"] == 1 and "Skipped 2 line(s) above the table." in r.json()["notes"]


def test_same_statement_twice_adds_nothing(setup, no_model):
    client, h, sid, acc = setup
    upload(client, h, sid, acc, OCTOBER)
    again = upload(client, h, sid, acc, OCTOBER).json()
    assert again["rows_added"] == 0 and again["rows_duplicate"] == 5
    assert client.get(f"/api/v1/startups/{sid}/transactions", headers=h).json()["total"] == 5


def test_correction_is_remembered_for_the_vendor(setup, no_model):
    client, h, sid, acc = setup
    upload(client, h, sid, acc, OCTOBER)
    aws = by_desc(client, h, sid)["NEFT DR-HDFC0001234-AMAZON WEB SERVICES"]
    r = client.patch(f"/api/v1/startups/{sid}/transactions/{aws['id']}", json={"category": "software_it"}, headers=h)
    assert r.status_code == 200 and r.json()["transaction"]["category_source"] == "founder"
    upload(client, h, sid, acc, NOVEMBER)  # next month: the same vendor is filed from memory
    nov = next(t for t in client.get(f"/api/v1/startups/{sid}/transactions", params={"month": "2026-11"}, headers=h)
               .json()["items"])
    assert nov["category"] == "software_it" and nov["category_source"] == "memory" and not nov["needs_review"]
    summary = client.get(f"/api/v1/startups/{sid}/transactions/summary", headers=h).json()
    assert summary["total"] == 6 and summary["by_source"]["memory"] == 1


def test_model_files_confident_expenses(setup, tiny_model):
    client, h, sid, acc = setup
    upload(client, h, sid, acc, OCTOBER)
    power = by_desc(client, h, sid)["ELECTRICITY BILL BESCOM"]
    assert power["category"] == "utilities" and power["category_source"] == "model" and not power["needs_review"]
    assert power["confidence"] >= 0.5 and "Utilities" in power["reason"] and power["suggestions"] == []


def test_review_filter_and_bad_category(setup, no_model):
    client, h, sid, acc = setup
    upload(client, h, sid, acc, OCTOBER)
    review = client.get(f"/api/v1/startups/{sid}/transactions", params={"needs_review": True}, headers=h).json()
    assert review["total"] == 3  # AWS and electricity (no model) and the payment to a person
    tid = review["items"][0]["id"]
    assert client.patch(f"/api/v1/startups/{sid}/transactions/{tid}", json={"category": "pizza"}, headers=h).status_code == 422


def test_bad_uploads_are_refused(setup, no_model):
    client, h, sid, acc = setup
    assert upload(client, h, sid, acc, "hello,world\n1,2\n").status_code == 422
    assert upload(client, h, sid, acc, OCTOBER, name="statement.pdf").status_code == 415


def test_other_startups_cannot_touch_these_transactions(setup, no_model):
    client, h, sid, acc = setup
    upload(client, h, sid, acc, OCTOBER)
    tid = client.get(f"/api/v1/startups/{sid}/transactions", headers=h).json()["items"][0]["id"]
    other = signup(client, "other@example.com")
    other_sid = client.post("/api/v1/startups", json=STARTUP, headers=other).json()["id"]
    assert client.get(f"/api/v1/startups/{sid}/transactions", headers=other).status_code == 404
    assert upload(client, other, sid, acc, OCTOBER).status_code == 404
    # even through their own startup, they cannot reach this account or transaction
    assert upload(client, other, other_sid, acc, OCTOBER).status_code == 404
    r = client.patch(f"/api/v1/startups/{other_sid}/transactions/{tid}", json={"category": "travel"}, headers=other)
    assert r.status_code == 404
