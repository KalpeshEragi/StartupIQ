"""Rules and cleaning used by the categoriser. These run without downloaded data."""
import pytest

from categoriser.formats import classify
from categoriser.keywords import keyword_category
from categoriser.taxonomy import EXPENSE_CATEGORIES, ipsa_category
from categoriser.text import clean


@pytest.mark.parametrize("raw, expected", [
    ("NEFT Dr-347896958678-SOFTWARE SUBSCRIPTION", "software subscription"),
    ("UPI/P2M/695829655510/COURSERA", "coursera"),
    ("CANVA [***] [200013638-67]", "canva"),
    ("NEFT Cr-197068208951-ICIC0SF0002-COMMERCE SOLUTIONS LTD--", "commerce solutions ltd"),
    ("POS DEBIT 222080699339 LOCALGYM", "localgym"),
    ("UPI/DR/886456578909/Nisha Desai/9723204789@pay", "nisha desai"),
    ("", ""),
])
def test_clean_removes_bank_noise(raw, expected):
    assert clean(raw) == expected


@pytest.mark.parametrize("cost_type, category", [
    ("Software & applications", "software_it"), ("Computer SW Purchase", "software_it"),
    ("Hotel - London", "travel"), ("Rail", "travel"), ("Food & Drink Volunteer", "meals"),
    ("Utilities", "utilities"), ("Const Office Electricity", "utilities"),
    ("Accom Tel Usg/Rent (Wind. Up)", "telecom"), ("Mobile telephone - contract & usage", "telecom"),
    ("Const Office Hire of Premises", "rent_premises"), ("Business rates", "rent_premises"),
    ("Venue hire, meetings & surgeries", "marketing"), ("Payroll", "people"),
    ("Insurance - buildings", "insurance"), ("Cleaning services", "facilities"),
    ("Other", None), ("Mortgage Interest", None), ("MP Budget Overspend", None), (None, None),
])
def test_ipsa_mapping(cost_type, category):
    assert ipsa_category(cost_type) == category
    assert category is None or category in EXPENSE_CATEGORIES


@pytest.mark.parametrize("description, direction, category", [
    ("NEFT Dr-704015962232-GST PAYMENT-GSTIN 76C3280", "debit", "taxes"),
    ("NEFT Dr-165604696395-LOAN EMI-MONTHLY INSTALLMENT", "debit", "loan_repayment"),
    ("Service Charges-Cheque Bounce Charge-769895", "debit", "bank_charges"),
    ("ATM WDL-ATM CASH 13930", "debit", "cash"),
    ("CASH DEPOSIT-Simran Kumar", "credit", "cash"),
    ("Dividend Cr-CAPITAL VENTURES INDIA-EQUITY SHARES", "credit", "other_income"),
    ("NEFT Cr-197068208951-ICIC0SF0002-COMMERCE SOLUTIONS LTD--", "credit", "revenue"),
    ("RAZORPAY SETTLEMENT 8822", "credit", "revenue"),
    ("REVERSAL-NEFT-600978537666-RETAIL SOLUTIONS INDIA", "debit", "refund"),
    ("FAILED-INSUFFICIENT FUNDS-ATM WDL-51689", "debit", "failed"),
    ("Chq Paid-MICR Inward Clearing-PADMA VARMA-HDFC BANK LTD.", "debit", "person_payment"),
    ("UPI/DR/655539332400/VARUN MEHTA/UPI-TRANSFER TO 6791527638", "debit", "person_payment"),
    ("NEFT Dr-347896958678-SOFTWARE SUBSCRIPTION", "debit", None),   # an expense: the learned model decides
    ("NEFT Dr-297219212621-CLOUD SERVICES", "debit", None),
])
def test_indian_formats(description, direction, category):
    assert classify(description, direction, clean(description))[0] == category


def test_keyword_baseline():
    assert keyword_category("electricity bill march") == "utilities"
    assert keyword_category("adobe creative cloud") == "software_it"
    assert keyword_category("something unrelated") is None
