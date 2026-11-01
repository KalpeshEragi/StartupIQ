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
    ("TO TRANSFER-NEFT*HDFC0005949*N473860610*BLUE DART COURIER", "blue dart courier"),   # SBI wrapper words
    ("DEBIT-ACH INSTAGRAM PROMOTION 59641958", "instagram promotion"),
    ("UPI-LINKEDIN ADS-linkedin@hdfcbank-YESB0000001-592680656296-UPI", "linkedin ads"),   # keep words next to the UPI ID
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
    ("AIRTEL BROADBAND MONTHLY", "debit", None),                      # merchant lines are not payments to people
    ("UBER TRIP CLIENT MEETING", "debit", None),
    ("STATIONERY AND PRINTER TONER", "debit", None),
    ("IMPS Dr-176955629159-DIVYA RAMAN", "debit", "person_payment"),
    ("UPI/DR/577110510426/SWIGGY/swiggy@icici", "debit", None),             # a business UPI ID: the model decides
    ("UPI/DR/691937865764/GOOGLE ADS/googleads@okaxis", "debit", None),
    ("UPI/DR/886456578909/Nisha Desai/9723204789@pay", "debit", "person_payment"),
    ("UPI/954772172314/PRIYA SHARMA/8815454572@ybl/Payment", "debit", "person_payment"),   # phone-number UPI ID
    ("NEFT DR-HDFC0003000-BIPLAB MUKHERJEE-NETBANK", "debit", "person_payment"),            # "BIPLAB" is not a "lab"
    ("NEFT DR-HDFC0008139-GOOGLE ADS-NETBANK", "debit", None),                              # a business name
    ("BY TRANSFER-NEFT*ICIC0005970*CMS06174356*NIMBUS FOODS LLP", "credit", "revenue"),     # SBI receipt
    ("MMT/IMPS/149276023545/SUNRISE HOSPITALS LTD", "credit", "revenue"),                   # ICICI receipt
    ("TO TRANSFER-UPI/DR/940780676878/CLEARTAX GST FILING/YESB/cleartax@okaxis/", "debit", None),  # a fee, not tax
    ("GST PAYMENT/GSTIN 29ABCDE1234F1Z5/CPIN123456", "debit", "taxes"),
    ("NEFT Dr-347896958678-SOFTWARE SUBSCRIPTION", "debit", None),   # an expense: the learned model decides
    ("NEFT Dr-297219212621-CLOUD SERVICES", "debit", None),
])
def test_indian_formats(description, direction, category):
    assert classify(description, direction, clean(description))[0] == category


def test_keyword_baseline():
    assert keyword_category("electricity bill march") == "utilities"
    assert keyword_category("adobe creative cloud") == "software_it"
    assert keyword_category("something unrelated") is None
