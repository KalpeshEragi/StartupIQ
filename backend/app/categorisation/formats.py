"""Structural lines of an Indian business bank statement, recognised from the narration format.

These are not "which kind of expense" questions (that is the learned model's job) but what the line *is*: money in
from customers, tax paid to the government, a loan instalment, a bank fee, cash. Formats follow the NEFT / IMPS /
RTGS / UPI / cheque / ATM narrations Indian banks print; patterns were written from the public narration formats and
checked against the AgamiAI statements (ml/categoriser).

classify() returns (category, reason) or (None, reason) when the line should go to the learned model, or to the
founder because it is a payment to a named person with no purpose given.
"""
from __future__ import annotations

import re

from app.categorisation.categories import STRUCTURAL_CATEGORIES  # noqa: F401  (re-exported)

_C = lambda p: re.compile(p, re.IGNORECASE)  # noqa: E731
# Ordered: first match wins. (pattern, direction or None for both, category, reason)
_RULES: list[tuple[re.Pattern, str | None, str, str]] = [
    (_C(r"^failed\b|insufficient funds|\breturned\b|\brtn\b"), None, "failed", "the bank marks this line as failed"),
    (_C(r"^reversal\b|\brefund\b|\brev\b-"), None, "refund", "the bank marks this line as a reversal or refund"),
    (_C(r"\bgst\b|gstin|\btds\b|\bincome tax\b|advance tax|\bcbdt\b|\bchallan\b|\btin nsdl\b|\bpt challan|professional tax"),
     "debit", "taxes", "tax payment narration (GST / TDS / income tax challan)"),
    (_C(r"\bemi\b|loan\b.*(instal|repay|emi)|\bloan emi\b|nach.*loan|instal+ment"), "debit", "loan_repayment",
     "loan instalment narration"),
    (_C(r"service charges?|\bchgs?\b|\bcharges?\b|\bfee\b.*bank|\bsms alert|\bamc\b|cheque bounce|chq bounce"
        r"|min(imum)? bal|gst on charges"),
     "debit", "bank_charges", "bank fee narration"),
    (_C(r"^atm\b|atm wdl|atm cash|cash withdrawal|^cash-|cash wdl|\bself\b"), "debit", "cash", "cash withdrawn"),
    (_C(r"cash deposit|^cash-bna|^by cash|\bcdm\b|^cash\b"), "credit", "cash", "cash deposited"),
    (_C(r"dividend|\binterest\b|\bint\.? ?cr|int pd|\bsb int"), "credit", "other_income", "interest or dividend credit"),
    (_C(r"loan disb|disbursement|\bcapital infusion|share application|equity infusion"), "credit", "financing",
     "loan disbursement or capital received"),
    (_C(r"razorpay|cashfree|payu|paytm settlement|stripe|instamojo|ccavenue|phonepe settlement|settlement"), "credit",
     "revenue", "payment-gateway settlement (customer payments)"),
]
# Money in through a transfer or cheque with no other signal is, for a business account, a customer paying.
_TRANSFER = _C(r"^(neft|imps|rtgs|upi|by clg|clg|chq|cheque|inward|trf|transfer)")
# A payment out whose narration is only a person's name (no purpose words) cannot be categorised from the text.
_PERSON_NAME_ONLY = _C(r"^[a-z]+(?: [a-z]+){0,3}$")
# Words a bank adds around the payee's name, removed before deciding whether only a name is left.
_BANK_WORDS = _C(r"\b(paid|micr|inward|clearing|chq|cheque|dr|cr|none|urgent|upi|transfer|to|from|sent"
                 r"|(hdfc|icici|axis|sbi|kotak|yes|canara|pnb|idfc"
                 r"|indusind|citi|federal|union) bank( ltd)?|bank ltd|state bank of india)\b")
_PURPOSE_HINT = _C(r"ltd|limited|pvt|private|india|corp|services|solutions|systems|technolog|infotech|consult|industries"
                   r"|subscription|payment|rent|utility|cloud|insurance|software|bill|invoice|store|mart|enterprise")


def classify(description: str, direction: str, cleaned: str | None = None) -> tuple[str | None, str]:
    """(category, reason). category None means: not a structural line, use the learned expense model."""
    text = description or ""
    for pattern, only, category, reason in _RULES:
        if (only is None or only == direction) and pattern.search(text):
            return category, reason
    if direction == "credit" and _TRANSFER.search(text.strip()):
        return "revenue", "money received by transfer or cheque"
    if direction == "debit" and cleaned is not None:
        name = " ".join(_BANK_WORDS.sub(" ", cleaned).split())
        if name and _PERSON_NAME_ONLY.match(name) and not _PURPOSE_HINT.search(name):
            return "person_payment", "payment to a named person with no purpose given"
    return None, "not a structural line"
