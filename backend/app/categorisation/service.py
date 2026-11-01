"""Give every transaction a category, in this order, and say how it was decided:

1. memory  - the founder already filed this vendor; use their choice (this startup only)
2. rule    - the bank narration says what the line is (GST, EMI, bank charge, cash, customer receipt...)
3. model   - the learned categoriser predicts the kind of expense; filed if confident, otherwise suggested
4. review  - nothing confident: the founder is asked

Corrections are stored as training examples and, if the founder agrees, remembered for the vendor.
"""
from __future__ import annotations

import uuid

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.categorisation import formats
from app.categorisation.categories import ALL_CATEGORIES
from app.categorisation.model import load_model
from app.categorisation.text import clean
from app.models import CategoryCorrection, CategorySource, Transaction, VendorRule


def _label(key: str | None) -> str:
    return ALL_CATEGORIES.get(key or "", "uncategorised")


def categorise(db: Session, startup_id: uuid.UUID, txns: list[Transaction]) -> None:
    """Fill category, source, confidence, needs_review and reason on each transaction (founder-set ones are left alone)."""
    txns = [t for t in txns if t.category_source != CategorySource.founder]
    if not txns:
        return
    memory = {r.vendor_key: r for r in db.scalars(select(VendorRule).where(VendorRule.startup_id == startup_id))}
    for_model: list[Transaction] = []
    for t in txns:
        remembered = memory.get(t.vendor_key) if t.vendor_key else None
        if remembered is not None:
            t.category, t.category_source, t.confidence = remembered.category, CategorySource.memory, 1.0
            t.needs_review, t.model_version = False, None
            t.reason = f"You filed “{t.vendor_key}” as {_label(remembered.category)} before"
            remembered.times_applied += 1
            continue
        category, reason = formats.classify(t.description, str(t.direction), clean(t.description))
        if category is not None:
            t.category, t.category_source, t.confidence, t.model_version = category, CategorySource.rule, 1.0, None
            t.needs_review = category == "person_payment"  # who was paid, and for what, only the founder knows
            t.reason = reason[0].upper() + reason[1:]
        elif str(t.direction) == "debit":
            for_model.append(t)
        else:
            t.category, t.category_source, t.confidence, t.needs_review = None, CategorySource.none, None, True
            t.reason = "Money in that we could not recognise"
    if not for_model:
        return
    model = load_model()
    if model is None:
        for t in for_model:
            t.category, t.category_source, t.confidence, t.needs_review = None, CategorySource.none, None, True
            t.reason = "The categoriser model is not installed, so this needs your review"
        return
    predictions = model.predict([t.description for t in for_model], [float(t.amount) for t in for_model])
    for t, p in zip(for_model, predictions, strict=True):
        t.category, t.confidence, t.model_version = p.category, round(p.confidence, 4), model.version
        if p.confident:
            t.category_source, t.needs_review = CategorySource.model, False
            t.reason = f"Predicted {_label(p.category)} ({p.confidence:.0%} sure)"
        else:
            t.category_source, t.needs_review = CategorySource.none, True
            t.reason = "Not sure: " + ", ".join(f"{_label(c)} {s:.0%}" for c, s in p.alternatives)


def correct(db: Session, txn: Transaction, new_category: str, user_id: uuid.UUID, remember: bool) -> int:
    """Apply the founder's category. With remember=True the vendor is remembered and every other transaction of this
    startup with the same vendor that the founder has not set by hand is updated too. Returns how many others changed."""
    db.add(CategoryCorrection(startup_id=txn.startup_id, transaction_id=txn.id, created_by=user_id,
                              previous_category=txn.category, previous_source=str(txn.category_source),
                              new_category=new_category, text=txn.vendor_key or clean(txn.description)[:200],
                              direction=str(txn.direction), amount=txn.amount))
    txn.category, txn.category_source, txn.confidence, txn.needs_review = new_category, CategorySource.founder, 1.0, False
    txn.reason = "Set by you"
    if not (remember and txn.vendor_key):
        return 0
    rule = db.scalar(select(VendorRule).where(VendorRule.startup_id == txn.startup_id, VendorRule.vendor_key == txn.vendor_key))
    if rule is None:
        db.add(VendorRule(startup_id=txn.startup_id, vendor_key=txn.vendor_key, category=new_category, created_by=user_id))
    else:
        rule.category = new_category
    result = db.execute(
        update(Transaction)
        .where(Transaction.startup_id == txn.startup_id, Transaction.vendor_key == txn.vendor_key, Transaction.id != txn.id,
               Transaction.category_source != CategorySource.founder)
        .values(category=new_category, category_source=CategorySource.memory, confidence=1.0, needs_review=False,
                model_version=None, reason=f"You filed “{txn.vendor_key}” as {_label(new_category)}")
    )
    return result.rowcount or 0
