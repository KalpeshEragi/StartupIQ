"""Give every transaction a category, in this order, and say how it was decided:

1. memory  - the founder already filed this vendor; use their choice (this startup only)
2. rule    - the bank narration says what the line is (GST, EMI, bank charge, cash, customer receipt...)
3. model   - the learned categoriser predicts the kind of expense; filed if confident, otherwise suggested
4. review  - nothing confident: the founder is asked

Corrections are stored as training examples and, if the founder agrees, remembered for the vendor.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.categorisation import formats
from app.categorisation.categories import ALL_CATEGORIES
from app.categorisation.model import load_model
from app.categorisation.text import clean
from app.models import CategoryCorrection, CategorySource, Transaction, VendorRule


def _label(key: str | None) -> str:
    return ALL_CATEGORIES.get(key or "", "uncategorised")


@dataclass
class Decision:
    category: str | None
    source: CategorySource
    needs_review: bool
    reason: str
    confidence: float | None = None
    suggestions: str | None = None
    model_version: str | None = None


def decide(lines: list[tuple[str, str, float]]) -> list[Decision]:
    """Rules, then the model, for (description, direction, amount) lines; no founder memory. The product and the demo
    checker both use this, so what is measured is what founders get."""
    model = load_model()
    out: list[Decision | None] = []
    for_model: list[int] = []
    for i, (description, direction, _amount) in enumerate(lines):
        category, reason = formats.classify(description, direction, clean(description))
        if category == "person_payment":  # who was paid, and for what, only the founder knows
            out.append(Decision(category, CategorySource.rule, True, reason[0].upper() + reason[1:], 1.0,
                                "people,professional_services"))  # salary or a contractor, in most startups
        elif category is not None:
            out.append(Decision(category, CategorySource.rule, False, reason[0].upper() + reason[1:], 1.0))
        elif direction == "debit" and model is not None:
            out.append(None)
            for_model.append(i)
        elif direction == "debit":
            out.append(Decision(None, CategorySource.none, True, "The categoriser model is not installed, so this needs review"))
        else:
            out.append(Decision(None, CategorySource.none, True, "Money in that we could not recognise"))
    if for_model:
        predictions = model.predict([lines[i][0] for i in for_model], [float(lines[i][2]) for i in for_model])
        for i, p in zip(for_model, predictions, strict=True):
            if p.confident:
                out[i] = Decision(p.category, CategorySource.model, False,
                                  f"Looks like {_label(p.category)} from the description", round(p.confidence, 4), None,
                                  model.version)
            else:  # scores are margins, not probabilities, so founders see ranked suggestions rather than percentages
                why = ("New vendor. Best guesses: " if not model.knows(lines[i][0]) else "Not sure. Best guesses: ")
                out[i] = Decision(p.category, CategorySource.none, True, why + ", ".join(_label(c) for c, _ in p.alternatives),
                                  round(p.confidence, 4), ",".join(c for c, _ in p.alternatives), model.version)
    return out  # type: ignore[return-value]


def categorise(db: Session, startup_id: uuid.UUID, txns: list[Transaction]) -> None:
    """Fill category, source, confidence, needs_review and reason on each transaction (founder-set ones are left alone)."""
    txns = [t for t in txns if t.category_source != CategorySource.founder]
    if not txns:
        return
    memory = {r.vendor_key: r for r in db.scalars(select(VendorRule).where(VendorRule.startup_id == startup_id))}
    rest: list[Transaction] = []
    for t in txns:
        remembered = memory.get(t.vendor_key) if t.vendor_key else None
        if remembered is not None:
            t.category, t.category_source, t.confidence = remembered.category, CategorySource.memory, 1.0
            t.needs_review, t.model_version, t.suggestions = False, None, None
            t.reason = f"You filed “{t.vendor_key}” as {_label(remembered.category)} before"
            remembered.times_applied += 1
        else:
            rest.append(t)
    for t, d in zip(rest, decide([(t.description, str(t.direction), float(t.amount)) for t in rest]), strict=True):
        t.category, t.category_source, t.needs_review, t.reason = d.category, d.source, d.needs_review, d.reason
        t.confidence, t.suggestions, t.model_version = d.confidence, d.suggestions, d.model_version


def correct(db: Session, txn: Transaction, new_category: str, user_id: uuid.UUID, remember: bool) -> int:
    """Apply the founder's category. With remember=True the vendor is remembered and every other transaction of this
    startup with the same vendor that the founder has not set by hand is updated too. Returns how many others changed."""
    db.add(CategoryCorrection(startup_id=txn.startup_id, transaction_id=txn.id, created_by=user_id,
                              previous_category=txn.category, previous_source=str(txn.category_source),
                              new_category=new_category, text=txn.vendor_key or clean(txn.description)[:200],
                              direction=str(txn.direction), amount=txn.amount))
    txn.category, txn.category_source, txn.confidence, txn.needs_review = new_category, CategorySource.founder, 1.0, False
    txn.suggestions = None
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
                suggestions=None, model_version=None, reason=f"You filed “{txn.vendor_key}” as {_label(new_category)}")
    )
    return result.rowcount or 0
