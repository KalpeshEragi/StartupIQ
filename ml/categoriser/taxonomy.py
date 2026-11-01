"""Startup expense categories, and how IPSA cost types map onto them.

The categoriser learns *which kind of operating expense* a bank line is. Structural lines (customer receipts,
transfers between own accounts, tax payments, loan EMIs, bank charges, cash) are recognised from the bank format by
rules (formats.py), not learned here.
"""
from __future__ import annotations

import re

from app.categorisation.categories import EXPENSE_CATEGORIES  # noqa: F401  (re-exported for the training code)

# key -> label shown to founders. Order is the display order.

# Ordered (pattern, category) rules over the IPSA cost_type name, first match wins. None = excluded from the
# dataset, because the type is not a startup operating expense or its rows are not one kind of thing.
_RULES: list[tuple[str, str | None]] = [
    # excluded: MP-specific money movements, catch-alls and income
    (r"mortgage|budget repayment|overspend|contingency|expense outside|outside the scheme|residential deposit|rics valuation"
     r"|rental income|subletting|working from home|^other$|^miscellaneous$|^office miscellaneous|^winding up$|^staffing$"
     r"|aggregated travel|tv licence|television", None),
    (r"insur", "insurance"),
    (r"hotel", "travel"),
    (r"food & drink|hospitality|subsistence", "meals"),
    (r"rail|taxi|air travel|public tr|mileage|parking|congestion|vehicle|car hire|own vehicle|railcard|^travel|travel cost"
     r"|travel associated|motor|ferry|coach", "travel"),
    (r"payroll|salary|recruitment|training|health (and|&) welfare|pooled|volunteer - agreed|volunteer agreed|redundancy"
     r"|reward|staff costs|employment costs|staff one off", "people"),
    (r"advertising|contact cards", "marketing"),
    (r"hire of premises", "rent_premises"),
    (r"venue hire", "marketing"),  # too few recent rows with text to learn events on their own
    (r"newspaper|journal", "subscriptions"),
    (r"software|website|comp sw|computer sw", "software_it"),
    (r"stationery|postage|couriers|consumables|printing", "office_supplies"),
    (r"tel\.|\btel\b|telephone|mobile|landline|internet|fax machine hire|^fax", "telecom"),
    (r"utilit|electricity|gas|water|fuel|waste|shredding", "utilities"),
    (r"equip|computer|comp hw|furniture|furn |printer|photocopier|scanner|shredder|purchase of tv|install", "equipment"),
    (r"clean|repair|redecorat|maintenance|secur|dilapidation|alterations|removal|moving", "facilities"),
    (r"professional|accountancy|parliamentary account|translation|language|bought-in", "professional_services"),
    (r"rent|service charge|service chgs|business rates|council tax", "rent_premises"),
]
_COMPILED = [(re.compile(p, re.IGNORECASE), c) for p, c in _RULES]


def ipsa_category(cost_type: str | None) -> str | None:
    """Startup category for an IPSA cost type, or None when the type is excluded."""
    if not isinstance(cost_type, str) or not cost_type:
        return None
    for pattern, category in _COMPILED:
        if pattern.search(cost_type):
            return category
    return None
