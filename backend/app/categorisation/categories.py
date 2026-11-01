"""Every category a transaction can get.

Expense categories are learned by the model (ml/categoriser); structural categories come from the bank narration
format (formats.py). Keys are stored in the database; labels are what founders see.
"""

EXPENSE_CATEGORIES: dict[str, str] = {
    "rent_premises": "Rent & premises",
    "utilities": "Utilities",
    "telecom": "Phone & internet",
    "software_it": "Software & IT services",
    "equipment": "Equipment & furniture",
    "office_supplies": "Office supplies & postage",
    "subscriptions": "Subscriptions & publications",
    "marketing": "Marketing & events",
    "professional_services": "Professional services",
    "facilities": "Cleaning, repairs & security",
    "insurance": "Insurance",
    "people": "People costs",
    "travel": "Travel & accommodation",
    "meals": "Meals & hospitality",
}

STRUCTURAL_CATEGORIES: dict[str, str] = {
    "revenue": "Revenue (customer receipts)",
    "other_income": "Interest, dividends & other income",
    "refund": "Refunds & reversals",
    "financing": "Loans & investment received",
    "taxes": "Taxes (GST, TDS, income tax)",
    "loan_repayment": "Loan repayments & EMIs",
    "bank_charges": "Bank charges",
    "cash": "Cash withdrawals & deposits",
    "person_payment": "Payment to a person (needs review)",
    "failed": "Failed / not completed",
}

ALL_CATEGORIES: dict[str, str] = {**STRUCTURAL_CATEGORIES, **EXPENSE_CATEGORIES}
INCOME = {"revenue", "other_income", "refund", "financing"}  # money in; everything else is money out or neutral
