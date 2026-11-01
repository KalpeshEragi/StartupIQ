"""Keyword rules: the baseline every learned model must beat.

Written from general bookkeeping knowledge, not from the training labels, so it is the honest "what rules alone
can do" line. The first category whose keyword appears in the cleaned text wins; no keyword means no answer.
"""
from __future__ import annotations

import re

KEYWORDS: dict[str, list[str]] = {
    "insurance": ["insurance", "insur", "premium", "policy"],
    "travel": ["hotel", "rail", "train", "taxi", "uber", "ola", "flight", "airline", "air ", "parking", "mileage", "fuel",
               "petrol", "oyster", "tube", "bus", "travel", "ticket", "makemytrip", "irctc"],
    "meals": ["food", "lunch", "dinner", "breakfast", "coffee", "tea", "catering", "restaurant", "swiggy", "zomato",
              "sandwich", "refreshment", "hospitality", "meal"],
    "people": ["salary", "payroll", "wages", "recruitment", "training", "course", "staff", "intern", "volunteer", "pension"],
    "marketing": ["advert", "marketing", "leaflet", "flyer", "ads", "promotion", "campaign", "facebook", "google ads",
                  "venue", "hall hire", "room hire", "event"],
    "subscriptions": ["newspaper", "magazine", "journal", "times", "guardian", "subscription", "membership"],
    "software_it": ["software", "licence", "license", "website", "hosting", "domain", "adobe", "microsoft", "canva",
                    "mailchimp", "zoom", "app", "cloud", "aws", "it support"],
    "office_supplies": ["stationery", "stationary", "paper", "toner", "ink", "printing", "print", "envelope", "postage",
                        "stamps", "courier", "royal mail", "pens"],
    "telecom": ["phone", "mobile", "telephone", "broadband", "internet", "bt ", "vodafone", "o0", "ee ", "airtel", "jio",
                "line rental", "sim"],
    "utilities": ["electric", "electricity", "gas", "water", "energy", "power", "waste", "bin", "recycling", "edf"],
    "equipment": ["laptop", "computer", "printer", "monitor", "desk", "chair", "furniture", "equipment", "keyboard",
                  "shredder", "photocopier", "scanner", "headset"],
    "facilities": ["cleaning", "cleaner", "repair", "maintenance", "locksmith", "security", "alarm", "decorat", "removal"],
    "professional_services": ["accountant", "accountancy", "legal", "solicitor", "consultant", "translation",
                              "interpreter", "audit", "professional"],
    "rent_premises": ["rent", "lease", "service charge", "council tax", "business rates", "landlord"],
}
_COMPILED = [(c, re.compile(r"\b(?:" + "|".join(re.escape(k.strip()) for k in ks) + r")", re.IGNORECASE))
             for c, ks in KEYWORDS.items()]


def keyword_category(text: str) -> str | None:
    for category, pattern in _COMPILED:
        if pattern.search(text):
            return category
    return None
