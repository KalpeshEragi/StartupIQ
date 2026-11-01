"""Cleaning transaction text so models see words and merchant names, not reference numbers.

Used identically in training and in the product, so a model never meets text cleaned differently from how it learned.
"""
from __future__ import annotations

import re

_REDACTED = re.compile(r"\[\*+\]")                       # IPSA redactions: [***]
_BRACKET_REF = re.compile(r"\[[0-9][0-9\-/ ]*\]")         # [200013638-68]
_BANK_PREFIX = re.compile(r"^(?:upi|neft|imps|rtgs|pos debit|pos|ach|nach|ecs|chq|cheque|by clg|clg)\b"
                          r"[\s/:-]*(?:dr|cr|p2m|p2a)?\b[\s/:-]*",
                          re.IGNORECASE)
_IFSC = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")
_VPA = re.compile(r"[\w.\-]+@[\w.\-]+")                            # UPI handles and e-mail addresses
_LONG_NUMBER = re.compile(r"\b[A-Z]*\d[\dA-Z]{5,}\b")      # reference numbers, account numbers, card stubs
_NUMBER = re.compile(r"\d+")
_SEPARATORS = re.compile(r"[/_*|\\\-:;,.#()\[\]+&]+")
_SPACES = re.compile(r"\s+")


def clean(text: str | None) -> str:
    """Lower-case words with bank prefixes, reference numbers and identifiers removed. Numbers become '0'."""
    if not text:
        return ""
    t = _REDACTED.sub(" ", text)
    t = _BRACKET_REF.sub(" ", t)
    t = _IFSC.sub(" ", t)
    t = _VPA.sub(" ", t)
    t = _BANK_PREFIX.sub(" ", t.strip())
    t = _LONG_NUMBER.sub(" ", t.upper())
    t = _NUMBER.sub("0", t)
    t = _SEPARATORS.sub(" ", t)
    return _SPACES.sub(" ", t).strip().lower()
