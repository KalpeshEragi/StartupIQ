"""Text helpers shared by the cleaning steps."""

from __future__ import annotations

import re
import unicodedata

_TAGS = re.compile(r"<[^>]+>")
_URLS = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_SPACES = re.compile(r"\s+")


def is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return value != value  # NaN
    except Exception:
        return False


def normalize_text(value) -> str:
    """Unicode-normalise, strip HTML tags and URLs, collapse whitespace. Missing becomes ''."""
    if is_missing(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = _TAGS.sub(" ", text)
    text = _URLS.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def website_domain(url) -> str | None:
    """Host of a URL without scheme, 'www.', port, or path. None if empty."""
    if is_missing(url):
        return None
    value = str(url).strip().lower()
    if not value:
        return None
    value = re.sub(r"^[a-z]+://", "", value)
    value = value.split("/")[0].split("?")[0].split("#")[0].split(":")[0]
    if value.startswith("www."):
        value = value[4:]
    return value or None


def word_count(text: str) -> int:
    return len(text.split()) if text else 0
