"""Runtime code shared with the product lives in the backend; training imports it from there."""
from app.categorisation.formats import STRUCTURAL_CATEGORIES, classify  # noqa: F401
