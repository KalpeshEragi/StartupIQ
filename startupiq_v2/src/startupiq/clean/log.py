"""A small, explicit record of what each cleaning rule did."""

from __future__ import annotations

import json
from pathlib import Path


class CleaningLog:
    """Collects one entry per rule so the cleaning is auditable and reproducible."""

    def __init__(self, dataset: str):
        self.dataset = dataset
        self.steps: list[dict] = []

    def rows(self, rule: str, before: int, after: int, note: str = "") -> None:
        """A rule that removes rows."""
        self.steps.append(
            {"rule": rule, "kind": "rows", "rows_before": before, "rows_after": after,
             "rows_removed": before - after, "note": note}
        )

    def flag(self, rule: str, count: int, total: int, note: str = "") -> None:
        """A rule that changes or flags values without removing rows."""
        self.steps.append(
            {"rule": rule, "kind": "values", "affected": int(count), "of_total": int(total), "note": note}
        )

    def info(self, rule: str, value, note: str = "") -> None:
        self.steps.append({"rule": rule, "kind": "info", "value": value, "note": note})

    def to_dict(self) -> dict:
        return {"dataset": self.dataset, "steps": self.steps}

    def write(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
