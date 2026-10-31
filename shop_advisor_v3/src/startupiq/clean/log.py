"""Cleaning log: every rule records rows before and after, and why it ran."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Entry:
    rule: str
    rows_before: int
    rows_after: int
    note: str

    @property
    def changed(self) -> int:
        return self.rows_before - self.rows_after


@dataclass
class CleaningLog:
    dataset: str
    entries: list[Entry] = field(default_factory=list)

    def record(self, rule: str, before: int, after: int, note: str) -> None:
        self.entries.append(Entry(rule, before, after, note))

    def to_markdown(self) -> str:
        lines = [f"### {self.dataset}", "", "| Rule | Rows before | Rows after | Removed | Note |", "| --- | ---: | ---: | ---: | --- |"]
        for e in self.entries:
            lines.append(f"| {e.rule} | {e.rows_before:,} | {e.rows_after:,} | {e.changed:,} | {e.note} |")
        return "\n".join(lines) + "\n"

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{self.dataset}_cleaning_log.json").write_text(
            json.dumps([asdict(e) for e in self.entries], indent=2), encoding="utf-8")
