"""Helpers for recording what was downloaded, when, and from where."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

HEADER = (
    "| File | Source URL | Retrieved (UTC) | Size (bytes) | SHA-256 | Licence note |\n"
    "| --- | --- | --- | --- | --- | --- |\n"
)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_row(path: Path, url: str, licence_note: str, retrieved: datetime | None = None) -> str:
    path = Path(path)
    when = (retrieved or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M")
    return (
        f"| {path.name} | {url} | {when} | {path.stat().st_size} | "
        f"{sha256_file(path)} | {licence_note} |\n"
    )


def append_row(manifest_path: Path, row: str) -> None:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text("# Raw data manifest\n\n" + HEADER, encoding="utf-8")
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(row)
