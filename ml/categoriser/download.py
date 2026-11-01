"""Download the public datasets used to build and benchmark the transaction categoriser.

    python -m categoriser.download          (from ml/)

1. AgamiAI/Indian-Bank-Statements (Apache-2.0): synthetic Indian business current-account statements with realistic
   NEFT / IMPS / RTGS / UPI / cheque narrations. Only the JSON files are fetched (the PDFs hold the same statements).
2. DoDataThings/us-bank-transaction-categories-v2 (MIT): 68,000 labelled US bank descriptions, used as an external
   benchmark only.

Files land in ml/data/raw/ (not committed). Re-running skips files that are already there.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
HF = "https://huggingface.co"
INDIAN = "AgamiAI/Indian-Bank-Statements"
FOLDERS = ["India_Bank_Statement_Digital_Type1", "India_Bank_Statement_Digital_Type2",
           "India_Bank_Statement_Scanned_Type1", "India_Bank_Statement_Scanned_Type2"]
US = ("DoDataThings/us-bank-transaction-categories-v2", "transactions-synthetic.csv")


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 (fixed https URLs)
        return r.read()


def _fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(_get(url))


def main() -> None:
    for folder in FOLDERS:
        listing = json.loads(_get(f"{HF}/api/datasets/{INDIAN}/tree/main/train/{folder}"))
        files = [x["path"] for x in listing if x["path"].endswith(".json")]
        for path in files:
            _fetch(f"{HF}/datasets/{INDIAN}/resolve/main/{path}", RAW / "indian" / Path(path).relative_to("train"))
        print(f"{folder}: {len(files)} statements")
    repo, name = US
    _fetch(f"{HF}/datasets/{repo}/resolve/main/{name}", RAW / "us" / name)
    print(f"US benchmark: {RAW / 'us' / name}")


if __name__ == "__main__":
    main()
