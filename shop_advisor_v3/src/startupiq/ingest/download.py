"""Download raw datasets into data/raw and record them in the manifest.

Usage (run from the project root, on a machine with normal internet access):

    python -m startupiq.ingest.download yc
    set STARTUPIQ_CONTACT=your-name-and-email
    python -m startupiq.ingest.download formd --quarters 2024q1 2024q2

URL patterns below have NOT been verified against the live sites yet. If a
download fails with 404, check the publisher's page and pass --url explicitly.
Nothing downloaded here is committed to git (see .gitignore).
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from startupiq.ingest.manifest import append_row, manifest_row

YC_URL = "https://yc-oss.github.io/api/companies/all.json"  # path unverified
FORMD_URL = "https://www.sec.gov/files/structureddata/data/form-d-data-sets/{quarter}_d.zip"  # pattern unverified

YC_LICENCE = "None stated by publisher (unofficial yc-oss/api). Experiments only until confirmed."
FORMD_LICENCE = "SEC page states no licence; catalogue entry labels it US public domain. Confirm before commercial use."


def fetch(url: str, dest: Path, headers: dict[str, str]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response, dest.open("wb") as out:
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            out.write(block)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", choices=["yc", "formd"])
    parser.add_argument("--quarters", nargs="*", default=[], help="Form D quarters, e.g. 2024q1")
    parser.add_argument("--url", help="Override the URL (yc only, or one formd file)")
    parser.add_argument("--out", default="data/raw", help="Output folder")
    args = parser.parse_args(argv)

    out = Path(args.out)
    manifest = out / "MANIFEST.md"
    jobs: list[tuple[str, Path, str]] = []
    headers = {"User-Agent": "startupiq-research"}

    if args.source == "yc":
        jobs.append((args.url or YC_URL, out / "yc" / "all.json", YC_LICENCE))
    else:
        contact = os.environ.get("STARTUPIQ_CONTACT")
        if not contact:
            print("Set STARTUPIQ_CONTACT to your name and email; the SEC asks automated clients to identify themselves.")
            return 2
        headers["User-Agent"] = f"startupiq-research {contact}"
        if args.url:
            jobs.append((args.url, out / "formd" / Path(args.url).name, FORMD_LICENCE))
        else:
            if not args.quarters:
                print("Pass --quarters, e.g. --quarters 2024q1 2024q2")
                return 2
            for quarter in args.quarters:
                jobs.append((FORMD_URL.format(quarter=quarter), out / "formd" / f"{quarter}_d.zip", FORMD_LICENCE))

    failures = 0
    for url, dest, licence in jobs:
        try:
            print(f"Downloading {url}")
            fetch(url, dest, headers)
            append_row(manifest, manifest_row(dest, url, licence))
            print(f"  saved {dest} and added a manifest row")
        except (urllib.error.URLError, OSError) as error:
            failures += 1
            print(f"  FAILED: {error}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
