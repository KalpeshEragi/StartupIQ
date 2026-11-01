"""Download daily Wikipedia page views for every sector article (Wikimedia Analytics API, data under CC0).

    python -m market.pageviews            (from ml/)

Human views only (agent=user, so known bots and spiders are excluded), all devices. One CSV per article lands in
ml/data/raw/pageviews/<project>/ (not committed). Re-running only fetches days that are missing, so it can be run
every week to extend the data. Each article's views are summed over its own title and every redirect to it,
so renamed articles keep their full history.
"""
from __future__ import annotations

import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

from market.sectors import START, articles

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "pageviews"
API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
# Wikimedia asks every client to identify itself with a User-Agent that says who it is and where to find out more.
USER_AGENT = "StartupIQ-research/0.1 (https://github.com/KalpeshEragi/StartupIQ; term project)"
PAUSE = 0.15  # seconds between requests, well under the published rate limit


def url(project: str, title: str, start: date, end: date) -> str:
    article = urllib.parse.quote(title.replace(" ", "_"), safe="")
    return f"{API}/{project}/all-access/user/{article}/daily/{start:%Y%m%d}00/{end:%Y%m%d}00"


def path_for(project: str, title: str) -> Path:
    safe = urllib.parse.quote(title.replace(" ", "_"), safe="")
    return RAW / project / f"{safe}.csv"


def read(project: str, title: str) -> dict[str, int]:
    """Saved daily views for one article, {YYYY-MM-DD: views}."""
    p = path_for(project, title)
    if not p.exists():
        return {}
    with p.open(encoding="utf-8") as f:
        return {row["date"]: int(row["views"]) for row in csv.DictReader(f)}


def _get(link: str) -> list[dict]:
    req = urllib.request.Request(link, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 (fixed https API)
        return json.loads(r.read())["items"]


AGG = "https://wikimedia.org/api/rest_v1/metrics/pageviews/aggregate"


def totals_path(project: str) -> Path:
    return RAW / "_totals" / f"{project}.csv"


def read_totals(project: str) -> dict[str, int]:
    """All human page views on a whole Wikipedia (e.g. all of English Wikipedia) per day, {YYYY-MM-DD: views}."""
    p = totals_path(project)
    if not p.exists():
        return {}
    with p.open(encoding="utf-8") as f:
        return {row["date"]: int(row["views"]) for row in csv.DictReader(f)}


def fetch_totals(project: str, end: date) -> None:
    """Daily total human views of the whole project. Used to turn article views into a share of attention, so that
    Wikipedia gaining or losing readers overall does not look like every market rising or falling."""
    link = f"{AGG}/{project}/all-access/user/daily/{date.fromisoformat(START):%Y%m%d}00/{end:%Y%m%d}00"
    days = merge([_get(link)])
    p = totals_path(project)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "views"])
        w.writerows(sorted(days.items()))


def redirects(project: str, title: str) -> list[str]:
    """Other titles that redirect to this article. Views stay with the title a reader opened, so when an article
    is renamed (e.g. "Gemini (chatbot)" became "Google Gemini" in late 2025) its history sits under the old name."""
    lang = project.split(".")[0]
    q = urllib.parse.quote(title)
    link = (f"https://{lang}.wikipedia.org/w/api.php?action=query&format=json&prop=redirects"
            f"&rdnamespace=0&rdlimit=max&titles={q}")
    req = urllib.request.Request(link, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 (fixed https API)
        pages = json.loads(r.read())["query"]["pages"].values()
    return [rd["title"] for pg in pages for rd in pg.get("redirects", [])]


def merge(series: list[list[dict]]) -> dict[str, int]:
    """Add up daily views from several titles of the same article."""
    total: dict[str, int] = {}
    for items in series:
        for item in items:
            day = datetime.strptime(item["timestamp"][:8], "%Y%m%d").date().isoformat()
            total[day] = total.get(day, 0) + int(item["views"])
    return total


def fetch(project: str, title: str, end: date) -> str:
    """Bring one article up to date, summing its own title and every redirect to it.
    Returns 'new', 'updated', 'current' or 'missing'."""
    have = read(project, title)
    start = date.fromisoformat(START)
    if have:
        last = date.fromisoformat(max(have))
        if last >= end:
            return "current"
        start = last + timedelta(days=1)
    series = []
    for name in [title, *redirects(project, title)]:
        try:
            series.append(_get(url(project, name, start, end)))
        except urllib.error.HTTPError as e:
            if e.code != 404:  # 404 = no views for that title in the window
                raise
        time.sleep(PAUSE)
    if not series:
        return "missing"
    have.update(merge(series))
    p = path_for(project, title)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "views"])
        w.writerows(sorted(have.items()))
    return "updated" if start > date.fromisoformat(START) else "new"


def check_titles() -> list[str]:
    """Titles that are redirects or do not exist. Page views count only the exact title, so a redirect would
    silently collect almost nothing (e.g. "Saree" redirects to "Sari")."""
    problems, by_project = [], {}
    for _, project, title in articles():
        by_project.setdefault(project, []).append(title)
    for project, titles in by_project.items():
        lang = project.split(".")[0]
        for i in range(0, len(titles), 40):
            q = urllib.parse.quote("|".join(titles[i:i + 40]))
            link = f"https://{lang}.wikipedia.org/w/api.php?action=query&format=json&redirects=1&titles={q}"
            req = urllib.request.Request(link, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 (fixed https API)
                data = json.loads(r.read())["query"]
            for n in data.get("normalized", []):
                problems.append(f"{project}: '{n['from']}' should be written '{n['to']}'")
            for red in data.get("redirects", []):
                problems.append(f"{project}: '{red['from']}' redirects to '{red['to']}'")
            problems += [f"{project}: '{pg['title']}' does not exist" for pg in data["pages"].values() if "missing" in pg]
    return problems


def main() -> None:
    problems = check_titles()
    if problems:
        print("Fix these titles in market/sectors.py first:")
        print("\n".join(f"  {x}" for x in problems))
        raise SystemExit(1)
    end = date.today() - timedelta(days=1)  # yesterday is the last complete day
    for project in ("en.wikipedia", "hi.wikipedia"):
        fetch_totals(project, end)
    missing, counts = [], {"new": 0, "updated": 0, "current": 0, "missing": 0}
    for sector, project, title in articles():
        status = fetch(project, title, end)
        counts[status] += 1
        if status == "missing":
            missing.append(f"{sector}: {project} / {title}")
        time.sleep(PAUSE)
    print(f"Up to {end}: {counts}")
    if missing:
        print("Not found (fix the title in market/sectors.py or drop it):")
        print("\n".join(f"  {m}" for m in missing))


if __name__ == "__main__":
    main()
