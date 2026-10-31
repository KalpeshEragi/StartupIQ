"""Broken-file pack: every file must get the expected tier, roles and questions."""

from datetime import date
from pathlib import Path

import pytest
import yaml

from startupiq.intake import read_file

PACK = Path(__file__).parent / "fixtures" / "intake"
EXPECTED = yaml.safe_load((PACK / "expected.yaml").read_text(encoding="utf-8"))
TODAY = date(2026, 10, 24)


def test_every_file_has_an_expectation():
    assert sorted(p.name for p in (PACK / "files").iterdir()) == sorted(EXPECTED)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_pack_file(name):
    exp = EXPECTED[name]
    got = read_file(PACK / "files" / name, today=TODAY)
    assert got.tier.value == exp["tier"], got.messages
    for role, col in exp.get("roles", {}).items():
        assert got.column_map.get(role) == col, got.column_map
    if "ask" in exp:
        assert got.needs_confirmation == sorted(exp["ask"]), got.messages
    for key in ("layout", "granularity", "sheet"):
        if key in exp:
            assert getattr(got, key) == exp[key]
    if "message" in exp:
        assert any(exp["message"] in m for m in got.messages), got.messages
    if "log" in exp:
        assert any(exp["log"] in m for m in got.cleaning_log), got.cleaning_log
    if got.tier.value != "refused":
        assert got.series and all(len(s.dates) == len(s.values) for s in got.series)
