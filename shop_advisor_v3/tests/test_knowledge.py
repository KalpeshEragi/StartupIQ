"""Task 1: frames, semantic network and the consistency checker on 8 labelled synthetic scenarios."""

import math
from pathlib import Path

import pytest
import yaml

from startupiq.knowledge.frames import Frame
from startupiq.knowledge.kb import KnowledgeBase

SCENARIOS = yaml.safe_load((Path(__file__).parent / "fixtures" / "knowledge" / "scenarios.yaml").read_text(encoding="utf-8"))


def build(spec: dict) -> KnowledgeBase:
    kb = KnowledgeBase()
    kb.add("SmallSeller", "biz", **spec["business"])
    kb.add("Category", spec["category"])
    kb.relate("biz", "operates_in", spec["category"])
    for rival, cat in spec["rivals"].items():
        kb.add("Competitor", rival)
        if cat not in kb.instances:
            kb.add("Category", cat)
        kb.relate(rival, "operates_in", cat)
        kb.relate("biz", "competes_with", rival)
    return kb


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenario(name):
    spec = SCENARIOS[name]
    got = build(spec).check("biz", spec["moves"])
    assert sorted({v["constraint"] for v in got}) == sorted(spec["expect_violations"]), got


def test_slot_lookup_order():
    animal = Frame("Animal", is_class=True).define("legs", default=4).define("sound", default="...")
    bird = Frame("Bird", animal, is_class=True).define("legs", default=2).define("double_legs", if_needed=lambda f: f.get("legs") * 2)
    tweety = bird.instance("tweety", sound="tweet")
    assert tweety.get("sound") == "tweet"                                  # own value
    assert tweety.get("legs") == 2                                         # nearest default (Bird overrides Animal)
    assert tweety.get("double_legs") == 4                                  # if-needed procedure
    assert tweety.get("legs", explain=True) == (2, "default from Bird")
    assert tweety.isa("Animal") and tweety.lineage() == ["tweety", "Bird", "Animal"]


def test_range_violations():
    kb = KnowledgeBase()
    biz = kb.add("SmallSeller", "x", gross_margin=1.4, price_position="free")
    v = " ".join(biz.violations())
    assert "gross_margin" in v and "price_position" in v


def test_if_needed_runway_and_stock_cover():
    kb = KnowledgeBase()
    biz = kb.add("SmallSeller", "x", cash=60000, monthly_fixed_costs=40000, gross_margin=0.3, weekly_sales=10000, stock_value=14000)
    monthly = 10000 * 52 / 12
    assert biz.get("runway_months") == pytest.approx(60000 / (40000 - monthly * 0.3))
    assert biz.get("stock_cover_weeks") == pytest.approx(14000 / (10000 * 0.7))
    rich = kb.add("SmallSeller", "y", cash=1, monthly_fixed_costs=10, gross_margin=0.5, weekly_sales=10000)
    assert math.isinf(rich.get("runway_months"))


def test_competitor_defaults_come_from_base_rates():
    kb = KnowledgeBase()
    r = kb.add("Competitor", "r")
    assert r.get("observed") is False and r.get("p_price_cut") == pytest.approx(0.02)


def test_semantic_network_queries():
    kb = build(SCENARIOS["valid_healthy_cafe"])
    assert kb.related("biz", "competes_with") == ["cafe_bliss"]
    assert "promotion" in kb.moves_affecting("revenue", "+") and "cost_reduction" in kb.moves_affecting("runway", "+")
    assert kb.explain_path("price_coordination", "legal") == ["price_coordination --requires--> legal"]


def test_allowed_moves_never_include_illegal():
    kb = build(SCENARIOS["valid_healthy_cafe"])
    allowed = kb.allowed_moves("biz")
    assert "price_coordination" not in allowed and "hold" in allowed
