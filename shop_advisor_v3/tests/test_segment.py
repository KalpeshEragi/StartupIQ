"""Business profiling: features are scale-free and an upload lands in a named segment."""

import numpy as np
import pytest

from startupiq import segment


def _weekly(n=104, seed=0, zeros=0.0):
    rng = np.random.default_rng(seed)
    y = 1000 + 50 * rng.standard_normal(n)
    y[rng.random(n) < zeros] = 0
    return y


def test_profile_is_unit_free():
    y = _weekly()
    a, b = segment.profile(y), segment.profile(y * 83.0)  # same shop in pounds and in rupees
    assert all(abs(a[f] - b[f]) < 1e-6 for f in ("volatility", "intermittency", "peak_ratio", "season_concentration", "change_rate"))


def test_profile_needs_a_year():
    with pytest.raises(ValueError):
        segment.profile(_weekly(30))


@pytest.mark.skipif(segment.load() is None, reason="run python -m startupiq.segment first")
def test_steady_and_intermittent_shops_differ():
    steady, bursty = segment.assign(_weekly()), segment.assign(_weekly(zeros=0.45, seed=1) * 3)
    assert steady["segment"] == "Steady seller" and 0 < steady["confidence"] <= 1
    assert bursty["segment"] != "Steady seller"
    assert abs(sum(s["share"] for s in steady["all"]) - 1) < 0.02
    assert segment.assign(_weekly(30)) is None
