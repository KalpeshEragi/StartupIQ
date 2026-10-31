"""Task 7: correctness of the deep-learning building blocks (fast, synthetic data)."""

import numpy as np
import pandas as pd
import pytest
import torch

from startupiq.dl import windows as W
from startupiq.dl.models import ARCHS, pinball
from startupiq.dl.train import Config, fit, predict


def batch(n=8, l=52, h=12, past=3, fut=4):
    g = torch.Generator().manual_seed(0)
    return torch.randn(n, l, past, generator=g), torch.randn(n, h, fut, generator=g), torch.zeros(n, 1)


@pytest.mark.parametrize("arch", list(ARCHS))
def test_quantiles_never_cross(arch):
    m = ARCHS[arch](lookback=52)
    out = m(*batch())
    assert out.shape == (8, 12, 3)
    assert torch.all(out[..., 0] <= out[..., 1]) and torch.all(out[..., 1] <= out[..., 2])


def test_pinball_loss_values():
    pred = torch.tensor([[[0.0, 1.0, 2.0]]])
    # target 1: P10 under by 1 -> 0.1; P50 exact -> 0; P90 over by 1 -> (1 - 0.9) = 0.1; mean = 0.2 / 3
    assert float(pinball(pred, torch.tensor([[1.0]]))) == pytest.approx(0.2 / 3)


def test_pinball_mask_ignores_closed_days():
    pred = torch.zeros(1, 2, 3)
    target = torch.tensor([[1.0, 100.0]])
    masked = pinball(pred, target, torch.tensor([[1.0, 0.0]]))
    assert float(masked) == pytest.approx(float(pinball(pred[:, :1], target[:, :1])))


def test_attention_weights_sum_to_one():
    m = ARCHS["lstm"](lookback=52, attention=True)
    m(*batch())
    assert torch.allclose(m.last_attention.sum(-1), torch.ones(8, 12), atol=1e-5)


def test_tcn_is_causal():
    m = ARCHS["tcn"](lookback=52).eval()
    p, f, s = batch()
    m(p, f, s, keep_maps=True)
    before = [x.clone() for x in m.feature_maps]
    p2 = p.clone(); p2[:, 40:] += 5.0                      # change only weeks 40..51
    m(p2, f, s, keep_maps=True)
    for a, b in zip(before, m.feature_maps):
        assert torch.allclose(a[..., :40], b[..., :40], atol=1e-6)  # nothing before week 40 may change


def test_store_embedding_is_used():
    m = ARCHS["mlp"](lookback=52, n_ids=5).eval()  # eval: no dropout noise between the two calls
    p, f, s = batch(n=2)
    a = m(p, f, s, torch.tensor([0, 0])); b = m(p, f, s, torch.tensor([0, 3]))
    assert torch.allclose(a[0], b[0]) and not torch.allclose(a[1], b[1])


def test_weekly_window_never_reads_from_the_origin_on():
    y = np.arange(120, dtype=float) * 10 + 100
    weeks = pd.date_range("2021-01-03", periods=120, freq="W-SUN")
    a = W.window(y, weeks, 90, 0)
    y2 = y.copy(); y2[90:] = 1e9
    b = W.window(y2, weeks, 90, 0)
    for x, z in zip(a[:3], b[:3]):                         # past, future, static
        np.testing.assert_array_equal(x, z)
    assert a[4] == b[4]                                    # scale


def test_training_origins_end_before_evaluation():
    o = W.training_origins({"s": np.zeros(110)}, {"s": 100})["s"]
    assert o[0] == W.L and o[-1] + 12 <= 100


def test_fit_and_predict_learn_a_simple_signal():
    rng = np.random.default_rng(0)
    n = 600
    level = rng.uniform(0, 2, n).astype(np.float32)
    past = np.repeat(level[:, None, None], 52, 1).repeat(3, 2).astype(np.float32)
    w = W.Windows(past, np.zeros((n, 12, 4), np.float32), np.zeros((n, 1), np.float32), np.repeat(level[:, None], 12, 1),
                  np.ones(n), np.array([str(i % 60) for i in range(n)]), np.zeros(n, int))
    t = fit(w, Config("mlp", hidden=32, max_epochs=40, lr=3e-3))
    q = predict(t.model, w)
    assert np.corrcoef(np.log1p(q[:, 0, 1]), level)[0, 1] > 0.9  # the model recovers the level it was shown


def test_rossmann_training_origins_cover_every_weekday():
    from startupiq.dl.experiment_rossmann import L, STEP
    origins = range(L, L + 7 * STEP, STEP)
    assert {o % 7 for o in origins} == set(range(7))  # a step that is a multiple of 7 would fix one weekday
