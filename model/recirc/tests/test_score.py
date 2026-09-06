#!/usr/bin/env python3
"""Tests for the closed-form pair scorer.

The one that matters is `test_first_order_predicts_actual`: the whole premise of RECIRC-01 is
that a single forward+backward ranks pairs as well as running them, so the predicted loss
change must track the measured one in the small-alpha regime the paper operates in.
"""
import numpy as np
import pytest
import torch

from model.recirc.recirc import Recirculator, tap_list
from model.recirc.score import all_gains, collect, score_pairs
from model.recirc.tests.test_recirc import Stub


@pytest.fixture
def stub():
    torch.manual_seed(0)
    return Stub().eval()


def test_collect_covers_every_tap(stub):
    x = torch.randn(1, 5, 8)
    z, _ = collect(stub, lambda: stub(x))
    assert set(z) == {l for l, _ in tap_list(stub)}


def test_score_pairs_enumerates_ordered_pairs(stub):
    x0, x1 = torch.randn(1, 5, 8), torch.randn(1, 5, 8)
    tgt = torch.randn(1, 5, 8)
    rows = score_pairs(stub, lambda: stub(x0), lambda: stub(x1),
                       lambda o: ((o - tgt) ** 2).mean(), alpha=0.05)
    L = len(tap_list(stub))
    assert len(rows) == L * (L - 1) // 2
    assert rows == sorted(rows, key=lambda r: r["score"])


def test_first_order_predicts_actual(stub):
    """Predicted vs measured loss change across all pairs, at an alpha in the paper's range."""
    torch.manual_seed(1)
    x0, x1 = torch.randn(1, 5, 8), torch.randn(1, 5, 8)
    tgt = torch.randn(1, 5, 8)
    lf = lambda o: ((o - tgt) ** 2).mean()
    alpha = 0.05

    with torch.no_grad():
        base = float(lf(stub(x1)))
        prev, _ = collect(stub, lambda: stub(x0))

    rows = score_pairs(stub, lambda: stub(x0), lambda: stub(x1), lf, alpha=alpha)

    pred, actual = [], []
    for r in rows:
        rec = Recirculator(stub, r["source"], r["dest"], alpha=alpha, mode="shuffle")
        rec.set_state(prev[r["source"]])
        with torch.no_grad():
            actual.append(float(lf(stub(x1))) - base)
        rec.remove()
        pred.append(r["first_order"])

    rho = float(np.corrcoef(pred, actual)[0, 1])
    assert rho > 0.9, f"first-order score does not track measured change (r={rho:.3f})"
    # and the ranking, which is what actually gets used
    sp = float(np.corrcoef(np.argsort(np.argsort(pred)),
                           np.argsort(np.argsort(actual)))[0, 1])
    assert sp > 0.8, f"rank correlation too low ({sp:.3f})"


def test_gain_divides_score(stub):
    x0, x1 = torch.randn(1, 5, 8), torch.randn(1, 5, 8)
    tgt = torch.randn(1, 5, 8)
    lf = lambda o: ((o - tgt) ** 2).mean()
    g = {("aat.1", "enc.1"): 0.5}
    rows = score_pairs(stub, lambda: stub(x0), lambda: stub(x1), lf, alpha=0.05, gains=g)
    r = next(r for r in rows if r["source"] == "aat.1" and r["dest"] == "enc.1")
    assert r["score"] == pytest.approx(r["first_order"] / 0.5, rel=1e-5)


def test_all_gains_subcritical_on_stub(stub):
    x = torch.randn(1, 5, 8)
    g = all_gains(stub, lambda: stub(x), alpha=0.10,
                  pairs=[("aat.1", "enc.1"), ("aat.2", "enc.0")], iters=6)
    assert all(0.0 <= v < 1.0 for v in g.values()), g
