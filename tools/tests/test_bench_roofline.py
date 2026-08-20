"""Gate 0 decision arithmetic. The proceed/kill call must not rest on hand arithmetic."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bench_roofline import (  # noqa: E402
    D, DH, K, R, TOKENS, decide, ffn_gemms, penalty, predict_time_s, variant_cost,
)

PEAK_TOPS = 67e12
PEAK_BW = 102e9


def test_ffn_gemm_shapes_cover_both_matrices():
    assert ffn_gemms() == [(729, 768, 3072), (729, 3072, 768)]


def test_dense_cost_matches_spec():
    b, f = variant_cost("dense")
    assert b == 4_718_592                      # 2 * 768 * 3072, INT8
    assert f == pytest.approx(6.879e9, rel=1e-3)
    assert f / b == pytest.approx(1458, rel=1e-2)


def test_vanilla_reads_k_copies_but_does_dense_flops():
    bd, fd = variant_cost("dense")
    bv, fv = variant_cost("vanilla")
    assert bv == K * bd
    assert fv == fd                            # top-1: one expert's work per token
    assert bv == 18_874_368
    assert fv / bv == pytest.approx(365, rel=1e-2)


def test_residual_delta_accounting():
    bd, _ = variant_cost("dense")
    br, fr = variant_cost("residual")
    delta = br - bd
    assert delta == 2 * K * R * (D + DH) == 245_760
    assert br == 4_964_352
    assert delta / bd == pytest.approx(0.0521, rel=1e-2)     # +5.2% params vs dense
    assert variant_cost("vanilla")[0] / br == pytest.approx(3.80, rel=1e-2)
    assert fr / br == pytest.approx(1458, rel=1e-2)


def test_roofline_at_peak_reproduces_spec_table():
    r = penalty(PEAK_TOPS, PEAK_BW)
    assert r["dense"]["time_us"] == pytest.approx(103, rel=2e-2)
    assert r["vanilla"]["time_us"] == pytest.approx(185, rel=2e-2)
    assert r["residual"]["time_us"] == pytest.approx(108, rel=2e-2)
    assert r["dense"]["bound_by"] == "compute"
    assert r["vanilla"]["bound_by"] == "memory"
    assert r["residual"]["bound_by"] == "compute"
    assert r["vanilla_penalty"] == pytest.approx(1.80, rel=2e-2)


def test_threshold_is_37_tops():
    """Below ~37 TOPS achieved the vanilla arm stops being memory-bound and costs nothing."""
    just_above = penalty(38e12, PEAK_BW)
    just_below = penalty(36e12, PEAK_BW)
    assert just_above["vanilla"]["bound_by"] == "memory"
    assert just_below["vanilla"]["bound_by"] == "compute"
    assert just_below["vanilla_penalty"] == pytest.approx(1.0, abs=1e-6)


def test_lower_bandwidth_lowers_the_threshold():
    """At 70 GB/s sustained the threshold falls to ~25.5 TOPS, so the method looks better."""
    assert penalty(26e12, 70e9)["vanilla"]["bound_by"] == "memory"
    assert penalty(25e12, 70e9)["vanilla"]["bound_by"] == "compute"


def test_predict_time_takes_the_binding_constraint():
    assert predict_time_s(1e9, 1e6, 1e12, 1e12) == pytest.approx(1e-3)   # compute binds
    assert predict_time_s(1e9, 1e9, 1e15, 1e9) == pytest.approx(1.0)     # memory binds


@pytest.mark.parametrize("modes,expected", [
    ({"maxn": 1.80, "15w": 1.40}, "proceed"),
    ({"maxn": 1.30, "15w": 1.02}, "proceed"),      # one mode clearing is enough
    ({"maxn": 1.05, "15w": 1.00}, "kill"),
    ({"maxn": 1.18, "15w": 1.05}, "measure_directly"),
])
def test_decision_rule(modes, expected):
    assert decide(modes) == expected
