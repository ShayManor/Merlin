#!/usr/bin/env python3
"""Hook-logic tests for cross-frame recirculation, on a stub stack (no MapAnything needed).

What can silently go wrong and produce plausible numbers rather than an error:
  - hooks never fire (regex miss, wrong module) -> arm reads exactly like the control
  - source captured BEFORE the destination injects -> within-frame looping, not recurrence
  - warm-up frame already injected -> arms differ by one frame for the wrong reason
  - shuffle mode overwriting its injected state from its own stream -> control becomes `prev`
Each of those is asserted here.
"""
import re
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

import model.recirc.recirc as R


class Blk(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x):
        return x + 0.1 * torch.tanh(self.lin(x))


class Stub(nn.Module):
    """Mimics the student's two-stack naming so the real regexes match."""

    def __init__(self, n_enc=4, n_aat=3, d=8):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.model = nn.Module()
        self.encoder.model.blocks = nn.ModuleList([Blk(d) for _ in range(n_enc)])
        self.info_sharing = nn.Module()
        self.info_sharing.self_attention_blocks = nn.ModuleList([Blk(d) for _ in range(n_aat)])

    def forward(self, x):
        for b in self.encoder.model.blocks:
            x = b(x)
        for b in self.info_sharing.self_attention_blocks:
            x = b(x)
        return x


@pytest.fixture
def stub():
    torch.manual_seed(0)
    return Stub().eval()


def test_tap_list_order(stub):
    labels = [l for l, _ in R.tap_list(stub)]
    assert labels == ["enc.0", "enc.1", "enc.2", "enc.3", "aat.0", "aat.1", "aat.2"]


def test_rejects_source_before_dest(stub):
    with pytest.raises(ValueError):
        R.Recirculator(stub, source="enc.1", dest="aat.0")
    with pytest.raises(ValueError):
        R.Recirculator(stub, source="enc.2", dest="enc.2")


@torch.no_grad()
def test_warmup_frame_is_identity(stub):
    x0 = torch.randn(1, 5, 8)
    base = stub(x0)
    rec = R.Recirculator(stub, "aat.1", "enc.1", alpha=0.5)
    out0 = stub(x0)
    assert torch.allclose(base, out0, atol=1e-6), "frame 0 must not be injected into"
    assert rec.n_injected == 0
    rec.remove()


@torch.no_grad()
def test_second_frame_is_perturbed_and_reset_restores(stub):
    x0, x1 = torch.randn(1, 5, 8), torch.randn(1, 5, 8)
    base1 = stub(x1)
    rec = R.Recirculator(stub, "aat.1", "enc.1", alpha=0.5)
    stub(x0)
    out1 = stub(x1)
    assert rec.n_injected == 1
    assert not torch.allclose(base1, out1, atol=1e-5), "injection had no effect"
    rec.reset()
    out1b = stub(x1)
    assert torch.allclose(base1, out1b, atol=1e-6), "reset must restore warm-up behaviour"
    rec.remove()


@torch.no_grad()
def test_alpha_zero_is_identity(stub):
    x0, x1 = torch.randn(1, 5, 8), torch.randn(1, 5, 8)
    base1 = stub(x1)
    rec = R.Recirculator(stub, "aat.1", "enc.1", alpha=0.0)
    stub(x0)
    assert torch.allclose(base1, stub(x1), atol=1e-6)
    rec.remove()


@torch.no_grad()
def test_state_is_post_injection_single_stack(stub):
    """z_s must be computed from the RECIRCULATED stream, which is what makes this a genuine
    single-stack recurrence rather than a two-stack read of a clean pass."""
    x0, x1 = torch.randn(1, 5, 8), torch.randn(1, 5, 8)
    rec = R.Recirculator(stub, "aat.1", "enc.1", alpha=0.5)
    stub(x0)
    s_clean = rec.state.clone()
    stub(x1)
    s_after = rec.state.clone()
    rec.reset()
    stub(x1)
    s_noinject = rec.state.clone()
    assert not torch.allclose(s_after, s_noinject, atol=1e-6), \
        "source tap did not see the injection; ordering is wrong"
    assert s_clean.shape == s_after.shape
    rec.remove()


@torch.no_grad()
def test_norm_matching_rescales_source(stub):
    """Per Eq. 2 the source is rescaled to the destination's L2 norm, so a source scaled by
    1000x must give the same result as the unscaled one."""
    x = torch.randn(1, 5, 8)
    outs = []
    for k in (1.0, 1000.0):
        rec = R.Recirculator(stub, "aat.1", "enc.1", alpha=0.5, norm="token")
        rec.set_state(torch.randn(1, 5, 8, generator=torch.Generator().manual_seed(3)) * k)
        outs.append(stub(x))
        rec.remove()
    assert torch.allclose(outs[0], outs[1], atol=1e-5), "norm matching not applied"


@torch.no_grad()
def test_norm_none_does_not_rescale(stub):
    x = torch.randn(1, 5, 8)
    outs = []
    for k in (1.0, 10.0):
        rec = R.Recirculator(stub, "aat.1", "enc.1", alpha=0.5, norm="none")
        rec.set_state(torch.randn(1, 5, 8, generator=torch.Generator().manual_seed(3)) * k)
        outs.append(stub(x))
        rec.remove()
    assert not torch.allclose(outs[0], outs[1], atol=1e-5)


@torch.no_grad()
def test_shuffle_does_not_capture_own_stream(stub):
    """The control must inject only what the caller hands it; if it captured its own history
    it would silently become the `prev` arm and the control would be worthless."""
    x0, x1 = torch.randn(1, 5, 8), torch.randn(1, 5, 8)
    rec = R.Recirculator(stub, "aat.1", "enc.1", alpha=0.5, mode="shuffle")
    stub(x0)
    assert rec.state is None, "shuffle captured its own activation"
    foreign = torch.randn(1, 5, 8)
    rec.set_state(foreign)
    stub(x1)
    assert torch.allclose(rec.state, foreign, atol=1e-6), "shuffle state was overwritten"
    rec.remove()


@torch.no_grad()
def test_remove_restores_baseline(stub):
    x0, x1 = torch.randn(1, 5, 8), torch.randn(1, 5, 8)
    base1 = stub(x1)
    rec = R.Recirculator(stub, "aat.1", "enc.1", alpha=0.5)
    stub(x0)
    stub(x1)
    rec.remove()
    assert torch.allclose(base1, stub(x1), atol=1e-6)


@torch.no_grad()
def test_loop_gain_runs_and_is_positive(stub):
    x = torch.randn(1, 5, 8)
    g = R.loop_gain(stub, lambda: stub(x), "aat.1", "enc.1", alpha=0.15, iters=6)
    assert g["rho"] > 0
    assert g["G"] == pytest.approx(0.15 * g["rho"])


@torch.no_grad()
def test_center_zero_is_exactly_eq1(stub):
    """center=0.0 disables mean subtraction, leaving the paper's Eq. 1 untouched."""
    x0, x1 = torch.randn(1, 5, 8), torch.randn(1, 5, 8)
    rec = R.Recirculator(stub, "aat.1", "enc.1", alpha=0.3, center=0.0)
    stub(x0)
    a = stub(x1)
    rec.remove()
    rec = R.Recirculator(stub, "aat.1", "enc.1", alpha=0.3)
    stub(x0)
    b = stub(x1)
    rec.remove()
    assert torch.allclose(a, b, atol=1e-6)
    assert rec.mean_d is None and rec.mean_s is None


@torch.no_grad()
def test_centering_changes_the_injection(stub):
    """Centring must actually alter what is injected. Whether it RAISES the informative
    fraction is a property of the real residual stream, not of this stub: the stub's adjacent
    layers sit at cosine ~1 (blocks are x + 0.1*tanh), so raw injection is already fully
    informative here. That measurement belongs on the student and lives in
    eval/drone/diag_recirc_state.py.
    """
    torch.manual_seed(7)
    frames = [40.0 + 0.05 * torch.randn(1, 5, 8) for _ in range(4)]
    outs = []
    for c in (0.0, 0.9):
        rec = R.Recirculator(stub, "aat.1", "enc.1", alpha=0.3, center=c)
        for f in frames[:-1]:
            stub(f)
        outs.append(stub(frames[-1]))
        assert (rec.mean_d is not None) == (c > 0.0)
        rec.remove()
    assert not torch.allclose(outs[0], outs[1], atol=1e-6)


@torch.no_grad()
def test_lag_injects_from_n_steps_back(stub):
    """lag is a camera-baseline control: at the closed loop's 0.031 m per keyframe, lag=1 is the
    redundant regime and lag=3 lands on the ~0.09 m interval that actually helps."""
    frames = [torch.randn(1, 5, 8) for _ in range(6)]
    rec = R.Recirculator(stub, "aat.1", "enc.1", alpha=0.3, lag=3)
    seen = []
    for f in frames:
        stub(f)
        seen.append(None if rec.state is None else rec.state.clone())
    # nothing to inject until the buffer holds `lag` entries
    assert seen[0] is None and seen[1] is None
    assert seen[2] is not None
    assert rec.n_injected == len(frames) - 3, rec.n_injected
    rec.reset()
    assert rec.state is None and len(rec.buf) == 0
    rec.remove()


@torch.no_grad()
def test_lag_one_matches_default(stub):
    x = [torch.randn(1, 5, 8) for _ in range(3)]
    outs = []
    for lag in (1, None):
        rec = (R.Recirculator(stub, "aat.1", "enc.1", alpha=0.3, lag=1) if lag
               else R.Recirculator(stub, "aat.1", "enc.1", alpha=0.3))
        for f in x[:-1]:
            stub(f)
        outs.append(stub(x[-1]))
        rec.remove()
    assert torch.allclose(outs[0], outs[1], atol=1e-6)


@torch.no_grad()
def test_loop_gain_handles_mismatched_token_counts():
    """The two stacks do not share a token count (the AAT carries a scale token the encoder
    does not). loop_gain must slice like the injection does; it previously built the iterate in
    the SOURCE's space and crashed on the real model with 730 vs 731."""
    class Ragged(Stub):
        def forward(self, x):
            for b in self.encoder.model.blocks:
                x = b(x)
            x = torch.cat([x, x[:, :1]], dim=1)   # AAT adds a token
            for b in self.info_sharing.self_attention_blocks:
                x = b(x)
            return x

    torch.manual_seed(0)
    m = Ragged().eval()
    x = torch.randn(1, 5, 8)
    g = R.loop_gain(m, lambda: m(x), "aat.1", "enc.1", alpha=0.15, iters=5)
    assert g["rho"] >= 0 and np.isfinite(g["rho"])
