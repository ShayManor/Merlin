"""Correctness gate for the fused inference path (`enable_fast_inference`).

The fused path rewrites routing into three kernels and folds the bias, the p_sel scaling, the
residual add and the tail concatenation into the two rotation kernels. None of that is allowed
to change the function. It is measured on latency, so it has every incentive to be subtly wrong
and still look good, which is what these tests exist to stop.

Slot order inside an expert is deliberately non-deterministic (the placement kernel claims slots
with an atomic increment), so the tests assert on the OUTPUT, which is scattered back by token
id and must be order-independent.
"""
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from moe.ore import OREResidual, enable_fast_inference, set_route_feature  # noqa: E402

D_IN, D_OUT, M = 768, 3072, 729
CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="the fused path is triton/CUDA only")


def _site(k, r, n_l=2, route_dim=0, bm=16, d_in=D_IN, d_out=D_OUT):
    torch.manual_seed(0)
    base = nn.Linear(d_in, d_out, device="cuda", dtype=torch.float16)
    ore = OREResidual(base, num_experts=k, rank=r, n_layers=n_l, block_m=bm,
                      route_dim=route_dim).cuda().eval()
    with torch.no_grad():                       # break the function-preserving init everywhere
        ore.B.normal_(0, 0.02)
        ore.A.normal_(0, 0.02)
        ore.theta.angle.normal_(0, 0.4)
        ore.phi.angle.normal_(0, 0.4)
        ore.gate.weight.normal_(0, 0.05)
    return ore


def _rel(a, b):
    return float((a.float() - b.float()).abs().max() / b.float().abs().max())


def _fp16_tie(ore, rin):
    """Tokens whose top two gate logits are EQUAL once rounded to fp16.

    The eager router runs the gate through an fp16 Linear, so its argmax on such a token is
    arbitrary; the fused router accumulates in fp32 and resolves the tie by the true ordering.
    That is a difference in the reference, not in the path under test, and it must be excluded
    from the comparison rather than absorbed into a loose tolerance -- a token that flips expert
    changes its whole row, so one tie would otherwise hide any real bug behind a 0.95 rel error.
    """
    with torch.no_grad():
        top2 = ore.gate(rin).float().topk(2, dim=-1).values
    return (top2[:, 0] - top2[:, 1]) == 0


def _compare(ore, x, rin=None):
    """Eager vs fused on the same input, ignoring rows the reference cannot resolve."""
    rin = x.reshape(-1, ore.d_in) if rin is None else rin.reshape(-1, ore.gate.in_features)
    with torch.no_grad():
        ref = ore(x).reshape(-1, ore.d_out)
        assert enable_fast_inference(ore), "config was silently rejected by can_fast()"
        got = ore(x).reshape(-1, ore.d_out)
    tie = _fp16_tie(ore, rin)
    assert tie.float().mean() < 0.03, f"{int(tie.sum())} of {tie.numel()} tokens tie in fp16 -- " \
                                      "the comparison would be vacuous"
    keep = ~tie
    return _rel(got[keep], ref[keep]), int(tie.sum())


@pytest.mark.parametrize("k,r", [(32, 128), (64, 128), (128, 32)])
def test_fast_equals_eager(k, r):
    """THE test: the fused path computes the same function as the module it replaces."""
    ore = _site(k, r)
    x = torch.randn(1, M, D_IN, device="cuda", dtype=torch.float16)
    err, ties = _compare(ore, x)
    assert err < 3e-3, f"rel {err} on non-tie tokens ({ties} ties excluded)"


def test_fast_matches_with_frozen_feature_routing():
    """Deployed configuration: every site gates on one broadcast 768-d feature."""
    ore = _site(32, 128, route_dim=768)
    x = torch.randn(1, M, D_IN, device="cuda", dtype=torch.float16)
    feat = torch.randn(1, M, 768, device="cuda", dtype=torch.float16)
    set_route_feature(nn.ModuleList([ore]), feat)
    err, ties = _compare(ore, x, rin=feat)
    assert err < 3e-3, f"rel {err} on non-tie tokens ({ties} ties excluded)"


def test_only_fp16_ties_route_differently():
    """The fused router may disagree with the eager one ONLY where fp16 cannot order the logits.
    Any other disagreement is a routing bug, and routing bugs are invisible in a mean error."""
    ore = _site(32, 128)
    x = torch.randn(M, D_IN, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        idx_ref, _ = ore._route(x)
        enable_fast_inference(ore)
        ore(x)
    flips = ore._fbuf["idx"].long() != idx_ref
    assert not (flips & ~_fp16_tie(ore, x)).any(), \
        f"{int((flips & ~_fp16_tie(ore, x)).sum())} tokens re-routed without an fp16 tie"


def test_fast_routes_to_the_same_experts():
    """The fused router must select the same expert per token, not merely a similar output."""
    ore = _site(64, 32)
    x = torch.randn(M, D_IN, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        idx_ref, p_ref = ore._route(x)
        enable_fast_inference(ore)
        ore(x)
    buf = ore._fbuf
    keep = ~_fp16_tie(ore, x)
    assert torch.equal(buf["idx"].long()[keep], idx_ref[keep]), \
        (buf["idx"].long()[keep] != idx_ref[keep]).sum().item()
    assert torch.allclose(buf["psel"].float()[keep], p_ref.float()[keep], atol=2e-3)


def test_sorted_slots_are_a_permutation_of_the_tokens():
    """Every token appears exactly once; every non-padding block holds one expert only."""
    ore = _site(32, 32)
    x = torch.randn(M, D_IN, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        enable_fast_inference(ore)
        ore(x)
    buf = ore._fbuf
    stok, be, idx = buf["stok"], buf["be"], buf["idx"]
    live = stok[stok < M].long()
    assert torch.equal(live.sort().values, torch.arange(M, device=live.device))
    slots = stok.reshape(-1, ore.bm)
    for b in range(slots.shape[0]):
        row = slots[b][slots[b] < M].long()
        if row.numel():
            assert (idx[row] == be[b]).all(), f"block {b} mixes experts"


def test_histogram_is_left_zeroed_for_the_next_frame():
    """The prefix kernel zeroes HIST on the read; if it did not, frame 2 would mis-size every
    expert region and the second call would differ from the first."""
    ore = _site(32, 64)
    x = torch.randn(M, D_IN, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        ref = ore(x)
        enable_fast_inference(ore)
        first = ore(x)
        second = ore(x)
    keep = ~_fp16_tie(ore, x)
    assert int(ore._fbuf["hist"].sum()) == 0
    assert torch.equal(first, second)
    assert _rel(first[keep], ref[keep]) < 3e-3


def test_fast_survives_cuda_graph_capture_and_tracks_new_routing():
    """Deployment replays a captured graph, so the path must be allocation- and sync-free, and
    a replay must re-route rather than reuse the captured frame's partition."""
    ore = _site(32, 128)
    x = torch.randn(M, D_IN, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        enable_fast_inference(ore)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                ore(x)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = ore(x)
        x.copy_(torch.randn_like(x))            # new frame, new routing
        g.replay()
        torch.cuda.synchronize()
        ore.fast = False
        ref = ore(x)
    keep = ~_fp16_tie(ore, x)
    assert _rel(out[keep], ref[keep]) < 3e-3, _rel(out[keep], ref[keep])


def test_can_fast_rejects_what_it_does_not_implement():
    assert not _site(8, 16, n_l=0).can_fast(), "n_l=0 has no fused rotation kernel"
    ore = _site(8, 16)
    ore.balance = "bias"
    assert not ore.can_fast(), "sel_bias selection is not implemented in the fused router"
