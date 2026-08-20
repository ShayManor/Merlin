"""Correctness gate for ORE (G1). Nothing trains until these pass.

The three properties that make the method what it claims to be:
  1. function-preserving init -- wrapping must not change the network at step 0
  2. sparse dispatch == dense-masked reference -- the fast path computes the stated function
  3. N > 1 by construction -- a config that is secretly dense should fail loudly, not after a
     six-hour training run
"""
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from moe.ore import OREResidual, ore_parameters  # noqa: E402

D_IN, D_OUT, M = 768, 3072, 729
CUDA = torch.cuda.is_available()


def _base(dtype=torch.float32, device="cpu"):
    torch.manual_seed(0)
    return nn.Linear(D_IN, D_OUT, device=device, dtype=dtype)


def test_function_preserving_at_init():
    """Theta = Phi = I and B = 0, so the wrapped module IS its dense parent at step 0."""
    base = _base()
    x = torch.randn(4, 32, D_IN)
    ref = base(x)
    ore = OREResidual(base, num_experts=8, rank=16, n_layers=2, b_init="zero")
    out = ore(x, sparse=False)
    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=1e-5), (out - ref).abs().max()


def test_rotation_is_orthogonal_after_training_angles():
    """A butterfly with non-zero angles must preserve norm -- that is the spectral claim."""
    base = _base()
    ore = OREResidual(base, num_experts=4, rank=8, n_layers=2)
    with torch.no_grad():
        ore.theta.angle.normal_(0, 0.5)
    x = torch.randn(64, ore.rot_in_dim)
    idx = torch.randint(0, 4, (64,))
    y = ore.theta(x, idx)
    assert torch.allclose(x.norm(dim=1), y.norm(dim=1), atol=1e-4), \
        "butterfly is not norm-preserving; the 'same singular values' argument fails"


@pytest.mark.parametrize("k,r,n_l", [(8, 16, 2), (32, 32, 2), (16, 8, 0)])
def test_param_accounting(k, r, n_l):
    base = _base()
    ore = OREResidual(base, num_experts=k, rank=r, n_layers=n_l)
    rot = n_l * (ore.rot_in_dim // 2 + ore.rot_out_dim // 2)
    assert ore.expert_params() == rot + r * (D_IN + D_OUT)
    # every expert stored, exactly one computed
    assert ore.total_params() - ore.active_params() == (k - 1) * ore.expert_params()
    assert ore.N() > 1.0, "config is secretly dense"


def test_degenerate_cases_are_the_baselines():
    """n_l=0 is DeRS, r=0 is ButterflyViT, K=1 is LoRA -- all reachable from one implementation."""
    base = _base()
    ders = OREResidual(base, num_experts=8, rank=16, n_layers=0)
    assert ders.expert_params() == 16 * (D_IN + D_OUT)
    bfly = OREResidual(_base(), num_experts=8, rank=0, n_layers=2)
    assert bfly.expert_params() == 2 * (bfly.rot_in_dim // 2 + bfly.rot_out_dim // 2)
    lora = OREResidual(_base(), num_experts=1, rank=16, n_layers=2)
    assert lora.N() == 1.0, "K=1 must be exactly dense-active (it is LoRA)"


def test_trainables_exclude_the_substrate():
    base = _base()
    ore = OREResidual(base, num_experts=8, rank=16, n_layers=2).freeze_base()
    ids = {id(p) for p in ore_parameters(ore)}
    assert id(ore.base.weight) not in ids
    assert not ore.base.weight.requires_grad


@pytest.mark.skipif(not CUDA, reason="sparse dispatch needs CUDA + triton")
@pytest.mark.parametrize("k,r", [(8, 32), (64, 32), (128, 16)])
def test_sparse_equals_dense_reference(k, r):
    """THE test: the fast path must compute the same function as the masked reference."""
    torch.manual_seed(0)
    base = _base(torch.float16, "cuda")
    ore = OREResidual(base, num_experts=k, rank=r, n_layers=2).cuda()
    with torch.no_grad():                       # break the identity init so B != 0
        ore.B.normal_(0, 0.02)
        ore.theta.angle.normal_(0, 0.3)
        ore.phi.angle.normal_(0, 0.3)
    x = torch.randn(M, D_IN, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        a = ore(x, sparse=False)                       # gather path (training)
        b = ore(x, sparse=True)                        # triton path (inference)
        idx, _ = ore._route(x)
        c = ore._residual_dense(x, idx)                # masked reference
        d = ore._residual_gather(x, idx)
    err = (a - b).abs().max().item()
    scale = a.abs().max().item()
    assert err <= 2e-2 * max(scale, 1.0), f"sparse != gather: max err {err} vs scale {scale}"
    # all THREE residual paths must agree, or a fast path is computing a different function
    e2 = (c - d).abs().max().item()
    assert e2 <= 2e-2 * max(c.abs().max().item(), 1.0), f"masked-reference != gather: {e2}"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_training_path_gives_A_and_B_gradient():
    """THE regression test for the bug that voided the first 9000-step run.

    Triton kernels have no autograd backward, so routing training through the sparse path
    detaches A and B: .grad comes back None, 1.0B of 1.04B trainable params never move, and the
    run reports a plausible null rather than an error. forward() must pick the dense path
    whenever grad is enabled.
    """
    base = _base(torch.float32, "cuda")
    ore = OREResidual(base, num_experts=32, rank=16, n_layers=2).cuda().freeze_base()
    x = torch.randn(64, D_IN, device="cuda")
    ore(x).sum().backward()                      # no sparse= override: must choose the safe path
    for name in ("A", "B"):
        g = getattr(ore, name).grad
        assert g is not None, f"{name}.grad is None -- training path is detached"
        assert g.abs().sum() > 0, f"{name}.grad is all zero"
    assert ore.theta.angle.grad is not None
    assert ore.gate.weight.grad is not None, "gate must get signal; B=0 init kills it"


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_inference_still_uses_sparse():
    base = _base(torch.float16, "cuda")
    ore = OREResidual(base, num_experts=32, rank=16, n_layers=2).cuda()
    x = torch.randn(128, D_IN, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        a = ore(x)                                # should take the sparse path
        b = ore(x, sparse=False)
    assert (a - b).abs().max().item() <= 2e-2 * max(a.abs().max().item(), 1.0)
