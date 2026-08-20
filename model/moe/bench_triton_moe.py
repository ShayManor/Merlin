#!/usr/bin/env python3
"""Fused grouped-GEMM top-1 MoE vs the gather-scatter loop, at the AAT FFN site.

Correctness first: the fused kernel is checked against MoETop1 on identical weights before any
timing is reported. A faster kernel that computes something else is not a result.

Baselines are the same three bench_moe_aat.py measures, so numbers here drop straight into the
existing moe_aat.json comparison.
"""
import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from triton_moe import FusedMoETop1

N, D, DH = 729, 768, 3072


class DenseFFN(nn.Module):
    # built directly on device in the target dtype: materialising K experts in fp32 on the host
    # first costs 2.4 GB at K=64, and on a shared 8 GB pool that perturbs every later timing
    def __init__(s, **kw):
        super().__init__(); s.w1 = nn.Linear(D, DH, **kw); s.w2 = nn.Linear(DH, D, **kw)
    def forward(s, x): return s.w2(F.gelu(s.w1(x)))


class MoETop1(nn.Module):
    """The existing path: gather tokens per expert, one GEMM each, scatter back."""
    def __init__(s, k, **kw):
        super().__init__(); s.k = k
        s.e = nn.ModuleList([DenseFFN(**kw) for _ in range(k)]); s.g = nn.Linear(D, k, **kw)
    def forward(s, x):
        B, n, _ = x.shape
        flat = x.reshape(-1, D)
        idx = s.g(flat).argmax(-1)
        out = torch.empty_like(flat)
        for i in range(s.k):
            sel = (idx == i).nonzero(as_tuple=True)[0]
            if sel.numel():
                out[sel] = s.e[i](flat[sel])
        return out.reshape(B, n, D)


@torch.no_grad()
def bench(m, x, iters=50, warmup=10):
    for _ in range(warmup): m(x)
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters): m(x)
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


@torch.no_grad()
def check(ref, fused, x, tol):
    """Same weights, same routing => same output. Returns (max_abs_err, routing_agreement)."""
    r, f = ref(x), fused(x)
    flat = x.reshape(-1, D)
    agree = (ref.g(flat).argmax(-1) == fused.gate(flat).argmax(-1)).float().mean().item()
    err = (r - f).abs().max().item()
    scale = r.abs().max().item()
    return err, scale, agree, err <= tol * max(scale, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--tol", type=float, default=2e-2)   # fp16 accumulate over 3072 terms
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="/home/evc/merlin/triton_moe.json")
    args = ap.parse_args()

    dt, dev = torch.float16, "cuda"
    torch.manual_seed(args.seed)
    x = torch.randn(1, N, D, device=dev, dtype=dt)

    kw = {"device": dev, "dtype": dt}
    dense = DenseFFN(**kw).eval()
    dense_ms = bench(dense, x, args.iters)
    print(f"site: Linear {D}->{DH}->{D}, {N} tokens, fp16, dense = {dense_ms:.3f} ms\n")

    def med(vals):
        return sorted(vals)[len(vals) // 2]

    print(f"{'K':>4} {'act':>6} {'ref ms':>9} {'fused ms':>10} {'speedup':>8} "
          f"{'roofline':>9} {'off':>6} {'err':>8} {'route':>7} {'ok':>5}")
    res = {"dense": {"ms": dense_ms}, "seed": args.seed}
    for k in args.ks:
        refs, fuses = [], []
        for rep in range(args.reps):
            # identical gate every rep and across both arms: how many distinct experts the
            # routing activates decides how much weight is read, so it must not drift
            torch.manual_seed(args.seed)
            ref = MoETop1(k, **kw).eval()
            fused = FusedMoETop1(k, D, DH, device=dev, dtype=dt).eval().load_from(ref.e, ref.g)
            if rep == 0:
                err, scale, agree, ok = check(ref, fused, x, args.tol)
                counts = torch.bincount(ref.g(x.reshape(-1, D)).argmax(-1), minlength=k)
                active = int((counts > 0).sum())
            refs.append(bench(ref, x, args.iters))
            # ref holds K full experts; free it so its footprint cannot perturb fused timing
            del ref
            torch.cuda.empty_cache()
            fuses.append(bench(fused, x, args.iters))
            del fused
            torch.cuda.empty_cache()

        ref_ms, fus_ms = med(refs), med(fuses)
        roof = k * 2 * D * DH * 2 / 90e9 * 1e3      # both weight matrices at 90 GB/s measured
        print(f"{k:>4} {active:>3}/{k:<2} {ref_ms:>9.3f} {fus_ms:>10.3f} {ref_ms/fus_ms:>7.2f}x "
              f"{roof:>9.3f} {fus_ms/roof:>5.1f}x {err:>8.4f} {agree:>6.1%} "
              f"{'PASS' if ok else 'FAIL':>5}")
        res[f"k{k}"] = {"k": k, "ref_ms": ref_ms, "fused_ms": fus_ms,
                        "ref_all": refs, "fused_all": fuses,
                        "speedup": ref_ms / fus_ms, "vs_dense": fus_ms / dense_ms,
                        "roofline_ms": roof, "off_roofline": fus_ms / roof,
                        "active_experts": active,
                        "max_err": err, "out_scale": scale, "route_agree": agree, "ok": bool(ok)}

    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
