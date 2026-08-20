#!/usr/bin/env python3
"""G0.5: does ORE actually keep its latency promise on the Jetson? Synthetic, dummy weights.

The sizing table (size_moe.py) claims ~24 FPS at K=512 by assuming top-1 dispatch is free. That
assumption is the whole method, it is unmeasured, and every way it can be wrong flatters us. So
measure it before building a training pipeline on top of it.

Measured per AAT block at deployed shapes (d=768 -> F=3072 -> d=768, 729 tokens):
  dense        the frozen FFN alone. The floor everything is compared against.
  ore          shared frozen base + top-1 grouped-GEMM low-rank residual (+ optional butterfly)
  gate         the router in isolation, to attribute overhead
  ders_dense   DeRS as implemented today: all K*r computed then masked (total == active)

The low-rank residual reuses triton_moe._grouped_gemm unchanged: x@A is (M,d)x(E,d,r) and z@B is
(M,r)x(E,r,F), which is the same kernel with different N/K. Sorting tokens by expert is what
makes the small per-expert GEMMs contiguous.
"""
import argparse
import json
import math

import torch
import torch.nn as nn
import triton

from triton_moe import _grouped_gemm, align_block_size, padded_capacity

D, F, N_TOK = 768, 3072, 729


class Butterfly(nn.Module):
    """n_l layers of paired Givens rotations, per expert, applied to activations.

    Cost is O(n_l * dim) per token and params are n_l * dim/2 angles per expert -- the point is
    that this re-orients the FULL-RANK shared substrate for the price of a rank-n_l residual.
    Strided pairing (stride 2^l) is the standard butterfly connectivity.
    """

    def __init__(self, k, dim, n_l, device, dtype):
        super().__init__()
        self.dim, self.n_l = dim, n_l
        self.ang = nn.Parameter(torch.zeros(k, n_l, dim // 2, device=device, dtype=dtype))

    @torch.no_grad()
    def freeze_trig(self):
        """Precompute cos/sin per EXPERT. Angles are frozen at inference, so recomputing
        transcendentals for every token of every frame is pure waste -- and it is 4.04 ms/block on
        the Orin Nano, 4.5x the entire dense FFN this decorates. Cached, the forward is a gather
        plus two multiplies."""
        self._cos, self._sin = torch.cos(self.ang), torch.sin(self.ang)
        return self

    def forward(self, x, eidx):
        # x (M, dim), eidx (M,). Layer l pairs element i with i+s at stride s = 2^l; viewing as
        # (M, dim/(2s), 2, s) makes those pairs adjacent on axis 2. nb*s == dim/2 always, so the
        # dim/2 angles reshape onto the pair grid exactly.
        cached = getattr(self, "_cos", None) is not None
        a = None if cached else self.ang[eidx]                  # (M, n_l, dim/2)
        for l in range(self.n_l):
            s = 1 << l
            nb = self.dim // (2 * s)
            v = x.reshape(-1, nb, 2, s)
            if cached:
                c = self._cos[eidx, l].reshape(-1, nb, s)
                sn = self._sin[eidx, l].reshape(-1, nb, s)
            else:
                ang = a[:, l].reshape(-1, nb, s)
                c, sn = torch.cos(ang), torch.sin(ang)
            x0, x1 = v[:, :, 0], v[:, :, 1]                     # (M, nb, s)
            x = torch.stack([x0 * c - x1 * sn, x0 * sn + x1 * c], 2).reshape(-1, self.dim)
        return x


_ZB = {}


def _zero_bias(e, n, device, dtype):
    """Cached. Allocating+zeroing a (K,N) bias every forward was 4 allocations per block and
    showed up as a flat ~1.5 ms that looked like dispatch overhead."""
    key = (e, n, str(device), dtype)
    if key not in _ZB:
        _ZB[key] = torch.zeros(e, n, device=device, dtype=dtype)
    return _ZB[key]


def grouped(a, w, sorted_tok, blk_e, M, N, Kd, bm, gather, scatter, out):
    grid = lambda meta: ((sorted_tok.numel() // bm) * triton.cdiv(N, meta["BLOCK_N"]),)
    _grouped_gemm[grid](
        a, w, _zero_bias(w.shape[0], N, w.device, w.dtype), out,
        sorted_tok, blk_e, M, N, Kd,
        a.stride(0), a.stride(1), w.stride(0), w.stride(1), w.stride(2),
        out.stride(0), out.stride(1), N,
        GATHER_A=gather, SCATTER_C=scatter, ACT_GELU=False, BLOCK_M=bm)
    return out


class ORESite(nn.Module):
    """One linear site: frozen shared base + top-1 low-rank residual (+ optional butterfly)."""

    def __init__(self, k, d_in, d_out, r, n_l, block_m=64, device="cuda", dtype=torch.float16):
        super().__init__()
        self.k, self.d_in, self.d_out, self.r, self.bm = k, d_in, d_out, r, block_m
        self.base = nn.Linear(d_in, d_out, device=device, dtype=dtype)
        self.gate = nn.Linear(d_in, k, device=device, dtype=dtype)
        self.A = nn.Parameter(torch.randn(k, d_in, r, device=device, dtype=dtype) * 0.02)
        self.B = nn.Parameter(torch.zeros(k, r, d_out, device=device, dtype=dtype))
        self.bfly = Butterfly(k, d_in, n_l, device, dtype) if n_l else None
        self.active_experts = 0

    def forward(self, x):
        m = x.shape[0]
        # A random gate over random activations spreads 729 tokens across ALL K experts, so every
        # expert gets a padded block and the pad blows up as K grows (12.1x at K=128). Real
        # routing does not do that: measured on this model, ONE FRAME touches only ~14 distinct
        # experts of 128, because the AAT token cloud has effective rank ~9 and 75% of its
        # variance is within-frame (diag_frame.py, 2026-08-10). active_experts models that -- K
        # experts stay resident, a frame touches n of them.
        lg = self.gate(x)
        if self.active_experts:
            lg = lg[:, :self.active_experts]
        idx = lg.argmax(-1).to(torch.int32)
        ns = padded_capacity(m, self.k, self.bm)
        st, be = align_block_size(idx, self.k, self.bm, ns)
        z = torch.empty((ns, self.r), device=x.device, dtype=x.dtype)
        y = torch.empty((m, self.d_out), device=x.device, dtype=x.dtype)
        xin = self.bfly(x, idx.long()) if self.bfly is not None else x
        grouped(xin, self.A, st, be, m, self.r, self.d_in, self.bm, True, False, z)
        grouped(z, self.B, st, be, m, self.d_out, self.r, self.bm, False, True, y)
        return self.base(x) + y


class DeRSDense(nn.Module):
    """DeRS as implemented today: fused A/B over all K*r, masked. total == active."""

    def __init__(self, k, d_in, d_out, r, device="cuda", dtype=torch.float16):
        super().__init__()
        kr = k * r
        self.k, self.r = k, r
        self.base = nn.Linear(d_in, d_out, device=device, dtype=dtype)
        self.gate = nn.Linear(d_in, k, device=device, dtype=dtype)
        self.A = nn.Linear(d_in, kr, bias=False, device=device, dtype=dtype)
        self.B = nn.Linear(kr, d_out, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        g = self.gate(x).softmax(-1)
        z = self.A(x)
        g = g.repeat_interleave(self.r, dim=-1)
        return self.base(x) + self.B(z * g)


def bench(fn, iters=30, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[32, 128, 512])
    ap.add_argument("--ranks", type=int, nargs="+", default=[8, 32])
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--block-m", type=int, nargs="+", default=[64])
    ap.add_argument("--active-experts", type=int, default=0,
                    help="distinct experts a single frame touches (0 = all K). Measured ~14 of "
                         "128 on real activations; the default spreads over all K and is "
                         "pessimistic by the padding factor.")
    ap.add_argument("--freeze-trig", action="store_true",
                    help="precompute the butterfly cos/sin (valid at inference: angles frozen)")
    ap.add_argument("--blocks", type=int, default=8, help="AAT blocks, to scale per-block -> model")
    ap.add_argument("--out", default="/home/evc/merlin/ore_latency.json")
    args = ap.parse_args()

    dev, dt = "cuda", torch.float16
    x = torch.randn(N_TOK, D, device=dev, dtype=dt)
    h = torch.randn(N_TOK, F, device=dev, dtype=dt)

    fc1 = nn.Linear(D, F, device=dev, dtype=dt)
    fc2 = nn.Linear(F, D, device=dev, dtype=dt)
    with torch.no_grad():
        dense_ms = bench(lambda: fc2(torch.nn.functional.gelu(fc1(x))))
    print(f"shapes {D}->{F}->{D}, {N_TOK} tokens, fp16, {args.blocks} blocks\n")
    print(f"dense FFN block: {dense_ms:.3f} ms  ->  model {args.blocks * dense_ms:.1f} ms\n")

    print(f"{'K':>5} {'r':>4} {'variant':>12} {'ms/blk':>8} {'vs dense':>9} "
          f"{'gate ms':>8} {'model ms':>9} {'FPS*':>7}")
    res = {"dense_ms": dense_ms, "blocks": args.blocks, "cells": {}}
    for r in args.ranks:
        for k in args.ks:
            with torch.no_grad():
                bmv = args.block_m[0] if len(args.block_m)==1 else None
                gate_ms = float("nan"); ore_ms = float("inf"); best_bm = None
                for bm in args.block_m:
                    s1 = ORESite(k, D, F, r, args.n_layers, block_m=bm, device=dev, dtype=dt)
                    s2 = ORESite(k, F, D, r, args.n_layers, block_m=bm, device=dev, dtype=dt)
                    ae = min(args.active_experts, k) if args.active_experts else 0
                    s1.active_experts = s2.active_experts = ae
                    if args.freeze_trig:
                        for st in (s1, s2):
                            if st.bfly is not None:
                                st.bfly.freeze_trig()
                    if gate_ms != gate_ms:
                        gate_ms = bench(lambda: (s1.gate(x).argmax(-1), s2.gate(h).argmax(-1)))
                    try:
                        t = bench(lambda: s2(torch.nn.functional.gelu(s1(x))))
                        slots = padded_capacity(N_TOK, k, bm)
                        print(f"{k:>5} {r:>4} {'ore bm=%d'%bm:>12} {t:>8.3f} {t/dense_ms:>8.2f}x "
                              f"{gate_ms:>8.3f}  slots={slots} ({slots/N_TOK:.1f}x pad)")
                        if t < ore_ms: ore_ms, best_bm = t, bm
                    except Exception as e:
                        print(f"{k:>5} {r:>4} bm={bm} FAILED {type(e).__name__}: {str(e)[:60]}")
                    del s1, s2; torch.cuda.empty_cache()
                try:
                    d1 = DeRSDense(k, D, F, r, device=dev, dtype=dt)
                    d2 = DeRSDense(k, F, D, r, device=dev, dtype=dt)
                    ders_ms = bench(lambda: d2(torch.nn.functional.gelu(d1(x))))
                    del d1, d2
                except Exception:
                    ders_ms = float("nan")
                torch.cuda.empty_cache()

            for name, ms in (("ore", ore_ms), ("ders_dense", ders_ms)):
                if ms != ms:
                    continue
                model_ms = args.blocks * ms + (169.49 - args.blocks * dense_ms)
                print(f"{k:>5} {r:>4} {name:>12} {ms:>8.3f} {ms/dense_ms:>8.2f}x "
                      f"{gate_ms:>8.3f} {model_ms:>9.1f} {1000/model_ms:>7.2f}")
                res["cells"][f"{name}_k{k}_r{r}"] = {
                    "k": k, "r": r, "variant": name, "ms_block": ms,
                    "vs_dense": ms / dense_ms, "gate_ms": gate_ms,
                    "model_ms_fp16": model_ms, "fps_fp16": 1000 / model_ms}
            print()

    json.dump(res, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")
    print("* FPS is fp16 PyTorch on the WHOLE model (block cost scaled to 8 blocks + the measured"
          "\n  169.49 ms base minus its FFN share). INT8-TRT measured 4.17x faster on the dense model.")


if __name__ == "__main__":
    main()
