#!/usr/bin/env python3
"""Per-frame latency of the REAL ORE block on the Jetson, and where the time goes.

`fullgraph.py` measured a simplification: one butterfly per site, applied to the residual input.
The deployed module (`ore.OREResidual`) applies TWO butterflies per site -- Theta on the input of
the frozen substrate and Phi on its output -- so it moves 5120 rotated elements per token per
block against fullgraph's 3840, and it carries the training-time router bookkeeping. This bench
drives the real module so the numbers transfer to deployment.

One AAT block = two sites, 768 -> 3072 -> 768, 729 tokens, fp16, frozen-feature routing.
Reported per block and scaled to the model: the deployed INT8-TRT dense student is 49.66 ms of
which the 8 FFN blocks are 2.10 ms, so model_ms = 47.56 + 8 * block_ms.
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from moe import ore as _ore                                          # noqa: E402
from moe.ore import OREResidual, enable_fast_inference, set_route_feature   # noqa: E402

# ore imports its kernels relatively and falls back to a pure-torch path on ImportError, so a
# broken import measures the reference implementation and reports it as the deployed one.
assert _ore._HAVE_TRITON, "triton path not live -- this would benchmark the reference path"

D, F, N_TOK = 768, 3072, 729
BASE_MS, DENSE_FFN_INT8_MS = 49.66, 2.10          # measured INT8-TRT dense student


def bench(fn, iters=50, warm=15):
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


def graphed(fn, warm=3):
    """Capture fn into a CUDA graph and return a replay callable.

    The ORE block is ~30 kernels over small tensors, so eager launch latency dominates on a Nano
    (60-120 us per launch). Deployment replays a captured graph; benching eager measures the
    harness, not the method.
    """
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warm):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    return g.replay, out


def build_block(k, r, n_l, bm, route_dim, active, dev, dt):
    """Two ORE sites wrapping the AAT FFN, routed on one frozen feature.

    `active` caps how many experts a frame can reach. Real routing touches ~14 distinct experts
    of 128 in one frame (diag_frame.py); a random gate over random activations spreads tokens
    across all K and over-counts weight traffic by the padding factor.
    """
    s1 = OREResidual(nn.Linear(D, F, device=dev, dtype=dt), num_experts=k, rank=r,
                     n_layers=n_l, block_m=bm, route_dim=route_dim).to(dev)
    s2 = OREResidual(nn.Linear(F, D, device=dev, dtype=dt), num_experts=k, rank=r,
                     n_layers=n_l, block_m=bm, route_dim=route_dim).to(dev)
    for s in (s1, s2):
        with torch.no_grad():
            s.gate.bias[active:] = -1e4                      # cap the reachable expert set
            s.theta.angle.normal_(0, 0.3)
            s.phi.angle.normal_(0, 0.3)
            s.A.normal_(0, 0.02)
            s.B.normal_(0, 0.02)
        s.eval()
    return s1, s2


def decompose(blk, label):
    """Per-kernel GPU time for one block, from the profiler rather than from stage-wise graphs.

    Capturing one CUDA graph per stage is the obvious way to attribute time, and it took the
    Jetson down twice: each CUDAGraph pins a private memory pool, and a dozen of them alive at
    once exhausts an 8 GB unified-memory board, which reboots rather than raising. The profiler
    reports each kernel's own GPU duration, so summing it gives the same launch-free attribution
    from a single eager pass and no extra memory at all.
    """
    from torch.profiler import ProfilerActivity, profile                 # noqa: PLC0415

    with torch.no_grad():
        for _ in range(5):
            blk()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(20):
                blk()
            torch.cuda.synchronize()
    per = {}
    for e in prof.key_averages():
        if e.self_device_time_total > 0:
            per[e.key] = per.get(e.key, 0.0) + e.self_device_time_total / 20 / 1000
    tot = sum(per.values())
    print(f"\n{label}: {tot:.3f} ms/block over {len(per)} distinct kernels")
    print(f"{'kernel':>52} {'ms':>8} {'share':>7}")
    for name, ms in sorted(per.items(), key=lambda kv: -kv[1])[:14]:
        print(f"{name[:52]:>52} {ms:>8.3f} {100*ms/tot:>6.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--block-m", type=int, nargs="+", default=[16])
    ap.add_argument("--route-dim", type=int, default=768)
    ap.add_argument("--active-experts", type=int, default=14)
    ap.add_argument("--eager", action="store_true", help="also time the un-captured path")
    ap.add_argument("--decompose", action="store_true", help="per-stage attribution, first K only")
    ap.add_argument("--no-fuse-epilogue", action="store_true",
                    help="keep the oriented epilogue as its own kernel instead of folding it "
                         "into the residual GEMM")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dev, dt = "cuda", torch.float16
    torch.manual_seed(0)
    x = torch.randn(N_TOK, D, device=dev, dtype=dt)
    rfeat = torch.randn(N_TOK, args.route_dim, device=dev, dtype=dt)

    fc1 = nn.Linear(D, F, device=dev, dtype=dt)
    fc2 = nn.Linear(F, D, device=dev, dtype=dt)
    with torch.no_grad():
        dense = bench(lambda: fc2(torch.nn.functional.gelu(fc1(x))))
    print(f"729 tokens, {D}->{F}->{D}, fp16, r={args.rank} n_l={args.n_layers} "
          f"bm={args.block_m} active={args.active_experts}")
    print(f"dense FFN block (the substrate alone): {dense:.3f} ms\n")
    print(f"{'K':>5} {'bm':>4} {'path':>6} {'eager':>9} {'graph':>9} {'vs dense':>9} {'block x8':>9} "
          f"{'model ms':>9} {'FPS':>6} {'rel err':>9}")

    res = {"dense_ms": dense, "cells": {}}
    for k, bm in [(k, bm) for k in args.ks for bm in args.block_m]:
        s1, s2 = build_block(k, args.rank, args.n_layers, bm,
                             args.route_dim, min(args.active_experts, k), dev, dt)
        sites = nn.ModuleList([s1, s2])
        set_route_feature(sites, rfeat)
        for s in sites:
            s.fuse_epilogue = not args.no_fuse_epilogue

        def blk():
            with torch.no_grad():
                return s2(torch.nn.functional.gelu(s1(x)))

        ref = None
        for path in ("base", "fast"):
            if path == "fast" and not enable_fast_inference(sites):
                print(f"{k:>5} {bm:>4} {'fast':>6}   config rejected by can_fast()")
                continue
            with torch.no_grad():
                out = blk()
                # gate every reported number on the function being unchanged; the fused path is
                # judged on latency, so a silently wrong one would look like a win
                err = float("nan") if ref is None else float(
                    (out.float() - ref.float()).abs().max() / ref.float().abs().max())
                ref = out.clone() if ref is None else ref
                eager = bench(blk) if args.eager else float("nan")
                replay, _ = graphed(blk)
                g = bench(replay)
            model_ms = BASE_MS - DENSE_FFN_INT8_MS + 8 * g
            print(f"{k:>5} {bm:>4} {path:>6} {eager:>9.3f} {g:>9.3f} {g/dense:>8.2f}x {8*g:>9.2f} "
                  f"{model_ms:>9.1f} {1000/model_ms:>6.2f} {err:>9.2e}")
            res["cells"][f"k{k}_bm{bm}_{path}"] = {"k": k, "bm": bm, "path": path, "eager_ms": eager,
                                            "block_ms": g, "model_ms": model_ms,
                                            "fps": 1000 / model_ms, "rel_err": err}
            del replay
            torch.cuda.empty_cache()
        if args.decompose:
            decompose(blk, f"fused K={k} bm={bm}")
            enable_fast_inference(sites, False)
            decompose(blk, f"baseline K={k} bm={bm}")
            args.decompose = False
        enable_fast_inference(sites, False)
        del s1, s2, sites, ref
        torch.cuda.empty_cache()

    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
