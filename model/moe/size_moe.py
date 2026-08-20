#!/usr/bin/env python3
"""Phase 0: is there a (K, r) design point that carries N-times the parameters at matched latency?

N is THE quantity. Let X be the total params of a dense model that hits our measured latency;
our method must carry N*X at that same latency. N > 1 or there is no method; N >= 5 is the
target. Everything here is arithmetic over MEASURED constants -- no model is built and no GPU
is touched, so this is the cheapest possible way to find out the architecture cannot work.

Three independent ceilings, all of which must hold:
  compute    active params set the GEMM time
  bandwidth  ALL K experts' weights are read every frame. With 729 tokens and top-1, essentially
             every expert is touched (measured 83-99% on TUM), so total params -- not active --
             set the weight traffic. This is the ceiling that bites first at large K.
  memory     total weights must fit the model-only budget on an 8 GB Orin Nano.

The trap this is built to expose: the frozen backbone is ALWAYS active, so N is capped by how
much of the model is base. With a 230M always-on base, experts confined to the 37.8M of AAT FFN
cannot move N much regardless of K.
"""
import argparse
import math

# ---------------------------------------------------------------- measured constants
BASE_PARAMS = 230_286_550      # student_hab_nav, capacity.json base:8
D, F = 768, 3072               # AAT FFN dims
N_TOK = 729                    # 27x27 tokens at 378px
N_BLOCKS = 8                   # aat_depth 8
FFN_PER_BLOCK = 2 * D * F      # 4.72M params in one block's fc1+fc2

MS_INT8_FOLDED = 40.65         # trtexec v2_folded2.engine  -> 24.6 FPS
MS_FP16_TORCH = 169.49         # capacity.json base:8
SPEEDUP_INT8 = MS_FP16_TORCH / MS_INT8_FOLDED          # 4.17x
MS_PER_MPARAM_FP16 = 0.2182    # slope of the measured dense curve (capacity.json)
BW_GBPS = 90.0                 # measured achievable read bandwidth
RIDGE = 80.0                   # measured 7.2 TFLOPS / 90 GB/s, fp16
MEM_BUDGET_GB = 5.5            # weights only, inside the ~6.5 GB model-node budget
TRAIN_FRAMES = 3519            # data/nearfield/train at stride 1
BATCH = 4                      # run_cell default


def expert_params(r, n_l):
    """One expert, one AAT block (both fc1 and fc2 sites).

    butterfly: n_l * d/2 angles per rotated dimension; fc1 rotates 768 in and 3072 out, fc2 the
               reverse, so both sites cost n_l * (D + F) / 2 * 2 = n_l * (D+F).
    low-rank : A(d->r) + B(r->out) = r * (D + F) per site, two sites.
    Note butterfly with n_l layers costs the SAME params as rank n_l -- but re-orients the full
    -rank shared base rather than adding a rank-n_l correction. That asymmetry is the bet.
    """
    bfly = n_l * (D + F)
    lowr = 2 * r * (D + F)
    return bfly + lowr


def design(K, r, n_l, bytes_per_param=1.0, fps_target=20.0):
    e_blk = expert_params(r, n_l)              # one expert, one block
    e_all = e_blk * N_BLOCKS                   # one expert across all 8 blocks
    total = BASE_PARAMS + K * e_all            # every expert stored
    active = BASE_PARAMS + e_all               # top-1: base + exactly one expert

    # --- compute: only active params do work
    add_active_M = (active - BASE_PARAMS) / 1e6
    ms_compute = MS_INT8_FOLDED + add_active_M * MS_PER_MPARAM_FP16 / SPEEDUP_INT8

    # --- bandwidth: only experts actually TOUCHED are read each frame. With top-1 over N_TOK
    # tokens, at most N_TOK distinct experts can fire, so the read set saturates at
    # min(K, N_TOK) -- past K=729 extra experts cost memory but NOT bandwidth. (Below that,
    # measured routing touches 83-99% of experts, so min() is the right approximation.)
    touched = min(K, N_TOK)
    bytes_frame = (BASE_PARAMS + touched * e_all) * bytes_per_param
    ms_bw = bytes_frame / (BW_GBPS * 1e9) * 1e3
    bytes_total = total * bytes_per_param

    ms = max(ms_compute, ms_bw)
    fps = 1000.0 / ms

    # --- N: params we carry vs params a dense model carries at OUR latency.
    # The dense model at latency ms has (ms - fixed)/slope params, from the measured curve.
    dense_at_ms = BASE_PARAMS + max(0.0, (ms - MS_INT8_FOLDED)) * SPEEDUP_INT8 / MS_PER_MPARAM_FP16 * 1e6
    N = total / dense_at_ms

    gb = bytes_total / 1e9

    # --- DATA: can K experts actually be trained? Top-1 splits the token stream K ways, so each
    # expert sees train_frames*N_TOK/K token-instances TOTAL. An expert with more parameters than
    # it has training tokens is under-determined no matter how good the routing is. This ceiling
    # bites long before bandwidth or memory do.
    tok_total = TRAIN_FRAMES * N_TOK
    tok_per_expert = tok_total / K
    # identifiability: each token routed to an expert supervises (D+F) residual outputs across
    # the block's two sites, not one scalar. Constraints per parameter is the ratio that decides
    # whether a rank-r map is determined; tokens-per-parameter is the wrong statistic and is
    # ~3840x too pessimistic.
    con_per_param = tok_per_expert * (D + F) / e_blk
    # practicality: with batch B, an expert is touched ~B*N_TOK/K times per STEP. Below ~1 the
    # updates are so sparse that optimisation, not identifiability, becomes the limit.
    tok_per_step = BATCH * N_TOK / K

    return dict(K=K, r=r, n_l=n_l, total=total, active=active, e_all=e_all,
                ms_compute=ms_compute, ms_bw=ms_bw, ms=ms, fps=fps, N=N, gb=gb,
                touched=touched, tok_per_expert=tok_per_expert, con_per_param=con_per_param,
                tok_per_step=tok_per_step,
                bound="bandwidth" if ms_bw > ms_compute else "compute",
                ok=(N > 1 and fps >= fps_target and gb <= MEM_BUDGET_GB))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 32, 128, 512, 2048, 8192])
    ap.add_argument("--n-layers", type=int, default=2, help="butterfly layers")
    ap.add_argument("--bytes", type=float, default=1.0, help="bytes/param (1=INT8, 2=fp16)")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--base", type=float, default=None,
                    help="override always-active base params (M), to test a smaller backbone")
    args = ap.parse_args()

    global BASE_PARAMS
    if args.base:
        BASE_PARAMS = int(args.base * 1e6)

    print(f"base (always active) = {BASE_PARAMS/1e6:.0f}M | butterfly n_l={args.n_layers} | "
          f"{args.bytes:g} B/param | target {args.fps:.0f} FPS | mem budget {MEM_BUDGET_GB} GB")
    print(f"one expert (8 blocks): " + ", ".join(
        f"r={r}: {expert_params(r, args.n_layers)*N_BLOCKS/1e6:.2f}M" for r in args.ranks))
    print()
    print(f"{'K':>6} {'r':>4} {'total':>9} {'active':>9} {'N':>6} {'ms':>7} {'FPS':>6} "
          f"{'GB':>6} {'tok/exp':>8} {'con/par':>8} {'tok/step':>9} {'bound':>10}  ok")
    best = []
    for r in args.ranks:
        for K in args.ks:
            d = design(K, r, args.n_layers, args.bytes, args.fps)
            flag = "OK" if d["ok"] else ("mem" if d["gb"] > MEM_BUDGET_GB
                                         else ("fps" if d["fps"] < args.fps else "N"))
            print(f"{K:>6} {r:>4} {d['total']/1e6:>8.0f}M {d['active']/1e6:>8.0f}M "
                  f"{d['N']:>6.2f} {d['ms']:>7.1f} {d['fps']:>6.1f} {d['gb']:>6.2f} "
                  f"{d['tok_per_expert']:>8.0f} {d['con_per_param']:>8.1f} {d['tok_per_step']:>9.2f} "
                  f"{d['bound']:>10}  {flag}")
            if d["ok"]:
                best.append(d)
        print()

    if best:
        best.sort(key=lambda d: -d["N"])
        print("PASSING design points (N>1, FPS and memory OK), best N first:")
        for d in best[:5]:
            print(f"  K={d['K']:<6} r={d['r']:<4} N={d['N']:.2f}  {d['fps']:.1f} FPS  "
                  f"{d['total']/1e6:.0f}M total / {d['active']/1e6:.0f}M active  {d['gb']:.2f} GB"
                  f"  con/param={d['con_per_param']:.0f} tok/step={d['tok_per_step']:.2f}"
                  f"{'  <-- UNDER-DETERMINED' if d['con_per_param'] < 10 else ''}"
                  f"{'  <-- SPARSE UPDATES' if d['tok_per_step'] < 1 else ''}")
    else:
        print("NO design point passes. The always-active base is the binding term:")
        print(f"  N <= (base + K*e)/(base + e); with base={BASE_PARAMS/1e6:.0f}M you need")
        print(f"  K*e >> base, i.e. total expert mass >> the backbone.")


if __name__ == "__main__":
    main()
