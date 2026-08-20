#!/usr/bin/env python3
"""Analyse the AAT-site experiments: F vs R, the deployable-rank ladder, the K sweep.

Cross-references accuracy against the Jetson cost measurements, because the deployment
question is not "which is more accurate at matched parameters" but "what accuracy does each
buy at the cost it actually incurs". A dense fine-tune edits weights in place: 1.0x compute,
zero added parameters. The mixture adds deltas and has to earn its 1.67-1.87x.
"""
import argparse
import glob
import json
import math
import os

import numpy as np

BINS = ["0.2-0.5", "0.5-0.8", "0.8-1.2", "1.2-1.8", "1.8-2.5"]

# Measured on the Jetson at the AAT FFN site (tools/bench_moe_edge.py), fp16, 729 tokens.
JETSON_COST = {(4, 8): 1.67, (8, 8): 1.69, (16, 8): 1.76, (32, 8): 1.87,
               (4, 29): 1.85, (4, 64): 1.85}


def primary(ev):
    v = [ev["bins"][b]["rel"] for b in BINS if ev["bins"][b]["n"]]
    return float(np.mean(v)) if v else float("nan")


def best(r):
    """Best validation point on the curve, not the final step.

    At least one dense run finished worse than its own optimum, so scoring on the final step
    alone penalises whichever arm overtrains and would make the comparison look better for the
    other one than it is. Both are reported."""
    c = r.get("curve") or []
    vals = [x["primary"] for x in c] + [primary(r["final_eval"])]
    return min(vals)


def welch(a, b):
    a, b = np.array(a), np.array(b)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    va, vb, na, nb = a.var(ddof=1), b.var(ddof=1), len(a), len(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return float("inf"), 0.0
    t = (a.mean() - b.mean()) / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return t, p


def cost_of(k, rank):
    """Measured cost if we have it, otherwise interpolate: cost is dominated by fixed
    orchestration, so it grows slowly and sub-linearly in both K and rank."""
    if (k, rank) in JETSON_COST:
        return JETSON_COST[(k, rank)], "measured"
    lo = 1.67 + 0.06 * math.log2(max(k, 1) / 4)          # K trend, 4->32 is 1.67->1.87
    hi = 0.18 * math.log2(max(rank, 1) / 8) / 3          # rank trend, 8->64 adds ~0.18
    return lo + max(hi, 0), "interp"


def load(out):
    runs = {}
    for f in sorted(glob.glob(os.path.join(out, "*.json"))):
        name = os.path.basename(f)[:-5]
        runs[name] = json.load(open(f))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/results/rmoe/aat")
    args = ap.parse_args()
    runs = load(args.out)
    if not runs:
        print("no results yet"); return

    F = [primary(r["final_eval"]) for n, r in runs.items() if n.startswith("Faat")]
    R = [primary(r["final_eval"]) for n, r in runs.items() if n.startswith("Raat")]
    Fb = [best(r) for n, r in runs.items() if n.startswith("Faat")]
    Rb = [best(r) for n, r in runs.items() if n.startswith("Raat")]

    print("=" * 78)
    print("1. MATCHED-PARAMETER COMPARISON (both ~37.8M trainable, 16 AAT FFN Linears)")
    print("=" * 78)
    if F and R:
        t, p = welch(R, F)
        print(f"  F dense    n={len(F)}  mean {np.mean(F):.4f}  {[round(x, 4) for x in F]}")
        print(f"  R mixture  n={len(R)}  mean {np.mean(R):.4f}  {[round(x, 4) for x in R]}")
        print(f"  R - F = {np.mean(R)-np.mean(F):+.4f}"
              + (f"   Welch t={t:.2f} p={p:.4f}" if np.isfinite(t) else "   (need n>=2 both)"))
        tb, pb = welch(Rb, Fb)
        print(f"\n  BEST-checkpoint (guards against either arm overtraining):")
        print(f"    F best  mean {np.mean(Fb):.4f}  {[round(x,4) for x in Fb]}")
        print(f"    R best  mean {np.mean(Rb):.4f}  {[round(x,4) for x in Rb]}")
        print(f"    R - F = {np.mean(Rb)-np.mean(Fb):+.4f}"
              + (f"   Welch t={tb:.2f} p={pb:.4f}" if np.isfinite(tb) else ""))
        print(f"\n  reference: conv site F 0.2196 / R 0.2366 (R LOST there);"
              f" deployed whole-model M 0.1124")
        print("\n  per-bin (mean):")
        print(f"    {'arm':<10}" + "".join(f"{b:>10}" for b in BINS))
        for tag, pre in [("F dense", "Faat"), ("R mixture", "Raat")]:
            rs = [r for n, r in runs.items() if n.startswith(pre)]
            if rs:
                print(f"    {tag:<10}" + "".join(
                    f"{np.mean([r['final_eval']['bins'][b]['rel'] for r in rs]):>10.4f}"
                    for b in BINS))

    lad = {n: r for n, r in runs.items() if n.startswith("Rladder")}
    if lad:
        print("\n" + "=" * 78)
        print("2. DEPLOYABLE LADDER: accuracy vs the cost it actually incurs")
        print("=" * 78)
        print(f"  {'config':<16}{'primary':>9}{'near-field':>12}{'added par':>12}"
              f"{'Jetson cost':>13}{'':>3}")
        if F:
            print(f"  {'F dense':<16}{np.mean(F):>9.4f}"
                  f"{np.mean([r['final_eval']['bins']['0.2-0.5']['rel'] for n, r in runs.items() if n.startswith('Faat')]):>12.4f}"
                  f"{0:>12,}{1.00:>13.2f}x   baseline, edits weights in place")
        for n, r in sorted(lad.items()):
            k, rank = r["k"], r["rank"]
            added = 8 * 2 * k * rank * (768 + 3072)
            c, how = cost_of(k, rank)
            print(f"  {'K=%d r=%d' % (k, rank):<16}{primary(r['final_eval']):>9.4f}"
                  f"{r['final_eval']['bins']['0.2-0.5']['rel']:>12.4f}{added:>12,}"
                  f"{c:>13.2f}x   {how}")

    ks = {n: r for n, r in runs.items() if n.startswith("Ksweep")}
    if ks or R:
        print("\n" + "=" * 78)
        print("3. K SWEEP AT CONSTANT CAPACITY: granularity, not budget")
        print("=" * 78)
        print("  Fixed ~37.8M added params; only the split changes. If more, smaller experts")
        print("  win, the gain is specialisation. If flat, it is the low-rank structure.")
        print(f"  {'config':<16}{'K*rank':>9}{'primary':>9}{'near-field':>12}{'Jetson':>10}")
        rows = []
        for n, r in runs.items():
            if n.startswith("Ksweep") or n.startswith("Raat"):
                rows.append((r["k"], r["rank"], primary(r["final_eval"]),
                             r["final_eval"]["bins"]["0.2-0.5"]["rel"]))
        for k, rank, pr, nf in sorted(rows):
            c, _ = cost_of(k, rank)
            print(f"  {'K=%d r=%d' % (k, rank):<16}{k*rank:>9}{pr:>9.4f}{nf:>12.4f}{c:>9.2f}x")

    print("\n" + "=" * 78)
    print("CHECKS")
    print("=" * 78)
    inits = {round(primary(r["init_eval"]), 4) for r in runs.values()}
    print(f"  init_eval identical across runs: {inits}  "
          f"{'OK' if len(inits) == 1 else 'MISMATCH -- runs are not comparable'}")
    ns = {r["final_eval"]["n"] for r in runs.values()}
    print(f"  eval frame count identical: {ns}  {'OK' if len(ns) == 1 else 'MISMATCH'}")
    steps = {r["steps"] for r in runs.values()}
    print(f"  steps identical: {steps}  {'OK' if len(steps) == 1 else 'MISMATCH'}")
    for n, r in runs.items():
        c = r.get("curve", [])
        if len(c) >= 3 and c[-1]["primary"] > min(x["primary"] for x in c) + 0.004:
            print(f"  WARN {n}: final ({c[-1]['primary']:.4f}) worse than its best "
                  f"({min(x['primary'] for x in c):.4f}) -- overtrained past the optimum")


if __name__ == "__main__":
    main()
