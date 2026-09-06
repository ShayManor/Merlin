#!/usr/bin/env python3
"""Does the mixture's closed-loop win over dense survive a change of training seed?

The C4 campaign's headline (top-1 vs dense, b=234 c=45, p=2.8e-32 over 13 classes) came from ONE
training seed per arm. Seeds 1 and 2 retrain both arms with byte-identical flags -- only `--seed`
differs, verified token-by-token -- and re-fly the two high-n classes on the SAME
`--seed0 7000+20*chunk` layout grid, so every episode is paired across arms and comparable across
seeds.

Reads out/c4_seed1/ and out/c4_seed2/ (dirs named `S<n>_<arm>_<class>`), and compares against the
seed-0 figures below, which came from c4stats.py and are fixed.

Reports, per seed: paired McNemar per class and pooled, then the three-seed tally. Also scores the
two candidate TRAINING-TIME indicators (final-eval near_rel and effective experts) against the
outcome, because an indicator that predicts closed-loop behaviour without flying would be the first
thing in this project able to do that.
"""
import collections
import glob
import json
import os
import statistics as st
import sys
from math import comb

VOID = ("px4_failsafe", "px4_frozen")

# From c4stats.py on the seed-0 campaign. Paired subsets differ per comparison, which is why the
# dense value is not identical between rows -- that is correct, not a transcription slip.
SEED0 = {"wall_cabinet_01": (0.640, 0.335), "clothes_hanger_02": (0.550, 0.039)}
SEED0_POOLED = (234, 45)          # b, c over all 13 classes

# Final-eval training values (L-FCR-32). FINAL ONLY -- mid-training values here are non-monotone
# and misled this analysis twice (L-FCR-30, L-FCR-31).
TRAIN = {0: {"near_rel": 0.1457, "eff_experts": 7.80},
         1: {"near_rel": 0.1293, "eff_experts": 6.87},
         2: {"near_rel": 0.1402, "eff_experts": 6.92}}


def mcnemar(b, c):
    """One-sided P(B >= b | B ~ Binom(b+c, 1/2)): chance the mixture's edge is coin-flip luck."""
    n = b + c
    return 1.0 if n == 0 else sum(comb(n, k) for k in range(b, n + 1)) / 2 ** n


def load(root):
    """(class, arm) -> {(scene, seed): success}, voids dropped."""
    out = collections.defaultdict(dict)
    nvoid = ntot = 0
    for p in glob.glob(os.path.join(root, "**", "obstacles.jsonl"), recursive=True):
        # The fly harness writes <root>/S<n>_<arm>_<class>/chunk<k>/obstacles.jsonl, so the cell
        # name is NOT the parent directory -- that is `chunk0`. Scan the path for the component
        # that actually encodes arm and class. Reading the parent silently yields zero paired
        # episodes and a bogus "tie" for every seed.
        cell = next((part for part in p.split(os.sep)
                     if "_top1_" in part or "_ft_" in part), None)
        if cell is None:
            continue
        who = "top1" if "_top1_" in cell else "dense"
        cls = cell.split("_", 2)[2]
        for line in open(p):
            try:
                r = json.loads(line)
            except Exception:
                continue
            ntot += 1
            if r.get("ended") in VOID:
                nvoid += 1
                continue
            if r.get("arm") != "student":
                continue
            out[(cls, who)][(r["scene"], r["seed"])] = bool(r.get("success"))
    return out, ntot, nvoid


def report(seed, root):
    if not os.path.isdir(root):
        print(f"  seed {seed}: {root} not present")
        return None
    data, ntot, nvoid = load(root)
    if not data:
        print(f"  seed {seed}: no rows yet")
        return None
    print(f"  seed {seed}  ({ntot} rows, {nvoid} voids = {100*nvoid/max(ntot,1):.1f}%)")
    tb = tc = 0
    for cls in sorted({c for c, _ in data}):
        A, B = data.get((cls, "top1"), {}), data.get((cls, "dense"), {})
        keys = sorted(set(A) & set(B))
        if len(keys) < 30:
            print(f"    {cls:<20} paired n={len(keys)} -- BELOW n=30, not called")
            continue
        b = sum(1 for k in keys if A[k] and not B[k])
        c = sum(1 for k in keys if B[k] and not A[k])
        tb += b
        tc += c
        s0 = SEED0.get(cls)
        ref = f"   [seed0 {s0[0]:.3f} vs {s0[1]:.3f}]" if s0 else ""
        print(f"    {cls:<20} n={len(keys):<4} top1={st.mean([A[k] for k in keys]):.3f} "
              f"dense={st.mean([B[k] for k in keys]):.3f}  b={b:<3} c={c:<3} "
              f"p(top1>dense)={mcnemar(b, c):.3g}{ref}")
    if tb + tc:
        print(f"    {'POOLED':<20} b={tb} c={tc}  p(top1>dense)={mcnemar(tb, tc):.4g}  "
              f"p(dense>top1)={mcnemar(tc, tb):.4g}")
    return tb, tc


def main():
    roots = {1: sys.argv[1] if len(sys.argv) > 1 else "out/c4_seed1",
             2: sys.argv[2] if len(sys.argv) > 2 else "out/c4_seed2"}
    print("THREE-SEED REPLICATION of the C4 headline\n")
    print(f"  seed 0  (13 classes, from c4stats)  b={SEED0_POOLED[0]} c={SEED0_POOLED[1]}  "
          f"p(top1>dense)=2.8e-32   -> MIXTURE")
    outcomes = {0: "mixture"}
    for s in (1, 2):
        r = report(s, roots[s])
        if r:
            tb, tc = r
            outcomes[s] = "mixture" if mcnemar(tb, tc) < 0.05 else (
                "dense" if mcnemar(tc, tb) < 0.05 else "tie")

    print("\nTALLY")
    for s in sorted(outcomes):
        print(f"  seed {s}: {outcomes[s]}")
    wins = sum(1 for v in outcomes.values() if v == "mixture")
    if len(outcomes) == 3:
        print(f"  -> {wins} of 3 seeds favour the mixture.")
        print("     3/3 = the claim holds (but was never tested this way before).")
        print("     2/3 = BIMODAL training, a characterisation of instability, NOT")
        print("           'conditional capacity improves navigation'.")
        print("     1/3 = the claim is DEAD; report the MoE work as a negative result.")

        # Which training-time indicator, if either, ordered the seeds correctly?
        print("\nTRAINING-TIME INDICATORS (final evals only, L-FCR-32)")
        for key, better in (("near_rel", "higher"), ("eff_experts", "higher")):
            vals = {s: TRAIN[s][key] for s in outcomes}
            pred = {s: ("mixture" if vals[s] >= st.median(list(vals.values())) else "dense")
                    for s in outcomes}
            ok = sum(1 for s in outcomes if pred[s] == outcomes[s])
            print(f"  {key:<12} " + "  ".join(f"s{s}={vals[s]}" for s in sorted(vals))
                  + f"   predicts {ok}/{len(outcomes)} correctly"
                  + ("  <-- consistent" if ok == len(outcomes) else "  <-- FALSIFIED"))


if __name__ == "__main__":
    main()
