#!/usr/bin/env python3
"""Aggregate the phase-A screen, apply the fixed decision rules, emit the figure and table.

Spec: docs/superpowers/specs/2026-08-08-rmoe-00-crossing-curves.md

PRIMARY METRIC: macro-averaged relative error across the five true-range bins. Fixed before any
result was seen. The hypothesis is that every regime improves once experts stop sharing weights,
so averaging across bins is the right scalar; it also does not rest on any single bin's frame
count the way a near-field-only metric would.
"""
import argparse
import glob
import json
import os

import numpy as np

BINS = ["0.2-0.5", "0.5-0.8", "0.8-1.2", "1.2-1.8", "1.8-2.5"]


def primary(ev):
    """Macro-average of per-bin relative error, over bins that have support."""
    v = [ev["bins"][b]["rel"] for b in BINS
         if ev["bins"][b]["n"] > 0 and np.isfinite(ev["bins"][b]["rel"])]
    return float(np.mean(v)) if v else float("nan")


def load(out):
    runs = {}
    for f in sorted(glob.glob(os.path.join(out, "*.json"))):
        if os.path.basename(f) == "summary.json":      # this script's own output
            continue
        r = json.load(open(f))
        # D, M and F have no experts; they carry argparse's default k, normalise to 0.
        k = 0 if r["arm"] in ("D", "M", "F") else r["k"]
        runs.setdefault((r["arm"], k), []).append(r)
    return runs


def cell(runs, arm, k):
    rs = runs.get((arm, k), [])
    return ([primary(r["final_eval"]) for r in rs], rs)


def decide(runs, floor):
    p32, _ = cell(runs, "P", 32)
    p8, _ = cell(runs, "P", 8)
    l32, _ = cell(runs, "L_top1", 32)
    l8, _ = cell(runs, "L_top1", 8)
    o32, _ = cell(runs, "O", 32)
    notes = []
    if not (p32 and l32 and o32 and l8 and p8):
        return "INCOMPLETE", ["not all screen cells present"]

    m = {k: float(np.mean(v)) for k, v in
         dict(p32=p32, p8=p8, l32=l32, l8=l8, o32=o32).items()}

    # Premise check first: if learned routing does not roll over, there is no barrier to remove.
    if m["l32"] <= m["l8"] + floor:
        notes.append(f"learned routing did NOT roll over (L@32 {m['l32']:.4f} vs "
                     f"L@8 {m['l8']:.4f}, floor {floor:.4f}): no barrier to remove")
        return "DEAD", notes
    # Oracle check: can this axis pay at all?
    if m["o32"] >= m["l32"] - floor:
        notes.append(f"oracle did not beat learned (O@32 {m['o32']:.4f} vs L@32 "
                     f"{m['l32']:.4f}): the axis carries no signal")
        return "DEAD", notes
    # The deployable claim.
    beats_all_seeds = all(p < min(l32) - floor for p in p32) if len(p32) == len(l32) else False
    if m["p32"] < m["l32"] - floor and beats_all_seeds and m["p32"] < m["p8"]:
        notes.append(f"P@32 {m['p32']:.4f} beats L@32 {m['l32']:.4f} in every seed and "
                     f"improves on P@8 {m['p8']:.4f}")
        return "PROCEED", notes
    notes.append(f"oracle works (O@32 {m['o32']:.4f} < L@32 {m['l32']:.4f}) but the predicted "
                 f"key does not (P@32 {m['p32']:.4f}): key quality, not the axis")
    return "REDIRECT", notes


LABEL = {"M_K0": "MERLIN (nf finetune)", "D_K0": "init (untrained)",
         "F_K0": "dense finetune", "L_top1_K8": "learned K=8", "L_top1_K32": "learned K=32",
         "P_K8": "forced K=8", "P_K32": "forced K=32", "O_K32": "routed K=32 (oracle)"}


def per_bin_table(runs):
    """Per-bin relative error and signed bias for every arm, against deployed MERLIN.

    This is the comparison that says whether experts beat what is already shipped, rather than
    only beating the checkpoint they were initialised from.
    """
    order = ["M_K0", "D_K0", "F_K0", "L_top1_K32", "P_K32", "O_K32", "L_top1_K8", "P_K8"]
    cols = [c for c in order if any(f"{a}_K{k}" == c for a, k in runs)]
    if not cols:
        return
    print("\nper-bin RELATIVE ERROR (signed bias in parentheses, + = predicted farther)")
    w = 22
    print(f"{'bin':<10}" + "".join(f"{LABEL.get(c, c)[:w - 2]:>{w}}" for c in cols))
    for b in BINS:
        line = f"{b:<10}"
        for c in cols:
            arm, k = c.rsplit("_K", 1)
            rs = runs.get((arm, int(k)), [])
            vals = [r["final_eval"]["bins"][b] for r in rs if r["final_eval"]["bins"][b]["n"]]
            if not vals:
                line += f"{'-':>{w}}"
            else:
                rel = np.mean([v["rel"] for v in vals])
                bias = np.mean([v["bias"] for v in vals])
                line += f"{rel:.4f} ({bias:+.3f})".rjust(w)
        print(line)


def figure(runs, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    style = {"L_top1": ("#c1440e", "learned"), "P": ("#1f6feb", "range-prescribed"),
             "O": ("#7d7d7d", "oracle (GT key)")}
    for arm, (colour, label) in style.items():
        ks = sorted({k for a, k in runs if a == arm})
        if not ks:
            continue
        mu = [np.mean(cell(runs, arm, k)[0]) for k in ks]
        lo = [np.min(cell(runs, arm, k)[0]) for k in ks]
        hi = [np.max(cell(runs, arm, k)[0]) for k in ks]
        ax.plot(ks, mu, "o-", color=colour, label=label, lw=1.8, ms=5)
        ax.fill_between(ks, lo, hi, color=colour, alpha=0.15, lw=0)
    for arm, colour, label, ls in [("D", "black", "init (untrained)", ":"),
                                   ("F", "#2e7d32", "dense finetune", "--"),
                                   ("M", "#8e44ad", "MERLIN (nf finetune)", "-.")]:
        v, _ = cell(runs, arm, 0)
        if v:
            ax.axhline(np.mean(v), color=colour, ls=ls, lw=1.2, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 32], ["8", "32"])
    ax.set_xlabel("experts")
    ax.set_ylabel("relative error (macro over range bins)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/results/rmoe/screen")
    ap.add_argument("--png", default="/data/results/rmoe/screen/curves.png")
    ap.add_argument("--json", default="/data/results/rmoe/screen/summary.json")
    args = ap.parse_args()

    runs = load(args.out)
    # The noise floor comes from TRAINED arms. D and M do no training, so their seeds are
    # identical and their spread is zero, which would make any difference look significant.
    spreads = [float(np.max(v) - np.min(v)) for (a, k), rs in runs.items()
               if a not in ("D", "M") and len(v := cell(runs, a, k)[0]) > 1]
    floor = float(np.max(spreads)) if spreads else 0.0

    print(f"noise floor (max seed spread over trained arms): {floor:.4f}\n")
    hdr = f"{'cell':<14}{'primary':>9}{'spread':>9}{'whole':>9}{'fnear':>9}{'bf1':>7}{'ent':>7}{'n':>4}"
    print(hdr)
    rows = {}
    for (arm, k), rs in sorted(runs.items()):
        p, _ = cell(runs, arm, k)
        ev = [r["final_eval"] for r in rs]
        rows[f"{arm}_K{k}"] = dict(
            primary=float(np.mean(p)), spread=float(np.max(p) - np.min(p)),
            whole=float(np.mean([e["whole_absrel"] for e in ev])),
            false_near=float(np.mean([e["false_near"] for e in ev])),
            boundary_f1=float(np.mean([e["boundary_f1"] for e in ev])),
            entropy=float(np.mean([r.get("utilisation_entropy", np.nan) for r in rs])),
            seeds=len(rs), per_seed=p)
        r = rows[f"{arm}_K{k}"]
        print(f"{arm+'_K'+str(k):<14}{r['primary']:>9.4f}{r['spread']:>9.4f}"
              f"{r['whole']:>9.4f}{r['false_near']:>9.4f}{r['boundary_f1']:>7.3f}"
              f"{r['entropy']:>7.3f}{r['seeds']:>4}")

    per_bin_table(runs)

    verdict, notes = decide(runs, floor)
    # A whole-frame win paid for with phantom near surfaces is the documented trap, not a result.
    if verdict == "PROCEED":
        p_fn = rows.get("P_K32", {}).get("false_near", 0)
        d_fn = rows.get("D_K0", {}).get("false_near", 0)
        if p_fn > d_fn * 1.25:
            verdict = "TRAP"
            notes.append(f"P@32 false-near {p_fn:.4f} vs dense {d_fn:.4f}: map-degradation trap")

    print(f"\nVERDICT: {verdict}")
    for n in notes:
        print(f"  - {n}")

    if any(runs):
        figure(runs, args.png)
        print(f"\nfigure: {args.png}")
    json.dump({"noise_floor": floor, "cells": rows, "verdict": verdict, "notes": notes,
               "primary_metric": "macro-mean relative error over true-range bins"},
              open(args.json, "w"), indent=2)


if __name__ == "__main__":
    main()
