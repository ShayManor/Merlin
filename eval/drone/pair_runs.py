"""Pair a sweep cell against a frozen baseline on identical (scene, seed) layouts.

Every rule enforced here is one this harness has already been burned by:
  - `episode_logs/` filenames encode a LAYOUT INDEX, not the seed, so two runs on disjoint
    seeds produce identical filenames. Pairing is therefore keyed on the JSON body.
  - PX4-voided episodes log as ordinary timeouts and drag every arm down together.
  - Cells below ~30 paired episodes have swung by 0.3 between seed subsets, so n is
    reported next to every number and small cells are flagged rather than quietly printed.

Usage:
    python -m eval.drone.pair_runs --runs out/e3/l200_cloth_04 --baseline out/big5fix
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

from eval.drone.metrics import boot_ci

VOID = ("px4_failsafe", "px4_frozen")
MIN_N = 30


def obstacle_class(ep):
    """Injected-obstacle class, e.g. 'cloth_04'. Empty string when none was placed."""
    obs = ep.get("obstacles") or []
    if not obs:
        return ""
    return os.path.basename(obs[0].get("handle", "")).split(".")[0]


def load(run_dir, arm="student"):
    """Episodes for one arm, keyed by (scene, seed) read from the JSON body."""
    rows, voided = {}, 0
    for p in sorted(glob.glob(f"{run_dir}/episode_logs/*.json")):
        try:
            ep = json.load(open(p))
        except (ValueError, OSError):
            continue
        if ep.get("arm") != arm:
            continue
        if ep.get("ended") in VOID:
            voided += 1
            continue
        rows[(ep.get("scene"), int(ep.get("seed", -1)))] = ep
    return rows, voided


def _succ(ep):
    return float(bool(ep.get("success")))


def summarise(rows):
    """Per-class and balanced success. Balanced = mean of class means, so a class that
    happened to place more often cannot dominate the headline."""
    by = defaultdict(list)
    for k, ep in rows.items():
        by[obstacle_class(ep)].append(_succ(ep))
    per = {c: (float(np.mean(v)), len(v)) for c, v in sorted(by.items())}
    bal = float(np.mean([m for m, _ in per.values()])) if per else float("nan")
    return per, bal


def pair(treat, base):
    """Paired difference on the layouts both runs actually flew."""
    keys = sorted(set(treat) & set(base))
    d = np.array([_succ(treat[k]) - _succ(base[k]) for k in keys], float)
    if d.size == 0:
        return None
    mean, lo, hi = boot_ci(d)
    return {"n": int(d.size), "delta": mean, "ci": [lo, hi],
            "won": int((d > 0).sum()), "lost": int((d < 0).sum()),
            "tied": int((d == 0).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="treatment cell directories")
    ap.add_argument("--baseline", default=None, help="frozen baseline directory")
    ap.add_argument("--arm", default="student")
    ap.add_argument("--json", default=None, help="write the table here as JSON")
    args = ap.parse_args()

    base, base_void = ({}, 0)
    if args.baseline:
        base, base_void = load(args.baseline, args.arm)
        per, bal = summarise(base)
        print(f"baseline {args.baseline}  n={len(base)} voided={base_void} balanced={bal:.3f}")
        for c, (m, n) in per.items():
            print(f"    {c or '(none)':<20} {m:.3f}  n={n}")

    out = []
    for run in args.runs:
        rows, voided = load(run, args.arm)
        per, bal = summarise(rows)
        rec = {"run": run, "n": len(rows), "voided": voided, "balanced": bal,
               "per_class": {c: {"success": m, "n": n} for c, (m, n) in per.items()}}
        print(f"\n{run}  n={len(rows)} voided={voided} balanced={bal:.3f}")
        for c, (m, n) in per.items():
            flag = "  [n<30, do not quote]" if n < MIN_N else ""
            print(f"    {c or '(none)':<20} {m:.3f}  n={n}{flag}")
        if base:
            p = pair(rows, base)
            rec["paired"] = p
            if p is None:
                print("    paired: NO overlapping (scene, seed) with the baseline")
            else:
                flag = "  [n<30, do not quote]" if p["n"] < MIN_N else ""
                print(f"    paired vs baseline: {p['delta']:+.3f} "
                      f"95% CI [{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}]  n={p['n']}  "
                      f"won {p['won']} / lost {p['lost']} / tied {p['tied']}{flag}")
        out.append(rec)

    if args.json:
        json.dump(out, open(args.json, "w"), indent=2)


if __name__ == "__main__":
    main()
