#!/usr/bin/env python3
"""The final MERLIN comparison table: what each method achieves, and what it costs to carry.

Two questions have to be answered together, and neither answers the other:

  capability   does the method navigate? measured end-to-end in the C4 closed loop, all arms
               driving the SAME map and planner on the SAME layouts, so the arms differ only
               in the depth they produce. Run on a datacenter GPU where every arm is fast,
               which is what isolates depth quality from throughput.
  deployability could you carry it? measured on the Orin Nano the robot actually flies, same
               resolution, same INA3221 rails, power sampled over the timed loop only.

Reporting only the first would credit a model that cannot fly. Reporting only the second would
credit a model that flies badly. The table is the join.

Capability columns are also marked, because throughput comparisons between methods that emit
different things are not comparisons: a depth-only network is solving a strictly easier problem
than one producing metric depth AND relative pose.
"""
import argparse
import collections
import glob
import json
import os

import numpy as np

# name -> (params_M, outputs, notes). Filled from measurement where available.
CAPABILITY = {
    "student": ("MERLIN (distilled)", "metric depth + pose"),
    "teacher": ("MapAnything 1.2B", "metric depth + pose"),
    "gt": ("GT depth (oracle)", "perfect depth"),
    "blind": ("route only (control)", "none"),
}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(c - h, 0.0), min(c + h, 1.0)


def load_c4(root):
    """-> {arm: {class: rows}} from the four-arm comparison."""
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in glob.glob(os.path.join(root, "*", "**", "*.jsonl"), recursive=True):
        cls = os.path.relpath(f, root).split(os.sep)[0]
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("ended") == "px4_frozen":
                continue
            per[r.get("arm")][cls].append(r)
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c4", default="out/c4cmp", help="four-arm C4 comparison root")
    ap.add_argument("--onboard", nargs="*", default=[], help="bench_swapc.py JSON outputs")
    ap.add_argument("--out", default="results/swapc/final_table.json")
    args = ap.parse_args()

    per = load_c4(args.c4)
    print("=" * 100)
    print("CAPABILITY -- C4 closed loop (same map, same planner, same layouts; arms differ "
          "only in depth)")
    print("=" * 100)
    print("%-24s %-22s %6s %8s %8s %8s %14s" % (
        "method", "outputs", "n", "success", "coll%", "TO%", "95% CI"))
    cap = {}
    for arm in ("blind", "student", "teacher", "gt"):
        by_cls = per.get(arm) or {}
        if not by_cls:
            continue
        rows = [r for v in by_cls.values() for r in v]
        n = len(rows)
        succ = float(np.mean([r["success"] for r in rows]))
        e = collections.Counter(r["ended"] for r in rows)
        lo, hi = wilson(int(round(succ * n)), n)
        label, outputs = CAPABILITY.get(arm, (arm, "?"))
        # balanced over classes, so a class with more episodes cannot dominate
        bal = float(np.mean([np.mean([r["success"] for r in v]) for v in by_cls.values()]))
        cap[arm] = {"n": n, "success": round(succ, 3), "balanced": round(bal, 3),
                    "coll_pct": round(100 * e["collision"] / n, 1),
                    "to_pct": round(100 * e["timeout"] / n, 1),
                    "ci": [round(lo, 3), round(hi, 3)]}
        print("%-24s %-22s %6d %8.3f %8.1f %8.1f   [%.3f, %.3f]" % (
            label, outputs, n, succ, 100 * e["collision"] / n, 100 * e["timeout"] / n, lo, hi))

    dep = []
    for p in args.onboard:
        if os.path.exists(p):
            dep += [r for r in json.load(open(p)) if r.get("kind") != "capacity"]
    if dep:
        print()
        print("=" * 100)
        print("DEPLOYABILITY -- measured on the Orin Nano 8GB (same res, same rails, power "
              "over the timed loop)")
        print("=" * 100)
        print("%-30s %8s %9s %8s %8s %9s" % (
            "candidate", "FPS", "lat(ms)", "mem(GB)", "power(W)", "J/kf"))
        for r in sorted(dep, key=lambda x: -x.get("fps", 0)):
            if "error" in r:
                print("%-30s %s" % (r["name"][:30], "DID NOT RUN: " + r["error"][:44]))
                continue
            print("%-30s %8.2f %9.1f %8.2f %8.1f %9.3f" % (
                r["name"][:30], r.get("fps", 0), r.get("latency_ms", {}).get("median", 0),
                r.get("peak_mem_gb", 0), r.get("power_w", 0), r.get("energy_j_kf", 0)))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"capability": cap, "deployability": dep}, open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
