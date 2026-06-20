#!/usr/bin/env python3
"""Compare distillation runs (baseline vs M1 nav-weighted) -> the misalignment table.

Reads the per-run JSON logs written by distill_cache.py and prints the final-eval
metrics side by side. The M1 thesis: nav-weighting lowers near-field / column-range
error (what control uses) at the cost of far-field error (what it doesn't).
"""
import argparse
import glob
import json
import os


def last_eval(path):
    d = json.load(open(path))
    ev = d.get("eval", [])
    return ev[-1] if ev else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="/workspace/ckpt/*_*.json")
    args = ap.parse_args()
    runs = {}
    for p in sorted(glob.glob(args.glob)):
        tag = os.path.basename(p).replace(".json", "").split("_")[-1]
        runs[tag] = last_eval(p)
    if not runs:
        print("no run logs found at", args.glob); return

    keys = ["step", "fid_absrel_vs_teacher", "gt_absrel", "gt_scale_err",
            "nav_near_absrel", "nav_far_absrel", "nav_col_range_mae", "nav_obstacle_iou"]
    w = 22
    print(f"{'metric':<{w}}" + "".join(f"{t:>14}" for t in runs))
    for k in keys:
        row = f"{k:<{w}}"
        for t in runs:
            v = runs[t].get(k, "-")
            row += f"{v:>14}" if isinstance(v, str) else f"{v:>14.4f}" if isinstance(v, float) else f"{v:>14}"
        print(row)

    # the M1 headline: near-field improvement vs far-field cost (nav1 vs baseline)
    if "baseline" in runs and "nav1" in runs:
        b, m = runs["baseline"], runs["nav1"]
        def d(k):
            try:
                return round(m[k] - b[k], 4)
            except Exception:
                return None
        print("\n=== M1 (nav1) - baseline ===")
        print(f"  near_absrel   delta = {d('nav_near_absrel')}  (negative = M1 better near-field)")
        print(f"  far_absrel    delta = {d('nav_far_absrel')}   (positive = far-field traded away)")
        print(f"  col_range_mae delta = {d('nav_col_range_mae')} (negative = better planner scan)")
        print(f"  obstacle_iou  delta = {d('nav_obstacle_iou')}  (positive = better free-space)")


if __name__ == "__main__":
    main()
