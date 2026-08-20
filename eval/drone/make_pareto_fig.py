#!/usr/bin/env python3
"""Pareto figure for the occupancy-representation benchmark: collision rate vs goal completion.

The axes are the two failure modes that trade against each other in every monocular-depth
navigation system: MonoMPC (arXiv:2508.07387) calls it "stuck or collide" and measures it on
two mapping baselines. A method that only moves along the frontier is buying safety with
completion; a method that moves the frontier out is worth having.

Each method contributes a curve, traced by sweeping its own conservatism knob, so the
comparison is between frontiers rather than between arbitrarily-tuned single points.
MonoNav's published 47.4% completion at 13% collision is plotted as an external reference.
"""
import argparse
import collections
import glob
import json
import os

import numpy as np


def load_cell(root, cell):
    rows = []
    for f in glob.glob(os.path.join(root, cell, "**", "*.jsonl"), recursive=True):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("arm") == "student" and r.get("ended") != "px4_frozen":
                rows.append(r)
    return rows


def summarise(rows):
    if not rows:
        return None
    e = collections.Counter(r["ended"] for r in rows)
    n = len(rows)
    kf = [r["n_keyframes"] for r in rows if r.get("n_keyframes")]
    return {"n": n, "goal": 100.0 * e["goal"] / n, "coll": 100.0 * e["collision"] / n,
            "to": 100.0 * e["timeout"] / n,
            "kf": float(np.median(kf)) if kf else float("nan")}


def wilson(k, n, z=1.96):
    """Wilson interval, which behaves at the 0-count end where a normal interval does not."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(c - h, 0.0), 100 * min(c + h, 1.0)


CURVES = [
    ("MERLIN counting map (n=228)", "o", "#444444", [("baseline", "min_hits 4")]),
    ("log-odds, sigma-wide deposit", "s", "#1f77b4", [("lo04", "l=0.4"), ("lo07", "l=0.7")]),
    ("log-odds, narrow deposit", "P", "#9467bd", [("lon04", "l=0.4")]),
    ("NanoMap, no fusion", "^", "#d62728", [("nano1", "agree 1")]),
    ("risk-inflated clearance", "D", "#2ca02c", [("risk015", "k=0.15")]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out/pareto")
    ap.add_argument("--baseline", default="out/big5fix/cloth_04",
                    help="the counting-map cell, run before --map existed")
    ap.add_argument("--out", default="results/drone/pareto_maps.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = {}
    for _, _, _, cells in CURVES:
        for cell, _lab in cells:
            rows = load_cell(os.path.dirname(args.baseline), os.path.basename(args.baseline)) \
                if cell == "baseline" else load_cell(args.root, cell)
            s = summarise(rows)
            if s:
                k = int(round(s["coll"] * s["n"] / 100))
                s["ci"] = wilson(k, s["n"])
                data[cell] = s

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for name, marker, colour, cells in CURVES:
        pts = [(data[c]["coll"], data[c]["goal"], lab, data[c]) for c, lab in cells if c in data]
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if len(pts) > 1:
            ax.plot(xs, ys, "-", color=colour, lw=1.4, alpha=0.85, zorder=2)
        ax.scatter(xs, ys, marker=marker, s=64, color=colour, label=name, zorder=3,
                   edgecolor="white", linewidth=0.8)
        for x, y, lab, s in pts:
            ax.errorbar(x, y, xerr=[[x - s["ci"][0]], [s["ci"][1] - x]], color=colour,
                        alpha=0.35, capsize=2, lw=1.0, zorder=1)
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(6, -9),
                        fontsize=7, color=colour)

    ax.scatter([13.0], [47.4], marker="*", s=150, color="#888888", zorder=3,
               edgecolor="white", linewidth=0.8)
    ax.annotate("MonoNav\n(published)", (13.0, 47.4), textcoords="offset points",
                xytext=(-10, -4), fontsize=7, color="#666666", ha="right", va="center")

    ax.set_xlabel("collision rate (%)")
    ax.set_ylabel("goal completion (%)")
    ax.legend(frameon=False, fontsize=8, loc="upper right", handletextpad=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.12, lw=0.5, axis="y")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=180)
    print("wrote", args.out)

    print("\n%-26s %5s %7s %7s %7s %6s" % ("cell", "n", "goal%", "coll%", "TO%", "kf"))
    for name, _, _, cells in CURVES:
        for c, lab in cells:
            if c in data:
                s = data[c]
                print("%-26s %5d %7.1f %7.1f %7.1f %6.0f  [%s]"
                      % (f"{name[:18]} {lab}", s["n"], s["goal"], s["coll"], s["to"], s["kf"], c))


if __name__ == "__main__":
    main()
