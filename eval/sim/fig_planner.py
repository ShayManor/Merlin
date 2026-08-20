#!/usr/bin/env python3
"""The planner-mediation figure: the SAME depth-quality ladder produces OPPOSITE outcomes
under a naive reactive planner vs a proper Nav2-style stack. Left: success vs depth (reactive
falls with fidelity = the inverse law; Nav2 is flat-high). Right: Nav2 collisions/m fall
monotonically with fidelity (22x). Perception fidelity helps navigation -- as safety -- but
only once the planner can exploit it."""
import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 140})
LAD = [("orig", "orig\n(0.33)"), ("adapted", "adapted\n(0.10)"), ("gt", "GT\n(0.0)")]


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def boot(xs, n=2000, seed=0):
    xs = np.asarray([x for x in xs if x is not None and np.isfinite(x)], float)
    if xs.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    bs = [rng.choice(xs, xs.size, replace=True).mean() for _ in range(n)]
    return float(xs.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def ser(rows, metric):
    ys, lo, hi = [], [], []
    for sub, _ in LAD:
        m, l, h = boot([r[metric] for r in rows if r["arm"].split("_")[0] == sub])
        ys.append(m); lo.append(m - l); hi.append(h - m)
    return ys, [lo, hi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reactive", required=True)   # sim_ladder_slide rows (reactive)
    ap.add_argument("--nav2", required=True)        # sim_nav2_ladder2 rows
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rea, nav = load(args.reactive), load(args.nav2)
    x = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    # success: reactive vs nav2
    for rows, lab, col in [(rea, "reactive planner", "#888888"), (nav, "Nav2 stack", "#1f77b4")]:
        ys, err = ser(rows, "success")
        axes[0].errorbar(x, ys, yerr=err, fmt="-o", color=col, label=lab, lw=1.8, ms=5, capsize=3)
    axes[0].set_xticks(x); axes[0].set_xticklabels([l for _, l in LAD])
    axes[0].set_xlabel("depth quality (median abs_rel)"); axes[0].set_ylabel("success rate")
    axes[0].set_title("success: opposite slopes by planner"); axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_ylim(0, 1.0)
    # nav2 collisions monotone down
    ys, err = ser(nav, "collisions_per_m")
    axes[1].errorbar(x, ys, yerr=err, fmt="-o", color="#d62728", lw=1.8, ms=5, capsize=3)
    axes[1].set_xticks(x); axes[1].set_xticklabels([l for _, l in LAD])
    axes[1].set_xlabel("depth quality (median abs_rel)"); axes[1].set_ylabel("collisions / m")
    axes[1].set_title("Nav2: fidelity buys safety (22x)")
    fig.tight_layout(); fig.savefig(args.out, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
