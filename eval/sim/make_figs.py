#!/usr/bin/env python3
"""Paper-grade figures from the closed-loop experiment outputs (clean/minimal style)."""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 140})


def load_rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def boot(xs, n=2000, seed=0):
    xs = np.asarray([x for x in xs if x is not None and np.isfinite(x)], float)
    if xs.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    bs = [rng.choice(xs, xs.size, replace=True).mean() for _ in range(n)]
    return float(xs.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def _arm_kind(a):
    if "base" in a:
        return "baseline", "#888888"
    if "nav1" in a:
        return "M1 (nav)", "#1f77b4"
    if "gt" in a:
        return "GT (oracle)", "#2ca02c"
    return a, "#1f77b4"


def fig_m1(rows, out):
    # order baseline, M1, GT
    order = {"baseline": 0, "M1 (nav)": 1, "GT (oracle)": 2}
    arms = sorted({r["arm"] for r in rows}, key=lambda a: order.get(_arm_kind(a)[0], 9))
    metrics = [("success", "success rate"), ("spl", "SPL"),
               ("collisions_per_m", "collisions / m"), ("fwd_perr_m", "fwd range err (m)")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.0 * len(metrics), 3.2))
    for ax, (m, label) in zip(axes, metrics):
        labels = []
        for i, a in enumerate(arms):
            lab, col = _arm_kind(a)
            labels.append(lab)
            mean, lo, hi = boot([r[m] for r in rows if r["arm"] == a])
            ax.bar(i, mean, color=col, width=0.6)
            ax.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]], color="k", capsize=4, lw=1)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(label)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


def fig_m2(rows, out):
    ctrls = sorted({r["arm"].split("@")[0] for r in rows})
    speeds = sorted({float(r["arm"].split("v")[-1]) for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    cmap = {"fixed_252": "#2ca02c", "fixed_378": "#ff7f0e", "fixed_518": "#d62728",
            "adaptive": "#1f77b4"}
    for c in ctrls:
        ys, los, his, es = [], [], [], []
        for sp in speeds:
            rr = [r for r in rows if r["arm"] == f"{c}@v{sp}"]
            m, lo, hi = boot([r["success"] for r in rr])
            ys.append(m); los.append(lo); his.append(hi)
            es.append(np.mean([r["energy_per_m"] for r in rr]) if rr else np.nan)
        axes[0].plot(speeds, ys, "-o", color=cmap.get(c, None), label=c, lw=1.6, ms=4)
        axes[0].fill_between(speeds, los, his, color=cmap.get(c, None), alpha=0.15)
        # success per energy/m (SWaP-C): success rate / (J per meter)
        spe = [y / e if e and np.isfinite(e) else np.nan for y, e in zip(ys, es)]
        axes[1].plot(speeds, spe, "-o", color=cmap.get(c, None), label=c, lw=1.6, ms=4)
    axes[0].set_xlabel("max speed (m/s)"); axes[0].set_ylabel("success rate")
    axes[0].set_title("success vs speed"); axes[0].legend(frameon=False, fontsize=9)
    axes[1].set_xlabel("max speed (m/s)"); axes[1].set_ylabel("success / (J·m⁻¹)")
    axes[1].set_title("success per energy")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


def fig_smooth(rows, out):
    # arm names like "gt_s4.0" / "student_s0.0": success vs blur sigma per ckpt
    import re
    pts = {}
    for r in rows:
        m = re.match(r"(.+)_s([0-9.]+)$", r["arm"])
        if not m:
            continue
        ck, sig = m.group(1), float(m.group(2))
        pts.setdefault(ck, {}).setdefault(sig, []).append(r["success"])
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    cmap = {"gt": "#2ca02c", "student": "#1f77b4", "baseline": "#888888"}
    lab = {"gt": "GT (oracle)", "student": "distilled student", "baseline": "baseline"}
    for ck in sorted(pts):
        sigs = sorted(pts[ck])
        ys, los, his = [], [], []
        for s in sigs:
            m, lo, hi = boot(pts[ck][s])
            ys.append(m); los.append(lo); his.append(hi)
        ax.plot(sigs, ys, "-o", color=cmap.get(ck), label=lab.get(ck, ck), lw=1.8, ms=5)
        ax.fill_between(sigs, los, his, color=cmap.get(ck), alpha=0.15)
    ax.set_xlabel("depth blur sigma (px)"); ax.set_ylabel("success rate")
    ax.set_title("smoothing the oracle's depth recovers navigation")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--mode", choices=["m1", "m2", "smooth"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = load_rows(args.rows)
    {"m1": fig_m1, "m2": fig_m2, "smooth": fig_smooth}[args.mode](rows, args.out)


if __name__ == "__main__":
    main()
