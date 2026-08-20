#!/usr/bin/env python3
"""Arm-U (full-FFN upcycling) against the existing 150k controls, at MATCHED steps.

Two things this figure has to be honest about:

1. The U cells are mid-flight. Everything is plotted only as far as the cells have actually run,
   so no arm is compared against another at a different step (L-ORE-91: the 2026-08-10 "MoE beats
   dense" claim was the same runs read before the ~30k crossover).
2. The controls' 147k endpoints are drawn as horizontal references precisely because the visible
   ordering is pre-crossover. ORE led by 15% at 3k and ended the WORST of three at 147k, so the
   gap still to close is the point of the figure, not the current lead.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
import numpy as np                                                     # noqa: E402

U = sys.argv[1] if len(sys.argv) > 1 else "out/upcycle"
C = sys.argv[2] if len(sys.argv) > 2 else "out/data"
OUT = sys.argv[3] if len(sys.argv) > 3 else "out/upcycle/upcycle_progress.png"

# validated categorical palette, same as make_ore_curves.py
CELLS = [("U8 top-1", "U8_top1", "#C25E1E"),
         ("U8 top-2", "U8_top2", "#B02F7C"),
         ("U8 council-3", "U8_council3", "#7A3FBF"),
         ("U2 top-1", "U2_top1", "#2E8B57")]
CTRL = [("dense FT", "L150_ft", "#1A6FB5"),
        ("dense widen", "L150_widen_matched", "#5A9BD4"),
        ("ORE K=32 unfrozen", "L150_ore_k32_unf", "#9a9a9a")]
INK, INK2, MUTED, GRID = "#1a1a1a", "#4a4a4a", "#7a7a7a", "#e3e3e3"
PANELS = [("near_rel", "near-field 0.2-0.5 m, rel. error", "primary metric"),
          ("whole", "whole-frame rel. error", ""),
          ("unseen", "unseen scenes (ord), rel. error", "")]


def load(d, tag):
    for p in (os.path.join(d, tag + ".json"), os.path.join(d, tag + ".json.partial")):
        if os.path.exists(p):
            try:
                return json.load(open(p))
            except json.JSONDecodeError:
                continue
    return None


def series(r, key):
    c = [(x["step"], x.get(key)) for x in r.get("curve", []) if x.get(key) is not None]
    return np.array([p[0] for p in c]), np.array([p[1] for p in c])


def smooth(v, w=5):
    """Rolling mean. Eval-to-eval scatter here is 0.017-0.023 on near-field, which is larger than
    every between-arm gap on show, so the raw traces cross each other constantly and carry no
    information at that resolution. Raw stays plotted, faint, so nothing is hidden."""
    if len(v) < w:
        return v
    k = np.ones(w) / w
    pad = np.r_[np.full(w // 2, v[0]), v, np.full(w // 2, v[-1])]
    return np.convolve(pad, k, mode="valid")[:len(v)]


def main():
    runs = {t: load(U, t) for _, t, _ in CELLS} | {t: load(C, t) for _, t, _ in CTRL}
    xmax = max(series(r, "near_rel")[0].max() for t, r in runs.items()
               if r and t.startswith("U"))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    fig.patch.set_facecolor("white")

    for ax, (key, ylab, note) in zip(axes, PANELS):
        for label, tag, color in CTRL:                     # controls first, behind
            r = runs.get(tag)
            if not r:
                continue
            s, v = series(r, key)
            m = s <= xmax
            ax.plot(s[m], smooth(v[m]), color=color, lw=1.6, ls="--", alpha=.9, zorder=2)
            if len(v):
                ax.axhline(v[-1], color=color, lw=.9, ls=":", alpha=.7, zorder=1)
        for label, tag, color in CELLS:
            r = runs.get(tag)
            if not r:
                continue
            s, v = series(r, key)
            ax.plot(s, v, color=color, lw=.7, alpha=.22, zorder=2)
            ax.plot(s, smooth(v), color=color, lw=2.1, zorder=3, solid_capstyle="round")
        ax.axvspan(28000, 32000, color="#f0d9a0", alpha=.45, zorder=0)
        ax.set_xlabel("training step", color=INK2, fontsize=9)
        ax.set_title(ylab + (f"   ({note})" if note else ""), color=INK, fontsize=10,
                     fontweight="600", loc="left", pad=8)
        ax.grid(True, color=GRID, lw=.7)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8.5)
        ax.set_xlim(0, xmax * 1.02)

    # one legend, on the first panel only
    h = [plt.Line2D([], [], color=c, lw=2.1) for _, _, c in CELLS]
    h += [plt.Line2D([], [], color=c, lw=1.6, ls="--") for _, _, c in CTRL]
    axes[0].legend(h, [l for l, _, _ in CELLS] + [l for l, _, _ in CTRL],
                   fontsize=8, frameon=False, loc="upper right", ncol=2,
                   labelcolor=INK2, handlelength=1.6, columnspacing=1.1)

    fig.text(.006, .035,
             f"Arm U is at {int(xmax):,} of 150,000 steps. Solid = upcycled experts, dashed = the "
             "existing 150k controls at MATCHED steps, dotted = where each control ENDS at 147k. "
             "Faint traces are raw evals; bold is a 5-eval rolling mean.",
             fontsize=7.5, color=MUTED)
    common = min(series(r, "near_rel")[0].max() for t, r in runs.items()
                 if r and t.startswith("U"))
    tail = ("Readings past 100k are inside the decision window; before it, ordering is "
            "pre-crossover and is not a result." if common >= 100000 else
            "Every ordering visible here is pre-crossover and is NOT a result.")
    fig.text(.006, .012,
             f"Shaded band is the ~30k crossover, where ORE's 15% lead at 3k began inverting into "
             f"the worst of three at 147k (L-ORE-91). Arms share {int(common):,} steps. " + tail,
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=[0, .065, 1, 1])
    fig.savefig(OUT, dpi=160, facecolor="white")
    print("wrote", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
