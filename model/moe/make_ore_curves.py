#!/usr/bin/env python3
"""Training-loss trajectories for the ORE arms — is K=128 converged, or just undertrained?

Two things this figure has to be honest about:

1. `loss_trace` records the loss AFTER the aux term is added
   (`loss = si_loss + aux_coef*aux`), and aux is None for the dense arms. So the raw traces are
   NOT comparable in LEVEL across arms. The aux term is estimated as aux_coef*K/K^entropy and
   subtracted, giving an approximate task loss; the estimate is marked as such.
2. Only the SHAPE answers the convergence question, and shape is unaffected by a constant offset.

There is no validation curve: `curve` is empty in every run because no --eval-every was set, so
the only val numbers are init and final. That is a real gap and the figure says so rather than
implying a val trajectory exists.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

import sys
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "rmoe", "ore")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(D), "ore_curves.png")

# validated categorical palette (scripts/validate_palette.js, light mode: ALL CHECKS PASS)
ARMS = [
    ("ORE K=128 r=32",  "ore_k128_lr3e-4",   "#C25E1E"),
    ("ORE K=128 unfrozen", "oreB_unfrozen_r32", "#B02F7C"),
    ("ORE K=1 r=32 (no mixture)", "oreD_k1_r32", "#2E8B57"),
    ("Fine-tune (dense)", "ft_lr3e-4",        "#1A6FB5"),
]
INK, INK2, MUTED, GRID = "#1a1a1a", "#4a4a4a", "#7a7a7a", "#e3e3e3"


def aux_estimate(d):
    """aux_coef * K * sum(f*P) ~ aux_coef * K / K^entropy. Zero for the dense arms."""
    k, e = d.get("k") or 0, d.get("utilisation_entropy")
    if not k:
        return 0.0
    if k == 1 or e != e:                      # K=1: Switch term is exactly 1.0
        return 0.01
    return 0.01 * k / (k ** e)


def main():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), width_ratios=[1.85, 1])
    fig.patch.set_facecolor("white")

    finals = []
    ends = []
    nsteps = 6000
    for label, tag, color in ARMS:
        p = os.path.join(D, tag + ".json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        nsteps = d["steps"]
        lt = np.array(d["loss_trace"], dtype=float)
        steps = np.linspace(0, d["steps"], len(lt))
        task = lt - aux_estimate(d)
        ax.plot(steps, task, color=color, lw=2, label=label, solid_capstyle="round")
        ends.append((label, steps[-1], task[-1], color))
        finals.append((label, d["final_eval"]["whole_absrel"],
                       d["final_eval"]["bins"]["0.2-0.5"]["rel"], color))
        # slope over the last quarter: is it still descending?
        q = task[int(len(task) * .75):]
        slope = np.polyfit(np.arange(len(q)), q, 1)[0]
        print(f"{label:<28} final task~{task[-1]:.4f}  last-quarter slope {slope:+.2e}/100steps"
              f"  {'STILL DESCENDING' if slope < -1e-5 else 'flat'}")

    # fan the labels out around their mean so none overlap, with a leader to the endpoint
    ends.sort(key=lambda e: -e[2])
    span = 0.016
    base = np.mean([e[2] for e in ends])
    for j, (label, x, yv, color) in enumerate(ends):
        ty = base + (len(ends) - 1) / 2 * span - j * span
        ax.annotate(label, xy=(x, yv), xytext=(x + nsteps * .045, ty),
                    color=color, fontsize=8.5, va="center", fontweight="600",
                    arrowprops=dict(arrowstyle="-", color=color, lw=.8, alpha=.55,
                                    shrinkA=0, shrinkB=2))

    ax.set_xlabel("training step", color=INK2, fontsize=9)
    ax.set_ylabel("training loss  (aux term removed, estimated)", color=INK2, fontsize=9)
    ax.set_title("All arms are still descending at 6000 steps", color=INK, fontsize=11,
                 fontweight="600", loc="left", pad=10)
    ax.grid(True, color=GRID, lw=.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.set_xlim(0, nsteps * 1.42)

    # right panel: the only val numbers that exist are final
    y = np.arange(len(finals))
    for i, (label, whole, near, color) in enumerate(finals):
        ax2.barh(i - .18, whole, height=.32, color=color, zorder=3)
        ax2.barh(i + .18, near, height=.32, color=color, alpha=.42, zorder=3)
        ax2.text(whole + .004, i - .18, f"{whole:.4f}", va="center", fontsize=8, color=INK2)
        ax2.text(near + .004, i + .18, f"{near:.4f}", va="center", fontsize=8, color=MUTED)
    ax2.set_yticks(y)
    ax2.set_yticklabels([f[0].replace(" (no mixture)", "") for f in finals], fontsize=8.5, color=INK2)
    ax2.invert_yaxis()
    ax2.set_xlabel("rel. depth error at 6000 steps", color=INK2, fontsize=9)
    ax2.set_title("solid = whole frame · faded = near field",
                  color=INK, fontsize=9.5, loc="left", pad=10)
    ax2.grid(True, axis="x", color=GRID, lw=.7)
    ax2.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax2.spines[s].set_visible(False)
    ax2.spines["bottom"].set_color(GRID)
    ax2.tick_params(colors=MUTED, labelsize=8.5)
    ax2.set_xlim(0, max(max(f[1] for f in finals), max(f[2] for f in finals)) * 1.22)

    fig.text(.008, .015,
             "No validation curve exists: --eval-every was never set, so only init and final val "
             "were measured. Loss levels are estimates (aux term subtracted); shape is exact.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=[0, .04, 1, 1])
    fig.savefig(OUT, dpi=160, facecolor="white")
    print("wrote", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
