#!/usr/bin/env python3
"""Robustness figure: depth-error degradation under realistic deployment perturbations (TUM
fr2_xyz held-out, 378px, scale-aligned gt abs_rel vs clean 0.0587). The rover is a cheap
global-shutter webcam on a jerky skid-steer. Finding: robust to sensor noise, JPEG, and moderate
motion blur; FRAGILE to low light (~2x error) -- a concrete, actionable deployment limitation.
Clean/minimal."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "swapc")
os.makedirs(ROOT, exist_ok=True)

# (label, % degradation vs clean, category) category: robust / fail / fixed
D = [
    ("motion blur\n(11px)", 2, "robust"), ("JPEG\n(q10)", 4, "robust"),
    ("sensor noise\n(σ50)", 10, "robust"), ("motion blur\n(21px)", 18, "robust"),
    ("low light\n(×0.5)", 135, "fail"), ("low light\n+ auto-gain", 4, "fixed"),
]
CMAP = {"robust": "#2e86ab", "fail": "#d1495b", "fixed": "#2a9d52"}
fig, ax = plt.subplots(figsize=(6.8, 3.7))
cols = [CMAP[c] for _, _, c in D]
xs = range(len(D))
bars = ax.bar(xs, [d[1] for d in D], color=cols, width=0.66)
for b, d in zip(bars, D):
    ax.text(b.get_x()+b.get_width()/2, d[1]+3, f"+{d[1]}%", ha="center", fontsize=9, fontweight="bold")
# arrow from fail -> fixed
ax.annotate("", xy=(5, 20), xytext=(4, 120), arrowprops=dict(arrowstyle="->", color="0.45", lw=1.5))
ax.text(4.5, 70, "1-line\nauto-gain", fontsize=7.5, color="0.4", ha="center")
ax.set_xticks(list(xs)); ax.set_xticklabels([d[0] for d in D], fontsize=8.5)
ax.set_ylabel("depth-error increase vs clean (%)")
ax.set_title("Robust to webcam noise/blur/JPEG; low-light failure is fixable", fontsize=10.5)
ax.set_ylim(0, 150)
ax.spines[["top", "right"]].set_visible(False)
ax.text(0.02, 0.92, "blue = robust;  red = failure mode;  green = recovered by free preprocessing",
        transform=ax.transAxes, fontsize=7.5, color="0.4", va="top")
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "robustness.png"), dpi=130)
print("saved", os.path.join(ROOT, "robustness.png"))
