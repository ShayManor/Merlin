#!/usr/bin/env python3
"""The thesis in one figure: closed-loop navigation vs on-robot deployability.

y-axis  balanced C4 success over the five saved obstacle classes (same map, planner, seeds;
        arms differ only in depth). x-axis on-board FPS on the Orin Nano the robot flies.

Every point is measured. The two 1.2B foundation models are the fair top-of-line baselines --
MERLIN's own model class (metric depth + pose), not depth-only -- and they do not run on the
board, so they sit in the "does not deploy" band at left with their C4 success still plotted.
MERLIN is the only point that is both high-success and on-board real-time, which is the claim.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, params_M, c4_balanced, nano_fps or None, colour, marker, note)
POINTS = [
    ("MERLIN (ours)\n230M INT8", 230, 0.881, 22.9, "#2e86ab", "o",
     "runs on-board\n20-26 FPS, 2.0 GB, 0.72 J/kf"),
    ("MapAnything\n1.2B", 1228, 0.796, None, "#d1495b", "s", "does not load (swap thrash)"),
    ("VGGT\n1.26B", 1256, 0.826, None, "#e0a458", "^", "loads 5.0 GB, forward pass OOMs board"),
    ("GT depth\n(oracle)", 0, 1.000, None, "#8d99ae", "D", "perfect-depth ceiling"),
]

fig, ax = plt.subplots(figsize=(7.0, 4.6))

# "does not deploy" band at left
band_x = -3.0
ax.axvspan(-6, 0, color="0.93", zorder=0)
ax.text(band_x, 0.44, "does not run\non the Nano", ha="center", va="center",
        fontsize=8.5, color="0.45", rotation=0)

for label, pm, c4, fps, col, mk, note in POINTS:
    x = fps if fps is not None else band_x
    ax.scatter(x, c4, s=180, c=col, marker=mk, zorder=3, edgecolor="white", linewidth=1.2)
    dy = 0.035 if not label.startswith("GT") else -0.055
    ha = "left" if fps is not None else "center"
    ax.annotate(label, (x, c4), textcoords="offset points",
                xytext=(10 if fps else 0, 6 if fps else (18 if not label.startswith("GT") else -18)),
                fontsize=8.5, ha=ha, color=col, fontweight="bold")
    ax.annotate(note, (x, c4), textcoords="offset points",
                xytext=(10 if fps else 0, -6 if fps else (6 if not label.startswith("GT") else -34)),
                fontsize=7, ha=ha, color="0.4")

ax.axvline(0, color="0.7", lw=0.8)
ax.axhline(0.5, color="0.9", lw=0.6, zorder=0)
ax.set_xlim(-6, 30)
ax.set_ylim(0.35, 1.03)
ax.set_xlabel("on-board FPS (Orin Nano 8GB)")
ax.set_ylabel("balanced C4 success (5 classes)")
ax.set_title("Metric-3D foundation models: navigation vs deployability", fontsize=12)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.12, lw=0.6)
fig.tight_layout()
out = os.environ.get("MERLIN_FIG_OUT", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "results", "swapc", "capability_vs_deploy.png"))
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=170)
print("wrote", out)
