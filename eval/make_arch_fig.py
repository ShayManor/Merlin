#!/usr/bin/env python3
"""MERLIN system-architecture figure: the two-board split. The Jetson runs ONE node (the model,
inference only); the Rubik Pi companion does IMU fusion (scale/drift), map assembly, Nav2, control.
Only compact depth + pose cross the wire -- never raw dense point clouds. Clean/minimal."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "swapc")
os.makedirs(ROOT, exist_ok=True)

fig, ax = plt.subplots(figsize=(8.2, 3.7))
ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")

def box(x, y, w, h, title, lines, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2",
                                fc=fc, ec=ec, lw=2))
    ax.text(x+w/2, y+h-4.5, title, ha="center", va="top", fontsize=10, fontweight="bold", color=ec)
    ax.text(x+w/2, y+h-11, "\n".join(lines), ha="center", va="top", fontsize=8, color="0.25")

def arrow(x1, y1, x2, y2, label="", col="0.35", lab_dy=2.2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 lw=1.8, color=col))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2+lab_dy, label, ha="center", fontsize=7.5, color=col)

# inputs
ax.text(5, 38, "USB cam", ha="center", fontsize=8.5, fontweight="bold")
ax.text(5, 12, "IMU\n(MPU-9250)", ha="center", fontsize=8.5, fontweight="bold")
# Jetson model node
box(20, 26, 30, 18, "Jetson Orin Nano 8GB  (~$249, ~10W)",
    ["MODEL NODE ONLY (inference)", "RGB -> metric depth + rel pose",
     "230M distilled, INT8-TRT", "20 FPS / 49ms / 10.4W"], "#eaf2f7", "#2e86ab")
# Rubik Pi companion
box(64, 14, 32, 30, "Rubik Pi 3 companion  (on-rover)",
    ["IMU fusion: metric scale (C2)", "drift bounding (C3)",
     "map assembly", "Nav2 + control"], "#f4efe6", "#b9770e")

arrow(9, 38, 20, 36)                                  # cam -> Jetson
arrow(9, 13, 64, 18, "accel/gyro", "#b9770e", -3)     # IMU -> Rubik
# the wire
arrow(50, 35, 64, 35, "compact depth + pose", "#2e86ab", 2.4)
ax.text(57, 31.5, "(NEVER raw clouds)", ha="center", fontsize=7, color="#d1495b", style="italic")
ax.text(57, 40, "DDS / Ethernet", ha="center", fontsize=7.5, color="0.4")

ax.text(50, 3, "All compute on the rover. No base station, no datacenter, no Thor-class accelerator.",
        ha="center", fontsize=8, color="0.4")
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "architecture.png"), dpi=130)
print("saved", os.path.join(ROOT, "architecture.png"))
