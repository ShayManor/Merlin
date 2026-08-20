#!/usr/bin/env python3
"""C3 drift-bounding figure: ATE (% of path) vs per-keyframe VIO rotation-drift rate, on the
TUM fr1_room GT trajectory (15.8 m), 15 trials/point, C2 scale error 2%. Shows the IMU's
absolute references bound drift to the scale-error floor while dead-reckoned rotation diverges.
Sim only (GT traj + synthetic VIO noise, short path) -- the 100 m+ real-traversal validation is
future work. Honest: gyro+accel needs a visual-yaw reference indoors (mag unreliable); 9-axis
here stands in for gyro+accel + the VP/Manhattan visual-yaw fusion."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "c3")
os.makedirs(ROOT, exist_ok=True)

sigma = [0.3, 0.6, 1.0, 1.5, 2.0, 3.0]
noimu = [1.6, 1.7, 1.9, 2.3, 2.6, 3.3]
gyro  = [1.5, 1.5, 1.5, 1.7, 1.8, 2.3]
nine  = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5]

fig, ax = plt.subplots(figsize=(5.8, 3.8))
ax.plot(sigma, noimu, "o-", color="#d1495b", lw=2, label="no IMU (3-DOF dead reckoning)")
ax.plot(sigma, gyro,  "s-", color="#e0a458", lw=2, label="gyro+accel (yaw drifts indoors)")
ax.plot(sigma, nine,  "^-", color="#2e86ab", lw=2, label="9-axis / +visual-yaw (bounded)")
ax.axhspan(0, 2.0, color="#2e86ab", alpha=0.05)
ax.text(0.32, 1.62, "C2 scale-error floor", fontsize=8, color="#2e86ab")
ax.set_xlabel("per-keyframe VIO rotation-drift rate (deg)")
ax.set_ylabel("ATE (% of path length)")
ax.set_title("C3: the IMU bounds drift to the scale-error floor", fontsize=11)
ax.set_ylim(1.0, 3.6)
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
ax.text(0.98, 0.02, "TUM fr1_room (15.8 m), sim; 100 m+ real traversal = future work",
        transform=ax.transAxes, ha="right", fontsize=7, color="0.5")
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "c3_drift.png"), dpi=130)
print("saved", os.path.join(ROOT, "c3_drift.png"))
