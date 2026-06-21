#!/usr/bin/env python3
"""C3: does the IMU bound pose drift over distance, without a global backend?

The decoupled architecture (student dense depth + VO/IMU pose) integrates relative poses, so
error accumulates. The C3 claim is that the IMU bounds it: gravity is an ABSOLUTE reference
for 2 of 3 orientation DOF (roll/pitch), and the C2 module anchors scale -- so only YAW drifts
freely (no magnetometer here). This characterizes that on real TUM trajectories, CPU-only (no
model, no sim): take the GT trajectory as ground truth, simulate a VIO front end by injecting a
rotation random-walk (gyro-bias-like drift) + a small C2 scale error into the relative poses,
integrate, and measure ATE vs path length for:
  - NO-IMU: full 3-DOF rotation drift (roll+pitch+yaw all walk).
  - IMU-GRAVITY: gravity (from the accelerometer low-freq = down) resets the roll/pitch
    components of the drift each keyframe; only yaw accumulates.
If IMU-GRAVITY drift << NO-IMU drift, the IMU bounds drift as C3 claims. TUM is short (<~50 m),
so this measures the drift RATE and the gravity-bounding effect, not the full 100 m claim.
"""
import argparse
import glob
import os

import numpy as np


def quat_to_R(q):
    x, y, z, w = q
    n = (x*x+y*y+z*z+w*w) ** 0.5 + 1e-12
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])


def so3_exp(w):
    th = np.linalg.norm(w)
    if th < 1e-9:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th)*K + (1-np.cos(th))*(K@K)


def load_tum(seq):
    G = np.array([[float(x) for x in l.split()] for l in open(seq+"/groundtruth.txt") if not l.startswith("#")])
    A = np.array([[float(x) for x in l.split()] for l in open(seq+"/accelerometer.txt") if not l.startswith("#")])
    return G, A


def grav_body(A, at, t):
    """Low-frequency accelerometer = gravity direction in the body frame at time t."""
    m = np.abs(A[:, 0] - t) < 0.15
    if m.sum() < 3:
        m = np.abs(A[:, 0] - t) < 0.5
    g = A[m, 1:4].mean(0)
    return g / (np.linalg.norm(g) + 1e-9)


def run(seq, sigma_deg, scale_err, rng, stride=10):
    G, A = load_tum(seq)
    fs = sorted(glob.glob(seq+"/rgb/*.png"))
    ts = np.array([float(f.split("/")[-1][:-4]) for f in fs])[::stride]
    # GT poses at keyframe times
    def gt_at(t):
        i = np.argmin(np.abs(G[:, 0]-t)); return G[i, 1:4], quat_to_R(G[i, 4:8])
    P, Rk = zip(*[gt_at(t) for t in ts])
    P = np.array(P); Rk = list(Rk)
    sigma = np.deg2rad(sigma_deg)

    def integrate(gravity_correct):
        est_p = [P[0].copy()]; est_R = Rk[0].copy()
        for i in range(1, len(ts)):
            dR_gt = Rk[i-1].T @ Rk[i]                       # true relative rotation
            drift = so3_exp(rng.randn(3) * sigma)           # per-keyframe VIO rotation drift
            dR = dR_gt @ drift
            dt_gt = Rk[i-1].T @ (P[i] - P[i-1])             # true relative translation (body)
            dt = dt_gt * (1.0 + scale_err)                  # C2 scale error
            est_R = est_R @ dR
            if gravity_correct:
                # gravity gives absolute down -> correct the roll/pitch of est_R so its
                # gravity direction matches the accel-measured one (only yaw left to drift).
                g_meas = grav_body(A, A[:, 0], ts[i])       # gravity in body (accel)
                g_world_est = est_R @ g_meas                # where est thinks down is, in world
                g_world_true = np.array([0, -1.0, 0])       # TUM world: -y is down-ish (approx)
                # rotation that aligns est's gravity to true gravity (about the horiz axis)
                v = np.cross(g_world_est, g_world_true); s = np.linalg.norm(v)
                if s > 1e-6:
                    c = float(np.dot(g_world_est, g_world_true))
                    ang = np.arctan2(s, c)
                    est_R = so3_exp(v/s * ang) @ est_R
            est_p.append(est_p[-1] + est_R @ dt)
        return np.array(est_p)

    def ate(est):
        # umeyama align (rot+trans, no scale) then RMSE
        X, Y = est, P
        mx, my = X.mean(0), Y.mean(0); Xc, Yc = X-mx, Y-my
        U, S, Vt = np.linalg.svd(Yc.T@Xc); R = U@Vt
        if np.linalg.det(R) < 0: Vt[-1] *= -1; R = U@Vt
        al = (R@Xc.T).T + my
        return np.sqrt(((al-Y)**2).sum(1).mean())

    path_len = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    return ate(integrate(False)), ate(integrate(True)), path_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", default=[
        "/workspace/data/tum/rgbd_dataset_freiburg1_room",
        "/workspace/data/tum/rgbd_dataset_freiburg1_desk",
        "/workspace/data/tum/rgbd_dataset_freiburg2_desk"])
    ap.add_argument("--sigma-deg", type=float, default=0.3, help="per-keyframe VIO rotation drift std")
    ap.add_argument("--scale-err", type=float, default=0.02, help="C2 scale error")
    ap.add_argument("--trials", type=int, default=10)
    args = ap.parse_args()
    rng = np.random.RandomState(0)
    print(f"[c3] VIO drift sigma={args.sigma_deg} deg/kf, C2 scale_err={args.scale_err}", flush=True)
    for seq in args.seqs:
        if not os.path.exists(seq):
            continue
        noimu, imu, pl = [], [], 0
        for _ in range(args.trials):
            a, b, pl = run(seq, args.sigma_deg, args.scale_err, rng)
            noimu.append(a); imu.append(b)
        nm, im = np.median(noimu), np.median(imu)
        name = seq.split("/")[-1].replace("rgbd_dataset_", "")
        print(f"  {name}: path {pl:.1f}m | NO-IMU ATE {nm:.3f}m ({100*nm/pl:.1f}%) | "
              f"IMU-gravity ATE {im:.3f}m ({100*im/pl:.1f}%) | drift cut {100*(1-im/max(nm,1e-6)):.0f}%", flush=True)
    print("=== C3_DRIFT_DONE ===", flush=True)


if __name__ == "__main__":
    main()
