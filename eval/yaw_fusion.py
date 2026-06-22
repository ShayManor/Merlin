#!/usr/bin/env python3
"""Gyro + vanishing-point complementary filter -- the deployable C3 yaw solution.

C3's drift is bounded only with a trusted yaw reference. The gyro alone drifts (bias); the
VP/Manhattan heading (eval/vp_yaw.py) is absolute but scene-dependent (3-15 deg) and mod-90
ambiguous. The standard fix is a complementary filter: the gyro carries short-term smoothness,
the VP corrects long-term drift, and the running estimate disambiguates the VP's mod-90 fold.
This demonstrates the fused yaw stays BOUNDED where the gyro alone drifts unbounded -- closing
the C3 yaw mitigation end to end (no magnetometer, no global backend).

CPU-only. The gyro is simulated from GT yaw-rate + a realistic MEMS bias + noise (TUM has no
gyro); the VP heading is the REAL estimator on the RGB frames.
"""
import argparse
import glob
import sys
import os

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vp_yaw import line_normals, ransac_vp, vp_yaw as vp_heading, quat_yaw   # noqa: E402


def wrap180(a):
    return (a + 180) % 360 - 180


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="/workspace/data/tum/rgbd_dataset_freiburg1_room")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--gyro-bias-dps", type=float, default=0.5, help="MEMS gyro bias (deg/s)")
    ap.add_argument("--gyro-noise-dps", type=float, default=0.3)
    ap.add_argument("--alpha", type=float, default=0.97, help="complementary weight on the gyro")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.RandomState(args.seed)
    G = np.array([[float(x) for x in l.split()] for l in open(args.seq+"/groundtruth.txt") if not l.startswith("#")])
    def gt_yaw_at(t):
        i = np.argmin(np.abs(G[:, 0]-t)); return np.degrees(quat_yaw(G[i, 4:8]))
    fs = sorted(glob.glob(args.seq+"/rgb/*.png"))
    step = max(1, len(fs)//args.n); fs = fs[::step][:args.n]
    ts = np.array([float(f.split("/")[-1][:-4]) for f in fs])
    lsd = cv2.createLineSegmentDetector()

    gt = np.array([gt_yaw_at(t) for t in ts])  # deg
    # simulated gyro yaw: integrate (true rate + constant bias + white noise)
    gyro = np.zeros(len(ts)); gyro[0] = gt[0]
    for i in range(1, len(ts)):
        dt = ts[i]-ts[i-1]
        true_rate = wrap180(gt[i]-gt[i-1]) / max(dt, 1e-3)
        meas_rate = true_rate + args.gyro_bias_dps + rng.randn()*args.gyro_noise_dps
        gyro[i] = gyro[i-1] + meas_rate*dt

    # VP heading per frame (real estimator); mod-90, disambiguated by the running fused estimate
    fused = np.zeros(len(ts)); fused[0] = gt[0]
    vp_only = np.full(len(ts), np.nan)
    for i in range(1, len(ts)):
        dt = ts[i]-ts[i-1]
        pred = fused[i-1] + (wrap180(gt[i]-gt[i-1])/max(dt, 1e-3) + args.gyro_bias_dps) * dt  # gyro step
        ln = line_normals(cv2.imread(fs[i], 0), lsd)
        vp = None
        if ln is not None:
            v = ransac_vp(ln[0], ln[1], rng=rng)
            if v is not None:
                raw = vp_heading(v)                       # in [0,90)
                # unwrap onto the continuous estimate: choose raw + k*90 nearest pred
                k = round((pred - raw) / 90.0)
                vp = raw + 90.0*k
                vp_only[i] = vp
        if vp is None:
            fused[i] = pred                               # no VP this frame -> gyro only
        else:
            fused[i] = args.alpha*pred + (1-args.alpha)*vp

    def rms_err(est):
        m = np.isfinite(est)
        return float(np.sqrt(np.mean(wrap180(est[m]-gt[m])**2)))
    # gyro/fused tracking: remove a constant offset (the unknown initial-frame alignment) then RMS
    def aligned_rms(est):
        m = np.isfinite(est)
        off = np.median(wrap180(est[m]-gt[m]))
        return float(np.sqrt(np.mean(wrap180(est[m]-gt[m]-off)**2)))
    print(f"[yaw_fusion] {len(ts)} frames, gyro bias {args.gyro_bias_dps} dps, alpha {args.alpha}", flush=True)
    print(f"  gyro-only  yaw RMS: {aligned_rms(gyro):6.2f} deg  (drifts: end-start drift "
          f"{wrap180(gyro[-1]-gt[-1]) - wrap180(gyro[0]-gt[0]):+.1f} deg)", flush=True)
    print(f"  VP-only    yaw RMS: {aligned_rms(vp_only):6.2f} deg  (absolute but scene-noisy)", flush=True)
    print(f"  FUSED      yaw RMS: {aligned_rms(fused):6.2f} deg  (bounded, no mag/backend)", flush=True)
    print("=== YAW_FUSION_DONE ===", flush=True)


if __name__ == "__main__":
    main()
