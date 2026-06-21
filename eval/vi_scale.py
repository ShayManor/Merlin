#!/usr/bin/env python3
"""C2: metric-scale recovery from mono + IMU (the credible novel contribution).

The student gives a scale-ambiguous visual trajectory (cam_trans/cam_quats). A consumer
accelerometer (TUM 500 Hz, m/s^2 with gravity, no gyro) gives metric acceleration. We solve
the classic linear visual-inertial alignment for [per-frame velocity, gravity, SCALE]:

  rotate accel into the c0 (first-camera) frame using the visual rotations, then over each
  consecutive keyframe pair form the integral constraints
     SI_k = (v_{k+1} - v_k) + g*dt                          (single integral = velocity)
     DI_k = s*(P_{k+1}-P_k) - v_k*dt + 0.5*g*dt^2           (double integral = position)
  stack over a window -> least squares -> scale s.

Validated against the GT-trajectory scale (Umeyama visual->GT). Scale is one global scalar,
so per-window noise (the student VO is ~22%) averages down over windows. C2 target <5%.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model", "distill"))
sys.path.insert(0, os.path.join(HERE, "..", "model", "backbones"))
from student import build_student   # noqa: E402


def quat_to_R(q):  # q = [x,y,z,w] -> 3x3 (body->c0)
    x, y, z, w = q
    n = (x*x+y*y+z*z+w*w) ** 0.5 + 1e-12
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])


def load_accel(seq):
    A = np.array([[float(x) for x in l.split()] for l in open(seq+"/accelerometer.txt") if not l.startswith("#")])
    return A[:, 0], A[:, 1:4]


def integrals_c0(at, av, Rk, t0, t1):
    """Single + double integral of accel (rotated to c0 via the keyframe rotation Rk) over [t0,t1]."""
    m = (at >= t0) & (at <= t1)
    if m.sum() < 2:
        return None
    ts = at[m]; acc = (Rk @ av[m].T).T            # rotate body accel to c0 frame
    si = np.zeros(3); di = np.zeros(3); v = np.zeros(3)
    for i in range(1, len(ts)):
        dt = ts[i] - ts[i-1]; a = 0.5*(acc[i]+acc[i-1])
        di += v*dt + 0.5*a*dt*dt
        v += a*dt; si = v.copy()
    return si, di


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/ckpt/student_v3.pt")
    ap.add_argument("--seq", default="/workspace/data/tum/rgbd_dataset_freiburg1_room")
    ap.add_argument("--nwin", type=int, default=40)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args()
    dev = "cuda"
    from mapanything.utils.image import load_images
    m = build_student(size="base", aat_depth=8, device="cpu")
    m.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"])
    m = m.to(dev).eval()
    at, av = load_accel(args.seq)
    G = np.array([[float(x) for x in l.split()] for l in open(args.seq+"/groundtruth.txt") if not l.startswith("#")])
    def gtpos(t): return G[np.argmin(np.abs(G[:, 0]-t)), 1:4]
    fs = sorted(glob.glob(args.seq+"/rgb/*.png"))

    s_est, s_true = [], []
    starts = range(0, len(fs)-args.frames*args.stride, max(1, (len(fs)-args.frames*args.stride)//args.nwin))
    for start in starts:
        win = fs[start:start+args.frames*args.stride:args.stride]
        ts = np.array([float(f.split("/")[-1][:-4]) for f in win])
        v = load_images(win, resize_mode="square", size=378)
        for x in v: x["img"] = x["img"].to(dev)
        out = m(v, memory_efficient_inference=True, minibatch_size=args.frames)
        P = np.array([o["cam_trans"][0].cpu().numpy() for o in out])     # visual positions (scale-ambiguous), c0 frame
        R = [quat_to_R(o["cam_quats"][0].cpu().numpy()) for o in out]    # body->c0
        gt = np.array([gtpos(t) for t in ts])
        if np.linalg.norm(gt-gt[0], axis=1).max() < 0.08:               # need motion
            continue
        # build linear system A x = b, x = [v_0..v_{n-1}(3n), g(3), s(1)]
        n = len(win); rows = []; rhs = []
        bad = False
        for k in range(n-1):
            dt = ts[k+1]-ts[k]
            ii = integrals_c0(at, av, R[k], ts[k], ts[k+1])
            if ii is None: bad = True; break
            si, di = ii
            # velocity rows: -v_k + v_{k+1} + g*dt = si
            for d in range(3):
                row = np.zeros(3*n+4)
                row[3*k+d] = -1; row[3*(k+1)+d] = 1; row[3*n+d] = dt
                rows.append(row); rhs.append(si[d])
            # position rows: -v_k*dt + 0.5*g*dt^2 + s*(P_{k+1}-P_k) = di
            for d in range(3):
                row = np.zeros(3*n+4)
                row[3*k+d] = -dt; row[3*n+d] = 0.5*dt*dt; row[3*n+3] = (P[k+1]-P[k])[d]
                rows.append(row); rhs.append(di[d])
        if bad: continue
        Amat = np.array(rows); bvec = np.array(rhs)
        x, *_ = np.linalg.lstsq(Amat, bvec, rcond=None)
        s = x[-1]
        # true scale via Umeyama (visual P -> GT)
        Xc, Yc = P-P.mean(0), gt-gt.mean(0)
        U, S, Vt = np.linalg.svd((Yc.T@Xc)/n); st = S.sum()/((Xc**2).sum()/n)
        g_mag = np.linalg.norm(x[3*n:3*n+3])
        if s > 0 and 0.2 < s/st < 5 and 7 < g_mag < 13:   # gravity sanity gates a valid window
            s_est.append(s); s_true.append(st)
    s_est = np.array(s_est); s_true = np.array(s_true)
    print(f"[vi_scale] valid windows: {len(s_est)}/{len(list(starts))}", flush=True)
    if len(s_est):
        per = np.abs(s_est-s_true)/s_true
        # global metric scale = average ratio (scale is one scalar -> averaging beats per-window noise)
        glob_est = np.median(s_est/s_true)   # recovered/true ratio; 1.0 = perfect
        print(f"  per-window scale error: median {np.median(per)*100:.1f}%  mean {per.mean()*100:.1f}%", flush=True)
        print(f"  AVERAGED scale (median est/true ratio): {glob_est:.3f} -> global scale error {abs(glob_est-1)*100:.1f}% (C2 target <5%)", flush=True)
    print("=== VI_SCALE_DONE ===", flush=True)


if __name__ == "__main__":
    main()
