#!/usr/bin/env python3
"""C2 decoupled-architecture test: does IMU metric-scale recovery work with GOOD poses?

The student-pose VI estimator (vi_scale.py) failed because the student's ego-motion is too
noisy. This isolates that: feed the SAME linear VI scale solver GT-quality camera rotations
(stand-in for a classical VO / decoupled pose front end) instead of student poses. If metric
scale then recovers to <5%, it confirms (a) the IMU + linear formulation are sound, (b) the
only blocker was student pose noise, and (c) the decoupled architecture (student=dense depth,
classical VO=pose) solves C2. CPU-only (GT poses + accelerometer, no model inference).

Steps: calibrate the accel->camera extrinsic R_ac by making gravity constant in world (needs
good rotations, which GT provides), then solve the per-window linear system for scale s using
GT metric positions as the 'visual' trajectory -> expect s ~ 1.0.
"""
import argparse
import numpy as np


def quat_to_R(q):  # [x,y,z,w] -> R_wc (camera->world), TUM groundtruth convention
    x, y, z, w = q
    n = (x*x+y*y+z*z+w*w) ** 0.5 + 1e-12
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])


def kabsch(X, Y):  # R mapping X->Y
    H = X.T @ Y; U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def integ(at, av, Rfn, t0, t1):
    m = (at >= t0) & (at <= t1)
    if m.sum() < 2:
        return None
    ts = at[m]; acc = np.array([Rfn(t) @ av[i] for t, i in zip(ts, np.where(m)[0])])
    v = np.zeros(3); di = np.zeros(3)
    for i in range(1, len(ts)):
        dt = ts[i]-ts[i-1]; a = 0.5*(acc[i]+acc[i-1])
        di += v*dt + 0.5*a*dt*dt; v += a*dt
    return v, di  # SI (velocity), DI (position)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="/workspace/data/tum/rgbd_dataset_freiburg1_room")
    ap.add_argument("--nwin", type=int, default=60)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args()
    A = np.array([[float(x) for x in l.split()] for l in open(args.seq+"/accelerometer.txt") if not l.startswith("#")])
    at, av = A[:, 0], A[:, 1:4]
    G = np.array([[float(x) for x in l.split()] for l in open(args.seq+"/groundtruth.txt") if not l.startswith("#")])
    gt_t = G[:, 0]
    def gpos(t): return G[np.argmin(np.abs(gt_t-t)), 1:4]
    def gR(t): return quat_to_R(G[np.argmin(np.abs(gt_t-t)), 4:8])

    # 1) calibrate accel->camera extrinsic R_ac via gravity-constant-in-world (low-motion samples)
    lm = np.abs(np.linalg.norm(av, axis=1) - 9.81) < 0.4
    idx = np.where(lm)[0][::20][:400]
    Rs = np.array([gR(at[i]) for i in idx]); ac = av[idx]
    Rac = np.eye(3); g = np.array([0, 0, 9.81])
    for _ in range(25):
        W = np.array([Rs[k].T @ g for k in range(len(ac))])
        Rac = kabsch(ac, W)
        g = np.mean([Rs[k] @ Rac @ ac[k] for k in range(len(ac))], 0)
    gstd = np.std([Rs[k] @ Rac @ ac[k] for k in range(len(ac))], 0)
    print(f"[calib] |g|={np.linalg.norm(g):.2f}  gravity-in-world std={gstd.round(2)} (low=good extrinsic)", flush=True)
    Rfn = lambda t: gR(t) @ Rac    # accel -> world

    # 2) VI scale solve per window using GT metric positions as 'visual' -> expect s~1
    import glob
    fs = sorted(glob.glob(args.seq+"/rgb/*.png"))
    s_est = []
    starts = list(range(0, len(fs)-args.frames*args.stride, max(1, (len(fs)-args.frames*args.stride)//args.nwin)))
    for start in starts:
        win = fs[start:start+args.frames*args.stride:args.stride]
        ts = np.array([float(f.split("/")[-1][:-4]) for f in win])
        P = np.array([gpos(t) for t in ts])
        if np.linalg.norm(P-P[0], axis=1).max() < 0.1:
            continue
        n = len(win); rows = []; rhs = []; bad = False
        for k in range(n-1):
            dt = ts[k+1]-ts[k]; ii = integ(at, av, Rfn, ts[k], ts[k+1])
            if ii is None: bad = True; break
            si, di = ii
            for d in range(3):
                r = np.zeros(3*n+4); r[3*k+d] = -1; r[3*(k+1)+d] = 1; r[3*n+d] = dt
                rows.append(r); rhs.append(si[d])
            for d in range(3):
                r = np.zeros(3*n+4); r[3*k+d] = -dt; r[3*n+d] = 0.5*dt*dt; r[3*n+3] = (P[k+1]-P[k])[d]
                rows.append(r); rhs.append(di[d])
        if bad: continue
        x, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
        s = x[-1]
        if 0.2 < s < 5:
            s_est.append(s)
    s_est = np.array(s_est)
    print(f"[vi_gt] valid windows {len(s_est)}/{len(starts)}", flush=True)
    if len(s_est):
        # GT positions are metric so true s=1; error = |median(s)-1|
        med = np.median(s_est)
        print(f"  recovered scale: median {med:.3f} (true=1.0) -> scale error {abs(med-1)*100:.1f}% (C2 target <5%)", flush=True)
        print(f"  per-window scale std {np.std(s_est):.3f}", flush=True)
    print("=== VI_GT_DONE ===", flush=True)


if __name__ == "__main__":
    main()
