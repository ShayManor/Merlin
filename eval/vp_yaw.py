#!/usr/bin/env python3
"""Vanishing-point / Manhattan-frame yaw estimator (the C3 indoor yaw reference).

C3's drift bound needs a yaw reference, and the indoor magnetometer is unreliable. Indoor
scenes are near-Manhattan (orthogonal walls/edges), so the visual input gives an ABSOLUTE
heading without a magnetometer or a global backend. This is a proper estimator (vs the crude
LSD-direction-median proof-of-concept): detect line segments, find the dominant horizontal
vanishing point by RANSAC on the calibrated interpretation-plane normals, and read off the
camera azimuth w.r.t. that wall direction. We validate that the recovered yaw tracks GT yaw
(delta is constant up to the unknown room-vs-world offset) -> std of (vp_yaw - gt_yaw) is the
residual yaw uncertainty the reference would inject (lower than the ~7 deg crude version).

CPU-only (OpenCV LSD + numpy). TUM freiburg1 intrinsics. Manhattan/VP heading is mod 90 deg
(walls come in two orthogonal families), so we unwrap against GT to measure tracking.
"""
import argparse
import glob

import cv2
import numpy as np

K = np.array([[517.3, 0, 318.6], [0, 516.5, 255.3], [0, 0, 1.0]])
Kinv = np.linalg.inv(K)


def quat_yaw(q):
    x, y, z, w = q
    # yaw about world-up (TUM world); use atan2 of the forward vector's horizontal components
    return np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def line_normals(im_gray, lsd, min_len=40):
    lines = lsd.detect(im_gray)[0]
    if lines is None:
        return None
    segs = lines.reshape(-1, 4)
    L = np.hypot(segs[:, 2]-segs[:, 0], segs[:, 3]-segs[:, 1])
    segs = segs[L > min_len]; L = L[L > min_len]
    if len(segs) < 6:
        return None
    p1 = np.c_[segs[:, 0], segs[:, 1], np.ones(len(segs))]
    p2 = np.c_[segs[:, 2], segs[:, 3], np.ones(len(segs))]
    n = np.cross(p1 @ Kinv.T, p2 @ Kinv.T)        # interpretation-plane normal (calibrated)
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    return n, L


def ransac_vp(n, L, iters=400, thresh_deg=1.5, rng=None, horiz_max=0.35):
    """Dominant HORIZONTAL vanishing point: a direction v perpendicular to many plane normals
    (|n.v| small). Constrained to horizontal (|v_y| small) so it locks onto a wall direction,
    not the vertical VP (vertical edges) whose azimuth is meaningless for yaw."""
    best_v, best_in, best_mask = None, -1, None
    th = np.sin(np.deg2rad(thresh_deg))
    idx = np.arange(len(n))
    for _ in range(iters):
        i, j = rng.choice(idx, 2, replace=False)
        v = np.cross(n[i], n[j]); nv = np.linalg.norm(v)
        if nv < 1e-9:
            continue
        v /= nv
        if abs(v[1]) > horiz_max:                  # reject near-vertical VPs (image-y ~ up)
            continue
        inl = np.abs(n @ v) < th
        score = L[inl].sum()
        if score > best_in:
            best_in, best_v, best_mask = score, v, inl
    if best_v is None or best_mask is None:
        return None
    A = n[best_mask] * L[best_mask, None]
    _, _, Vt = np.linalg.svd(A)
    v = Vt[-1]
    return v if abs(v[1]) < horiz_max else best_v   # keep horizontal after refine


def vp_yaw(v):
    """Azimuth of a (horizontal) vanishing direction in the camera frame -> heading proxy.
    Camera looks -z; horizontal VPs lie near the x-z plane. Use atan2(vx, vz), folded mod 90."""
    az = np.arctan2(v[0], v[2])
    return (np.degrees(az) % 90.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="/workspace/data/tum/rgbd_dataset_freiburg1_room")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.RandomState(args.seed)
    G = np.array([[float(x) for x in l.split()] for l in open(args.seq+"/groundtruth.txt") if not l.startswith("#")])
    def gt_yaw_at(t):
        i = np.argmin(np.abs(G[:, 0]-t)); return np.degrees(quat_yaw(G[i, 4:8]))
    fs = sorted(glob.glob(args.seq+"/rgb/*.png"))
    step = max(1, len(fs)//args.n); fs = fs[::step][:args.n]
    lsd = cv2.createLineSegmentDetector()

    rows = []
    for f in fs:
        im = cv2.imread(f, 0)
        ln = line_normals(im, lsd)
        if ln is None:
            continue
        v = ransac_vp(ln[0], ln[1], rng=rng)
        if v is None:
            continue
        t = float(f.split("/")[-1][:-4])
        rows.append((vp_yaw(v), gt_yaw_at(t)))
    if len(rows) < 5:
        print("too few frames"); return
    vpy = np.array([r[0] for r in rows]); gty = np.array([r[1] for r in rows]) % 90.0
    # the VP heading tracks GT up to a constant room offset; remove the circular-median offset (mod 90)
    d = (vpy - gty + 45) % 90 - 45
    off = np.median(d)
    resid = (d - off + 45) % 90 - 45
    print(f"[vp_yaw] {len(rows)} frames", flush=True)
    print(f"  VP-heading tracking residual (after room-offset): std {resid.std():.2f} deg, "
          f"median |err| {np.median(np.abs(resid)):.2f} deg", flush=True)
    print(f"  (crude LSD-median proof-of-concept was ~7 deg; proper VP-RANSAC should be tighter)", flush=True)
    print("=== VP_YAW_DONE ===", flush=True)


if __name__ == "__main__":
    main()
