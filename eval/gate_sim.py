#!/usr/bin/env python3
"""EXP-G: metric-residual vs appearance vs fixed-rate gating for compute-skip.

Tests the M-E wedge: a cheap absolute-metric ToF residual predicts when a reused
(ego-motion-warped) dense depth map has gone stale BETTER than an internal
appearance proxy (Eventful-style) or a fixed refresh rate. If true, an
innovation-gated foundation-model refresh skips more frames at the same depth
error -> higher model-slice FPS/watt.

Runs on TUM RGB-D (GT depth + GT pose). Pure numpy; no model forward needed --
staleness is a geometric, scene-motion-driven property (the model adds a roughly
constant fidelity offset on top). 8x8 ToF is simulated from GT depth.

Outputs JSON: per-gate Spearman(signal, true_reuse_error), skip-rate/error Pareto,
and the two adversarial-case numbers (lighting-over-static, smooth-motion-to-wall).
"""
import argparse, json, os, glob
import numpy as np
from PIL import Image

INTR = {  # fx, fy, cx, cy  (TUM defaults per camera)
    "freiburg1": (517.306, 516.469, 318.643, 255.314),
    "freiburg2": (520.909, 521.007, 325.142, 249.701),
    "freiburg3": (535.4,   539.2,   320.1,   247.6),
}
DEPTH_SCALE = 5000.0


def read_list(path):
    out = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            p = ln.split()
            out.append((float(p[0]), p[1:]))
    return out


def associate(a, b, max_dt=0.02):
    """nearest-neighbour timestamp match a->b."""
    bt = np.array([t for t, _ in b])
    pairs = []
    for ta, va in a:
        j = int(np.argmin(np.abs(bt - ta)))
        if abs(bt[j] - ta) <= max_dt:
            pairs.append((ta, va, b[j][1]))
    return pairs


def quat_to_T(p):
    tx, ty, tz, qx, qy, qz, qw = map(float, p)
    n = np.sqrt(qx*qx+qy*qy+qz*qz+qw*qw) + 1e-12
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    R = np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
    ])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [tx, ty, tz]
    return T  # camera-to-world


def load_depth(path):
    d = np.asarray(Image.open(path)).astype(np.float32) / DEPTH_SCALE
    return d  # meters, 0 = invalid


def load_gray(path):
    return np.asarray(Image.open(path).convert("L")).astype(np.float32)


def reproject(depth_k, T_k, T_t, K):
    """Render keyframe depth into frame t (reproject + z-buffer). Returns D_reuse[t]."""
    fx, fy, cx, cy = K
    H, W = depth_k.shape
    ys, xs = np.nonzero(depth_k > 0)
    z = depth_k[ys, xs]
    X = (xs - cx) / fx * z
    Y = (ys - cy) / fy * z
    pts = np.stack([X, Y, z, np.ones_like(z)], 0)          # 4xN cam-k
    Pkt = np.linalg.inv(T_t) @ T_k                          # cam-k -> cam-t
    q = Pkt @ pts
    zc = q[2]
    valid = zc > 1e-3
    q, zc = q[:, valid], zc[valid]
    u = (q[0] / zc * fx + cx).round().astype(np.int32)
    v = (q[1] / zc * fy + cy).round().astype(np.int32)
    m = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, zc = u[m], v[m], zc[m]
    out = np.full((H, W), np.inf, np.float32)
    flat = v * W + u
    order = np.argsort(-zc)                                  # far first, near overwrites
    np.minimum.at(out.reshape(-1), flat[order], zc[order])
    out[~np.isfinite(out)] = 0.0
    return out


def abs_rel(pred, gt):
    m = (pred > 0) & (gt > 0)
    if m.sum() < 200:
        return np.nan
    return float(np.mean(np.abs(pred[m] - gt[m]) / gt[m]))


def tof_residual(pred, sensor_gt, grid=8):
    """median over 8x8 zones of |median(pred)-median(sensor)|/median(sensor)."""
    H, W = sensor_gt.shape
    hs, ws = H // grid, W // grid
    res = []
    for i in range(grid):
        for j in range(grid):
            ps = pred[i*hs:(i+1)*hs, j*ws:(j+1)*ws]
            ss = sensor_gt[i*hs:(i+1)*hs, j*ws:(j+1)*ws]
            pv, sv = ps[ps > 0], ss[ss > 0]
            if len(pv) < 5 or len(sv) < 5:
                continue
            mp, ms = np.median(pv), np.median(sv)
            res.append(abs(mp - ms) / ms)
    return float(np.median(res)) if res else np.nan


def spearman(a, b):
    a, b = np.asarray(a), np.asarray(b)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan
    ra = np.argsort(np.argsort(a[m])); rb = np.argsort(np.argsort(b[m]))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    return float((ra @ rb) / (np.sqrt(ra @ ra) * np.sqrt(rb @ rb) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="TUM sequence dir")
    ap.add_argument("--stride", type=int, default=10, help="keyframe stride (frames)")
    ap.add_argument("--horizon", type=int, default=40, help="max coast horizon")
    ap.add_argument("--max_kf", type=int, default=60, help="cap #keyframes for speed")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seq = args.seq.rstrip("/")
    cam = [k for k in INTR if k in os.path.basename(seq)]
    K = INTR[cam[0]] if cam else INTR["freiburg1"]

    rgb = read_list(os.path.join(seq, "rgb.txt"))
    dep = read_list(os.path.join(seq, "depth.txt"))
    gt  = read_list(os.path.join(seq, "groundtruth.txt"))
    # associate depth->rgb and depth->pose
    rd = associate([(t, v) for t, v in dep], rgb)            # depth_t, depth_path, rgb_path
    frames = []
    gtt = np.array([t for t, _ in gt])
    for td, dpath, rpath in rd:
        j = int(np.argmin(np.abs(gtt - td)))
        if abs(gtt[j] - td) > 0.02:
            continue
        frames.append((td, os.path.join(seq, dpath[0]), os.path.join(seq, rpath[0]), gt[j][1]))
    frames.sort()
    N = len(frames)
    print(f"{os.path.basename(seq)}: {N} associated frames, cam={cam}")

    # sample keyframes
    kf_idx = list(range(0, N - 2, args.stride))[:args.max_kf]
    rows = []  # (true_reuse_err, tof_res, appear_delta, lag, k, t)
    for k in kf_idx:
        td_k, dpk, rpk, posek = frames[k]
        Dk = load_depth(dpk); Tk = quat_to_T(posek); Gk = load_gray(rpk)
        for t in range(k + 1, min(k + 1 + args.horizon, N)):
            td_t, dpt, rpt, poset = frames[t]
            Dt = load_depth(dpt); Tt = quat_to_T(poset); Gt = load_gray(rpt)
            Dreuse = reproject(Dk, Tk, Tt, K)
            err = abs_rel(Dreuse, Dt)                        # true staleness oracle
            tof = tof_residual(Dreuse, Dt)                   # metric gate signal
            app = float(np.mean(np.abs(Gt - Gk)) / 255.0)    # appearance gate signal (Eventful proxy)
            if np.isfinite(err):
                rows.append([err, tof, app, t - k, k, t])
    rows = np.array(rows, float)
    print(f"  {len(rows)} (k,t) reuse pairs")

    err, tof, app, lag = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    res = {
        "seq": os.path.basename(seq), "n_pairs": len(rows), "n_kf": len(kf_idx),
        "spearman_tof_vs_trueerr": spearman(tof, err),
        "spearman_appear_vs_trueerr": spearman(app, err),
        "spearman_lag_vs_trueerr": spearman(lag, err),
        "mean_true_err": float(np.nanmean(err)),
    }

    # Pareto: at matched skip-rate, mean true reuse-error on SKIPPED frames.
    # Each gate ranks pairs by its signal; "skip" = signal below threshold (predicted safe).
    def pareto(signal):
        s = signal.copy(); s[~np.isfinite(s)] = np.inf
        order = np.argsort(s)                                # safest-looking first
        out = []
        for frac in np.linspace(0.1, 0.9, 9):
            n = int(frac * len(order))
            skipped = order[:n]
            out.append([round(float(frac), 2), float(np.mean(err[skipped]))])
        return out
    res["pareto_tof"] = pareto(tof)        # [skip_frac, mean_err_on_skipped]
    res["pareto_appear"] = pareto(app)
    res["pareto_fixedrate"] = pareto(lag.astype(float))      # skip = low lag (recent), proxy for fixed-rate
    res["pareto_oracle"] = pareto(err)                       # lower bound: skip truly-lowest-error

    # Adversarial A: lighting change over static geometry. Take low-true-error pairs
    # (geometry essentially unchanged) and darken frame t by 0.5; appearance signal
    # should spike (over-fire) while ToF/true-err unchanged.
    safe = rows[err < np.nanpercentile(err, 25)]
    if len(safe) > 5:
        # recompute appearance under 0.5x brightness for these (gray scales linearly)
        app_dark = []
        for r in safe:
            k, t = int(r[4]), int(r[5])
            Gk = load_gray(frames[k][2]); Gt = load_gray(frames[t][2]) * 0.5
            app_dark.append(float(np.mean(np.abs(Gt - Gk)) / 255.0))
        res["advA_lighting"] = {
            "appear_normal_median": float(np.median(safe[:, 2])),
            "appear_dark_median": float(np.median(app_dark)),
            "tof_median": float(np.nanmedian(safe[:, 1])),
            "true_err_median": float(np.median(safe[:, 0])),
            "note": "static geometry; appear_dark spikes (over-fire), tof/true_err unchanged",
        }

    # Adversarial B: smooth motion, high true error but low appearance delta.
    # Fraction of high-error pairs that the appearance gate would WRONGLY skip
    # (appearance below median) vs ToF would correctly catch (ToF above median).
    hi = rows[err > np.nanpercentile(err, 75)]
    if len(hi) > 5:
        app_med, tof_med = np.nanmedian(app), np.nanmedian(tof)
        res["advB_discontinuity"] = {
            "n_high_err": len(hi),
            "frac_appear_would_skip": float(np.mean(hi[:, 2] < app_med)),
            "frac_tof_would_skip": float(np.mean(hi[:, 1] < tof_med)),
            "note": "high true-error pairs; lower frac_*_would_skip = better gate (fewer missed refreshes)",
        }

    out = args.out or f"/home/evc/merlin/tum/gate_{res['seq']}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
