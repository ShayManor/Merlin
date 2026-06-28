#!/usr/bin/env python3
"""EXP-RG: innovation-gated refresh measured END-TO-END through the REAL student.

Coast between refreshes by warping the REAL prior-keyframe depth (model intrinsics +
GT relative pose); refresh (re-run the real student) only when a gate fires. Compare a
metric gate (sim 8x8 ToF residual vs the coasted depth) against an appearance gate
(RGB delta) and a fixed rate. Sweep thresholds -> (skip-rate, output abs_rel vs GT)
Pareto. With the measured per-frame energy (0.027 + (1-h)*0.54 J GPU-rail) this gives
the real FPS/W-vs-accuracy curve.

Phase 1 runs the real model once per frame and caches depth_z + intrinsics (fast).
Phase 2 is pure numpy over the cache (warp + gate + select), so threshold sweeps are cheap.
"""
import sys, os, json, argparse
import numpy as np
from PIL import Image
import torch
sys.path.insert(0, "/home/evc/merlin/repo/model/backbones")
sys.path.insert(0, "/home/evc/merlin/repo/model/distill")
from teacher import patch_linalg_cpu_fallback, _upcast_geometry_postproc, add_dtype_casting_hooks
from student import build_student
from mapanything.utils.image import load_images
from mapanything.utils.cropping import crop_resize_if_necessary
from PIL.ImageOps import exif_transpose

SIZE = 378
DEPTH_SCALE = 5000.0

def load_model(ckpt="/home/evc/merlin/student_distilled.pt"):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = build_student(size=ck.get("size", "base"), aat_depth=ck.get("aat_depth", 8), device="cpu")
    m.load_state_dict(ck["state_dict"]); m = m.to(torch.bfloat16).eval()
    add_dtype_casting_hooks(m); _upcast_geometry_postproc(); patch_linalg_cpu_fallback()
    return m.to("cuda")

def quat_to_T(p):
    tx, ty, tz, qx, qy, qz, qw = map(float, p)
    n = (qx*qx+qy*qy+qz*qz+qw*qw) ** 0.5 + 1e-12; qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    R = np.array([[1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
                  [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
                  [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [tx, ty, tz]; return T

def assoc(seq):
    def rd(f):
        o = []
        for ln in open(os.path.join(seq, f)):
            ln = ln.strip()
            if ln and not ln.startswith("#"): p = ln.split(); o.append((float(p[0]), p[1:]))
        return o
    rgb, dep, gt = rd("rgb.txt"), rd("depth.txt"), rd("groundtruth.txt")
    dt = np.array([t for t, _ in dep]); gtt = np.array([t for t, _ in gt]); out = []
    for tr, pr in rgb:
        j = int(np.argmin(abs(dt - tr)));
        if abs(dt[j] - tr) > 0.02: continue
        g = int(np.argmin(abs(gtt - tr)))
        if abs(gtt[g] - tr) > 0.02: continue
        out.append((os.path.join(seq, pr[0]), os.path.join(seq, dep[j][1][0]), gt[g][1]))
    return out

def reproject(depth_k, K, T_rel):
    """render keyframe depth into frame t. T_rel = inv(T_t)@T_k (cam-k->cam-t). K=3x3 (378 frame)."""
    H, W = depth_k.shape; fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    ys, xs = np.nonzero(depth_k > 0); z = depth_k[ys, xs]
    X = (xs - cx) / fx * z; Y = (ys - cy) / fy * z
    pts = np.stack([X, Y, z, np.ones_like(z)], 0); q = T_rel @ pts
    zc = q[2]; m = zc > 1e-3; q, zc = q[:, m], zc[m]
    u = np.round(q[0] / zc * fx + cx).astype(np.int32); v = np.round(q[1] / zc * fy + cy).astype(np.int32)
    mm = (u >= 0) & (u < W) & (v >= 0) & (v < H); u, v, zc = u[mm], v[mm], zc[mm]
    out = np.full(H * W, np.inf, np.float32); flat = v * W + u; order = np.argsort(-zc)
    np.minimum.at(out, flat[order], zc[order]); out[~np.isfinite(out)] = 0.0
    return out.reshape(H, W)

def abs_rel(pred, gt):
    m = (gt > 0.1) & (gt < 10) & (pred > 0) & np.isfinite(pred)
    if m.sum() < 500: return np.nan
    p = pred.copy(); p *= np.median(gt[m] / p[m]); return float(np.mean(np.abs(p[m] - gt[m]) / gt[m]))

def tof_resid(coast, gt_t, grid=8):
    H, W = gt_t.shape; hs, ws = H // grid, W // grid; r = []
    for i in range(grid):
        for j in range(grid):
            cc = coast[i*hs:(i+1)*hs, j*ws:(j+1)*ws]; ss = gt_t[i*hs:(i+1)*hs, j*ws:(j+1)*ws]
            cv, sv = cc[cc > 0], ss[ss > 0]
            if len(cv) < 5 or len(sv) < 5: continue
            r.append(abs(np.median(cv) - np.median(sv)) / np.median(sv))
    return float(np.median(r)) if r else 1.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True); ap.add_argument("--nframes", type=int, default=120)
    ap.add_argument("--out", default=None)
    ap.add_argument("--darken", action="store_true", help="0.5x lighting on middle third (perturbs appearance signal only)")
    ap.add_argument("--ckpt", default="/home/evc/merlin/student_distilled.pt")
    a = ap.parse_args()
    m = load_model(a.ckpt)
    frames = assoc(a.seq)[::3][:a.nframes]
    print(f"{os.path.basename(a.seq)}: {len(frames)} frames; caching real depths...", flush=True)
    # Phase 1: cache real model depth + intrinsics + aligned GT + gray, per frame
    D, K, GT, GRAY, POSE = [], [], [], [], []
    for fi, (rp, dp, pose) in enumerate(frames):
        v = load_images([rp], resize_mode="square", size=SIZE)
        for x in v: x["img"] = x["img"].to("cuda")
        with torch.no_grad():
            o = m.infer(v, memory_efficient_inference=True, minibatch_size=1, use_amp=False, apply_mask=False, mask_edges=False)[0]
        d = o["depth_z"].float().squeeze().cpu().numpy()
        K.append(o["intrinsics"].float().squeeze().cpu().numpy())
        img_pil = exif_transpose(Image.open(rp)).convert("RGB")
        gt_raw = np.asarray(Image.open(dp)).astype(np.float32) / DEPTH_SCALE
        img378, gt = crop_resize_if_necessary(img_pil, (SIZE, SIZE), depthmap=gt_raw)
        # C2/ToF makes the map metric: scale-align each frame's depth to metric (sim ToF anchor).
        mm = (gt > 0.1) & (gt < 10) & (d > 0)
        if mm.sum() > 500: d = d * float(np.median(gt[mm] / d[mm]))
        D.append(d)
        GT.append(gt); GRAY.append(np.asarray(img378.convert("L")).astype(np.float32)); POSE.append(quat_to_T(pose))
        if fi % 20 == 0: print(f"  cache {fi}/{len(frames)}", flush=True)
    n = len(D)
    if a.darken:   # perturb ONLY the appearance signal on a static middle segment (geometry/ToF untouched)
        for t in range(n // 3, 2 * n // 3): GRAY[t] = GRAY[t] * 0.5

    # Phase 2: gated policy over the cache (pure numpy), sweep thresholds per gate
    def run_policy(gate, thr):
        kf = 0; refreshes = 1; errs = []
        for t in range(n):
            if t == kf:
                out = D[t]
            else:
                Trel = np.linalg.inv(POSE[t]) @ POSE[kf]
                coast = reproject(D[kf], K[kf], Trel)
                if gate == "metric": sig = tof_resid(coast, GT[t])
                elif gate == "appear": sig = float(np.mean(np.abs(GRAY[t] - GRAY[kf])) / 255.0)
                else: sig = (t - kf)            # fixedrate: larger lag -> refresh sooner
                if sig > thr:                   # stale -> refresh
                    kf = t; refreshes += 1; out = D[t]
                else:
                    out = coast
            e = abs_rel(out, GT[t])
            if np.isfinite(e): errs.append(e)
        return 1 - refreshes / n, float(np.mean(errs))   # (skip_rate, mean abs_rel)

    grids = {"metric": [0.02, 0.04, 0.06, 0.09, 0.13, 0.2, 0.3],
             "appear": [0.02, 0.04, 0.06, 0.09, 0.13, 0.2, 0.3],
             "fixedrate": [2, 3, 5, 8, 12, 20, 40]}
    res = {"seq": os.path.basename(a.seq), "nframes": n, "pareto": {}}
    for gate, ths in grids.items():
        res["pareto"][gate] = [run_policy(gate, th) for th in ths]
        print(gate, [(round(h, 2), round(e, 4)) for h, e in res["pareto"][gate]], flush=True)
    # energy curve from measured states
    res["energy_J_per_frame"] = {f"{h:.2f}": round(0.027 + (1 - h) * 0.54, 3) for h in [0.0, 0.5, 0.7, 0.8, 0.9]}
    out = a.out or f"/home/evc/merlin/tum/realgate_{res['seq']}.json"
    json.dump(res, open(out, "w"), indent=2); print("wrote", out)

if __name__ == "__main__":
    main()
