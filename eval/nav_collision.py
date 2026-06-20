#!/usr/bin/env python3
"""Navigation-safety eval: collision-relevant metrics bridging depth proxies toward
the actual M1 navigation claim.

A local planner steering the robot forward only cares about one thing per frame: is
the drivable corridor ahead clear within braking distance? We turn each depth map into
that decision and compare it to the GT-depth decision:
  - COLLISION (false negative): GT says an obstacle is within brake_dist in the forward
    corridor, but the student's depth says clear -> the robot would not stop -> crash.
    Safety-critical; this is what M1 (near-field-focused) should reduce.
  - FALSE STOP (false positive): student sees an obstacle that is not there -> needless halt.

Forward corridor = central image columns x lower (drivable) rows; corridor range = the
nearest valid depth in that region. Run baseline vs M1 students; M1 should cut collisions.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model", "distill"))
sys.path.insert(0, os.path.join(HERE, "..", "model", "backbones"))
from data import tum_pairs, load_gt_depth   # noqa: E402
from student import build_student            # noqa: E402


def corridor_range(depth, col_frac=0.4, row_lo=0.4, min_d=0.1):
    """Nearest valid depth in the forward drivable corridor (central cols, lower rows)."""
    H, W = depth.shape
    c0 = int(W * (0.5 - col_frac / 2)); c1 = int(W * (0.5 + col_frac / 2))
    r0 = int(H * row_lo)
    region = depth[r0:, c0:c1]
    valid = region[(np.isfinite(region)) & (region > min_d)]
    return float(np.min(valid)) if valid.size > 50 else np.inf


@torch.no_grad()
def eval_ckpt(ckpt, pairs, dev, res, brake_dist, near_scale=False):
    from mapanything.utils.image import load_images
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = build_student(size=ck.get("size", "base"), aat_depth=ck.get("aat_depth", 8), device="cpu")
    m.load_state_dict(ck["state_dict"]); m = m.to(dev).eval()
    s = (res // 14) * 14
    coll = fstop = obst = clear = 0
    near_err = []
    for rp, dp in pairs:
        v = load_images([rp], resize_mode="square", size=s); v[0]["img"] = v[0]["img"].to(dev)
        sd = m(v, memory_efficient_inference=True, minibatch_size=1)[0]["pts3d_cam"][0, ..., 2].float().cpu().numpy()
        gt = load_gt_depth(dp, sd.shape[0], sd.shape[1])
        # scale-align student to GT. GLOBAL = scale from whole frame (what SLAM/IMU gives
        # globally); NEAR = scale from the corridor region (best case for local scale).
        vmask = np.isfinite(gt) & np.isfinite(sd) & (gt > 0.1) & (gt < 10) & (sd > 0.05)
        if vmask.sum() < 200:
            continue
        H, W = sd.shape; c0 = int(W * 0.3); c1 = int(W * 0.7); r0 = int(H * 0.4)
        nm = np.zeros_like(vmask); nm[r0:, c0:c1] = True; nm &= vmask
        scale = np.median(gt[nm] / sd[nm]) if (nm.sum() > 50 and near_scale) else np.median(gt[vmask] / sd[vmask])
        sd = sd * scale
        g_rng, s_rng = corridor_range(gt), corridor_range(sd)
        g_obst = g_rng < brake_dist
        s_obst = s_rng < brake_dist
        if g_obst:
            obst += 1
            if not s_obst:
                coll += 1                       # missed a real obstacle -> collision
            near_err.append(abs(g_rng - min(s_rng, brake_dist * 3)))
        else:
            clear += 1
            if s_obst:
                fstop += 1                      # hallucinated obstacle -> false stop
    del m; torch.cuda.empty_cache()
    return {"frames_with_obstacle": obst, "frames_clear": clear,
            "collision_rate": round(coll / max(1, obst), 4),
            "false_stop_rate": round(fstop / max(1, clear), 4),
            "corridor_range_mae_m": round(float(np.mean(near_err)) if near_err else float("nan"), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="tag=path ...")
    ap.add_argument("--seq", default="/workspace/data/tum/rgbd_dataset_freiburg1_room")
    ap.add_argument("--res", type=int, default=378)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--brake-dist", type=float, default=1.0)
    ap.add_argument("--near-scale", action="store_true", help="scale-align from corridor (local) not global")
    ap.add_argument("--out", default="/workspace/ckpt/nav_collision.json")
    args = ap.parse_args()
    dev = "cuda"
    pairs = tum_pairs(args.seq)
    step = max(1, len(pairs) // args.n); pairs = pairs[::step][:args.n]
    print(f"[nav] {len(pairs)} frames, brake_dist={args.brake_dist}m", flush=True)
    report = {}
    for c in args.ckpts:
        tag, path = c.split("=", 1)
        report[tag] = eval_ckpt(path, pairs, dev, args.res, args.brake_dist, args.near_scale)
        print(f"  {tag}: {report[tag]}", flush=True)
    if "baseline" in report and "nav1" in report:
        b, m = report["baseline"], report["nav1"]
        print(f"\n=== M1 vs baseline (safety) ===")
        print(f"  collision_rate: {b['collision_rate']} -> {m['collision_rate']} "
              f"(delta {round(m['collision_rate']-b['collision_rate'],4)}; negative = fewer crashes)")
        print(f"  corridor_range_mae: {b['corridor_range_mae_m']} -> {m['corridor_range_mae_m']}")
    json.dump(report, open(args.out, "w"), indent=2)
    print("=== NAV_COLLISION_DONE ===", flush=True)


if __name__ == "__main__":
    main()
