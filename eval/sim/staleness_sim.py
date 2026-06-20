#!/usr/bin/env python3
"""Latency-coupled staleness sim: the dynamic test of M2 the offline evals cannot do.

Offline, M2's accuracy/latency frontier is flat (the student saturates). But M2's REAL
claim is dynamic: a slow operating point makes the map STALE while the robot keeps moving,
so the planner acts on outdated geometry and collides. That only exists in closed loop with
latency. This sim tests it without any view synthesis:

- Replay a real TUM trajectory (real poses + timestamps).
- At control tick i (agent at pose i), the planner's belief about the forward corridor comes
  from perception issued `latency(op_point)` ago -> frame j with t_j ~ t_i - latency, at the
  chosen resolution. So a slow op point -> larger staleness -> older belief.
- A collision occurs when the STALE belief says clear (range_j >= brake) but the TRUE current
  corridor (GT depth at i) is blocked (range_i < brake) while the agent is moving. Faster
  motion -> the agent covers more ground during staleness -> stale belief more wrong.
- Controllers: fixed_252/378/518 vs adaptive (faster motion -> faster op point -> fresher).
- Sweep a speed multiplier so the deadline binds; report collision rate per controller.

If adaptive holds collisions down as speed rises while fixed-slow degrades, M2 has dynamic
value the offline frontier hid. If not, the saturation/negative story stands -- now under a
closed-loop-relevant test.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "model", "distill"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "model", "backbones"))
sys.path.insert(0, os.path.join(HERE, ".."))
from data import tum_pairs, load_gt_depth          # noqa: E402
from student import build_student                    # noqa: E402
from nav_collision import corridor_range             # noqa: E402

LATENCY_S = {252: 0.084, 378: 0.147, 518: 0.189}     # measured Jetson bf16, seconds
RES_LIST = [252, 378, 518]


def load_traj(seq):
    gt = os.path.join(seq, "groundtruth.txt")
    T = []
    for line in open(gt):
        if line.startswith("#"):
            continue
        p = line.split()
        if len(p) >= 4:
            T.append((float(p[0]), np.array([float(p[1]), float(p[2]), float(p[3])])))
    ts = np.array([t for t, _ in T]); xs = np.stack([x for _, x in T])
    return ts, xs


@torch.no_grad()
def precompute(ckpt, pairs, dev):
    """Per frame: timestamp, GT corridor range, and student corridor range at each resolution."""
    from mapanything.utils.image import load_images
    m = build_student(size="base", aat_depth=8, device="cpu")
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"])
    m = m.to(dev).eval()
    rows = []
    for rp, dp in pairs:
        t = float(os.path.basename(rp)[:-4])
        gt = load_gt_depth(dp, 476, 476)
        g_rng = corridor_range(gt)
        s_rng = {}
        for res in RES_LIST:
            s = (res // 14) * 14
            v = load_images([rp], resize_mode="square", size=s); v[0]["img"] = v[0]["img"].to(dev)
            sd = m(v, memory_efficient_inference=True, minibatch_size=1)[0]["pts3d_cam"][0, ..., 2].float().cpu().numpy()
            # near-field scale align (best case for local/IMU scale) then corridor range
            import cv2
            sd = cv2.resize(sd, (476, 476))
            vm = np.isfinite(gt) & np.isfinite(sd) & (gt > 0.1) & (gt < 10) & (sd > 0.05)
            H, W = sd.shape; c0 = int(W*0.3); c1 = int(W*0.7); r0 = int(H*0.4)
            nm = np.zeros_like(vm); nm[r0:, c0:c1] = True; nm &= vm
            if nm.sum() > 50:
                sd = sd * np.median(gt[nm] / sd[nm])
            s_rng[res] = corridor_range(sd)
        rows.append({"t": t, "g_rng": g_rng, "s_rng": s_rng})
    del m; torch.cuda.empty_cache()
    return rows


def simulate(rows, ts, xs, speed_mult, brake=1.0):
    """For each control tick, planner uses stale perception; collision if stale-clear but true-blocked."""
    # per-frame speed from trajectory (scaled); staleness in seconds -> stale frame index
    frame_t = np.array([r["t"] for r in rows])
    # interpolate agent speed at each frame time
    spd = np.zeros(len(ts)); spd[1:] = np.linalg.norm(np.diff(xs, axis=0), axis=1) / np.clip(np.diff(ts), 1e-3, None)
    def speed_at(t):
        return float(np.interp(t, ts, spd)) * speed_mult

    def stale_frame(i, lat):
        # frame whose timestamp is ~ t_i - lat (the perception the planner is acting on)
        tgt = frame_t[i] - lat
        j = int(np.searchsorted(frame_t, tgt))
        return max(0, min(len(rows) - 1, j))

    def run(policy):
        coll = mov = 0
        for i, r in enumerate(rows):
            v = speed_at(r["t"])
            if v < 0.05:                      # not moving -> no collision risk this tick
                continue
            mov += 1
            op = policy(v)
            j = stale_frame(i, LATENCY_S[op])
            perceived = rows[j]["s_rng"][op]  # stale belief about the corridor
            true_now = r["g_rng"]             # actual obstacle range now (GT)
            # extra staleness penalty: agent advanced v*lat meters since perception
            advanced = v * LATENCY_S[op]
            if perceived - advanced >= brake and true_now < brake:
                coll += 1                     # believed clear (even accounting for closing), actually blocked
        return coll / max(1, mov)

    pols = {f"fixed_{res}": (lambda v, res=res: res) for res in RES_LIST}
    # adaptive: faster motion -> tighter deadline -> faster op point. deadline = D / v.
    def adaptive(v):
        D = 0.06
        dl = D / max(v, 0.02)
        ok = [res for res in RES_LIST if LATENCY_S[res] <= dl]
        return max(ok) if ok else min(RES_LIST, key=lambda x: LATENCY_S[x])  # sharpest that fits, else fastest
    pols["adaptive"] = adaptive
    return {name: round(run(p), 4) for name, p in pols.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/ckpt/student_v3.pt")
    ap.add_argument("--seq", default="/workspace/data/tum/rgbd_dataset_freiburg1_room")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--brake", type=float, default=1.0)
    ap.add_argument("--out", default="/workspace/ckpt/staleness_sim.json")
    args = ap.parse_args()
    dev = "cuda"
    pairs = tum_pairs(args.seq)
    step = max(1, len(pairs) // args.n); pairs = pairs[::step][:args.n]
    ts, xs = load_traj(args.seq)
    print(f"[sim] precomputing corridor ranges for {len(pairs)} frames x {len(RES_LIST)} res", flush=True)
    rows = precompute(args.ckpt, pairs, dev)
    report = {}
    for sm in (1.0, 2.0, 4.0, 8.0):
        report[f"speed_x{sm}"] = simulate(rows, ts, xs, sm, args.brake)
        print(f"speed x{sm}: {report[f'speed_x{sm}']}", flush=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print("=== STALENESS_SIM_DONE ===", flush=True)


if __name__ == "__main__":
    main()
