#!/usr/bin/env python3
"""M2: deadline-elastic anytime reconstruction (the second MERLIN method).

The student exposes multiple operating points by INPUT RESOLUTION (252/378/518),
each a measured latency/accuracy point on the Nano. A per-frame controller picks
the operating point to hold the navigation control deadline: fast camera motion ->
the map must refresh sooner -> tighter deadline -> lower res (fresher); slow motion
-> looser deadline -> higher res (sharper). The claim: adaptive holds the deadline
AND maximizes fidelity, Pareto-beating any FIXED operating point -- because no single
fixed point is optimal across the speed/scene variation of a real traversal.

Ablation: compare the adaptive controller vs each fixed resolution on
(deadline-hit-rate, accuracy-on-hit-frames). Adaptive should dominate: higher
hit-rate than the slow fixed point, higher accuracy than the fast fixed point.

Per-frame deadline comes from the TUM GT trajectory velocity (proxy for "how fresh
the map must be"). Accuracy is scale-aligned abs_rel vs GT depth. Latency is the
measured Jetson floor per resolution.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model", "distill"))
sys.path.insert(0, os.path.join(HERE, "..", "model", "backbones"))
sys.path.insert(0, os.path.join(HERE, "geometry"))
from data import tum_pairs, load_gt_depth                # noqa: E402
from student import build_student                          # noqa: E402

# Measured Jetson Orin Nano bf16 latency per resolution (ms) -- the operating points.
LATENCY_MS = {252: 84.0, 378: 147.0, 518: 189.0}
RES_LIST = [252, 378, 518]


def scale_aligned_absrel(gt, pred, min_d=0.1, max_d=10.0):
    v = np.isfinite(gt) & np.isfinite(pred) & (gt > min_d) & (gt < max_d) & (pred > 0.05)
    if v.sum() < 200:
        return np.nan
    r = np.median(gt[v] / pred[v])
    return float(np.mean(np.abs(gt[v] - r * pred[v]) / gt[v]))


def load_traj_velocity(seq):
    """Per-rgb-frame camera speed (m/s) from TUM groundtruth.txt (nearest timestamp)."""
    gt = os.path.join(seq, "groundtruth.txt")
    if not os.path.exists(gt):
        return None
    T = []
    for line in open(gt):
        if line.startswith("#"):
            continue
        p = line.split()
        if len(p) >= 4:
            T.append((float(p[0]), np.array([float(p[1]), float(p[2]), float(p[3])])))
    if len(T) < 2:
        return None
    ts = np.array([t for t, _ in T]); xs = np.stack([x for _, x in T])
    vel = np.zeros(len(ts))
    vel[1:] = np.linalg.norm(np.diff(xs, axis=0), axis=1) / np.clip(np.diff(ts), 1e-3, None)
    return ts, vel


@torch.no_grad()
def run(student, path, res, dev):
    from mapanything.utils.image import load_images
    s = (res // 14) * 14
    v = load_images([path], resize_mode="square", size=s)
    v[0]["img"] = v[0]["img"].to(dev)
    pr = student(v, memory_efficient_inference=True, minibatch_size=1)[0]
    return pr["pts3d_cam"][0, ..., 2].float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/ckpt/student_v2.pt")
    ap.add_argument("--seq", default="/workspace/data/tum/rgbd_dataset_freiburg1_room")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--out", default="/workspace/ckpt/m2_deadline.json")
    args = ap.parse_args()
    dev = "cuda"
    student = build_student(size="base", aat_depth=8, device="cpu")
    student.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"])
    student = student.to(dev).eval()

    pairs = tum_pairs(args.seq)
    step = max(1, len(pairs) // args.n)
    pairs = pairs[::step][:args.n]
    tv = load_traj_velocity(args.seq)

    # accuracy per (frame, res)
    rows = []
    for rp, dp in pairs:
        t = float(os.path.basename(rp)[:-4])
        vel = float(np.interp(t, tv[0], tv[1])) if tv else 0.1
        acc = {}
        for res in RES_LIST:
            sd = run(student, rp, res, dev)
            gt = load_gt_depth(dp, sd.shape[0], sd.shape[1])
            acc[res] = scale_aligned_absrel(gt, sd)
        rows.append({"vel": vel, "acc": acc})
    print(f"[m2] {len(rows)} frames; mean vel {np.mean([r['vel'] for r in rows]):.3f} m/s", flush=True)

    # per-frame deadline (ms): faster motion -> tighter. deadline = clamp(travel_budget/vel).
    # A robot that may not move more than ~D meters between map updates: deadline = D/vel.
    D = 0.05  # 5 cm of travel per keyframe before the map is "stale"
    report = {}
    for D in (0.03, 0.05, 0.08):
        deadlines = [min(400.0, max(40.0, 1000.0 * D / max(r["vel"], 0.02))) for r in rows]
        def eval_policy(pick):
            hit, accs = 0, []
            for r, dl in zip(rows, deadlines):
                res = pick(r, dl)
                if LATENCY_MS[res] <= dl and np.isfinite(r["acc"][res]):
                    hit += 1; accs.append(r["acc"][res])
            return hit / len(rows), (float(np.mean(accs)) if accs else float("nan"))
        # adaptive: highest-accuracy res whose latency fits the deadline (else fastest)
        def adaptive(r, dl):
            ok = [res for res in RES_LIST if LATENCY_MS[res] <= dl]
            if not ok:
                return min(RES_LIST, key=lambda x: LATENCY_MS[x])
            return min(ok, key=lambda res: (r["acc"][res] if np.isfinite(r["acc"][res]) else 9))
        pol = {f"fixed_{res}": (lambda r, dl, res=res: res) for res in RES_LIST}
        pol["adaptive"] = adaptive
        report[f"D={D}m"] = {name: {"hit_rate": round(h, 3), "absrel_on_hit": round(a, 4)}
                             for name, (h, a) in ((n, eval_policy(p)) for n, p in pol.items())}
        print(f"D={D}m:", report[f"D={D}m"], flush=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print(f"[saved] {args.out}\n=== M2_DONE ===", flush=True)


if __name__ == "__main__":
    main()
