#!/usr/bin/env python3
"""Does a TAIL statistic of obstacle-region depth error predict closed-loop navigation,
where every mean statistic fails?

The C4 campaign measured a clean dissociation: the mixtures navigate ~2.6x better than dense
while being WORSE on near-field AbsRel, whole-frame AbsRel, far-field signed bias and offline
clear-route blockage. All four of those are means. The hypothesis this script tests is that the
dissociation is structural rather than mysterious:

    a collision needs ONE bad frame at one moment, so closed-loop success is an extreme-value
    event over a trajectory, while AbsRel is an expectation over pixels. The two are linked only
    through the tail, and conditional capacity trades mean error for tail error.

If that is right, some tail statistic of the signed error on obstacle pixels should reproduce the
measured per-class ordering of the arms, including the two reversals that make the C4 table
interesting (top-2 owns umbrella 0.950 vs top-1's 0.350; top-1 owns clothes_hanger_02 0.550 vs
top-2's 0.032). If NO tail statistic does, the mechanism is wrong.

Sign convention, and it is the whole point: e = pred*scale - gt on obstacle pixels. e > 0 means
the model places the obstacle FARTHER than it is, so the planner does not brake. That error is
hazardous. e < 0 is conservative and nearly free. Every candidate here is therefore asymmetric in
a way no standard depth metric is.

This script only COLLECTS per-frame summaries; `tail_rank.py` sweeps the candidate statistics and
scores them against the measured closed-loop table. Splitting them means the statistic sweep is a
few seconds of CPU and never re-renders.

Rendering is done ONCE per pose and every arm is scored on the identical pixels, so arm-vs-arm
differences cannot come from a different frame, route or obstacle placement.
"""
import argparse
import json
import os
import time

import numpy as np

from eval.drone.diag_depth import route_poses, spawn, PORT
from eval.drone.perception import Perception
from eval.drone.sim_client import SimClient

# Quantiles of the signed error kept per frame. The sweep in tail_rank.py builds every candidate
# statistic out of these, so adding a candidate later costs no rendering.
QS = [0.50, 0.75, 0.90, 0.95, 0.99]
# Hazard thresholds in metres. 0.27 is the measured displacement of an object's near face in the
# pre-fix model, and the body radius is 0.18, so this brackets the regime that actually collides.
TAUS = [0.10, 0.15, 0.20, 0.27, 0.35, 0.50]


def frame_stats(e, gt):
    """Per-frame summary of the signed obstacle-region error, sufficient for every candidate."""
    q = np.quantile(e, QS)
    return {"n": int(e.size),
            "gt": round(float(np.median(gt)), 3),
            "mean": round(float(e.mean()), 4),
            "absrel": round(float(np.mean(np.abs(e) / gt)), 4),
            "max": round(float(e.max()), 4),
            "q": [round(float(x), 4) for x in q],
            "haz": [round(float((e > t).mean()), 4) for t in TAUS]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="tag=path, one per arm; all arms are scored on identical pixels")
    ap.add_argument("--classes", nargs="+", required=True)
    ap.add_argument("--scenes", nargs="+",
                    default=["apt_0", "apt_1", "apt_2", "apt_3", "apt_4", "apt_5"])
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=9000)
    ap.add_argument("--yaws", default=None, help="best_yaw.json, to match the flown orientation")
    # Defaults below MUST match the closed-loop fly config, else the frames scored here are not
    # the frames that flew and the whole comparison is void.
    ap.add_argument("--res", type=int, default=378)
    ap.add_argument("--alt", type=float, default=0.75)
    ap.add_argument("--spacing", type=float, default=0.25)
    ap.add_argument("--max_range", type=float, default=2.5)
    ap.add_argument("--route_clear", type=float, default=0.6)
    ap.add_argument("--obs_clear", type=float, default=0.3)
    ap.add_argument("--intrude", type=float, default=0.9)
    ap.add_argument("--min_geo", type=float, default=3.0)
    ap.add_argument("--max_geo", type=float, default=14.0)
    ap.add_argument("--nav_radius", type=float, default=0.35)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    arms = {}
    for spec in args.ckpts:
        tag, path = spec.split("=", 1)
        print(f"[load] {tag} <- {path}", flush=True)
        arms[tag] = Perception(path)
    yaws = json.load(open(args.yaws)) if args.yaws else {}

    proc = spawn(PORT, args.res, 90.0)
    rows, t0 = [], time.time()
    try:
        time.sleep(12)
        client = SimClient(PORT)
        for cls in args.classes:
            Y = float(yaws.get(cls, 0.0))
            kept = 0
            for scene in args.scenes:
                client.reset(scene, args.dataset, args.nav_radius)
                for k in range(args.episodes):
                    seed = args.seed0 + k
                    epi = client.sample(scene, args.min_geo, args.max_geo, seed=seed,
                                        alt=args.alt, dataset=args.dataset,
                                        route_clear=args.route_clear)
                    if epi is None:
                        continue
                    wps = epi.get("waypoints_hab") or [epi["start_hab"], epi["goal_hab"]]
                    poses = route_poses(wps, args.spacing)
                    if not poses:
                        continue
                    client.clear_obstacles()
                    bg = [client.render(p, R)["depth"] for p, R in poses]
                    obs = client.place_obstacles(wps, 1, seed=seed, alt=args.alt,
                                                 intrude=args.intrude, obs_clear=args.obs_clear,
                                                 min_width=0.35, block_margin=-0.10,
                                                 jitter=0.10, include=[cls], yaw=Y)
                    if not obs.get("placed"):
                        continue
                    shots = [client.render(p, R) for p, R in poses]
                    # One render pass, every arm scored on it.
                    for tag, perc in arms.items():
                        ep_scale, per = None, []
                        for out, d0 in zip(shots, bg):
                            gt, rgb = out["depth"], out["rgb"]
                            pred = perc._raw_depth(rgb, args.res)
                            v = np.isfinite(gt) & (gt > 0.1) & (gt < 10) & (pred > 0.05)
                            if ep_scale is None and v.sum() > 200:
                                # The episode-constant scale the deployed arm actually flies with.
                                ep_scale = float(np.median(gt[v] / pred[v]))
                            if ep_scale is None:
                                continue
                            m = (d0 - gt > 0.15) & (gt > 0.1) & (gt < args.max_range)
                            if m.sum() < 100:
                                continue
                            e = pred[m] * ep_scale - gt[m]
                            per.append(frame_stats(e, gt[m]))
                        if per:
                            rows.append({"cls": cls, "arm": tag, "scene": scene, "seed": seed,
                                         "frames": per})
                            kept += 1
            print(f"[{cls}] episode-arms kept={kept}  ({time.time()-t0:.0f}s)", flush=True)
    finally:
        proc.terminate()
        proc.wait(timeout=20)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} episode-arm rows -> {args.out}")


if __name__ == "__main__":
    main()
