#!/usr/bin/env python3
"""Render SEQUENTIAL (RGB, GT-depth) trajectories for the recirculation study.

Cross-frame recirculation needs consecutive frames along a real camera path. The existing
`data/nearfield/` set cannot serve: its poses are independently sampled near-field views, not
a trajectory, so there is no frame t-1 to carry state from.

This flies the geodesic route open-loop exactly as `diag_depth.py` does (poses from the route,
no PX4) and dumps one npz per sequence. Rendering happens once; every arm, layer pair and alpha
in `eval_recirc.py` then runs offline from the cache with no renderer in the loop.

`--spacing` is the variable that matters. The deployed model runs every frame at 20 FPS and
0.177 m/s, i.e. ~0.009 m of camera motion between invocations; keyframe spacing is coarser.
Sweep it: if a gain exists only at wide spacing it is not the deployed regime, and if it exists
only at tight spacing it is a temporal-smoothing effect worth naming as such.
"""
import argparse
import os
import time

import numpy as np

from eval.drone.diag_depth import route_poses
from eval.drone.run_experiments import WS, spawn_renderer
from eval.drone.sim_client import SimClient

PORT = 5601 + int(os.environ.get("MERLIN_PX4_INSTANCE", 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--episodes", type=int, default=6, help="routes per scene")
    ap.add_argument("--spacing", type=float, default=0.10, help="metres between frames")
    ap.add_argument("--max_frames", type=int, default=60)
    ap.add_argument("--res", type=int, default=378)
    ap.add_argument("--alt", type=float, default=0.75)
    ap.add_argument("--min_geo", type=float, default=3.0)
    ap.add_argument("--max_geo", type=float, default=12.0)
    ap.add_argument("--nav_radius", type=float, default=0.45)
    ap.add_argument("--route_clear", type=float, default=0.60)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    proc = spawn_renderer(PORT, args.res)
    try:
        client = SimClient(PORT)
        for _ in range(60):
            try:
                client.ping()
                break
            except Exception:
                time.sleep(2.0)
        n_seq = 0
        for scene in args.scenes:
            client.reset(scene, args.dataset, args.nav_radius)
            for k in range(args.episodes):
                seed = args.seed0 + k
                epi = client.sample(scene, args.min_geo, args.max_geo, seed=seed,
                                    alt=args.alt, dataset=args.dataset,
                                    route_clear=args.route_clear)
                if epi is None:
                    print(f"  {os.path.basename(scene)} s{seed}: no route", flush=True)
                    continue
                wps = epi.get("waypoints_hab") or [epi["start_hab"], epi["goal_hab"]]
                poses = route_poses(wps, args.spacing)[: args.max_frames]
                rgbs, deps = [], []
                for p, R in poses:
                    out = client.render(p, R)
                    rgbs.append(np.asarray(out["rgb"], np.uint8))
                    deps.append(np.asarray(out["depth"], np.float16))
                if len(rgbs) < 8:
                    continue
                tag = f"{os.path.splitext(os.path.basename(scene))[0]}_s{seed}"
                np.savez_compressed(os.path.join(args.out, tag + ".npz"),
                                    rgb=np.stack(rgbs), dep=np.stack(deps),
                                    spacing=args.spacing)
                n_seq += 1
                print(f"  {tag}: {len(rgbs)} frames", flush=True)
        print(f"wrote {n_seq} sequences to {args.out}", flush=True)
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
