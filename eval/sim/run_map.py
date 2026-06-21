#!/usr/bin/env python3
"""Smarter-planner experiment: navigate with a GLOBAL occupancy-mapping + A* planner that
plans on the PERCEIVED map (map_planner.py), instead of the reactive navmesh-guided planner.

Tests the closed-loop report's open question: does perception fidelity (M1) re-enter once the
planner actually builds and uses a map? Arms: gt (perfect depth) / baseline / nav1, same
episodes (paired). --validate runs gt-only on a few episodes to confirm the projection (a
GT-depth mapping agent should navigate well; if it can't, the projection/A* is broken).
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sim_client import SimClient        # noqa: E402
from planner import LocalPlanner        # noqa: E402
from costmap import range_scan          # noqa: E402
from map_planner import MapPlanner       # noqa: E402


def spawn_server(port, conda_sh, env_name):
    cmd = (f"source {conda_sh} && conda activate {env_name} && "
           f"MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet "
           f"python {os.path.join(HERE, 'habitat_server.py')} --port {port}")
    p = subprocess.Popen(["bash", "-lc", cmd], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print("[server]", line.rstrip(), flush=True)
        if "HABSERVER_READY" in line:
            break
    return p


def boot_ci(xs, n=2000, seed=0):
    xs = np.asarray([x for x in xs if x is not None and np.isfinite(x)], float)
    if xs.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    bs = [rng.choice(xs, xs.size, replace=True).mean() for _ in range(n)]
    return (float(xs.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def run_episode_map(client, perception, lp, ep, res=378, success_radius=0.4,
                    max_sim_time=120.0, stuck_patience=90):
    obs = client.reset(ep["scene"], ep["start"], ep["yaw"], ep["goal"],
                       ep.get("dataset"), guide="straight")
    hfov = obs["hfov"]
    if perception is not None:
        perception.calibrate(obs["rgb"], obs["depth"], res=378)
    mapper = MapPlanner(ep["start"], ep["goal"])
    from perception import LATENCY_MS
    sim_t = 0.0; best_geo = obs["geo_dist"]; stuck = 0; success = False
    while sim_t < max_sim_time:
        if np.isfinite(obs["geo_dist"]) and obs["geo_dist"] <= success_radius:
            success = True; break
        depth = obs["depth"] if perception is None else perception.depth(obs["rgb"], res=res)
        mapper.update(obs["pos"], obs["yaw"], depth, hfov)
        gb = mapper.bearing(obs["pos"], obs["yaw"])
        bearings, ranges = range_scan(depth, hfov)
        v, yaw_rate, info = lp.command(gb, bearings, ranges)
        L = LATENCY_MS[res] / 1000.0
        obs = client.step(v, yaw_rate, L); sim_t += L
        g = obs["geo_dist"]
        if np.isfinite(g):
            if g < best_geo - 0.05: best_geo = g; stuck = 0
            else: stuck += 1
        if stuck >= stuck_patience: break
    pl = obs["path_len"]
    spl = (ep["geo"] / max(pl, ep["geo"])) if success else 0.0
    return {"scene": ep["scene"], "success": int(success), "spl": round(spl, 4),
            "collisions": int(obs["collisions"]), "path_len": round(pl, 3),
            "collisions_per_m": round(obs["collisions"]/max(pl, 1e-3), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--dataset", default="/root/habitat-data/versioned_data/replica_cad_dataset/replicaCAD.scene_dataset_config.json")
    ap.add_argument("--ckpts", nargs="*", default=[], help="tag=path ...")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--validate", action="store_true", help="gt-only smoke to check projection")
    ap.add_argument("--min-geo", type=float, default=2.5)
    ap.add_argument("--max-geo", type=float, default=8.0)
    ap.add_argument("--port", type=int, default=5562)
    ap.add_argument("--conda-sh", default="/workspace/miniconda3/etc/profile.d/conda.sh")
    ap.add_argument("--env-name", default="habenv")
    ap.add_argument("--out", default="/workspace/ckpt/sim_map")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    arms = [("gt", None)]
    if not args.validate:
        from perception import Perception
        for c in args.ckpts:
            tag, path = c.split("=", 1)
            arms.append((tag, Perception(path)))

    server = spawn_server(args.port, args.conda_sh, args.env_name)
    client = SimClient(args.port)
    lp = LocalPlanner(max_speed=0.5, brake_dist=0.30)

    # fixed episode set (paired)
    eps = []
    for scene in args.scenes:
        got = 0; seed = 0
        while got < args.episodes and seed < args.episodes * 25:
            try:
                ep = client.sample(scene, args.min_geo, args.max_geo, seed, args.dataset)
            except Exception as e:
                ep = None
            seed += 1
            if ep:
                ep["dataset"] = args.dataset; eps.append(ep); got += 1
    print(f"[map] {len(eps)} episodes, arms={[a for a,_ in arms]}", flush=True)

    rows = []
    for tag, perc in arms:
        succ = []
        for ep in eps:
            try:
                r = run_episode_map(client, perc, lp, ep)
            except Exception as e:
                print(f"[ep fail {tag}: {repr(e)[:80]}]", flush=True); continue
            r["arm"] = tag; rows.append(r); succ.append(r["success"])
        m, lo, hi = boot_ci(succ)
        print(f"  {tag}: success {m:.3f} [{lo:.3f},{hi:.3f}] n={len(succ)} "
              f"collisions/m {np.mean([x['collisions_per_m'] for x in rows if x['arm']==tag]):.3f}", flush=True)
    json.dump(rows, open(os.path.join(args.out, "map_rows.json"), "w"), indent=2)
    print("=== RUN_MAP_DONE ===", flush=True)
    try: client.close(); server.terminate()
    except Exception: pass


if __name__ == "__main__":
    main()
