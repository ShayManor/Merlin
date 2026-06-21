#!/usr/bin/env python3
"""Driver for the closed-loop M1/M2 experiments.

Spawns the Habitat render server in the conda `habenv` (subprocess), samples a FIXED
episode set per scene (deterministic seeds), then runs every arm on the SAME episodes
(paired comparison -> high power for M1's subtle near-field effect). Aggregates with
bootstrap 95% CIs and writes per-episode JSONL + a summary JSON.

Arms are (ckpt_tag, controller, max_speed) triples. M1: two ckpts, fixed controller, one
speed. M2: one ckpt, four controllers, a speed sweep. GT-perception (ckpt_tag='gt') is
the perfect-perception upper bound.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "model", "distill"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "model", "backbones"))

from sim_client import SimClient        # noqa: E402
from planner import LocalPlanner        # noqa: E402
from nav_loop import run_episode        # noqa: E402


def spawn_server(port, conda_sh, env_name):
    cmd = (f"source {conda_sh} && conda activate {env_name} && "
           f"MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet "
           f"python {os.path.join(HERE, 'habitat_server.py')} --port {port}")
    p = subprocess.Popen(["bash", "-lc", cmd], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    # stream until ready marker
    for line in p.stdout:
        print("[server]", line.rstrip(), flush=True)
        if "HABSERVER_READY" in line:
            break
    return p


def boot_ci(xs, n=2000, seed=0):
    xs = np.asarray([x for x in xs if x is not None and np.isfinite(x)], float)
    if xs.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    bs = [rng.choice(xs, xs.size, replace=True).mean() for _ in range(n)]
    return (float(xs.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def aggregate(rows, key):
    out = {}
    arms = sorted({r["arm"] for r in rows})
    for a in arms:
        rr = [r for r in rows if r["arm"] == a]
        agg = {"n": len(rr)}
        for m in ("success", "spl", "collisions_per_m", "energy_per_m", "fwd_perr_m",
                  "mean_stale_m", "mean_res", "mean_speed", "path_len"):
            mean, lo, hi = boot_ci([r[m] for r in rr])
            agg[m] = {"mean": round(mean, 4), "ci": [round(lo, 4), round(hi, 4)]}
        out[a] = agg
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "m1", "m2"], required=True)
    ap.add_argument("--scenes", nargs="+", required=True, help="scene id or .glb path")
    ap.add_argument("--dataset", default=None, help="scene_dataset_config.json (ReplicaCAD)")
    ap.add_argument("--ckpts", nargs="*", default=[], help="tag=path ...")
    ap.add_argument("--episodes", type=int, default=20, help="episodes per scene")
    ap.add_argument("--min-geo", type=float, default=2.5)
    ap.add_argument("--max-geo", type=float, default=8.0)
    ap.add_argument("--speeds", nargs="+", type=float, default=[0.5])
    ap.add_argument("--controllers", nargs="+", default=["fixed_378"])
    ap.add_argument("--time-scale", type=float, default=1.0)
    ap.add_argument("--port", type=int, default=5557)
    ap.add_argument("--conda-sh", default="/workspace/miniconda3/etc/profile.d/conda.sh")
    ap.add_argument("--env-name", default="habenv")
    ap.add_argument("--out", default="/workspace/ckpt/sim")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--guide", choices=["geodesic", "straight"], default="geodesic",
                    help="straight = robot must avoid obstacles from perception (puts the "
                         "student's depth on the critical path)")
    ap.add_argument("--brake-dist", type=float, default=0.45,
                    help="planner safety margin; tight values stress perception precision")
    ap.add_argument("--robot-radius", type=float, default=0.18)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ckpts = dict(c.split("=", 1) for c in args.ckpts)

    # build arms
    arms = []
    if args.mode == "smoke":
        arms = [{"name": "gt_fixed_378", "ckpt": "gt", "controller": "fixed_378",
                 "speed": args.speeds[0]}]
    elif args.mode == "m1":
        for tag in ckpts:
            arms.append({"name": f"{tag}_{args.controllers[0]}", "ckpt": tag,
                         "controller": args.controllers[0], "speed": args.speeds[0]})
    elif args.mode == "m2":
        tag = next(iter(ckpts))
        for sp in args.speeds:
            for ctrl in args.controllers:
                arms.append({"name": f"{ctrl}@v{sp}", "ckpt": tag,
                             "controller": ctrl, "speed": sp})

    # respawnable server: habitat-sim can sporadically die across many episodes; a hung
    # server would otherwise hang the client and lose the whole run. Keep a mutable handle
    # and restart on any RPC failure, skipping just the offending episode.
    state = {"server": spawn_server(args.port, args.conda_sh, args.env_name),
             "client": SimClient(args.port), "port": args.port}

    def restart():
        try:
            state["client"].close()
        except Exception:
            pass
        try:
            state["server"].terminate()
        except Exception:
            pass
        state["port"] += 1
        print(f"[restart server -> port {state['port']}]", flush=True)
        state["server"] = spawn_server(state["port"], args.conda_sh, args.env_name)
        state["client"] = SimClient(state["port"])

    jsonl = os.path.join(args.out, f"{args.mode}_rows.jsonl")
    fout = open(jsonl, "w")
    rows = []
    try:
        # fixed episode set per scene (shared across all arms)
        episodes = []
        for si, scene in enumerate(args.scenes):
            for ei in range(args.episodes):
                seed = args.seed + 1000 * si + ei
                for _try in range(2):
                    try:
                        ep = state["client"].sample(scene, args.min_geo, args.max_geo,
                                                     seed, args.dataset)
                        break
                    except Exception as e:
                        print(f"[sample fail {scene} seed{seed}: {e}]", flush=True)
                        restart(); ep = None
                if ep:
                    ep["dataset"] = args.dataset
                    episodes.append(ep)
            print(f"[scene {scene}] sampled {sum(1 for e in episodes if e['scene']==scene)} eps",
                  flush=True)

        # group arms by ckpt so at most ONE student is GPU-resident at a time
        # (paired comparison preserved: every arm runs the same fixed episode set)
        from collections import OrderedDict
        groups = OrderedDict()
        for arm in arms:
            groups.setdefault(arm["ckpt"], []).append(arm)
        for tag, garms in groups.items():
            perc = None
            if tag != "gt":
                from perception import Perception
                print(f"[load] {tag} <- {ckpts[tag]}", flush=True)
                perc = Perception(ckpts[tag], device="cuda")
            for ei, ep in enumerate(episodes):
                for arm in garms:
                    planner = LocalPlanner(max_speed=arm["speed"],
                                           brake_dist=args.brake_dist,
                                           robot_radius=args.robot_radius)
                    ep_r = {"scene": ep["scene"], "start": ep["start"], "yaw": ep["yaw"],
                            "goal": ep["goal"], "geo": ep["geo"]}
                    try:
                        # _EpClient threads the episode's dataset into reset (ReplicaCAD)
                        m = run_episode(_EpClient(state["client"], ep.get("dataset"),
                                                  args.guide),
                                        perc, planner, ep_r, controller=arm["controller"],
                                        time_scale=args.time_scale)
                    except Exception as e:
                        print(f"[ep {ei} arm {arm['name']} FAILED: {e}; restarting]", flush=True)
                        restart()
                        continue
                    m.update({"arm": arm["name"], "ep_idx": ei})
                    rows.append(m)
                    fout.write(json.dumps(m) + "\n"); fout.flush()
                if (ei + 1) % 10 == 0:
                    print(f"[{tag} ep {ei+1}/{len(episodes)}]", flush=True)
            if perc is not None:
                del perc
                import torch
                torch.cuda.empty_cache()
    finally:
        fout.close()
        try:
            state["client"].close()
        except Exception:
            pass
        try:
            state["server"].terminate()
        except Exception:
            pass

    summ = aggregate(rows, "arm")
    json.dump(summ, open(os.path.join(args.out, f"{args.mode}_summary.json"), "w"), indent=2)
    print(json.dumps(summ, indent=2), flush=True)
    print(f"=== SIM_{args.mode.upper()}_DONE n={len(rows)} ===", flush=True)


class _EpClient:
    """Wraps SimClient so run_episode's reset() carries the episode's dataset + guide."""
    def __init__(self, client, dataset, guide="geodesic"):
        self.c = client; self.dataset = dataset; self.guide = guide
    def reset(self, scene, start, yaw, goal):
        return self.c.reset(scene, start, yaw, goal, self.dataset, self.guide)
    def step(self, v, yaw_rate, dt):
        return self.c.step(v, yaw_rate, dt)


if __name__ == "__main__":
    main()
