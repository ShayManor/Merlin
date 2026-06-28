#!/usr/bin/env python3
"""Run the depth-quality ladder under the faithful Nav2-style stack (nav2_stack.py): persistent
inflated costmap + clearance-aware A* + DWA local controller. THE make-or-break test of whether
'perception fidelity does not help reactive navigation' survives a properly engineered planner.

If success now RISES with depth fidelity (gt/adapted > orig), fidelity matters once the planner
is engineered (inverse law was a naive-planner artifact). If the inverse/flat behaviour
survives, the finding is robust to standard practice.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "model", "distill"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "model", "backbones"))
from sim_client import SimClient        # noqa: E402
from nav2_stack import Nav2Stack         # noqa: E402


def spawn_server(port, conda_sh, env_name):
    cmd = (f"source {conda_sh} && conda activate {env_name} && "
           f"MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet "
           f"python {os.path.join(HERE, 'habitat_server.py')} --port {port}")
    p = subprocess.Popen(["bash", "-lc", cmd], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
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


def run_episode(client, perception, ep, res=378, success_radius=0.4, max_sim_time=120.0,
                stuck_patience=90, no_slide=False, max_ticks=400, replan_every=3, max_speed=0.5):
    obs = client.reset(ep["scene"], ep["start"], ep["yaw"], ep["goal"],
                       ep.get("dataset"), guide="straight", no_slide=no_slide)
    hfov = obs["hfov"]
    if perception is not None:
        perception.calibrate(obs["rgb"], obs["depth"], res=378)
    nav = Nav2Stack(ep["start"], ep["goal"], max_speed=max_speed)
    from perception import LATENCY_MS
    sim_t = 0.0; best_geo = obs["geo_dist"]; stuck = 0; success = False; ticks = 0; v_prev = 0.0
    nav.plan(ep["start"])
    while sim_t < max_sim_time and ticks < max_ticks:
        ticks += 1
        if np.isfinite(obs["geo_dist"]) and obs["geo_dist"] <= success_radius:
            success = True; break
        depth = obs["depth"] if perception is None else perception.depth(obs["rgb"], res=res)
        nav.update(obs["pos"], obs["yaw"], depth, hfov)
        nav.cost_field()
        if ticks % replan_every == 1:
            nav.plan(obs["pos"])
        v, yaw_rate = nav.dwa_command(obs["pos"], obs["yaw"], v_prev)
        L = LATENCY_MS[res] / 1000.0
        obs = client.step(v, yaw_rate, L); sim_t += L; v_prev = v
        g = obs["geo_dist"]
        if np.isfinite(g):
            if g < best_geo - 0.05:
                best_geo = g; stuck = 0
            else:
                stuck += 1
        if stuck >= stuck_patience:
            break
    pl = obs["path_len"]
    spl = (ep["geo"] / max(pl, ep["geo"])) if success else 0.0
    return {"scene": ep["scene"], "success": int(success), "spl": round(spl, 4),
            "collisions": int(obs["collisions"]), "path_len": round(pl, 3),
            "collisions_per_m": round(obs["collisions"] / max(pl, 1e-3), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--dataset", default="/root/habitat-data/replica_cad/replicaCAD.scene_dataset_config.json")
    ap.add_argument("--ckpts", nargs="*", default=[], help="tag=path ... (plus implicit gt)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--min-geo", type=float, default=2.0)
    ap.add_argument("--max-geo", type=float, default=5.5)
    ap.add_argument("--no-slide", action="store_true")
    ap.add_argument("--max-speed", type=float, default=0.5)
    ap.add_argument("--port", type=int, default=5850)
    ap.add_argument("--conda-sh", default="/workspace/miniconda3/etc/profile.d/conda.sh")
    ap.add_argument("--env-name", default="/root/conda_envs/habenv")
    ap.add_argument("--out", default="/workspace/ckpt/sim_nav2")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    state = {"server": spawn_server(args.port, args.conda_sh, args.env_name),
             "client": None, "port": args.port}
    state["client"] = SimClient(state["port"], timeout=120.0)

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
        print(f"[restart -> {state['port']}]", flush=True)
        state["server"] = spawn_server(state["port"], args.conda_sh, args.env_name)
        state["client"] = SimClient(state["port"], timeout=120.0)

    arms = [("gt", None)]
    if not args.validate:
        from perception import Perception
        for c in args.ckpts:
            tag, path = c.split("=", 1)
            arms.append((tag, Perception(path)))

    eps = []
    for si, scene in enumerate(args.scenes):
        got = 0; seed = 0
        while got < args.episodes and seed < args.episodes * 25:
            try:
                ep = state["client"].sample(scene, args.min_geo, args.max_geo,
                                            7000 + 1000 * si + seed, args.dataset)
            except Exception:
                restart(); ep = None
            seed += 1
            if ep:
                ep["dataset"] = args.dataset; eps.append(ep); got += 1
    print(f"[nav2] {len(eps)} episodes, arms={[a for a, _ in arms]}", flush=True)

    rows = []
    fout = open(os.path.join(args.out, "nav2_rows.jsonl"), "w")
    for tag, perc in arms:
        succ = []
        for ei, ep in enumerate(eps):
            try:
                r = run_episode(state["client"], perc, ep, no_slide=args.no_slide,
                                max_speed=args.max_speed)
            except Exception as e:
                print(f"[ep {ei} {tag} fail: {repr(e)[:80]}; restart]", flush=True)
                restart(); continue
            r["arm"] = tag; r["ep_idx"] = ei; rows.append(r); succ.append(r["success"])
            fout.write(json.dumps(r) + "\n"); fout.flush()
        m, lo, hi = boot_ci(succ)
        cpm = np.mean([x["collisions_per_m"] for x in rows if x["arm"] == tag] or [0])
        print(f"  {tag}: success {m:.3f} [{lo:.3f},{hi:.3f}] coll/m {cpm:.3f} n={len(succ)}", flush=True)
    fout.close()
    print("=== RUN_NAV2_DONE ===", flush=True)
    try:
        state["client"].close(); state["server"].terminate()
    except Exception:
        pass


if __name__ == "__main__":
    main()
