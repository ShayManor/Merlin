#!/usr/bin/env python3
"""One closed-loop navigation episode with latency coupling -- the core of the M1/M2
sim. Perception runs at a controller-chosen operating point; its latency advances sim
time while the robot keeps executing its last command, so a slow op point makes the map
the robot acts on STALE. This is the dynamic effect offline depth proxies cannot capture.

Controllers (M2): fixed_252 / fixed_378 / fixed_518, and `adaptive` (pick the highest-res
op point whose Jetson latency fits the motion deadline derived from the robot's current
speed -- faster motion -> tighter deadline -> lower res -> fresher map).

Perception=None uses the sim GT depth (the perfect-perception upper bound, for M0 smoke).
"""
import numpy as np

from costmap import range_scan, min_forward_range
from perception import LATENCY_MS, WATTS, RES_LIST


def choose_res(controller, prev_speed, travel_budget):
    if controller.startswith("fixed_"):
        return int(controller.split("_")[1])
    if controller == "adaptive":
        deadline_ms = 1000.0 * travel_budget / max(prev_speed, 0.02)
        ok = [r for r in RES_LIST if LATENCY_MS[r] <= deadline_ms]
        return max(ok) if ok else min(RES_LIST, key=lambda r: LATENCY_MS[r])
    raise ValueError(controller)


def run_episode(client, perception, planner, ep, controller="fixed_378",
                success_radius=0.4, max_sim_time=120.0,
                travel_budget=0.10, time_scale=1.0, stuck_patience=70):
    """Returns a per-episode metrics dict. time_scale multiplies op-point latency
    (lets us probe the staleness mechanism without retraining)."""
    obs = client.reset(ep["scene"], ep["start"], ep["yaw"], ep["goal"])
    hfov = obs["hfov"]
    geo0 = ep["geo"]
    if perception is not None:
        perception.calibrate(obs["rgb"], obs["depth"], res=378)

    sim_t = 0.0
    prev_speed = planner.max_speed
    res_hist, speed_hist, perr, stale_m = [], [], [], []
    best_geo = obs["geo_dist"]; stuck = 0
    energy_J = 0.0
    success = False

    while sim_t < max_sim_time:
        if np.isfinite(obs["geo_dist"]) and obs["geo_dist"] <= success_radius:
            success = True
            break
        res = choose_res(controller, prev_speed, travel_budget)
        if perception is not None:
            depth = perception.depth(obs["rgb"], res=res)
        else:
            depth = obs["depth"]                      # GT-perception upper bound
        bearings, ranges = range_scan(depth, hfov)
        v, yaw_rate, info = planner.command(obs["goal_bearing"], bearings, ranges)

        # perception-error diagnostic (forward range vs GT) -- ties behavior to M1 metric
        if perception is not None:
            gb, gr = range_scan(obs["depth"], hfov)
            pf, gf = min_forward_range(bearings, ranges), min_forward_range(gb, gr)
            if np.isfinite(pf) and np.isfinite(gf):
                perr.append(abs(pf - gf))

        # latency window: robot executes (v, yaw_rate) for L seconds on this stale map.
        # One navmesh step per window (motion < ~0.2 m): gives the standard
        # one-collision-per-decision semantics (substepping a blocked window would
        # slide-and-recount, exploding collisions/m while path_len stays ~0).
        L = (LATENCY_MS[res] / 1000.0) * time_scale
        obs = client.step(v, yaw_rate, L)
        sim_t += L
        energy_J += WATTS[res] * L
        stale_m.append(v * L)
        res_hist.append(res); speed_hist.append(v)
        prev_speed = max(v, 0.05)

        g = obs["geo_dist"]
        if np.isfinite(g):
            if g < best_geo - 0.05:
                best_geo = g; stuck = 0
            else:
                stuck += 1
        if stuck >= stuck_patience:
            break

    path_len = obs["path_len"]
    spl = (geo0 / max(path_len, geo0)) if success else 0.0
    cpm = obs["collisions"] / max(path_len, 1e-3)
    return {
        "scene": ep["scene"], "geo0": round(geo0, 3), "controller": controller,
        "success": int(success), "spl": round(spl, 4),
        "collisions": int(obs["collisions"]), "path_len": round(path_len, 3),
        "collisions_per_m": round(cpm, 4), "n_ticks": len(res_hist),
        "sim_time": round(sim_t, 2), "energy_J": round(energy_J, 2),
        "energy_per_m": round(energy_J / max(path_len, 1e-3), 3),
        "mean_res": round(float(np.mean(res_hist)), 1) if res_hist else 0,
        "mean_speed": round(float(np.mean(speed_hist)), 4) if speed_hist else 0.0,
        "mean_stale_m": round(float(np.mean(stale_m)), 4) if stale_m else 0.0,
        "fwd_perr_m": round(float(np.mean(perr)), 4) if perr else float("nan"),
        "final_geo": round(float(obs["geo_dist"]), 3) if np.isfinite(obs["geo_dist"]) else -1,
    }
