#!/usr/bin/env python3
"""One closed-loop flight episode (C4). PX4 SIH supplies GT pose (flight state) + IMU;
the drone navigates using ONLY Merlin depth -> voxel map -> 3D planner to avoid obstacles
and reach the goal. Collisions are geometric (renderer ray casts against the scene mesh).

Scope: this loop produces the C4 metrics (success / SPL / collisions / path). It does NOT
compute drift (C3) or scale (C2) inline -- those need a visual-inertial odometry chain vs
GT and are computed by the driver from the per-keyframe log this returns (gt pose + IMU +,
when available, the student visual pose). Computing an ATE from GT-vs-GT here would be
degenerate.
"""
import time

import numpy as np

from eval.drone.frames import HAB_CAM_TO_OPTICAL
from eval.drone.mapper import VoxelMap
from eval.drone.maps import build_map
from eval.drone.planner import Planner3D
from eval.drone.staleness import Staleness
from eval.drone.vi import path_length
from eval.drone.metrics import spl


def _unit(v):
    n = float(np.linalg.norm(v))
    return np.asarray(v, float) / n if n > 1e-9 else np.array([0.0, 0.0, -1.0])


def obstacle_map_stats(vmap, boxes, pos_hab):
    """(nearest occupied voxel INSIDE an injected obstacle, how many there are).

    map_dist alone cannot tell "the obstacle is missing from the map" from "the obstacle is
    mapped and the planner flew into it anyway": both leave the nearest voxel sitting at the
    safety radius, because that is where the planner holds station relative to whatever else
    it can see. Restricting the query to the obstacle's own bounding box separates them.
    """
    if not boxes:
        return None, 0
    pts = vmap.occupied_points()
    if pts.shape[0] == 0:
        return None, 0
    inside = np.zeros(len(pts), bool)
    for c, s in boxes:
        h = 0.5 * np.asarray(s, float) + vmap.voxel + 0.10
        inside |= (np.abs(pts - np.asarray(c, float)) <= h).all(1)
    if not inside.any():
        return None, 0
    d = np.linalg.norm(pts[inside] - np.asarray(pos_hab, float), axis=1)
    return round(float(d.min()), 3), int(inside.sum())


def run_episode(bridge, client, depth_provider, ep, waypoints_ned, cfg):
    """depth_provider(render_out, res) -> (H,W) metric depth. The student arm wraps
    Perception.depth(out['rgb'], res); the GT upper-bound arm returns out['depth'] directly.
    Rendering at the model resolution keeps out['K'] consistent with the returned depth.
    ep: eval.drone.frames.EpisodeFrame placing the PX4-NED frame at the interior start.
    waypoints_ned: the geodesic route (episode-relative NED subgoals; last = final goal).
    The drone steers toward the current waypoint (global route around walls) while the
    perception voxel map handles local obstacle avoidance -- global+local, like a nav stack."""
    wps = [np.asarray(w, float) for w in waypoints_ned]
    if len(wps) == 0:
        return {"success": 0.0, "spl": 0.0, "collisions": 0, "path_len": 0.0,
                "geo": 0.0, "mean_age": 0.0, "n_keyframes": 0, "log": []}
    goal_ned = wps[-1]
    # wps[0] is the start (~origin in episode-relative NED), so the route length is the
    # geodesic distance used for SPL.
    geo = float(path_length(np.array(wps))) if len(wps) > 1 else float(np.linalg.norm(goal_ned))

    # hit_w must be forwarded like carve_w. It was not, so in flight the map always ran
    # symmetric 1:1 evidence while the offline diagnostic swept it -- the same two-mappers,
    # two-conclusions trap that carve_w fell into.
    # The planner reaches the map only through any_occupied_within/unknown/voxel, so the
    # representation is swappable without touching it. Default "voxel" is the counting map,
    # so every existing cell is unchanged; eval/drone/maps.py holds the benchmark alternatives.
    vmap = build_map(cfg.get("map", "voxel"),
                     voxel=cfg.get("voxel", 0.15), min_hits=cfg.get("min_hits", 2),
                     carve_w=cfg.get("carve_w", 1), hit_w=cfg.get("hit_w", 1),
                     sigma_table=cfg.get("sigma_table"), l_thresh=cfg.get("l_thresh", 1.2),
                     k_sigma=cfg.get("k_sigma", 2.0), free_w=cfg.get("free_w", 0.55),
                     window=cfg.get("window", 12), agree=cfg.get("agree", 1),
                     risk_k=cfg.get("risk_k", 1.5))
    plan = Planner3D(step=cfg.get("step", 0.4), safety=cfg.get("safety", 0.5),
                     max_speed=cfg.get("max_speed", 1.0), escape=cfg.get("escape", True),
                     require_known=cfg.get("require_known", False),
                     min_clear=cfg.get("robot_radius", 0.18)
                     + cfg.get("voxel", 0.15) * 0.8661)
    stal = Staleness(res=cfg.get("res", 378), latency_ms=cfg.get("latency_ms"))

    bridge.set_speed_limit(cfg.get("max_speed", 1.0))
    bridge.set_mode_offboard_and_arm()
    bridge.takeoff(cfg.get("alt", 0.75))
    start_ned = bridge.get_state()["pos_ned"].copy()
    # stream the target at 20 Hz from a background thread so OFFBOARD survives slow
    # perception/planning ticks (the loop below only updates the target).
    bridge.start_streaming(start_ned[0], start_ned[1], start_ned[2])

    gt_path = []
    log = []
    ages = []
    collisions = 0
    collision_events = []
    crashed = False      # a crash ends the episode as a failure (drone-nav convention)
    failsafe = False     # PX4 took control; the episode is void, not a nav result
    in_contact = False   # count collision EVENTS (no-hit -> hit), not every grazing tick
    success = False
    rng = np.random.default_rng(cfg.get("noise_seed", 0))
    # drift is per-episode state (it integrates distance flown), unlike the stateless jitter
    drift = None
    if cfg.get("pose_drift") or cfg.get("yaw_drift_deg"):
        from eval.drone.noise import PoseDrift
        drift = PoseDrift(cfg.get("pose_drift", 0.0), cfg.get("yaw_drift_deg", 0.0), rng)
    blocked_ticks = 0
    prev_pos = None
    last_vel = None
    obs_first = None      # range at which the injected obstacle first entered the map
    obs_vox_max = 0
    n_ticks = 0
    stuck = 0
    idx = 0
    t0 = time.time()
    timeout = cfg.get("timeout", 90.0)
    goal_tol = cfg.get("goal_tol", 0.6)
    wp_tol = cfg.get("wp_tol", 0.6)
    robot_r = cfg.get("robot_radius", 0.18)   # < navmesh clearance (0.35) so no spawn collision

    last_px4_t = None
    frozen = 0
    while time.time() - t0 < timeout:
        st = bridge.get_state()
        # PX4's IMU clock must always advance. If it stops, PX4 died (e.g. another job's
        # cleanup killed it) and every later episode silently records a motionless drone as
        # an ordinary timeout -- so surface it instead of logging plausible-looking garbage.
        if st["t"] == last_px4_t:
            frozen += 1
        else:
            frozen = 0
            last_px4_t = st["t"]
        if frozen > 200:
            bridge.stop_streaming()
            raise RuntimeError("PX4 telemetry frozen (clock stopped advancing) -- PX4 is dead")
        # A failsafe silently overrides OFFBOARD: setpoints stream on and are ignored, and
        # the episode records as a timeout that has nothing to do with navigation.
        if hasattr(bridge, "in_failsafe") and bridge.in_failsafe():
            failsafe = True
            break
        rel_ned = st["pos_ned"] - start_ned
        pos_hab = ep.ned_to_hab_pos(rel_ned)
        vel_hab = (pos_hab - prev_pos) if prev_pos is not None else None
        if vel_hab is not None and np.linalg.norm(vel_hab) < 1e-3:
            vel_hab = last_vel
        else:
            last_vel = vel_hab
        prev_pos = pos_hab.copy()
        R_cam2world = ep.ned_quat_to_hab_R(st["quat_frd2ned"])
        sim_t = time.time() - t0

        if stal.due(sim_t):
            out = client.render(pos_hab, R_cam2world)
            depth = depth_provider(out, cfg.get("res", 378))
            # the map is normally assembled from PX4 GT pose (perfect odometry); pose_noise
            # degrades that oracle towards the real-rover condition when enabled
            ipos, iR = pos_hab, R_cam2world
            if drift is not None:
                ipos, iR = drift.apply(ipos, iR)
            if cfg.get("pose_noise") or cfg.get("rot_noise_deg"):
                from eval.drone.noise import perturb_pose
                ipos, iR = perturb_pose(ipos, iR, cfg.get("pose_noise", 0.0),
                                        cfg.get("rot_noise_deg", 0.0), rng)
            # iR is the Habitat camera matrix render() was driven with; the map needs the
            # optical convention or the whole point cloud lands behind the drone
            vmap.integrate(depth, out["K"], iR @ HAB_CAM_TO_OPTICAL, ipos,
                           max_range=cfg.get("max_range", 8.0),
                           carve=cfg.get("carve", True))
            stal.mark(sim_t)
            od, ov = obstacle_map_stats(vmap, cfg.get("obs_boxes"), pos_hab)
            if ov and obs_first is None:
                obs_first = round(float(min(np.linalg.norm(pos_hab - np.asarray(c, float))
                                            for c, _ in cfg["obs_boxes"])), 2)
            obs_vox_max = max(obs_vox_max, ov)
            log.append({"t": sim_t, "pos_ned": rel_ned.copy(),
                        "quat_frd2ned": st["quat_frd2ned"].copy(),
                        "acc": st["acc"].copy(), "gyro": st["gyro"].copy()})
        ages.append(stal.age(sim_t))

        col = client.collide(pos_hab, robot_r)
        hit = bool(col.get("hit"))
        if hit and not in_contact:
            collisions += 1
            # What the map believed at the moment of impact splits the two failure modes:
            # a large distance means the obstacle was never mapped (perception), a small one
            # means it was mapped and the planner flew into it anyway (planning).
            collision_events.append({"t": sim_t, "pos_ned": rel_ned.copy().tolist(),
                                     "object_id": col.get("object_id"),
                                     "map_dist": round(vmap.nearest_obstacle_dist(pos_hab), 3),
                                     "n_voxels": len(vmap._occ_codes()),
                                     "obs_map_dist": obstacle_map_stats(
                                         vmap, cfg.get("obs_boxes"), pos_hab)[0],
                                     "obs_voxels": obstacle_map_stats(
                                         vmap, cfg.get("obs_boxes"), pos_hab)[1],
                                     # Had the camera ever looked where the drone now is?
                                     # A collision in unobserved space is an FOV/planning
                                     # failure, not a depth-accuracy one: the map cannot be
                                     # wrong about geometry it never saw.
                                     "unseen": bool(vmap.unknown(
                                         np.atleast_2d(pos_hab))[0]),
                                     # how far the drone had to turn to be looking where it
                                     # was going; the camera spans +-45 deg
                                     "look_err_deg": round(float(np.degrees(np.arccos(
                                         np.clip(np.dot(
                                             R_cam2world @ np.array([0.0, 0.0, -1.0]),
                                             _unit(vel_hab)), -1, 1)))), 1)
                                     if vel_hab is not None else None,
                                     "is_novel": bool(col.get("is_novel", False))})
        in_contact = hit
        if hit:
            crashed = True
            break

        # advance along the route as waypoints are reached
        while idx < len(wps) - 1 and np.linalg.norm(rel_ned - wps[idx]) < wp_tol:
            idx += 1
        cur_wp_hab = ep.ned_to_hab_pos(wps[idx])
        tgt_hab, blocked = plan.next_setpoint(vmap, pos_hab, cur_wp_hab)
        # The planner validates a fixed short segment; the commanded setpoint is pushed out
        # along that validated direction to `carrot` so the position controller actually
        # saturates the speed limit. Keeping the clearance horizon fixed across ablation cells
        # means speed varies without also varying how conservative the planner is.
        carrot = cfg.get("carrot", 0.0)
        d = tgt_hab - pos_hab
        nd = float(np.linalg.norm(d))
        if carrot > nd > 1e-6:
            tgt_hab = pos_hab + d / nd * carrot
        n_ticks += 1
        blocked_ticks += int(bool(blocked))
        # A subgoal swallowed by an obstacle can never be reached, and the drone would orbit
        # it until timeout. The global route is a suggestion, so drop a subgoal the planner
        # has been unable to approach and steer at the next one. The final goal is never
        # skipped -- reaching it is the success criterion.
        stuck = stuck + 1 if blocked else 0
        if stuck > cfg.get("skip_ticks", 40) and idx < len(wps) - 1:
            idx += 1
            stuck = 0
        tgt_ned = ep.hab_to_ned_pos(tgt_hab) + start_ned   # command in absolute PX4 NED
        yaw = float(np.arctan2(tgt_ned[1] - st["pos_ned"][1], tgt_ned[0] - st["pos_ned"][0]))
        bridge.set_target(float(tgt_ned[0]), float(tgt_ned[1]), float(tgt_ned[2]), yaw=yaw)

        gt_path.append(rel_ned.copy())
        if np.linalg.norm(rel_ned - goal_ned) < goal_tol:
            success = True
            break
        time.sleep(cfg.get("ctrl_dt", 0.05))

    bridge.stop_streaming()

    gt = np.array(gt_path)
    dur = time.time() - t0
    pl = path_length(gt) if len(gt) > 1 else 0.0
    final_dist = float(np.linalg.norm(gt[-1] - goal_ned)) if len(gt) else float("nan")
    mean_age = float(np.mean(ages)) if ages else 0.0
    return {"success": 0.0 if crashed else float(success),
            "spl": 0.0 if crashed else spl(success, geo, pl),
            "collisions": collisions, "collision_events": collision_events,
            "path_len": pl, "geo": geo,
            "mean_age": mean_age,
            # achieved speed and the distance flown on a held map -- the ablation's real
            # independent variable; both must be non-zero for a cell to mean anything.
            "mean_speed": round(pl / dur, 3) if dur > 1e-6 else 0.0,
            "blind_dist": round(mean_age * pl / dur, 3) if dur > 1e-6 else 0.0,
            "n_keyframes": len(log), "log": log,
            "wp_reached": idx, "n_wps": len(wps), "final_dist": round(final_dist, 2),
            "blocked_frac": round(blocked_ticks / max(1, n_ticks), 2),
            "obs_first": obs_first, "obs_vox_max": obs_vox_max,
            # A drone that never flew is not a navigation result. PX4's SIH degrades over a
            # long job (battery drain overriding OFFBOARD) and the drone then takes off,
            # drifts a metre or two and stops -- which `in_failsafe` does not always catch,
            # and which otherwise records as an ordinary timeout. Measured: one 4-hour job
            # logged 88 of 91 episodes this way, and the arms' success rates all collapsed
            # together, which reads as "hard scenes" rather than "dead simulator".
            "ended": ("collision" if crashed else "goal" if success
                      else "px4_failsafe" if failsafe
                      else "px4_frozen" if (dur > 0.5 * timeout and pl / max(dur, 1e-6) < 0.05)
                      else "timeout")}
