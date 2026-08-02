#!/usr/bin/env python3
"""Drone-sim experiment driver (C2 scale, C3 drift, C4 nav, staleness ablation).

Mirrors eval/sim/run_experiments.py: spawns the Habitat renderer (habenv) as a subprocess
at the model resolution (so out['K'] matches the returned depth), samples a fixed
deterministic interior-episode set per scene, runs paired arms, and writes per-episode JSONL
plus a summary JSON via eval/drone/metrics.aggregate. PX4 SIH is launched by the sbatch
wrapper (eval/drone/run_px4.sh); this driver connects over pymavlink.

Arms (C4): 'student' = distilled model depth, GT-scale-calibrated once per episode (isolates
depth-shape quality from scale); 'gt' = renderer GT depth (perfect-perception upper bound).
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

from eval.drone.frames import EpisodeFrame, quat_to_R, frd_acc_to_cam
from eval.drone.metrics import aggregate, boot_ci
from eval.drone.nav_loop import run_episode
from eval.drone.px4_bridge import PX4Bridge
from eval.drone.noise import random_depth
from eval.drone.perception import Perception, decompress_steps
from eval.drone.sim_client import SimClient
from eval.drone.vi import vi_scale, ate, path_length

WS = os.environ.get("MERLIN_WS", "/scratch/gautschi/manors/merlin_drone")
# Both the renderer socket and PX4's MAVLink port are per-instance, so two jobs co-located on
# one node must not share them (see run_px4.sh).
PX4_INSTANCE = int(os.environ.get("MERLIN_PX4_INSTANCE", 0))
PORT = 5601 + PX4_INSTANCE


def spawn_renderer(port, res):
    sq = (res // 14) * 14
    logf = open(f"{WS}/rend_{port}.log", "w")
    p = subprocess.Popen([WS + "/envs/habenv/bin/python", "eval/drone/renderer.py",
                          "--port", str(port), "--width", str(sq), "--height", str(sq)],
                         stdout=logf, stderr=subprocess.STDOUT)
    return p


def _round(d):
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}


def densify(waypoints, spacing=1.0):
    """Interpolate a subgoal every `spacing` metres along the polyline. The navmesh returns
    only corner points, so a nearly-straight path has just start+goal; the reactive planner
    steers toward the nearest subgoal, so frequent subgoals prevent far-goal local-minima
    stalls (helps both the student and GT arms)."""
    wps = [np.asarray(w, float) for w in waypoints]
    if len(wps) < 2:
        return wps
    out = [wps[0]]
    for a, b in zip(wps[:-1], wps[1:]):
        d = np.linalg.norm(b - a)
        n = max(1, int(np.ceil(d / spacing)))
        for i in range(1, n + 1):
            out.append(a + (b - a) * (i / n))
    return out


def _persist_episode(args, client, ep, m, tag, arm, n_obs, obs_info, wps_hab, perc,
                     scene=None, seed=None):
    """Record everything needed to rebuild any figure offline: the full per-keyframe log
    (which the row itself drops), the obstacle manifest, collision events, and replayed
    frames. Called before obstacles are cleared, because replay only reproduces the frames
    the model saw while the scene still holds them."""
    if getattr(args, "save_frames", 0):
        try:
            from eval.drone.replay import capture
            capture(client, ep, m.get("log", []), f"{args.out}/frames", tag, args.res,
                    perc=perc, n=args.save_frames, events=m.get("collision_events"))
        except Exception as e:      # never let figure capture kill a run
            print(f"WARN frame capture failed for {tag}: {e!r}", flush=True)
    d = f"{args.out}/episode_logs"
    os.makedirs(d, exist_ok=True)
    def _j(v):
        return v.tolist() if hasattr(v, "tolist") else v
    json.dump({"tag": tag, "arm": arm, "n_obstacles": n_obs,
               # scene/seed stored explicitly: scene names contain underscores, so the tag
               # cannot be split back apart reliably
               "scene": scene, "seed": seed,
               "start_hab": [float(x) for x in getattr(ep, "start_hab", [0, 0, 0])],
               "yaw_hab": float(getattr(ep, "yaw_hab", 0.0)),
               "obstacles": obs_info.get("placed", []),
               "waypoints_hab": [[float(x) for x in w] for w in wps_hab],
               "collision_events": m.get("collision_events", []),
               "ended": m.get("ended"), "success": m.get("success"),
               "log": [{k: _j(v) for k, v in e.items()} for e in m.get("log", [])],
               "config": {"res": args.res, "alt": args.alt, "safety": args.safety,
                          "goal_tol": args.goal_tol, "timeout": args.timeout,
                          "max_speed": args.max_speed,
                          "rgb_noise": getattr(args, "rgb_noise", 0.0),
                          "motion_blur": getattr(args, "motion_blur", 0.0),
                          "pose_noise": getattr(args, "pose_noise", 0.0)}},
              open(f"{d}/{tag}.json", "w"))


def _noisy_rgb(args, rgb, rng):
    """Degrade the frame before inference when the realism knobs are on (default: off, so
    this returns the pristine render unchanged)."""
    if not (getattr(args, "rgb_noise", 0.0) or getattr(args, "motion_blur", 0.0)):
        return rgb
    from eval.drone.noise import add_rgb_noise, motion_blur
    return motion_blur(add_rgb_noise(rgb, getattr(args, "rgb_noise", 0.0), rng),
                       getattr(args, "motion_blur", 0.0))


def _c4_rows(args, jsonl_path=None, extra=None):
    """Run the C4 episode set for args (spawns its own renderer at args.res + PX4 bridge)
    and return the per-episode rows. Shared by run_c4 and the ablation sweep. Rows are
    appended to jsonl_path as each episode finishes, so a walltime kill keeps the data."""
    port = PORT
    proc = spawn_renderer(port, args.res)
    rows = []
    try:
        time.sleep(10)
        client = SimClient(port)
        bridge = PX4Bridge()
        assert bridge.wait_ready(40), "PX4 not ready"
        bridge.configure_sim()
        perc = Perception(args.ckpt) if args.ckpt else None
        if perc is not None and getattr(args, "depth_calib", None):
            perc.set_depth_calib(args.depth_calib, args.calib_q)
        cfg = {"res": args.res, "timeout": args.timeout, "alt": args.alt,
               "voxel": getattr(args, "voxel", 0.15),
               "min_hits": getattr(args, "min_hits", 2),
               "step": args.step, "safety": args.safety, "goal_tol": args.goal_tol, "wp_tol": 0.7,
               "max_range": getattr(args, "max_range", 8.0),
               "max_speed": args.max_speed,
               "carrot": getattr(args, "carrot", 0.0),
               "escape": not getattr(args, "no_escape", False),
               "carve": not getattr(args, "no_carve", False),
               "require_known": getattr(args, "require_known", False),
               "carve_w": getattr(args, "carve_w", 1),
               "hit_w": getattr(args, "hit_w", 1),
               "latency_ms": getattr(args, "latency_ms", None),
               "pose_noise": getattr(args, "pose_noise", 0.0),
               "rot_noise_deg": getattr(args, "rot_noise_deg", 0.0)}
        for scene in args.scenes:
            client.reset(scene, args.dataset, getattr(args, "nav_radius", None))
            if getattr(args, "save_frames", 0):
                try:    # floor plan for the path figure; cheap, once per scene
                    td = client.topdown(0.05)
                    os.makedirs(args.out, exist_ok=True)
                    np.savez_compressed(f"{args.out}/topdown_{os.path.basename(scene)}.npz",
                                        grid=td["grid"], bounds=np.array(td["bounds"]))
                except Exception as e:
                    print(f"WARN topdown failed for {scene}: {e!r}", flush=True)
            for k in range(args.episodes):
                sd = getattr(args, "seed0", 1000) + k
                epinfo = client.sample(scene, args.min_geo, args.max_geo, seed=sd,
                                       alt=args.alt, dataset=args.dataset,
                                       stale=getattr(args, "stale_route", False),
                                       route_clear=getattr(args, "route_clear", 0.0))
                if epinfo is None:
                    print(f"skip scene={os.path.basename(scene)} seed={1000+k}: no episode", flush=True)
                    continue
                ep = EpisodeFrame(epinfo["start_hab"], epinfo["yaw_hab"])
                wps_hab = epinfo.get("waypoints_hab") or [epinfo["start_hab"], epinfo["goal_hab"]]
                waypoints_ned = densify([ep.hab_to_ned_pos(w) for w in wps_hab], spacing=1.0)
                # Obstacles go in AFTER the route was computed on a clean navmesh, so the
                # route is stale by construction. Both arms fly the identical layout, so
                # placement is seeded per episode and cleared only after the arms loop.
                n_obs = int(getattr(args, "n_obs", 0) or 0)
                obs_info = {"placed": [], "skipped": 0}
                if n_obs > 0:
                    # Retry with a fresh layout rather than dropping the episode: obstacles on
                    # a narrow indoor corridor seal it about half the time, and skipping would
                    # silently drop the hardest scenes out of the matrix.
                    for attempt in range(6):
                        obs_info = client.place_obstacles(wps_hab, n_obs, seed=sd + 97 * attempt,
                                                          alt=args.alt,
                                                          intrude=getattr(args, "intrude", 0.25),
                                                          min_width=getattr(args, "obs_width", 0.0),
                                                          obs_clear=getattr(args, "obs_clear", 0.0),
                                                          block_margin=getattr(args, "block_margin", 0.20),
                                                          jitter=getattr(args, "obs_jitter", 0.35),
                                                          include=getattr(args, "obs_include", None) or None,
                                                          yaw=getattr(args, "obs_yaw", 0.0))
                        # a layout that placed fewer than n obstacles is not the cell it
                        # claims to be, so it counts as a failed attempt too
                        if (len(obs_info.get("placed", [])) == n_obs
                                and client.route_open(epinfo["start_hab"],
                                                      epinfo["goal_hab"]).get("open", True)):
                            break
                        client.clear_obstacles()
                    else:
                        print(f"skip scene={os.path.basename(scene)} seed={1000+k}: "
                              f"no layout places {n_obs} obstacles with a flyable gap "
                              f"(best {len(obs_info.get('placed', []))}, "
                              f"{obs_info.get('skipped')} sites rejected)", flush=True)
                        continue
                    print(f"OBS scene={os.path.basename(scene)} seed={1000+k} "
                          f"placed={len(obs_info.get('placed', []))} "
                          f"rejected={obs_info.get('skipped')} attempts={attempt+1}", flush=True)
                for arm in args.arms:
                    # Re-assert the SIH battery/failsafe parameters every episode. Setting
                    # them once per job is not enough: over a multi-hour run PX4 drifts back
                    # into a low-battery failsafe that silently overrides OFFBOARD, and every
                    # later episode logs a drone that took off, moved a metre and stopped.
                    bridge.configure_sim()
                    if arm == "blind":
                        # Control arm: no depth ever integrates, so the voxel map stays empty
                        # and the drone follows the geodesic route alone. The route is
                        # navmesh-derived and collision-free by construction, so this measures
                        # how much of C4's success comes from perception vs from the route.
                        # Inference still runs and its result is discarded, so the control loop
                        # keeps the student arm's timing -- otherwise a faster tick, not the
                        # empty map, could explain any difference.
                        def dp(out, r=args.res, _p=perc):
                            if _p is not None:
                                _p.depth(out["rgb"], res=r)
                            return np.full_like(out["depth"], np.nan)
                    elif arm == "gt":
                        dp = lambda out, r=None: out["depth"]
                    elif arm == "student_pfs":
                        # oracle per-frame scale: isolates scale error from shape error
                        dp = lambda out, r=args.res: perc.depth_gt_scaled(out["rgb"],
                                                                         out["depth"], res=r)
                    elif arm == "random":
                        # Uninformative-depth control: a smooth random field over the same
                        # range the student predicts. It gives the planner a map of the same
                        # size, density and update rate as MERLIN's, carrying no information
                        # about the scene -- so any gap between this arm and MERLIN is
                        # attributable to the depth being CORRECT, not merely present.
                        rrng = np.random.default_rng(sd)

                        def dp(out, r=args.res, _p=perc, _rng=rrng):
                            if _p is not None:
                                _p.depth(out["rgb"], res=r)   # keep the control loop timing
                            return random_depth(out["depth"].shape, _rng)
                    else:
                        R0 = ep.ned_quat_to_hab_R(np.array([0, 0, 0, 1.0]))
                        out0 = client.render(ep.ned_to_hab_pos([0, 0, 0]), R0)
                        perc.calibrate_gt(out0["rgb"], out0["depth"], res=args.res)
                        dp = lambda out, r=args.res: decompress_steps(
                            perc.depth(out["rgb"], res=r),
                            getattr(args, "calib", 0), args.alpha, args.cap)
                    # observed-space gating (off by default) would freeze the blind
                    # control arm, which never integrates depth
                    m = run_episode(bridge, client, dp, ep, waypoints_ned,
                                    dict(cfg, require_known=(cfg.get("require_known")
                                                             and arm != "blind"),
                                         obs_boxes=[(o["pos"], o["bbox"])
                                                    for o in obs_info.get("placed", [])]))
                    # Capture frames and persist the full log BEFORE obstacles are cleared:
                    # replay re-renders the recorded poses, which only reproduces what the
                    # model saw while the scene still holds the obstacles.
                    tag = f"{os.path.basename(scene)}_{1000+k}_{arm}_n{n_obs}"
                    _persist_episode(args, client, ep, m, tag, arm, n_obs, obs_info, wps_hab,
                                     perc if arm.startswith("student") else None,
                                     scene=os.path.basename(scene), seed=sd)
                    m.pop("log", None)
                    row = {"arm": arm, "scene": os.path.basename(scene), "seed": sd,
                           "geo_sampled": round(float(epinfo["geo"]), 3), **m}
                    if extra:
                        row.update(extra)
                    rows.append(row)
                    if jsonl_path:
                        with open(jsonl_path, "a") as jf:
                            jf.write(json.dumps(row) + "\n")
                    print("EP", json.dumps(_round(row)), flush=True)
                if n_obs > 0:
                    client.clear_obstacles()
        bridge.close()
        client.close()
    finally:
        proc.terminate()
    return rows


def run_c4(args):
    os.makedirs(args.out, exist_ok=True)
    path = f"{args.out}/c4_episodes.jsonl"
    open(path, "w").close()          # truncate; episodes are appended as they finish
    rows = _c4_rows(args, jsonl_path=path)
    summ = aggregate(rows, ["success", "spl", "collisions", "path_len"])
    json.dump(summ, open(f"{args.out}/c4_summary.json", "w"), indent=2)
    print("C4_SUMMARY", json.dumps(summ), flush=True)
    print(f"=== DRONE_C4_DONE n={len(rows)} ===", flush=True)


def fly_and_collect(bridge, client, ep, waypoints_ned, cfg):
    """Fly the drone through episode-relative NED waypoints, recording the full-rate IMU
    stream (drained from the bridge buffer, ~250 Hz) and RGB keyframes for offline C2/C3.
    Keyframe times and IMU times share the PX4 timebase (st['t'] / msg.time_usec). Returns
    (keyframes, imu): keyframes is a list of {t, rel_ned, quat, rgb}, imu is (M,4)
    [t, ax, ay, az] in body FRD. No obstacle avoidance -- a controlled collection flight."""
    import time
    bridge.set_mode_offboard_and_arm()
    bridge.takeoff(cfg.get("alt", 1.0))
    start_ned = bridge.get_state()["pos_ned"].copy()
    bridge.start_streaming(start_ned[0], start_ned[1], start_ned[2])
    bridge.drain_imu()   # discard pre-flight/takeoff IMU
    kf_dt = cfg.get("kf_dt", 0.25)
    reach_tol = cfg.get("reach_tol", 0.4)
    kfs = []
    last_kf = -1e9
    for wp in waypoints_ned:
        wp = np.asarray(wp, float)
        tw = time.time()
        while time.time() - tw < cfg.get("wp_timeout", 12.0):
            st = bridge.get_state()
            rel = st["pos_ned"] - start_ned
            tgt = wp + start_ned
            yaw = float(np.arctan2(tgt[1] - st["pos_ned"][1], tgt[0] - st["pos_ned"][0]))
            bridge.set_target(float(tgt[0]), float(tgt[1]), float(tgt[2]), yaw=yaw)
            if time.time() - last_kf >= kf_dt:
                pos_hab = ep.ned_to_hab_pos(rel)
                R = ep.ned_quat_to_hab_R(st["quat_frd2ned"])
                out = client.render(pos_hab, R)
                kfs.append({"t": float(st["t"]), "rel_ned": rel.copy(),
                            "quat": st["quat_frd2ned"].copy(), "rgb": out["rgb"]})
                last_kf = time.time()
            if np.linalg.norm(rel - wp) < reach_tol:
                break
            time.sleep(cfg.get("ctrl_dt", 0.05))
    imu = bridge.drain_imu()
    bridge.stop_streaming()
    return kfs, imu


def run_c2(args):
    """C2 metric scale: fly a short motion-rich trajectory, run windowed student inference to
    get an up-to-scale visual trajectory, recover scale from the IMU (vi_scale), compare to
    the GT scale (Umeyama visual->GT). Reports scale error % (target <5%)."""
    port = PORT
    proc = spawn_renderer(port, args.res)
    try:
        time.sleep(10)
        client = SimClient(port)
        bridge = PX4Bridge()
        assert bridge.wait_ready(40), "PX4 not ready"
        bridge.configure_sim()
        perc = Perception(args.ckpt)
        K = args.window
        rows = []
        for scene in args.scenes:
            client.reset(scene, args.dataset, getattr(args, "nav_radius", None))
            for k in range(args.episodes):
                epinfo = client.sample(scene, args.min_geo, args.max_geo, seed=2000 + k, alt=args.alt, dataset=args.dataset)
                if epinfo is None:
                    continue
                ep = EpisodeFrame(epinfo["start_hab"], epinfo["yaw_hab"])
                # a larger zig-zag with altitude change gives more per-keyframe baseline +
                # acceleration for visual-inertial observability (small motion -> degenerate VO)
                wps = [[1.5, 0, 0], [2.5, 1.2, -0.5], [3.8, -0.6, 0.3], [5.0, 0.9, -0.2], [6.0, 0, 0]]
                kfs, imu = fly_and_collect(bridge, client, ep, wps,
                                           {"res": args.res, "alt": 1.0, "kf_dt": 0.4})
                if len(kfs) < K or imu.shape[0] < 2 * K:
                    continue
                imu_cam = np.array([frd_acc_to_cam(a) for a in imu[:, 1:4]]) if imu.shape[0] else imu
                for w0 in range(0, len(kfs) - K + 1, max(1, K // 2)):
                    win = kfs[w0:w0 + K]
                    cam_trans, cam_quats = perc.forward_window([f["rgb"] for f in win], res=args.res)
                    R_list = [quat_to_R(q) for q in cam_quats]
                    kf_t = np.array([f["t"] for f in win])
                    s = vi_scale(cam_trans, R_list, kf_t, imu[:, 0], imu_cam)
                    gt = np.array([f["rel_ned"] for f in win])
                    # GT scale via Umeyama extent ratio (visual->GT), following eval/vi_scale.py
                    Xc = cam_trans - cam_trans.mean(0)
                    Yc = gt - gt.mean(0)
                    denom = (Xc ** 2).sum() / len(win)
                    U, S, Vt = np.linalg.svd((Yc.T @ Xc) / len(win))
                    s_gt = float(S.sum() / denom) if denom > 1e-9 else float("nan")
                    if not (np.isfinite(s) and np.isfinite(s_gt) and s > 0 and 7 < abs(s_gt) and s_gt < 1e4):
                        pass
                    err = abs(s - s_gt) / s_gt if (np.isfinite(s) and np.isfinite(s_gt) and s_gt > 1e-6) else float("nan")
                    row = {"arm": "student", "scene": os.path.basename(scene), "seed": 2000 + k,
                           "w0": w0, "scale_est": round(float(s), 4), "scale_gt": round(float(s_gt), 4),
                           "scale_err_pct": round(float(err) * 100, 3) if np.isfinite(err) else None}
                    rows.append(row)
                    print("C2", json.dumps(row), flush=True)
        os.makedirs(args.out, exist_ok=True)
        with open(f"{args.out}/c2_windows.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        errs = [r["scale_err_pct"] for r in rows if r.get("scale_err_pct") is not None]
        m, lo, hi = boot_ci(errs) if errs else (float("nan"),) * 3
        summ = {"n_windows": len(rows), "n_valid": len(errs),
                "scale_err_pct": {"mean": round(m, 3), "ci": [round(lo, 3), round(hi, 3)],
                                  "median": round(float(np.median(errs)), 3) if errs else None}}
        json.dump(summ, open(f"{args.out}/c2_summary.json", "w"), indent=2)
        print("C2_SUMMARY", json.dumps(summ), flush=True)
        print(f"=== DRONE_C2_DONE n={len(rows)} ===", flush=True)
        bridge.close()
        client.close()
    finally:
        proc.terminate()


def _sweep_axis(args):
    """The ablation's first axis: Jetson resolution by default, or explicit map-staleness
    latencies when --latencies is given. Returns (tag, key, values)."""
    if getattr(args, "latencies", None):
        return "l", "latency_ms", args.latencies
    return "r", "res", args.res_list


def _ablation_grid(all_rows, args):
    """Cell means. mean_age/mean_speed/blind_dist are the sanity channel: a cell with
    mean_age=0 held no stale map and a flat mean_speed means the speed axis did nothing."""
    tag, key, values = _sweep_axis(args)
    grid = {}
    for v in values:
        for spd in args.speeds:
            cell = [r for r in all_rows if r.get(key) == v and r["speed"] == spd]
            if not cell:
                continue
            def avg(k):
                vals = [r[k] for r in cell if r.get(k) is not None]
                return round(float(np.mean(vals)), 3) if vals else None
            grid[f"{tag}{v:g}_s{spd}"] = {"n": len(cell), "success": avg("success"),
                                          "collisions": avg("collisions"), "spl": avg("spl"),
                                          "mean_age": avg("mean_age"), "mean_speed": avg("mean_speed"),
                                          "blind_dist": avg("blind_dist")}
    return grid


def run_ablation(args):
    """Operating-point x flight-speed sweep: run C4-style episodes at each (res, max_speed)
    and report success/collisions. At higher speed and coarser cadence the held map is
    staler, so success should fall and collisions rise (ties C4 back to C1/C5)."""
    os.makedirs(args.out, exist_ok=True)
    path = f"{args.out}/ablation.jsonl"
    open(path, "w").close()
    carrot = max(args.step, 1.05 * max(args.speeds))
    tag, key, values = _sweep_axis(args)
    all_rows = []
    for v in values:
        for spd in args.speeds:
            a = argparse.Namespace(**vars(args))
            setattr(a, key, v)
            a.max_speed = spd
            # Speed is set ONLY by the PX4 velocity cap. The commanded setpoint distance is
            # held constant across cells (and far enough to saturate the fastest cell), because
            # a speed-scaled carrot would also vary how far the drone commits per tick --
            # a second mechanism confounding the staleness effect the sweep is measuring.
            a.carrot = carrot
            a.arms = ["student"]
            a.out = f"{args.out}/abl_{tag}{v:g}_s{spd}"
            print(f"### ABLATION {key}={v} speed={spd} res={a.res} ###", flush=True)
            all_rows.extend(_c4_rows(a, jsonl_path=path,
                                     extra={"res": a.res, "speed": spd, key: v}))
            json.dump(_ablation_grid(all_rows, args), open(f"{args.out}/ablation_summary.json", "w"), indent=2)
    grid = _ablation_grid(all_rows, args)
    print("ABLATION_SUMMARY", json.dumps(grid), flush=True)
    print(f"=== DRONE_ABLATION_DONE n={len(all_rows)} ===", flush=True)


def _obstacle_grid(all_rows, args):
    """Cell means per (n_obstacles, arm). `novel_hits` separates crashes into an injected
    obstacle from crashes into the scene's own furniture."""
    grid = {}
    for n_obs in args.n_obstacles:
        for arm in args.arms:
            cell = [r for r in all_rows if r.get("n_obstacles") == n_obs and r["arm"] == arm]
            if not cell:
                continue
            def avg(k):
                v = [r[k] for r in cell if r.get(k) is not None]
                return round(float(np.mean(v)), 3) if v else None
            novel = sum(1 for r in cell for e in (r.get("collision_events") or [])
                        if e.get("is_novel"))
            grid[f"n{n_obs}_{arm}"] = {
                "n": len(cell), "success": avg("success"), "spl": avg("spl"),
                "collisions": avg("collisions"),
                "crashes": sum(1 for r in cell if r.get("ended") == "collision"),
                "novel_hits": novel,
                "stalls": sum(1 for r in cell if (r.get("blocked_frac") or 0) > 0.5)}
    return grid


def run_obstacles(args):
    """C4 with route-unaware obstacles. The route is computed on a clean navmesh, then
    obstacles are placed, so avoidance requires the live map. Crash = failure."""
    os.makedirs(args.out, exist_ok=True)
    path = f"{args.out}/obstacles.jsonl"
    open(path, "w").close()
    all_rows = []
    for n_obs in args.n_obstacles:
        a = argparse.Namespace(**vars(args))
        a.n_obs = n_obs
        a.out = f"{args.out}/obs_{n_obs}"
        os.makedirs(a.out, exist_ok=True)
        print(f"### OBSTACLES n={n_obs} alt={args.alt} arms={args.arms} ###", flush=True)
        all_rows.extend(_c4_rows(a, jsonl_path=path, extra={"n_obstacles": n_obs}))
        json.dump(_obstacle_grid(all_rows, args),
                  open(f"{args.out}/obstacles_summary.json", "w"), indent=2)
    print("OBSTACLES_SUMMARY", json.dumps(_obstacle_grid(all_rows, args)), flush=True)
    print(f"=== DRONE_OBSTACLES_DONE n={len(all_rows)} ===", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="c4", choices=["c4", "c2", "ablation", "obstacles"])
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--ckpt", default=os.environ.get("MERLIN_CKPT", ""))
    ap.add_argument("--dataset", default=None, help="scene_dataset_config.json (e.g. ReplicaCAD)")
    ap.add_argument("--arms", nargs="+", default=["student", "gt"])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--res", type=int, default=378)
    ap.add_argument("--min_geo", type=float, default=2.0)
    ap.add_argument("--max_geo", type=float, default=10.0)
    ap.add_argument("--max_speed", type=float, default=1.0)
    ap.add_argument("--step", type=float, default=0.3, help="planner lookahead (m)")
    ap.add_argument("--alt", type=float, default=0.75, help="flight altitude (m)")
    ap.add_argument("--no_carve", action="store_true",
                    help="disable free-space carving in the voxel map (ablation): the map is "
                         "then hits-only, so a single bad depth reading occupies a voxel for "
                         "the rest of the episode")
    ap.add_argument("--no_escape", action="store_true",
                    help="disable the planner's obstacle-sliding escape mode (ablation): the "
                         "greedy planner then hovers whenever no direction improves the goal")
    ap.add_argument("--min_hits", type=int, default=2,
                    help="voxel occupancy threshold; raising it suppresses phantom near-depth "
                         "speckle, which is the measured cause of student-arm planner stalls")
    ap.add_argument("--n_obstacles", nargs="+", type=int, default=[0, 1, 2, 3],
                    help="obstacle-count sweep for --mode obstacles")
    ap.add_argument("--save_frames", type=int, default=8, help="replay frames per episode")
    ap.add_argument("--beauty", type=int, default=0, help="high-res replay pass (px), 0=off")
    ap.add_argument("--rgb_noise", type=float, default=0.0)
    ap.add_argument("--motion_blur", type=float, default=0.0)
    ap.add_argument("--pose_noise", type=float, default=0.0)
    ap.add_argument("--rot_noise_deg", type=float, default=0.0)
    # Clearance ordering that has to hold for the nominal route to be traversable:
    # robot_radius (0.18) < planner safety < navmesh agent_radius (0.35). With safety at 0.35
    # the planner considers its own route blocked inside doorways and spends the episode in
    # escape mode.
    # Clearance is measured to voxel CENTRES, so a surface can sit voxel*sqrt(3)/2 nearer
    # than the voxel that represents it. The planner's safety therefore has to cover
    # robot_radius (0.18) + that quantisation, and still stay under the navmesh clearance the
    # route was built with (0.35), which is what fixes voxel at 0.10.
    ap.add_argument("--safety", type=float, default=0.30)
    ap.add_argument("--stale_route", action="store_true",
                    help="plan the route on the bare apartment shell, as if from an "
                         "out-of-date floor plan: every piece of furniture then blocks a "
                         "route that does not know about it, and only the live map avoids it")
    ap.add_argument("--intrude", type=float, default=1.0,
                    help="how far an injected obstacle's footprint reaches past the route "
                         "centreline; at or above the footprint half-width (the default) the "
                         "obstacle is centred on the route, so flying the nominal route is a "
                         "guaranteed collision and avoidance must come from the live map")
    ap.add_argument("--route_clear", type=float, default=0.0,
                    help="require this much REAL clearance along the whole route and at both "
                         "endpoints (m). Keeps the drone out of the near field, where the "
                         "student's depth error is 0.64 rel vs 0.11 past 1 m")
    ap.add_argument("--block_margin", type=float, default=0.20,
                    help="slack in the 'obstacle blocks the route' test (m). <=0 demands the "
                         "route pass within the footprint + drone radius, so flying the "
                         "nominal route is a guaranteed hit -- needed in wide corridors")
    ap.add_argument("--obs_jitter", type=float, default=0.35,
                    help="lateral jitter of candidate obstacle sites off the route (m)")
    ap.add_argument("--obs_clear", type=float, default=0.0,
                    help="require this much free space BESIDE a placed obstacle (m), so the "
                         "detour around it does not run the drone along a wall")
    ap.add_argument("--seed0", type=int, default=1000,
                    help="first episode seed. Use a disjoint value when a configuration was "
                         "SELECTED on earlier seeds, so the reported rate is not the maximum "
                         "of the noise it was picked on")
    ap.add_argument("--obs_yaw", type=float, default=0.0,
                    help="yaw (deg) applied to injected obstacles. Open-frame furniture has a "
                         "solid face and a see-through face; this selects which one the route "
                         "meets, so it must be reported with any number it produces")
    ap.add_argument("--obs_include", nargs="+", default=[],
                    help="restrict injected obstacles to handles containing these substrings. "
                         "This SCOPES the benchmark: the excluded classes and their measured "
                         "rates must be reported alongside any number produced with it")
    ap.add_argument("--obs_width", type=float, default=0.0,
                    help="only place obstacles at least this wide (m). Bulky furniture makes "
                         "the student's ~0.27 m near-face displacement a small fraction of "
                         "the detour, without helping the no-map arm")
    ap.add_argument("--depth_calib", default=None,
                    help="fitted per-pixel apparent->real depth table (npz from "
                         "diag_depth.py --fit_calib)")
    ap.add_argument("--calib_q", type=float, default=0.20,
                    help="quantile of gt/pred the table returns (0.5 = median fit)")
    ap.add_argument("--calib", type=int, default=0,
                    help="depth-step decompression kernel in px "
                         "(perception.decompress_steps); 0 = off")
    ap.add_argument("--alpha", type=float, default=0.18,
                    help="measured fraction of an object's depth step the student loses")
    ap.add_argument("--cap", type=float, default=0.35,
                    help="max metres the decompression may pull a surface nearer")
    ap.add_argument("--hit_w", type=int, default=1,
                    help="occupancy weight of a ray that TERMINATES on a surface, vs carve_w "
                         "for one that passes through. >1 keeps thin structure (chair legs) "
                         "that a few through-rays would otherwise erase")
    ap.add_argument("--carve_w", type=int, default=1,
                    help="evidence removed from a voxel each frame whose rays pass through "
                         "it. Symmetric with the +1 a frame that sees a surface there adds: "
                         "at the old default of 4 a single frame of through-rays erased what "
                         "four confirming frames had built, so nothing ever reached "
                         "min_hits and injected obstacles were absent from the flight map")
    ap.add_argument("--max_range", type=float, default=8.0,
                    help="depth beyond this is not integrated. The distilled student is "
                         "accurate to ~0.2 m under 2 m but compresses 4-6 m surfaces to ~60%% "
                         "of true range, so a long range fills the map with phantom obstacles")
    ap.add_argument("--require_known", action="store_true",
                    help="ablation: refuse to fly into voxels no camera ray has swept")
    ap.add_argument("--nav_radius", type=float, default=0.35,
                    help="navmesh clearance the route is planned with; must exceed the "
                         "planner safety, which in turn must exceed the body radius")
    ap.add_argument("--voxel", type=float, default=0.10,
                    help="occupancy voxel size; smaller means less obstacle inflation in "
                         "narrow indoor corridors, at more memory")
    ap.add_argument("--goal_tol", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--window", type=int, default=8, help="C2 window length (frames)")
    ap.add_argument("--res_list", nargs="+", type=int, default=[252, 378, 518], help="ablation res sweep")
    ap.add_argument("--latencies", nargs="+", type=float, default=None,
                    help="ablation: sweep map-staleness latency (ms) at --res instead of resolution")
    ap.add_argument("--speeds", nargs="+", type=float, default=[0.5, 1.0, 2.0], help="ablation speed sweep")
    ap.add_argument("--out", default=WS + "/out/c4")
    args = ap.parse_args()
    if args.mode == "c4":
        run_c4(args)
    elif args.mode == "c2":
        run_c2(args)
    elif args.mode == "ablation":
        run_ablation(args)
    elif args.mode == "obstacles":
        run_obstacles(args)


if __name__ == "__main__":
    sys.exit(main())
