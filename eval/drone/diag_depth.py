#!/usr/bin/env python3
"""Offline depth/map diagnostic for the C4 loop: why does the student's map block a route
the GT map flies cleanly?

Flies the geodesic route open-loop (poses come from the route, not from PX4), renders each
pose, and builds two voxel maps side by side -- one from GT depth, one from student depth.
Reports, per configuration:
  scale   spread of the per-frame median GT/pred ratio (a drifting scale smears surfaces
          across the map even when every single frame looks accurate)
  absrel  depth error at the episode-constant scale the student arm actually flies with
  vox     occupied voxel count of each map (the student's was measured 3-10x the GT map's
          in the closed-loop runs, which is what blocks the planner)
  block   fraction of route points where the planner would refuse to move: the route is
          navmesh-derived and collision-free, so every blocked point is a phantom
  miss     fraction of GT-map voxels with no student voxel within one voxel (holes: what
          the drone flies into)
No PX4 needed, so this runs anywhere the renderer + student do.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

from eval.drone.frames import HAB_CAM_TO_OPTICAL, EpisodeFrame
from eval.drone.mapper import VoxelMap
from eval.drone.perception import Perception, sharpen_edges, decompress_steps
from eval.drone.run_experiments import WS, densify, spawn_renderer
from eval.drone.sim_client import SimClient

PORT = 5601 + int(os.environ.get("MERLIN_PX4_INSTANCE", 0))


def look_along(fwd):
    """Habitat camera matrix (columns = right/up/back in world) looking along horizontal
    `fwd` with +Y up -- the pose the drone holds while it steers at a waypoint."""
    f = np.asarray(fwd, float).copy()
    f[1] = 0.0
    n = np.linalg.norm(f)
    f = np.array([0.0, 0.0, -1.0]) if n < 1e-9 else f / n
    right = np.array([-f[2], 0.0, f[0]])
    return np.stack([right, np.array([0.0, 1.0, 0.0]), -f], 1)


def spawn(port, res, hfov):
    sq = (res // 14) * 14
    logf = open(f"{WS}/rend_{port}.log", "w")
    return subprocess.Popen([WS + "/envs/habenv/bin/python", "eval/drone/renderer.py",
                             "--port", str(port), "--width", str(sq), "--height", str(sq),
                             "--hfov", str(hfov)], stdout=logf, stderr=subprocess.STDOUT)


def route_poses(wps_hab, spacing=0.25):
    """Dense (pos, R) samples along the route: what a perfect flight would have seen."""
    pts = []
    for a, b in zip(wps_hab[:-1], wps_hab[1:]):
        a = np.asarray(a, float); b = np.asarray(b, float)
        d = np.linalg.norm(b - a)
        n = max(1, int(np.ceil(d / spacing)))
        for i in range(n):
            p = a + (b - a) * (i / n)
            pts.append((p, look_along(b - a)))
    pts.append((np.asarray(wps_hab[-1], float), pts[-1][1] if pts else np.eye(3)))
    return pts


BINS = [(0.2, 0.5), (0.5, 0.8), (0.8, 1.2), (1.2, 1.8), (1.8, 2.5)]


def chamfer_miss(a, b, tol):
    """fraction of points in `a` with no point of `b` within tol (a, b: (N,3) voxel centres)"""
    if len(a) == 0:
        return 0.0
    if len(b) == 0:
        return 1.0
    k = np.round(b / tol).astype(np.int64)
    seen = set(map(tuple, k))
    off = np.array([(i, j, l) for i in (-1, 0, 1) for j in (-1, 0, 1) for l in (-1, 0, 1)])
    ka = np.round(a / tol).astype(np.int64)
    hit = np.array([any(tuple(x) in seen for x in (kk + off)) for kk in ka])
    return float(1.0 - hit.mean())


def _detect(o, p, mg, ms, mhs, voxel):
    """Record how far away the drone was the first time each map put a surface on the
    obstacle, at each occupancy threshold. This is the number that decides a closed-loop
    episode: an obstacle first mapped at 0.4 m cannot be steered around at 0.6 m/s no matter
    how good the planner is. Raising min_hits cleans the map but delays this -- the trade the
    threshold is actually choosing."""
    c = np.asarray(o["pos"], float)[None, :]
    rad = 0.5 * max(o["bbox"]) + voxel
    d = round(float(np.linalg.norm(np.asarray(p, float) - c[0])), 2)
    for tag, m in (("det_gt", mg), ("det_st", ms)):
        for mh in mhs:
            k = f"{tag}{mh}"
            if o.get(k) is None:
                keep, m.min_hits, m._codes_cache = m.min_hits, mh, None
                if m.any_occupied_within(c, rad)[0]:
                    o[k] = d
                m.min_hits, m._codes_cache = keep, None


def visibility(client, perc, args):
    """Does the obstacle appear in the student's DEPTH at all?

    Flies each route twice: once on the clean scene to record background depth, once with the
    obstacle injected. Pixels where GT depth jumps closer are exactly the obstacle's silhouette.
    Inside that silhouette, `recov` = (background - predicted) / (background - true) is 1.0 if
    the student resolves the object off the wall behind it and 0.0 if it renders straight
    through it. A map-side threshold cannot recover an object that was never in the depth, so
    this separates "the mapper discarded it" from "the model never saw it".
    """
    rows = []
    for scene in args.scenes:
        client.reset(scene, args.dataset, args.nav_radius)
        for k in range(args.episodes):
            epi = client.sample(scene, args.min_geo, args.max_geo, seed=1000 + k,
                                alt=args.alt, dataset=args.dataset,
                                route_clear=args.route_clear)
            if epi is None:
                continue
            wps = epi.get("waypoints_hab") or [epi["start_hab"], epi["goal_hab"]]
            poses = route_poses(wps, args.spacing)
            client.clear_obstacles()
            bg = [client.render(p, R)["depth"] for p, R in poses]
            # Placement must match the closed-loop config, otherwise a given seed injects a
            # different object here than the one that actually flew and the per-asset
            # comparison is meaningless.
            obs = client.place_obstacles(wps, args.n_obs, seed=1000 + k, alt=args.alt,
                                         intrude=args.intrude, obs_clear=args.obs_clear,
                                         min_width=0.35, block_margin=-0.10,
                                         jitter=0.10).get("placed", [])
            if not obs:
                continue
            ep_scale, per = None, []
            for (p, R), d0 in zip(poses, bg):
                out = client.render(p, R)
                gt, pred = out["depth"], perc._raw_depth(out["rgb"], args.res)
                v = np.isfinite(gt) & (gt > 0.1) & (gt < 10) & (pred > 0.05)
                if ep_scale is None and v.sum() > 200:
                    ep_scale = float(np.median(gt[v] / pred[v]))
                m = (d0 - gt > 0.15) & (gt > 0.1) & (gt < args.max_range)
                if m.sum() < 100 or ep_scale is None:
                    continue
                bgd, tru = float(np.median(d0[m])), float(np.median(gt[m]))
                if bgd - tru < 0.2:
                    continue
                # Is the student SCALE-ambiguous (one multiplier) or AFFINE-ambiguous
                # (multiplier + offset)? Recovering only the scale, as the deployed arm does,
                # fits mostly walls -- so a leftover offset leaves near objects displaced
                # while flat surfaces look correct, which is exactly the non-uniform error
                # measured closed-loop. Fit both on this frame and compare.
                A = np.stack([pred[v], np.ones(int(v.sum()))], 1)
                ab = np.linalg.lstsq(A, gt[v], rcond=None)[0]
                rec = {"scale": round((bgd - float(np.median(pred[m] * ep_scale)))
                                      / (bgd - tru), 3),
                       "affine": round((bgd - float(np.median(pred[m] * ab[0] + ab[1])))
                                       / (bgd - tru), 3)}
                for kk in args.calib_sweep:
                    prd = float(np.median(decompress_steps(pred * ep_scale, kk,
                                                          args.alpha, args.cap)[m]))
                    rec[str(kk)] = round((bgd - prd) / (bgd - tru), 3)
                per.append({"px": int(m.sum()), "range": round(tru, 2),
                            "bg": round(bgd, 2), "gt": round(tru, 2),
                            "a": round(float(ab[0]), 4), "b": round(float(ab[1]), 4),
                            "s": round(ep_scale, 4), "recov": rec})
            if per:
                rows.append({"scene": os.path.basename(scene), "seed": 1000 + k,
                             "handle": os.path.basename(obs[0]["handle"]).split(".")[0],
                             "bbox": [round(float(x), 3) for x in obs[0]["bbox"]],
                             "n": len(per), "frames": per})
                print(f"  {os.path.basename(scene)} s{1000+k} frames={len(per)} "
                      + " ".join(f"k{kk}={np.median([f['recov'][str(kk)] for f in per]):.2f}"
                                 for kk in args.calib_sweep), flush=True)
    fr = [f for r in rows for f in r["frames"]]
    if fr:
        print(f"\nobstacle-silhouette depth-step recovery ({len(fr)} frames); 1.00 = the "
              f"object's near face is placed correctly")
        print(f"{'calib k':>8s} {'recov (all)':>12s} {'recov <1.5 m':>13s}")
        for kk in ["scale", "affine"] + [str(x) for x in args.calib_sweep]:
            a = [f["recov"][kk] for f in fr]
            n = [f["recov"][kk] for f in fr if f["range"] < 1.5]
            print(f"{kk:>8s} {np.median(a):12.2f} {np.median(n):13.2f}")
        bs = [f["b"] for f in fr]; as_ = [f["a"] for f in fr]; ss = [f["s"] for f in fr]
        print(f"\nper-frame affine fit gt ~ a*pred + b:  a median {np.median(as_):.3f} "
              f"(scale-only fit gives {np.median(ss):.3f}),  b median {np.median(bs):+.3f} m "
              f"[{np.percentile(bs,25):+.3f}, {np.percentile(bs,75):+.3f}]")
    return rows


def fit_calib(client, perc, args):
    """Sample student depth against GT depth along the route and accumulate the per-pixel
    apparent->real calibration table (eval/drone/depth_calib).

    Flown exactly as the closed-loop student arm is: the same route poses, the same injected
    obstacles, and the same episode-constant scale from the first frame's median GT/pred
    ratio. Obstacles are injected during fitting even though nothing here plans around them,
    because the near-field object pixels are the only ones that decide an episode and a clean
    route never produces them.

    Seeds come from --seed0 (3000), disjoint from the evaluation seeds (1000+), so the table
    is never fit on the episodes it is later scored on.
    """
    from eval.drone.depth_calib import DepthCalib
    tab = DepthCalib()
    n = 0
    for scene in args.scenes:
        client.reset(scene, args.dataset, args.nav_radius)
        for k in range(args.episodes):
            epi = client.sample(scene, args.min_geo, args.max_geo, seed=args.seed0 + k,
                                alt=args.alt, dataset=args.dataset,
                                route_clear=args.route_clear)
            if epi is None:
                continue
            wps = epi.get("waypoints_hab") or [epi["start_hab"], epi["goal_hab"]]
            client.clear_obstacles()
            if args.n_obs:
                client.place_obstacles(wps, args.n_obs, seed=args.seed0 + k, alt=args.alt,
                                       intrude=args.intrude, obs_clear=args.obs_clear,
                                       min_width=0.35, block_margin=-0.10, jitter=0.10)
            ep_scale = None
            for p, R in route_poses(wps, args.spacing):
                out = client.render(p, R)
                gt, pred = out["depth"], perc._raw_depth(out["rgb"], args.res)
                v = np.isfinite(gt) & (gt > 0.1) & (gt < 10) & (pred > 0.05)
                if ep_scale is None and v.sum() > 200:
                    ep_scale = float(np.median(gt[v] / pred[v]))
                if ep_scale is None:
                    continue
                n += tab.accumulate(pred * ep_scale, gt)
            print(f"  {os.path.basename(scene)} s{args.seed0+k}: {n/1e6:.1f}M samples",
                  flush=True)
    out_path = args.calib_out or (args.out or "calib") + ".npz"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tab.save(out_path)
    print(f"\nwrote {out_path}  ({n/1e6:.1f}M pixel samples)")
    for q in args.q_sweep:
        print(f"\nq={q}: predicted range -> ratio applied (pooled), and how much the "
              f"correction varies ACROSS pixel blocks")
        print(f"{'pred range (m)':>14s} {'n':>10s} {'ratio':>7s} {'block IQR':>10s}")
        for lo, hi, cnt, r, iqr in tab.spread(q):
            if cnt > 0 and lo < 4.0:
                print(f"  {lo:5.2f}-{hi:5.2f} {cnt:10d} {r:7.3f} {iqr:10.3f}")
    return []


def true_clearance(cl, p, lo=0.05, hi=1.60, tol=0.03):
    """Radius of the largest sphere that fits at p without touching real geometry.

    Bisection on the renderer's own collision test, so it is the same notion of clearance the
    episode is scored against -- not a navmesh proxy.
    """
    if cl.collide([float(x) for x in p], lo)["hit"]:
        return lo
    if not cl.collide([float(x) for x in p], hi)["hit"]:
        return hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if cl.collide([float(x) for x in p], mid)["hit"]:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def clearance_calib(client, perc, args):
    """Map-reported clearance vs TRUE clearance, at points along the flown route.

    This is the quantity the planner actually queries: it refuses a step when a sample lies
    within `safety` of an occupied voxel, i.e. when nearest_obstacle_dist < safety. If that
    reported distance is systematically larger than the truth, the planner's block condition
    fires too late -- which is exactly the measured failure (11 of 13 losses had
    blocked_frac 0.00 and hit an obstacle the map held 0.49 m away).

    Unlike a per-pixel depth calibration, this aggregates hundreds of voxels over many frames,
    so it is not obviously defeated by the regression dilution that killed the pixel version.
    """
    rows = []
    for scene in args.scenes:
        client.reset(scene, args.dataset, args.nav_radius)
        for k in range(args.episodes):
            epi = client.sample(scene, args.min_geo, args.max_geo, seed=1000 + k,
                                alt=args.alt, dataset=args.dataset,
                                route_clear=args.route_clear)
            if epi is None:
                continue
            wps = epi.get("waypoints_hab") or [epi["start_hab"], epi["goal_hab"]]
            client.clear_obstacles()
            obs = client.place_obstacles(wps, max(1, args.n_obs), seed=1000 + k, alt=args.alt,
                                         intrude=0.9, obs_clear=args.obs_clear,
                                         min_width=0.35, block_margin=-0.10, jitter=0.10)
            if not obs.get("placed"):
                continue
            poses = route_poses(wps, args.spacing)
            ms = VoxelMap(voxel=args.voxel, min_hits=args.min_hits, carve_w=1, cap=8)
            mg = VoxelMap(voxel=args.voxel, min_hits=args.min_hits, carve_w=1, cap=8)
            ep_scale = None
            for p, R in poses:
                out = client.render(p, R)
                gt, pred = out["depth"], perc._raw_depth(out["rgb"], args.res)
                v = np.isfinite(gt) & (gt > 0.1) & (gt < 10) & (pred > 0.05)
                if ep_scale is None and v.sum() > 200:
                    ep_scale = float(np.median(gt[v] / pred[v]))
                if ep_scale is None:
                    continue
                Ropt = R @ HAB_CAM_TO_OPTICAL
                ms.integrate(perc.apply_depth_calib(pred * ep_scale), out["K"], Ropt, p,
                             max_range=args.max_range)
                mg.integrate(gt, out["K"], Ropt, p, max_range=args.max_range)
            # The route itself never enters the near field (route_clear keeps it out), which
            # is exactly the regime collisions happen in. Probe a grid AROUND the obstacle so
            # the calibration covers the distances that decide an episode, and record whether
            # the nearest mapped voxel belongs to the obstacle or to the room -- if the two
            # obey different laws, apparent->real is not a function of apparent alone.
            boxes = [(o["pos"], o["bbox"]) for o in obs["placed"]]
            occ = ms.occupied_points()
            for c, bb in boxes:
                c = np.asarray(c, float)
                for dx in np.arange(-1.4, 1.45, 0.2):
                    for dz in np.arange(-1.4, 1.45, 0.2):
                        q = np.array([c[0] + dx, float(poses[0][0][1]), c[2] + dz])
                        md = float(ms.nearest_obstacle_dist(q, 2.0))
                        if md >= 2.0:
                            continue
                        near_obs = False
                        if occ.shape[0]:
                            j = int(np.argmin(np.linalg.norm(occ - q, axis=1)))
                            h = 0.5 * np.asarray(bb, float) + ms.voxel + 0.10
                            near_obs = bool((np.abs(occ[j] - c) <= h).all())
                        rows.append({"scene": os.path.basename(scene), "seed": 1000 + k,
                                     "map_st": round(md, 3),
                                     "map_gt": round(float(mg.nearest_obstacle_dist(q, 2.0)), 3),
                                     "true": round(float(true_clearance(client, q)), 3),
                                     "near_obs": near_obs})
            print(f"  {os.path.basename(scene)} s{1000+k}: {len(rows)} probes", flush=True)
    if rows:
        import numpy as _np
        BINS2 = [(0.0, 0.35), (0.35, 0.5), (0.5, 0.7), (0.7, 1.0), (1.0, 1.4), (1.4, 2.1)]
        print("\nMAP-REPORTED clearance -> TRUE clearance (m), split by what the nearest "
              "mapped voxel is")
        print(f"{'map says':>12s} {'n(obst)':>8s} {'true|obst':>10s} "
              f"{'n(room)':>8s} {'true|room':>10s} {'gap':>7s}")
        for lo, hi in [(0.0, 0.3), (0.3, 0.45), (0.45, 0.6), (0.6, 0.8), (0.8, 1.1),
                       (1.1, 1.5)]:
            b = [r for r in rows if lo <= r["map_st"] < hi]
            o = [r for r in b if r.get("near_obs")]
            w = [r for r in b if not r.get("near_obs")]
            if len(b) < 15:
                continue
            to = _np.median([r["true"] for r in o]) if len(o) >= 8 else float("nan")
            tw = _np.median([r["true"] for r in w]) if len(w) >= 8 else float("nan")
            print(f"{lo:5.2f}-{hi:4.2f} {len(o):8d} {to:10.3f} {len(w):8d} {tw:10.3f} "
                  f"{(to - tw):+7.3f}")
    return rows


def run_cfg(client, perc, args, hfov):
    rows = []
    for scene in args.scenes:
        client.reset(scene, args.dataset, args.nav_radius)
        for k in range(args.episodes):
            # route_clear must reach here too: without it this samples wall-hugging routes
            # while the closed-loop experiment flies 0.6 m-clearance ones, and the offline
            # gate then describes a different near-field distribution than the loop it gates.
            epi = client.sample(scene, args.min_geo, args.max_geo, seed=1000 + k,
                                alt=args.alt, dataset=args.dataset,
                                route_clear=args.route_clear)
            if epi is None:
                continue
            wps = epi.get("waypoints_hab") or [epi["start_hab"], epi["goal_hab"]]
            poses = route_poses(wps, args.spacing)
            # Same novel obstacle the closed-loop experiment injects, placed after the route
            # was computed. The question this answers is the one the route statistics cannot:
            # does the student map the obstacle EARLY ENOUGH to steer around it?
            obs = []
            if args.n_obs:
                client.clear_obstacles()
                obs = client.place_obstacles(wps, args.n_obs, seed=1000 + k, alt=args.alt,
                                             intrude=args.intrude, obs_clear=args.obs_clear,
                                             min_width=0.35, block_margin=-0.10, jitter=0.10,
                                             include=args.obs_include or None).get("placed", [])
            mg = VoxelMap(voxel=args.voxel, carve_w=1, hit_w=1, cap=8)
            # (hit_w, carve_w): how strongly a terminating ray counts against one that seems
            # to pass through. All variants are fed the identical depth in the same pass.
            mss = {w: VoxelMap(voxel=args.voxel, hit_w=w[0], carve_w=w[1], cap=8)
                   for w in [tuple(map(int, s.split(":"))) for s in args.weights]}
            scales, absrel, ep_scale, bins = [], [], None, {}
            for i, (p, R) in enumerate(poses):
                out = client.render(p, R)
                gt = out["depth"]
                pred = perc._raw_depth(out["rgb"], args.res)
                v = np.isfinite(gt) & (gt > 0.1) & (gt < 10) & (pred > 0.05)
                if v.sum() > 200:
                    s = float(np.median(gt[v] / pred[v]))
                    scales.append(s)
                    if ep_scale is None:
                        ep_scale = s
                    absrel.append(float(np.median(np.abs(pred[v] * ep_scale - gt[v]) / gt[v])))
                    # Error vs RANGE. Collisions happen pressed against a surface, not
                    # mid-corridor, so a model that is accurate at 2 m and wrong at 0.4 m
                    # fails exactly where avoidance needs it -- and a whole-frame absrel,
                    # dominated by mid-range pixels, cannot show that.
                    err = np.abs(pred * ep_scale - gt)
                    # Binned by PREDICTED depth, not GT: at run time the predicted value is
                    # all a correction curve can key on.
                    for lo, hi in BINS:
                        b = v & (pred * ep_scale >= lo) & (pred * ep_scale < hi)
                        if b.sum() > 100:
                            bins.setdefault((lo, hi), []).append(
                                (float(np.median(err[b])),
                                 float(np.median(err[b] / gt[b])),
                                 float(np.median((pred * ep_scale - gt)[b]))))
                Ropt = R @ HAB_CAM_TO_OPTICAL
                mg.integrate(gt, out["K"], Ropt, p, max_range=args.max_range)
                # The depth-error columns above describe the RAW model; the map columns below
                # describe what the planner sees, so --depth_calib enters here only. A low-q
                # table is deliberately not minimum-error, so absrel is expected to rise even
                # when block_st/miss_st improve.
                for ms in mss.values():
                    ms.integrate(decompress_steps(perc.apply_depth_calib(
                                                      pred * (ep_scale or 1.0)),
                                                  args.calib, args.alpha, args.cap),
                                 out["K"], Ropt, p, max_range=args.max_range)
                for o in obs:
                    _detect(o, p, mg, mss[tuple(map(int, args.detect_w.split(":")))], args.min_hits_sweep, args.voxel)
            rp = np.array([p for p, _ in poses])
            # min_hits is only a threshold on the same evidence counts, so every candidate
            # is evaluated on one pass of rendering + inference rather than one run each.
            for w, ms in mss.items():
              for mh in args.min_hits_sweep:
                ms.min_hits = mg.min_hits = mh
                ms._codes_cache = mg._codes_cache = None
                pg = mg.occupied_points()
                ps = ms.occupied_points()
                rows.append({
                    "hfov": hfov, "min_hits": mh, "w": f"{w[0]}:{w[1]}",
                    "scene": os.path.basename(scene), "seed": 1000 + k,
                    "geo": round(float(epi["geo"]), 2), "n_frames": len(poses),
                    "scale_med": round(float(np.median(scales)), 3) if scales else None,
                    "scale_iqr": round(float(np.subtract(*np.percentile(scales, [75, 25]))), 3)
                                 if scales else None,
                    "absrel": round(float(np.median(absrel)), 3) if absrel else None,
                    "vox_gt": len(pg), "vox_st": len(ps),
                    "block_gt": round(float(mg.any_occupied_within(rp, args.safety).mean()), 3),
                    "block_st": round(float(ms.any_occupied_within(rp, args.safety).mean()), 3),
                    "miss_st": round(chamfer_miss(pg[::7], ps, args.voxel * 1.5), 3),
                    # How far the nearest mapped surface sits from the flight path in each
                    # map. A student value well above the GT one means the map RECEDES from
                    # real geometry, which is the closed-loop signature of flying into an
                    # obstacle the map still shows half a metre away.
                    "clear_gt": round(float(np.mean([mg.nearest_obstacle_dist(p) for p in rp])), 3),
                    "clear_st": round(float(np.mean([ms.nearest_obstacle_dist(p) for p in rp])), 3),
                    "bins": {f"{lo}-{hi}": [round(float(np.median([x[i] for x in v])), 3)
                                            for i in (0, 1, 2)]
                             for (lo, hi), v in bins.items()},
                    "det_gt": [o.get(f"det_gt{mh}") for o in obs],
                    "det_st": [o.get(f"det_st{mh}") for o in obs],
                })
                print("  " + json.dumps(rows[-1]), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--res", type=int, default=378)
    ap.add_argument("--hfovs", nargs="+", type=float, default=[90.0])
    ap.add_argument("--alt", type=float, default=0.75)
    ap.add_argument("--voxel", type=float, default=0.10)
    ap.add_argument("--min_hits", type=int, default=1)
    ap.add_argument("--min_hits_sweep", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--weights", nargs="+", default=["1:1", "2:1", "3:1", "1:0"],
                    help="hit_w:carve_w occupancy evidence weights to compare")
    ap.add_argument("--n_obs", type=int, default=0)
    ap.add_argument("--obs_include", nargs="+", default=[],
                    help="restrict injected obstacles to handles containing these substrings")
    ap.add_argument("--calib", type=int, default=0,
                    help="edge-sharpening calibration kernel (px); 0 = off")
    ap.add_argument("--calib_sweep", nargs="+", type=int, default=[0, 21, 41, 61, 81])
    ap.add_argument("--alpha", type=float, default=0.18)
    ap.add_argument("--cap", type=float, default=0.35)
    ap.add_argument("--clearance_calib", action="store_true",
                    help="measure map-reported vs true clearance (the planner's own query)")
    ap.add_argument("--fit_calib", action="store_true",
                    help="fit the per-pixel apparent->real depth table and write --calib_out")
    ap.add_argument("--calib_out", default=None, help="npz path for --fit_calib")
    ap.add_argument("--seed0", type=int, default=3000,
                    help="first episode seed; kept off the evaluation seeds (1000+)")
    ap.add_argument("--q_sweep", nargs="+", type=float, default=[0.10, 0.20, 0.35, 0.50],
                    help="quantiles to report after fitting")
    ap.add_argument("--depth_calib", default=None,
                    help="apply a fitted table (npz) to the student depth")
    ap.add_argument("--calib_q", type=float, default=0.20,
                    help="quantile of gt/pred the table returns (0.5 = median fit)")
    ap.add_argument("--route_clear", type=float, default=0.0)
    ap.add_argument("--obs_clear", type=float, default=0.0)
    ap.add_argument("--visibility", action="store_true",
                    help="measure whether the obstacle appears in the student depth")
    ap.add_argument("--intrude", type=float, default=0.25)
    ap.add_argument("--detect_w", default="1:1", help="weights used for the detection-range stat")
    ap.add_argument("--detect_mh", type=int, default=2)
    ap.add_argument("--carve_w", type=int, default=4)
    ap.add_argument("--safety", type=float, default=0.30)
    ap.add_argument("--max_range", type=float, default=8.0)
    ap.add_argument("--spacing", type=float, default=0.25)
    ap.add_argument("--min_geo", type=float, default=4.0)
    ap.add_argument("--max_geo", type=float, default=12.0)
    ap.add_argument("--nav_radius", type=float, default=0.35)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    perc = Perception(args.ckpt)
    if args.depth_calib:
        perc.set_depth_calib(args.depth_calib, args.calib_q)
    allrows = []
    for hfov in args.hfovs:
        print(f"### hfov={hfov}", flush=True)
        proc = spawn(PORT, args.res, hfov)
        try:
            time.sleep(12)
            client = SimClient(PORT)
            allrows += (fit_calib(client, perc, args) if args.fit_calib
                        else clearance_calib(client, perc, args) if args.clearance_calib
                        else visibility(client, perc, args) if args.visibility
                        else run_cfg(client, perc, args, hfov))
        finally:
            proc.terminate()
            proc.wait(timeout=20)
            time.sleep(2)

    if args.fit_calib:
        return
    if args.visibility or args.clearance_calib:
        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w") as f:
                for r in allrows:
                    f.write(json.dumps(r) + "\n")
        return
    bb = {}
    for r in allrows:
        for k, v in r.get("bins", {}).items():
            bb.setdefault(k, []).append(v)
    if bb:
        print("\nDEPTH ERROR vs PREDICTED RANGE (episode-constant scale, as flown).")
        print("subtract the signed bias from a prediction in that bin to calibrate it.")
        print(f"{'pred range (m)':>14s} {'abs err (m)':>11s} {'rel err':>8s} {'signed bias (m)':>15s}")
        for k in [f"{lo}-{hi}" for lo, hi in BINS]:
            if k in bb:
                a = np.array(bb[k])
                print(f"{k:>14s} {np.median(a[:, 0]):11.3f} {np.median(a[:, 1]):8.3f} "
                      f"{np.median(a[:, 2]):+15.3f}")
    print("\n w    mh  n   absrel vox_gt vox_st block_gt block_st  miss clr_gt clr_st")
    for w in args.weights:
        for mh in args.min_hits_sweep:
            rs = [r for r in allrows if r["w"] == w and r["min_hits"] == mh]
            if not rs:
                continue
            def m(k):
                v = [r[k] for r in rs if r[k] is not None]
                return float(np.mean(v)) if v else float("nan")
            print(f"{w:>4s} {mh:3d} {len(rs):3d} "
                  f"{m('absrel'):6.3f} {m('vox_gt'):6.0f} {m('vox_st'):6.0f} "
                  f"{m('block_gt'):8.3f} {m('block_st'):8.3f} {m('miss_st'):5.3f} "
                  f"{m('clear_gt'):6.3f} {m('clear_st'):6.3f}")
    for mh in args.min_hits_sweep:
        rs = [r for r in allrows if r["min_hits"] == mh and r["w"] == args.detect_w]
        dg = [d for r in rs for d in r["det_gt"]]
        ds = [d for r in rs for d in r["det_st"]]
        if dg:
            print(f"mh={mh} obstacles={len(dg)}  first mapped at (m): "
                  f"gt {np.mean([d for d in dg if d] or [0]):.2f} "
                  f"(missed {sum(d is None for d in dg)})  "
                  f"student {np.mean([d for d in ds if d] or [0]):.2f} "
                  f"(missed {sum(d is None for d in ds)})")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            for r in allrows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
