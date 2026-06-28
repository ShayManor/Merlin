#!/usr/bin/env python3
"""C4 closed-loop navigation evaluation harness (real-rover).

Turns recorded rover navigation EPISODES into the paper's C4 metrics: success rate, SPL
(Success weighted by Path Length, Anderson et al. 2018, arXiv:1807.06757), soft-SPL, collision
rate, and path efficiency. The rover only has to log each episode as a JSON in the format below;
this harness does the rest, so the closed-loop runs are turnkey once the hardware is ready.

Episode JSON schema (one file per nav goal, written by the companion / a rosbag post-processor):
{
  "scene": "lab_room_3",                 # unseen indoor scene id
  "start":  [x, y],                       # meters, map frame
  "goal":   [x, y],
  "shortest_path_m": 7.4,                 # geodesic dist on the (GT or onboard) free-space map;
                                          #   if absent, falls back to straight-line ||goal-start||
  "trajectory": [[t, x, y, yaw], ...],    # executed pose stream (any rate); t in seconds
  "collisions": [[t, x, y], ...],         # contact / bumper / costmap-violation events (may be [])
  "reached": true,                        # OPTIONAL explicit flag; else inferred from final dist
  "outcome": "goal" | "collision" | "timeout" | "stuck"   # OPTIONAL, for bookkeeping
}

Success = reached the goal within --success-radius AND no FATAL collision (a fatal collision ends
the episode; see --fatal-collisions). SPL weights success by path optimality. We report per-scene
and aggregate, matching C4 ("nav success >=80-90% in N>=6 unseen indoor scenes from the live
onboard map only").
"""
import argparse
import glob
import json
import os

import numpy as np


def _path_len(traj):
    """Total executed path length (m) from an [[t,x,y,yaw],...] stream."""
    if len(traj) < 2:
        return 0.0
    p = np.asarray(traj, float)[:, 1:3]
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def _duration(traj):
    return float(traj[-1][0] - traj[0][0]) if len(traj) >= 2 else 0.0


def episode_metrics(ep, success_radius=0.5, fatal_collisions=True):
    """Per-episode C4 metrics. Returns a dict; robust to missing optional fields."""
    traj = ep.get("trajectory", [])
    goal = np.asarray(ep["goal"], float)
    start = np.asarray(ep["start"], float)
    cols = ep.get("collisions", []) or []

    actual = _path_len(traj)
    shortest = float(ep.get("shortest_path_m") or np.linalg.norm(goal - start))
    final_xy = np.asarray(traj[-1][1:3], float) if traj else start
    final_dist = float(np.linalg.norm(final_xy - goal))

    reached = ep["reached"] if "reached" in ep else (final_dist <= success_radius)
    # a fatal collision invalidates success even if the goal pose was reached
    fatal = fatal_collisions and len(cols) > 0
    success = bool(reached and not fatal)

    # SPL = S * shortest / max(shortest, actual); guard the degenerate actual==0
    denom = max(shortest, actual, 1e-6)
    spl = (shortest / denom) if success else 0.0
    # soft-SPL (Datta et al. 2020): replace binary success by goal-progress, never penalizes
    # partial progress to zero -> better signal with few episodes
    progress = max(0.0, 1.0 - final_dist / max(np.linalg.norm(goal - start), 1e-6))
    soft_spl = progress * (shortest / denom)

    return {
        "scene": ep.get("scene", "?"),
        "success": success,
        "spl": spl,
        "soft_spl": soft_spl,
        "reached": bool(reached),
        "fatal_collision": fatal,
        "n_collisions": len(cols),
        "path_m": round(actual, 2),
        "shortest_m": round(shortest, 2),
        "path_efficiency": round(shortest / denom, 3),   # <=1; 1 = optimal
        "final_dist_m": round(final_dist, 2),
        "duration_s": round(_duration(traj), 1),
        "collisions_per_m": round(len(cols) / max(actual, 1e-6), 4),
    }


def aggregate(rows):
    """Aggregate over episodes + per-scene (matches the C4 reporting in the paper)."""
    if not rows:
        return {"n": 0}
    succ = np.array([r["success"] for r in rows], float)
    agg = {
        "n_episodes": len(rows),
        "n_scenes": len(set(r["scene"] for r in rows)),
        "success_rate": round(float(succ.mean()), 3),
        "spl": round(float(np.mean([r["spl"] for r in rows])), 3),
        "soft_spl": round(float(np.mean([r["soft_spl"] for r in rows])), 3),
        "collisions_per_m": round(float(np.mean([r["collisions_per_m"] for r in rows])), 4),
        "mean_path_efficiency": round(float(np.mean([r["path_efficiency"] for r in rows])), 3),
    }
    per_scene = {}
    for sc in sorted(set(r["scene"] for r in rows)):
        s = [r for r in rows if r["scene"] == sc]
        per_scene[sc] = {
            "n": len(s),
            "success_rate": round(float(np.mean([r["success"] for r in s])), 3),
            "spl": round(float(np.mean([r["spl"] for r in s])), 3),
        }
    agg["per_scene"] = per_scene
    return agg


def _self_test():
    """Synthetic episodes to verify the metric logic with no hardware (run with --self-test)."""
    eps = [
        # clean success, near-optimal path
        {"scene": "A", "start": [0, 0], "goal": [4, 0], "shortest_path_m": 4.0,
         "trajectory": [[0, 0, 0, 0], [1, 2, 0.2, 0], [2, 4, 0.0, 0]], "collisions": []},
        # reached goal but collided -> fatal -> not success
        {"scene": "A", "start": [0, 0], "goal": [4, 0], "shortest_path_m": 4.0,
         "trajectory": [[0, 0, 0, 0], [2, 4, 0, 0]], "collisions": [[1.0, 2, 0]]},
        # timeout, stalled halfway -> partial soft-SPL
        {"scene": "B", "start": [0, 0], "goal": [6, 0], "shortest_path_m": 6.0,
         "trajectory": [[0, 0, 0, 0], [3, 3, 0, 0]], "collisions": []},
    ]
    rows = [episode_metrics(e) for e in eps]
    assert rows[0]["success"] and rows[0]["spl"] > 0.95, rows[0]
    assert (not rows[1]["success"]) and rows[1]["spl"] == 0.0 and rows[1]["fatal_collision"], rows[1]
    assert (not rows[2]["success"]) and 0.0 < rows[2]["soft_spl"] < 0.6, rows[2]
    agg = aggregate(rows)
    assert agg["n_episodes"] == 3 and agg["n_scenes"] == 2
    assert abs(agg["success_rate"] - 1/3) < 0.01, agg   # 1/3, allowing for rounding
    print("[c4-eval] self-test PASSED")
    print(json.dumps(agg, indent=2))


def main():
    ap = argparse.ArgumentParser(description="C4 closed-loop nav metrics from rover episode logs.")
    ap.add_argument("--episodes", default="", help="glob of episode JSON files, e.g. runs/*.json")
    ap.add_argument("--success-radius", type=float, default=0.5, help="m to count goal reached")
    ap.add_argument("--no-fatal-collisions", action="store_true",
                    help="count success even if a collision occurred (default: collision is fatal)")
    ap.add_argument("--out", default="", help="write the aggregate JSON here")
    ap.add_argument("--self-test", action="store_true", help="run the synthetic logic test and exit")
    args = ap.parse_args()

    if args.self_test or not args.episodes:
        _self_test()
        return

    files = sorted(glob.glob(args.episodes))
    if not files:
        print(f"[c4-eval] no episode files matched: {args.episodes}")
        return
    rows = [episode_metrics(json.load(open(f)), args.success_radius, not args.no_fatal_collisions)
            for f in files]
    agg = aggregate(rows)
    for r in rows:
        print(f"  {r['scene']:<14} success={int(r['success'])} spl={r['spl']:.2f} "
              f"coll={r['n_collisions']} eff={r['path_efficiency']:.2f} dist={r['final_dist_m']}m")
    print("\n=== C4 aggregate ===")
    print(json.dumps(agg, indent=2))
    if args.out:
        json.dump({"episodes": rows, "aggregate": agg}, open(args.out, "w"), indent=2)
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
