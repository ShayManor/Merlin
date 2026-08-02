#!/usr/bin/env python3
"""Frame-convention validation (the classic silent-failure axis).

Moving the drone forward in PX4 NED (+N) must move the Habitat camera in the direction it
faces, so the central GT depth DECREASES ~1:1 as it approaches a wall. If the frames.py
NED<->Habitat composition is mirrored, depth would increase instead and the map would be
silently wrong. Renderer-only (GT depth, pure geometry): no model, no PX4.

Samples a navigable interior pose (the scene origin is usually empty space), scans headings
to find one facing a wall at usable range, then steps forward along that heading.

Run on the cluster (needs habenv renderer + assets):
  PYTHONPATH=. $WS/envs/habenv/bin/python eval/drone/tests/test_frame_render_consistency.py
"""
import os
import subprocess
import sys
import time

import numpy as np

WS = "/scratch/gautschi/manors/merlin_drone"
SCENE = WS + "/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"


def _central_median_depth(depth, frac=0.12):
    h, w = depth.shape
    ch, cw = int(h * frac), int(w * frac)
    patch = depth[h // 2 - ch:h // 2 + ch, w // 2 - cw:w // 2 + cw]
    v = patch[np.isfinite(patch) & (patch > 0.05) & (patch < 50.0)]
    return float(np.median(v)) if v.size else float("nan")


def run_check(port=5624):
    sys.path.insert(0, os.getcwd())
    from eval.drone.sim_client import SimClient
    from eval.drone.frames import EpisodeFrame
    proc = subprocess.Popen([WS + "/envs/habenv/bin/python", "eval/drone/renderer.py",
                             "--port", str(port), "--width", "504", "--height", "504"])
    try:
        time.sleep(9)
        c = SimClient(port)
        c.reset(SCENE, None)
        ep0 = c.sample(SCENE, min_geo=2.0, max_geo=12.0, seed=0)
        if ep0 is None:
            print("FRAME_CHECK SKIP (no navigable episode sampled)", flush=True)
            return False
        start = ep0["start_hab"]
        # scan headings; pick the one facing the nearest wall at usable range (1-8 m)
        best = None
        for yaw in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            ep = EpisodeFrame(start, float(yaw))
            out = c.render(ep.ned_to_hab_pos([0, 0, 0]), ep.ned_quat_to_hab_R(np.array([0, 0, 0, 1.0])))
            d = _central_median_depth(out["depth"])
            if np.isfinite(d) and 1.0 < d < 8.0 and (best is None or d < best[1]):
                best = (yaw, d)
        if best is None:
            print("FRAME_CHECK SKIP (no heading faces a wall in 1-8 m)", flush=True)
            return False
        yaw, d0 = best
        ep = EpisodeFrame(start, float(yaw))
        print(f"start_hab={np.round(start,2).tolist()} heading_yaw={yaw:.2f} facing wall at {d0:.2f} m", flush=True)
        depths = []
        for fwd in (0.0, 0.4, 0.8, 1.2):
            out = c.render(ep.ned_to_hab_pos([fwd, 0, 0]), ep.ned_quat_to_hab_R(np.array([0, 0, 0, 1.0])))
            dd = _central_median_depth(out["depth"])
            depths.append(dd)
            print(f"fwd={fwd:.1f}m  central_GT_depth={dd:.3f}m", flush=True)
        c.close()
        diffs = np.diff(depths)
        drop = depths[0] - depths[-1]
        # forward motion toward the wall must shrink central depth, and by ~the distance moved
        ok = np.all(diffs < 0.05) and (0.7 < drop < 1.6)
        print("FRAME_CHECK", "PASS" if ok else "FAIL",
              f"(total drop {drop:.2f} m over 1.2 m forward; expect ~1.2)", flush=True)
        return ok
    finally:
        proc.terminate()


def test_frame_convention():
    import pytest
    if not os.path.exists(SCENE):
        pytest.skip("needs habitat assets on cluster")
    assert run_check()


if __name__ == "__main__":
    sys.exit(0 if run_check() else 1)
