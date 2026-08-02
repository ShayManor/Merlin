"""Headless kinematic nav testbed: planner + voxel map against analytic box geometry.

Habitat and PX4 are only available on the cluster, so planner behaviour could previously
only be exercised by a full 90 s closed-loop episode. This reproduces the closed loop
(render -> depth -> integrate -> plan -> move) in pure numpy against axis-aligned boxes,
which makes planner regressions unit-testable in milliseconds. Same conventions as the real
loop: Habitat world (+Y up), optical camera frame (+Z forward, +X right, +Y down).
"""
import numpy as np

from eval.drone.mapper import VoxelMap
from eval.drone.planner import Planner3D


def look_at_R(direction):
    """R_cam2world for a camera looking along the horizontal component of `direction`."""
    d = np.asarray(direction, float).copy()
    d[1] = 0.0
    n = float(np.linalg.norm(d))
    d = np.array([1.0, 0.0, 0.0]) if n < 1e-9 else d / n
    z = d
    y = np.array([0.0, -1.0, 0.0])          # world down = optical +Y
    x = np.cross(y, z)
    return np.stack([x, y, z], 1)


class Boxes:
    """Axis-aligned boxes: centre (3,) and half-extent (3,) each."""

    def __init__(self, boxes):
        self.c = np.array([b[0] for b in boxes], float).reshape(-1, 3)
        self.h = np.array([b[1] for b in boxes], float).reshape(-1, 3)

    def depth(self, pos, R, K, far=7.0):
        """Perspective z-depth image by slab intersection.

        Misses read as `far` rather than infinity: a real room always has a wall, and without
        one nothing carries the carving that marks free space as observed."""
        H = W = int(K[0, 2] * 2)
        us, vs = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        d_cam = np.stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1],
                          np.ones_like(us)], -1).reshape(-1, 3)
        d_w = d_cam @ R.T                                   # (P,3) unnormalised, z_cam = 1
        o = np.asarray(pos, float)
        t = np.full(len(d_w), np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            for c, h in zip(self.c, self.h):
                inv = 1.0 / d_w
                t1 = (c - h - o) * inv
                t2 = (c + h - o) * inv
                tn = np.nanmax(np.minimum(t1, t2), 1)
                tf = np.nanmin(np.maximum(t1, t2), 1)
                hit = (tf >= np.maximum(tn, 0.0))
                t = np.where(hit, np.minimum(t, np.maximum(tn, 0.0)), t)
        z = np.where(np.isfinite(t), t, far).reshape(H, W)   # t is in z_cam units already
        return np.clip(z, 0.0, far).astype(np.float32)

    def hit(self, pos, radius):
        p = np.asarray(pos, float)
        d = np.abs(p - self.c) - self.h
        return bool(np.any(np.linalg.norm(np.maximum(d, 0.0), axis=1) <= radius))


def fly(boxes, waypoints, cfg=None, blind=False):
    """Run the closed loop over `waypoints` (list of (3,) Habitat points). Returns a dict
    with the same outcome vocabulary as nav_loop.run_episode."""
    cfg = dict(cfg or {})
    res = cfg.get("res", 48)
    f = res / (2 * np.tan(np.deg2rad(cfg.get("hfov", 90.0)) / 2))
    K = np.array([[f, 0, res / 2], [0, f, res / 2], [0, 0, 1.0]])
    vmap = VoxelMap(voxel=cfg.get("voxel", 0.15), min_hits=cfg.get("min_hits", 2))
    plan = Planner3D(step=cfg.get("step", 0.4), safety=cfg.get("safety", 0.35),
                     max_speed=cfg.get("max_speed", 0.5),
                     escape=cfg.get("escape", True),
                     # the blind control arm has no map at all, so requiring
                     # observed space would simply freeze it on the spot
                     require_known=cfg.get("require_known", False))
    dt = cfg.get("dt", 0.1)
    kf_dt = cfg.get("kf_dt", 0.3)
    speed = cfg.get("max_speed", 0.5)   # validated indoor speed; see test_simnav
    robot_r = cfg.get("robot_radius", 0.18)
    wp_tol = cfg.get("wp_tol", 0.7)
    goal_tol = cfg.get("goal_tol", 0.6)
    wps = [np.asarray(w, float) for w in waypoints]
    pos = wps[0].copy()
    vel = np.zeros(3)
    accel = cfg.get("accel", 2.0)          # m/s^2, a modest indoor multirotor limit
    idx = min(1, len(wps) - 1)
    path = [pos.copy()]
    last_kf = -1e9
    t = 0.0
    blocked_ticks = 0
    n_ticks = 0
    stuck = 0
    while t < cfg.get("timeout", 60.0):
        if not blind and t - last_kf >= kf_dt:
            R = look_at_R(wps[idx] - pos)
            vmap.integrate(boxes.depth(pos, R, K), K, R, pos,
                           max_range=cfg.get("max_range", 8.0))
            last_kf = t
        if boxes.hit(pos, robot_r):
            return {"ended": "collision", "path": np.array(path), "t": t, "wp": idx,
                    "blocked_frac": blocked_ticks / max(1, n_ticks)}
        while idx < len(wps) - 1 and np.linalg.norm(pos - wps[idx]) < wp_tol:
            idx += 1
        tgt, blocked = plan.next_setpoint(vmap, pos, wps[idx])
        n_ticks += 1
        blocked_ticks += int(bool(blocked))
        stuck = stuck + 1 if blocked else 0      # mirrors nav_loop's unreachable-subgoal skip
        if stuck > cfg.get("skip_ticks", 20) and idx < len(wps) - 1:
            idx += 1
            stuck = 0
        # Accel-limited, not teleporting: a purely kinematic drone stops dead the tick the
        # planner says stop, which hides the fact that the real vehicle needs stopping
        # distance inside the clearance budget. That gap is why the testbed passed cases the
        # PX4 loop crashed.
        want = tgt - pos
        n = float(np.linalg.norm(want))
        want = want / n * speed if n > 1e-9 else want * 0.0
        dv = want - vel
        a = float(np.linalg.norm(dv))
        if a > accel * dt:
            dv = dv / a * accel * dt
        vel = vel + dv
        pos = pos + vel * dt
        path.append(pos.copy())
        t += dt
        if np.linalg.norm(pos - wps[-1]) < goal_tol:
            return {"ended": "goal", "path": np.array(path), "t": t, "wp": idx,
                    "blocked_frac": blocked_ticks / max(1, n_ticks)}
    return {"ended": "timeout", "path": np.array(path), "t": t, "wp": idx,
            "blocked_frac": blocked_ticks / max(1, n_ticks)}
