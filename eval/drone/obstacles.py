"""Placement geometry for novel (route-unaware) obstacles.

Pure numpy so it unit-tests without habitat-sim; renderer.py does the habitat calls.
Habitat is Y-up: positions are (x, y, z) and the horizontal plane is XZ.

Obstacles must straddle the drone's flight altitude. Furniture shorter than the drone
passes underneath it and the experiment silently measures nothing, so height is a gate,
not a preference.
"""
import numpy as np

MIN_OBSTACLE_H = 0.8      # flight alt 0.75 + robot radius 0.18 -> drone spans ~0.57-0.93


def usable_templates(bboxes, min_height=MIN_OBSTACLE_H, min_width=0.0, include=None):
    """bboxes: {handle: (x, y, z) extent}. Keep handles tall enough to be real obstacles.

    min_width additionally restricts to BULKY furniture. The student maps an object's near
    face ~0.27 m too far, so a narrow object leaves the planner almost no true clearance to
    round it, while a wide one makes that displacement a small fraction of the detour. It
    makes the task easier for a live map without helping the no-map arm at all -- a wider
    obstacle on the route is if anything harder to miss blind.

    `include` restricts to handles containing any of the given substrings. It exists for
    BUILDING TRAINING DATA: the chair recovers only 0.56 of its depth step against 0.76-0.97
    for every other object, so thin strut structure has to be over-represented deliberately
    when generating near-field frames. It is not for the evaluation set -- restricting the
    obstacles a run is scored on selects on the outcome.
    """
    hs = [h for h, b in bboxes.items()
          if b[1] >= min_height and max(b[0], b[2]) >= min_width]
    if include:
        hs = [h for h in hs if any(s in h for s in include)]
    return hs


def floor_lift(bb_min_y):
    """Vertical shift that puts an object's lowest point on the floor. Wall-mounted
    ReplicaCAD templates have their bb origin near the top (bb.min.y ~ -1.066)."""
    return -float(bb_min_y)


def _cum_len(pts):
    seg = np.linalg.norm(np.diff(pts[:, [0, 2]], axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)]), seg


def _at_arclen(pts, cum, s):
    i = int(np.clip(np.searchsorted(cum, s) - 1, 0, len(pts) - 2))
    span = cum[i + 1] - cum[i]
    t = 0.0 if span < 1e-9 else (s - cum[i]) / span
    return pts[i] + (pts[i + 1] - pts[i]) * t, i


def route_points(waypoints, n, seed, edge_margin=1.0, lateral_jitter=0.35):
    """n placement points along the route polyline, at route altitude.

    Points sit at evenly spaced arc-length fractions with a seeded along-track and lateral
    jitter, so obstacles are not always dead-centre (which would look artificial and would
    make the blocking gate vacuous). Excludes edge_margin at both ends so nothing spawns on
    the drone or the goal.
    """
    pts = np.asarray(waypoints, float)
    cum, _ = _cum_len(pts)
    total = float(cum[-1])
    lo, hi = edge_margin, max(edge_margin, total - edge_margin)
    if hi <= lo:                       # route too short to respect the margins
        lo, hi = 0.25 * total, 0.75 * total
    rng = np.random.default_rng(int(seed))
    out = []
    for k in range(n):
        frac = (k + 0.5) / n
        s = lo + (hi - lo) * frac
        jit = (hi - lo) / max(n, 1) * 0.25
        s = float(np.clip(s + rng.uniform(-jit, jit), lo, hi))
        p, i = _at_arclen(pts, cum, s)
        d = pts[i + 1] - pts[i]
        nrm = np.linalg.norm([d[0], d[2]])
        if nrm > 1e-9 and lateral_jitter > 0:
            perp = np.array([-d[2] / nrm, 0.0, d[0] / nrm])
            p = p + perp * rng.uniform(-lateral_jitter, lateral_jitter)
        out.append(p)
    return np.array(out, float)


def densify(waypoints, spacing=1.0):
    """Mirror of the driver's densification, in the Habitat frame.

    Placement must respect the subgoals the planner actually steers to, not the raw navmesh
    corners: nav_loop follows the densified route.
    """
    w = [np.asarray(x, float) for x in waypoints]
    if len(w) < 2:
        return np.array(w, float)
    out = [w[0]]
    for a, b in zip(w[:-1], w[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / spacing)))
        for i in range(1, n + 1):
            out.append(a + (b - a) * (i / n))
    return np.array(out, float)


def route_perp(waypoints, pos):
    """Unit vector perpendicular to the route at the point nearest `pos` (Habitat XZ)."""
    P = np.asarray(waypoints, float)
    p = np.asarray(pos, float)
    j = int(np.argmin(np.linalg.norm(P[:, [0, 2]] - p[[0, 2]], axis=1)))
    i = min(max(j, 0), len(P) - 2)
    d = P[i + 1] - P[i]
    n = float(np.linalg.norm([d[0], d[2]]))
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0])
    return np.array([-d[2] / n, 0.0, d[0] / n])


def covers_waypoint(pos, bbox, wps, robot_radius=0.18, margin=0.10):
    """True if the obstacle's footprint (inflated by the drone) swallows any waypoint.

    A buried waypoint makes the episode unwinnable for ANY perception system: the drone is
    commanded into the obstacle, and the greedy planner only accepts strictly goal-improving
    directions, so no lateral escape is ever legal.
    """
    r = 0.5 * max(float(bbox[0]), float(bbox[2])) + robot_radius + margin
    W = np.asarray(wps, float)[:, [0, 2]]
    return bool(np.any(np.linalg.norm(W - np.asarray(pos, float)[[0, 2]], axis=1) <= r))


def shift_clear_of_waypoints(pos, perp, bbox, wps, robot_radius=0.18, margin=0.10,
                             step=0.12, max_shift=2.0):
    """Slide the obstacle sideways until it clears every waypoint's box.

    Keeps the obstacle intruding on the flight corridor (so avoidance still needs the live
    map) while leaving the goal reachable. Returns the adjusted position, or None if no
    offset within max_shift clears the waypoints.
    """
    pos = np.asarray(pos, float)
    if not covers_waypoint(pos, bbox, wps, robot_radius, margin):
        return pos
    perp = np.asarray(perp, float)
    for i in range(1, int(max_shift / step) + 1):
        for sgn in (1.0, -1.0):
            cand = pos + perp * (sgn * i * step)
            if not covers_waypoint(cand, bbox, wps, robot_radius, margin):
                return cand
    return None


def straddles_alt(bb_min_y, bb_max_y, alt):
    """True if the PLACED object's vertical extent covers the flight altitude."""
    return float(bb_min_y) <= float(alt) <= float(bb_max_y)


def blocks_route(waypoints, pos, bbox, robot_radius=0.18, margin=0.20):
    """True if the route passes close enough that the drone must react.

    The footprint is approximated by a circle of radius 0.5*max(x, z) extent; ReplicaCAD
    furniture is roughly convex at this scale and the margin absorbs the approximation.

    `margin` is the slack: positive means "close enough to matter", which is right for tight
    corridors where any intrusion forces a deviation. In a WIDE corridor that slack lets the
    no-map arm sail past on the nominal route (measured: blind 0.67 with route_clear 0.6), so
    a strict test wants margin <= 0 -- the route must pass within the footprint plus the
    drone's own radius, i.e. flying it is a guaranteed contact.
    """
    pts = np.asarray(waypoints, float)[:, [0, 2]]
    p = np.asarray(pos, float)[[0, 2]]
    r = 0.5 * max(float(bbox[0]), float(bbox[2])) + robot_radius + margin
    best = np.inf
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        L2 = float(ab @ ab)
        t = 0.0 if L2 < 1e-12 else float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
        best = min(best, float(np.linalg.norm(p - (a + ab * t))))
    return best <= r


def lateral_place(pos, perp, bbox, intrude=0.25, sign=1.0):
    """Offset an obstacle sideways so its footprint edge sits `intrude` past the route line.

    Placing obstacle centres on the route line seals indoor corridors (half the layouts were
    unroutable, and the rest were unavoidable even with perfect depth); clearing every
    densified subgoal pushes them ~0.8 m clear and the blind arm flies past untouched. Fixing
    the INTRUSION instead makes the blocked fraction of the corridor independent of how big
    the chosen furniture is: the drone must deviate by intrude + its own clearance, and the
    far side of the corridor stays open.
    """
    half = 0.5 * max(float(bbox[0]), float(bbox[2]))
    return np.asarray(pos, float) + np.asarray(perp, float) * (sign * max(0.0, half - intrude))
