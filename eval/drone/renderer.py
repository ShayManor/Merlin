#!/usr/bin/env python3
"""6DOF Habitat render server for the drone sim (runs in the conda `habenv`).

Kept in its own process+env so habitat-sim's torch/numpy never collide with the model
env (the student runs in the client). Protocol: length-prefixed pickle over a localhost
TCP socket, identical wire format to eval/sim/habitat_server.py.

Unlike the ground sim (2D navmesh step), the drone camera is a FREE 6DOF sensor: the
client drives it to an arbitrary world position + orientation each frame (PX4 supplies
the pose), and the renderer returns RGB + GT depth + pinhole intrinsics. Collisions are
geometric: cast_ray along +/-X,Y,Z against the scene collision mesh (physics enabled),
returning the min hit distance (the renderer is decoupled from PX4 rigid-body physics).
"""
import argparse
import pickle
import socket
import struct

import numpy as np
import quaternion  # numpy-quaternion, a habitat-sim dependency

import habitat_sim


def make_cfg(scene, hfov, width, height, dataset=None):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene
    if dataset:
        sim_cfg.scene_dataset_config_file = dataset
    sim_cfg.enable_physics = True   # needed for cast_ray against the stage collision mesh
    specs = []
    for uuid, stype in (("rgba", habitat_sim.SensorType.COLOR),
                        ("depth", habitat_sim.SensorType.DEPTH)):
        s = habitat_sim.CameraSensorSpec()
        s.uuid = uuid
        s.sensor_type = stype
        s.resolution = [height, width]
        s.position = [0.0, 0.0, 0.0]   # sensor at the agent origin (free 6DOF camera)
        s.hfov = hfov
        specs.append(s)
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = specs
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def intrinsics(hfov_deg, width, height):
    f = (width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]])


class Renderer:
    def __init__(self, hfov=90.0, width=504, height=504):
        self.hfov = hfov; self.width = width; self.height = height
        self.sim = None; self.scene = None
        self.K = intrinsics(hfov, width, height)
        self._obs = []        # object ids of OUR injected obstacles (for collision attribution)
        self._bbcache = {}    # template handle -> measured bbox, filled once per scene
        self._nav_alt = 0.75; self._floor_y = None; self._solid_cache = None
        self.nav_radius = 0.35
        self._nav_furn = True

    def _ensure_sim(self, scene, dataset=None):
        key = (scene, dataset)
        if self.scene == key and self.sim is not None:
            return
        if self.sim is not None:
            self.sim.close()
        self.sim = habitat_sim.Simulator(make_cfg(scene, self.hfov, self.width, self.height, dataset))
        self.scene = key
        self._obs = []          # object ids do not survive a Simulator swap
        self._bbcache = {}
        self._nav_alt = 0.75
        self._floor_y = None
        self._solid_cache = None
        self._nav_furn = True

    def reset(self, scene, dataset=None):
        self._ensure_sim(scene, dataset)
        return {"ok": True, "K": self.K.copy()}

    def _freeze_scene_objects(self):
        """Mark the scene's own furniture STATIC so recompute_navmesh can see it.

        ReplicaCAD loads furniture as separate rigid objects, so the default navmesh is the
        bare apartment shell: routes ran straight through sofas and kitchen counters, and
        every arm -- including perfect depth -- crashed on the nominal route. Nothing steps
        physics here, so the motion type is a label only. Our injected obstacles are placed
        AFTER the route is computed and are unaffected."""
        rom = self.sim.get_rigid_object_manager()
        for h in rom.get_object_handles():
            o = rom.get_object_by_handle(h)
            if o is not None:
                o.motion_type = habitat_sim.physics.MotionType.STATIC

    def _ensure_navmesh(self, alt=None, furniture=True):
        pf = self.sim.pathfinder
        if not pf.is_loaded or (alt is not None and alt != self._nav_alt) \
                or furniture != self._nav_furn:
            if furniture:
                self._freeze_scene_objects()
            ns = habitat_sim.NavMeshSettings()
            ns.set_defaults()
            ns.agent_radius = float(self.nav_radius)   # > planner safety, see run_experiments
            # A navmesh cell is navigable only if the whole agent column is clear, so the
            # height must cover the drone's flight envelope: at 0.75 m altitude an agent of
            # height 0.5 happily routes under a table the drone flies into.
            ns.agent_height = float(alt or self._nav_alt) + 0.30
            ns.include_static_objects = bool(furniture)   # attribute, NOT a kwarg
            self.sim.recompute_navmesh(pf, ns)
            self._nav_alt = float(alt or self._nav_alt)
            self._nav_furn = bool(furniture)
        return pf

    def _floor_height(self, pf, n=500):
        """Median height of the navigable surface = the floor.

        Including furniture in the navmesh also makes the TOP faces of tall furniture
        navigable, so ~30% of sampled endpoints landed on counters and wardrobes and the
        drone was asked to fly at 3.5 m. Sampling is area-weighted and floor dominates, so
        the median is the floor; elevated islands are then rejected by tolerance."""
        if self._floor_y is None:
            pf.seed(0)
            ys = [float(pf.get_random_navigable_point()[1]) for _ in range(n)]
            self._floor_y = float(np.median(ys))
        return self._floor_y

    def _same_side(self, pf, p, q, rad):
        """True if q is reachable from p without going around anything."""
        sp = habitat_sim.ShortestPath()
        sp.requested_start = [float(x) for x in p]
        sp.requested_end = [float(x) for x in q]
        return bool(pf.find_path(sp)) and sp.geodesic_distance <= 1.5 * rad + 0.05

    def _inflate_route(self, pf, floor_pts, want=0.55, max_shift=0.45):
        """Push interior route corners away from obstacles.

        A geodesic path touches the navmesh boundary at every corner, so the nominal route
        grazes furniture with exactly agent_radius of clearance and a drone that cuts the
        corner at all clips it. Real stacks avoid this with costmap inflation; this is the
        same idea applied to the waypoints. Start and goal are left alone."""
        out = [list(map(float, floor_pts[0]))]
        for p in floor_pts[1:-1]:
            p = np.asarray(p, float)
            best, bd = p, float(pf.distance_to_closest_obstacle(p))
            if bd < want:
                for rad in (0.15, 0.30, max_shift):
                    for th in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                        q = p + np.array([np.cos(th) * rad, 0.0, np.sin(th) * rad])
                        # navigable is not enough: a shift can land across a thin wall in the
                        # next room, which would tear the route in half
                        if not pf.is_navigable(q) or not self._same_side(pf, p, q, rad):
                            continue
                        d = float(pf.distance_to_closest_obstacle(q))
                        if d > bd:
                            best, bd = q, d
                        if bd >= want:
                            break
                    if bd >= want:
                        break
            out.append([float(best[0]), float(best[1]), float(best[2])])
        if len(floor_pts) > 1:
            out.append(list(map(float, floor_pts[-1])))
        return out

    def _route_flyable(self, pts, clearance=0.30, spacing=0.25):
        """Ray-cast the whole route at flight altitude. The navmesh guarantees clearance for
        a cylinder on the floor, which is not quite the same as clearance for a sphere at
        altitude; this is the check that actually matches the collision test."""
        P = np.asarray(pts, float)
        for a, b in zip(P[:-1], P[1:]):
            n = max(1, int(np.ceil(np.linalg.norm(b - a) / spacing)))
            for i in range(n + 1):
                if self.collide(a + (b - a) * (i / n), clearance)["hit"]:
                    return False
        return True

    def sample_episode(self, scene, min_geo, max_geo, seed, alt=1.0, dataset=None,
                       stale=False, route_clear=0.0):
        """Sample an interior drone episode: a navigable floor start + goal with geodesic
        distance in [min_geo, max_geo], raised to drone altitude `alt` (Habitat +Y). Returns
        start_hab, goal_hab, geo, and yaw_hab (azimuth from start toward goal). Deterministic
        in `seed`. Placing episodes at navigable interior points avoids the empty scene origin.
        """
        self._ensure_sim(scene, dataset)
        # A stale route is planned on the bare apartment shell, as if from an out-of-date
        # floor plan: every piece of furniture is then an obstacle the route does not know
        # about, and only the live map can avoid it. Endpoints and route feasibility are
        # still checked against the furnished scene so the episode remains winnable.
        pf = self._ensure_navmesh(alt, furniture=not stale)
        floor = self._floor_height(pf)
        pf.seed(int(seed))
        rejected = 0
        for _ in range(400):
            s = pf.get_random_navigable_point()
            g = pf.get_random_navigable_point()
            if not (pf.is_navigable(s) and pf.is_navigable(g)):
                continue
            if abs(float(s[1]) - floor) > 0.3 or abs(float(g[1]) - floor) > 0.3:
                continue                       # endpoint on top of furniture, not the floor
            # A random navigable point can sit hard against the navmesh boundary; starting
            # there fills the map with wall on the first keyframe and the planner has no room
            # to manoeuvre before it has seen anything. The threshold is expressed in REAL
            # clearance (the navmesh boundary already excludes agent_radius) so that a tight
            # prior map makes the ROUTE harder without also spawning the drone against a wall
            # -- otherwise every arm fails at small nav_radius for a reason that has nothing
            # to do with perception.
            # route_clear demands a genuinely OPEN route: every point on it, and both
            # endpoints, keep this much real clearance from geometry. The default gate below
            # is capped at 0.30 m, which is fine for "not inside a wall" but still routes the
            # drone down corridors barely wider than itself -- and a surface entering the
            # near field is exactly where the student's depth fails (0.64 rel err inside
            # 0.5 m). Asking for space up front removes that regime instead of trying to
            # survive it.
            endpoint_clear = max(max(0.15, 0.50 - self.nav_radius), float(route_clear))
            if min(pf.distance_to_closest_obstacle(s),
                   pf.distance_to_closest_obstacle(g)) < endpoint_clear:
                continue
            sp = habitat_sim.ShortestPath()
            sp.requested_start = s
            sp.requested_end = g
            if not pf.find_path(sp):
                continue
            geo = sp.geodesic_distance
            if not np.isfinite(geo) or geo < min_geo or geo > max_geo:
                continue
            eu = float(np.linalg.norm(np.asarray(s) - np.asarray(g)))
            if eu < 1e-3 or geo / eu > 1.8:   # skip degenerate / very twisty pairs
                continue
            start = [float(s[0]), float(s[1]) + alt, float(s[2])]
            goal = [float(g[0]), float(g[1]) + alt, float(g[2])]
            d = np.asarray(goal) - np.asarray(start)
            yaw_hab = float(np.arctan2(d[0], -d[2]))   # Habitat azimuth start->goal
            # geodesic path waypoints at drone altitude: global routing around walls, which
            # the reactive local planner cannot do on its own. Perception handles local
            # obstacle avoidance between waypoints (global + local, like a real nav stack).
            # Inflation has to scale with the prior map's guarantee too. distance_to_closest
            # _obstacle measures to the navmesh boundary, which already sits agent_radius from
            # real geometry, so a fixed 0.55 m target hands the route ~0.65 m of real
            # clearance even at nav_radius 0.10 -- undoing the tightness the sweep is trying
            # to create. Scaling keeps the 0.35 setting byte-identical to the runs already
            # measured.
            want = 0.55 * min(1.0, self.nav_radius / 0.35)
            pts = self._inflate_route(pf, [np.asarray(p, float) for p in sp.points],
                                      want=want, max_shift=0.45 * min(1.0, self.nav_radius / 0.35))
            waypoints = [[float(p[0]), float(p[1]) + alt, float(p[2])] for p in pts]
            # The route must be flyable before any obstacle is injected, otherwise a crash
            # cannot be attributed to the injected obstacle and even perfect depth fails on
            # the nominal path.
            # The gate must scale with the clearance being asked for, or a deliberately tight
            # prior map (small nav_radius) rejects every route it produces. The floor has to
            # sit BELOW the body radius (0.18): a floor of 0.20 silently guarantees the route
            # clears the drone no matter how small nav_radius is, which defeats the knob at
            # exactly the settings meant to be hard. It only has to keep the route out of
            # solid geometry.
            gate = max(min(0.30, max(0.10, self.nav_radius - 0.05)), float(route_clear))
            if not stale and not self._route_flyable(waypoints, clearance=gate):
                rejected += 1
                continue
            if stale and (self.collide(start, 0.30)["hit"] or self.collide(goal, 0.30)["hit"]):
                rejected += 1          # endpoints must still be clear of real furniture
                continue
            return {"start_hab": start, "goal_hab": goal, "geo": float(geo),
                    "yaw_hab": yaw_hab, "waypoints_hab": waypoints, "rejected": rejected}
        return None

    def render(self, pos_hab, R_cam2world):
        agent = self.sim.get_agent(0)
        st = agent.get_state()
        st.position = np.asarray(pos_hab, dtype=np.float32)
        st.rotation = quaternion.from_rotation_matrix(np.asarray(R_cam2world, dtype=np.float64))
        agent.set_state(st)
        o = self.sim.get_sensor_observations()
        rgb = np.asarray(o["rgba"])[:, :, :3].copy()
        depth = np.asarray(o["depth"]).astype(np.float32).copy()
        return {"rgb": rgb, "depth": depth, "K": self.K.copy()}

    def collide(self, pos_hab, radius, max_dist=5.0):
        origin = np.asarray(pos_hab, dtype=np.float32)
        best = float(max_dist); best_id = None
        for ax in (0, 1, 2):
            for sgn in (1.0, -1.0):
                d = np.zeros(3, dtype=np.float32); d[ax] = sgn
                ray = habitat_sim.geo.Ray(origin, d)
                res = self.sim.cast_ray(ray, max_distance=max_dist)
                if res.has_hits() and float(res.hits[0].ray_distance) < best:
                    best = float(res.hits[0].ray_distance)
                    best_id = int(res.hits[0].object_id)
        # stage id is 0 and the scene's own furniture also has ids >= 0, so a crash is only
        # attributable to an injected obstacle by checking our own recorded id list
        return {"hit": bool(best < radius), "dist": best, "object_id": best_id,
                "is_novel": bool(best_id is not None and best_id in self._obs)}

    def _template_bboxes(self):
        """{handle: {size, min_y}} for every ReplicaCAD file template. Instantiate, measure,
        remove -- once per scene, then cached (92 templates takes seconds)."""
        if self._bbcache:
            return self._bbcache
        otm = self.sim.get_object_template_manager()
        rom = self.sim.get_rigid_object_manager()
        for h in otm.get_file_template_handles(""):
            o = rom.add_object_by_template_handle(h)
            if o is None:
                continue
            bb = o.root_scene_node.cumulative_bb
            self._bbcache[h] = {"size": (float(bb.size()[0]), float(bb.size()[1]),
                                         float(bb.size()[2])),
                                "min_y": float(bb.min[1])}
            rom.remove_object_by_id(o.object_id)
        return self._bbcache

    def _solid_templates(self, handles, tol=0.08, n_dirs=12, alt=0.75, span=0.18):
        """Keep only templates whose collision proxy matches their visible surface.

        Habitat collides against a convex decomposition, but the depth sensor renders the
        visual mesh. For non-convex furniture (bike, chair, plant, clothes hanger) the hull
        bulges through the gaps, so the drone collides with geometry the camera correctly
        reports as empty -- measured at 32% of approach directions, with the proxy up to 1.7 m
        proud of the visible surface. That makes an episode unwinnable for ANY depth model,
        which is exactly what we measured: perfect depth scored 0/4. Each template is probed
        once, in empty space outside the scene, by comparing the centre pixel of a render
        against a ray cast along the same optical axis."""
        if self._solid_cache is not None:
            return [h for h in handles if h in self._solid_cache]
        rom = self.sim.get_rigid_object_manager()
        far = np.array([200.0, 0.0, 200.0])          # empty space: nothing else to hit
        keep = set()
        for h in handles:
            o = rom.add_object_by_template_handle(h)
            if o is None:
                continue
            bb0 = o.root_scene_node.cumulative_bb
            o.translation = [float(far[0]), -float(bb0.min[1]), float(far[2])]  # base on y=0
            o.motion_type = habitat_sim.physics.MotionType.STATIC
            worst = 0.0
            # only the heights the drone can actually occupy matter
            for hgt in (alt - span, alt, alt + span):
                for th in np.linspace(0, 2 * np.pi, n_dirs, endpoint=False):
                    d = np.array([np.cos(th), 0.0, np.sin(th)])
                    cam = np.array([far[0], hgt, far[2]]) - d * 3.0
                    dep = self.render(cam, self._look_at(d))["depth"]
                    k = dep.shape[0] // 2
                    res = self.sim.cast_ray(habitat_sim.geo.Ray(cam.astype(np.float32),
                                                                d.astype(np.float32)),
                                            max_distance=8.0)
                    if not res.has_hits():
                        continue
                    # depth reads 0 where nothing was rendered, so only a POSITIVE gap (proxy
                    # nearer than the visible surface) counts against the template
                    worst = max(worst, float(dep[k, k]) - float(res.hits[0].ray_distance))
            rom.remove_object_by_id(o.object_id)
            if worst <= tol:
                keep.add(h)
        self._solid_cache = keep
        return [h for h in handles if h in keep]

    @staticmethod
    def _look_at(direction):
        d = np.asarray(direction, float)
        z = d / max(float(np.linalg.norm(d)), 1e-9)
        y = np.array([0.0, -1.0, 0.0])
        return np.stack([np.cross(y, z), y, z], 1)

    def place_obstacles(self, waypoints, n, seed, alt=0.75, intrude=1.0, min_width=0.0,
                        include=None,
                        obs_clear=0.0, block_margin=0.20, jitter=0.35, yaw=0.0):
        """Instance n route-unaware obstacles. Called AFTER sample_episode, so the navmesh
        that produced the route never saw them -- that staleness is the whole experiment."""
        from eval.drone import obstacles as ob
        rom = self.sim.get_rigid_object_manager()
        bbs = self._template_bboxes()
        usable = self._solid_templates(
            ob.usable_templates({h: v["size"] for h, v in bbs.items()},
                                min_width=float(min_width), include=include))
        if not usable:
            return {"ok": False, "placed": [], "skipped": n,
                    "reason": f"no template >= {ob.MIN_OBSTACLE_H} m"}
        pf = self._ensure_navmesh()
        floor_y_probe = self._floor_height(pf)
        rng = np.random.default_rng(int(seed))
        # 1.5 m of clear run-up: an obstacle 1 m from the start is inside the drone's
        # stopping distance before the first keyframes have built any map.
        # Oversample candidate sites: only a fraction of a route passes the height, blocking
        # and open-space gates, and taking exactly n candidates left 20 of 22 episodes with no
        # obstacle at all (the blind arm then flew clean and the cell measured nothing).
        pts = ob.route_points(waypoints, max(6 * n, 12), seed, edge_margin=1.5,
                              lateral_jitter=float(jitter))
        # Only the start and the goal have to stay clear. Requiring every densified subgoal
        # to be clear pushed obstacles ~0.8 m off the corridor, and the blind arm then flew
        # past all of them untouched. Subgoals buried by an obstacle are now handled by the
        # nav loop, which drops a subgoal it cannot approach.
        dense = [waypoints[0], waypoints[-1]]
        # Work the roomiest sites first, and fit the furniture to the site rather than the
        # other way round: drawing a template blind and then testing whether it fits rejected
        # nearly every candidate, because the tall templates that straddle flight altitude are
        # also the wide ones.
        room = {}
        for p in pts:
            room[tuple(p)] = float(pf.distance_to_closest_obstacle(
                [float(p[0]), floor_y_probe, float(p[2])]))
        pts = sorted(pts, key=lambda p: -room[tuple(p)])
        placed, skipped = [], 0
        for p in pts:
            if len(placed) >= n:
                break
            # the drone needs robot_radius + planner safety of room beside the obstacle, and
            # the navmesh has already excluded agent_radius from this clearance
            # obs_clear reserves a genuinely wide gap BESIDE the obstacle, so routing round
            # it never puts the drone within near-field range of a wall. Without it an
            # obstacle can sit in a corridor only marginally wider than itself, and the
            # detour is precisely the manoeuvre that crashes.
            fits = [h for h in usable
                    if 0.5 * max(bbs[h]["size"][0], bbs[h]["size"][2])
                    <= room[tuple(p)] - 0.15 - float(obs_clear)]
            if not fits:
                skipped += 1
                continue
            handle = fits[int(rng.integers(len(fits)))]
            size, min_y = bbs[handle]["size"], bbs[handle]["min_y"]
            perp = ob.route_perp(waypoints, p)
            p = ob.lateral_place(p, perp, size, intrude=intrude,
                                 sign=1.0 if rng.random() < 0.5 else -1.0)
            q = ob.shift_clear_of_waypoints(p, perp, size, dense, margin=0.30)
            if q is None:
                skipped += 1
                continue
            p = q
            if not ob.blocks_route(waypoints, p, size, margin=float(block_margin)):
                skipped += 1
                continue
            # Re-check clearance at the FINAL position (lateral_place and the waypoint shift
            # both move it). An obstacle dropped into a doorway leaves a gap narrower than the
            # planner's own clearance, so the episode is unwinnable for every arm -- measured
            # at 0/6 even with perfect depth.
            fp = [float(p[0]), floor_y_probe, float(p[2])]
            half = 0.5 * max(float(size[0]), float(size[2]))
            if float(pf.distance_to_closest_obstacle(fp)) < half + 0.15 + float(obs_clear):
                skipped += 1
                continue
            lift = ob.floor_lift(min_y)
            floor_y = float(p[1]) - float(alt)          # route altitude -> floor
            pos = [float(p[0]), floor_y + lift, float(p[2])]
            o = rom.add_object_by_template_handle(handle)
            if o is None:
                skipped += 1
                continue
            o.translation = pos
            if yaw:
                # Templates are instanced in their authored orientation. Yaw about the
                # habitat up-axis turns an asymmetric object so a different face meets the
                # route -- which matters for open-frame furniture (shelving, chairs), where
                # one face is solid and the other is mostly gaps.
                import magnum as mn
                o.rotation = mn.Quaternion.rotation(mn.Deg(float(yaw)), mn.Vector3.y_axis())
            o.motion_type = habitat_sim.physics.MotionType.STATIC
            bb = o.root_scene_node.cumulative_bb
            lo, hi = pos[1] + float(bb.min[1]), pos[1] + float(bb.max[1])
            if not ob.straddles_alt(lo, hi, float(p[1])):
                rom.remove_object_by_id(o.object_id)
                skipped += 1
                continue
            if any(np.linalg.norm(np.asarray(q["pos"])[[0, 2]] - np.asarray(pos)[[0, 2]]) < 1.0
                   for q in placed):
                rom.remove_object_by_id(o.object_id)
                skipped += 1
                continue
            self._obs.append(int(o.object_id))
            # Footprint after yaw: the axis-aligned extents the map and the blockage tests
            # use are not the template's once it has been turned.
            th = np.deg2rad(float(yaw))
            fw = abs(size[0] * np.cos(th)) + abs(size[2] * np.sin(th))
            fd = abs(size[0] * np.sin(th)) + abs(size[2] * np.cos(th))
            placed.append({"handle": handle, "object_id": int(o.object_id),
                           "pos": pos, "bbox": [float(fw), float(size[1]), float(fd)],
                           "lift": lift, "yaw": float(yaw)})
        return {"ok": len(placed) > 0, "placed": placed, "skipped": skipped}

    def clear_obstacles(self):
        rom = self.sim.get_rigid_object_manager()
        k = 0
        for oid in self._obs:
            try:
                rom.remove_object_by_id(oid)
                k += 1
            except Exception:
                pass
        self._obs = []
        return {"ok": True, "removed": k}

    def route_still_open(self, start_hab, goal_hab):
        """Gate 1: with our obstacles included, is start->goal still connected?
        Uses a scratch pathfinder so the episode's own navmesh is untouched. Scene furniture
        is STATIC (see _freeze_scene_objects), so this sees the whole scene."""
        ns = habitat_sim.NavMeshSettings()
        ns.set_defaults()
        # The gap left beside an obstacle has to fit the drone AND the clearance its planner
        # insists on (safety 0.25 around a 0.18 m body), not just the bare airframe: a gap
        # the body would squeeze through is one the planner will refuse to enter.
        ns.agent_radius = 0.35
        ns.agent_height = self._nav_alt + 0.30
        ns.include_static_objects = True    # NavMeshSettings attribute, NOT a recompute kwarg
        pf = habitat_sim.nav.PathFinder()
        if not self.sim.recompute_navmesh(pf, ns):
            return {"open": True, "geo": None, "note": "recompute failed; not blocking"}
        # start/goal are episode points raised to flight altitude; the pathfinder needs
        # navigable FLOOR points, so snap them down before querying or find_path always fails
        s = pf.snap_point([float(x) for x in start_hab])
        g = pf.snap_point([float(x) for x in goal_hab])
        s, g = np.asarray(s, float), np.asarray(g, float)
        if not (np.all(np.isfinite(s)) and np.all(np.isfinite(g))):
            return {"open": False, "geo": None, "note": "endpoint unsnappable"}
        sp = habitat_sim.ShortestPath()
        sp.requested_start = [float(s[0]), float(s[1]), float(s[2])]
        sp.requested_end = [float(g[0]), float(g[1]), float(g[2])]
        ok = pf.find_path(sp)
        return {"open": bool(ok and np.isfinite(sp.geodesic_distance)),
                "geo": float(sp.geodesic_distance) if ok else None}

    def topdown(self, meters_per_pixel=0.05, height=None):
        """Floor plan for the path figure. ReplicaCAD apartments have ceilings, so an overhead
        camera would render the ceiling; the navmesh view is the clean option."""
        pf = self._ensure_navmesh()
        # slice at the FLOOR, not the scene bounding box minimum: the stage geometry extends
        # well below the floor, and a slice taken down there returns a different navigable
        # set, which put the goal inside a wall on the path figure
        h = float(height) if height is not None else self._floor_height(pf) + 0.1
        grid = np.asarray(pf.get_topdown_view(meters_per_pixel, h))
        lo, hi = pf.get_bounds()
        return {"grid": grid.astype(bool),
                "bounds": [[float(lo[0]), float(lo[2])], [float(hi[0]), float(hi[2])]],
                "mpp": float(meters_per_pixel)}

    def handle(self, msg):
        c = msg["cmd"]
        if c == "reset":
            if msg.get("nav_radius"):
                self.nav_radius = float(msg["nav_radius"])
            return self.reset(msg["scene"], msg.get("dataset"))
        if c == "sample":
            return self.sample_episode(msg["scene"], msg["min_geo"], msg["max_geo"],
                                       msg["seed"], msg.get("alt", 1.0), msg.get("dataset"),
                                       msg.get("stale", False),
                                       msg.get("route_clear", 0.0))
        if c == "render":
            return self.render(msg["pos"], msg["R"])
        if c == "collide":
            return self.collide(msg["pos"], msg.get("radius", 0.25), msg.get("max_dist", 5.0))
        if c == "place_obstacles":
            return self.place_obstacles(msg["waypoints"], msg["n"], msg["seed"],
                                        msg.get("alt", 0.75), msg.get("intrude", 0.25),
                                        msg.get("min_width", 0.0),
                                        msg.get("include"), msg.get("obs_clear", 0.0),
                                        msg.get("block_margin", 0.20),
                                        msg.get("jitter", 0.35), msg.get("yaw", 0.0))
        if c == "clear_obstacles":
            return self.clear_obstacles()
        if c == "route_open":
            return self.route_still_open(msg["start"], msg["goal"])
        if c == "topdown":
            return self.topdown(msg.get("mpp", 0.05), msg.get("height"))
        if c == "ping":
            return {"ok": True}
        raise ValueError(c)


def _recv(conn, n):
    buf = b""
    while len(buf) < n:
        b = conn.recv(n - len(buf))
        if not b:
            return None
        buf += b
    return buf


def serve(port, hfov=90.0, width=504, height=504):
    r = Renderer(hfov=hfov, width=width, height=height)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", port)); s.listen(1)
    print(f"DRONE_RENDERER_READY port={port}", flush=True)
    conn, _ = s.accept()
    while True:
        hdr = _recv(conn, 4)
        if hdr is None:
            break
        n = struct.unpack(">I", hdr)[0]
        msg = pickle.loads(_recv(conn, n))
        if msg.get("cmd") == "close":
            break
        try:
            rep = r.handle(msg)
        except Exception as e:
            rep = {"error": repr(e)}
        out = pickle.dumps(rep)
        conn.sendall(struct.pack(">I", len(out)) + out)
    conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5601)
    ap.add_argument("--hfov", type=float, default=90.0)
    ap.add_argument("--width", type=int, default=504)
    ap.add_argument("--height", type=int, default=504)
    args = ap.parse_args()
    serve(args.port, args.hfov, args.width, args.height)
