#!/usr/bin/env python3
"""Global occupancy-mapping + A* planner -- the 'smarter planner' test.

The reactive planner (planner.py) brakes on any perceived obstacle and is guided globally
by the navmesh, so perception barely matters (the closed-loop 'planner-bound' finding). This
planner instead BUILDS its own 2D occupancy grid from PERCEIVED depth accumulated over time
and plans the global path on THAT map with A*. Now perception errors flow into the global
route: under-seeing an obstacle -> A* plans through it -> collision; over-seeing -> phantom
walls -> detours; and a GT-depth oracle builds a perfect map -> optimal paths (not the
over-conservative braking that made the oracle look bad reactively). This is the test of
whether perception fidelity (M1) RE-ENTERS once the planner actually uses the map.

No navmesh guide: the goal bearing comes from A* on the self-built map. Projection uses the
Habitat camera convention (+x right, +y up, -z forward; depth = perpendicular z-distance).
"""
import heapq

import numpy as np


class MapPlanner:
    def __init__(self, start, goal, cell=0.10, margin=6.0, robot_radius=0.18,
                 obs_lo=0.15, obs_hi=0.95, max_range=4.0, replan_every=3, occ_thresh=1):
        s = np.asarray(start, np.float32); g = np.asarray(goal, np.float32)
        lo = np.minimum(s, g) - margin; hi = np.maximum(s, g) + margin
        self.x0, self.z0 = float(lo[0]), float(lo[2])
        self.cell = cell
        self.nx = int((hi[0]-lo[0]) / cell) + 1
        self.nz = int((hi[2]-lo[2]) / cell) + 1
        self.occ = np.zeros((self.nx, self.nz), np.int32)     # hit count per cell
        self.free = np.zeros((self.nx, self.nz), np.int32)    # free observations per cell
        self.goal = g
        self.robot_radius = robot_radius
        self.obs_lo, self.obs_hi, self.max_range = obs_lo, obs_hi, max_range
        self.replan_every = replan_every
        self.occ_thresh = occ_thresh
        self._tick = 0
        self._path = None

    def _cell(self, x, z):
        return int((x-self.x0)/self.cell), int((z-self.z0)/self.cell)

    def _inb(self, ix, iz):
        return 0 <= ix < self.nx and 0 <= iz < self.nz

    def update(self, pos, yaw, depth, hfov, sensor_height=0.88):
        H, W = depth.shape
        fx = (W/2) / np.tan(np.deg2rad(hfov)/2); fy = fx; cx, cy = W/2, H/2
        # subsample for speed
        vs = np.arange(0, H, 4); us = np.arange(0, W, 4)
        uu, vv = np.meshgrid(us, vs)
        d = depth[vv, uu]
        m = np.isfinite(d) & (d > 0.1) & (d < self.max_range)
        d = d[m]; uu = uu[m]; vv = vv[m]
        xn = (uu - cx)/fx; yn = (cy - vv)/fy
        pc = np.stack([xn*d, yn*d, -d], 1)                    # camera frame (-z forward)
        cy_, sy = np.cos(yaw), np.sin(yaw)
        Ry = np.array([[cy_, 0, sy], [0, 1, 0], [-sy, 0, cy_]])  # R_y(yaw)
        wp = pc @ Ry.T + np.array([pos[0], pos[1]+sensor_height, pos[2]])
        h = wp[:, 1]
        is_obs = (h > self.obs_lo) & (h < self.obs_hi)
        cam = np.array([pos[0], pos[2]])
        for k in range(len(wp)):
            ix, iz = self._cell(wp[k, 0], wp[k, 2])
            if not self._inb(ix, iz):
                continue
            if is_obs[k]:
                self.occ[ix, iz] += 1
            # mark free cells along the ray from camera to this point (Bresenham-ish)
            c0 = self._cell(cam[0], cam[1]); c1 = (ix, iz)
            for fc in self._ray(c0, c1):
                if fc != c1 and self._inb(*fc):
                    self.free[fc] += 1
        self._tick += 1
        self._path = None

    def _ray(self, a, b):
        (x0, y0), (x1, y1) = a, b
        pts = []; dx = abs(x1-x0); dy = abs(y1-y0)
        sx = 1 if x0 < x1 else -1; sy = 1 if y0 < y1 else -1
        err = dx-dy; n = 0
        while n < 200:
            pts.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2*err
            if e2 > -dy: err -= dy; x0 += sx
            if e2 < dx: err += dx; y0 += sy
            n += 1
        return pts

    def _grid_blocked(self):
        # a cell is an obstacle if hits dominate; inflate by robot radius. occ_thresh raises
        # the hit count needed (stricter = less clutter from thin/fine obstacles -> tests
        # whether the 'perfect depth hurts' effect is a mapping-clutter artifact).
        block = (self.occ >= self.occ_thresh) & (self.occ >= self.free)
        r = int(np.ceil(self.robot_radius/self.cell))
        if r > 0 and block.any():
            from scipy.ndimage import binary_dilation
            block = binary_dilation(block, iterations=r)
        return block

    def _astar(self, start_c, goal_c, block):
        if not (self._inb(*start_c) and self._inb(*goal_c)):
            return None
        # if goal blocked, aim for nearest free neighbor
        nbrs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        openh = [(0, start_c)]; g = {start_c: 0.0}; came = {}
        while openh:
            _, cur = heapq.heappop(openh)
            if cur == goal_c:
                path = [cur]
                while cur in came:
                    cur = came[cur]; path.append(cur)
                return path[::-1]
            for dx, dz in nbrs:
                nx_, nz_ = cur[0]+dx, cur[1]+dz
                if not self._inb(nx_, nz_) or block[nx_, nz_]:
                    continue
                ng = g[cur] + (1.414 if dx and dz else 1.0)
                nc = (nx_, nz_)
                if nc not in g or ng < g[nc]:
                    g[nc] = ng; came[nc] = cur
                    h = abs(nx_-goal_c[0]) + abs(nz_-goal_c[1])
                    heapq.heappush(openh, (ng + h, nc))
        return None

    def bearing(self, pos, yaw):
        """Goal bearing (robot frame) from A* on the self-built map. Replans periodically."""
        if self._path is None or self._tick % self.replan_every == 0:
            block = self._grid_blocked()
            sc = self._cell(pos[0], pos[2]); gc = self._cell(self.goal[0], self.goal[2])
            block[sc] = False; block[gc] = False
            self._path = self._astar(sc, gc, block)
        if not self._path or len(self._path) < 2:
            # no path on current map: head straight at goal (optimistic)
            wx, wz = float(self.goal[0]), float(self.goal[2])
        else:
            look = min(6, len(self._path)-1)               # waypoint a few cells ahead
            cx_, cz_ = self._path[look]
            wx = self.x0 + (cx_+0.5)*self.cell; wz = self.z0 + (cz_+0.5)*self.cell
        dx = wx - pos[0]; dz = wz - pos[2]
        world_ang = np.arctan2(-dx, -dz)
        return float(np.arctan2(np.sin(world_ang - yaw), np.cos(world_ang - yaw)))
