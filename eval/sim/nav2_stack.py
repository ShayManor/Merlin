#!/usr/bin/env python3
"""A faithful (ROS2-Nav2-style) navigation stack for the closed-loop sim, to test whether
the 'perception fidelity does not help / hurts reactive navigation' finding survives a
PROPERLY ENGINEERED planner -- the make-or-break experiment.

The reactive planner (planner.py) and the A* planner (map_planner.py + run_map.py) both set
SPEED by proportional braking on the raw nearest perceived depth, so accurate depth makes them
over-conservative. Nav2 does not do this. Its defining behaviours, reproduced here:
  1. a persistent 2D costmap with INFLATION (a decaying cost gradient around obstacles, so
     clearance comes from the cost field, not from braking on every reading);
  2. a global plan (A*) that routes on the INFLATED costmap (paths keep their distance);
  3. a DWA/DWB local controller that follows the path by rolling out candidate (v, omega)
     trajectories and scoring them against the inflated costmap + path progress + speed --
     so the robot maintains speed along a clear path and only slows for trajectories that
     would actually enter the inflated/lethal zone, planning AROUND obstacles instead of
     braking on them;
  4. a rotate-in-place recovery when no admissible trajectory exists.

If the fidelity finding flips here (good depth -> higher success), perception fidelity matters
once the planner is properly engineered. If it survives, the finding is robust to standard
practice -- a much stronger result.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt

from map_planner import MapPlanner


class Nav2Stack(MapPlanner):
    def __init__(self, start, goal, inflation_radius=0.45, inflation_cost=8.0,
                 lookahead=0.8, max_speed=0.5, max_yaw=1.6, horizon=1.0, dt=0.1, **kw):
        super().__init__(start, goal, **kw)
        self.infl_r = inflation_radius          # metres of decaying cost beyond the obstacle
        self.infl_c = inflation_cost
        self.lookahead = lookahead
        self.max_speed = max_speed
        self.max_yaw = max_yaw
        self.horizon = horizon
        self.dt = dt
        self._cost = None

    # ---- costmap with inflation (Nav2 inflation_layer) ----
    def cost_field(self):
        block = (self.occ >= self.occ_thresh) & (self.occ >= self.free)
        # lethal footprint = obstacle dilated by robot radius; beyond it, an exponentially
        # decaying inflation cost out to inflation_radius (standard Nav2 inflation).
        dist = distance_transform_edt(~block) * self.cell      # metres to nearest obstacle
        lethal = dist <= self.robot_radius
        infl = self.infl_c * np.exp(-(dist - self.robot_radius) / max(self.infl_r, 1e-3))
        cost = np.where(lethal, 1e6, np.where(dist < self.robot_radius + self.infl_r, infl, 0.0))
        self._cost = cost
        return cost

    # ---- A* that routes on the inflated costmap (prefers clearance) ----
    def _astar_cost(self, start_c, goal_c, cost):
        import heapq
        if not (self._inb(*start_c) and self._inb(*goal_c)):
            return None
        lethal = cost >= 1e5
        if lethal[goal_c]:           # snap goal to nearest non-lethal cell
            free = np.argwhere(~lethal)
            if len(free) == 0:
                return None
            j = np.argmin(np.abs(free[:, 0] - goal_c[0]) + np.abs(free[:, 1] - goal_c[1]))
            goal_c = (int(free[j, 0]), int(free[j, 1]))
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        openh = [(0.0, start_c)]; g = {start_c: 0.0}; came = {}
        while openh:
            _, cur = heapq.heappop(openh)
            if cur == goal_c:
                path = [cur]
                while cur in came:
                    cur = came[cur]; path.append(cur)
                return path[::-1]
            for dx, dz in nbrs:
                nx_, nz_ = cur[0] + dx, cur[1] + dz
                if not self._inb(nx_, nz_) or lethal[nx_, nz_]:
                    continue
                step = (1.414 if dx and dz else 1.0)
                ng = g[cur] + step + 0.05 * cost[nx_, nz_]      # clearance-aware edge cost
                nc = (nx_, nz_)
                if nc not in g or ng < g[nc]:
                    g[nc] = ng; came[nc] = cur
                    h = abs(nx_ - goal_c[0]) + abs(nz_ - goal_c[1])
                    heapq.heappush(openh, (ng + h, nc))
        return None

    def plan(self, pos):
        cost = self.cost_field()
        sc = self._cell(pos[0], pos[2]); gc = self._cell(self.goal[0], self.goal[2])
        cells = self._astar_cost(sc, gc, cost)
        if not cells:
            self._path_world = None
            return None
        self._path_world = np.array([[self.x0 + (ix + 0.5) * self.cell,
                                      self.z0 + (iz + 0.5) * self.cell] for ix, iz in cells])
        return self._path_world

    def _carrot(self, pos):
        """Lookahead point on the planned path (pure-pursuit carrot)."""
        if getattr(self, "_path_world", None) is None or len(self._path_world) == 0:
            return np.array([self.goal[0], self.goal[2]])
        p = np.array([pos[0], pos[2]])
        d = np.linalg.norm(self._path_world - p, axis=1)
        i0 = int(np.argmin(d))
        for i in range(i0, len(self._path_world)):
            if np.linalg.norm(self._path_world[i] - p) >= self.lookahead:
                return self._path_world[i]
        return self._path_world[-1]

    def _cost_at(self, x, z):
        ix, iz = self._cell(x, z)
        if not self._inb(ix, iz):
            return 1e6
        return float(self._cost[ix, iz])

    # ---- DWA/DWB local controller ----
    def dwa_command(self, pos, yaw, v_prev):
        if self._cost is None:
            self.cost_field()
        carrot = self._carrot(pos)
        best, best_cmd = -1e18, (0.0, 0.0)
        n_steps = int(self.horizon / self.dt)
        for v in np.linspace(0.0, self.max_speed, 6):
            for w in np.linspace(-self.max_yaw, self.max_yaw, 11):
                x, z, th = pos[0], pos[2], yaw
                worst, hit = 0.0, False
                for _ in range(n_steps):
                    th += w * self.dt
                    x += -np.sin(th) * v * self.dt          # forward = [-sin,_,-cos]
                    z += -np.cos(th) * v * self.dt
                    c = self._cost_at(x, z)
                    if c >= 1e5:
                        hit = True; break
                    worst = max(worst, c)
                if hit:
                    continue
                # progress toward the carrot + speed reward - inflation cost
                end = np.array([x, z]); p = np.array([pos[0], pos[2]])
                progress = np.linalg.norm(carrot - p) - np.linalg.norm(carrot - end)
                # heading alignment to carrot at the end pose
                to_c = carrot - end
                want = np.arctan2(-to_c[0], -to_c[1])       # world azimuth (matches yaw)
                align = np.cos(want - th)
                score = 2.0 * progress + 0.6 * align + 0.4 * v - 0.05 * worst
                if score > best:
                    best, best_cmd = score, (float(v), float(w))
        if best <= -1e17:
            # boxed in -> rotate-in-place recovery toward the carrot
            to_c = carrot - np.array([pos[0], pos[2]])
            want = np.arctan2(-to_c[0], -to_c[1])
            return 0.0, float(np.clip(np.arctan2(np.sin(want - yaw), np.cos(want - yaw)),
                                      -self.max_yaw, self.max_yaw))
        return best_cmd
