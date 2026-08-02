"""Greedy 3D safe-direction planner. Samples directions on a unit sphere, rejects any whose
short segment passes within `safety` of an occupied voxel, and picks the safe candidate that
most reduces distance to goal. Deliberately reactive and conservative (no global A*): the C4
claim is 'live onboard map only', and reactivity is what the staleness ablation stresses.
"""
import numpy as np

def _fib_sphere(n=64):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2*i/n)
    theta = np.pi * (1 + 5**0.5) * i
    return np.stack([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)], 1)

class Planner3D:
    def __init__(self, step=0.4, safety=0.5, max_speed=1.0, n_dirs=64, escape=True,
                 esc_bonus=0.6, require_known=False, min_clear=0.27):
        self.step = float(step); self.safety = float(safety)
        self.max_speed = float(max_speed); self.dirs = _fib_sphere(n_dirs)
        # When no direction reduces the distance to goal, a purely greedy planner hovers
        # until timeout -- an obstacle wider than it is deep is a local minimum it can never
        # leave. Escape mode takes the least-bad safe direction instead, and commits to it
        # (esc_bonus rewards agreement with the previous escape step) so the drone slides
        # consistently around one side rather than oscillating in place.
        self.escape = bool(escape); self.esc_bonus = float(esc_bonus)
        self.esc_look = 3.0            # escape lookahead, in units of `step`
        self.esc_flip = 1.2            # metres of fruitless sliding before trying the far side
        # Optional: only fly where the camera has looked. This is the principled reading of
        # a 90 deg forward FOV -- an unobserved voxel is unknown, not empty. Measured on the
        # real loop it is far too conservative: the drone spends 64% of ticks blocked and
        # scores 0/5 on routes it otherwise flies cleanly, so it is off by default and kept
        # as an ablation.
        self.require_known = bool(require_known)
        self.min_clear = float(min_clear)   # robot radius + voxel*sqrt(3)/2
        self._esc = None
        self._banned = None            # side already tried and abandoned
        self._esc_dist = 0.0
        self._last = None

    def next_setpoint(self, vmap, cur, goal):
        cur = np.asarray(cur, float); goal = np.asarray(goal, float)
        to_goal = goal - cur
        dist = np.linalg.norm(to_goal)
        if dist < 1e-6:
            return cur.copy(), False
        # bias sampling toward goal: include the straight-at-goal direction first, and the
        # committed escape direction second so it can be re-tested exactly
        head = [to_goal/dist] if self._esc is None else [to_goal/dist, self._esc]
        cand = np.vstack(head + [self.dirs])
        step = min(self.step, dist)
        tgts = cur + cand * step
        # every candidate segment is sampled and clearance-tested in one vectorised query;
        # per-candidate Python looping here dominated the control tick.
        n = max(2, int(step / (vmap.voxel*0.5)))
        s = np.linspace(0, 1, n)

        def seg_blocked(reach):
            ends = cur + cand * reach
            p = (cur + (ends - cur)[:, None, :] * s[None, :, None]).reshape(-1, 3)
            bad = vmap.any_occupied_within(p, self.safety)
            if self.require_known:
                bad = bad | vmap.unknown(p)
            return bad.reshape(len(cand), n).any(1)

        blocked = seg_blocked(step)
        cost = np.linalg.norm(goal - tgts, axis=1)
        cost[blocked] = np.inf
        i = int(np.argmin(cost))
        # While escaping, a move back toward the goal is almost always available (sliding
        # sideways makes the goal direction improvable again) and taking it walks straight
        # back into the obstacle. So escape is left only once the goal direction itself is
        # clear, as in the classic bug algorithms -- otherwise the two modes alternate and
        # the drone jitters on the safety boundary until timeout.
        if np.isfinite(cost[i]) and cost[i] < dist and (self._esc is None or not blocked[0]):
            self._reset_escape()
            return tgts[i], False
        if not np.isfinite(cost).any():
            # Already inside the safety margin, so every segment is blocked at its first
            # sample -- the drone's own position. Holding here is what turns a near miss into
            # a crash: it cannot retreat, and any residual velocity carries it in. Back out
            # to whichever endpoint recovers the most clearance, relaxing the requirement
            # until something is reachable.
            self._reset_escape()
            known = ~vmap.unknown(tgts) if self.require_known else np.ones(len(tgts), bool)
            # Never relax below the body radius plus voxel quantisation: a retreat that
            # accepts less clearance than the drone physically occupies is just a collision
            # with extra steps.
            floor = max(self.min_clear, 0.5 * self.safety)
            for rad in (self.safety, 0.5 * (self.safety + floor), floor):
                free = (~vmap.any_occupied_within(tgts, rad)) & known
                if free.any():
                    c = np.where(free, np.linalg.norm(goal - tgts, axis=1), np.inf)
                    return tgts[int(np.argmin(c))], True
            return cur.copy(), True
        if not self.escape:
            self._reset_escape()
            return cur.copy(), True
        # Which side of an obstacle is open is not observable until the drone has looked
        # down it, so a first choice is a guess. Track how far the slide has gone and, if it
        # has not opened the goal direction, ban that side and commit to the other one --
        # otherwise a guess into the pocket between an obstacle and a wall ends in a crash.
        if self._last is not None:
            self._esc_dist += float(np.linalg.norm(cur - self._last))
        self._last = cur.copy()
        if self._esc is not None and self._esc_dist > self.esc_flip:
            # Alternate: ban the side just tried, but drop the ban on the next flip. Holding
            # a ban indefinitely deadlocks when neither side works on the first pass and the
            # map has since changed.
            self._banned = self._esc if self._banned is None else None
            self._esc = None
            self._esc_dist = 0.0
            cand = np.vstack([to_goal/dist, self.dirs])
            tgts = cur + cand * step
            blocked = seg_blocked(step)
            cost = np.linalg.norm(goal - tgts, axis=1)
            cost[blocked] = np.inf
        # Local minimum: slide along the obstacle instead of hovering. Getting past an
        # obstacle wider than the safety radius needs many consecutive sideways steps, none
        # of which reduce the distance to goal, so the committed direction is held until it
        # is blocked -- re-optimising every tick just oscillates in place.
        if self._esc is not None and not blocked[1]:
            return tgts[1], True
        # Pick the escape direction by how far it stays clear, not by whether it clears the
        # next 0.4 m. Both sides of an obstacle look identical one step out, so a cost-only
        # choice commits to whichever the candidate ordering happens to list first -- and
        # half the time that is the dead end between the obstacle and the wall.
        m = max(2, int(self.esc_look * step / vmap.voxel))
        reach = (np.arange(m) + 1) * (self.esc_look * step / m)
        probe = (cur + cand[:, None, :] * reach[None, :, None]).reshape(-1, 3)
        occ = vmap.any_occupied_within(probe, self.safety).reshape(len(cand), m)
        # free run before the first blocked sample, in metres
        free = np.where(occ.any(1), occ.argmax(1), m) * (self.esc_look * step / m)
        score = -free + 0.25 * cost / max(dist, 1e-6)
        if self._esc is not None:
            score = score - self.esc_bonus * (cand @ self._esc)
        if self._banned is not None:
            score = score + self.esc_bonus * (cand @ self._banned)
        score[blocked] = np.inf
        if not np.isfinite(score).any():
            score = cost
        j = int(np.argmin(score))
        self._esc = cand[j] / max(float(np.linalg.norm(cand[j])), 1e-9)
        return tgts[j], True

    def _reset_escape(self):
        self._esc = None
        self._banned = None
        self._esc_dist = 0.0
        self._last = None
