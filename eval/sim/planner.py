#!/usr/bin/env python3
"""Local planner: a geodesic-follower with a perceived-obstacle veto.

The global guide (navmesh ShortestPath) supplies goal_bearing = direction to the next
navigable waypoint, so heading toward it is collision-free on the TRUE geometry. The
local planner's only job is to react to PERCEIVED obstacles: drive toward goal_bearing
unless the perceived range in that direction is inside the braking bubble, in which case
sidestep to the nearest clear heading (or stop+rotate if boxed in). This keeps perception
on the critical path -- a student that under-sees a near obstacle won't veto and will
collide; a stale map vetoes late -- while not letting side walls in a tight apartment
stall a reactive controller. Deliberately simple: the experiment is about perception,
not planner cleverness.
"""
import numpy as np


class LocalPlanner:
    def __init__(self, max_speed=0.5, max_yaw_rate=1.6, brake_dist=0.45,
                 robot_radius=0.18, d_ref=1.5):
        self.max_speed = max_speed
        self.max_yaw_rate = max_yaw_rate
        self.brake_dist = brake_dist
        self.robot_radius = robot_radius
        self.d_ref = d_ref

    def command(self, goal_bearing, bearings, ranges):
        """goal_bearing: rad, robot frame (0 ahead, + left). Returns (v, yaw_rate, info)."""
        safe = self.brake_dist + self.robot_radius
        clear = ~np.isfinite(ranges) | (ranges > safe)        # inf range => clear
        gb = float(np.clip(goal_bearing, bearings.min(), bearings.max()))
        if clear.any():
            cand = bearings[clear]
            heading = float(cand[np.argmin(np.abs(cand - gb))])  # clear bin nearest goal
            blocked = False
        else:
            heading = 0.0; blocked = True

        # forward range in a cone around the chosen heading -> speed
        cone = np.abs(bearings - heading) <= 0.22
        fr = ranges[cone]
        fr = fr[np.isfinite(fr)]
        fwd_range = float(np.min(fr)) if fr.size else self.d_ref * 3

        if blocked or fwd_range <= safe:
            # boxed in or wall straight ahead: rotate toward the clearest opening (biased
            # to goal) to find a way out instead of spinning at a blocked goal bearing.
            score = np.where(np.isfinite(ranges), ranges, self.d_ref * 3) \
                + 0.5 * np.cos(bearings - gb)
            target = float(bearings[int(np.argmax(score))])
            yaw = float(np.clip(target * 2.5, -self.max_yaw_rate, self.max_yaw_rate))
            return 0.0, yaw, {"blocked": True, "fwd_range": fwd_range}

        speed = self.max_speed * float(np.clip((fwd_range - safe) / self.d_ref, 0.0, 1.0))
        speed *= float(np.clip(1.0 - abs(heading) / (np.pi / 2), 0.35, 1.0))  # ease in turns
        yaw = float(np.clip(heading * 2.5, -self.max_yaw_rate, self.max_yaw_rate))
        return speed, yaw, {"blocked": False, "heading": heading, "fwd_range": fwd_range}
