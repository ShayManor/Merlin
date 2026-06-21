#!/usr/bin/env python3
"""Turn a perceived depth image into the thing a local planner consumes: a forward
angular range scan ("fake 2D lidar") of OBSTACLES (floor and ceiling removed).

A level camera at robot height sees the floor in the lower rows; naively taking the
nearest depth there makes the ground look like an obstacle a metre ahead and the robot
crawls. So we back-project each pixel to 3D (camera level, no pitch), keep only points
whose world height is above the floor and below the robot, and reduce to a per-azimuth
nearest ground-plane range. This is the standard height-band costmap a 2D nav stack uses,
and it keeps the near-field obstacle band (what M1 targets) as exactly what drives control.
"""
import numpy as np


def _cam_rays(H, W, hfov_deg):
    """Per-pixel camera-frame ray slopes (x_cam, y_cam) for a square-FOV pinhole.
    z_cam = -1 (optical axis), up = +y. hfov is horizontal; vfov scaled by aspect."""
    th = np.tan(np.deg2rad(hfov_deg) / 2.0)
    u = ((np.arange(W) + 0.5) / W) * 2.0 - 1.0          # -1 left .. +1 right
    v = ((np.arange(H) + 0.5) / H) * 2.0 - 1.0          # -1 top .. +1 bottom
    aspect = H / W
    x_cam = u * th                                       # right+
    y_cam = -v * (th * aspect)                           # up+
    return x_cam, y_cam


def range_scan(depth, hfov_deg, n_bins=21, cam_height=0.88, h_floor=0.18,
               h_ceil=1.1, min_d=0.1, max_d=10.0):
    """depth: (H,W) metric z-depth. Returns (bearings_rad, ranges_m) of obstacles in
    the height band [h_floor, h_ceil] above ground. bearing 0 = ahead, + = left."""
    H, W = depth.shape
    x_cam, y_cam = _cam_rays(H, W, hfov_deg)
    X = depth * x_cam[None, :]                            # lateral (m), right+
    height = cam_height + depth * y_cam[:, None]          # world height (m)
    grange = depth * np.sqrt(x_cam[None, :] ** 2 + 1.0)   # ground-plane range (m)
    az = np.arctan(x_cam)                                 # azimuth per column, right+
    obst = (np.isfinite(depth) & (depth > min_d) & (depth < max_d)
            & (height > h_floor) & (height < h_ceil))
    # bearing convention: + = left, so flip azimuth sign (right+ -> left+)
    bearing_col = -az
    edges = np.linspace(-np.deg2rad(hfov_deg) / 2, np.deg2rad(hfov_deg) / 2, n_bins + 1)
    bearings = 0.5 * (edges[:-1] + edges[1:])
    ranges = np.full(n_bins, np.inf)
    bcol = np.broadcast_to(bearing_col[None, :], depth.shape)
    for b in range(n_bins):
        m = obst & (bcol >= edges[b]) & (bcol < edges[b + 1])
        if m.sum() >= 6:
            ranges[b] = float(np.min(grange[m]))
    return bearings, ranges


def min_forward_range(bearings, ranges, half_width_rad=0.30):
    """Nearest obstacle within a narrow forward cone (the braking-distance check)."""
    m = np.abs(bearings) <= half_width_rad
    r = ranges[m]
    r = r[np.isfinite(r)]
    return float(np.min(r)) if r.size else np.inf
