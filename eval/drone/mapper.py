"""Lift metric depth to a world point cloud and hash into a 3D voxel occupancy set.

Optical camera frame is +Z forward, +X right, +Y down (standard pinhole). integrate() takes
R_cam2world (columns = optical axes in world) and the camera world position. A voxel is
'occupied' once >= min_hits keyframes have seen a surface there net of the frames that saw
through it, so occupancy is a multi-view consensus rather than a union of every frame's noise.
"""
import numpy as np

def backproject(depth, K, max_range=8.0):
    H, W = depth.shape
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    z = depth.reshape(-1)
    u = us.reshape(-1); v = vs.reshape(-1)
    good = np.isfinite(z) & (z > 0.05) & (z < max_range)
    z = z[good]; u = u[good]; v = v[good]
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    return np.stack([x, y, z], 1)   # (N,3) optical/camera frame

class VoxelMap:
    """Voxels are hashed by a single int64 code (grid index packed base-S) rather than a tuple,
    so integration and the planner's clearance queries are numpy-vectorised: a keyframe lifts
    ~1e5 points and the planner runs ~1e3 sphere queries per control tick, and per-point Python
    is far too slow for a closed loop."""

    def __init__(self, voxel=0.15, bound=20.0, min_hits=2, carve_w=1, cap=6, hit_w=1):
        self.voxel = float(voxel); self.bound = float(bound)
        self.min_hits = int(min_hits); self.carve_w = int(carve_w); self.cap = int(cap)
        # Hit and miss evidence are weighted separately, as in a log-odds occupancy grid: a
        # ray that terminates ON a surface is stronger evidence than one that appears to pass
        # through it, because depth error at an object boundary sends rays past the object
        # onto whatever is behind it. With symmetric weights a handful of such rays erase an
        # obstacle that many good frames had already confirmed.
        self.hit_w = int(hit_w)
        self._hits = {}   # int-key -> count
        self._seen = set()  # voxel codes a camera ray has passed through or landed on
        self._B = int(np.ceil(self.bound / self.voxel)) + 2
        self._S = 2 * self._B + 1
        self._codes_cache = None
        self._off_cache = {}

    def _encode(self, keys):
        """(N,3) int grid indices -> (N,) int64 codes; -1 for out-of-grid keys."""
        q = keys + self._B
        bad = (q < 0).any(1) | (q >= self._S).any(1)
        code = (q[:, 0] * self._S + q[:, 1]) * self._S + q[:, 2]
        code[bad] = -1
        return code

    def _decode(self, codes):
        c = np.asarray(codes, np.int64)
        z = c % self._S; c = c // self._S
        y = c % self._S; x = c // self._S
        return np.stack([x, y, z], 1) - self._B

    def _key(self, p):
        q = np.round(np.asarray(p, float) / self.voxel).astype(np.int64)
        return int(self._encode(q.reshape(1, 3))[0])

    def integrate(self, depth, K, R_cam2world, t_cam_world, max_range=8.0, carve=True):
        pts_cam = backproject(depth, K, max_range)
        if pts_cam.shape[0] == 0:
            return
        pts_world = (R_cam2world @ pts_cam.T).T + np.asarray(t_cam_world, float)
        pts_world = pts_world[np.max(np.abs(pts_world), axis=1) <= self.bound]
        if pts_world.shape[0] == 0:
            return
        keys = np.round(pts_world / self.voxel).astype(np.int64)
        codes = np.unique(self._encode(keys))
        self._seen.update(codes.tolist())
        # One vote per KEYFRAME, not per point. A 378x378 depth map drops ~1e5 points into
        # ~2e4 voxels, so counting points made every voxel any single frame touched instantly
        # occupied and outvoted per-frame free-space carving ~100:1. The map then held the
        # UNION of every frame's depth noise instead of the frames' consensus: measured
        # offline, the student's map carried 4x the GT map's voxels and declared 42-62% of a
        # collision-free navmesh route blocked (GT: 0%), while missing almost no real surface.
        for c in codes.tolist():
            h = self._hits.get(c, 0) + self.hit_w
            self._hits[c] = h if h < self.cap else self.cap
        if carve and self.carve_w:
            self._carve(pts_world, np.asarray(t_cam_world, float), codes)
        self._codes_cache = None

    def _carve(self, pts_world, origin, hit_codes, stride=8):
        """Clear the free space between the camera and each measured surface.

        A hits-only map is monotone: one bad depth reading occupies a voxel forever. The
        distilled student's dominant error is phantom NEAR depth, so without carving its map
        fills with obstacles that are never retracted and the planner has nowhere to go --
        no occupancy threshold fixes it, because the phantoms accumulate. Rays are
        subsampled (stride) because carving only has to outvote the phantoms, not resolve
        surfaces."""
        p = pts_world[::stride]
        d = p - origin
        r = np.linalg.norm(d, axis=1)
        good = r > 2.0 * self.voxel
        if not good.any():
            return
        d, r = d[good], r[good]
        ss = (np.arange(int(np.ceil(r.max() / self.voxel))) + 0.5) * self.voxel
        q = origin + (d / r[:, None])[:, None, :] * ss[None, :, None]
        m = ss[None, :] < (r[:, None] - 1.5 * self.voxel)     # stop short of the surface
        keys = np.round(q[m] / self.voxel).astype(np.int64)
        codes = np.setdiff1d(np.unique(self._encode(keys)), hit_codes)
        self._seen.update(codes.tolist())
        for c in codes.tolist():
            h = self._hits.get(c)
            if h is not None:
                self._hits[c] = h - self.carve_w
                if self._hits[c] <= 0:
                    del self._hits[c]

    def _occ_keys(self):
        return [k for k, c in self._hits.items() if c >= self.min_hits]

    def _occ_codes(self):
        """Sorted int64 codes of the occupied voxels (cached; invalidated by integrate)."""
        if self._codes_cache is None:
            ks = self._occ_keys()
            self._codes_cache = np.sort(np.array(ks, np.int64)) if ks else np.zeros(0, np.int64)
        return self._codes_cache

    def is_occupied(self, p_world):
        return self._hits.get(self._key(p_world), 0) >= self.min_hits

    def occupied_points(self):
        ks = self._occ_codes()
        if ks.size == 0:
            return np.zeros((0, 3))
        return self._decode(ks).astype(float) * self.voxel

    def any_occupied_within(self, pts, radius):
        """(Q,3) query points -> (Q,) bool: is an occupied voxel centre within `radius`?
        Looks up only the local neighbourhood in the hash, so cost is independent of map size
        (the planner calls this on every candidate segment sample, every control tick)."""
        pts = np.atleast_2d(np.asarray(pts, float))
        codes = self._occ_codes()
        if codes.size == 0:
            return np.zeros(len(pts), bool)
        off = self._offsets(radius)
        keys = np.round(pts / self.voxel).astype(np.int64)
        cand = keys[:, None, :] + off[None, :, :]                     # (Q,M,3)
        near = np.linalg.norm(cand * self.voxel - pts[:, None, :], axis=2) <= radius
        cc = self._encode(cand.reshape(-1, 3)).reshape(cand.shape[:2])
        i = np.searchsorted(codes, cc).clip(0, codes.size - 1)
        return (near & (codes[i] == cc)).any(1)

    def _offsets(self, radius):
        """Grid offsets whose voxel centre can lie within `radius` of a query point. The point
        sits up to voxel*sqrt(3)/2 from its own voxel centre, hence the padded cutoff."""
        k = round(float(radius), 4)
        off = self._off_cache.get(k)
        if off is None:
            r = int(np.ceil(radius / self.voxel)) + 1
            g = np.arange(-r, r + 1)
            off = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
            off = off[np.linalg.norm(off, axis=1) * self.voxel <= radius + self.voxel * 0.8661]
            self._off_cache[k] = off
        return off

    def nearest_obstacle_dist(self, p_world, max_search=3.0):
        pts = self.occupied_points()
        if pts.shape[0] == 0:
            return float('inf')
        d = np.linalg.norm(pts - np.asarray(p_world, float), axis=1)
        m = d.min()
        return float(m) if m <= max_search else float(max_search)

    def unknown(self, pts):
        """(Q,3) -> (Q,) bool: has no camera ray ever passed through this voxel?

        A hits-only map cannot distinguish empty from unobserved, so the planner happily
        flew into space behind and beside the drone that the forward camera had never seen --
        the measured cause of collisions where the nearest mapped voxel was still 0.3-0.5 m
        away. Free-space carving records what was actually swept, which makes the difference
        expressible.
        """
        pts = np.atleast_2d(np.asarray(pts, float))
        if not self._seen:
            return np.ones(len(pts), bool)
        codes = self._encode(np.round(pts / self.voxel).astype(np.int64))
        return np.array([c not in self._seen for c in codes.tolist()], bool)
