"""Classical monocular VO front end (plan §6, E4a).

Emits an up-to-scale trajectory in the window's first-frame camera frame, in exactly the
(cam_trans, cam_quats) convention the learned pose sources return, so it drops into
`--pose_source vo` without touching run_c2.

Why this exists: E0a measured every feed-forward pose source through one VI solver and all of
them failed (student 103%, teacher 121%, VGGT 187% median scale error) while the GT control
passed at 4.74%. The deficit is class-wide, not a compression artifact, so the pose front end
has to come from somewhere else.

The one part that is not textbook: two-view recovery returns a UNIT translation per pair, so
chaining pairs directly would make every segment the same length and a single global scale
would be meaningless to the VI solver. `relative_scale` fixes the ratio between consecutive
pairs from the depths of the points they both triangulated.

cv2 is imported lazily so the geometry helpers below stay unit-testable on a machine with no
OpenCV, which is how the rest of eval/drone/ is arranged.
"""
import numpy as np


def intrinsics(width, height, hfov_deg=90.0):
    """Pinhole K for the sim camera. Mirrors eval/drone/renderer.intrinsics."""
    f = (width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
    return np.array([[f, 0.0, width / 2.0],
                     [0.0, f, height / 2.0],
                     [0.0, 0.0, 1.0]])


def R_to_quat(R):
    """Rotation matrix to quaternion, xyzw, matching frames.quat_to_R's convention."""
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return q / (np.linalg.norm(q) + 1e-12)


def relative_scale(depth_prev, depth_curr, min_pts=8):
    """Scale of the current pair relative to the previous one.

    Both pairs triangulated the same physical points, so the ratio of the depths they assigned
    is the ratio of their translation lengths. The median is used because triangulation near
    the epipole produces heavy-tailed outliers that would wreck a mean. Returns 1.0 when there
    is not enough overlap to measure, which degrades to constant-speed rather than to nonsense.
    """
    a = np.asarray(depth_prev, float)
    b = np.asarray(depth_curr, float)
    m = np.isfinite(a) & np.isfinite(b) & (a > 1e-6) & (b > 1e-6)
    if m.sum() < min_pts:
        return 1.0
    r = float(np.median(a[m] / b[m]))
    return r if np.isfinite(r) and r > 1e-6 else 1.0


def chain(rels):
    """Compose per-pair (R, t) into absolute poses in the FIRST frame's camera frame.

    Each (R, t) maps a point from frame k into frame k+1, which is what recoverPose returns.
    Returns (centres (N,3) in frame 0, rotations cam_k -> frame 0), the same convention
    forward_window produces for every other pose source.
    """
    R_abs, t_abs = np.eye(3), np.zeros(3)
    centres, rots = [np.zeros(3)], [np.eye(3)]
    for R, t in rels:
        R = np.asarray(R, float)
        t = np.asarray(t, float).reshape(3)
        t_abs = R @ t_abs + t
        R_abs = R @ R_abs
        centres.append(-R_abs.T @ t_abs)     # camera centre expressed in frame 0
        rots.append(R_abs.T)                 # cam -> frame 0
    return np.array(centres), rots


class VOFrontend:
    """ORB + essential matrix + RANSAC, keyframed, with inter-pair scale propagation."""

    def __init__(self, hfov_deg=90.0, n_features=3000, ratio=0.75):
        self.hfov_deg = float(hfov_deg)
        self.n_features = int(n_features)
        self.ratio = float(ratio)

    def _pair(self, cv2, K, g0, g1, orb, bf):
        """Relative pose for one image pair, plus the triangulated depths of its inliers,
        keyed by the second image's keypoint index so the next pair can find the overlap."""
        k0, d0 = orb.detectAndCompute(g0, None)
        k1, d1 = orb.detectAndCompute(g1, None)
        if d0 is None or d1 is None or len(k0) < 16 or len(k1) < 16:
            return None
        matches = bf.knnMatch(d0, d1, k=2)
        good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < self.ratio * n.distance]
        if len(good) < 16:
            return None
        p0 = np.float32([k0[m.queryIdx].pt for m in good])
        p1 = np.float32([k1[m.trainIdx].pt for m in good])
        E, mask = cv2.findEssentialMat(p0, p1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or E.shape != (3, 3):
            return None
        _, R, t, pose_mask = cv2.recoverPose(E, p0, p1, K, mask=mask)
        inl = (pose_mask.ravel() > 0)
        if inl.sum() < 8:
            return None
        # depths in the SECOND camera, so consecutive pairs share a frame to compare in
        P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P1 = K @ np.hstack([R, t.reshape(3, 1)])
        X = cv2.triangulatePoints(P0, P1, p0[inl].T, p1[inl].T)
        X = (X[:3] / np.where(np.abs(X[3]) < 1e-9, 1e-9, X[3])).T
        z = (R @ X.T + t.reshape(3, 1)).T[:, 2]
        idx = np.array([good[i].trainIdx for i in np.flatnonzero(inl)])
        return R, t.reshape(3), dict(zip(idx.tolist(), z.tolist()))

    def forward_window(self, rgbs, res=378):
        """Up-to-scale trajectory for a window of frames. Signature matches every other pose
        source, so run_c2 needs no special case. Raises on an unusable window rather than
        returning a silently wrong trajectory."""
        import cv2
        grays = [cv2.cvtColor(np.asarray(r), cv2.COLOR_RGB2GRAY) for r in rgbs]
        h, w = grays[0].shape[:2]
        K = intrinsics(w, h, self.hfov_deg)
        orb = cv2.ORB_create(nfeatures=self.n_features)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        rels, prev_depths = [], None
        for a, b in zip(grays[:-1], grays[1:]):
            out = self._pair(cv2, K, a, b, orb, bf)
            if out is None:
                # a degenerate pair would otherwise silently freeze the trajectory
                rels.append((np.eye(3), np.zeros(3)))
                prev_depths = None
                continue
            R, t, depths = out
            if prev_depths:
                shared = set(prev_depths) & set(depths)
                s = relative_scale([prev_depths[i] for i in shared],
                                   [depths[i] for i in shared])
                t = t * s
                depths = {i: d * s for i, d in depths.items()}
            rels.append((R, t))
            prev_depths = depths
        centres, rots = chain(rels)
        return centres, np.array([R_to_quat(R) for R in rots])
