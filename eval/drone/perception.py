#!/usr/bin/env python3
"""Perception wrapper for the drone closed-loop sim: RGB frame -> metric z-depth at a
chosen operating point (input resolution), using a distilled MERLIN student.

Mirrors eval/sim/perception.py (same MapAnything load_images preprocessing + student
forward, so drone perception is byte-identical to the offline evals), with one change:
metric scale comes from the IMU visual-inertial solve (eval/drone/vi.vi_scale), not a
per-episode GT-depth align. set_scale() lets a caller inject a known scale (e.g. a GT
upper-bound arm); calibrate_from_imu() recovers it from the first keyframes + PX4 IMU.
"""
import os
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "model", "distill"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "model", "backbones"))
from student import build_student  # noqa: E402


def decompress_steps(depth, k=41, alpha=0.18, cap=0.35):
    """Invert the student's measured depth-step compression (a calibration, not a retrain).

    Measured on held-out frames: inside an object's silhouette the student recovers only
    ~82% of the background-to-object depth step, i.e. pred = gt + alpha*(bg - gt) with
    alpha ~= 0.18. Because the error is proportional to the LOCAL STEP it vanishes on a flat
    wall (where bg == gt) and is largest for a small object against a far background --
    which is exactly the non-uniformity that made a single global safety margin unable to
    cover both. Estimating bg by a local maximum over a k x k window and solving for gt:

        gt = (pred - alpha * bg) / (1 - alpha)

    On a flat surface bg == pred and this is the identity, so unlike a global margin or a
    uniform depth rescale it costs nothing where the student was already right. k must be
    wide enough to reach past a typical obstacle to the surface behind it.

    `cap` bounds the correction, and is not optional: alpha was fitted on obstacle silhouettes
    whose depth step is ~1-1.5 m, but a window straddling a doorway or an open corridor sees a
    step of many metres, and the unbounded form drove those pixels to zero -- they then fell
    out of the map entirely and the drone hit the obstacle in 5 of 5 episodes. The measured
    displacement is ~0.27 m, so the correction is clamped just above it.
    """
    if not k or alpha <= 0:
        return depth
    import cv2
    d = np.asarray(depth, np.float32)
    bg = cv2.dilate(d, np.ones((int(k), int(k)), np.uint8))
    corr = np.minimum(float(alpha) * (bg - d) / (1.0 - float(alpha)), float(cap))
    return np.maximum(d - corr, 0.05)


def sharpen_edges(depth, k):
    """Undo the student's over-smoothing of depth discontinuities (a calibration, not a model
    change).

    Measured on held-out frames: inside an injected obstacle's silhouette the student recovers
    only ~82% of the background-to-object depth step, so it places the object's near face
    ~0.27 m too far and the planner keeps its margin from a surface that is not there. A local
    MINIMUM over a k x k window restores it: on a flat wall the neighbourhood minimum is the
    centre value, so the transform is near-identity, and it only bites where depth changes
    quickly -- which is exactly where the smoothing error lives. The cost is that obstacles
    dilate by k/2 pixels, so k is chosen offline against phantom-blockage, not by eye.
    """
    if not k or k < 2:
        return depth
    import cv2
    return cv2.erode(np.asarray(depth, np.float32), np.ones((int(k), int(k)), np.uint8))


class Perception:
    def __init__(self, ckpt, device="cuda", aat_depth=None):
        import cv2  # noqa
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.model = build_student(size=ck.get("size", "base"),
                                   aat_depth=aat_depth or ck.get("aat_depth", 8),
                                   device="cpu")
        self.model.load_state_dict(ck["state_dict"])
        self.model = self.model.to(device).eval()
        self.device = device
        self.scale = 1.0  # set by set_scale() or calibrate_from_imu()
        self.calib_tab = None  # set by set_depth_calib()
        self.calib_q = 0.20
        self._tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        from mapanything.utils.image import load_images
        self._load_images = load_images

    @torch.no_grad()
    def _forward(self, rgb, res, aat_depth=None):
        """rgb: (H,W,3) uint8 RGB. Returns (up-to-scale z-depth, cam_trans, cam_quats) from
        one student forward. cam_trans/cam_quats are the student's per-frame visual pose
        (up-to-scale translation, xyzw rotation) used for the C2/C3 visual-inertial chain."""
        import cv2
        cv2.imwrite(self._tmp, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        s = (res // 14) * 14
        v = self._load_images([self._tmp], resize_mode="square", size=s)
        v[0]["img"] = v[0]["img"].to(self.device)
        if aat_depth is not None:
            self.model.info_sharing.depth = aat_depth
        pr = self.model(v, memory_efficient_inference=True, minibatch_size=1)[0]
        depth = pr["pts3d_cam"][0, ..., 2].float().cpu().numpy()
        cam_trans = pr["cam_trans"][0].float().cpu().numpy()
        cam_quats = pr["cam_quats"][0].float().cpu().numpy()
        return depth, cam_trans, cam_quats

    def _raw_depth(self, rgb, res, aat_depth=None):
        """rgb: (H,W,3) uint8 RGB. Returns up-to-scale z-depth at student native size."""
        return self._forward(rgb, res, aat_depth)[0]

    def depth_and_pose(self, rgb, res=378, aat_depth=None):
        """Metric depth + the student visual pose (cam_trans up-to-scale, cam_quats xyzw)
        for one frame. Used by the C2/C3 driver to log a visual trajectory alongside GT."""
        depth, cam_trans, cam_quats = self._forward(rgb, res, aat_depth)
        return depth * self.scale, cam_trans, cam_quats

    @torch.no_grad()
    def forward_window(self, rgbs, res=378):
        """Run the student on a WINDOW of frames (list of (H,W,3) uint8 RGB) in one forward.
        MapAnything estimates each frame's pose in the window's c0 (first-frame) frame, so a
        single window yields a short up-to-scale visual trajectory. Single-frame inference
        gives no useful relative pose, so C2 (scale) and C3 (drift) run windowed like
        eval/vi_scale.py. Returns (cam_trans (N,3) up-to-scale, cam_quats (N,4) xyzw)."""
        import cv2
        s = (res // 14) * 14
        paths = []
        for i, rgb in enumerate(rgbs):
            p = f"{self._tmp[:-4]}_w{i}.png"
            cv2.imwrite(p, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            paths.append(p)
        v = self._load_images(paths, resize_mode="square", size=s)
        for x in v:
            x["img"] = x["img"].to(self.device)
        out = self.model(v, memory_efficient_inference=True, minibatch_size=len(v))
        cam_trans = np.array([o["cam_trans"][0].float().cpu().numpy() for o in out])
        cam_quats = np.array([o["cam_quats"][0].float().cpu().numpy() for o in out])
        return cam_trans, cam_quats

    def set_depth_calib(self, path, q=0.20):
        """Load a fitted per-pixel apparent->real depth table (eval/drone/depth_calib).

        `q` is the quantile of gt/pred the table returns, and it is the difference between a
        correction that helps and one that hurts: q=0.5 is the minimum-error fit, which is
        defeated by regression dilution, while a low q asks how close the surface could be and
        so pulls in exactly the cells where the student is unreliable.
        """
        from eval.drone.depth_calib import DepthCalib
        self.calib_tab = DepthCalib.load(path) if path else None
        self.calib_q = float(q)
        return self.calib_tab

    def apply_depth_calib(self, depth):
        """Metric depth -> calibrated depth; identity if no table was loaded."""
        if self.calib_tab is None:
            return depth
        return self.calib_tab.apply(depth, self.calib_q)

    def set_scale(self, s):
        """Inject a known metric scale (e.g. a GT-scale upper-bound arm)."""
        self.scale = float(s)
        return self.scale

    def calibrate_gt(self, rgb, gt_depth, res=378):
        """Set the per-episode metric scale from the median GT/pred ratio (the 'gt-scale
        student' arm, which isolates depth-shape quality from the scale problem; the IMU
        path in calibrate_from_imu is the C2 alternative). Mirrors eval/sim/perception."""
        import cv2
        pred = self._raw_depth(rgb, res)
        if pred.shape != gt_depth.shape:
            gt = cv2.resize(gt_depth, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            gt = gt_depth
        v = np.isfinite(gt) & np.isfinite(pred) & (gt > 0.1) & (gt < 10) & (pred > 0.05)
        if v.sum() > 200:
            self.scale = float(np.median(gt[v] / pred[v]))
        return self.scale

    def calibrate_from_imu(self, P_vis, R_list, kf_t, imu_t, imu_acc):
        """Recover the metric scale (C2) from the visual keyframe trajectory + PX4 IMU
        via the visual-inertial linear solve. Keeps the previous scale if the solve is
        degenerate (non-finite or non-positive)."""
        from eval.drone.vi import vi_scale
        s = vi_scale(P_vis, R_list, kf_t, imu_t, imu_acc)
        if np.isfinite(s) and s > 0:
            self.scale = float(s)
        return self.scale

    def depth_gt_scaled(self, rgb, gt_depth, res=378):
        """Depth with the metric scale re-fitted against GT on THIS frame.

        An oracle diagnostic, not a deployable arm. calibrate_gt fixes one scale for the whole
        episode; measured offline, re-fitting per frame cuts median depth error 0.270 -> 0.205 m
        (24%). Running it closed-loop separates the two ways the student can be wrong: if this
        navigates well, what limits MERLIN is metric SCALE (which the IMU chain is meant to
        supply on the robot), and if it does not, the limit is depth SHAPE and no scale
        estimator will help.
        """
        import cv2
        pred = self._raw_depth(rgb, res)
        gt = gt_depth
        if pred.shape != gt.shape:
            gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
        v = np.isfinite(gt) & np.isfinite(pred) & (gt > 0.1) & (gt < 10) & (pred > 0.05)
        if v.sum() < 200:
            return pred * self.scale
        return pred * float(np.median(gt[v] / pred[v]))

    def depth(self, rgb, res=378, aat_depth=None, calib=0):
        """Metric z-depth (meters) at the chosen operating point, scaled by the
        recovered/injected metric scale. `calib` applies the edge-sharpening calibration
        (see sharpen_edges). Returns (H,W) float."""
        d = self.apply_depth_calib(self._raw_depth(rgb, res, aat_depth) * self.scale)
        return sharpen_edges(d, calib)
