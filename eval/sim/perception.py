#!/usr/bin/env python3
"""Perception wrapper for the closed-loop sim: RGB frame -> metric z-depth at a
chosen operating point (input resolution), using a distilled MERLIN student.

Preprocessing goes through MapAnything's load_images (square resize, dinov2 norm)
by writing the in-memory RGB to a tmp PNG -- this keeps the closed-loop perception
byte-identical to the offline evals (eval/m2_deadline.py, eval/nav_collision.py), so
the M1 offline finding transfers without a preprocessing confound.

Metric scale: the student is up-to-scale mono. We scale-align to the sim's GT depth
ONCE per episode start (stands in for the IMU module that is not built yet). This is
the documented scale assumption (see CLAUDE.md: scale ambiguity is the silent failure).
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

# Measured Jetson Orin Nano bf16 latency per resolution (ms) -- the M2 operating points.
LATENCY_MS = {252: 84.0, 378: 147.0, 518: 189.0}
# GPU-rail watts per resolution on the Nano (README deployment table).
WATTS = {252: 7.4, 378: 8.5, 518: 8.8}
RES_LIST = [252, 378, 518]


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
        self.scale = 1.0  # set by calibrate()
        self._tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        from mapanything.utils.image import load_images
        self._load_images = load_images

    @torch.no_grad()
    def _raw_depth(self, rgb, res, aat_depth=None):
        """rgb: (H,W,3) uint8 RGB. Returns up-to-scale z-depth at student native size."""
        import cv2
        cv2.imwrite(self._tmp, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        s = (res // 14) * 14
        v = self._load_images([self._tmp], resize_mode="square", size=s)
        v[0]["img"] = v[0]["img"].to(self.device)
        if aat_depth is not None:
            self.model.info_sharing.depth = aat_depth
        pr = self.model(v, memory_efficient_inference=True, minibatch_size=1)[0]
        return pr["pts3d_cam"][0, ..., 2].float().cpu().numpy()

    def calibrate(self, rgb, gt_depth, res=378):
        """Set the per-episode metric scale from the median GT/pred ratio (IMU stand-in)."""
        pred = self._raw_depth(rgb, res)
        import cv2
        if pred.shape != gt_depth.shape:
            gt = cv2.resize(gt_depth, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            gt = gt_depth
        v = np.isfinite(gt) & np.isfinite(pred) & (gt > 0.1) & (gt < 10) & (pred > 0.05)
        if v.sum() > 200:
            self.scale = float(np.median(gt[v] / pred[v]))
        return self.scale

    def depth(self, rgb, res=378, aat_depth=None):
        """Metric z-depth (meters) at the chosen operating point, scaled by the
        per-episode calibration. Returns (H,W) float."""
        return self._raw_depth(rgb, res, aat_depth) * self.scale
