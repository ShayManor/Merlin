"""Depth / metric-scale accuracy metrics for MERLIN.

Used two ways:
  - student-vs-teacher fidelity (how well the distilled/quantized student
    reproduces the teacher's metric depth), and
  - vs ground-truth depth (TUM/7-Scenes/ScanNet++), for the metric claims C2/C3.

All depth inputs are metric Z-depth maps in meters, shape (H, W).
"""
import numpy as np


def _valid(gt, pred, mask=None, min_d=1e-3, max_d=80.0):
    v = np.isfinite(gt) & np.isfinite(pred) & (gt > min_d) & (gt < max_d) & (pred > min_d)
    if mask is not None:
        v &= mask.astype(bool)
    return v


def depth_metrics(gt, pred, mask=None):
    """Standard monocular depth metrics. gt/pred are metric depth (m)."""
    gt = np.asarray(gt, np.float64)
    pred = np.asarray(pred, np.float64)
    v = _valid(gt, pred, mask)
    if v.sum() == 0:
        return {k: float("nan") for k in
                ("abs_rel", "sq_rel", "rmse", "rmse_log", "delta1", "delta2", "delta3")}
    g, p = gt[v], pred[v]
    thresh = np.maximum(g / p, p / g)
    return {
        "abs_rel": float(np.mean(np.abs(g - p) / g)),
        "sq_rel": float(np.mean((g - p) ** 2 / g)),
        "rmse": float(np.sqrt(np.mean((g - p) ** 2))),
        "rmse_log": float(np.sqrt(np.mean((np.log(g) - np.log(p)) ** 2))),
        "delta1": float(np.mean(thresh < 1.25)),
        "delta2": float(np.mean(thresh < 1.25 ** 2)),
        "delta3": float(np.mean(thresh < 1.25 ** 3)),
        "n_valid": int(v.sum()),
    }


def scale_error(gt, pred, mask=None):
    """Absolute metric-scale error (claim C2: <5%).

    Estimates the single global scale that best aligns pred to gt (median ratio),
    then reports |scale - 1|. A model with correct metric scale gives ~0.
    """
    gt = np.asarray(gt, np.float64)
    pred = np.asarray(pred, np.float64)
    v = _valid(gt, pred, mask)
    if v.sum() == 0:
        return float("nan")
    ratio = np.median(gt[v] / pred[v])
    return float(abs(ratio - 1.0))


def fidelity_vs_teacher(teacher_depth, student_depth, mask=None):
    """How close the student is to the teacher (no GT needed).

    Reports abs_rel against the teacher and the scale offset between them.
    """
    m = depth_metrics(teacher_depth, student_depth, mask)
    m["scale_offset"] = scale_error(teacher_depth, student_depth, mask)
    return m
