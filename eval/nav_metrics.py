"""Navigation-relevant metrics for MERLIN's M1 method (task-aware allocation).

The misalignment thesis: accuracy-optimal distillation spends capacity matching
the teacher everywhere (incl. far-field texture control never consumes), while a
robot's local planner only consumes NEAR-FIELD obstacle geometry. These metrics
score what control actually uses, so they can show nav-weighted distillation (M1)
beating uniform reconstruction at the same budget.

All depths are metric Z-depth (m), shape (H, W). "near" = within `near_m`.
"""
import numpy as np


def _valid(a, b, min_d=0.1, max_d=10.0):
    return np.isfinite(a) & np.isfinite(b) & (a > min_d) & (a < max_d) & (b > min_d)


def near_far_absrel(ref_depth, pred_depth, near_m=2.0):
    """abs_rel split into near-field (obstacles) and far-field (background).

    ref is the target (teacher or GT). Returns dict; near is what nav uses.
    """
    v = _valid(ref_depth, pred_depth)
    near = v & (ref_depth < near_m)
    far = v & (ref_depth >= near_m)
    def ar(m):
        if m.sum() < 50:
            return float("nan")
        r, p = ref_depth[m], pred_depth[m]
        return float(np.mean(np.abs(r - p) / r))
    return {"near_absrel": ar(near), "far_absrel": ar(far),
            "near_frac": float(near.sum()) / max(1, v.sum())}


def column_range_mae(ref_depth, pred_depth, near_m=4.0):
    """Per-column nearest-obstacle range error -- a 2D-laser-scan proxy.

    For each image column the closest valid depth is the obstacle range in that
    bearing, which is exactly what a local costmap / DWA planner consumes. MAE of
    pred vs ref over columns where ref sees an obstacle within near_m. Lower=better.
    """
    H, W = ref_depth.shape
    rv = np.where(np.isfinite(ref_depth) & (ref_depth > 0.1), ref_depth, np.inf)
    pv = np.where(np.isfinite(pred_depth) & (pred_depth > 0.1), pred_depth, np.inf)
    r_min = rv.min(axis=0)   # (W,)
    p_min = pv.min(axis=0)
    cols = np.isfinite(r_min) & (r_min < near_m)
    if cols.sum() < 5:
        return float("nan")
    pm = np.where(np.isfinite(p_min[cols]), p_min[cols], near_m)
    return float(np.mean(np.abs(r_min[cols] - pm)))


def obstacle_iou(ref_depth, pred_depth, near_m=2.0):
    """IoU of the near-field obstacle mask (depth < near_m). Free-space accuracy."""
    v = np.isfinite(ref_depth) & np.isfinite(pred_depth) & (ref_depth > 0.1) & (pred_depth > 0.1)
    r = v & (ref_depth < near_m)
    p = v & (pred_depth < near_m)
    inter = (r & p).sum(); union = (r | p).sum()
    return float(inter / union) if union > 0 else float("nan")


def nav_metrics(ref_depth, pred_depth, near_m=2.0):
    out = near_far_absrel(ref_depth, pred_depth, near_m)
    out["col_range_mae"] = column_range_mae(ref_depth, pred_depth, near_m=2 * near_m)
    out["obstacle_iou"] = obstacle_iou(ref_depth, pred_depth, near_m)
    return {k: (round(v, 4) if np.isfinite(v) else v) for k, v in out.items()}
