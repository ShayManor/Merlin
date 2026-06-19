"""Frame indexing + batched mono-view construction for MERLIN distillation.

On the A40 we do ONLINE distillation (teacher + student co-resident), so this
module just turns RGB frame paths into batched single-view inputs the model
accepts, plus a GT-depth loader for metric eval (TUM RGB-D).

A "batched mono view" is one view dict whose img is (B,3,H,W): the model treats
it as a single-view batch of B independent images, which is what we want for
monocular distillation (no cross-view info sharing).
"""
import glob
import os

import numpy as np
import torch

from mapanything.utils.image import load_images


def tum_pairs(seq_dir, max_dt=0.02):
    """Associate TUM rgb<->depth frames by nearest timestamp. Returns [(rgb, depth)]."""
    rgb = sorted(glob.glob(os.path.join(seq_dir, "rgb", "*.png")))
    dep = sorted(glob.glob(os.path.join(seq_dir, "depth", "*.png")))
    if not rgb or not dep:
        return []
    dts = np.array([float(os.path.basename(d)[:-4]) for d in dep])
    out = []
    for r in rgb:
        t = float(os.path.basename(r)[:-4])
        j = int(np.argmin(np.abs(dts - t)))
        if abs(dts[j] - t) < max_dt:
            out.append((r, dep[j]))
    return out


def list_tum_sequences(root):
    return sorted(glob.glob(os.path.join(root, "rgbd_dataset_*")))


def gather_frames(roots, stride=1, with_depth=False):
    """Collect frame paths across TUM sequences. Returns list of rgb paths, or
    list of (rgb, depth) if with_depth."""
    frames = []
    for root in roots:
        for seq in list_tum_sequences(root):
            pairs = tum_pairs(seq)
            for i, (r, d) in enumerate(pairs):
                if i % stride:
                    continue
                frames.append((r, d) if with_depth else r)
    return frames


def make_views(paths, size, device, norm_type="dinov2"):
    """Load a list of image paths into ONE batched mono view: img (B,3,H,W)."""
    imgs, shapes = [], []
    for p in paths:
        v = load_images([p], resize_mode="square", size=size, norm_type=norm_type)[0]
        imgs.append(v["img"])           # (1,3,H,W)
        shapes.append(v["true_shape"])  # (1,2)
    img = torch.cat(imgs, 0).to(device)
    true_shape = np.concatenate(shapes, 0)
    B = img.shape[0]
    return [dict(img=img, true_shape=true_shape, idx=0,
                 instance=[str(i) for i in range(B)],
                 data_norm_type=[norm_type] * B)]


def load_gt_depth(depth_path, H, W, divisor=5000.0):
    """TUM 16-bit depth PNG -> metric meters, center-square cropped + resized to HxW."""
    import cv2
    d = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float64) / divisor
    gh, gw = d.shape
    s = min(gh, gw)
    y0, x0 = (gh - s) // 2, (gw - s) // 2
    dc = d[y0:y0 + s, x0:x0 + s]
    return cv2.resize(dc, (W, H), interpolation=cv2.INTER_NEAREST)
