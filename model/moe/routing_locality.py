#!/usr/bin/env python3
"""Does a depth-range routing key give a smaller, more reusable expert working set than a
feature-driven one, on real video?

This is the measurement that decides whether "physically-grounded routing makes expert prefetch
exact" is worth pursuing as a method. The systems quantity is not accuracy, it is how many
DISTINCT experts a frame activates (that is what must be resident) and how much of the next
frame's set is already there (that is the prefetch hit rate).

Three routers over the 27x27 = 729 AAT token grid:
  range    hard assignment to equal-population log-depth quantile bins, i.e. PrescribedRangeGate
  appear   fixed random projection of patch RGB -> argmax. Stand-in for a feature-driven router.
  balanced iid uniform per token. Not a strawman: the Switch load-balancing loss explicitly
           optimises toward uniform expert utilisation, so this is what a trained sparse router
           is pushed to become, and it is the worst case for caching.

Strides map to deployment rates: TUM is 30 Hz, so stride 1/2/6 is 30/15/5 Hz.
"""
import argparse
import glob
import json
import os

import numpy as np

try:
    import cv2
    def _read16(p): return cv2.imread(p, cv2.IMREAD_UNCHANGED)
except ImportError:
    from PIL import Image
    def _read16(p): return np.array(Image.open(p))

GRID = 27          # AAT token grid: 378/14
TUM_SCALE = 5000.0


def load_depth(path, grid=GRID):
    """TUM depth PNG (uint16, mm*5) -> (grid, grid) metres, averaging valid pixels per patch."""
    d = _read16(path).astype(np.float32) / TUM_SCALE
    h, w = d.shape
    ph, pw = h // grid, w // grid
    d = d[:ph * grid, :pw * grid].reshape(grid, ph, grid, pw).transpose(0, 2, 1, 3)
    valid = d > 0
    n = valid.sum((2, 3))
    return np.where(n > 0, d.sum((2, 3)) / np.maximum(n, 1), 0.0)


def load_rgb(path, grid=GRID):
    im = _read16(path).astype(np.float32)
    if im.ndim == 2:
        im = im[..., None]
    h, w = im.shape[:2]
    ph, pw = h // grid, w // grid
    im = im[:ph * grid, :pw * grid].reshape(grid, ph, grid, pw, -1)
    return im.mean((1, 3))                                  # (grid, grid, C)


def quantile_centres(depths, k):
    """Equal-population bin centres in log-depth, frozen over the sequence."""
    v = np.concatenate([d[d > 0].ravel() for d in depths])
    return np.quantile(np.log(v), (np.arange(k) + 0.5) / k)


def route_range(depth, centres):
    ld = np.log(np.maximum(depth, 1e-3))
    return np.abs(ld[..., None] - centres).argmin(-1)


def route_appear(rgb, proj):
    return (rgb.reshape(-1, rgb.shape[-1]) @ proj).argmax(-1).reshape(rgb.shape[:2])


def working_sets(assign, k):
    return set(np.unique(assign).tolist())


def analyse(seq, k, stride, limit, rng):
    dpaths = sorted(glob.glob(os.path.join(seq, "depth", "*.png")))[:limit]
    rpaths = sorted(glob.glob(os.path.join(seq, "rgb", "*.png")))[:limit]
    n = min(len(dpaths), len(rpaths))
    dpaths, rpaths = dpaths[:n:stride], rpaths[:n:stride]

    depths = [load_depth(p) for p in dpaths]
    centres = quantile_centres(depths, k)
    proj = rng.standard_normal((3, k))

    out = {}
    for name in ("range", "appear", "balanced"):
        sizes, reuse, fetch = [], [], []
        prev = None
        for i, dp in enumerate(depths):
            if name == "range":
                a = route_range(dp, centres)
            elif name == "appear":
                a = route_appear(load_rgb(rpaths[i]), proj)
            else:
                a = rng.integers(0, k, size=(GRID, GRID))
            cur = working_sets(a, k)
            sizes.append(len(cur))
            if prev is not None:
                hit = len(cur & prev)
                reuse.append(hit / max(len(cur), 1))
                fetch.append(len(cur) - hit)             # experts to pull for this frame
            prev = cur
        out[name] = {"ws_mean": float(np.mean(sizes)), "ws_frac": float(np.mean(sizes) / k),
                     "reuse": float(np.mean(reuse)), "fetch": float(np.mean(fetch))}
    return out, len(depths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/evc/merlin/tum")
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--strides", type=int, nargs="+", default=[1, 2, 6])
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--out", default="/home/evc/merlin/routing_locality.json")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    seqs = sorted(d for d in glob.glob(os.path.join(args.root, "*")) if os.path.isdir(d))
    res = {}
    for seq in seqs:
        name = os.path.basename(seq)
        print(f"\n=== {name} ===")
        print(f"{'K':>4} {'Hz':>4} {'n':>5} | " + " | ".join(
            f"{r:>8} ws  reuse  fetch" for r in ("range", "appear", "balanced")))
        for k in args.ks:
            for st in args.strides:
                r, n = analyse(seq, k, st, args.limit, rng)
                res[f"{name}|k{k}|s{st}"] = r
                cells = " | ".join(
                    f"{r[m]['ws_mean']:>8.1f} {r[m]['reuse']:>6.2f} {r[m]['fetch']:>5.1f}"
                    for m in ("range", "appear", "balanced"))
                print(f"{k:>4} {30//st:>4} {n:>5} | {cells}")

    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
