#!/usr/bin/env python3
"""Pack TUM RGB-D sequences into the same npz format `gen_traj.py` emits, for eval_recirc.py.

Why TUM and not the sim: the habitat renderer needs EGL+CUDA and the GPU partitions are
account-capped, whereas the recirculation eval itself is pure torch and runs on CPU at
0.31 s/frame. TUM is real sequential RGB-D, is already the repo's offline-geometry dataset, and
needs no renderer.

One limitation to state whenever a number from here is quoted: the Kinect does not return depth
below ~0.5 m, so the **0.2-0.5 m bin is empty on TUM**. That is the bin that predicted
navigation in the C4 work, so TUM can answer "does cross-frame state change the depth at all"
but cannot answer "does it help the near field". The sim render covers that.

RGB and depth are captured on different clocks, so frames are associated by nearest timestamp
within `--max_dt`. Both are centre-cropped 640x480 -> 480x480 before the student's square
resize, so no aspect distortion is introduced and GT stays pixel-aligned with the prediction.
"""
import argparse
import os

import numpy as np

DEPTH_SCALE = 5000.0  # TUM 16-bit PNG units per metre


def read_list(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            rows.append((float(parts[0]), parts[1]))
    return rows


def associate(rgb, dep, max_dt):
    """Nearest-timestamp match, each depth frame used at most once."""
    out, j = [], 0
    used = set()
    for t, p in rgb:
        while j + 1 < len(dep) and abs(dep[j + 1][0] - t) <= abs(dep[j][0] - t):
            j += 1
        if j not in used and abs(dep[j][0] - t) <= max_dt:
            used.add(j)
            out.append((p, dep[j][1]))
    return out


def center_square(a):
    h, w = a.shape[:2]
    s = min(h, w)
    y, x = (h - s) // 2, (w - s) // 2
    return a[y:y + s, x:x + s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir containing rgbd_dataset_* folders")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=48, help="frames per chunk")
    ap.add_argument("--chunks", type=int, default=6, help="chunks per sequence")
    ap.add_argument("--stride", type=int, default=1, help="sample every Nth associated frame")
    ap.add_argument("--res", type=int, default=378)
    ap.add_argument("--max_dt", type=float, default=0.02)
    args = ap.parse_args()

    import cv2
    os.makedirs(args.out, exist_ok=True)
    seqs = sorted(d for d in os.listdir(args.root)
                  if d.startswith("rgbd_dataset_") and
                  os.path.isdir(os.path.join(args.root, d)))
    n_out = 0
    for name in seqs:
        d = os.path.join(args.root, name)
        try:
            pairs = associate(read_list(os.path.join(d, "rgb.txt")),
                              read_list(os.path.join(d, "depth.txt")), args.max_dt)
        except FileNotFoundError:
            print(f"  {name}: missing rgb.txt/depth.txt, skipped", flush=True)
            continue
        pairs = pairs[::args.stride]
        need = args.frames
        for c in range(args.chunks):
            chunk = pairs[c * need:(c + 1) * need]
            if len(chunk) < need:
                break
            rgbs, deps = [], []
            for rp, dp in chunk:
                im = cv2.imread(os.path.join(d, rp), cv2.IMREAD_COLOR)
                dm = cv2.imread(os.path.join(d, dp), cv2.IMREAD_UNCHANGED)
                if im is None or dm is None:
                    break
                im = center_square(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                dm = center_square(dm).astype(np.float32) / DEPTH_SCALE
                dm[dm <= 0] = np.nan
                rgbs.append(cv2.resize(im, (args.res, args.res), interpolation=cv2.INTER_AREA))
                deps.append(cv2.resize(dm, (args.res, args.res),
                                       interpolation=cv2.INTER_NEAREST).astype(np.float16))
            if len(rgbs) < need:
                continue
            tag = f"{name.replace('rgbd_dataset_', '')}_c{c}"
            np.savez_compressed(os.path.join(args.out, tag + ".npz"),
                                rgb=np.stack(rgbs), dep=np.stack(deps), spacing=-1.0)
            n_out += 1
            valid = float(np.mean(np.isfinite(np.stack(deps))))
            print(f"  {tag}: {len(rgbs)} frames, {valid*100:.0f}% valid depth", flush=True)
    print(f"wrote {n_out} sequences to {args.out}", flush=True)


if __name__ == "__main__":
    main()
