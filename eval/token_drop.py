#!/usr/bin/env python3
"""Conclusive token-saliency test through the REAL MERLIN student.

Question: when you drop encoder tokens, does a GEOMETRY-derived saliency keep depth
accuracy better than (a) input-RGB-gradient saliency (LiteVGGT 2512.04939), (b) the
model's own feature-norm saliency (ToMe-style internal importance), (c) random?
If geometry wins at matched keep-ratio, the "geometry-source token selection" sliver
is real and measured (not the marginal effect a reviewer assumed).

Drop = zero the low-saliency tokens at the ViT encoder output (forward hook), keeping
the 729-length grid so AAT+DPT shapes are untouched. Tests SELECTION QUALITY through
the real model -> depth_z, scale-aligned abs_rel vs TUM GT. (Latency ceiling measured
separately in tok_bench: ~2.4x at keep 0.5.)
"""
import sys, os, json, argparse, time
import numpy as np
from PIL import Image
import torch
sys.path.insert(0, "/home/evc/merlin/repo/model/backbones")
sys.path.insert(0, "/home/evc/merlin/repo/model/distill")
from teacher import patch_linalg_cpu_fallback, _upcast_geometry_postproc, add_dtype_casting_hooks
from student import build_student
from mapanything.utils.image import load_images
from mapanything.utils.cropping import crop_resize_if_necessary
from PIL.ImageOps import exif_transpose

GRID = 27          # 378/14
N = GRID * GRID     # 729
DEPTH_SCALE = 5000.0

# ---- model ----
def load_model(ckpt="/home/evc/merlin/student_distilled.pt"):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = build_student(size=ck.get("size", "base"), aat_depth=ck.get("aat_depth", 8), device="cpu")
    m.load_state_dict(ck["state_dict"]); m = m.to(torch.bfloat16).eval()
    add_dtype_casting_hooks(m); _upcast_geometry_postproc(); patch_linalg_cpu_fallback()
    return m.to("cuda")

# ---- saliency-gated token zeroing hook on the encoder ----
class Gate:
    def __init__(s): s.keep = None; s.mode = None; s.ext = None; s.fill = "nearest"  # ext:(729,) external saliency
    def hook(s, module, inp, out):
        feats = out.features                       # (V,768,27,27) channels-first grid
        V, C, H, W = feats.shape; Ntok = H * W
        if s.keep is None or s.keep >= Ntok: return out
        if s.mode == "featnorm":
            sal = feats.float().norm(dim=1).mean(0).reshape(-1)   # (729,) row-major
        elif s.mode == "random":
            sal = torch.rand(Ntok, device=feats.device)
        else:                                       # external (rgb / geom), (729,) row-major
            sal = s.ext
        k = int(s.keep)
        drop = torch.topk(sal, Ntok - k, largest=False).indices
        ff = feats.view(V, C, Ntok)
        if s.fill == "nearest":                    # ToMe-style merge: dropped <- nearest kept token
            keepm = torch.ones(Ntok, dtype=torch.bool, device=feats.device); keepm[drop] = False
            kept = torch.nonzero(keepm, as_tuple=False).squeeze(1)
            hy = (torch.arange(Ntok, device=feats.device) // W).float(); wx = (torch.arange(Ntok, device=feats.device) % W).float()
            dh = hy[drop][:, None] - hy[kept][None, :]; dw = wx[drop][:, None] - wx[kept][None, :]
            nn = kept[torch.argmin(dh * dh + dw * dw, dim=1)]
            ff[:, :, drop] = ff[:, :, nn]
        else:
            ff[:, :, drop] = 0                     # zero (destructive)
        return out

def infer_depth(m, views):
    with torch.no_grad():
        o = m.infer(views, memory_efficient_inference=True, minibatch_size=1,
                    use_amp=False, apply_mask=False, mask_edges=False)[0]
    return o["depth_z"].float().squeeze().cpu().numpy()      # (378,378)

# ---- TUM ----
def assoc(seq):
    def rd(f):
        o = []
        for ln in open(os.path.join(seq, f)):
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                p = ln.split(); o.append((float(p[0]), p[1]))
        return o
    rgb, dep = rd("rgb.txt"), rd("depth.txt"); dt = np.array([t for t, _ in dep]); out = []
    for tr, pr in rgb:
        j = int(np.argmin(abs(dt - tr)))
        if abs(dt[j] - tr) < 0.02: out.append((os.path.join(seq, pr), os.path.join(seq, dep[j][1])))
    return out

def square_resize(arr, size, nearest=False):
    """center-crop to square then resize to size (matches load_images square)."""
    H, W = arr.shape[:2]; s = min(H, W); y0, x0 = (H - s) // 2, (W - s) // 2
    a = arr[y0:y0 + s, x0:x0 + s]
    im = Image.fromarray(a)
    im = im.resize((size, size), Image.NEAREST if nearest else Image.BILINEAR)
    return np.asarray(im)

def patch_grad(img2d):
    """mean gradient magnitude per 14x14 patch -> (729,) on the 27x27 grid."""
    g = np.abs(np.diff(img2d, axis=1, prepend=img2d[:, :1])) + np.abs(np.diff(img2d, axis=0, prepend=img2d[:1, :]))
    P = 14; return g[:GRID * P, :GRID * P].reshape(GRID, P, GRID, P).mean((1, 3)).ravel()

def abs_rel(pred, gt):
    m = (gt > 0.1) & (gt < 10) & np.isfinite(pred) & (pred > 0)
    if m.sum() < 500: return np.nan
    p = pred.copy(); p *= np.median(gt[m] / p[m])      # scale-align (paper convention)
    return float(np.mean(np.abs(p[m] - gt[m]) / gt[m]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--nframes", type=int, default=40)
    ap.add_argument("--keeps", default="1.0,0.7,0.5,0.35")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    keeps = [float(x) for x in a.keeps.split(",")]
    size = 378
    m = load_model(); gate = Gate(); m.encoder.register_forward_hook(gate.hook)
    pairs = assoc(a.seq)[::4][:a.nframes]
    print(f"{os.path.basename(a.seq)}: {len(pairs)} frames")

    modes = ["random", "rgb", "featnorm", "geom"]
    acc = {md: {k: [] for k in keeps} for md in modes}
    for fi, (rp, dp) in enumerate(pairs):
        views = load_images([rp], resize_mode="square", size=size)
        for v in views: v["img"] = v["img"].to("cuda")
        # GT depth + RGB aligned to the SAME 378 transform the model uses
        img_pil = exif_transpose(Image.open(rp)).convert("RGB")
        gt_raw = np.asarray(Image.open(dp)).astype(np.float32) / DEPTH_SCALE
        img378, gt = crop_resize_if_necessary(img_pil, (size, size), depthmap=gt_raw)
        gray = np.asarray(img378.convert("L")).astype(np.float32)   # 378x378 aligned
        rgb_sal = torch.tensor(patch_grad(gray), device="cuda")
        gtsq = gt.copy(); gtsq[gtsq == 0] = np.nan
        geo = patch_grad(np.nan_to_num(gtsq, nan=float(np.nanmedian(gtsq))))
        geo_sal = torch.tensor(geo / (np.nanmedian(gtsq) + 1e-6), device="cuda")
        for md in modes:
            gate.mode = md
            gate.ext = rgb_sal if md == "rgb" else (geo_sal if md == "geom" else None)
            for k in keeps:
                gate.keep = int(round(k * N))
                er = abs_rel(infer_depth(m, views), gt)
                if np.isfinite(er): acc[md][k].append(er)
        if fi % 10 == 0: print(f"  {fi}/{len(pairs)}", flush=True)
    res = {"seq": os.path.basename(a.seq), "nframes": len(pairs), "keeps": keeps,
           "abs_rel": {md: {str(k): round(float(np.mean(acc[md][k])), 4) if acc[md][k] else None for k in keeps} for md in modes}}
    out = a.out or f"/home/evc/merlin/tum/tokdrop_{res['seq']}.json"
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
