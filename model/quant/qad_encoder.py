#!/usr/bin/env python3
"""M-test-2 (CENTERPIECE): Quantization-Aware Distillation of the ENCODER.

E-Q1/2/3 found the distilled 3D-reasoning stack is ~3-bit-native but the pretrained
DINOv2 encoder is the sole low-bit bottleneck (outlier-heavy). M-test-1 showed
distillation monotonically removes those outliers (raw 4.6 -> light-distill 3.2 ->
from-scratch AAT 1.5). This script tests the implied method: fine-tune the encoder
ONLY with straight-through fake-quant in the forward pass, self-distilling against the
frozen fp32 student, so the encoder becomes robust to W{bits} quantization while its
fp behavior is preserved. If encoder@bits error drops toward AAT levels, uniform
low-bit (W3/W4 A8) metric-3D deployment becomes viable -> the largest byte saving on
the bandwidth-bound Jetson (encoder = 86.6M, the biggest single block).
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "distill"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "eval", "geometry"))
from student import build_student          # noqa: E402
from data import make_views, tum_pairs, load_gt_depth  # noqa: E402
from metrics import depth_metrics          # noqa: E402


def fq_ste(w, bits):
    """Per-out-channel symmetric weight fake-quant with straight-through gradient."""
    if bits >= 16:
        return w
    qmax = 2 ** (bits - 1) - 1
    amax = w.abs().amax(1, keepdim=True).clamp_min(1e-12)
    s = amax / qmax
    wq = torch.clamp(torch.round(w / s), -qmax, qmax) * s
    return w + (wq - w).detach()


class QLinear(torch.nn.Module):
    def __init__(self, lin, bits):
        super().__init__()
        self.lin = lin
        self.bits = bits
        self.enabled = True
    def forward(self, x):
        w = fq_ste(self.lin.weight, self.bits) if self.enabled else self.lin.weight
        return F.linear(x, w, self.lin.bias)


def wrap_encoder(model, bits):
    qmods = []
    for name, mod in list(model.named_modules()):
        if isinstance(mod, torch.nn.Linear) and name.startswith("encoder"):
            parent = model
            *path, leaf = name.split(".")
            for p in path:
                parent = getattr(parent, p)
            q = QLinear(mod, bits)
            setattr(parent, leaf, q)
            qmods.append(q)
    return qmods


def set_quant(qmods, on):
    for q in qmods:
        q.enabled = on


def depth_of(model, view):
    pr = model(view, memory_efficient_inference=True, minibatch_size=view[0]["img"].shape[0])[0]
    return pr["pts3d_cam"][..., 2].float()


def si_logl1(pred, target):
    """Scale-invariant log-L1 between two metric depth maps (self-distill loss)."""
    v = torch.isfinite(pred) & torch.isfinite(target) & (pred > 0.05) & (target > 0.05)
    if v.sum() < 200:
        return None
    lp, lt = torch.log(pred.clamp_min(0.05)), torch.log(target.clamp_min(0.05))
    d = (lp - lt)
    shift = (d * v).sum() / v.sum()
    return ((d - shift).abs() * v).sum() / v.sum()


@torch.no_grad()
def evaluate(model, qmods, ref_model, frames, dev, res, n=24):
    """gt_absrel (noise-dominated by the student floor) AND geom_vs_ref (the sensitive
    metric: scale-aligned abs_rel of model-depth vs the FROZEN original fp32 student).
    geom_vs_ref_q at init == E-Q2 only_encoder@3 (~0.023); QAD should drive it toward 0."""
    model.eval()
    s = (res // 14) * 14
    out = {}
    refd = []
    for rp, dp in frames[:n]:
        view = make_views([rp], s, dev)
        refd.append(depth_of(ref_model, view)[0].cpu().numpy())
    for label, on in [("fp", False), ("q", True)]:
        set_quant(qmods, on)
        ar, gd = [], []
        for i, (rp, dp) in enumerate(frames[:n]):
            view = make_views([rp], s, dev)
            pz = depth_of(model, view)[0].cpu().numpy()
            gt = load_gt_depth(dp, pz.shape[0], pz.shape[1])
            ar.append(depth_metrics(gt, pz)["abs_rel"])
            r = refd[i]
            m = np.isfinite(r) & np.isfinite(pz) & (r > 1e-3) & (pz > 1e-3)
            if m.sum() > 200:
                sc = np.median(r[m] / pz[m])
                gd.append(depth_metrics(r, pz * sc)["abs_rel"])
        out[f"gt_absrel_{label}"] = round(float(np.nanmedian(ar)), 4)
        out[f"geom_vs_ref_{label}"] = round(float(np.nanmedian(gd)), 4) if gd else None
    set_quant(qmods, True)
    model.train()
    return out


def robust_save(obj, path):
    """Save to local overlay (fast, no MooseFS short-write) with retry; the /workspace
    network FS truncates torch.save zip streams non-deterministically (runpod memory)."""
    import shutil
    local = "/root/" + os.path.basename(path)
    for attempt in range(4):
        try:
            torch.save(obj, local + ".tmp")
            os.replace(local + ".tmp", local)
            break
        except Exception as e:
            print(f"[save] local attempt {attempt} failed: {e}", flush=True)
    # best-effort mirror to /workspace (retry; ok if it fails, /root copy is the source of truth)
    for attempt in range(4):
        try:
            shutil.copyfile(local, path + ".tmp")
            os.replace(path + ".tmp", path)
            print(f"[save] mirrored to {path}", flush=True); break
        except Exception as e:
            print(f"[save] mirror attempt {attempt} failed: {e}", flush=True)
    return local


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/ckpt/student_v3.pt")
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--tum-root", default="/workspace/data/tum")
    ap.add_argument("--train-seqs", default="rgbd_dataset_freiburg1_360,rgbd_dataset_freiburg1_floor,rgbd_dataset_freiburg2_xyz")
    ap.add_argument("--eval-seqs", default="rgbd_dataset_freiburg1_xyz,rgbd_dataset_freiburg1_plant")
    ap.add_argument("--res", type=int, default=378)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--out", default="/workspace/ckpt/student_qad3.pt")
    args = ap.parse_args()
    dev = "cuda"

    # frozen fp32 reference (self-distill target)
    ref = build_student(size="base", aat_depth=8, device="cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ref.load_state_dict(ck["state_dict"]); ref = ref.to(dev).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    # trainable student, encoder wrapped with STE fake-quant; freeze everything but encoder
    model = build_student(size="base", aat_depth=8, device="cpu")
    model.load_state_dict(ck["state_dict"]); model = model.to(dev)
    qmods = wrap_encoder(model, args.bits)
    for n, p in model.named_parameters():
        p.requires_grad_(n.startswith("encoder"))
    enc_params = [p for n, p in model.named_parameters() if p.requires_grad]
    print(f"[qad] bits={args.bits} encoder params trainable={sum(p.numel() for p in enc_params)/1e6:.1f}M "
          f"wrapped_linears={len(qmods)}", flush=True)

    train = []
    for sq in args.train_seqs.split(","):
        train += tum_pairs(os.path.join(args.tum_root, sq))[::3]
    eval_f = []
    for sq in args.eval_seqs.split(","):
        eval_f += tum_pairs(os.path.join(args.tum_root, sq))[::15]
    print(f"[qad] train frames {len(train)}  eval frames {len(eval_f)}", flush=True)
    print(f"[qad] init eval:", evaluate(model, qmods, ref, eval_f, dev, args.res), flush=True)

    opt = torch.optim.AdamW(enc_params, lr=args.lr, weight_decay=0.0)
    rng = np.random.default_rng(0)
    s = (args.res // 14) * 14
    model.train()
    for step in range(args.steps):
        idx = rng.integers(0, len(train), args.batch)
        paths = [train[i][0] for i in idx]
        view = make_views(paths, s, dev)
        set_quant(qmods, True)
        pred = depth_of(model, view)
        with torch.no_grad():
            tgt = depth_of(ref, view)
        loss = si_logl1(pred, tgt)
        if loss is None:
            continue
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(enc_params, 1.0)
        opt.step()
        if step % 100 == 0:
            print(f"[qad] step {step} loss {loss.item():.4f}", flush=True)
        if step > 0 and step % args.eval_every == 0:
            print(f"[qad] step {step} eval:", evaluate(model, qmods, ref, eval_f, dev, args.res), flush=True)

    print(f"[qad] final eval:", evaluate(model, qmods, ref, eval_f, dev, args.res), flush=True)
    # save clean state_dict (strip the .lin infix so a plain student can load it)
    sd = {}
    for k, v in model.state_dict().items():
        sd[k.replace(".lin.weight", ".weight").replace(".lin.bias", ".bias")] = v.clone()
    local = robust_save({"state_dict": sd, "size": "base", "aat_depth": 8, "qad_bits": args.bits}, args.out)
    print(f"=== QAD_DONE {local} (mirror {args.out}) ===", flush=True)


if __name__ == "__main__":
    main()
