#!/usr/bin/env python3
"""QAD v2: Quantization-Aware Distillation of the encoder with LSQ + fp-anchor.

v1 (fixed-range STE) closed only 40% of the encoder's 3-bit gap and drifted fp
accuracy. v2 fixes both:
  - LSQ: per-output-channel LEARNABLE quant scale, so the optimizer can clip the
    encoder's input-channel outliers (the mechanism in our paper) instead of being
    stuck with the outlier-determined fixed range.
  - fp-anchor: also distill the UNQUANTIZED forward against the frozen fp32 ref, so
    fp accuracy is preserved while quant robustness is gained.
  - same-distribution eval: train/eval are disjoint frames of the SAME sequences, so
    geom_vs_ref_fp drift reflects true overfitting, not a train/eval domain gap.
Success = encoder@bits geom_vs_ref drops toward the AAT's ~0.001 with ~0 fp drift.
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "distill"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "eval", "geometry"))
from student import build_student          # noqa: E402
from data import make_views, tum_pairs, load_gt_depth  # noqa: E402
from metrics import depth_metrics          # noqa: E402


class LSQLinear(nn.Module):
    """Weight-only LSQ: per-out-channel learnable scale, STE round."""
    def __init__(self, lin, bits):
        super().__init__()
        self.lin = lin
        self.bits = bits
        self.qmax = 2 ** (bits - 1) - 1
        amax = lin.weight.detach().abs().amax(1, keepdim=True).clamp_min(1e-8)
        self.log_s = nn.Parameter(torch.log(amax / self.qmax))
        self.enabled = True

    def forward(self, x):
        if not self.enabled or self.bits >= 16:
            return F.linear(x, self.lin.weight, self.lin.bias)
        s = self.log_s.exp()
        v = self.lin.weight / s
        vr = v + (torch.round(v) - v).detach()          # STE round
        vc = torch.clamp(vr, -self.qmax, self.qmax)      # differentiable clamp
        wq = vc * s
        return F.linear(x, wq, self.lin.bias)


def wrap_encoder(model, bits):
    qmods = []
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Linear) and name.startswith("encoder"):
            parent = model
            *path, leaf = name.split(".")
            for p in path:
                parent = getattr(parent, p)
            q = LSQLinear(mod, bits)
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
    v = torch.isfinite(pred) & torch.isfinite(target) & (pred > 0.05) & (target > 0.05)
    if v.sum() < 200:
        return None
    d = torch.log(pred.clamp_min(0.05)) - torch.log(target.clamp_min(0.05))
    shift = (d * v).sum() / v.sum()
    return ((d - shift).abs() * v).sum() / v.sum()


@torch.no_grad()
def evaluate(model, qmods, ref_model, frames, dev, res, n=24):
    model.eval()
    s = (res // 14) * 14
    out = {}
    refd = [depth_of(ref_model, make_views([rp], s, dev))[0].cpu().numpy() for rp, _ in frames[:n]]
    for label, on in [("fp", False), ("q", True)]:
        set_quant(qmods, on)
        gd = []
        for i, (rp, dp) in enumerate(frames[:n]):
            pz = depth_of(model, make_views([rp], s, dev))[0].cpu().numpy()
            r = refd[i]
            m = np.isfinite(r) & np.isfinite(pz) & (r > 1e-3) & (pz > 1e-3)
            if m.sum() > 200:
                sc = np.median(r[m] / pz[m])
                gd.append(depth_metrics(r, pz * sc)["abs_rel"])
        out[f"geom_vs_ref_{label}"] = round(float(np.nanmedian(gd)), 4) if gd else None
    set_quant(qmods, True)
    model.train()
    return out


def robust_save(obj, path):
    local = "/root/" + os.path.basename(path)
    for _ in range(4):
        try:
            torch.save(obj, local + ".tmp"); os.replace(local + ".tmp", local); break
        except Exception as e:
            print(f"[save] retry: {e}", flush=True)
    return local


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/ckpt/student_v3.pt")
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--tum-root", default="/workspace/data/tum")
    ap.add_argument("--seqs", default="rgbd_dataset_freiburg1_xyz,rgbd_dataset_freiburg1_plant")
    ap.add_argument("--res", type=int, default=378)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lr-scale", type=float, default=1e-3)
    ap.add_argument("--fp-anchor", type=float, default=0.5)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--out", default="/workspace/ckpt/student_qad3v2.pt")
    args = ap.parse_args()
    dev = "cuda"

    ref = build_student(size="base", aat_depth=8, device="cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ref.load_state_dict(ck["state_dict"]); ref = ref.to(dev).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    model = build_student(size="base", aat_depth=8, device="cpu")
    model.load_state_dict(ck["state_dict"]); model = model.to(dev)
    qmods = wrap_encoder(model, args.bits)
    # train: encoder weights + LSQ scales only
    w_params, s_params = [], []
    for n, p in model.named_parameters():
        if "log_s" in n:
            p.requires_grad_(True); s_params.append(p)
        elif n.startswith("encoder."):   # ONLY the DINOv2 image encoder (not geometric *_encoder)
            p.requires_grad_(True); w_params.append(p)
        else:
            p.requires_grad_(False)
    print(f"[qadv2] bits={args.bits} enc_w={sum(p.numel() for p in w_params)/1e6:.1f}M "
          f"lsq_scales={len(s_params)} fp_anchor={args.fp_anchor}", flush=True)

    # same-distribution split: even frames train, odd frames eval
    allf = []
    for sq in args.seqs.split(","):
        allf += tum_pairs(os.path.join(args.tum_root, sq))
    train = allf[0::2]
    eval_f = allf[1::2][::8]
    print(f"[qadv2] train {len(train)} eval {len(eval_f)}", flush=True)
    print("[qadv2] init eval:", evaluate(model, qmods, ref, eval_f, dev, args.res), flush=True)

    opt = torch.optim.AdamW([{"params": w_params, "lr": args.lr},
                             {"params": s_params, "lr": args.lr_scale}], weight_decay=0.0)
    rng = np.random.default_rng(0)
    s = (args.res // 14) * 14
    model.train()
    for step in range(args.steps):
        idx = rng.integers(0, len(train), args.batch)
        paths = [train[i][0] for i in idx]
        view = make_views(paths, s, dev)
        with torch.no_grad():
            tgt = depth_of(ref, view)
        set_quant(qmods, True)
        lq = si_logl1(depth_of(model, view), tgt)
        loss = lq
        if args.fp_anchor > 0:
            set_quant(qmods, False)
            lfp = si_logl1(depth_of(model, view), tgt)
            set_quant(qmods, True)
            if lfp is not None:
                loss = (lq if lq is not None else 0) + args.fp_anchor * lfp
        if loss is None or not torch.is_tensor(loss):
            continue
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(w_params + s_params, 1.0)
        opt.step()
        if step % 100 == 0:
            print(f"[qadv2] step {step} loss {float(loss):.4f}", flush=True)
        if step > 0 and step % args.eval_every == 0:
            print(f"[qadv2] step {step} eval:", evaluate(model, qmods, ref, eval_f, dev, args.res), flush=True)

    print("[qadv2] final eval:", evaluate(model, qmods, ref, eval_f, dev, args.res), flush=True)
    # fold LSQ scales into quantized weights and save a plain student state_dict
    sd = {}
    set_quant(qmods, True)
    qmap = {id(q.lin): q for q in qmods}
    for name, mod in model.named_modules():
        if isinstance(mod, LSQLinear):
            s_ = mod.log_s.exp()
            v = mod.lin.weight / s_
            wq = torch.clamp(torch.round(v), -mod.qmax, mod.qmax) * s_
            sd[name + ".weight"] = wq.detach().clone()
            if mod.lin.bias is not None:
                sd[name + ".bias"] = mod.lin.bias.detach().clone()
    # everything else unchanged from ref
    base = {k: v.clone() for k, v in model.state_dict().items()
            if ".lin." not in k and "log_s" not in k}
    # remap LSQLinear weights: stored above already as name+'.weight'
    full = dict(base)
    full.update(sd)
    local = robust_save({"state_dict": full, "size": "base", "aat_depth": 8,
                         "qad_bits": args.bits, "qad": "v2_lsq_folded"}, args.out)
    print(f"=== QADV2_DONE {local} ===", flush=True)


if __name__ == "__main__":
    main()
