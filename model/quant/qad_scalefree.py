#!/usr/bin/env python3
"""Scale-free QAD (the method that flows from IMU-anchored quantization).

Since a visual-inertial system recovers absolute metric scale from the IMU, an
aggressively-quantized model should NOT spend its scarce low-bit capacity preserving
absolute scale -- only the relative geometry that survives and that the planner uses.
We test, at the brutal W2 regime, whether a SCALE-INVARIANT QAD loss (penalize geometry
only) yields better IMU-recovered geometry than a SCALE-DEPENDENT loss (penalize absolute
log-depth). Both whole-model LSQ + fp-anchor; compared on the IMU-recovered (scale-aligned)
held-out abs_rel that a VI deployment actually achieves.
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
    def __init__(self, lin, bits):
        super().__init__()
        self.lin = lin; self.bits = bits; self.qmax = 2 ** (bits - 1) - 1
        amax = lin.weight.detach().abs().amax(1, keepdim=True).clamp_min(1e-8)
        self.log_s = nn.Parameter(torch.log(amax / self.qmax)); self.enabled = True
    def forward(self, x):
        if not self.enabled or self.bits >= 16:
            return F.linear(x, self.lin.weight, self.lin.bias)
        s = self.log_s.exp()
        v = self.lin.weight / s
        vr = v + (torch.round(v) - v).detach()
        return F.linear(x, torch.clamp(vr, -self.qmax, self.qmax) * s, self.lin.bias)


def wrap_all(model, bits):
    qmods = []
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Linear):
            parent = model; *path, leaf = name.split(".")
            for p in path: parent = getattr(parent, p)
            q = LSQLinear(mod, bits); setattr(parent, leaf, q); qmods.append(q)
    return qmods


def set_q(qmods, on):
    for q in qmods: q.enabled = on


def depth_of(model, view):
    pr = model(view, memory_efficient_inference=True, minibatch_size=view[0]["img"].shape[0])[0]
    return pr["pts3d_cam"][..., 2].float()


def loss_si(pred, tgt):
    v = torch.isfinite(pred) & torch.isfinite(tgt) & (pred > 0.05) & (tgt > 0.05)
    if v.sum() < 200: return None
    d = torch.log(pred.clamp_min(0.05)) - torch.log(tgt.clamp_min(0.05))
    shift = (d * v).sum() / v.sum()                       # SCALE-INVARIANT: remove global shift
    return ((d - shift).abs() * v).sum() / v.sum()


def loss_abs(pred, tgt):
    v = torch.isfinite(pred) & torch.isfinite(tgt) & (pred > 0.05) & (tgt > 0.05)
    if v.sum() < 200: return None
    d = torch.log(pred.clamp_min(0.05)) - torch.log(tgt.clamp_min(0.05))
    return (d.abs() * v).sum() / v.sum()                  # SCALE-DEPENDENT: absolute


@torch.no_grad()
def eval_imu(model, qmods, frames, dev, res, n=24):
    """IMU-recovered (scale-aligned) held-out abs_rel = the deployment metric."""
    model.eval(); set_q(qmods, True); s = (res // 14) * 14; ar = []
    for rp, dp in frames[:n]:
        pz = depth_of(model, make_views([rp], s, dev))[0].cpu().numpy()
        gt = load_gt_depth(dp, pz.shape[0], pz.shape[1])
        m = np.isfinite(gt) & np.isfinite(pz) & (gt > 0.1) & (pz > 0.05)
        if m.sum() > 200:
            sg = np.median(gt[m] / pz[m]); ar.append(depth_metrics(gt, pz * sg)["abs_rel"])
    model.train()
    return round(float(np.nanmedian(ar)), 4) if ar else None


def run_one(loss_name, args, dev, ref, train, evf):
    lossf = loss_si if loss_name == "si" else loss_abs
    model = build_student(size="base", aat_depth=8, device="cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"]); model = model.to(dev)
    qmods = wrap_all(model, args.bits)
    w = [p for n, p in model.named_parameters() if "log_s" not in n]
    s = [p for n, p in model.named_parameters() if "log_s" in n]
    opt = torch.optim.AdamW([{"params": w, "lr": args.lr}, {"params": s, "lr": 1e-3}], weight_decay=0.0)
    rng = np.random.default_rng(0); sz = (args.res // 14) * 14
    print(f"[{loss_name}] init IMU-recovered abs_rel:", eval_imu(model, qmods, evf, dev, args.res), flush=True)
    model.train()
    for step in range(args.steps):
        idx = rng.integers(0, len(train), args.batch)
        view = make_views([train[i][0] for i in idx], sz, dev)
        with torch.no_grad(): tgt = depth_of(ref, view)
        set_q(qmods, True); lq = lossf(depth_of(model, view), tgt)
        set_q(qmods, False); lfp = lossf(depth_of(model, view), tgt); set_q(qmods, True)
        loss = (lq if lq is not None else 0) + 0.5 * (lfp if lfp is not None else 0)
        if not torch.is_tensor(loss): continue
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(w + s, 1.0); opt.step()
        if step % 200 == 0:
            print(f"[{loss_name}] step {step} loss {float(loss):.4f}", flush=True)
    final = eval_imu(model, qmods, evf, dev, args.res)
    print(f"=== SCALEFREE {loss_name} bits={args.bits} final IMU-recovered abs_rel = {final} ===", flush=True)
    del model; torch.cuda.empty_cache()
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/ckpt/student_v3.pt")
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--seqs", default="rgbd_dataset_freiburg1_xyz,rgbd_dataset_freiburg1_plant")
    ap.add_argument("--tum-root", default="/workspace/data/tum")
    ap.add_argument("--res", type=int, default=378)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()
    dev = "cuda"
    ref = build_student(size="base", aat_depth=8, device="cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ref.load_state_dict(ck["state_dict"]); ref = ref.to(dev).eval()
    for p in ref.parameters(): p.requires_grad_(False)
    allf = []
    for sq in args.seqs.split(","): allf += tum_pairs(os.path.join(args.tum_root, sq))
    train, evf = allf[0::2], allf[1::2][::8]
    print(f"[scalefree] bits={args.bits} train {len(train)} eval {len(evf)}", flush=True)
    si = run_one("si", args, dev, ref, train, evf)
    ab = run_one("abs", args, dev, ref, train, evf)
    print(f"=== SCALEFREE_RESULT bits={args.bits} scale_invariant={si} scale_dependent={ab} "
          f"(lower IMU-recovered abs_rel is better) ===", flush=True)


if __name__ == "__main__":
    main()
