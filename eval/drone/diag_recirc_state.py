#!/usr/bin/env python3
"""Is the shuffle control actually a control?

`eval_recirc.py` compares injecting frame t-1's z_s against injecting a foreign frame's z_s.
That comparison is only meaningful if those two tensors DIFFER. Residual streams are usually
dominated by a large shared component (a per-token mean that barely moves between inputs), so
two arbitrary frames can sit at cosine ~0.99 of each other. If that holds here, `prev` and
`shuffle` inject nearly the same tensor, the arms cannot separate by construction, and a null
result says nothing about temporal state.

Reports, per tap:
  cos_consec   cosine(z_s^t, z_s^{t-1})            -- what `prev` injects
  cos_foreign  cosine(z_s^t, z_s^{random other})   -- what `shuffle` injects
  and both again after removing the per-token mean over frames, which is the part that
  actually carries frame-specific information.

Read it as: if cos_consec and cos_foreign are both near 1 and near each other, the experiment
was underpowered, not negative. The centred columns say how much signal was available at all.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "model"))

from eval.drone.perception import Perception  # noqa: E402
from recirc.recirc import tap_list  # noqa: E402


def cos(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seqs", required=True)
    ap.add_argument("--taps", nargs="+", default=["enc.3", "enc.9", "aat.0", "aat.2", "aat.6"])
    ap.add_argument("--n_seq", type=int, default=4)
    ap.add_argument("--n_frames", type=int, default=12)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--res", type=int, default=378)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--frac", action="store_true",
                    help="also measure the informative fraction, raw vs centred")
    ap.add_argument("--pairs", nargs="+", default=["enc.9:enc.3", "aat.2:enc.6", "aat.6:aat.0"])
    args = ap.parse_args()

    perc = Perception(args.ckpt, device=args.device)
    mods = dict(tap_list(perc.model))
    want = [t for t in args.taps if t in mods]
    grab, hooks = {}, []
    for t in want:
        def mk(label):
            def hook(m, i, o):
                grab[label] = (o[0] if isinstance(o, tuple) else o).detach().clone()
                return o
            return hook
        hooks.append(mods[t].register_forward_hook(mk(t)))

    files = sorted(glob.glob(os.path.join(args.seqs, "*.npz")))[: args.n_seq]
    # acts[tap] = list over (seq, frame) of tensors, kept in stream order per sequence
    acts = {t: [] for t in want}
    for f in files:
        rgb = np.load(f)["rgb"][:: args.stride][: args.n_frames]
        per = {t: [] for t in want}
        for i in range(len(rgb)):
            perc._raw_depth(rgb[i], args.res)
            for t in want:
                per[t].append(grab[t])
        for t in want:
            acts[t].append(per[t])
    for h in hooks:
        h.remove()

    rng = np.random.default_rng(0)
    print(f"{'tap':>8} {'cos_consec':>11} {'cos_foreign':>12} "
          f"{'ctr_consec':>11} {'ctr_foreign':>12} {'frame_var%':>11}")
    for t in want:
        flat = [z for seq in acts[t] for z in seq]
        mean = torch.stack(flat).mean(0)
        var_frac = float(torch.stack([(z - mean).norm() ** 2 for z in flat]).sum()
                         / torch.stack([z.norm() ** 2 for z in flat]).sum())
        cc, cf, kc, kf = [], [], [], []
        for si, seq in enumerate(acts[t]):
            for i in range(1, len(seq)):
                cc.append(cos(seq[i], seq[i - 1]))
                kc.append(cos(seq[i] - mean, seq[i - 1] - mean))
                sj = rng.integers(len(acts[t]))
                fj = rng.integers(len(acts[t][sj]))
                if sj == si and abs(fj - i) < 3:
                    continue
                cf.append(cos(seq[i], acts[t][sj][fj]))
                kf.append(cos(seq[i] - mean, acts[t][sj][fj] - mean))
        print(f"{t:>8} {np.mean(cc):>11.4f} {np.mean(cf):>12.4f} "
              f"{np.mean(kc):>11.4f} {np.mean(kf):>12.4f} {var_frac*100:>10.2f}%")

    if args.frac:
        pairs = [tuple(p.split(":")) for p in args.pairs]
        informative_fraction(perc, files, pairs, alphas=[0.15], centers=[0.0, 0.9],
                             n_frames=min(8, args.n_frames), res=args.res)


@torch.no_grad()
def informative_fraction(perc, files, pairs, alphas, centers, n_frames=8, res=378):
    """Of the change an injection causes, how much depends on WHICH frame supplied the state?

        frac = |out_prev - out_shuffle| / |out_prev - out_base|

    This decides whether the prev-vs-shuffle comparison can separate at all. If frac is a few
    percent, the injection is dominated by the frame-independent offset between the two layers'
    mean activations, both arms move the metrics together, and a null is uninformative rather
    than negative. Centring should raise it.
    """
    from recirc.recirc import Recirculator

    streams = [np.load(f)["rgb"][:n_frames] for f in files]
    foreign_img = np.load(files[-1])["rgb"][-1]

    print(f"\n{'source':>8} {'dest':>8} {'alpha':>6} {'center':>7} "
          f"{'frac_informative':>17} {'perturb_rel':>12}")
    rows = []
    for src, dst in pairs:
        for a in alphas:
            for c in centers:
                cap = Recirculator(perc.model, src, dst, alpha=0.0, center=c)
                perc._raw_depth(foreign_img, res)
                foreign_state = cap.last.clone()
                cap.remove()

                fr, pr = [], []
                for rgb in streams:
                    base = perc._raw_depth(rgb[-1], res)

                    rec = Recirculator(perc.model, src, dst, alpha=a, center=c, mode="prev")
                    for i in range(len(rgb) - 1):
                        perc._raw_depth(rgb[i], res)
                    out_prev = perc._raw_depth(rgb[-1], res)
                    rec.remove()

                    rec = Recirculator(perc.model, src, dst, alpha=a, center=c, mode="shuffle")
                    for i in range(len(rgb) - 1):
                        rec.set_state(foreign_state if i > 0 else None)
                        perc._raw_depth(rgb[i], res)
                    rec.set_state(foreign_state)
                    out_shuf = perc._raw_depth(rgb[-1], res)
                    rec.remove()

                    dp = float(np.linalg.norm(out_prev - base))
                    fr.append(float(np.linalg.norm(out_prev - out_shuf)) / max(dp, 1e-9))
                    pr.append(dp / max(float(np.linalg.norm(base)), 1e-9))
                print(f"{src:>8} {dst:>8} {a:>6.2f} {c:>7.2f} "
                      f"{np.mean(fr):>17.3f} {np.mean(pr):>12.4f}", flush=True)
                rows.append({"source": src, "dest": dst, "alpha": a, "center": c,
                             "frac": float(np.mean(fr)), "perturb": float(np.mean(pr))})
    return rows


if __name__ == "__main__":
    main()
