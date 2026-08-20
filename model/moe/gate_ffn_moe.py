#!/usr/bin/env python3
"""Gates G1, G3, G4, G5, G6 for arm U. Nothing 12-hour runs until this prints ALL GATES PASS.

Each gate exists because this repo has already paid for the failure it catches; the reasons are
in docs/superpowers/specs/2026-08-14-fcr-00-upcycling-control.md section 3. G2 (K=1 == arm F) is
two short run_cell invocations and is driven by the sbatch, not from here.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from moe.ffn_moe import FFNMoE, ffn_modules, ffn_parameters, upcycle_ffn  # noqa: E402
from moe.ore import set_route_feature                                     # noqa: E402
import moe.run_cell as rc                                                 # noqa: E402

SITES = [f"info_sharing.self_attention_blocks.{b}.mlp" for b in range(8)]


def wrap(model, k, top_k, council_m, route_dim=768):
    mods = upcycle_ffn(model, SITES, num_experts=k, top_k=top_k, council_m=council_m,
                       balance="aux", router_norm=True, route_dim=route_dim)
    if route_dim:
        model.info_sharing.self_attention_blocks[0].register_forward_pre_hook(
            lambda m, i: set_route_feature(model, i[0].detach()))
    return mods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--val-root", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--res", type=int, default=378)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wall-hours", type=float, default=11.0)
    ap.add_argument("--min-steps", type=int, default=60000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dev = "cuda"
    size = (args.res // 14) * 14
    torch.manual_seed(0)
    np.random.seed(0)
    rep, fails = {}, []

    train = rc.load_pairs(args.train_root, 1)
    val = rc.load_pairs(args.val_root, 1)
    print(f"[gate] train {len(train)} val {len(val)}", flush=True)

    # ---------------------------------------------------------------- G1 + G3
    ref_model = rc.load_model(args.init, dev)
    ref_model.eval()
    with torch.no_grad():
        ref = rc.predict_depth(ref_model, [p[0] for p in val[:2]], size, dev, batch=2).clone()
    dense_site = sum(p.numel() for b in ref_model.info_sharing.self_attention_blocks
                     for p in b.mlp.parameters())
    del ref_model
    torch.cuda.empty_cache()

    for tag, (k, tk, cm) in {"top1": (args.k, 1, 0), "top2": (args.k, 2, 0),
                             "council3": (args.k, 2, 3), "k1": (1, 1, 0)}.items():
        m = rc.load_model(args.init, dev)
        mods = wrap(m, k, tk, cm)
        m.to(dev).eval()
        # MODULE level, on real activations. This is the actual claim -- identical copies plus
        # renormalised gate values reproduce the parent MLP -- and it is measured where a
        # violation would be visible, not after 8 blocks have amplified fp rounding.
        site = m.info_sharing.self_attention_blocks[0].mlp
        cap = {}
        h = site.register_forward_pre_hook(lambda mod, i: cap.setdefault("x", i[0].detach()))
        with torch.no_grad():
            y = rc.predict_depth(m, [p[0] for p in val[:2]], size, dev, batch=2)
        h.remove()
        x = cap["x"]
        with torch.no_grad():
            parent = site.experts[0](x)
            mod_d = float((site(x) - parent).abs().max() / parent.abs().max())
        # MODEL level. K=1 routes every token to one expert, so it is one GEMM over all rows and
        # comes out bit-exact; K>1 does the same arithmetic in per-expert row subsets, which
        # reorders the reduction. That is fp noise, and it is 3 orders of magnitude away from
        # what a real error would cost: a mis-normalised top-2 doubles the FFN contribution.
        d = float((y - ref).abs().max())
        rel = float(((y - ref).abs() / ref.abs().clamp_min(1e-3)).median())
        rep.update({f"G1_{tag}_maxabs": d, f"G1_{tag}_medrel": rel, f"G1_{tag}_module": mod_d})
        # 1e-4 relative, not 1e-5: top-k renormalisation forms w1*y + w2*y where w1+w2 sums to
        # one only to floating-point precision, so each product rounds separately. Measured 1.7e-5
        # for top-2 against 0.0 for K=1. A real violation -- an unnormalised gate value -- would
        # be O(1), four orders away, so the bound still separates the two cases.
        ok = mod_d < 1e-4 and rel < 0.01
        print(f"[G1] {tag}: module {mod_d:.2e} | depth max {d:.2e} m, median rel {rel:.2e} "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            fails.append(f"G1 {tag} module {mod_d:.2e} rel {rel:.2e}")
        if tag == "top1":
            tot = sum(x.total_params() for x in mods)
            act = sum(x.active_params() for x in mods)
            trainable = sum(p.numel() for p in ffn_parameters(m))
            exp = mods[0].expert_params()
            rep.update({"G3_total": tot, "G3_active": act, "G3_trainable": trainable,
                        "G3_dense_site": dense_site, "G3_expert": exp, "G3_sites": len(mods)})
            # active per token must EQUAL the dense fine-tune's trainable set; that equality is
            # the entire point of the arm and L-ORE-92/93 turned on it not holding for ORE
            ok3 = (len(mods) == 8 and exp * k * len(mods) == tot - sum(
                x.gate_params() for x in mods) and act - sum(
                x.gate_params() for x in mods) == dense_site)
            print(f"[G3] sites {len(mods)} expert {exp:,} total {tot:,} active {act:,} "
                  f"trainable {trainable:,} vs dense site {dense_site:,} "
                  f"{'PASS' if ok3 else 'FAIL'}", flush=True)
            if not ok3:
                fails.append("G3 parameter audit")
        del m, mods
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- G5 + G4 + G6
    model = rc.load_model(args.init, dev)
    mods = wrap(model, args.k, 1, 0)
    model.to(dev)
    ceiling = rc.calibrate_routers(model, mods, train, size, dev, 64, kmeans=True, seed=0)
    for p in model.parameters():
        p.requires_grad_(False)
    params = ffn_parameters(model)
    for p in params:
        p.requires_grad_(True)
    model.train()
    for m in mods:
        m.tok_count.zero_()

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    touched = torch.zeros(args.k)
    effs = []
    t0 = None
    for step in range(args.steps):
        if step == 20:                       # skip warm-up/allocator noise in the rate
            torch.cuda.synchronize()
            t0 = time.time()
        idx = rng.integers(0, len(train), args.batch)
        pz = rc.predict_depth(model, [train[i][0] for i in idx], size, dev, batch=args.batch)
        gts = torch.stack([torch.from_numpy(rc.load_depth(train[i][1], *pz.shape[1:])).float()
                           for i in idx], 0).to(dev)
        task = rc.si_loss(pz, gts)
        if task is None:
            continue
        a = sum(m.aux_loss for m in mods) / len(mods)
        opt.zero_grad()
        (task + 0.1 * a).backward()
        if step < 100:
            for i, e in enumerate(mods[0].experts):
                if e.fc1.weight.grad is not None and float(e.fc1.weight.grad.norm()) > 0:
                    touched[i] = 1
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        effs.append(np.mean([m.load_stats()["effective_experts"] for m in mods]))
        for m in mods:
            m.update_bias(0.005)                 # zeroes tok_count, so read load_stats first
        if step % 50 == 0:
            print(f"[gate] step {step} task {float(task):.4f} "
                  f"eff {mods[0].load_stats()['effective_experts']:.2f}", flush=True)
    torch.cuda.synchronize()
    sec = (time.time() - t0) / (args.steps - 20)
    div = mods[0].expert_divergence()
    reach = int(touched.sum())
    ok4 = reach == args.k and abs(div["expert_cos"]) < 0.999
    rep.update({"G4_reached": reach, "G4_cos": div["expert_cos"], "G6_sec_per_step": sec,
                "G6_projected_steps": int(args.wall_hours * 3600 / sec),
                "G6_peak_gb": torch.cuda.max_memory_allocated() / 1e9})
    print(f"[G4] experts with nonzero grad {reach}/{args.k}, expert cos {div['expert_cos']:+.4f} "
          f"{'PASS' if ok4 else 'FAIL'}", flush=True)
    if not ok4:
        fails.append(f"G4 reached {reach} cos {div['expert_cos']:.4f}")
    # G5 is judged against the CEILING a k-means partition can reach on a real batch, not
    # against K: the token cloud's effective rank means 4 frames cannot reach 8 groups, and
    # run_cell's own abort gate uses 0.35 x ceiling for exactly that reason. Averaged over the
    # first 10 optimiser steps, because a single batch's reading swings by 3x.
    eff10, effN = float(np.mean(effs[:10])), float(np.mean(effs[-10:]))
    rep.update({"G5_eff_first10": eff10, "G5_eff_last10": effN, "G5_ceiling": ceiling})
    ok5 = eff10 >= 0.5 * ceiling
    print(f"[G5] effective experts {eff10:.2f} (first 10 steps) -> {effN:.2f} (last 10), "
          f"ceiling {ceiling:.1f}, floor {0.5 * ceiling:.1f} {'PASS' if ok5 else 'FAIL'}",
          flush=True)
    if not ok5:
        fails.append(f"G5 effective {eff10:.2f} vs ceiling {ceiling:.1f}")
    proj = rep["G6_projected_steps"]
    ok6 = proj >= args.min_steps
    print(f"[G6] {sec:.3f} s/step -> {proj:,} steps in {args.wall_hours} h, peak "
          f"{rep['G6_peak_gb']:.1f} GB {'PASS' if ok6 else 'FAIL'}", flush=True)
    if not ok6:
        fails.append(f"G6 projected {proj}")

    rep["FAILS"] = fails
    if args.out:
        json.dump(rep, open(args.out, "w"), indent=2)
    print("[gate] " + ("ALL GATES PASS" if not fails else "GATES FAILED: " + "; ".join(fails)),
          flush=True)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
