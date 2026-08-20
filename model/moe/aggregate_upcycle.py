#!/usr/bin/env python3
"""Read the arm-U cells against the existing 150k controls and apply the preregistered rule.

The controls are NOT re-run: the cells are hyperparameter-identical to g_150k.sb, so
out/data/L150_*.json is the comparison set. Reads .json or .json.partial, whichever is newer,
because a wall-killed cell keeps a usable curve.

Rules this encodes, all from log.md:
  - never quote an arm-vs-arm number without its step (L-ORE-91)
  - compare at the LARGEST COMMON step, and judge a delta against the last-5-eval scatter of the
    curve rather than one eval (L-ORE-45, eval-to-eval sd 0.0025 on unseen)
  - near-field 0.2-0.5 m is the primary metric; it is the only one that separated the arms
    (L-ORE-76) and the regime C4 collisions come from
"""
import argparse
import glob
import json
import os

import numpy as np

CONTROLS = {"dense FT": "L150_ft", "dense widen": "L150_widen_matched",
            "ORE K=32 unfroz": "L150_ore_k32_unf", "ORE K=1 r128": "L150_k1_r128",
            "ORE K=32 r128": "L150_ore_k32_r128"}
CELLS = ["U8_top1", "U8_top2", "U8_council3", "U2_top1"]


def load(path):
    for p in (path, path + ".partial"):
        if os.path.exists(p):
            try:
                return json.load(open(p))
            except json.JSONDecodeError:          # a partial written mid-flush
                continue
    return None


def curve(r):
    return {c["step"]: c for c in r.get("curve", []) if c.get("whole") is not None}


def scatter(cv, key, n=5):
    v = [c[key] for c in list(cv.values())[-n:] if c.get(key) is not None]
    return float(np.std(v)) if len(v) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upcycle", default="out/upcycle")
    ap.add_argument("--controls", default="out/data")
    ap.add_argument("--at", default="", help="comma-separated steps to also tabulate")
    args = ap.parse_args()

    runs = {}
    for name, tag in CONTROLS.items():
        r = load(os.path.join(args.controls, tag + ".json"))
        if r:
            runs[name] = r
    for tag in CELLS:
        r = load(os.path.join(args.upcycle, tag + ".json"))
        if r:
            runs[tag] = r
    for p in sorted(glob.glob(os.path.join(args.upcycle, "*.json"))):
        tag = os.path.basename(p)[:-5]
        if tag not in runs and tag not in ("gates",) and not tag.startswith("G2"):
            r = load(p)
            if r:
                runs[tag] = r
    if not runs:
        print("no runs found")
        return

    cvs = {k: curve(v) for k, v in runs.items()}
    live = {k: v for k, v in cvs.items() if v}
    common = set.intersection(*[set(v) for v in live.values()]) if live else set()
    steps = [max(common)] if common else []
    steps += [int(s) for s in args.at.split(",") if s.strip()]

    print(f"{'arm':<18}{'last step':>10}{'whole':>9}{'near':>9}{'unseen':>9}"
          f"{'sd near':>9}{'s/step':>8}{'K_eff':>7}")
    for k, cv in cvs.items():
        if not cv:
            print(f"{k:<18}{'(no curve)':>10}")
            continue
        s = max(cv)
        c = cv[s]
        ls = (runs[k].get("load_trace") or [{}])[-1].get("effective_experts", float("nan"))
        print(f"{k:<18}{s:>10}{c['whole']:>9.4f}{c['near_rel']:>9.4f}"
              f"{(c.get('unseen') or float('nan')):>9.4f}{scatter(cv, 'near_rel'):>9.4f}"
              f"{runs[k].get('sec_per_step', float('nan')):>8.3f}{ls:>7.1f}")

    for s in sorted(set(steps)):
        rows = {k: cv[s] for k, cv in cvs.items() if s in cv}
        if len(rows) < 2:
            continue
        print(f"\n--- step {s:,} ({len(rows)} arms) ---")
        for k, c in sorted(rows.items(), key=lambda kv: kv[1]["near_rel"]):
            print(f"  {k:<18} whole {c['whole']:.4f}  near {c['near_rel']:.4f}  "
                  f"unseen {(c.get('unseen') or float('nan')):.4f}")

    # ---- preregistered decision rule, spec section 5
    if common:
        s = max(common)
        d = {k: cv[s] for k, cv in cvs.items() if s in cv}
        dense = d.get("dense FT")
        us = {k: v for k, v in d.items() if k in CELLS}
        if dense and us:
            sd = max(scatter(cvs[k], "near_rel") for k in us)
            best = min(us.items(), key=lambda kv: kv[1]["near_rel"])
            print(f"\n[rule] at step {s:,}, near-field scatter {sd:.4f}")
            print(f"[rule] best U cell {best[0]} near {best[1]['near_rel']:.4f} vs dense "
                  f"{dense['near_rel']:.4f}, unseen {best[1].get('unseen'):.4f} vs "
                  f"{dense.get('unseen'):.4f}")
            # The step check comes FIRST. Ordering it after the GO test printed "GO" at 27,000
            # steps, which is the exact error L-ORE-91 records: ORE led by 15% at 3k, crossed at
            # ~30k and ended the worst of three at 147k. No verdict before the crossover.
            if s < 100000:
                print(f"[rule] UNDECIDED: only {s:,} steps, and the rule needs >= 100k "
                      "(the ORE crossover is ~30k, L-ORE-56). Any ordering seen here is "
                      "pre-crossover and must not be recorded as a result.")
            elif best[1]["near_rel"] < dense["near_rel"] - sd:
                print("[rule] GO: a U cell beats dense on near-field. Next: C4 on the 30k and "
                      "final snapshots BEFORE any claim, then matched-latency on the Nano.")
            elif ("U8_top2" in d and "U8_top1" in d
                  and d["U8_top2"]["near_rel"] < d["U8_top1"]["near_rel"] - sd):
                print("[rule] CONDITIONAL: top-2 beats top-1 beyond scatter but neither beats "
                      "dense. The top-k axis is real; next cells are the dense-preserving "
                      "variants (always-on shared expert, residual-demand gate).")
            else:
                print("[rule] STOP candidate: no U cell beats dense. Check whole and unseen too, "
                      "then close the MoE direction and keep the Orin inversion + the clean "
                      "negative for the systems paper.")


if __name__ == "__main__":
    main()
