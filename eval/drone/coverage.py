"""Obstacle coverage: the one offline-computable quantity that predicts C4 success.

coverage = obs_vox_max / (obstacle bbox surface area / voxel^2), i.e. how many voxel
layers thick a crust the live map builds on the injected obstacle, normalised for object
size. The planner's only obstacle test is any_occupied_within(p, safety), so a crust with
holes lets the drone aim at a hole and the test never fires.

Reads the campaign's obstacles.jsonl rows and writes a per-cell table, then reports
class-demeaned and leave-one-class-out R2 against C4 success for coverage and for the
depth metrics it replaces.

  python -m eval.drone.coverage --roots out/c4_* --out results/drone/coverage
"""
import argparse
import glob
import json
import os

import numpy as np

VOX = 0.10
VOID = {"px4_failsafe", "px4_frozen"}
# final-eval depth metrics at step 147k, for the arms that were flown (log L-FCR-02/21/32)
DEPTH = {"dense_s0": (0.0927, 0.1156), "widen_s0": (0.0873, 0.1185),
         "top2_s0": (0.0862, 0.1323), "top1_s0": (0.0957, 0.1457),
         "dense_s1": (0.0903, 0.1164), "top1_s1": (0.0904, 0.1293)}


def arm_of(p):
    """Cell directory -> arm identity. Campaign roots name the arm inconsistently."""
    if "c4_seed1/S1_top1" in p:    return "top1_s1"
    if "c4_seed1/S1_ft" in p:      return "dense_s1"
    if "c4_matched/HN_top1" in p:  return "top1_s0"
    if "c4_top1_classes/ft2" in p: return "dense_s0"
    if "/top1/" in p:              return "top1_s0"
    if "/top2/" in p or "HN_top2" in p: return "top2_s0"
    if "HN_widen" in p:            return "widen_s0"
    if "/ft/" in p or "HN_ft" in p: return "dense_s0"
    return None


def obstacle_bboxes(roots):
    """class -> bbox extents, read from any episode log that placed that object."""
    bb = {}
    for r in roots:
        for f in glob.glob(os.path.join(r, "**", "episode_logs", "*.json"), recursive=True):
            for o in json.load(open(f)).get("obstacles", []):
                c = os.path.basename(o["handle"]).replace(".object_config.json", "")
                bb.setdefault(c.replace("frl_apartment_", ""), o["bbox"])
    return bb


def build(roots, min_n=15):
    bb = obstacle_bboxes(roots)
    rows = {}
    for r in roots:
        for f in glob.glob(os.path.join(r, "**", "obstacles.jsonl"), recursive=True):
            arm = arm_of(f)
            cls = next((c for c in sorted(bb, key=len, reverse=True) if c in f), None)
            if arm is None or cls is None:
                continue
            for line in open(f):
                if not line.strip():
                    continue
                # a voided row is not a navigation result (PX4 SIH degradation)
                d = json.loads(line.replace("Infinity", '"1e9"'))
                if d.get("arm") != "student" or d.get("ended") in VOID:
                    continue
                rows.setdefault((arm, cls), []).append(d)
    out = []
    for (arm, cls), R in sorted(rows.items()):
        if len(R) < min_n:
            continue
        b = np.asarray(bb[cls], float)
        skin = 2 * (b[0]*b[1] + b[1]*b[2] + b[0]*b[2]) / VOX**2
        vox = float(np.mean([d.get("obs_vox_max", 0) for d in R]))
        out.append(dict(arm=arm, cls=cls, n=len(R),
                        succ=float(np.mean([d["success"] for d in R])),
                        vox=vox, coverage=vox / skin,
                        whole=DEPTH[arm][0], near=DEPTH[arm][1]))
    return out


def _demean(v, g, w):
    o = np.asarray(v, float).copy()
    for k in set(g):
        i = g == k
        o[i] -= np.average(o[i], weights=w[i])
    return o


def _wfit(X, y, w):
    A = np.column_stack([X, np.ones(len(X))])
    b, *_ = np.linalg.lstsq(A * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)
    return b, A @ b


def score(cells, name):
    """Class-demeaned and leave-one-class-out R2 on logit success, binomial weights."""
    y = np.array([c["succ"] for c in cells])
    n = np.array([c["n"] for c in cells], float)
    cls = np.array([c["cls"] for c in cells])
    x = np.array([c[name] for c in cells], float)
    # Haldane-corrected logit keeps the 0.000 and 1.000 cells; weights stop a 20-episode
    # cell outvoting a 200-episode one
    yl = np.log((y * n + .5) / ((1 - y) * n + .5))
    w = n * np.clip(y, .02, .98) * (1 - np.clip(y, .02, .98))
    xd, yd = _demean(x, cls, w), _demean(yl, cls, w)
    b, pred = _wfit(xd[:, None], yd, w)
    within = 1 - (np.average((yd - pred)**2, weights=w)
                  / np.average((yd - np.average(yd, weights=w))**2, weights=w))
    err, tot = [], []
    for k in set(cls):
        te, tr = cls == k, cls != k
        if te.sum() < 2 or tr.sum() < 6:
            continue
        bb, _ = _wfit(xd[tr][:, None], yd[tr], w[tr])
        p = np.column_stack([xd[te], np.ones(te.sum())]) @ bb
        err += list(np.sqrt(w[te]) * (yd[te] - p))
        tot += list(np.sqrt(w[te]) * (yd[te] - np.average(yd[tr], weights=w[tr])))
    err, tot = np.array(err), np.array(tot)
    return float(b[0]), float(within), float(1 - (err**2).sum() / (tot**2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", default="results/drone/coverage")
    a = ap.parse_args()
    cells = build(a.roots)
    os.makedirs(a.out, exist_ok=True)
    json.dump(cells, open(os.path.join(a.out, "cells.json"), "w"), indent=1)
    print(f"{len(cells)} cells, {sum(c['n'] for c in cells)} episodes, "
          f"{len(set(c['arm'] for c in cells))} arms, {len(set(c['cls'] for c in cells))} classes")
    print(f"{'predictor':12s} {'slope':>8s} {'within-class R2':>16s} {'LOCO CV R2':>12s}")
    for p in ("coverage", "near", "whole"):
        s, wi, cv = score(cells, p)
        print(f"{p:12s} {s:8.2f} {wi:16.3f} {cv:12.3f}")


if __name__ == "__main__":
    main()
