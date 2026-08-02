"""Post-episode frame capture by deterministic replay.

Re-renders the episode's recorded poses after it ends but BEFORE obstacles are cleared.
Habitat is deterministic given pose + scene state, so these frames are what the model saw --
without paying image I/O inside the control loop, which would distort the very staleness and
speed the experiment measures.
"""
import os

import numpy as np


def _pick(log, n, events=None):
    """Sample poses, biased to the approach into a collision -- that is where the story is."""
    if not log:
        return []
    idx = list(np.linspace(0, len(log) - 1, min(n, len(log))).astype(int))
    for e in (events or []):
        t = e.get("t")
        if t is None:
            continue
        j = int(np.argmin([abs(k["t"] - t) for k in log]))
        idx += [max(0, j - 2), max(0, j - 1), j]
    return sorted(set(int(i) for i in idx))


def capture(client, ep, log, out_dir, tag, res, perc=None, n=8, hi_res=None, events=None):
    """Save model-res frames (and optionally a high-res 'beauty' pass) at recorded poses."""
    os.makedirs(out_dir, exist_ok=True)
    picks = _pick(log, n, events)
    frames = []
    for i in picks:
        k = log[i]
        pos_hab = ep.ned_to_hab_pos(np.asarray(k["pos_ned"], float))
        R = ep.ned_quat_to_hab_R(np.asarray(k["quat_frd2ned"], float))
        out = client.render(pos_hab, R)
        rec = {"i": int(i), "t": float(k["t"]), "rgb": out["rgb"],
               "gt_depth": out["depth"], "K": out["K"]}
        if perc is not None:
            rec["pred_depth"] = perc.depth(out["rgb"], res=res)
        frames.append(rec)
    path = os.path.join(out_dir, f"frames_{tag}.npz")
    np.savez_compressed(
        path,
        idx=np.array([f["i"] for f in frames]),
        t=np.array([f["t"] for f in frames]),
        rgb=np.stack([f["rgb"] for f in frames]) if frames else np.zeros((0,)),
        gt_depth=np.stack([f["gt_depth"] for f in frames]) if frames else np.zeros((0,)),
        pred_depth=(np.stack([f["pred_depth"] for f in frames])
                    if frames and "pred_depth" in frames[0] else np.zeros((0,))))
    return {"path": path, "n": len(frames), "picks": picks}
