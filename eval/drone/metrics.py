"""Closed-loop nav metrics + bootstrap aggregation. boot_ci/aggregate mirror
eval/sim/run_experiments.py so drone results are reported the same way as the ground sim.
"""
import numpy as np

def spl(success, geo, path_len):
    if not success:
        return 0.0
    return float(geo) / max(float(path_len), float(geo), 1e-6)

def boot_ci(xs, n=2000, seed=0):
    xs = np.asarray([x for x in xs if x is not None and np.isfinite(x)], float)
    if xs.size == 0:
        return (float('nan'), float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    bs = [rng.choice(xs, xs.size, replace=True).mean() for _ in range(n)]
    return (float(xs.mean()), float(np.percentile(bs,2.5)), float(np.percentile(bs,97.5)))

def aggregate(rows, metrics):
    out = {}
    # Episodes PX4 voided (failsafe took control, or the drone never flew) say nothing about
    # navigation, and averaging them in drags every arm down together -- which reads as a
    # harder scene set rather than a dead simulator.
    rows = [r for r in rows if r.get("ended") not in ("px4_failsafe", "px4_frozen")]
    for a in sorted({r["arm"] for r in rows}):
        rr = [r for r in rows if r["arm"] == a]
        agg = {"n": len(rr)}
        for m in metrics:
            mean, lo, hi = boot_ci([r.get(m) for r in rr])
            agg[m] = {"mean": round(mean,4), "ci": [round(lo,4), round(hi,4)]}
        out[a] = agg
    return out
