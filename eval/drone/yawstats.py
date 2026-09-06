#!/usr/bin/env python3
"""C2 test: is obstacle yaw a nuisance variable for LEARNED depth but not for a GT oracle?

Reads the yawgen cells (out/yawgen/<class>_yaw<Y>/) and reports, per class, success vs yaw for
every arm. All arms fly the SAME layouts inside a cell, so the comparison is paired by
construction and no seed matching is needed -- only the per-arm rate per cell.

The claim under test (log L-FCR-17): obstacle presentation geometry is an uncontrolled variable
in closed-loop obstacle-avoidance benchmarks for learned monocular depth. The decisive number is
the SWING, max-min success across yaw, per arm:

    if swing(gt) ~ 0 while swing(student), swing(vggt) and swing(da) are all large
        -> the effect is a property of monocular-depth navigation, not of our student. C2 lives.
    if only swing(student) is large
        -> C2 is a self-critique of our checkpoint and dies.

`gt` is the oracle control and the whole experiment hinges on it staying flat; `vggt` and `da` are
INDEPENDENT published models (VGGT-1B, Depth-Anything-V2). The MapAnything teacher is deliberately
absent -- it is the student's own teacher and would prove nothing about generality.

Also prints mean_speed per cell. Every closed-loop number in this repo was flown at ~0.177; a cell
that drifts off that was flown on different dynamics and must not be pooled with the rest.
"""
import collections
import glob
import json
import os
import re
import sys

VOID = ("px4_failsafe", "px4_frozen")
ARM_ORDER = ["gt", "student", "vggt", "da"]


def perm_p(data, cls, arm, yaws, iters=20000, seed=0):
    """P(swing >= observed) when yaw labels are shuffled, holding each cell's n fixed.

    Answers the only question that matters here: given how few episodes per cell we fly, could
    this much yaw-to-yaw spread have arisen with yaw carrying no information at all? Exact,
    assumption-free, and needs no scipy.
    """
    import random as _r
    rng = _r.Random(seed)
    pool, sizes = [], []
    for y in yaws:
        ar = [r for r in data[(cls, y)] if r.get("arm") == arm]
        # A yaw still in flight can have rows for some arms and none for others. Skip those
        # rather than bailing to nan -- otherwise one partially-written cell blanks the whole
        # column and hides a verdict that the completed yaws can already support.
        if not ar:
            continue
        sizes.append(len(ar))
        pool += [bool(r.get("success")) for r in ar]
    if len(sizes) < 2:
        return float("nan")
    if len(set(pool)) < 2:          # arm is all-success or all-fail: no swing possible
        return 1.0

    def swing_of(seq):
        out, i = [], 0
        for n in sizes:
            out.append(sum(seq[i:i + n]) / n)
            i += n
        return max(out) - min(out)

    obs = swing_of(pool)
    hits = 0
    for _ in range(iters):
        rng.shuffle(pool)
        if swing_of(pool) >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (iters + 1)


def fmt_p(p):
    return " nan " if p != p else f"{p:.4f}" if p >= 1e-4 else "<1e-4"


def cells(roots):
    """(class, yaw) -> list of episode rows, voids dropped."""
    out = collections.defaultdict(list)
    for root in roots:
        for p in glob.glob(os.path.join(root, "**", "obstacles.jsonl"), recursive=True):
            m = re.search(r"([A-Za-z_0-9]+)_yaw(\d+)", p)
            if not m:
                continue
            cls, yaw = m.group(1), int(m.group(2))
            for line in open(p):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("ended") in VOID:
                    continue
                out[(cls, yaw)].append(r)
    return out


def main():
    roots = sys.argv[1:] or sorted(glob.glob("out/yawgen*"))
    data = cells(roots)
    if not data:
        print("no yawgen cells found under: " + ", ".join(roots))
        return
    classes = sorted({c for c, _ in data})
    swings = collections.defaultdict(dict)

    for cls in classes:
        yaws = sorted(y for c, y in data if c == cls)
        print("=" * 74)
        print(f"{cls}")
        print(f"  {'yaw':>5} | " + " | ".join(f"{a:>13}" for a in ARM_ORDER) + " | speed  void")
        rates = collections.defaultdict(dict)
        for y in yaws:
            rows = data[(cls, y)]
            cellstr = []
            for a in ARM_ORDER:
                ar = [r for r in rows if r.get("arm") == a]
                if ar:
                    s = sum(bool(r.get("success")) for r in ar) / len(ar)
                    rates[a][y] = s
                    cellstr.append(f"{s:>7.3f} n={len(ar):<3}")
                else:
                    cellstr.append(f"{'--':>13}")
            sp = [r.get("mean_speed") for r in rows if isinstance(r.get("mean_speed"), (int, float))]
            spd = f"{sum(sp)/len(sp):.3f}" if sp else "  --  "
            nv = sum(1 for r in rows if r.get("ended") in VOID)
            print(f"  {y:>5} | " + " | ".join(cellstr) + f" | {spd} {nv}")
        for a in ARM_ORDER:
            if len(rates[a]) >= 2:
                v = list(rates[a].values())
                swings[cls][a] = max(v) - min(v)
        if swings[cls]:
            print("  swing (max-min across yaw): "
                  + "  ".join(f"{a}={swings[cls][a]:.3f}" for a in ARM_ORDER if a in swings[cls]))
            # RAW SWING IS NOT INTERPRETABLE ON ITS OWN. At n=12/cell and p~0.5 the binomial SD is
            # 0.144, so five yaws of PURE NOISE give an expected max-min near 0.40. A synthetic
            # null (flat vggt/da) produced swings of 0.50 and 0.33 and would have been misread as
            # "every arm swings". Calibrate against shuffling the yaw labels at the observed n.
            print("  permutation p (yaw carries no signal): " + "  ".join(
                f"{a}=" + fmt_p(perm_p(data, cls, a, yaws))
                for a in ARM_ORDER if a in swings[cls]))

    print("=" * 74)
    print("VERDICT INPUTS -- swing per arm, per class (higher = more yaw-sensitive)")
    for cls in classes:
        if cls in swings:
            print(f"  {cls:<22} " + "  ".join(f"{a}={swings[cls].get(a, float('nan')):.3f}"
                                              for a in ARM_ORDER if a in swings[cls]))
    # Verdict is decided on the PERMUTATION p, never on raw swing (see note above).
    learned = [a for a in ARM_ORDER if a != "gt"]
    print("\n  permutation p per class/arm:")
    sig = collections.defaultdict(list)
    for cls in classes:
        yaws = sorted(y for c, y in data if c == cls)
        ps = {a: perm_p(data, cls, a, yaws) for a in ARM_ORDER if a in swings.get(cls, {})}
        print(f"    {cls:<22} " + "  ".join(f"{a}={fmt_p(p)}" for a, p in ps.items()))
        for a, p in ps.items():
            if p == p:
                sig[a].append(p < 0.05)
    print("\n  VERDICT")
    gt_ok = sig.get("gt") and not any(sig["gt"])
    ind = [a for a in ("vggt", "da") if sig.get(a) and all(sig[a])]
    print(f"    GT oracle flat (no significant yaw effect): {bool(gt_ok)}")
    print(f"    independent published models that DO swing significantly: "
          f"{ind if ind else 'NONE'}")
    if gt_ok and len(ind) == 2:
        print("    -> C2 SURVIVES: yaw sensitivity is a property of monocular-depth navigation,")
        print("       not of our checkpoint. This is the paper.")
    elif not ind:
        print("    -> C2 DIES: only our student is yaw-sensitive, so this is a self-critique")
        print("       of one checkpoint, not a benchmark-validity claim. Record it and stop.")
    else:
        print("    -> PARTIAL: mixed evidence. Do NOT write the claim; state exactly which arms")
        print("       swung and treat the rest as unresolved.")


if __name__ == "__main__":
    main()
