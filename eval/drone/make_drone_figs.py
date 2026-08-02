#!/usr/bin/env python3
"""Figures for the drone-in-sim experiments (C2 scale, C4 nav, staleness ablation).

Reads the summary JSON / JSONL written by run_experiments.py and renders clean, minimal
matplotlib figures to results/drone/. Targets are marked (C2 <5%, C4 >=80%). C3 (drift)
is added once the chained-VO mode lands.

  python eval/drone/make_drone_figs.py --results <dir with c4_summary.json etc> --out results/drone
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 11, "figure.dpi": 130})
STUDENT = "#2b6cb0"
GT = "#a0aec0"
TARGET = "#e53e3e"


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def fig_c4(results, out):
    summ = _load(os.path.join(results, "c4_summary.json"))
    if not summ:
        return
    arms = [a for a in ("student", "gt") if a in summ]
    metrics = ["success", "spl"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(7, 3.2))
    for ax, m in zip(axes, metrics):
        vals = [summ[a][m]["mean"] for a in arms]
        cis = [summ[a][m]["ci"] for a in arms]
        err = [[v - lo for v, (lo, hi) in zip(vals, cis)], [hi - v for v, (lo, hi) in zip(vals, cis)]]
        colors = [STUDENT if a == "student" else GT for a in arms]
        ax.bar(range(len(arms)), vals, yerr=err, color=colors, width=0.6, capsize=4)
        if m == "success":
            ax.axhline(0.8, color=TARGET, ls="--", lw=1)
            ax.text(len(arms) - 0.5, 0.81, "target 0.80", color=TARGET, fontsize=8, ha="right")
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(["MERLIN" if a == "student" else "GT depth" for a in arms])
        ax.set_ylim(0, 1.05)
        ax.set_title(m.upper() if m == "spl" else "Success rate")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "c4_nav.png"))
    plt.close(fig)


def fig_c4_control(results, out):
    """Three-arm C4 with the blind control. The control has an always-empty perception map, so
    it isolates how much of C4's success comes from the (collision-free) navmesh route rather
    than from the live map."""
    rows = []
    for f in ("c4_episodes.jsonl", "c4_blind.jsonl"):
        p = os.path.join(results, f)
        if os.path.exists(p):
            rows += [json.loads(l) for l in open(p)]
    arms = [a for a in ("blind", "student", "gt") if any(r["arm"] == a for r in rows)]
    if len(arms) < 2:
        return
    labels = {"blind": "no map\n(route only)", "student": "MERLIN", "gt": "GT depth"}
    colors = {"blind": "#dd6b20", "student": STUDENT, "gt": GT}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    succ, coll = [], []
    for a in arms:
        r = [x for x in rows if x["arm"] == a]
        succ.append(np.mean([x["success"] for x in r]))
        coll.append(sum(x["collisions"] for x in r) / sum(x["path_len"] for x in r))
    axes[0].bar(range(len(arms)), succ, color=[colors[a] for a in arms], width=0.6)
    axes[0].set_ylim(0, 1.05); axes[0].set_title("Success rate")
    axes[1].bar(range(len(arms)), coll, color=[colors[a] for a in arms], width=0.6)
    axes[1].set_title("Collisions per metre")
    for ax in axes:
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([labels[a] for a in arms], fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "c4_blind_control.png"))
    plt.close(fig)


def _episode_frame(d):
    """Rebuild the episode frame. Uses the stored start/yaw when present, else reconstructs
    yaw from start->goal exactly as renderer.sample_episode does (older logs lack the field)."""
    from eval.drone.frames import EpisodeFrame
    W = np.array(d["waypoints_hab"], float)
    start = d.get("start_hab") or list(W[0])
    yaw = d.get("yaw_hab")
    if yaw is None:
        dv = W[-1] - W[0]
        yaw = float(np.arctan2(dv[0], -dv[2]))
    return EpisodeFrame(list(start), float(yaw))


def _to_hab_xz(ep, pos_ned):
    """Episode-relative NED -> habitat XZ. The map, obstacles and floor plan are all in the
    habitat frame, so the flown path must be converted or the figure mixes frames."""
    h = ep.ned_to_hab_pos(np.asarray(pos_ned, float))
    return np.array([h[0], h[2]])


def _topdown(results, scene):
    """Floor plan for `scene`, wherever the driver wrote it (obstacle runs nest it one level
    down under obs_<n>/)."""
    import glob
    hits = glob.glob(os.path.join(results, "**", f"topdown_{scene}.npz"), recursive=True)
    if not hits:
        return None
    z = np.load(hits[0])
    # grid is [z_row, x_col] and bounds are [[x_min, z_min], [x_max, z_max]]; transposing it
    # swaps the two spans and silently stretches the plan across the wrong axis
    return np.asarray(z["grid"]), np.asarray(z["bounds"], float)


def fig_paths(results, out, extra_results=()):
    """Top-down floor plan, planned route, and the flown path of every arm on one episode.

    The paths are what the drone actually did; the dashed line is the route it was given.
    Arms that only track the route sit on top of it, and an arm using its live map visibly
    departs from it.
    """
    import glob
    dirs = [results] + list(extra_results)
    logs = []
    for d in dirs:
        logs += sorted(glob.glob(os.path.join(d, "**", "episode_logs", "*.json"),
                                 recursive=True))
    if not logs:
        return 0
    byep = {}
    for p in logs:
        d = json.load(open(p))
        # scene names contain underscores, so the tag is not safely splittable
        key = f"{d.get('scene', 'scene')}_{d.get('seed', 0)}_n{d['n_obstacles']}"
        byep.setdefault(key, []).append(d)
    order = ["blind", "random", "gt", "student"]
    colors = {"blind": "#dd6b20", "random": "#805ad5", "student": STUDENT, "gt": "#38a169"}
    names = {"blind": "no map", "random": "random depth", "student": "MERLIN",
             "gt": "GT depth"}
    made = 0
    for key, arms in sorted(byep.items()):
        arms = sorted(arms, key=lambda d: order.index(d["arm"]) if d["arm"] in order else 9)
        if len(arms) < 2:
            continue
        fig, ax = plt.subplots(figsize=(4.6, 4.6))
        td = _topdown(results, arms[0].get("scene", ""))
        if td is not None:
            grid, b = td
            # shade what the drone CANNOT occupy: free space reads as white, walls and
            # furniture as grey, which is how a floor plan is normally read
            ax.imshow(~grid.astype(bool), cmap="Greys", alpha=0.28, origin="lower",
                      interpolation="nearest", vmin=0, vmax=1,
                      extent=[b[0][0], b[1][0], b[0][1], b[1][1]], zorder=0)
        W = np.array(arms[0]["waypoints_hab"], float)
        ax.plot(W[:, 0], W[:, 2], "--", color="#4a5568", lw=1.1, dashes=(4, 3), zorder=2,
                label="planned route")
        for o in arms[0].get("obstacles", []):
            r = 0.5 * max(o["bbox"][0], o["bbox"][2])
            ax.add_patch(plt.Circle((o["pos"][0], o["pos"][2]), r, color="#2d3748",
                                    alpha=0.40, zorder=1))
        xs_all, zs_all = [W[:, 0]], [W[:, 2]]
        for d in arms:
            pts = [e["pos_ned"] for e in d.get("log", [])]
            if not pts:
                continue
            ep = _episode_frame(d)
            p = np.array([_to_hab_xz(ep, x) for x in pts], float)
            ax.plot(p[:, 0], p[:, 1], color=colors.get(d["arm"], "#000"), lw=1.7, zorder=3,
                    label=names.get(d["arm"], d["arm"]), solid_capstyle="round")
            xs_all.append(p[:, 0]); zs_all.append(p[:, 1])
            for e in d.get("collision_events", []):
                c = _to_hab_xz(ep, e["pos_ned"])
                ax.plot(c[0], c[1], "x", color=TARGET, ms=9, mew=2.2, zorder=5)
        ax.plot(W[0, 0], W[0, 2], "o", color="#1a202c", ms=6, zorder=4)
        ax.plot(W[-1, 0], W[-1, 2], "*", color="#1a202c", ms=13, zorder=4)
        # an episode succeeds on reaching goal_tol of the goal, so paths legitimately stop
        # short of the star; without the circle that reads as a failure
        tol = float((arms[0].get("config") or {}).get("goal_tol", 1.0))
        ax.add_patch(plt.Circle((W[-1, 0], W[-1, 2]), tol, fill=False, ls=":", lw=1.0,
                                ec="#1a202c", alpha=0.55, zorder=4))
        # crop to what was actually flown, or the room swallows the trajectory
        xs = np.concatenate(xs_all); zs = np.concatenate(zs_all)
        m = 1.0
        ax.set_xlim(xs.min() - m, xs.max() + m)
        ax.set_ylim(zs.min() - m, zs.max() + m)
        ax.set_aspect("equal")
        ax.legend(fontsize=8, frameon=False, loc="best")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"path_{key}.png"))
        plt.close(fig)
        made += 1
    return made


def fig_path_disagree(results, out, extra_results=(), fname="paths_disagree.png"):
    """Only the episodes where the arms disagree.

    Where every arm succeeds, a path figure shows four lines on top of each other and says
    nothing. The episodes worth plotting are the ones where the depth input changed the
    outcome -- in both directions, since perception hurt more often than it helped here.
    """
    import glob
    dirs = [results] + list(extra_results)
    logs = []
    for d in dirs:
        logs += sorted(glob.glob(os.path.join(d, "**", "episode_logs", "*.json"),
                                 recursive=True))
    byep = {}
    for p in logs:
        d = json.load(open(p))
        byep.setdefault((d.get("scene"), d.get("seed")), {})[d["arm"]] = d
    # only the arms this run actually flew: a fixed list silently matched nothing whenever a
    # run omitted one (the random-depth control is not part of every experiment)
    order = [a for a in ("blind", "random", "gt", "student")
             if any(a in v for v in byep.values())]
    picks = [k for k, v in sorted(byep.items())
             if all(a in v for a in order)
             and 0 < sum(v[a]["success"] > .5 for a in order) < len(order)]
    if not picks:
        return 0
    colors = {"blind": "#dd6b20", "random": "#805ad5", "student": STUDENT, "gt": "#38a169"}
    names = {"blind": "no map", "random": "random depth", "student": "MERLIN",
             "gt": "GT depth"}
    ncol = min(3, len(picks))
    nrow = int(np.ceil(len(picks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, key in zip(axes, picks):
        v = byep[key]
        td = _topdown(results, key[0])
        if td is not None:
            grid, b = td
            ax.imshow(~grid.astype(bool), cmap="Greys", alpha=0.28, origin="lower",
                      interpolation="nearest", vmin=0, vmax=1,
                      extent=[b[0][0], b[1][0], b[0][1], b[1][1]], zorder=0)
        W = np.array(v[order[0]]["waypoints_hab"], float)
        ax.plot(W[:, 0], W[:, 2], "--", color="#4a5568", lw=1.0, dashes=(4, 3), zorder=2)
        xs_all, zs_all = [W[:, 0]], [W[:, 2]]
        for a in order:
            d = v[a]
            pts = [e["pos_ned"] for e in d.get("log", [])]
            if not pts:
                continue
            ep = _episode_frame(d)
            p = np.array([_to_hab_xz(ep, x) for x in pts], float)
            ok = d["success"] > .5
            ax.plot(p[:, 0], p[:, 1], color=colors[a], lw=1.6, zorder=3,
                    ls="-" if ok else "-", alpha=1.0 if ok else 0.85,
                    solid_capstyle="round")
            xs_all.append(p[:, 0]); zs_all.append(p[:, 1])
            for e in d.get("collision_events", []):
                c = _to_hab_xz(ep, e["pos_ned"])
                ax.plot(c[0], c[1], "x", color=TARGET, ms=10, mew=2.4, zorder=5)
        tol = float((v[order[0]].get("config") or {}).get("goal_tol", 1.0))
        ax.add_patch(plt.Circle((W[-1, 0], W[-1, 2]), tol, fill=False, ls=":", lw=0.9,
                                ec="#1a202c", alpha=0.5, zorder=4))
        ax.plot(W[0, 0], W[0, 2], "o", color="#1a202c", ms=5, zorder=4)
        ax.plot(W[-1, 0], W[-1, 2], "*", color="#1a202c", ms=11, zorder=4)
        xs = np.concatenate(xs_all); zs = np.concatenate(zs_all)
        m = 0.8
        ax.set_xlim(xs.min() - m, xs.max() + m); ax.set_ylim(zs.min() - m, zs.max() + m)
        ax.set_aspect("equal")
        crashed = [names[a] for a in order if v[a]["success"] < .5]
        ax.set_title(f"{key[0].replace('_', ' ')} seed {key[1]}\ncrashed: " +
                     ", ".join(crashed), fontsize=8, pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    for ax in axes[len(picks):]:
        ax.axis("off")
    handles = [plt.Line2D([], [], color="#4a5568", ls="--", lw=1.0, label="planned route")]
    handles += [plt.Line2D([], [], color=colors[a], lw=1.6, label=names[a]) for a in order]
    handles += [plt.Line2D([], [], color=TARGET, marker="x", ls="", mew=2.2, label="collision")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(out, fname))
    plt.close(fig)
    return 1


def fig_path_grid(results, out, extra_results=(), seed=1000, fname="paths_grid.png"):
    """One panel per scene: floor plan, planned route, and every arm's flown path.

    A single episode does not show whether an arm generalises; six unseen apartments on one
    sheet do, and put the arms side by side in the same visual frame.
    """
    import glob
    dirs = [results] + list(extra_results)
    logs = []
    for d in dirs:
        logs += sorted(glob.glob(os.path.join(d, "**", "episode_logs", "*.json"),
                                 recursive=True))
    if not logs:
        return 0
    byscene = {}
    for p in logs:
        d = json.load(open(p))
        # seed=None: one panel per episode rather than per scene, for runs where each
        # scene+seed carries a different obstacle layout
        if seed is not None and d.get("seed") != seed:
            continue
        key = d.get("scene", "?") if seed is not None else f"{d.get('scene')} s{d.get('seed')}"
        byscene.setdefault(key, []).append(d)
    scenes = sorted(byscene)[:6]
    if not scenes:
        return 0
    order = ["blind", "random", "gt", "student"]
    colors = {"blind": "#dd6b20", "random": "#805ad5", "student": STUDENT, "gt": "#38a169"}
    names = {"blind": "no map", "random": "random depth", "student": "MERLIN",
             "gt": "GT depth"}
    ncol = min(3, len(scenes))
    nrow = int(np.ceil(len(scenes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.4 * nrow))
    axes = np.atleast_1d(axes).ravel()
    seen = []
    for ax, scene in zip(axes, scenes):
        arms = sorted(byscene[scene],
                      key=lambda d: order.index(d["arm"]) if d["arm"] in order else 9)
        td = _topdown(results, scene.split(" ")[0])
        if td is not None:
            grid, b = td
            ax.imshow(~grid.astype(bool), cmap="Greys", alpha=0.28, origin="lower",
                      interpolation="nearest", vmin=0, vmax=1,
                      extent=[b[0][0], b[1][0], b[0][1], b[1][1]], zorder=0)
        for o in arms[0].get("obstacles", []):
            r = 0.5 * max(o["bbox"][0], o["bbox"][2])
            ax.add_patch(plt.Circle((o["pos"][0], o["pos"][2]), r, color="#c53030",
                                    alpha=0.35, zorder=1))
        W = np.array(arms[0]["waypoints_hab"], float)
        ax.plot(W[:, 0], W[:, 2], "--", color="#4a5568", lw=1.0, dashes=(4, 3), zorder=2)
        xs_all, zs_all = [W[:, 0]], [W[:, 2]]
        for d in arms:
            pts = [e["pos_ned"] for e in d.get("log", [])]
            if not pts:
                continue
            ep = _episode_frame(d)
            p = np.array([_to_hab_xz(ep, x) for x in pts], float)
            ax.plot(p[:, 0], p[:, 1], color=colors.get(d["arm"], "#000"), lw=1.5, zorder=3,
                    solid_capstyle="round")
            xs_all.append(p[:, 0]); zs_all.append(p[:, 1])
            if d["arm"] not in seen:
                seen.append(d["arm"])
            for e in d.get("collision_events", []):
                c = _to_hab_xz(ep, e["pos_ned"])
                # an injected obstacle and the scene's own furniture are different failures:
                # the first is what the experiment injects, the second is collateral damage
                # from the detour around it
                novel = bool(e.get("is_novel"))
                ax.plot(c[0], c[1], "x", color=TARGET if novel else "#4a5568",
                        ms=9 if novel else 7, mew=2.2 if novel else 1.6, zorder=5)
        tol = float((arms[0].get("config") or {}).get("goal_tol", 1.0))
        ax.add_patch(plt.Circle((W[-1, 0], W[-1, 2]), tol, fill=False, ls=":", lw=0.9,
                                ec="#1a202c", alpha=0.5, zorder=4))
        ax.plot(W[0, 0], W[0, 2], "o", color="#1a202c", ms=5, zorder=4)
        ax.plot(W[-1, 0], W[-1, 2], "*", color="#1a202c", ms=11, zorder=4)
        xs = np.concatenate(xs_all); zs = np.concatenate(zs_all)
        m = 0.9
        ax.set_xlim(xs.min() - m, xs.max() + m)
        ax.set_ylim(zs.min() - m, zs.max() + m)
        ax.set_aspect("equal")
        nob = len(arms[0].get("obstacles", []))
        bits = []
        for n, a in (("no map", "blind"), ("random", "random"), ("GT", "gt"),
                     ("MERLIN", "student")):
            m = [d for d in arms if d["arm"] == a]
            if not m or m[0]["success"] > .5:
                continue
            ce = m[0].get("collision_events") or [{}]
            bits.append(n + ("*" if ce[0].get("is_novel") else ""))
        ax.set_title(scene.replace("_", " ") + (f"  ({nob} obstacles)" if nob else "") +
                     ("\ncrashed: " + ", ".join(bits) if bits else "\nall reached goal"),
                     fontsize=8, pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    for ax in axes[len(scenes):]:
        ax.axis("off")
    handles = [plt.Line2D([], [], color="#4a5568", ls="--", lw=1.0, label="planned route")]
    handles += [plt.Line2D([], [], color=colors[a], lw=1.6, label=names[a])
                for a in order if a in seen]
    handles += [plt.Line2D([], [], color=TARGET, marker="x", ls="", mew=2.2,
                           label="hit injected obstacle"),
                plt.Line2D([], [], color="#4a5568", marker="x", ls="", mew=1.6,
                           label="hit scene furniture")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(os.path.join(out, fname))
    plt.close(fig)
    return 1


def fig_filmstrip(results, out, tag=None):
    """RGB / GT depth / MERLIN depth across the approach to an obstacle."""
    import glob
    files = sorted(glob.glob(os.path.join(results, "**", "frames", "frames_*.npz"),
                             recursive=True))
    if not files:
        return 0
    sel_files = [p for p in files if tag and tag in p] or files
    z = np.load(sel_files[0])
    rgb, pred, gt = z["rgb"], z["pred_depth"], z["gt_depth"]
    if rgb.shape[0] == 0:
        return 0
    m = min(4, rgb.shape[0])
    sel = np.linspace(0, rgb.shape[0] - 1, m).astype(int)
    rows = 3 if pred.shape[0] else 2
    fig, axes = plt.subplots(rows, m, figsize=(2.0 * m, 2.05 * rows))
    axes = np.atleast_2d(axes)
    vmax = float(np.nanpercentile(gt[sel], 95)) if gt.shape[0] else 5.0
    for j, i in enumerate(sel):
        axes[0, j].imshow(rgb[i])
        axes[1, j].imshow(gt[i], cmap="viridis", vmin=0, vmax=vmax)
        if rows == 3:
            axes[2, j].imshow(pred[i], cmap="viridis", vmin=0, vmax=vmax)
    for a in axes.ravel():
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
    for a, lab in zip(axes[:, 0], ["RGB", "GT depth", "MERLIN"][:rows]):
        a.set_ylabel(lab, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "obstacle_filmstrip.png"), dpi=200)
    plt.close(fig)
    return 1


def fig_obstacle_sweep(results, out):
    """Success vs obstacle count per arm -- the dose-response the pivot exists to produce."""
    grid = _load(os.path.join(results, "obstacles_summary.json"))
    if not grid:
        return 0
    arms, counts = [], []
    for k in grid:
        n, a = k.split("_", 1)
        counts.append(int(n[1:])); arms.append(a)
    counts = sorted(set(counts)); arms = [a for a in ("blind", "student", "gt") if a in set(arms)]
    colors = {"blind": "#dd6b20", "student": STUDENT, "gt": GT}
    names = {"blind": "no map", "student": "MERLIN", "gt": "GT depth"}
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    for a in arms:
        ys = [grid.get(f"n{c}_{a}", {}).get("success") for c in counts]
        ax.plot(counts, ys, "o-", color=colors[a], label=names[a], lw=1.8, ms=5)
    ax.set_xticks(counts)
    ax.set_xlabel("novel obstacles on route")
    ax.set_ylabel("success rate")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "obstacle_sweep.png"))
    plt.close(fig)
    return 1


def fig_c2(results, out):
    summ = _load(os.path.join(results, "c2_summary.json"))
    rows = []
    jp = os.path.join(results, "c2_windows.jsonl")
    if os.path.exists(jp):
        rows = [json.loads(l) for l in open(jp)]
    errs = [r["scale_err_pct"] for r in rows if r.get("scale_err_pct") is not None]
    if not errs and not summ:
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    if errs:
        ax.hist(errs, bins=min(20, max(5, len(errs) // 2)), color=STUDENT, alpha=0.85)
    ax.axvline(5.0, color=TARGET, ls="--", lw=1)
    ax.text(5.1, ax.get_ylim()[1] * 0.9, "target 5%", color=TARGET, fontsize=8)
    if summ:
        med = summ["scale_err_pct"].get("median")
        if med is not None:
            ax.axvline(med, color="#2f855a", lw=1.5)
            ax.text(med, ax.get_ylim()[1] * 0.75, f"median {med:.1f}%", color="#2f855a", fontsize=8)
    ax.set_xlabel("metric scale error (%)")
    ax.set_ylabel("windows")
    ax.set_title("C2: IMU metric-scale recovery")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "c2_scale.png"))
    plt.close(fig)


def fig_ablation(results, out):
    grid = _load(os.path.join(results, "ablation_summary.json"))
    if not grid:
        return
    # keys are r<res>_s<speed> (resolution sweep) or l<latency_ms>_s<speed> (staleness sweep)
    tag = next(iter(grid))[0]
    res_set = sorted({float(k.split("_")[0][1:]) for k in grid})
    spd_set = sorted({float(k.split("_")[1][1:]) for k in grid})
    M = np.full((len(res_set), len(spd_set)), np.nan)
    for i, r in enumerate(res_set):
        for j, sp in enumerate(spd_set):
            cell = grid.get(f"{tag}{r:g}_s{sp}")
            if cell and cell.get("success") is not None:
                M[i, j] = cell["success"]
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto", origin="lower")
    ax.set_xticks(range(len(spd_set)))
    ax.set_xticklabels([f"{s:g}" for s in spd_set])
    ax.set_yticks(range(len(res_set)))
    ax.set_yticklabels([f"{r:g}" for r in res_set])
    ax.set_xlabel("flight speed (m/s)")
    ax.set_ylabel("map latency (ms)" if tag == "l" else "input resolution")
    ax.set_title("Nav success vs operating point")
    for i in range(len(res_set)):
        for j in range(len(spd_set)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "ablation_success.png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/drone")
    ap.add_argument("--out", default="results/drone")
    ap.add_argument("--also", nargs="*", default=[],
                    help="extra result dirs whose episodes belong on the same "
                         "path figure (e.g. an arm run as a separate job)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig_c4(args.results, args.out)
    fig_c4_control(args.results, args.out)
    fig_obstacle_sweep(args.results, args.out)
    fig_paths(args.results, args.out, extra_results=args.also)
    fig_path_grid(args.results, args.out, extra_results=args.also)
    fig_path_disagree(args.results, args.out, extra_results=args.also)
    # the student arm is the only one with a predicted-depth row, so name it
    fig_filmstrip(args.results, args.out, tag="student")
    fig_c2(args.results, args.out)
    fig_ablation(args.results, args.out)
    print("wrote figures to", args.out)


if __name__ == "__main__":
    main()


def fig_c4_summary(episode_files, clean_files, out, fname="c4_summary.png",
                   labels=("obstacle on route", "clean route")):
    """Success by arm, with and without a novel obstacle on the route.

    Bootstrap CIs over episode-runs; episodes PX4 voided (the drone never flew) are dropped,
    because averaging them in drags every arm down together and reads as a harder scene set.
    """
    def load(paths):
        eps = {}
        for p in paths:
            tag = os.path.basename(p)
            for line in open(p):
                if line.strip().startswith("{"):
                    r = json.loads(line)
                    eps.setdefault((tag, r["scene"], r["seed"]), {})[r["arm"]] = r
        return {k: v for k, v in eps.items()
                if not any(x["mean_speed"] < 0.05 for x in v.values())}

    def boot(v, n=20000):
        rng = np.random.default_rng(0)
        m = np.sort(rng.choice(v, (n, len(v))).mean(1))
        return float(np.mean(v)), float(m[int(.025 * n)]), float(m[int(.975 * n)])

    arms = ["blind", "student", "gt"]
    names = {"blind": "no map", "student": "MERLIN", "gt": "GT depth"}
    colors = {"blind": "#dd6b20", "student": STUDENT, "gt": "#38a169"}
    cells = [(labels[0], load(episode_files)), (labels[1], load(clean_files))]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharex=True)
    for ax, (title, eps) in zip(axes, cells):
        for i, a in enumerate(arms):
            v = [e[a]["success"] for e in eps.values() if a in e]
            if not v:
                continue
            mean, lo, hi = boot(v)
            ax.barh(i, mean, height=0.55, color=colors[a])
            ax.plot([lo, hi], [i, i], color="#2d3748", lw=1.4)
            ax.text(max(hi, mean) + 0.04, i, f"{mean:.2f}", va="center", fontsize=9,
                    color="#2d3748")
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels([names[a] for a in arms])
        ax.set_xlim(0, 1.24)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_title(f"{title}  (n={len(eps)})", fontsize=10)
        ax.invert_yaxis()
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_xlabel("success rate")
    axes[1].set_xlabel("success rate")
    fig.tight_layout()
    fig.savefig(os.path.join(out, fname), dpi=150)
    plt.close(fig)
    return 1


def fig_map_fix(weights_jsonl, obs_jsonl, out, fname="map_fix.png", prefix_block=0.455):
    """What the occupancy fix bought, and what the evidence threshold costs.

    Left: fraction of a collision-free navmesh route the map calls blocked. Every such point
    is a phantom, so GT's level is the floor. Right: how far away the injected obstacle is
    first mapped. Raising min_hits cleans the map and delays detection; the operating point
    has to buy the first without spending too much of the second.
    """
    rows = [json.loads(l) for l in open(weights_jsonl) if l.strip().startswith("{")]
    rows = [r for r in rows if r.get("w") == "1:1"]
    det = [json.loads(l) for l in open(obs_jsonl) if l.strip().startswith("{")]
    mhs = sorted({r["min_hits"] for r in rows})
    blk_s = [np.mean([r["block_st"] for r in rows if r["min_hits"] == m]) for m in mhs]
    blk_g = [np.mean([r["block_gt"] for r in rows if r["min_hits"] == m]) for m in mhs]
    dmhs = sorted({r["min_hits"] for r in det})
    def dmean(m, k):
        v = [d for r in det if r["min_hits"] == m for d in r[k] if d]
        return float(np.mean(v)) if v else np.nan
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.3))
    a1.axhline(prefix_block, color="#c53030", lw=1.3, ls="--")
    a1.text(mhs[0], prefix_block + 0.012, "before the occupancy fix", color="#c53030",
            fontsize=9, va="bottom", ha="left")
    a1.plot(mhs, blk_s, "o-", color=STUDENT, lw=1.8, ms=5, label="MERLIN")
    a1.plot(mhs, blk_g, "s-", color="#38a169", lw=1.8, ms=5, label="GT depth")
    a1.set_xlabel("min_hits (confirming keyframes)")
    a1.set_ylabel("route falsely blocked")
    a1.set_ylim(0, 0.5)
    a1.legend(frameon=False, fontsize=9)
    a2.plot(dmhs, [dmean(m, "det_st") for m in dmhs], "o-", color=STUDENT, lw=1.8, ms=5,
            label="MERLIN")
    a2.plot(dmhs, [dmean(m, "det_gt") for m in dmhs], "s-", color="#38a169", lw=1.8, ms=5,
            label="GT depth")
    a2.set_xlabel("min_hits (confirming keyframes)")
    a2.set_ylabel("obstacle first mapped (m)")
    a2.legend(frameon=False, fontsize=9)
    for ax in (a1, a2):
        ax.set_xticks(mhs if ax is a1 else dmhs)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, fname), dpi=150)
    plt.close(fig)
    return 1


def _paired(paths):
    """(scene, seed, source) -> {arm: row}, dropping runs PX4 voided."""
    eps = {}
    for p in paths:
        tag = os.path.basename(p)
        for line in open(p):
            if line.strip().startswith("{"):
                r = json.loads(line)
                eps.setdefault((tag, r["scene"], r["seed"]), {})[r["arm"]] = r
    return {k: v for k, v in eps.items()
            if not any(x["mean_speed"] < 0.05 for x in v.values())}


def fig_calibration_ledger(cells, out, fname="calibration_ledger.png"):
    """Every depth calibration tried, offline score against closed-loop outcome.

    The left bars are what each calibration was designed to fix: the fraction of an obstacle's
    depth step the student recovers, 1.00 being correct placement. The right bars are what
    happened when the same calibration flew. They disagree completely -- the two transforms
    that scored best offline are the two that never reached the goal -- because the offline
    metric only looks INSIDE the obstacle silhouette and cannot see the damage a global
    transform does everywhere else.
    """
    names = [c["name"] for c in cells]
    recov = [c["recov"] for c in cells]
    succ = [c["success"] for c in cells]
    n = [c.get("n") for c in cells]
    y = np.arange(len(cells))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 0.62 * len(cells) + 1.9), sharey=True)
    a1.barh(y, [0 if r is None else r for r in recov], height=0.6,
            color=["#e2e8f0" if r is None else STUDENT for r in recov])
    a1.axvline(1.0, color="#38a169", lw=1.4, ls="--")
    for i, r in enumerate(recov):
        if r is None:
            a1.text(0.02, i, "not measured", va="center", fontsize=9, color="#718096")
        else:
            a1.text(min(r + 0.03, 1.06), i, f"{r:.2f}", va="center", fontsize=9,
                    color="#2d3748")
    a1.set_xlim(0, 1.25)
    a1.set_title("offline: obstacle depth-step recovered\n(dashed = correct placement)",
                 fontsize=10)
    ok = [s for s in succ if s is not None]
    a2.barh(y, [0 if s is None else s for s in succ], height=0.6,
            color=["#e2e8f0" if s is None else ("#c53030" if s < 0.05 else STUDENT)
                   for s in succ])
    for i, s in enumerate(succ):
        if s is None:
            a2.text(0.02, i, "not flown", va="center", fontsize=9, color="#718096")
        else:
            a2.text(s + 0.02, i, f"{s:.2f}" + (f"  (n={n[i]})" if n[i] else ""),
                    va="center", fontsize=9, color="#2d3748")
    a2.set_xlim(0, 1.25)
    a2.set_title("closed loop: success rate", fontsize=10)
    a1.set_yticks(y); a1.set_yticklabels(names)
    a1.invert_yaxis()
    for ax in (a1, a2):
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xticks([0, 0.5, 1.0])
    fig.tight_layout()
    fig.savefig(os.path.join(out, fname), dpi=150)
    plt.close(fig)
    return 1


def fig_collision_split(cells, out, fname="collision_split.png"):
    """Where the collisions go.

    MERLIN roughly halves strikes on the injected obstacle -- the map is doing its job -- but
    total collisions barely move, because the avoidance manoeuvres put it into scene furniture
    instead. Widening the margin only shifts weight between the two bars. That trade is why
    no global setting closes the gap to GT depth.
    """
    fig, axes = plt.subplots(1, len(cells), figsize=(4.4 * len(cells), 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    arms = ["blind", "student", "gt"]
    labels = {"blind": "no map", "student": "MERLIN", "gt": "GT depth"}
    for ax, cell in zip(axes, cells):
        eps = _paired(cell["files"])
        nov, furn = [], []
        for a in arms:
            rs = [e[a] for e in eps.values() if a in e]
            if not rs:
                nov.append(0); furn.append(0); continue
            nov.append(np.mean([any(x.get("is_novel") for x in r["collision_events"])
                                for r in rs]))
            furn.append(np.mean([bool(r["collision_events"])
                                 and not any(x.get("is_novel")
                                             for x in r["collision_events"]) for r in rs]))
        x = np.arange(len(arms))
        ax.bar(x, nov, width=0.6, color="#c53030", label="hit injected obstacle")
        ax.bar(x, furn, width=0.6, bottom=nov, color="#f6ad55", label="hit scene furniture")
        ax.set_xticks(x); ax.set_xticklabels([labels[a] for a in arms], fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{cell['name']}  (n={len(eps)})", fontsize=10)
    axes[0].set_ylabel("episodes ending in a collision")
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, fname), dpi=150)
    plt.close(fig)
    return 1
