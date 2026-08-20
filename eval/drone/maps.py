"""Alternative occupancy representations, as drop-in replacements for VoxelMap.

The planner only ever touches a map through `any_occupied_within`, `unknown` and `voxel`, so
a representation can be swapped without touching the planner or the loop. These exist to
benchmark MERLIN's counting map against what the literature does with noisy learned depth.

The measured problem (see docs/merlin-c4-handoff.md and the C4 section of CLAUDE.md): failure
episodes map the obstacle late, at 1.62-1.67 m against 1.98 m for successes, and then split on
how much of it lands in the map. Too many voxels (576 median) over-blocks the corridor and the
planner dithers until timeout; too few (348) under-blocks it and the block fires a tick after
contact. MonoMPC (arXiv:2508.07387) names this exact dichotomy, "stuck or collide", and
measures it on two mapping baselines.

VoxelMap counts integer hits against `min_hits`, so a sample at 0.3 m -- where the student's
error is 0.22 m absolute and 40-64% relative -- carries the same weight as one at 1.5 m, where
it is 0.15 m. Binary consensus over measurements whose reliability varies by a factor of four
is what manufactures both the inflated and the deflated map.

Three alternatives, each with one conservatism knob so a Pareto curve can be traced:

  LogOddsMap      probabilistic inverse sensor model. Hit evidence is spread over +-k*sigma(z)
                  along the ray and weighted by 1/sigma(z)^2, as in Voxblox/KinectFusion
                  (inverse-square) and Rosinol et al. (arXiv:2210.01276), who report 92% better
                  accuracy than unweighted fusion of monocular depth. Knob: `l_thresh`.
  NanoMapLocal    no fusion at all. Keeps the last K depth frames with their poses and answers
                  each query from the single minimum-uncertainty view of that point
                  (Florence et al., arXiv:1802.09076). Knob: `agree`.
  RiskInflatedMap chance-constrained inflation: the clearance radius grows with the uncertainty
                  of the measurement that produced each voxel, instead of a constant `safety`
                  (CCO-VOXEL, arXiv:2110.02904). Knob: `risk_k`.
"""
import numpy as np

from eval.drone.mapper import VoxelMap, backproject

# Inverse sensor model: per-range-bin (lo, hi, bias, sigma) in metres, MEASURED for
# student_nf.pt on the 385 held-out nearfield/val frames at the flown (fixed) scale, by
# bench_depth.py --global_scale --sigma_out. 4-13 M pixels per bin. Override per run with
# --sigma_model for a different checkpoint.
#
# Two things here are load-bearing and neither was obvious before it was measured. Sigma grows
# MONOTONICALLY with range, 0.163 -> 0.475, so the near-field fine-tune has made the near field
# the most reliable band, not the least; the pre-fine-tune table in CLAUDE.md says the opposite
# and no longer describes this checkpoint. And the bias flips sign, pushing near surfaces away
# (+0.078) while pulling far ones closer (-0.139).
#
# The consequence that matters for navigation: an obstacle first enters the map at 1.6-2.0 m,
# where sigma is 0.32-0.41 m against a 0.10 m voxel. First sightings are barely better than
# noise, which is why the same obstacle lands over-inflated in one episode and under-inflated
# in the next. Inverse-variance weighting exists to stop those sightings counting as strongly
# as the 0.163 m ones the drone gets later.
DEFAULT_SIGMA = [(0.2, 0.5, 0.078, 0.163), (0.5, 0.8, 0.032, 0.219), (0.8, 1.2, -0.029, 0.225),
                 (1.2, 1.8, -0.091, 0.321), (1.8, 2.5, -0.127, 0.407), (2.5, 8.0, -0.139, 0.475)]


def sigma_lookup(table=None):
    """-> f(z) giving (bias, sigma) per element of z, piecewise-constant over the bins."""
    tab = sorted(table or DEFAULT_SIGMA, key=lambda r: r[0])
    edges = np.array([r[0] for r in tab] + [tab[-1][1]], float)
    bias = np.array([r[2] for r in tab], float)
    sig = np.array([r[3] for r in tab], float)

    def f(z):
        i = np.clip(np.searchsorted(edges, np.asarray(z, float), "right") - 1, 0, len(tab) - 1)
        return bias[i], sig[i]
    return f


def _within(m, pts, radius, codes):
    """VoxelMap's neighbourhood test against an explicit sorted code array.

    Shared because two of the methods here need occupancy against a code set that is not the
    map's own: RiskInflatedMap queries one sigma bin at a time, NanoMapLocal queries a single
    frame's surface set.
    """
    pts = np.atleast_2d(np.asarray(pts, float))
    if len(pts) == 0 or codes.size == 0:
        return np.zeros(len(pts), bool)
    off = m._offsets(radius)
    keys = np.round(pts / m.voxel).astype(np.int64)
    cand = keys[:, None, :] + off[None, :, :]
    near = np.linalg.norm(cand * m.voxel - pts[:, None, :], axis=2) <= radius
    cc = m._encode(cand.reshape(-1, 3)).reshape(cand.shape[:2])
    i = np.searchsorted(codes, cc).clip(0, codes.size - 1)
    return (near & (codes[i] == cc)).any(1)


def load_sigma(path):
    """Read a bench_depth.py --sigma_out JSON into the (lo, hi, bias, sigma) form."""
    import json
    return [(r["lo"], r["hi"], r["bias"], r["sigma"]) for r in json.load(open(path))]


class LogOddsMap(VoxelMap):
    """Probabilistic inverse sensor model over the same voxel hash.

    Two departures from VoxelMap, both standard and both aimed at the measured failure:

    1. A measurement does not occupy one voxel. It deposits the Gaussian probability that the
       surface lies in each voxel, centred on the bias-corrected range, so a surface whose
       predicted face is displaced by less than a couple of sigma still puts evidence on the
       true face. VoxelMap's hard deposit is why the block fires late: the near face is placed
       too far and the map is confident about the wrong voxel.
    2. Every observation carries the SAME total evidence, so precision decides concentration
       rather than amount. A 0.163 m-sigma sighting from close up crosses the threshold in a
       few frames; a 0.407 m-sigma one at 2 m spreads too thin to occupy anything alone and
       must be corroborated. Weighting the amount by 1/sigma^2 as well would charge the same
       uncertainty twice.

    Occupancy is a log-odds threshold rather than a hit count, which gives a continuous
    conservatism knob (`l_thresh`) for the Pareto sweep. Free space is carved to the near edge
    of the bump so carving never erases the bump it belongs to.
    """

    def __init__(self, *a, sigma_table=None, k_sigma=2.0, l_thresh=0.4, l_max=4.0,
                 free_w=0.55, l_hit=1.0, max_spread=0.45, **kw):
        super().__init__(*a, **kw)
        self.sigma = sigma_lookup(sigma_table)
        self.k_sigma = float(k_sigma)
        self.l_thresh = float(l_thresh)
        self.l_max = float(l_max)
        self.free_w = float(free_w)
        self.l_hit = float(l_hit)
        # Hard cap on how far one measurement may smear a surface. 2 sigma at range is 0.95 m,
        # which is wider than the corridors this flies in.
        self.max_spread = float(max_spread)
        self._sig_ref = min(r[3] for r in (sigma_table or DEFAULT_SIGMA))
        self._l = {}          # voxel code -> log-odds

    # --- occupancy ------------------------------------------------------------------
    def _occ_keys(self):
        return [k for k, v in self._l.items() if v >= self.l_thresh]

    def is_occupied(self, p_world):
        return self._l.get(self._key(p_world), 0.0) >= self.l_thresh

    # --- integration ----------------------------------------------------------------
    def integrate(self, depth, K, R_cam2world, t_cam_world, max_range=8.0, carve=True):
        pts_cam = backproject(depth, K, max_range)
        if pts_cam.shape[0] == 0:
            return
        o = np.asarray(t_cam_world, float)
        pts_world = (R_cam2world @ pts_cam.T).T + o
        z = np.linalg.norm(pts_world - o, axis=1)
        keep = (z > 1e-3) & (np.max(np.abs(pts_world), axis=1) <= self.bound)
        if not keep.any():
            return
        pts_world, z = pts_world[keep], z[keep]
        dirs = (pts_world - o) / z[:, None]
        bias, sig = self.sigma(z)
        zc = np.maximum(z - bias, 1e-3)                 # de-biased range
        # Each observation deposits the SAME TOTAL evidence, spread as the Gaussian probability
        # that the surface lies in each voxel: w = l_hit * (voxel / (sigma*sqrt(2pi))) * exp(-t^2/2).
        # Depositing exp(-t^2/2) per voxel instead makes a wider bump carry more total evidence,
        # which is backwards and was measured to be catastrophic: at the true sigma of 0.407 m at
        # 2 m the bump spans 1.6 m, every voxel in it crossed the threshold, and the cell timed
        # out on 6 of 6 episodes because the planner had nowhere left to go.
        #
        # With the mass fixed, a precise near measurement concentrates it and crosses the
        # threshold quickly, while an imprecise far one spreads it too thin to occupy anything
        # on its own and has to be corroborated as the drone closes. That is the intended
        # behaviour: obstacles enter the map at 1.6-2.0 m where sigma is 0.32-0.41 m, and a
        # single sighting there should not be trusted to block a corridor.
        spread = np.minimum(self.k_sigma * sig, self.max_spread)
        w = self.l_hit * self.voxel / (np.maximum(sig, 1e-3) * np.sqrt(2 * np.pi))
        # Tap count follows the widest bump in voxels. At k_sigma=0 every tap lands on the same
        # voxel, and running 9 identical passes cost half the keyframe rate on the cluster
        # (98 vs 182 keyframes per successful episode at the same flight speed), which starved
        # the map and showed up as collisions rather than as a slow cell.
        ntap = int(np.clip(np.ceil(2 * float(np.max(spread)) / self.voxel) + 1, 1, 9))
        taps = np.linspace(-self.k_sigma, self.k_sigma, ntap) if ntap > 1 else np.zeros(1)
        frame = {}
        for t in taps:
            p = o + dirs * (zc + (t / max(self.k_sigma, 1e-9)) * spread)[:, None]
            ok = np.max(np.abs(p), axis=1) <= self.bound
            if not ok.any():
                continue
            codes = self._encode(np.round(p[ok] / self.voxel).astype(np.int64))
            ww = w[ok] * float(np.exp(-0.5 * t * t))
            good = codes >= 0
            self._accumulate_max(frame, codes[good], ww[good])
        for c, ww in frame.items():
            self._l[c] = min(self._l.get(c, 0.0) + ww, self.l_max)
        self._seen.update(frame.keys())

        if carve and self.free_w:
            self._carve_free(o, dirs, zc - spread, set(frame))
        self._codes_cache = None

    @staticmethod
    def _accumulate_max(frame, codes, weights):
        """frame[code] = max(existing, weight), vectorised over a keyframe's points."""
        if codes.size == 0:
            return
        order = np.argsort(codes, kind="stable")
        c, ww = codes[order], weights[order]
        edge = np.r_[True, c[1:] != c[:-1]]
        starts = np.flatnonzero(edge)
        best = np.maximum.reduceat(ww, starts)
        for k, v in zip(c[starts].tolist(), best.tolist()):
            if v > frame.get(k, 0.0):
                frame[k] = v

    def _carve_free(self, origin, dirs, stop, hit_codes, stride=8):
        """Subtract free-space evidence from the origin up to `stop` along each ray."""
        d, s = dirs[::stride], stop[::stride]
        good = s > 2.0 * self.voxel
        if not good.any():
            return
        d, s = d[good], s[good]
        ss = (np.arange(int(np.ceil(s.max() / self.voxel))) + 0.5) * self.voxel
        q = origin + d[:, None, :] * ss[None, :, None]
        m = ss[None, :] < s[:, None]
        codes = np.unique(self._encode(np.round(q[m] / self.voxel).astype(np.int64)))
        codes = codes[codes >= 0]
        self._seen.update(codes.tolist())
        for c in codes.tolist():
            if c in hit_codes:
                continue
            v = self._l.get(c)
            if v is not None:
                v -= self.free_w
                if v <= 0.0:
                    del self._l[c]
                else:
                    self._l[c] = v


class NanoMapLocal(VoxelMap):
    """No fusion. A window of recent depth frames, queried from the best single view.

    Fusing noisy depth into one grid commits every frame's error permanently and loses the
    information about which view was trustworthy. NanoMap (arXiv:1802.09076) instead keeps the
    raw measurements and, for each queried point, searches for the minimum-uncertainty view of
    it; it flies obstacle avoidance at up to 10 m/s on that.

    Uncertainty of a view here is sigma(z) at the queried range times an off-axis penalty,
    which stands in for NanoMap's pose-uncertainty chain: this harness maps from PX4 GT pose,
    so the relative-pose term is zero and what remains is depth uncertainty and viewing angle.

    `agree` is the conservatism knob and also selects the decision rule:

      agree <= 0  the paper's rule, answer from the single minimum-uncertainty view.
      agree >= 1  occupied if at least `agree` of the views that can see the point say so.

    Measured here, the paper's rule is unsafe for thin obstacles: one view whose pixel misses
    a 0.18 m-deep cloth and reads the wall behind it overrides every view that resolves it, and
    the drone flew straight in with the goal direction reading clear on 136 of 136 ticks. That
    is a real property of single-view querying against thin geometry, not an implementation
    slip, so both rules are kept and both are benchmarked.
    """

    def __init__(self, *a, sigma_table=None, window=12, agree=1, k_sigma=1.0, **kw):
        super().__init__(*a, **kw)
        self.sigma = sigma_lookup(sigma_table)
        self.window = int(window)
        self.agree = int(agree)
        self.k_sigma = float(k_sigma)
        self.frames = []      # (depth, K, R_world2cam, origin)

    def integrate(self, depth, K, R_cam2world, t_cam_world, max_range=8.0, carve=True):
        # Each frame keeps its OWN voxelised surface set. Nothing is merged, which is the
        # point of the method, but the query has to be a genuine 3D proximity search rather
        # than a depth comparison along the query ray. Measured: with a per-pixel depth test
        # a point 0.18 m to the side of a thin obstacle reads free, because the ray through
        # that pixel passes the obstacle edge and returns the wall behind it, and the drone
        # flew into the edge with the goal direction clear on every tick.
        d = np.asarray(depth, np.float32)
        pc = backproject(d, np.asarray(K, float), float(max_range))
        codes = np.zeros(0, np.int64)
        if pc.shape[0]:
            pw = (np.asarray(R_cam2world, float) @ pc.T).T + np.asarray(t_cam_world, float)
            pw = pw[np.max(np.abs(pw), axis=1) <= self.bound]
            if pw.shape[0]:
                c = self._encode(np.round(pw / self.voxel).astype(np.int64))
                codes = np.sort(np.unique(c[c >= 0]))
        self.frames.append((d, np.asarray(K, float), np.asarray(R_cam2world, float).T,
                            np.asarray(t_cam_world, float), float(max_range), codes))
        if len(self.frames) > self.window:
            self.frames.pop(0)
        self._codes_cache = None

    def _project(self, pts, frame):
        """-> (in_view, z_query, z_measured, off_axis) for (Q,3) world points."""
        depth, K, Rw2c, o, mr = frame[:5]
        pc = (Rw2c @ (pts - o).T).T
        zq = pc[:, 2]
        H, W = depth.shape
        with np.errstate(divide="ignore", invalid="ignore"):
            u = K[0, 0] * pc[:, 0] / zq + K[0, 2]
            v = K[1, 1] * pc[:, 1] / zq + K[1, 2]
        ok = (zq > 0.05) & (zq < mr) & np.isfinite(u) & np.isfinite(v)
        ui = np.clip(np.nan_to_num(u), 0, W - 1).astype(np.int32)
        vi = np.clip(np.nan_to_num(v), 0, H - 1).astype(np.int32)
        ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
        zm = depth[vi, ui]
        ok &= np.isfinite(zm) & (zm > 0.05)
        off = np.hypot((u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1])
        return ok, zq, zm, off

    def any_occupied_within(self, pts, radius):
        pts = np.atleast_2d(np.asarray(pts, float))
        if not self.frames:
            return np.zeros(len(pts), bool)
        votes = np.zeros(len(pts), np.int32)
        best_u = np.full(len(pts), np.inf)
        best_occ = np.zeros(len(pts), bool)
        # agree == 1 is an OR across frames, so a point already called occupied needs no
        # further views. Without this the query costs one full neighbourhood search per
        # retained frame, and nav_loop is wall-clock driven: a slow map silently spends the
        # episode's own time budget and would be penalised twice in the benchmark.
        one = self.agree == 1
        for fr in self.frames:
            if fr[5].size == 0:
                continue
            if one:
                todo = votes == 0
                if not todo.any():
                    break
                votes[todo] += _within(self, pts[todo],
                                       radius + self.k_sigma * float(np.median(
                                           self.sigma(self._project(pts[todo], fr)[1])[1])),
                                       fr[5]).astype(np.int32)
                continue
            # proximity against this frame's own surface set, widened by the measurement
            # uncertainty at the range the query point sits at
            ok, zq, zm, off = self._project(pts, fr)
            _, sq = self.sigma(zq)
            occ = _within(self, pts, radius + self.k_sigma * float(np.median(sq)), fr[5])
            votes += occ.astype(np.int32)
            u = np.where(ok, self.sigma(zm)[1] * (1.0 + off), np.inf)   # view uncertainty
            better = u < best_u
            best_u = np.where(better, u, best_u)
            best_occ = np.where(better, occ, best_occ)
        return best_occ if self.agree <= 0 else (votes >= self.agree)

    def unknown(self, pts):
        pts = np.atleast_2d(np.asarray(pts, float))
        if not self.frames:
            return np.ones(len(pts), bool)
        seen = np.zeros(len(pts), bool)
        for fr in self.frames:
            ok, zq, zm, _ = self._project(pts, fr)
            seen |= ok & (zq <= zm + self.voxel)      # in front of or at the surface
        return ~seen

    def occupied_points(self):
        """Surface points of the most recent frame, for the diagnostics that need a cloud."""
        if not self.frames:
            return np.zeros((0, 3))
        codes = self.frames[-1][5]
        if codes.size == 0:
            return np.zeros((0, 3))
        return self._decode(codes).astype(float) * self.voxel

    def _occ_codes(self):
        """The most recent frame's surface set. Nothing is fused, so there is no map-wide
        occupied set; this is what the diagnostics that ask for one should see."""
        return self.frames[-1][5] if self.frames else np.zeros(0, np.int64)

    def nearest_obstacle_dist(self, p_world, max_search=3.0):
        pts = self.occupied_points()
        if pts.shape[0] == 0:
            return float("inf")
        d = float(np.linalg.norm(pts - np.asarray(p_world, float), axis=1).min())
        return d if d <= max_search else float(max_search)


class RiskInflatedMap(LogOddsMap):
    """Chance-constrained clearance: the margin follows the uncertainty, not a constant.

    A fixed `safety` has to be right for both a 0.22 m-sigma near reading and a 0.15 m-sigma
    mid one, and cannot be: raising it eliminates obstacle strikes and re-introduces furniture
    strikes and timeouts (six margin increases measured, all net-negative). CCO-VOXEL
    (arXiv:2110.02904) instead plans against the distance distribution. This keeps the
    log-odds map and inflates each voxel's effective radius by risk_k * sigma(range at which
    that voxel was observed), so uncertain geometry is given room and well-observed geometry
    is not. risk_k is the risk level: 0 reduces to LogOddsMap, ~2 is a 2-sigma constraint.
    """

    def __init__(self, *a, risk_k=1.5, k_sigma=0.0, **kw):
        # k_sigma defaults to 0 here, unlike LogOddsMap: the uncertainty must enter ONCE. With
        # the inherited 2-sigma hit bump AND a 1.5-sigma clearance margin the drone was blocked
        # on 97% of ticks, because the same error was paid for twice. This map deposits at the
        # de-biased range and expresses all of the uncertainty as clearance.
        super().__init__(*a, k_sigma=k_sigma, **kw)
        self.risk_k = float(risk_k)
        self._rng_obs = {}     # voxel code -> range at which it was most recently observed
        self._rng_cache = None  # ranges aligned with _occ_codes(), rebuilt with it

    def integrate(self, depth, K, R_cam2world, t_cam_world, max_range=8.0, carve=True):
        before = dict(self._l)
        super().integrate(depth, K, R_cam2world, t_cam_world, max_range, carve)
        o = np.asarray(t_cam_world, float)
        fresh = [c for c in self._l if self._l[c] != before.get(c)]
        if fresh:
            pts = self._decode(np.array(fresh, np.int64)).astype(float) * self.voxel
            r = np.linalg.norm(pts - o, axis=1)
            for c, rr in zip(fresh, r.tolist()):
                self._rng_obs[c] = rr
        self._rng_cache = None

    def any_occupied_within(self, pts, radius):
        """Inflate the query by the largest per-voxel margin in the neighbourhood.

        Querying at radius + risk_k*max_sigma and then rejecting voxels whose own margin is
        smaller keeps the test exact rather than uniformly conservative.
        """
        pts = np.atleast_2d(np.asarray(pts, float))
        codes = self._occ_codes()
        if codes.size == 0:
            return np.zeros(len(pts), bool)
        _, sig = self.sigma(self._occ_ranges())
        out = np.zeros(len(pts), bool)
        # sigma is piecewise-constant over a handful of bins, so run one exact query per bin at
        # that bin's own radius. Querying every voxel at the largest margin instead inflated
        # the neighbourhood sixfold and ran the control tick at 3.8x real time.
        for s in np.unique(sig):
            todo = ~out
            if not todo.any():
                break
            sub = codes[sig == s]
            if sub.size:
                out[todo] = _within(self, pts[todo], radius + self.risk_k * float(s), sub)
        return out

    def _occ_ranges(self):
        """Observation range per entry of _occ_codes(), in the same order."""
        if self._rng_cache is None or self._rng_cache.shape[0] != self._occ_codes().shape[0]:
            codes = self._occ_codes()
            self._rng_cache = np.array([self._rng_obs.get(int(c), 0.5) for c in codes.tolist()],
                                       float) if codes.size else np.zeros(0)
        return self._rng_cache


REGISTRY = {"voxel": VoxelMap, "logodds": LogOddsMap,
            "nanomap": NanoMapLocal, "risk": RiskInflatedMap}


def build_map(name, **kw):
    """Construct a map by name, passing only the kwargs its constructor accepts."""
    import inspect
    cls = REGISTRY[name]
    sig = set()
    for k in cls.__mro__:
        try:
            sig |= set(inspect.signature(k.__init__).parameters)
        except (TypeError, ValueError):
            pass
    return cls(**{k: v for k, v in kw.items() if k in sig})
