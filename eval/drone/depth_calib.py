"""Per-pixel apparent->real depth calibration for the C4 loop.

Sampled in sim (student depth vs renderer GT over many poses) into a 4-D histogram indexed by
[pixel block, predicted-depth bin, gt/pred ratio bin], then applied at run time as
`depth * r_q(block, depth)`.

The quantile `q` is the whole point. The obvious fit -- replace a prediction by the MEDIAN GT
seen at that (pixel, predicted depth) -- is minimum-error and measured NET-NEGATIVE for
navigation: conditioned on TRUE range the student's near-field bias is +0.26 m, but
conditioned on PREDICTED range (the only key available at run time) it INVERTS to -0.21 m.
That is regression dilution, and correcting the diluted bias pushes near surfaces further
away, i.e. makes the drone less cautious in exactly the regime it crashes in. A LOW quantile
inverts that incentive: it answers "how close could this surface actually be?", so a cell
where the student is reliable (tight ratio distribution, e.g. a flat wall at 2 m) is left
alone, while a cell where it is noisy (40-64% relative error inside 0.5 m) pulls the surface
in. The correction is therefore keyed on the student's LOCAL RELIABILITY rather than on its
bias, which is what survives dilution.

Storing counts rather than a fitted curve keeps q an apply-time knob, so a q sweep costs no
extra sim time, and q=0.5 reproduces the median variant as a baseline.
"""
import numpy as np

PRED_LO, PRED_HI, N_PRED = 0.15, 6.0, 28
RATIO_LO, RATIO_HI, N_RATIO = 0.40, 2.50, 40
# Only ever pull a surface TOWARD the camera. Two earlier unbounded depth corrections drove
# high-dynamic-range pixels to zero, dropped them out of the map entirely and lost 5 of 5
# closed-loop episodes, so the correction is clamped on both sides.
R_LO, R_HI = 0.55, 1.00
# ...and never by more than this many metres. R_LO alone is a RELATIVE bound, so it would let
# a 2.5 m wall march in by 1.1 m: the student's far field is only ~0.11 relative error, but a
# low quantile of any distribution still sits below its centre, and applied to every wall that
# is a corridor-narrowing global margin change in disguise -- six of which are already
# measured net-negative. The failure being corrected is a ~0.3 m near-face displacement, so
# the absolute correction is clamped just above it (the same reason decompress_steps has cap).
MAX_PULL = 0.35


def _log_edges(lo, hi, n):
    return np.exp(np.linspace(np.log(lo), np.log(hi), n + 1))


class DepthCalib:
    """Histogram of gt/pred, per pixel block and predicted-depth bin."""

    def __init__(self, grid=27, block_shape=None):
        self.pred_edges = _log_edges(PRED_LO, PRED_HI, N_PRED)
        self.ratio_edges = _log_edges(RATIO_LO, RATIO_HI, N_RATIO)
        self.nby, self.nbx = block_shape or (int(grid), int(grid))
        self.counts = np.zeros((self.nby, self.nbx, self.n_pred, self.n_ratio), np.uint32)
        self._cache = {}

    @property
    def n_pred(self):
        return len(self.pred_edges) - 1

    @property
    def n_ratio(self):
        return len(self.ratio_edges) - 1

    # ---- fitting -------------------------------------------------------------------

    def _blocks(self, H, W):
        """Pixel -> block index. Derived from RELATIVE position, so a table fit at one
        resolution still applies at another."""
        key = ("blk", H, W)
        if key not in self._cache:
            by = (np.arange(H) * self.nby // H).clip(0, self.nby - 1)
            bx = (np.arange(W) * self.nbx // W).clip(0, self.nbx - 1)
            self._cache[key] = np.add.outer(by * self.nbx, bx)
        return self._cache[key]

    def accumulate(self, pred, gt):
        """One frame of metric student depth against renderer GT depth (same shape)."""
        pred = np.asarray(pred, np.float64)
        gt = np.asarray(gt, np.float64)
        if pred.shape != gt.shape:
            raise ValueError(f"shape mismatch {pred.shape} vs {gt.shape}")
        b = self._blocks(*pred.shape)
        v = (np.isfinite(pred) & np.isfinite(gt) & (gt > 0.1) & (gt < 10.0)
             & (pred >= self.pred_edges[0]) & (pred < self.pred_edges[-1]))
        if not v.any():
            return 0
        p, g, bb = pred[v], gt[v], b[v]
        pi = np.searchsorted(self.pred_edges, p, "right") - 1
        # Ratios outside the histogram range are CLIPPED, not dropped: they are real
        # observations of a badly wrong prediction and dropping them would bias the low
        # quantile upward, which is the direction that makes the drone less cautious.
        ri = (np.searchsorted(self.ratio_edges, g / p, "right") - 1).clip(0, self.n_ratio - 1)
        flat = (bb * self.n_pred + pi) * self.n_ratio + ri
        self.counts += np.bincount(flat, minlength=self.counts.size).reshape(
            self.counts.shape).astype(np.uint32)
        self._cache = {k: v for k, v in self._cache.items() if k[0] == "blk"}
        return int(v.sum())

    # ---- the correction ------------------------------------------------------------

    def ratio_map(self, q=0.20, min_count=200):
        """(nby, nbx, n_pred) q-th quantile of gt/pred, clipped to [R_LO, R_HI].

        Cells with too little evidence fall back to the same pred bin pooled over all blocks,
        and failing that to 1.0 (no correction). A sparse near-field cell must degrade to
        identity, never to a quantile of a handful of samples.
        """
        key = ("r", round(float(q), 4), int(min_count))
        if key in self._cache:
            return self._cache[key]
        c = self.counts.astype(np.float64)
        cell = self._quantile(c.reshape(-1, self.n_pred, self.n_ratio), q)
        tot = c.reshape(-1, self.n_pred, self.n_ratio).sum(-1)
        pooled = self._quantile(c.sum((0, 1))[None], q)[0]           # (n_pred,)
        pooled_tot = c.sum((0, 1, 3))
        r = np.where(tot >= min_count, cell, np.where(pooled_tot >= min_count, pooled, 1.0))
        r = np.clip(np.nan_to_num(r, nan=1.0), R_LO, R_HI)
        out = r.reshape(self.nby, self.nbx, self.n_pred)
        self._cache[key] = out
        return out

    def _quantile(self, c, q):
        """(..., n_pred, n_ratio) counts -> (..., n_pred) q-th ratio quantile.

        Interpolated within the containing bin (log-linear on the ratio axis) so the
        correction moves smoothly with q instead of snapping between bin edges.
        """
        tot = c.sum(-1)
        cum = np.cumsum(c, -1)
        target = q * tot
        i = (cum < target[..., None]).sum(-1).clip(0, self.n_ratio - 1)
        below = np.take_along_axis(cum, i[..., None], -1)[..., 0] \
            - np.take_along_axis(c, i[..., None], -1)[..., 0]
        n = np.take_along_axis(c, i[..., None], -1)[..., 0]
        f = np.where(n > 0, (target - below) / np.maximum(n, 1e-9), 0.0).clip(0.0, 1.0)
        lo = np.log(self.ratio_edges[:-1])[i]
        hi = np.log(self.ratio_edges[1:])[i]
        return np.where(tot > 0, np.exp(lo + f * (hi - lo)), np.nan)

    def apply(self, depth, q=0.20, min_count=200, max_pull=MAX_PULL):
        """Metric student depth -> calibrated depth.

        Log-linear interpolation ACROSS the predicted-depth axis, not a bin lookup: a step
        function of depth injects artificial discontinuities at the bin edges, and a depth
        discontinuity is precisely what the mapper turns into a phantom voxel.
        """
        d = np.asarray(depth, np.float32)
        R = self.ratio_map(q, min_count)
        cen = np.log(np.sqrt(self.pred_edges[:-1] * self.pred_edges[1:]))
        ld = np.clip(np.log(np.maximum(d, 1e-6)), cen[0], cen[-1])
        hi = np.searchsorted(cen, ld, "left").clip(1, self.n_pred - 1)
        lo = hi - 1
        w = (ld - cen[lo]) / (cen[hi] - cen[lo])
        b = self._blocks(*d.shape)
        by, bx = b // self.nbx, b % self.nbx
        r = R[by, bx, lo] * (1.0 - w) + R[by, bx, hi] * w
        if max_pull:
            r = np.maximum(r, 1.0 - float(max_pull) / np.maximum(d, 1e-6))
        out = np.where(np.isfinite(d) & (d > 0), np.maximum(d * r, 0.05), d)
        return out.astype(np.float32)

    # ---- reporting / io ------------------------------------------------------------

    def spread(self, q=0.20, min_count=200):
        """Per pred bin: (n samples, pooled ratio, IQR of the ratio ACROSS pixel blocks).

        The pixel axis only earns its place if that IQR is non-trivial. If the correction is
        flat across blocks the table has collapsed to a global depth rescale, which is a
        safety-margin change in disguise -- and six of those are already measured
        net-negative. Report it rather than assume it.
        """
        R = self.ratio_map(q, min_count)
        n = self.counts.sum((0, 1, 3))
        pooled = self._quantile(self.counts.astype(np.float64).sum((0, 1))[None], q)[0]
        rows = []
        for j in range(self.n_pred):
            v = R[:, :, j].reshape(-1)
            rows.append((float(self.pred_edges[j]), float(self.pred_edges[j + 1]), int(n[j]),
                         float(np.clip(np.nan_to_num(pooled[j], nan=1.0), R_LO, R_HI)),
                         float(np.percentile(v, 75) - np.percentile(v, 25))))
        return rows

    def save(self, path):
        np.savez_compressed(path, counts=self.counts, pred_edges=self.pred_edges,
                            ratio_edges=self.ratio_edges, nby=self.nby, nbx=self.nbx)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        t = cls(block_shape=(int(z["nby"]), int(z["nbx"])))
        t.pred_edges = z["pred_edges"]
        t.ratio_edges = z["ratio_edges"]
        t.counts = z["counts"]
        return t
