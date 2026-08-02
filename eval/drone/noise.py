"""Optional input-realism knobs for the drone sim, all default-off.

The sim otherwise feeds the model pristine renders and ground-truth pose. These let a
follow-up run degrade the two largest oracles (clean imagery, perfect localisation)
without touching the experiment code again.
"""
import numpy as np


def add_rgb_noise(rgb, sigma, rng):
    """Additive Gaussian sensor noise in 8-bit counts. sigma=0 is exactly identity."""
    if sigma <= 0:
        return rgb
    out = rgb.astype(np.float32) + rng.normal(0.0, float(sigma), rgb.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def motion_blur(rgb, speed_mps, gain=4.0):
    """Horizontal box blur with a kernel length proportional to flight speed.

    A cheap stand-in for exposure-time smear: enough to test whether obstacle detection
    survives non-sharp frames, without modelling the true per-pixel optical flow.
    """
    k = int(round(float(speed_mps) * float(gain)))
    if k < 2:
        return rgb
    pad = k // 2
    f = rgb.astype(np.float32)
    padded = np.pad(f, ((0, 0), (pad, pad), (0, 0)), mode="edge")
    acc = np.zeros_like(f)
    for i in range(k):
        acc += padded[:, i:i + f.shape[1], :]
    return np.clip(acc / k, 0, 255).astype(np.uint8)


def _so3(w):
    t = float(np.linalg.norm(w))
    if t < 1e-12:
        return np.eye(3)
    k = np.asarray(w, float) / t
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)


def perturb_pose(pos, R, pos_sigma, rot_sigma_deg, rng):
    """Perturb the pose used to integrate depth into the map.

    This is the largest oracle in the sim: the voxel map is currently assembled from PX4
    ground-truth pose, i.e. perfect odometry. Returns (pos, R) unchanged when both sigmas
    are zero. The returned R stays a proper rotation.
    """
    pos = np.asarray(pos, float)
    R = np.asarray(R, float)
    if pos_sigma <= 0 and rot_sigma_deg <= 0:
        return pos, R
    p = pos + rng.normal(0.0, float(pos_sigma), 3) if pos_sigma > 0 else pos
    if rot_sigma_deg > 0:
        w = rng.normal(0.0, np.deg2rad(float(rot_sigma_deg)), 3)
        R = _so3(w) @ R
    return p, R


def random_depth(shape, rng, lo=0.3, hi=3.0, cells=12):
    """Smooth random depth field: low-frequency noise bilinearly upsampled to the image.

    White noise per pixel would be rejected by the voxel map's hit threshold and would carry
    the map nowhere, which would make the control trivially equivalent to flying blind. A
    smooth field produces coherent surfaces of the same character as a real depth map."""
    g = rng.random((cells, cells))
    ys = np.linspace(0, cells - 1, shape[0])
    xs = np.linspace(0, cells - 1, shape[1])
    y0 = np.clip(np.floor(ys).astype(int), 0, cells - 2)
    x0 = np.clip(np.floor(xs).astype(int), 0, cells - 2)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    top = g[y0][:, x0] * (1 - fx) + g[y0][:, x0 + 1] * fx
    bot = g[y0 + 1][:, x0] * (1 - fx) + g[y0 + 1][:, x0 + 1] * fx
    return (lo + (hi - lo) * (top * (1 - fy) + bot * fy)).astype(np.float32)
