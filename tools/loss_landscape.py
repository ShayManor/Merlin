"""Filter-normalized 2D loss landscape, rendered as a 3D surface.

Method follows Li et al., "Visualizing the Loss Landscape of Neural Nets"
(arXiv:1712.09913): take two random Gaussian directions in parameter space,
normalize each direction filter-wise to the norm of the corresponding filter
in the trained weights, then evaluate the real loss on a grid

    L(alpha, beta) = L(theta* + alpha * d1 + beta * d2)

Filter-wise normalization is what makes the picture meaningful: without it the
apparent sharpness of a minimum is just an artifact of weight scale.

Numpy only (no torch on this box). The network is a 5x32 tanh MLP fit to a
high-frequency 2D wave field. That task is what buys the geometry: an easy
separable task trains to ~0 loss and its slice is a featureless bowl, while a
net holding many oscillations in place sits in a basin with real relief.

    python3 tools/loss_landscape.py          # defaults reproduce the figure
"""

import argparse

import numpy as np

RNG_SEED = 7


# ---------------------------------------------------------------- data + model

def wave_field(n, freq, rng):
    """High-frequency 2D regression target. The frequency matters: a net that
    has to hold many oscillations in place has a far more rugged landscape than
    one solving an easy separable task."""
    x = rng.uniform(-2.0, 2.0, (n, 2))
    r = np.linalg.norm(x, axis=1)
    y = np.sin(freq * x[:, 0]) * np.cos(freq * x[:, 1]) + 0.6 * np.sin(freq * r)
    return x, y


def init_params(sizes, rng):
    return [
        (rng.normal(0, np.sqrt(1.0 / a), (a, b)), np.zeros(b))
        for a, b in zip(sizes[:-1], sizes[1:])
    ]


ACT = "relu"


def forward(params, x):
    """Returns (output, per-layer activations) for backprop."""
    acts = [x]
    h = x
    for w, b in params[:-1]:
        h = h @ w + b
        h = np.maximum(h, 0.0) if ACT == "relu" else np.tanh(h)
        acts.append(h)
    w, b = params[-1]
    return (h @ w + b).ravel(), acts


def loss_of(params, x, y):
    z = forward(params, x)[0]
    return float(np.mean((z - y) ** 2))


def grads(params, x, y):
    z, acts = forward(params, x)
    n = len(y)
    delta = (2.0 * (z - y) / n)[:, None]
    g = [None] * len(params)
    for i in range(len(params) - 1, -1, -1):
        g[i] = (acts[i].T @ delta, delta.sum(0))
        if i:
            back = (acts[i] > 0) if ACT == "relu" else (1.0 - acts[i] ** 2)
            delta = (delta @ params[i][0].T) * back
    return g


def train(params, x, y, steps, lr, rng):
    vel = [(np.zeros_like(w), np.zeros_like(b)) for w, b in params]
    n = len(y)
    for s in range(steps):
        idx = rng.choice(n, 256, replace=False)
        g = grads(params, x[idx], y[idx])
        step = lr * 0.5 * (1 + np.cos(np.pi * s / steps))
        gn = np.sqrt(sum(float((a ** 2).sum() + (b ** 2).sum()) for a, b in g))
        if gn > 1.0:
            g = [(a / gn, b / gn) for a, b in g]
        for i, ((gw, gb), (vw, vb)) in enumerate(zip(g, vel)):
            vw[:] = 0.9 * vw + gw
            vb[:] = 0.9 * vb + gb
            params[i][0][:] -= step * vw
            params[i][1][:] -= step * vb
    return params


# ------------------------------------------------------------------- landscape

def filter_normalized_direction(params, rng):
    """Random direction, each filter (weight column) rescaled to the trained
    filter's norm. Biases are zeroed, per the paper."""
    d = []
    for w, b in params:
        dw = rng.normal(size=w.shape)
        dw *= np.linalg.norm(w, axis=0, keepdims=True) / (
            np.linalg.norm(dw, axis=0, keepdims=True) + 1e-10
        )
        d.append((dw, np.zeros_like(b)))
    return d


def landscape(params, d1, d2, x, y, grid, span):
    coords = np.linspace(-span, span, grid)
    z = np.empty((grid, grid))
    probe = [(w.copy(), b.copy()) for w, b in params]
    for i, a in enumerate(coords):
        for j, bb in enumerate(coords):
            for k, (w, b) in enumerate(params):
                probe[k][0][:] = w + a * d1[k][0] + bb * d2[k][0]
                probe[k][1][:] = b
            z[j, i] = loss_of(probe, x, y)
    return coords, z


# ---------------------------------------------------------------------- render

def render(coords, z, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource

    a, b = np.meshgrid(coords, coords)
    # log loss: standard for these plots, keeps the basin readable next to the walls
    zc = np.log(np.maximum(z, 1e-3))

    fig = plt.figure(figsize=(9, 7.2), dpi=220)
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(zc.min(), zc.max())
    shade = LightSource(azdeg=315, altdeg=45).shade(
        zc, cmap=cmap, norm=norm, blend_mode="soft", vert_exag=1.6
    )

    # mplot3d paints the projected contour over the surface whatever the add
    # order is, so the depth has to be pinned explicitly
    floor = zc.min() - 1.1 * (zc.max() - zc.min())
    cset = ax.contour(a, b, zc, levels=20, offset=floor, cmap=cmap,
                      linewidths=0.5, alpha=0.8)
    cset.set_zorder(0)

    surf = ax.plot_surface(
        a, b, zc,
        facecolors=shade, rstride=1, cstride=1,
        antialiased=True, shade=False,
    )
    surf.set_edgecolor((1, 1, 1, 0.11))
    surf.set_linewidth(0.1)
    surf.set_zorder(4)

    # the trained weights sit at the origin of the slice
    c = len(coords) // 2
    ax.plot([0, 0], [0, 0], [floor, zc[c, c]], color="#d94a3d", lw=0.9, alpha=0.7, zorder=8)
    ax.scatter([0], [0], [zc[c, c]], color="#d94a3d", s=26, depthshade=False, zorder=9)

    ax.set_zlim(floor, zc.max())
    ax.view_init(elev=48, azim=-58)
    ax.set_box_aspect((1, 1, 0.66), zoom=1.05)
    ax.set_xticks([-coords[-1], 0, coords[-1]])
    ax.set_yticks([-coords[-1], 0, coords[-1]])

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_visible(False)
        axis.line.set_color((0, 0, 0, 0.18))
        axis._axinfo["grid"].update(color=(0, 0, 0, 0.07), linewidth=0.5)
    ax.tick_params(colors="#666", labelsize=7, pad=0.5)
    ax.set_xlabel(r"$\alpha$", labelpad=-4, color="#444")
    ax.set_ylabel(r"$\beta$", labelpad=-4, color="#444")
    ax.set_zticks([])
    ax.text2D(0.035, 0.93, "log loss", transform=ax.transAxes, color="#444", fontsize=9)

    fig.savefig(out, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", type=int, default=61, help="grid is grid x grid evals")
    p.add_argument("--span", type=float, default=2.0)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--freq", type=float, default=3.5)
    p.add_argument("--act", default="tanh", choices=["relu", "tanh"])
    p.add_argument("--dirseed", type=int, default=9)
    p.add_argument("--out", default="results/landscape/loss_landscape.png")
    args = p.parse_args()

    global ACT
    ACT = args.act
    rng = np.random.default_rng(RNG_SEED)
    x, y = wave_field(800, args.freq, rng)
    params = init_params([2] + [args.width] * args.depth + [1], rng)
    params = train(params, x, y, args.steps, args.lr, rng)
    print(f"trained loss {loss_of(params, x, y):.4f}")

    drng = np.random.default_rng(args.dirseed)
    d1 = filter_normalized_direction(params, drng)
    d2 = filter_normalized_direction(params, drng)
    coords, z = landscape(params, d1, d2, x, y, args.grid, args.span)
    print(f"{args.grid ** 2} evals  loss min {z.min():.4f}  max {z.max():.4f}")

    np.savez(args.out.replace(".png", ".npz"), coords=coords, z=z)
    render(coords, z, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
