#!/usr/bin/env python3
"""Cross-frame recirculation for the frozen MERLIN student (arXiv 2608.17981, single-stack).

The paper mixes a deep layer's residual stream back into a shallow one at the NEXT input step:

    z_{t+1,t,d} = alpha * f(z_{t,t,s}) + (1 - alpha) * z_{t,t,d}      (their Eq. 1)
    f(z)        = ||z_d||_2 / ||z||_2 * z                             (their Eq. 2)

They pay 2x compute for it, running k+1 transformer stacks per input step at k iterations.
That cost is not intrinsic to the mechanism: in autoregressive decoding a "copy" of the
architecture is seeded by an injection at step t-1 and then has to keep serving step t through
its own KV cache, so two copies are live at once. MERLIN's deployed core is single-view and
stateless (model/quant/export_onnx.py builds one view per forward, no cross-frame attention,
no cache), so there is no cache lineage to preserve and one pass per frame suffices: tap z_s on
the way past, inject last frame's z_s at layer d. Cost is one L2 norm and one axpy.

Nothing here trains or modifies a weight. Injection is two forward hooks.
"""
import re
from collections import deque

import numpy as np
import torch

# Residual-stream tap points, in execution order. Both stacks are width 768 in the base
# student, so any (source, destination) pair is dimensionally trivial; only the token count
# differs (the AAT carries a scale token and registers past the 729 patch tokens).
ENC_RE = re.compile(r"^encoder\..*\bblocks\.(\d+)$")
AAT_RE = re.compile(r"^info_sharing\.self_attention_blocks\.(\d+)$")


def tap_list(model):
    """Flat ordered list [(label, module)]: encoder blocks 0..N then AAT blocks 0..M."""
    enc, aat = [], []
    for name, mod in model.named_modules():
        m = ENC_RE.match(name)
        if m:
            enc.append((int(m.group(1)), "enc.%d" % int(m.group(1)), mod))
            continue
        m = AAT_RE.match(name)
        if m:
            aat.append((int(m.group(1)), "aat.%d" % int(m.group(1)), mod))
    enc.sort(key=lambda r: r[0])
    aat.sort(key=lambda r: r[0])
    return [(lbl, mod) for _, lbl, mod in enc] + [(lbl, mod) for _, lbl, mod in aat]


def _tensor_of(out):
    """Blocks return either a tensor or a tuple whose first element is the residual stream."""
    if isinstance(out, tuple):
        return out[0], lambda t: (t,) + out[1:]
    return out, lambda t: t


class Recirculator:
    """Single-stack cross-frame recirculation on a frozen model.

    `source` and `destination` are labels from `tap_list` (e.g. "aat.5", "enc.3"). The
    destination must execute BEFORE the source, so that the destination hook consumes the
    PREVIOUS frame's state and the source hook then overwrites it with the current frame's.

    `mode`:
      prev    - inject frame t-1's source activation (the mechanism under test)
      shuffle - inject a source activation captured from an unrelated frame, held fixed.
                The control that separates temporal state from generic perturbation.
      off     - hooks installed but inert; the untouched student.
    """

    def __init__(self, model, source, dest, alpha=0.10, norm="token", mode="prev",
                 center=0.0, lag=1):
        labels = [l for l, _ in tap_list(model)]
        if source not in labels or dest not in labels:
            raise ValueError("unknown tap; have %s" % labels)
        if labels.index(source) <= labels.index(dest):
            raise ValueError("source %s must execute after destination %s" % (source, dest))
        mods = dict(tap_list(model))
        self.alpha, self.norm, self.mode = float(alpha), norm, mode
        # Residual streams here are 99.3-99.8% an input-independent constant (measured by
        # eval/drone/diag_recirc_state.py: frame-varying energy is 0.16% at aat.6). Injecting
        # the raw stream therefore delivers >99% generic mean and <1% frame-specific content,
        # and `prev` becomes indistinguishable from `shuffle` by construction. With center>0 the
        # mixture is applied to the CENTRED residual and the mean is added back, so only the
        # frame-specific part is exchanged. `center` is the EMA decay of that mean over frames
        # (causal, no offline statistics); center=0.0 disables it and gives exactly Eq. 1.
        # Inject the activation from `lag` steps back, not necessarily the previous one. The
        # useful recirculation interval is a CAMERA BASELINE (~0.10 m), not a frame count: at
        # the closed loop's measured 0.0308 m per keyframe, lag=1 sits at 0.031 m, which is the
        # redundant regime where the injection does nothing. lag=3 lands on 0.09 m.
        self.lag = max(1, int(lag))
        self.buf = deque(maxlen=self.lag)
        self.center = float(center)
        self.mean_s = None
        self.mean_d = None
        self.source, self.dest = source, dest
        self.state = None        # z_s carried into the next frame
        self.last = None         # z_s most recently seen (pool building)
        self.n_injected = 0
        self._h = [mods[dest].register_forward_hook(self._inject),
                   mods[source].register_forward_hook(self._capture)]

    # -- hooks -------------------------------------------------------------
    def _capture(self, mod, inp, out):
        t, _ = _tensor_of(out)
        self.last = t.detach()
        if self.center > 0.0:
            self.mean_s = t.detach() if self.mean_s is None else \
                self.center * self.mean_s + (1.0 - self.center) * t.detach()
        # In shuffle mode the caller supplies `state` per frame from a pool of foreign
        # activations, so the stream must not overwrite it with its own history.
        if self.mode == "prev":
            self.buf.append(t.detach())
            self.state = self.buf[0] if len(self.buf) == self.lag else None
        return out

    def _inject(self, mod, inp, out):
        src = self.state
        if self.mode == "off" or src is None or self.alpha == 0.0:
            return out
        t, rewrap = _tensor_of(out)
        n = min(src.shape[1], t.shape[1])
        s, d = src[:, :n].to(t.dtype), t[:, :n]
        if s.shape[0] != d.shape[0]:
            s = s[: d.shape[0]] if s.shape[0] > d.shape[0] else s.expand_as(d)
        base = torch.zeros((), dtype=d.dtype, device=d.device)
        if self.center > 0.0:
            self.mean_d = d.detach() if self.mean_d is None else \
                self.center * self.mean_d + (1.0 - self.center) * d.detach()
            ms = self.mean_s if self.mean_s is not None else s.detach()
            base, s, d = self.mean_d, s - ms[:, :n], d - self.mean_d
        if self.norm == "token":
            s = s * (d.norm(dim=-1, keepdim=True) / s.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        elif self.norm == "global":
            s = s * (d.norm() / s.norm().clamp_min(1e-6))
        new = t.clone()
        new[:, :n] = base + self.alpha * s + (1.0 - self.alpha) * d
        self.n_injected += 1
        return rewrap(new)

    # -- lifecycle ---------------------------------------------------------
    def reset(self):
        """Clear carried state. Call at every sequence boundary; frame 0 is the warm-up."""
        self.state = None
        self.buf.clear()
        self.mean_s = None
        self.mean_d = None

    def set_state(self, z):
        """Install the state to be injected on the next frame (shuffle control)."""
        self.state = None if z is None else z.detach()

    def remove(self):
        for h in self._h:
            h.remove()
        self._h = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.remove()


@torch.no_grad()
def loop_gain(model, forward_fn, source, dest, alpha, norm="token", iters=12, seed=0):
    """Power-iterate the linearised feedback map to get its spectral radius.

    Recirculation across frames is a genuine dynamical system: a perturbation injected at
    `dest` propagates to `source` and is re-injected next frame, scaled by alpha and by the
    norm-matching ratio. The loop gain G = alpha * gamma * rho decides whether the carried
    state is bounded (G < 1) and sets the memory horizon tau = -1/ln G in frames. This is the
    stability half of the closed-form selection criterion (RECIRC-01).

    The iterate must live in the DESTINATION's space: one loop is (perturb d) -> (forward to s)
    -> (the injection's own slice back into d). The two taps do not share a token count -- the
    AAT carries a scale token the encoder does not -- so the map back is a slice to the common
    leading patch tokens, exactly as `Recirculator._inject` does. Estimated by finite
    differences, so it needs no autograd graph.
    """
    mods = dict(tap_list(model))
    cap, inj = {}, {}

    def _at_dest(m, i, o):
        t, rewrap = _tensor_of(o)
        cap["d"] = t.detach()
        d = inj.get("delta")
        if d is None:
            return o
        n = min(d.shape[1], t.shape[1])
        new = t.clone()
        new[:, :n] = t[:, :n] + d[:, :n]
        return rewrap(new)

    def _at_src(m, i, o):
        cap["s"] = _tensor_of(o)[0].detach()
        return o

    hs = [mods[dest].register_forward_hook(_at_dest),
          mods[source].register_forward_hook(_at_src)]
    try:
        inj["delta"] = None
        forward_fn()
        z_d0, z_s0 = cap["d"].clone(), cap["s"].clone()
        n = min(z_d0.shape[1], z_s0.shape[1])

        g = torch.Generator(device="cpu").manual_seed(seed)
        v = torch.randn(z_d0.shape, generator=g).to(z_d0.device, z_d0.dtype)
        v /= v.norm().clamp_min(1e-12)
        eps = 1e-2 * z_d0.norm() / max(v.norm().item(), 1e-12)
        rho = 0.0
        for _ in range(iters):
            inj["delta"] = eps * v
            forward_fn()
            jv = (cap["s"] - z_s0) / eps          # response in SOURCE space
            w = torch.zeros_like(z_d0)            # map back into DESTINATION space
            w[:, :n] = jv[:, :n]
            nw = w.norm().item()
            rho = nw / max(v.norm().item(), 1e-12)
            if nw <= 0:
                break
            v = w / nw
        G = alpha * rho
        tau = None
        if 0.0 < G < 1.0:
            tau = float(-1.0 / np.log(G))
        return {"rho": rho, "gamma": 1.0, "alpha": alpha, "G": G, "tau_frames": tau}
    finally:
        for h in hs:
            h.remove()
