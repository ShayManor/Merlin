"""GateRoute: fixed-capacity Mixture-of-Depths token routing in the MERLIN student, STATIC shapes.

Routed blocks (default: DINOv2 encoder blocks 6-11 and AAT blocks 1, 3, 5, 7, the frame-attention
blocks of the alternating stack): a router scores the 729 patch tokens, TopK(k) picks k of them,
the block runs on [prefix tokens, k picked patch tokens, suffix tokens] and the picked tokens are
scattered back into the residual stream. Unpicked patch tokens ride the residual unchanged.
Prefix / suffix tokens (cls + registers in the encoder, registers + scale token in the AAT) always
run. k is a build-time constant, so the graph is TopK + GatherElements + block + ScatterElements,
the formulation timed in tools/spike_gateroute/e1_mod_trt.py ("gs").

Router kinds (one per arm; all share the graph, so they share the latency):
  learned  per-block LayerNorm(no affine) + Linear(C, 1). Trained ONLY by a ranking loss on a
           per-token label (planner label for GateRoute, depth-error label for the depth-error
           control); its input is detached, so the label never trains the backbone and the
           distillation loss never trains the router.
  norm     score = L2 norm of the token (feature-norm routing control), no parameters.
  random   score = uniform noise, fresh every forward (random routing control); a fixed
           generator seed makes eval repeatable.
MoD scales the routed update by sigmoid(router score) only so the TASK loss reaches the router.
GateRoute's router must not see the task loss, so the update is unscaled: picked tokens get
exactly the pretrained block, and k == 729 reproduces the dense student (checked by
gateroute_selfcheck below).

kv_full=True (latency variant "kv-full"): in a routed block only the picked tokens get Q, attention
output, out-proj and MLP, but attention reads K/V of ALL tokens, so unpicked tokens still give context
(they are just not updated). Per routed block: K/V projections on N tokens, attention k x N, the rest
on k tokens.

Batched (B >= 1) for training; export uses B = 1. Outputs are MonoCore's (dense, scale).
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.gateroute.come import _block

ENC_SITES = (6, 7, 8, 9, 10, 11)
AAT_SITES = (1, 3, 5, 7)


def _block_kvfull(blk, q_tok, kv_tok, scale=None):
    """Pre-norm residual block (DINOv2 Block or uniception SelfAttentionBlock, eval mode) updating
    q_tok only, with keys / values from kv_tok."""
    attn = blk.attn
    B, Nq, C = q_tok.shape
    h = attn.num_heads
    W, b = attn.qkv.weight, attn.qkv.bias
    q = F.linear(blk.norm1(q_tok), W[:C], None if b is None else b[:C]).reshape(B, Nq, h, C // h).transpose(1, 2)
    kv = F.linear(blk.norm1(kv_tok), W[C:], None if b is None else b[C:])
    kv = kv.reshape(B, kv_tok.shape[1], 2, h, C // h).permute(2, 0, 3, 1, 4)
    k, v = kv[0], kv[1]
    if hasattr(attn, "q_norm"):
        q, k = attn.q_norm(q), attn.k_norm(k)
    o = F.scaled_dot_product_attention(q, k, v, scale=scale)
    x = q_tok + blk.ls1(attn.proj(o.transpose(1, 2).reshape(B, Nq, C)))
    return x + blk.ls2(blk.mlp(blk.norm2(x)))


class Router(nn.Module):
    def __init__(self, dim, arch="linear"):
        super().__init__()
        self.arch = arch
        if arch == "ctx":  # sees its 3x3 token neighbourhood: 1x1 C->32, 3x3 32->32, 1x1 32->1 on the patch grid
            self.ctx = nn.Sequential(nn.Conv2d(dim, 32, 1), nn.GELU(), nn.Conv2d(32, 32, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(32, 1, 1))
            return
        self.lin = nn.Linear(dim, 1)
        nn.init.normal_(self.lin.weight, std=1e-3)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x):
        if self.arch == "ctx":
            B, N, C = x.shape
            g = int(round(int(N) ** 0.5))
            x = F.layer_norm(x, x.shape[-1:]).transpose(1, 2).reshape(B, C, g, g)
            return self.ctx(x).reshape(B, N)
        return self.lin(F.layer_norm(x, x.shape[-1:])).squeeze(-1)


class GateRouteCore(nn.Module):
    """MERLIN student with MoD routing at `enc_sites` and `aat_sites`, k patch tokens each."""

    def __init__(self, m, hw, k, enc_sites=ENC_SITES, aat_sites=AAT_SITES, router="learned", seed=0,
                 kv_full=False, heavy=0, k_heavy=0, heavy_router="learned", router_arch="linear", coverage=0.0):
        super().__init__()
        self.router_arch = router_arch
        self.coverage = coverage  # selection score = standardized router score - coverage * (blocks that already updated the token)
        self.m, self.hw, self.k, self.kv_full = m, hw, k, kv_full
        self.enc_sites, self.aat_sites = tuple(enc_sites), tuple(aat_sites)
        self.router_kind = router
        D_enc = m.encoder.model.embed_dim
        D_aat = m.info_sharing.dim
        self.routers = nn.ModuleDict(
            {f"enc{i}": Router(D_enc, router_arch) for i in self.enc_sites} | {f"aat{i}": Router(D_aat, router_arch) for i in self.aat_sites}
        ) if router == "learned" else nn.ModuleDict()
        self.gen = torch.Generator().manual_seed(seed) if router == "random" else None
        self.last_scores = {}
        # heavy path: `heavy` extra blocks (copies of the last AAT block, LayerScale at 0.1x so they start near
        # identity) after the AAT stack, run kv-full on the top-k_heavy patch tokens by a heavy router
        self.k_heavy, self.heavy_router = k_heavy, heavy_router
        self.heavy = nn.ModuleList(copy.deepcopy(m.info_sharing.self_attention_blocks[-1]) for _ in range(heavy))
        with torch.no_grad():
            for hb in self.heavy:
                hb.ls1.gamma.mul_(0.1)
                hb.ls2.gamma.mul_(0.1)
        if heavy and heavy_router == "learned":
            self.routers["heavy"] = Router(D_aat, router_arch)
        self.gen_heavy = torch.Generator().manual_seed(seed + 1) if heavy_router == "random" else None

    def score(self, name, x):
        if self.router_kind == "learned":
            s = self.routers[name](x.detach())
        elif self.router_kind == "norm":
            s = x.detach().norm(dim=-1)
        elif self.router_kind == "random":
            s = torch.rand(x.shape[:2], generator=self.gen).to(x.device)
        else:
            raise ValueError(self.router_kind)
        self.last_scores[name] = s
        return s

    def routed(self, name, fn, pre, patch, suf, blk=None, sc=None):
        """One MoD block. pre (B,P,C), patch (B,N,C), suf (B,S,C) -> updated (pre, patch, suf).
        blk / sc (the block and its attention scale) are needed only for kv_full."""
        k, N, C = self.k, patch.shape[1], patch.shape[2]
        s = self.score(name, patch)
        if k >= N:
            out = fn(torch.cat([pre, patch, suf], 1))
            P = pre.shape[1]
            return out[:, :P], out[:, P:P + N], out[:, P + N:]
        if self.coverage:
            if self._cnt is None:
                self._cnt = torch.zeros_like(s)
            s = (s - s.mean(-1, keepdim=True)) / (s.std(-1, keepdim=True) + 1e-6) - self.coverage * self._cnt
        idx = torch.topk(s, k, dim=-1).indices
        if self.coverage:
            self._cnt = self._cnt.scatter_add(1, idx, torch.ones_like(idx, dtype=s.dtype))
        idx = idx.unsqueeze(-1).expand(-1, -1, C)
        sel = torch.gather(patch, 1, idx)
        if self.kv_full:
            out = _block_kvfull(blk, torch.cat([pre, sel, suf], 1), torch.cat([pre, patch, suf], 1), sc)
        else:
            out = fn(torch.cat([pre, sel, suf], 1))
        P = pre.shape[1]
        patch = patch.scatter(1, idx, out[:, P:P + k])
        return out[:, :P], patch, out[:, P + k:]

    def forward(self, img):
        from uniception.models.prediction_heads.base import PredictionHeadTokenInput, AdaptorInput
        from uniception.models.utils.intermediate_feature_return import feature_take_indices

        m, H, W = self.m, self.hw[0], self.hw[1]
        B = img.shape[0]
        vit = m.encoder.model
        h, w = H // 14, W // 14
        n_pre = 1 + int(vit.num_register_tokens)
        self.last_scores = {}
        self._cnt = None

        # ---- encoder ----
        x = vit.prepare_tokens_with_masks(img)
        empty = x[:, :0]
        for i, blk in enumerate(vit.blocks):
            if i in self.enc_sites:
                pre, patch, _ = self.routed(f"enc{i}", blk, x[:, :n_pre], x[:, n_pre:], empty, blk)
                x = torch.cat([pre, patch], 1)
            else:
                x = blk(x)
        xn = vit.norm(x)
        C_enc = xn.shape[-1]
        feats = [xn[:, n_pre:].permute(0, 2, 1).reshape(B, C_enc, h, w).contiguous()]
        regs = [xn[:, :n_pre].permute(0, 2, 1).contiguous()] if m.use_register_tokens_from_encoder else None
        views = [{"img": img, "data_norm_type": ["dinov2"], "true_shape": torch.tensor([[H, W]] * B)}]
        feats = m._encode_and_fuse_optional_geometric_inputs(views, feats)
        scale_tok = m.scale_token.unsqueeze(0).unsqueeze(-1).repeat(B, 1, 1)   # (B, C, 1)

        # ---- alternating-attention stack (same token bookkeeping as come.ComeCore's port) ----
        isf = m.info_sharing
        assert isf.custom_positional_encoding is None and isf.norm_intermediate
        take, _ = feature_take_indices(isf.depth, isf.indices)
        feat = feats[0]
        HW = h * w
        toks = [feat.reshape(B, feat.shape[1], HW)]
        T_pv = 0
        if regs is not None:
            toks.append(regs[0])
            T_pv = regs[0].shape[2]
        z = torch.cat(toks, 2).permute(0, 2, 1)
        z = torch.cat([z, scale_tok.permute(0, 2, 1)], 1)
        z = isf.proj_embed(z)
        D = isf.dim
        n_view = HW + T_pv
        z = torch.cat([z[:, :n_view] + isf.view_pos_table[0].reshape(1, 1, D), z[:, n_view:]], 1)
        spatial, pv, add = z[:, :HW], z[:, HW:n_view], z[:, n_view:]

        def dmap(v):
            return isf.norm(v).reshape(B, h, w, D).permute(0, 3, 1, 2).contiguous()

        inters = []
        for d in range(isf.depth):
            blk = isf.self_attention_blocks[d]
            sc = blk.attn.scale
            fn = lambda t, blk=blk, sc=sc: _block(blk, t, None, sc)  # noqa: E731
            glob = d % 2 == 0
            suf = torch.cat([pv, add], 1) if glob else pv
            if d in self.aat_sites:
                _, spatial, suf = self.routed(f"aat{d}", fn, spatial[:, :0], spatial, suf, blk, sc)
            else:
                out = fn(torch.cat([spatial, suf], 1))
                spatial, suf = out[:, :HW], out[:, HW:]
            if glob:
                pv, add = suf[:, :T_pv], suf[:, T_pv:]
            else:
                pv = suf
            if d in take:
                inters.append(dmap(spatial))
        if len(self.heavy):
            if self.heavy_router == "learned":
                s = self.routers["heavy"](spatial.detach())
            else:
                s = torch.rand(spatial.shape[:2], generator=self.gen_heavy).to(spatial.device)
            self.last_scores["heavy"] = s
            idx = torch.topk(s, min(self.k_heavy, HW), dim=-1).indices.unsqueeze(-1).expand(-1, -1, D)
            suf = torch.cat([pv, add], 1)
            for hb in self.heavy:
                sel = torch.gather(spatial, 1, idx)
                out = _block_kvfull(hb, sel, torch.cat([spatial, suf], 1), hb.attn.scale)
                spatial = spatial.scatter(1, idx, out)
        final = dmap(spatial)
        add_final = isf.norm(add).permute(0, 2, 1).contiguous()
        dense = m.downstream_dense_head([feat, inters[0], inters[1], final], self.hw)
        sh = m.scale_head(PredictionHeadTokenInput(last_feature=add_final))
        sc = m.scale_adaptor(AdaptorInput(adaptor_feature=sh.decoded_channels, output_shape_hw=self.hw))
        return dense.value, sc.value


def save_gateroute(path, core, extra=None):
    torch.save({"k": core.k, "enc_sites": core.enc_sites, "aat_sites": core.aat_sites,
                "router": core.router_kind, "res": core.hw[0], "kv_full": core.kv_full,
                "heavy": len(core.heavy), "k_heavy": core.k_heavy, "heavy_router": core.heavy_router,
                "heavy_state_dict": core.heavy.state_dict(), "router_arch": core.router_arch, "coverage": core.coverage,
                "student_state_dict": core.m.state_dict(), "router_state_dict": core.routers.state_dict(),
                **(extra or {})}, path)


def load_gateroute(path, m, device="cuda", k=None, router=None, seed=0, coverage=None):
    """Build a GateRouteCore from a train_gateroute.py checkpoint on student `m` (weights replaced).
    k / router override the trained values (e.g. eval a learned router at another k)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["student_state_dict"])
    core = GateRouteCore(m, (ck["res"], ck["res"]), k or ck["k"], ck["enc_sites"], ck["aat_sites"],
                         router or ck["router"], seed, ck.get("kv_full", False), ck.get("heavy", 0),
                         ck.get("k_heavy", 0), ck.get("heavy_router", "learned"), ck.get("router_arch", "linear"),
                         ck.get("coverage", 0.0) if coverage is None else coverage)
    if core.router_kind == "learned" or "heavy" in core.routers:
        core.routers.load_state_dict(ck["router_state_dict"])
    core.heavy.load_state_dict(ck.get("heavy_state_dict", {}))
    return core.to(device).eval(), ck


@torch.no_grad()
def gateroute_selfcheck(device="cuda", res=378):
    """k = all tokens must reproduce MonoCore (the deployed dense graph)."""
    from model.gateroute.student_io import load_student, dense_core
    m = load_student(device=device)
    img = torch.randn(1, 3, res, res, device=device)
    a = dense_core(m, res)(img)
    b = GateRouteCore(m, (res, res), (res // 14) ** 2).to(device).eval()(img)
    return [float((x - y).abs().max()) for x, y in zip(a, b)]


if __name__ == "__main__":
    print("max |GateRoute(k=all) - MonoCore| (dense, scale):", gateroute_selfcheck())
