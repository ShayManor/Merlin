#!/usr/bin/env python3
"""Perception wrapper for the drone closed-loop sim: RGB frame -> metric z-depth at a
chosen operating point (input resolution), using a distilled MERLIN student.

Mirrors eval/sim/perception.py (same MapAnything load_images preprocessing + student
forward, so drone perception is byte-identical to the offline evals), with one change:
metric scale comes from the IMU visual-inertial solve (eval/drone/vi.vi_scale), not a
per-episode GT-depth align. set_scale() lets a caller inject a known scale (e.g. a GT
upper-bound arm); calibrate_from_imu() recovers it from the first keyframes + PX4 IMU.
"""
import json
import os
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "model", "distill"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "model", "backbones"))
from student import build_student  # noqa: E402


def decompress_steps(depth, k=41, alpha=0.18, cap=0.35):
    """Invert the student's measured depth-step compression (a calibration, not a retrain).

    Measured on held-out frames: inside an object's silhouette the student recovers only
    ~82% of the background-to-object depth step, i.e. pred = gt + alpha*(bg - gt) with
    alpha ~= 0.18. Because the error is proportional to the LOCAL STEP it vanishes on a flat
    wall (where bg == gt) and is largest for a small object against a far background --
    which is exactly the non-uniformity that made a single global safety margin unable to
    cover both. Estimating bg by a local maximum over a k x k window and solving for gt:

        gt = (pred - alpha * bg) / (1 - alpha)

    On a flat surface bg == pred and this is the identity, so unlike a global margin or a
    uniform depth rescale it costs nothing where the student was already right. k must be
    wide enough to reach past a typical obstacle to the surface behind it.

    `cap` bounds the correction, and is not optional: alpha was fitted on obstacle silhouettes
    whose depth step is ~1-1.5 m, but a window straddling a doorway or an open corridor sees a
    step of many metres, and the unbounded form drove those pixels to zero -- they then fell
    out of the map entirely and the drone hit the obstacle in 5 of 5 episodes. The measured
    displacement is ~0.27 m, so the correction is clamped just above it.
    """
    if not k or alpha <= 0:
        return depth
    import cv2
    d = np.asarray(depth, np.float32)
    bg = cv2.dilate(d, np.ones((int(k), int(k)), np.uint8))
    corr = np.minimum(float(alpha) * (bg - d) / (1.0 - float(alpha)), float(cap))
    return np.maximum(d - corr, 0.05)


def sharpen_edges(depth, k):
    """Undo the student's over-smoothing of depth discontinuities (a calibration, not a model
    change).

    Measured on held-out frames: inside an injected obstacle's silhouette the student recovers
    only ~82% of the background-to-object depth step, so it places the object's near face
    ~0.27 m too far and the planner keeps its margin from a surface that is not there. A local
    MINIMUM over a k x k window restores it: on a flat wall the neighbourhood minimum is the
    centre value, so the transform is near-identity, and it only bites where depth changes
    quickly -- which is exactly where the smoothing error lives. The cost is that obstacles
    dilate by k/2 pixels, so k is chosen offline against phantom-blockage, not by eye.
    """
    if not k or k < 2:
        return depth
    import cv2
    return cv2.erode(np.asarray(depth, np.float32), np.ones((int(k), int(k)), np.uint8))


class Perception:
    def __init__(self, ckpt, device="cuda", aat_depth=None):
        import cv2  # noqa
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.model = build_student(size=ck.get("size", "base"),
                                   aat_depth=aat_depth or ck.get("aat_depth", 8),
                                   device="cpu")
        # An ORE checkpoint carries OREResidual sub-modules at the wrapped sites, so the plain
        # student's key set will not match it. Rebuild the same wrapping from the metadata the
        # checkpoint records before loading, and re-register the frozen-feature routing hook if
        # the run used it -- without that hook every site raises for a missing routing feature.
        if ck.get("arm") == "ORE" and ck.get("sites"):
            import sys as _sys
            _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "..", "..", "model"))
            from moe.ore import upcycle_ore, set_route_feature
            rd = int(ck.get("route_dim") or 0)
            if not rd:
                # run_cell.py's post-loop save omits route_dim (and step), so a completed run's
                # untagged checkpoint carries None even though it was trained with frozen-feature
                # routing. Recover it from the gate: gate.weight is (K, gate_in), and gate_in is
                # route_dim when routing on a frozen feature and the site's own d_in otherwise.
                # Ambiguity to respect: a single (K, 768) gate could equally be route_dim=768 or an
                # fc1 site with route_dim=0. Frozen-feature routing gives EVERY site the same gate
                # input, whereas per-site routing gives fc1 768 and fc2 3072, so agreement across
                # all gates is what distinguishes them.
                widths = {v.shape[1] for k, v in ck["state_dict"].items()
                          if k.endswith(".gate.weight") and v.shape[0] == int(ck["k"])}
                if len(widths) == 1:
                    rd = widths.pop()
                    print(f"[perception] route_dim missing from checkpoint, inferred {rd} "
                          f"from {len(ck['sites'])} agreeing gates", flush=True)
            upcycle_ore(self.model, ck["sites"], num_experts=int(ck["k"]), rank=int(ck["rank"]),
                        n_layers=int(ck.get("n_layers", 2)), route_dim=rd)
            if rd:
                blk0 = self.model.info_sharing.self_attention_blocks[0]
                blk0.register_forward_pre_hook(
                    lambda m, i: set_route_feature(self.model, i[0].detach()))
            print(f"[perception] ORE checkpoint: K={ck['k']} r={ck['rank']} route_dim={rd}, "
                  f"{len(ck['sites'])} sites", flush=True)
        elif ck.get("arm") == "U" and ck.get("sites"):
            # Full-FFN upcycling (L-FCR-02). Same rebuild-before-load problem as ORE, plus one
            # that ORE does not have: arm U's FORWARD depends on top_k and council_m, and
            # checkpoints written before 2026-08-14 record neither. Loading a top-2 checkpoint
            # as top-1 evaluates a DIFFERENT FUNCTION and would look like a real C4 result, so
            # this reads the sidecar JSON and refuses to guess.
            import sys as _sys
            _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "..", "..", "model"))
            from moe.ffn_moe import upcycle_ffn
            from moe.ore import set_route_feature
            tk, cm = ck.get("top_k"), ck.get("council_m")
            # router_norm whitens the ROUTER INPUT (ore.moe._norm_in) and k-means init puts the
            # gate rows on centroids of whitened features, so it is part of the function exactly
            # as top_k is. Defaulting it cost the whole upcycling campaign: every cell trained
            # with --router-init kmeans and flew here at the upcycle_ffn default of False, which
            # on moe_k8_step1000 collapsed site-0 routing from expert 0 carrying 35% of tokens to
            # expert 4 carrying 73%, and moved whole_absrel 0.2075 -> 0.2239.
            rn = ck.get("router_norm")
            if tk is None or rn is None:
                side = os.path.splitext(ckpt)[0] + ".json"
                base = side.split("_step")[0] + ".json" if "_step" in side else side
                for cand in (side, base):
                    if os.path.exists(cand):
                        j = json.load(open(cand))
                        if tk is None:
                            tk, cm = j.get("top_k"), j.get("council_m")
                            print(f"[perception] top_k/council_m missing from checkpoint, read "
                                  f"{tk}/{cm} from {os.path.basename(cand)}", flush=True)
                        if rn is None and j.get("router_norm") is not None:
                            rn = j["router_norm"]
                            print(f"[perception] router_norm missing from checkpoint, read {rn} "
                                  f"from {os.path.basename(cand)}", flush=True)
                        break
            if tk is None:
                raise ValueError(
                    f"{ckpt} is an arm-U checkpoint with no top_k recorded and no sidecar JSON "
                    "beside it. Running it at the wrong top_k silently evaluates a different "
                    "model; copy the run's .json next to the checkpoint instead of guessing.")
            if rn is None:
                raise ValueError(
                    f"{ckpt} is an arm-U checkpoint with no router_norm recorded, in it or in a "
                    "sidecar JSON. It is True for any run trained with --router-init kmeans. "
                    "Verify which by rebuilding both ways and matching the run's recorded "
                    "whole_absrel, then record it; guessing evaluates a different model.")
            upcycle_ffn(self.model, ck["sites"], num_experts=int(ck["k"]), top_k=int(tk),
                        council_m=int(cm or 0), router_norm=bool(rn),
                        route_dim=int(ck.get("route_dim") or 0))
            if ck.get("route_dim"):
                blk0 = self.model.info_sharing.self_attention_blocks[0]
                blk0.register_forward_pre_hook(
                    lambda m, i: set_route_feature(self.model, i[0].detach()))
            print(f"[perception] upcycled checkpoint: K={ck['k']} top_k={tk} council={cm or 0} "
                  f"router_norm={bool(rn)}, {len(ck['sites'])} sites", flush=True)
        elif ck.get("ffn_widen"):
            # Rebuild the widened AAT FFNs. The widths come from the CHECKPOINT'S OWN SHAPES
            # rather than from the ffn_widen metadata: the tensor is the ground truth, and a
            # metadata/tensor mismatch would otherwise surface as a confusing strict-load error.
            # Only info_sharing is widened (run_cell.widen_ffn touches those 16 Linears); the
            # encoder blocks keep their 3072 hidden dim, so this must not loop over them.
            # No function-preserving init needed here -- every weight is overwritten by the load.
            import torch.nn as _nn
            sd = ck["state_dict"]
            for i, blk in enumerate(self.model.info_sharing.self_attention_blocks):
                pre = f"info_sharing.self_attention_blocks.{i}.mlp"
                hid, din = sd[f"{pre}.fc1.weight"].shape
                dout = sd[f"{pre}.fc2.weight"].shape[0]
                blk.mlp.fc1 = _nn.Linear(din, hid, bias=f"{pre}.fc1.bias" in sd)
                blk.mlp.fc2 = _nn.Linear(hid, dout, bias=f"{pre}.fc2.bias" in sd)
            print(f"[perception] widened checkpoint: AAT FFN hidden {hid} "
                  f"(+{ck['ffn_widen']}), {len(self.model.info_sharing.self_attention_blocks)} "
                  f"blocks", flush=True)
        self.model.load_state_dict(ck["state_dict"])
        self.model = self.model.to(device).eval()
        self.device = device
        self.scale = 1.0  # set by set_scale() or calibrate_from_imu()
        self.calib_tab = None  # set by set_depth_calib()
        self.calib_q = 0.20
        self._tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        from mapanything.utils.image import load_images
        self._load_images = load_images

    @torch.no_grad()
    def _forward(self, rgb, res, aat_depth=None):
        """rgb: (H,W,3) uint8 RGB. Returns (up-to-scale z-depth, cam_trans, cam_quats) from
        one student forward. cam_trans/cam_quats are the student's per-frame visual pose
        (up-to-scale translation, xyzw rotation) used for the C2/C3 visual-inertial chain."""
        import cv2
        cv2.imwrite(self._tmp, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        s = (res // 14) * 14
        v = self._load_images([self._tmp], resize_mode="square", size=s)
        v[0]["img"] = v[0]["img"].to(self.device)
        if aat_depth is not None:
            self.model.info_sharing.depth = aat_depth
        pr = self.model(v, memory_efficient_inference=True, minibatch_size=1)[0]
        depth = pr["pts3d_cam"][0, ..., 2].float().cpu().numpy()
        cam_trans = pr["cam_trans"][0].float().cpu().numpy()
        cam_quats = pr["cam_quats"][0].float().cpu().numpy()
        return depth, cam_trans, cam_quats

    def _raw_depth(self, rgb, res, aat_depth=None):
        """rgb: (H,W,3) uint8 RGB. Returns up-to-scale z-depth at student native size."""
        return self._forward(rgb, res, aat_depth)[0]

    def depth_and_pose(self, rgb, res=378, aat_depth=None):
        """Metric depth + the student visual pose (cam_trans up-to-scale, cam_quats xyzw)
        for one frame. Used by the C2/C3 driver to log a visual trajectory alongside GT."""
        depth, cam_trans, cam_quats = self._forward(rgb, res, aat_depth)
        return depth * self.scale, cam_trans, cam_quats

    @torch.no_grad()
    def forward_window(self, rgbs, res=378):
        """Run the student on a WINDOW of frames (list of (H,W,3) uint8 RGB) in one forward.
        MapAnything estimates each frame's pose in the window's c0 (first-frame) frame, so a
        single window yields a short up-to-scale visual trajectory. Single-frame inference
        gives no useful relative pose, so C2 (scale) and C3 (drift) run windowed like
        eval/vi_scale.py. Returns (cam_trans (N,3) up-to-scale, cam_quats (N,4) xyzw)."""
        import cv2
        s = (res // 14) * 14
        paths = []
        for i, rgb in enumerate(rgbs):
            p = f"{self._tmp[:-4]}_w{i}.png"
            cv2.imwrite(p, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            paths.append(p)
        v = self._load_images(paths, resize_mode="square", size=s)
        for x in v:
            x["img"] = x["img"].to(self.device)
        out = self.model(v, memory_efficient_inference=True, minibatch_size=len(v))
        cam_trans = np.array([o["cam_trans"][0].float().cpu().numpy() for o in out])
        cam_quats = np.array([o["cam_quats"][0].float().cpu().numpy() for o in out])
        return cam_trans, cam_quats

    def set_depth_calib(self, path, q=0.20):
        """Load a fitted per-pixel apparent->real depth table (eval/drone/depth_calib).

        `q` is the quantile of gt/pred the table returns, and it is the difference between a
        correction that helps and one that hurts: q=0.5 is the minimum-error fit, which is
        defeated by regression dilution, while a low q asks how close the surface could be and
        so pulls in exactly the cells where the student is unreliable.
        """
        from eval.drone.depth_calib import DepthCalib
        self.calib_tab = DepthCalib.load(path) if path else None
        self.calib_q = float(q)
        return self.calib_tab

    def apply_depth_calib(self, depth):
        """Metric depth -> calibrated depth; identity if no table was loaded."""
        if self.calib_tab is None:
            return depth
        return self.calib_tab.apply(depth, self.calib_q)

    def set_scale(self, s):
        """Inject a known metric scale (e.g. a GT-scale upper-bound arm)."""
        self.scale = float(s)
        return self.scale

    def calibrate_gt(self, rgb, gt_depth, res=378):
        """Set the per-episode metric scale from the median GT/pred ratio (the 'gt-scale
        student' arm, which isolates depth-shape quality from the scale problem; the IMU
        path in calibrate_from_imu is the C2 alternative). Mirrors eval/sim/perception."""
        import cv2
        pred = self._raw_depth(rgb, res)
        if pred.shape != gt_depth.shape:
            gt = cv2.resize(gt_depth, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            gt = gt_depth
        v = np.isfinite(gt) & np.isfinite(pred) & (gt > 0.1) & (gt < 10) & (pred > 0.05)
        if v.sum() > 200:
            self.scale = float(np.median(gt[v] / pred[v]))
        return self.scale

    def calibrate_from_imu(self, P_vis, R_list, kf_t, imu_t, imu_acc):
        """Recover the metric scale (C2) from the visual keyframe trajectory + PX4 IMU
        via the visual-inertial linear solve. Keeps the previous scale if the solve is
        degenerate (non-finite or non-positive)."""
        from eval.drone.vi import vi_scale
        s = vi_scale(P_vis, R_list, kf_t, imu_t, imu_acc)
        if np.isfinite(s) and s > 0:
            self.scale = float(s)
        return self.scale

    def depth_gt_scaled(self, rgb, gt_depth, res=378):
        """Depth with the metric scale re-fitted against GT on THIS frame.

        An oracle diagnostic, not a deployable arm. calibrate_gt fixes one scale for the whole
        episode; measured offline, re-fitting per frame cuts median depth error 0.270 -> 0.205 m
        (24%). Running it closed-loop separates the two ways the student can be wrong: if this
        navigates well, what limits MERLIN is metric SCALE (which the IMU chain is meant to
        supply on the robot), and if it does not, the limit is depth SHAPE and no scale
        estimator will help.
        """
        import cv2
        pred = self._raw_depth(rgb, res)
        gt = gt_depth
        if pred.shape != gt.shape:
            gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
        v = np.isfinite(gt) & np.isfinite(pred) & (gt > 0.1) & (gt < 10) & (pred > 0.05)
        if v.sum() < 200:
            return pred * self.scale
        return pred * float(np.median(gt[v] / pred[v]))

    def depth(self, rgb, res=378, aat_depth=None, calib=0):
        """Metric z-depth (meters) at the chosen operating point, scaled by the
        recovered/injected metric scale. `calib` applies the edge-sharpening calibration
        (see sharpen_edges). Returns (H,W) float."""
        d = self.apply_depth_calib(self._raw_depth(rgb, res, aat_depth) * self.scale)
        return sharpen_edges(d, calib)


class TeacherPerception:
    """The undistilled MapAnything backbone behind Perception's interface.

    The student is a distilled copy of this architecture, so `_forward` and everything
    downstream of it -- image loading, the pts3d_cam read, GT scale calibration, the depth
    calibration table -- are identical. Only the weights and their cost differ, which is
    exactly what a C4 arm comparison should isolate: the closed-loop question is whether the
    top-of-line model actually navigates better, not whether it reconstructs better offline.

    Implemented by borrowing Perception's methods rather than subclassing its __init__, since
    that loads a distilled checkpoint from disk and this loads pretrained weights from the hub.
    """

    _forward = Perception._forward
    _raw_depth = Perception._raw_depth
    depth_and_pose = Perception.depth_and_pose
    # same architecture and the same (model, device, _tmp, _load_images) attributes, so the
    # windowed pose forward is borrowed verbatim rather than reimplemented
    forward_window = Perception.forward_window
    set_depth_calib = Perception.set_depth_calib
    apply_depth_calib = Perception.apply_depth_calib
    set_scale = Perception.set_scale
    calibrate_gt = Perception.calibrate_gt
    depth_gt_scaled = Perception.depth_gt_scaled
    depth = Perception.depth

    def __init__(self, hf_id="facebook/map-anything-apache", device="cuda"):
        import cv2  # noqa
        from mapanything.models import MapAnything
        from mapanything.utils.image import load_images
        self.model = MapAnything.from_pretrained(hf_id).to(device).eval()
        self.device = device
        self.scale = 1.0
        self.calib_tab = None
        self.calib_q = 0.20
        self._tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        self._load_images = load_images




class DepthAnythingPerception:
    """Depth Anything V2 (metric) behind Perception's interface, as a C4 arm.

    The strongest independently-authored monocular depth model that runs on the Nano, and the
    one baseline whose CAPABILITY was unmeasured while its throughput was: the SWaP-C table had
    it at 13.97 FPS with no idea whether it can actually navigate.

    It emits depth only, no relative pose, which is why it cannot replace MERLIN in the full
    system -- but the C4 planner consumes only depth, so as a navigation arm the comparison is
    exact. Scale is handled by the same per-episode GT calibration every other arm uses, so an
    arm cannot win or lose on scale recovery here.
    """

    set_depth_calib = Perception.set_depth_calib
    apply_depth_calib = Perception.apply_depth_calib
    set_scale = Perception.set_scale

    def __init__(self, hf_id="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
                 device="cuda"):
        import torch as _t
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self._t = _t
        self.proc = AutoImageProcessor.from_pretrained(hf_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(hf_id).to(device).eval()
        self.device = device
        self.scale = 1.0
        self.calib_tab = None
        self.calib_q = 0.20

    def _raw_depth(self, rgb, res, aat_depth=None):
        import numpy as _np
        with self._t.no_grad():
            inp = self.proc(images=rgb, return_tensors="pt").to(self.device)
            d = self.model(**inp).predicted_depth
            d = self._t.nn.functional.interpolate(d[:, None], size=(res, res),
                                                  mode="bilinear", align_corners=False)
        return d[0, 0].float().cpu().numpy().astype(_np.float32)

    def calibrate_gt(self, rgb, gt_depth, res=378):
        """Same median-ratio calibration the student uses, so scale is not the variable."""
        import numpy as _np
        import cv2
        p = self._raw_depth(rgb, res)
        g = gt_depth
        if g.shape != p.shape:
            g = cv2.resize(g, (p.shape[1], p.shape[0]), interpolation=cv2.INTER_NEAREST)
        v = _np.isfinite(g) & _np.isfinite(p) & (g > 0.1) & (g < 10) & (p > 0.05)
        self.scale = float(_np.median(g[v] / p[v])) if v.sum() > 200 else 1.0
        return self.scale

    def depth(self, rgb, res=378, aat_depth=None, calib=0):
        d = self._raw_depth(rgb, res) * self.scale
        return self.apply_depth_calib(d) if self.calib_tab is not None else d


class VGGTPerception:
    """VGGT (Visual Geometry Grounded Transformer, CVPR 2025) behind Perception's interface.

    The fair top-of-line baseline: VGGT is MERLIN's own model class -- a feed-forward
    transformer that outputs depth AND camera pose from images -- unlike a depth-only network,
    which the C4 harness would silently hand the pose and scale it cannot itself produce. VGGT
    is up-to-scale, so it takes the same per-episode GT scale calibration the student does; the
    comparison is then depth SHAPE at matched scale, exactly as for the student and teacher.

    ~1B parameters, so like MapAnything it is a top-of-line point that does not deploy on the
    Nano; that is measured separately, not assumed.
    """

    set_depth_calib = Perception.set_depth_calib
    apply_depth_calib = Perception.apply_depth_calib
    set_scale = Perception.set_scale

    def __init__(self, hf_id="facebook/VGGT-1B-Commercial", device="cuda"):
        import cv2  # noqa
        import torch as _t
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        self._t = _t
        self._load = load_and_preprocess_images
        self.model = VGGT.from_pretrained(hf_id).to(device).eval()
        self.device = device
        self.scale = 1.0
        self.calib_tab = None
        self.calib_q = 0.20
        self._tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name

    def _raw_depth(self, rgb, res, aat_depth=None):
        import cv2
        import numpy as _np
        cv2.imwrite(self._tmp, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        imgs = self._load([self._tmp]).to(self.device)
        with self._t.no_grad(), self._t.autocast("cuda", dtype=self._t.bfloat16):
            tok, ps = self.model.aggregator(imgs[None])
            depth_map, _ = self.model.depth_head(tok, imgs[None], ps)
        d = depth_map[0, 0, ..., 0].float().cpu().numpy()
        if d.shape != (res, res):
            d = cv2.resize(d, (res, res), interpolation=cv2.INTER_NEAREST)
        return d.astype(_np.float32)

    def forward_window(self, rgbs, res=378):
        """Windowed VGGT pose, unpacked to the SAME convention MapAnything's forward_window
        returns: camera centres and orientations in the window's first-frame camera frame,
        quats xyzw. VGGT's camera head emits a world->camera extrinsic whose world IS frame 0,
        so the only work is inverting each pose; nothing is re-estimated here.
        """
        import cv2
        import numpy as _np
        from scipy.spatial.transform import Rotation
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        paths = []
        for i, rgb in enumerate(rgbs):
            p = f"{self._tmp[:-4]}_w{i}.png"
            cv2.imwrite(p, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            paths.append(p)
        imgs = self._load(paths).to(self.device)
        with self._t.no_grad(), self._t.autocast("cuda", dtype=self._t.bfloat16):
            tok, ps = self.model.aggregator(imgs[None])
            pose_enc = self.model.camera_head(tok)[-1]
        extr, _ = pose_encoding_to_extri_intri(pose_enc.float(), imgs.shape[-2:])
        extr = extr[0].float().cpu().numpy()          # (N,3,4), world->cam, world = frame 0
        trans, quats = [], []
        for E in extr:
            Rc2w = E[:3, :3].T                        # cam -> frame0
            trans.append(-Rc2w @ E[:3, 3])            # camera centre in frame-0 coords
            quats.append(Rotation.from_matrix(Rc2w).as_quat())   # xyzw
        return _np.array(trans), _np.array(quats)

    def calibrate_gt(self, rgb, gt_depth, res=378):
        import numpy as _np
        import cv2
        p = self._raw_depth(rgb, res)
        g = gt_depth
        if g.shape != p.shape:
            g = cv2.resize(g, (p.shape[1], p.shape[0]), interpolation=cv2.INTER_NEAREST)
        v = _np.isfinite(g) & _np.isfinite(p) & (g > 0.1) & (g < 10) & (p > 0.05)
        self.scale = float(_np.median(g[v] / p[v])) if v.sum() > 200 else 1.0
        return self.scale

    def depth(self, rgb, res=378, aat_depth=None, calib=0):
        d = self._raw_depth(rgb, res) * self.scale
        return self.apply_depth_calib(d) if self.calib_tab is not None else d
