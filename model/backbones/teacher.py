"""Loader + mono-inference for the MapAnything teacher on the 8GB Jetson.

The 1.23B teacher is 4.23GB in fp32, which does NOT fit the 8GB unified-memory
board (fp32 resident + activations, or even the fp32 conversion transient, OOMs
the GPU and hard-reboots the Nano). It must run in bf16 (~2.1GB resident).

But the model is written for fp32 + autocast: it mixes bf16 weights with
internally-created fp32 tensors (default geometric inputs, positional-embedding
interpolation) in Linear, Conv AND LayerNorm. autocast can't be used with bf16
weights (it promotes LayerNorm/softmax to fp32, re-introducing the mismatch),
and setting the default dtype to bf16 breaks the fp32 numpy postprocessing.

Fix: keep weights bf16, run with use_amp=False, and register input-casting
forward-pre-hooks on EVERY weighted module (Linear/Conv/LayerNorm/...), casting
each float input to that module's weight dtype. This makes the whole network
uniformly bf16 without touching the fp32 postprocessing.
"""
import torch

from mapanything.models import MapAnything


def _match_weight_dtype(module, args):
    if not args or not torch.is_tensor(args[0]) or not args[0].is_floating_point():
        return None
    try:
        wdt = module.weight.dtype
        if not torch.is_floating_point(module.weight) or args[0].dtype == wdt:
            return None
    except Exception:
        return None  # e.g. torchao-quantized weight tensor subclass
    return (args[0].to(wdt),) + tuple(args[1:])


def add_dtype_casting_hooks(model):
    """Hook every module that owns a float `weight` to cast its float input to
    that weight's dtype. Covers Linear, Conv*, LayerNorm, GroupNorm, etc."""
    n = 0
    for mod in model.modules():
        w = getattr(mod, "weight", None)
        if torch.is_tensor(w) and w.is_floating_point() and not isinstance(mod, torch.nn.Embedding):
            mod.register_forward_pre_hook(_match_weight_dtype)
            n += 1
    return n


def _upcast_geometry_postproc():
    """The geometry postprocessing (raydirs+depth+pose -> pointmap, pose math) is
    written in fp32 and breaks when fed the bf16 network outputs. Wrap those
    functions in the model's namespace to upcast float tensor args to fp32. Cheap
    (small per-frame math) and makes the whole postprocessing fp32-consistent."""
    import mapanything.models.mapanything.model as M
    names = [n for n in dir(M) if "pointmap" in n or "pose" in n or "ray_dirs" in n
             or "quats" in n or "depth_along_ray" in n]
    import functools

    def wrap(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            def up(x):
                return x.float() if torch.is_tensor(x) and x.dtype == torch.bfloat16 else x
            return fn(*[up(a) for a in args], **{k: up(v) for k, v in kwargs.items()})
        return inner

    for n in names:
        obj = getattr(M, n)
        if callable(obj) and not getattr(obj, "_merlin_wrapped", False):
            w = wrap(obj); w._merlin_wrapped = True
            setattr(M, n, w)


def patch_linalg_cpu_fallback():
    """JetPack 6.2's cuSOLVER lacks symbols torch 2.11's libtorch_cuda_linalg.so
    needs (cusolverDnXsyevBatched_bufferSize), so any CUDA torch.linalg.* op
    fails to dlopen. The postprocessing uses torch.linalg.solve (intrinsics from
    rays) on tiny systems -- route these ops through the CPU (negligible cost)."""
    import functools
    if getattr(torch.linalg, "_merlin_cpu_patched", False):
        return
    for name in ("solve", "lstsq", "svd", "inv", "eigh", "qr", "pinv", "cholesky"):
        fn = getattr(torch.linalg, name, None)
        if fn is None:
            continue

        def make(fn):
            @functools.wraps(fn)
            def inner(*args, **kwargs):
                dev = next((a.device for a in args if torch.is_tensor(a)), None)
                if dev is not None and dev.type == "cuda":
                    cargs = [a.cpu() if torch.is_tensor(a) else a for a in args]
                    ckw = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in kwargs.items()}
                    out = fn(*cargs, **ckw)
                    if torch.is_tensor(out):
                        return out.to(dev)
                    if isinstance(out, tuple):
                        return type(out)(o.to(dev) if torch.is_tensor(o) else o for o in out)
                    return out
                return fn(*args, **kwargs)
            return inner
        setattr(torch.linalg, name, make(fn))
    torch.linalg._merlin_cpu_patched = True


def load_teacher(hf_id="facebook/map-anything-apache", device="cuda"):
    """from_pretrained (fp32 CPU, swap-absorbed) -> uniform bf16 -> GPU (~2.1GB).
    Never materializes fp32 on the GPU."""
    model = MapAnything.from_pretrained(hf_id)
    model = model.to(torch.bfloat16).eval()
    add_dtype_casting_hooks(model)
    _upcast_geometry_postproc()
    patch_linalg_cpu_fallback()
    return model.to(device)


load_teacher_bf16 = load_teacher


def mono_infer(model, views):
    """Mono RGB inference, pure bf16 (no autocast). Hooks keep dtypes consistent."""
    with torch.no_grad():
        return model.infer(
            views, memory_efficient_inference=True, minibatch_size=1,
            use_amp=False, apply_mask=False, mask_edges=False,
        )
