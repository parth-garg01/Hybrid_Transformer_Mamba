#!/usr/bin/env python3
"""Mamba GPU compatibility check. Runs ON Kaggle (needs CUDA + mamba_ssm).

    python scripts/kaggle/compat_check.py [--out DIR] [--quick]

Writes DIR/compat_check.json (DIR defaults to $ABR_OUTPUT_DIR or ./compat_out). The file
is rewritten after every check, so partial results survive a crash. Every check is
wrapped in try/except; a failure records the full traceback and the script moves on.

Checks
  env                     GPU name, compute capability, torch/CUDA/driver, package versions
  mamba1_fwd_bwd          Mamba-1 forward+backward, d_model=256, L=256, B=8, fp32/fp16/bf16
  mamba1_fast_vs_ref      fused CUDA path vs the pure-PyTorch reference path (fp32)
  mamba2_fwd_bwd          Mamba-2 forward+backward, same shapes and dtypes
  chunk_scan_initial_states
                          mamba_chunk_scan_combined with initial_states + return_final_states:
                          output/final-state/grad(initial_states) vs a naive fp32 recurrence
  chunked_vs_single_kernel
                          one call over L vs several calls passing final_states -> initial_states
  chunked_vs_single_mamba2_module
                          full Mamba2 layer forward vs a block-by-block forward that carries
                          both the SSM state and the conv state (what gated CBSC needs)
  mamba1_initial_state_api
                          what Mamba-1 exposes for passing an initial state (signatures + source)
  throughput              tokens/s and peak memory, Mamba-1 and Mamba-2 layers, L=256/1024/4096
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

RESULTS: dict = {"started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "checks": {}}
OUT_PATH = None

D_MODEL, SEQ, BATCH = 256, 256, 8


def save():
    RESULTS["updated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(OUT_PATH, "w") as f:
        json.dump(RESULTS, f, indent=2, sort_keys=False, default=str)


def check(name):
    def deco(fn):
        def wrapper(*a, **k):
            t0 = time.time()
            print(f"=== {name} ===", flush=True)
            try:
                res = fn(*a, **k)
                entry = {"ok": True, **(res or {})}
            except Exception as e:
                entry = {"ok": False, "error": repr(e), "traceback": traceback.format_exc()}
                print(entry["traceback"], flush=True)
            entry["seconds"] = round(time.time() - t0, 2)
            RESULTS["checks"][name] = entry
            print(json.dumps({k: v for k, v in entry.items() if k != "traceback"},
                             default=str)[:3000], flush=True)
            save()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            return entry
        return wrapper
    return deco


def sub(fn, *a, **k):
    """Run one sub-case; return its result dict or the traceback."""
    try:
        return {"ok": True, **(fn(*a, **k) or {})}
    except Exception as e:
        return {"ok": False, "error": repr(e), "traceback": traceback.format_exc()}


# ---------------------------------------------------------------------------------
def dtypes_to_test(torch):
    cap = torch.cuda.get_device_capability(0)
    out = [("fp32", None), ("fp16_autocast", torch.float16)]
    # T4 (sm_75) and P100 (sm_60) have no native bf16; still try it but label it.
    out.append(("bf16_autocast" + ("" if cap[0] >= 8 else "_not_native"), torch.bfloat16))
    return out


@check("env")
def check_env():
    import torch
    info = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    try:
        info["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=60).stdout.strip()
        smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=60).stdout
        m = re.search(r"CUDA Version:\s*([\d.]+)", smi)
        info["driver_cuda_version"] = m.group(1) if m else None
        m = re.search(r"Driver Version:\s*([\d.]+)", smi)
        info["driver_version"] = m.group(1) if m else None
    except Exception as e:
        info["nvidia_smi_error"] = repr(e)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu_name=p.name, compute_capability=list(torch.cuda.get_device_capability(0)),
                    gpu_count=torch.cuda.device_count(), total_memory_gb=round(p.total_memory / 2**30, 2),
                    bf16_native=torch.cuda.get_device_capability(0)[0] >= 8,
                    torch_arch_list=torch.cuda.get_arch_list())
        try:
            info["torch_is_bf16_supported"] = torch.cuda.is_bf16_supported()
        except Exception as e:
            info["torch_is_bf16_supported"] = repr(e)
    import importlib.metadata as md
    vers = {}
    for pkg in ["mamba-ssm", "causal-conv1d", "triton", "transformers", "einops", "numpy"]:
        try:
            vers[pkg] = md.version(pkg)
        except Exception:
            vers[pkg] = None
    info["versions"] = vers
    RESULTS["gpu_name"] = info.get("gpu_name")
    RESULTS["compute_capability"] = info.get("compute_capability")
    return info


def _fwd_bwd(torch, layer, dtype, B=BATCH, L=SEQ, D=D_MODEL, ref_out=None):
    torch.manual_seed(0)
    x = torch.randn(B, L, D, device="cuda", requires_grad=True)
    layer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    if dtype is None:
        y = layer(x)
    else:
        with torch.autocast("cuda", dtype=dtype):
            y = layer(x)
    loss = y.float().pow(2).mean()
    loss.backward()
    torch.cuda.synchronize()
    grads = [p.grad for p in layer.parameters() if p.grad is not None]
    res = {
        "out_dtype": str(y.dtype), "out_shape": list(y.shape),
        "out_finite": bool(torch.isfinite(y).all()),
        "input_grad_finite": bool(torch.isfinite(x.grad).all()),
        "input_grad_abs_mean": float(x.grad.abs().mean()),
        "param_grads_finite": all(bool(torch.isfinite(g).all()) for g in grads),
        "n_params_with_grad": len(grads),
        "n_params": sum(1 for _ in layer.parameters()),
        "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
    }
    if ref_out is not None:
        res["max_abs_diff_vs_fp32"] = float((y.float() - ref_out).abs().max())
    return res, y.detach().float()


@check("mamba1_fwd_bwd")
def check_mamba1():
    import torch
    from mamba_ssm import Mamba
    torch.manual_seed(0)
    layer = Mamba(d_model=D_MODEL, d_state=16, d_conv=4, expand=2).cuda()
    out, ref = {}, None
    for label, dt in dtypes_to_test(torch):
        r = sub(lambda: _fwd_bwd(torch, layer, dt, ref_out=ref)[0])
        if label == "fp32" and r["ok"]:
            ref = _fwd_bwd(torch, layer, None)[1]
        out[label] = r
    return {"config": dict(d_model=D_MODEL, d_state=16, d_conv=4, expand=2, seq=SEQ, batch=BATCH),
            "cases": out, "all_ok": all(v["ok"] for k, v in out.items() if "not_native" not in k)}


@check("mamba1_fast_vs_ref")
def check_mamba1_ref():
    import torch
    from mamba_ssm import Mamba
    torch.manual_seed(0)
    fast = Mamba(d_model=D_MODEL, d_state=16, d_conv=4, expand=2, use_fast_path=True).cuda()
    slow = Mamba(d_model=D_MODEL, d_state=16, d_conv=4, expand=2, use_fast_path=False).cuda()
    slow.load_state_dict(fast.state_dict())
    x = torch.randn(BATCH, SEQ, D_MODEL, device="cuda")
    with torch.no_grad():
        a, b = fast(x), slow(x)
    return {"max_abs_diff": float((a - b).abs().max()), "out_abs_mean": float(a.abs().mean())}


@check("mamba2_fwd_bwd")
def check_mamba2():
    import torch
    from mamba_ssm import Mamba2
    torch.manual_seed(0)
    layer = Mamba2(d_model=D_MODEL, d_state=64, d_conv=4, expand=2, headdim=64).cuda()
    out, ref = {}, None
    for label, dt in dtypes_to_test(torch):
        r = sub(lambda: _fwd_bwd(torch, layer, dt, ref_out=ref)[0])
        if label == "fp32" and r["ok"]:
            ref = _fwd_bwd(torch, layer, None)[1]
        out[label] = r
    cfg = dict(d_model=D_MODEL, d_state=64, d_conv=4, expand=2, headdim=64,
               nheads=layer.nheads, ngroups=layer.ngroups, chunk_size=layer.chunk_size,
               d_ssm=getattr(layer, "d_ssm", None), seq=SEQ, batch=BATCH,
               ssm_state_shape_per_layer=["batch", layer.nheads, layer.headdim, layer.d_state],
               conv_state_shape_per_layer=["batch", layer.conv1d.weight.shape[0], layer.d_conv - 1])
    return {"config": cfg, "cases": out,
            "all_ok": all(v["ok"] for k, v in out.items() if "not_native" not in k)}


# ---------------------------------------------------------------------------------
# SSD reference + chunk scan checks
# ---------------------------------------------------------------------------------
def naive_ssd(torch, x, dt, A, B, C, D, h0):
    """fp32 recurrence. x (b,l,h,p) dt (b,l,h) [already positive] A (h) B,C (b,l,g,n) D (h)
    h0 (b,h,p,n). Returns y (b,l,h,p), h_final."""
    b, l, nh, p = x.shape
    g = B.shape[2]
    rep = nh // g
    Bh = B.repeat_interleave(rep, dim=2)  # (b,l,h,n)
    Ch = C.repeat_interleave(rep, dim=2)
    h = h0
    ys = []
    for t in range(l):
        dA = torch.exp(dt[:, t] * A)  # (b,h)
        h = h * dA[:, :, None, None] + (dt[:, t, :, None, None] * x[:, t, :, :, None]
                                         * Bh[:, t, :, None, :])
        y = (h * Ch[:, t, :, None, :]).sum(-1) + D[None, :, None] * x[:, t]
        ys.append(y)
    return torch.stack(ys, 1), h


def _ssd_inputs(torch, b=2, l=256, nh=8, p=64, g=1, n=64, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s: torch.randn(*s, device="cuda", generator=gen)
    x = r(b, l, nh, p)
    dt = torch.nn.functional.softplus(r(b, l, nh) - 1.0)
    A = -torch.exp(r(nh) * 0.5)
    B = r(b, l, g, n) * 0.5
    C = r(b, l, g, n) * 0.5
    D = r(nh)
    h0 = r(b, nh, p, n)
    return x, dt, A, B, C, D, h0


@check("chunk_scan_initial_states")
def check_chunk_scan():
    import torch
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    sig = str(inspect.signature(mamba_chunk_scan_combined))
    x, dt, A, B, C, D, h0 = _ssd_inputs(torch)
    h0k = h0.clone().requires_grad_(True)
    xk = x.clone().requires_grad_(True)
    y, hf = mamba_chunk_scan_combined(xk, dt, A, B, C, chunk_size=64, D=D,
                                      initial_states=h0k, return_final_states=True)
    gy = torch.randn_like(y)
    ghf = torch.randn_like(hf)
    ((y * gy).sum() + (hf * ghf).sum()).backward()
    h0r = h0.clone().requires_grad_(True)
    xr = x.clone().requires_grad_(True)
    yr, hfr = naive_ssd(torch, xr, dt, A, B, C, D, h0r)
    ((yr * gy).sum() + (hfr * ghf).sum()).backward()
    res = {
        "signature": sig,
        "y_shape": list(y.shape), "final_state_shape": list(hf.shape),
        "max_abs_diff_y_vs_naive": float((y - yr).abs().max()),
        "max_abs_diff_final_state_vs_naive": float((hf - hfr).abs().max()),
        "y_abs_max": float(yr.abs().max()),
        "grad_initial_states_is_none": h0k.grad is None,
    }
    if h0k.grad is not None:
        res.update(
            grad_initial_states_abs_mean=float(h0k.grad.abs().mean()),
            grad_initial_states_nonzero=bool(h0k.grad.abs().sum() > 0),
            grad_initial_states_finite=bool(torch.isfinite(h0k.grad).all()),
            max_abs_diff_grad_initial_states_vs_naive=float((h0k.grad - h0r.grad).abs().max()),
            grad_initial_states_ref_abs_max=float(h0r.grad.abs().max()),
            max_abs_diff_grad_x_vs_naive=float((xk.grad - xr.grad).abs().max()),
        )
    # gated hand-off: gradient must reach the gate parameters through initial_states
    gate = torch.nn.Parameter(torch.full((8, 64, 64), 4.0, device="cuda"))
    h_prev = h0.detach()
    h_init = torch.sigmoid(gate) * h_prev
    y2, _ = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size=64, D=D,
                                      initial_states=h_init, return_final_states=True)
    y2.pow(2).mean().backward()
    res["gate_grad_abs_mean"] = float(gate.grad.abs().mean())
    res["gate_grad_nonzero"] = bool(gate.grad.abs().sum() > 0)
    # dtype variants of the kernel with initial states
    dts = {}
    for label, dtp in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
        def run(dtp=dtp):
            hh = h0.clone().requires_grad_(True)
            yy, ff = mamba_chunk_scan_combined(x.to(dtp), dt.to(dtp), A, B.to(dtp), C.to(dtp),
                                               chunk_size=64, D=D, initial_states=hh,
                                               return_final_states=True)
            yy.float().pow(2).mean().backward()
            torch.cuda.synchronize()
            return {"max_abs_diff_y_vs_fp32_naive": float((yy.float() - yr.detach()).abs().max()),
                    "grad_initial_states_finite": bool(torch.isfinite(hh.grad).all()),
                    "final_state_dtype": str(ff.dtype)}
        dts[label] = sub(run)
    res["dtype_cases"] = dts
    return res


@check("chunked_vs_single_kernel")
def check_chunked_kernel():
    import torch
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    x, dt, A, B, C, D, h0 = _ssd_inputs(torch, l=512)
    zero = torch.zeros_like(h0)
    with torch.no_grad():
        y_full, h_full = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size=64, D=D,
                                                   initial_states=zero, return_final_states=True)
        y_none, _ = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size=64, D=D,
                                              return_final_states=True)
    res = {"zero_initial_states_vs_none_max_abs_diff": float((y_full - y_none).abs().max()),
           "y_abs_max": float(y_full.abs().max())}

    def run_split(bounds):
        h = None
        ys = []
        with torch.no_grad():
            for s, e in zip(bounds[:-1], bounds[1:]):
                kw = {} if h is None else {"initial_states": h}
                y, h = mamba_chunk_scan_combined(x[:, s:e], dt[:, s:e], A, B[:, s:e], C[:, s:e],
                                                 chunk_size=64, D=D, return_final_states=True, **kw)
                ys.append(y)
        y = torch.cat(ys, 1)
        return {"max_abs_diff_y": float((y - y_full).abs().max()),
                "max_abs_diff_final_state": float((h - h_full).abs().max())}

    res["two_chunks_at_256"] = sub(run_split, [0, 256, 512])
    res["two_chunks_at_200_unaligned"] = sub(run_split, [0, 200, 512])
    res["sixteen_blocks_of_32"] = sub(run_split, list(range(0, 513, 32)))
    return res


def mamba2_block_forward(torch, m, u, ssm_state=None, conv_state=None):
    """Non-fused Mamba2 forward over one block that takes and returns (ssm_state, conv_state).

    Mirrors mamba_ssm.modules.mamba2.Mamba2.forward (non-mem-efficient path) but with an
    explicit conv state (last d_conv-1 pre-conv inputs) and SSM initial state.
    """
    import torch.nn.functional as F
    from einops import rearrange
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    b, l, _ = u.shape
    d_ssm = getattr(m, "d_ssm", m.d_inner)
    zxbcdt = m.in_proj(u)
    A = -torch.exp(m.A_log.float())
    d_mlp = (zxbcdt.shape[-1] - 2 * d_ssm - 2 * m.ngroups * m.d_state - m.nheads) // 2
    z0, x0, z, xBC, dt = torch.split(
        zxbcdt, [d_mlp, d_mlp, d_ssm, d_ssm + 2 * m.ngroups * m.d_state, m.nheads], dim=-1)
    xBC_t = xBC.transpose(1, 2)  # (b, conv_dim, l)
    if conv_state is None:
        conv_state = torch.zeros(b, xBC_t.shape[1], m.d_conv - 1, device=u.device, dtype=xBC_t.dtype)
    padded = torch.cat([conv_state, xBC_t], dim=-1)
    new_conv_state = padded[..., -(m.d_conv - 1):]
    xBC = F.conv1d(padded, m.conv1d.weight, m.conv1d.bias, groups=xBC_t.shape[1]).transpose(1, 2)
    xBC = m.act(xBC)
    x, B, C = torch.split(xBC, [d_ssm, m.ngroups * m.d_state, m.ngroups * m.d_state], dim=-1)
    kw = {}
    dt_limit = getattr(m, "dt_limit", (0.0, float("inf")))
    if tuple(dt_limit) != (0.0, float("inf")):
        kw["dt_limit"] = dt_limit
    rmsnorm = getattr(m, "rmsnorm", True)
    D = rearrange(m.D, "(h p) -> h p", p=m.headdim) if getattr(m, "D_has_hdim", False) else m.D
    if ssm_state is not None:
        kw["initial_states"] = ssm_state
    y, new_ssm = mamba_chunk_scan_combined(
        rearrange(x, "b l (h p) -> b l h p", p=m.headdim), dt, A,
        rearrange(B, "b l (g n) -> b l g n", g=m.ngroups),
        rearrange(C, "b l (g n) -> b l g n", g=m.ngroups),
        chunk_size=m.chunk_size, D=D,
        z=None if rmsnorm else rearrange(z, "b l (h p) -> b l h p", p=m.headdim),
        dt_bias=m.dt_bias, dt_softplus=True, return_final_states=True, **kw)
    y = rearrange(y, "b l h p -> b l (h p)")
    if rmsnorm:
        y = m.norm(y, z)
    if d_mlp > 0:
        y = torch.cat([F.silu(z0) * x0, y], dim=-1)
    return m.out_proj(y), new_ssm, new_conv_state


@check("chunked_vs_single_mamba2_module")
def check_chunked_module():
    import torch
    from mamba_ssm import Mamba2
    torch.manual_seed(0)
    m = Mamba2(d_model=D_MODEL, d_state=64, d_conv=4, expand=2, headdim=64).cuda()
    u = torch.randn(4, 512, D_MODEL, device="cuda")
    with torch.no_grad():
        y_full = m(u)
        y_manual, _, _ = mamba2_block_forward(torch, m, u)
    res = {"manual_full_vs_module_max_abs_diff": float((y_manual - y_full).abs().max()),
           "y_abs_max": float(y_full.abs().max())}

    def blocks(bs, carry_conv=True):
        s_ssm = s_conv = None
        ys = []
        with torch.no_grad():
            for s in range(0, u.shape[1], bs):
                y, s_ssm, c = mamba2_block_forward(torch, m, u[:, s:s + bs], s_ssm,
                                                   s_conv if carry_conv else None)
                s_conv = c
                ys.append(y)
        return {"max_abs_diff_vs_module_full": float((torch.cat(ys, 1) - y_full).abs().max())}

    res["two_chunks_256_ssm_and_conv_state"] = sub(blocks, 256)
    res["blocks_of_32_ssm_and_conv_state"] = sub(blocks, 32)
    res["blocks_of_32_ssm_state_only_no_conv_state"] = sub(blocks, 32, False)

    # gradient through a gated state hand-off between two blocks
    def grad_case():
        gate_w = torch.nn.Parameter(torch.full((m.nheads, m.headdim, m.d_state), 4.0, device="cuda"))
        u1 = u.clone().requires_grad_(True)
        y1, h1, c1 = mamba2_block_forward(torch, m, u1[:, :32])
        y2, _, _ = mamba2_block_forward(torch, m, u1[:, 32:64], torch.sigmoid(gate_w) * h1, c1)
        y2.pow(2).mean().backward()
        g = u1.grad[:, :32]
        return {"grad_block2_wrt_block1_input_abs_mean": float(g.abs().mean()),
                "grad_nonzero": bool(g.abs().sum() > 0),
                "gate_grad_abs_mean": float(gate_w.grad.abs().mean()),
                "gate_shape": list(gate_w.shape),
                "gate_params": int(gate_w.numel())}
    res["gated_handoff_gradient"] = sub(grad_case)
    return res


@check("mamba1_initial_state_api")
def check_mamba1_api():
    import mamba_ssm
    from mamba_ssm import Mamba
    from mamba_ssm.ops import selective_scan_interface as ssi
    res = {"mamba_ssm_file": mamba_ssm.__file__, "signatures": {}}
    for name in ["selective_scan_fn", "selective_scan_ref", "mamba_inner_fn", "mamba_inner_ref"]:
        fn = getattr(ssi, name, None)
        res["signatures"][name] = str(inspect.signature(fn)) if fn is not None else None
    res["signatures"]["Mamba.forward"] = str(inspect.signature(Mamba.forward))
    res["signatures"]["Mamba.step"] = str(inspect.signature(Mamba.step)) if hasattr(Mamba, "step") else None
    try:
        from mamba_ssm.ops.triton.selective_state_update import selective_state_update
        res["signatures"]["selective_state_update"] = str(inspect.signature(selective_state_update))
    except Exception as e:
        res["signatures"]["selective_state_update"] = repr(e)
    try:
        import selective_scan_cuda
        res["selective_scan_cuda_fwd_doc"] = (selective_scan_cuda.fwd.__doc__ or "")[:2000]
        res["selective_scan_cuda_bwd_doc"] = (selective_scan_cuda.bwd.__doc__ or "")[:2000]
    except Exception as e:
        res["selective_scan_cuda"] = repr(e)
    src = inspect.getsource(ssi)
    pat = re.compile(r"initial_state|init_state|prev_state|h0|ssm_state|last_state", re.I)
    res["selective_scan_interface_matches"] = [
        f"{i + 1}: {ln.strip()}" for i, ln in enumerate(src.splitlines()) if pat.search(ln)][:60]
    msrc = inspect.getsource(Mamba)
    res["mamba_simple_matches"] = [
        f"{i + 1}: {ln.strip()}" for i, ln in enumerate(msrc.splitlines()) if pat.search(ln)][:60]
    fwd_params = set(inspect.signature(ssi.selective_scan_fn).parameters)
    res["selective_scan_fn_accepts_initial_state"] = any(
        p in fwd_params for p in ("initial_state", "initial_states", "h0", "prev_state", "ssm_state"))
    res["selective_scan_fn_returns_last_state"] = "return_last_state" in fwd_params
    res["mamba_forward_params"] = list(inspect.signature(Mamba.forward).parameters)
    return res


# ---------------------------------------------------------------------------------
@check("throughput")
def check_throughput(quick=False):
    import torch
    from mamba_ssm import Mamba, Mamba2
    tokens = 32768
    seqs = [256, 1024, 4096]
    iters, warm = (3, 1) if quick else (10, 3)
    cap = torch.cuda.get_device_capability(0)
    dts = [("fp32", None), ("fp16_autocast", torch.float16)]
    if cap[0] >= 8:
        dts.append(("bf16_autocast", torch.bfloat16))
    layers = {
        "mamba1": lambda: Mamba(d_model=D_MODEL, d_state=16, d_conv=4, expand=2),
        "mamba2": lambda: Mamba2(d_model=D_MODEL, d_state=64, d_conv=4, expand=2, headdim=64),
    }
    rows = []
    for lname, ctor in layers.items():
        torch.manual_seed(0)
        layer = ctor().cuda()
        for L in seqs:
            B = max(1, tokens // L)
            for dname, dt in dts:
                def bench(mode):
                    x = torch.randn(B, L, D_MODEL, device="cuda", requires_grad=(mode == "fwd_bwd"))
                    ctx = (lambda: torch.autocast("cuda", dtype=dt)) if dt else (lambda: torch.autocast("cuda", enabled=False))

                    def step():
                        if mode == "fwd":
                            with torch.no_grad(), ctx():
                                layer(x)
                        else:
                            with ctx():
                                y = layer(x)
                            y.float().pow(2).mean().backward()
                    for _ in range(warm):
                        step()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    t0 = time.time()
                    for _ in range(iters):
                        step()
                    torch.cuda.synchronize()
                    el = (time.time() - t0) / iters
                    return {"ms_per_iter": round(el * 1e3, 2), "tokens_per_s": round(B * L / el),
                            "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)}
                for mode in ("fwd", "fwd_bwd"):
                    r = sub(bench, mode)
                    rows.append({"layer": lname, "seq": L, "batch": B, "dtype": dname, "mode": mode, **r})
                    print(json.dumps({k: v for k, v in rows[-1].items() if k != "traceback"}), flush=True)
                    torch.cuda.empty_cache()
        del layer
    return {"d_model": D_MODEL, "tokens_per_batch": tokens, "iters": iters, "rows": rows}


def main():
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("ABR_OUTPUT_DIR", "compat_out"))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    OUT_PATH = os.path.join(a.out, "compat_check.json")
    RESULTS["code_commit"] = os.environ.get("CODE_COMMIT")
    RESULTS["kaggle_kernel_url"] = os.environ.get("KAGGLE_KERNEL_URL")
    save()
    env = check_env()
    import torch
    if not torch.cuda.is_available():
        RESULTS["fatal"] = "CUDA not available; skipping GPU checks"
        save()
        print(RESULTS["fatal"])
        return 1
    check_mamba1()
    check_mamba1_ref()
    check_mamba2()
    check_chunk_scan()
    check_chunked_kernel()
    check_chunked_module()
    check_mamba1_api()
    check_throughput(quick=a.quick)
    RESULTS["finished_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    RESULTS["summary"] = {k: v.get("ok") for k, v in RESULTS["checks"].items()}
    save()
    print(json.dumps(RESULTS["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
