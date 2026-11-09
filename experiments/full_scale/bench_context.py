"""
Throughput / memory benchmark for any config (PRD P2.5, fixes B7).

For every config and sequence length: tokens/sec and peak memory, for
forward-only (eval, no_grad) and for a full training step (fwd + bwd + AdamW).
Randomly initialised weights are enough: speed and memory do not depend on
the weight values, only on the routing fraction. For learned-routing ABR
models, pass the routing fraction the TRAINED model actually produces; the
model is then benchmarked with routing_mode="random" at that attention
fraction, and the fraction is reported next to the numbers.

    python bench_context.py                                    # all PHASE2_CONFIGS
    python bench_context.py --configs Transformer-Only ABR-4L-CBSC \
        --routing-fractions ABR-4L-CBSC=0.31
    python bench_context.py --routing-fractions-from results/phase2   # mean of final
                                                                     # transformer_pct per label

All Mamba-containing models must be benchmarked on the same Kaggle GPU as the
Transformer (PRD §0.3); local timings are for development only.
Output: <out>/<run_id>/result.json via provenance (rows under results["rows"]).
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C                        # noqa: E402
from train import (_ms, _prov, resolve_device, seed_everything,   # noqa: E402
                   _utcnow)

VOCAB = 50258   # GPT-2 + pad, same as training


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _reset_peak(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_gb(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    if device.type == "mps":   # no peak counter on MPS; report the current driver allocation
        return torch.mps.driver_allocated_memory() / 1e9
    return None


def _oom_types():
    t = [torch.cuda.OutOfMemoryError] if hasattr(torch.cuda, "OutOfMemoryError") else []
    return tuple(t) + (RuntimeError,)


def bench_one(model, seq_len, batch_size, device, warmup, iters, train_mode):
    x = torch.randint(0, VOCAB, (batch_size, seq_len), device=device)
    if train_mode:
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

        def step():
            logits = model(x)
            F.cross_entropy(logits.reshape(-1, logits.size(-1)), x.reshape(-1)).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
    else:
        model.eval()

        def step():
            with torch.no_grad():
                model(x)
    for _ in range(warmup):
        step()
    _sync(device)
    _reset_peak(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    _sync(device)
    dt = time.perf_counter() - t0
    return {"tokens_per_s": batch_size * seq_len * iters / dt,
            "latency_ms": dt / iters * 1000, "peak_mem_gb": _peak_gb(device)}


def parse_fractions(spec):
    out = {}
    for item in spec or []:
        for part in item.split(","):
            if part.strip():
                k, v = part.rsplit("=", 1)
                out[k.strip()] = float(v)
    return out


def fractions_from_results(out_dir):
    import stats
    acc = {}
    for r in stats.load_results(out_dir):
        rs = r.get("final_routing_stats") or {}
        if "transformer_pct" in rs:
            acc.setdefault(r["label"], []).append(float(rs["transformer_pct"]))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("--configs", nargs="+", default=None)
    ap.add_argument("--lengths", nargs="+", type=int, default=C.CONTEXT_LENGTHS)
    ap.add_argument("--batch-size", type=int, default=C.BENCH_BATCH_SIZE)
    ap.add_argument("--block-size", type=int, default=C.BLOCK_SIZE)
    ap.add_argument("--warmup", type=int, default=C.BENCH_WARMUP_ITERS)
    ap.add_argument("--iters", type=int, default=C.BENCH_ITERS)
    ap.add_argument("--modes", nargs="+", default=["forward", "train"],
                    choices=["forward", "train"])
    ap.add_argument("--routing-fractions", nargs="*", default=None,
                    help="LABEL=attn_frac pairs (comma or space separated)")
    ap.add_argument("--routing-fractions-from", default=None,
                    help="results dir: use each label's mean final transformer_pct")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--allow-gru-fallback", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--out", default="results/phase2_bench")
    args = ap.parse_args(argv)

    ms, prov = _ms(), _prov()
    device = resolve_device(args.device)
    labels = args.configs or [c["label"] for c in C.PHASE2_CONFIGS]
    cfgs = [C.get_config(lbl) for lbl in labels]
    fracs = fractions_from_results(args.routing_fractions_from) \
        if args.routing_fractions_from else {}
    fracs.update(parse_fractions(args.routing_fractions))

    backend = None
    if any(C.contains_mamba(c) for c in cfgs):
        ms.require_real_mamba(allow_fallback=args.allow_gru_fallback)
        backend = ms.MAMBA_BACKEND
    bench_cfg = {"label": "bench_context", "seed": 0, "configs": labels,
                 "lengths": args.lengths, "batch_size": args.batch_size,
                 "block_size": args.block_size, "warmup": args.warmup, "iters": args.iters,
                 "modes": args.modes, "routing_fractions": fracs}
    meta = prov.start_run(bench_cfg, debug=args.debug, extra={"mamba_backend": backend})
    if backend == "gru_fallback":
        meta["reportable"] = False
    out = os.path.abspath(args.out)
    log_path = os.path.join(out, meta["run_id"], "train.log")
    rows = []
    with prov.RunLogger(log_path):
        print(f"===== bench start {_utcnow()} device={device} backend={backend} =====")
        for cfg in cfgs:
            frac_used = None
            if cfg["expert_type"] == "abr" and cfg["routing_mode"] == "learned":
                if cfg["label"] in fracs:
                    frac_used = fracs[cfg["label"]]
                    cfg = {**cfg, "routing_mode": "random", "attn_frac": frac_used}
                else:
                    print(f"  WARNING: no trained routing fraction for {cfg['label']}; "
                          f"benchmarking the untrained learned router")
            elif cfg["expert_type"] == "abr":
                frac_used = cfg["attn_frac"]
            for T in args.lengths:
                seed_everything(0)
                model = ms.build_model(cfg, VOCAB, d_model=C.D_MODEL, n_heads=C.N_HEADS,
                                       block_size=args.block_size, ffn_mult=C.FFN_MULT,
                                       max_seq_len=T + 16).to(device)
                for mode in args.modes:
                    row = {"config": cfg["label"], "seq_len": T, "mode": mode,
                           "batch_size": args.batch_size, "params": ms.count_params(model),
                           "attn_frac_used": frac_used, "device": str(device)}
                    try:
                        row.update(bench_one(model, T, args.batch_size, device,
                                             args.warmup, args.iters, mode == "train"))
                        row["status"] = "ok"
                    except _oom_types() as e:
                        if "out of memory" not in str(e).lower() and \
                                not isinstance(e, getattr(torch.cuda, "OutOfMemoryError", ())):
                            raise
                        row["status"] = "oom"
                    rows.append(row)
                    print(f"  {cfg['label']:<22} T={T:<5} {mode:<7} {row['status']:<4} "
                          f"tok/s={row.get('tokens_per_s', 0):>10,.0f} "
                          f"mem={row.get('peak_mem_gb') or 0:.2f}GB", flush=True)
                del model
                _reset_peak(device)
    path = prov.finish_run(meta, {"rows": rows, "mamba_backend": backend,
                                  "gpu_name": torch.cuda.get_device_name(0)
                                  if device.type == "cuda" else str(device)},
                           out, log_path)
    print(f"[bench] -> {path}")
    return rows


if __name__ == "__main__":
    main()
