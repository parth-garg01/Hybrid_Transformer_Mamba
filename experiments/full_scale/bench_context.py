"""
Context-length benchmark: PPL + tokens/sec + peak memory + latency
for Transformer, Mamba, ABR at context lengths [128, 256, 512, 1024].

Loads the best checkpoint from train_ablation.py results, or trains fresh
for N_QUICK_STEPS if no checkpoint exists.

Run after train_ablation.py, or standalone with --quick flag.
"""
import sys, os, json, time, math, csv, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import numpy as np

from config import (D_MODEL, N_HEADS, BLOCK_SIZE, FFN_MULT,
                    CONTEXT_LENGTHS, BENCH_BATCH_SIZE,
                    BENCH_WARMUP_ITERS, BENCH_ITERS, RESULTS_DIR,
                    BATCH_SIZE, LR, SEEDS)
from dataset import WikiTextBPE
from models_scaled import TransformerLM, MambaLM, ABRLM, count_params

assert torch.cuda.is_available(), "CUDA required."
device = torch.device("cuda")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_QUICK_STEPS = 2000    # steps for quick training if no checkpoint exists
N_EVAL_STEPS  = 100

BENCH_MODELS = {
    "Transformer": None,
    "Mamba":       None,
    "ABR-2L":      None,
}


def quick_train(model, train_ds, seq_len, steps=N_QUICK_STEPS):
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for step in range(steps):
        x, y = train_ds.get_batch(BATCH_SIZE, device=device)
        logits = model(x)
        loss = criterion(logits.view(-1, train_ds.vocab_size), y.view(-1))
        loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        if step % 500 == 0:
            print(f"  quick-train step {step}/{steps}  loss={loss.item():.4f}", flush=True)
    return model


def evaluate_ppl(model, val_ds, criterion, n_steps):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(n_steps):
            x, y = val_ds.get_batch(BENCH_BATCH_SIZE, device=device)
            logits = model(x)
            total += criterion(logits.view(-1, val_ds.vocab_size), y.view(-1)).item()
    model.train()
    return total / n_steps


def measure_efficiency(model, seq_len):
    """Returns (tokens_per_sec, peak_mem_gb, latency_ms)."""
    model.eval()
    x = torch.randint(0, 1000, (BENCH_BATCH_SIZE, seq_len), device=device)
    criterion = nn.CrossEntropyLoss()

    # Warm-up
    with torch.no_grad():
        for _ in range(BENCH_WARMUP_ITERS):
            _ = model(x)
    torch.cuda.synchronize()

    # Throughput + latency
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(BENCH_ITERS):
            _ = model(x)
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_tokens    = BENCH_BATCH_SIZE * seq_len * BENCH_ITERS
    tokens_per_sec  = total_tokens / elapsed
    latency_ms      = elapsed / BENCH_ITERS * 1000
    peak_mem_gb     = torch.cuda.max_memory_allocated() / 1e9

    model.train()
    return tokens_per_sec, peak_mem_gb, latency_ms


def estimate_flops(model_name, d_model, n_heads, seq_len, n_layers):
    """Analytical FLOPs estimate per forward pass (1 sample)."""
    # Transformer: attention O(T^2 * d) + FFN O(T * d * 4d)
    attn_flops = 2 * seq_len ** 2 * d_model
    ffn_flops  = 2 * seq_len * d_model * 4 * d_model
    if model_name == "Transformer":
        return n_layers * (attn_flops + ffn_flops)
    elif model_name == "Mamba":
        # Mamba SSM: O(T * d^2) — no quadratic term
        mamba_flops = 2 * seq_len * d_model * d_model * 2  # expand=2
        return n_layers * mamba_flops
    else:  # ABR-2L: ~50% each
        return n_layers * (0.5 * (attn_flops + ffn_flops) +
                           0.5 * 2 * seq_len * d_model * d_model * 2)


def main():
    rows = []
    criterion = nn.CrossEntropyLoss()

    for seq_len in CONTEXT_LENGTHS:
        print(f"\n{'='*55}")
        print(f"Context length: {seq_len}")
        print(f"{'='*55}")

        train_ds = WikiTextBPE("train",      seq_len=seq_len, max_tokens=10_000_000)
        val_ds   = WikiTextBPE("validation", seq_len=seq_len)
        V = train_ds.vocab_size

        models_to_bench = {
            "Transformer": TransformerLM(V, D_MODEL, N_HEADS, 4, FFN_MULT, seq_len + 16),
            "Mamba":       MambaLM(V, D_MODEL, 4, seq_len + 16),
            "ABR-2L":      ABRLM(V, D_MODEL, N_HEADS, 2, BLOCK_SIZE, True, FFN_MULT, seq_len + 16),
        }

        for mname, model in models_to_bench.items():
            model = model.to(device)
            print(f"\n  Training {mname} (context={seq_len})...")
            model = quick_train(model, train_ds, seq_len)

            # Evaluate PPL
            ppl_vals = []
            for seed in SEEDS:
                torch.manual_seed(seed)
                ppl_vals.append(math.exp(min(
                    evaluate_ppl(model, val_ds, criterion, N_EVAL_STEPS), 20)))
            ppl_mean = float(np.mean(ppl_vals))
            ppl_std  = float(np.std(ppl_vals))

            # Efficiency
            tps, mem, lat = measure_efficiency(model, seq_len)

            # FLOPs
            n_layers = 4 if mname in ("Transformer", "Mamba") else 2
            gflops = estimate_flops(mname, D_MODEL, N_HEADS, seq_len, n_layers) / 1e9

            row = {
                "Context":     seq_len,
                "Model":       mname,
                "Params":      count_params(model),
                "PPL_mean":    round(ppl_mean, 2),
                "PPL_std":     round(ppl_std,  2),
                "Tokens_per_s":round(tps,      0),
                "Peak_mem_GB": round(mem,       3),
                "Latency_ms":  round(lat,       2),
                "GFLOPs":      round(gflops,    3),
            }
            rows.append(row)

            print(f"  {mname:<15} PPL={ppl_mean:.2f}±{ppl_std:.2f} | "
                  f"tps={tps:,.0f} | mem={mem:.2f}GB | "
                  f"lat={lat:.1f}ms | GFLOPs={gflops:.3f}")

            del model
            torch.cuda.empty_cache()

    # ── Save ─────────────────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "context_length_benchmark.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    json_path = csv_path.replace(".csv", ".json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"\nSaved -> {csv_path}")

    # ── Print formatted table ─────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print("CONTEXT LENGTH BENCHMARK")
    print(f"{'Model':<15} {'Ctx':>5} {'PPL':>10} {'tok/s':>10} "
          f"{'Mem(GB)':>8} {'Lat(ms)':>9} {'GFLOPs':>8}")
    print("-" * 75)
    for r in rows:
        print(f"{r['Model']:<15} {r['Context']:>5} "
              f"{r['PPL_mean']:>7.2f}±{r['PPL_std']:<4.2f} "
              f"{r['Tokens_per_s']:>10,.0f} "
              f"{r['Peak_mem_GB']:>8.3f} "
              f"{r['Latency_ms']:>9.1f} "
              f"{r['GFLOPs']:>8.3f}")


if __name__ == "__main__":
    main()
