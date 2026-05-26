"""
Long-sequence latency benchmark — phase diagram across (seq_len, block_size).

Extends the main benchmark to show where each model wins:
  - Transformer: O(n^2) attention, good at short sequences
  - Mamba:       O(n) recurrence, good at long sequences
  - Sparse Hybrid: block-level routing at varying granularities

Run:
    python -m experiments.exp1.benchmark_longseq
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import time

import torch

from models.hybrid_model import DynamicHybridModel
from models.transformer_only import TransformerOnly
from models.mamba_only import MambaOnly
from models.soft_hybrid import SoftHybridModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

SEQ_LENS   = [64, 128, 256, 512, 1024]
BLOCK_SIZES = [4, 8, 16, 32]
BATCH_SIZE  = 4
D_MODEL     = 128
WARMUP      = 5
REPEATS     = 20


def benchmark_model(model, x, warmup=WARMUP, repeats=REPEATS):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats


results = {}

for seq_len in SEQ_LENS:
    x = torch.randn(BATCH_SIZE, seq_len, D_MODEL, device=device)
    results[seq_len] = {}

    baselines = [
        ("Transformer",  TransformerOnly(d_model=D_MODEL)),
        ("Mamba",        MambaOnly(d_model=D_MODEL)),
        ("Soft Hybrid",  SoftHybridModel(d_model=D_MODEL)),
    ]
    for name, model in baselines:
        model = model.to(device)
        lat = benchmark_model(model, x)
        results[seq_len][name] = lat
        print(f"seq={seq_len:5d} | {name:<20s} | {lat * 1000:.2f} ms")

    for bs in BLOCK_SIZES:
        if seq_len % bs != 0:
            continue
        for use_state in (False, True):
            model = DynamicHybridModel(
                d_model=D_MODEL, block_size=bs, use_cross_block_state=use_state
            ).to(device)
            lat = benchmark_model(model, x)
            label = f"Hybrid(b={bs},state={'on' if use_state else 'off'})"
            results[seq_len][label] = lat
            print(f"seq={seq_len:5d} | {label:<30s} | {lat * 1000:.2f} ms")

    print()

out_path = os.path.join(os.path.dirname(__file__), "longseq_results.json")
with open(out_path, "w") as f:
    json.dump({str(k): v for k, v in results.items()}, f, indent=2)
print(f"Results saved -> {out_path}")
