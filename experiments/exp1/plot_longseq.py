"""
Plots the latency phase diagram from benchmark_longseq.py results.

Run after benchmark_longseq.py:
    python experiments/exp1/plot_longseq.py
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

results_path = os.path.join(os.path.dirname(__file__), "longseq_results.json")
with open(results_path) as f:
    raw = json.load(f)

results = {int(k): v for k, v in raw.items()}
seq_lens = sorted(results)

all_keys = sorted({k for v in results.values() for k in v})

# --- Line plot: latency vs sequence length per model ---
fig, ax = plt.subplots(figsize=(11, 6))
cmap = plt.cm.tab10(np.linspace(0, 1, len(all_keys)))

for key, color in zip(all_keys, cmap):
    lats_ms = []
    xs = []
    for s in seq_lens:
        if key in results[s]:
            xs.append(s)
            lats_ms.append(results[s][key] * 1000)
    if xs:
        ax.plot(xs, lats_ms, marker="o", label=key, color=color)

ax.set_xlabel("Sequence Length", fontsize=12)
ax.set_ylabel("Latency (ms)", fontsize=12)
ax.set_title("Latency Phase Diagram: Sequence Length × Model", fontsize=13)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, which="both", alpha=0.3)

line_path = os.path.join(os.path.dirname(__file__), "longseq_phase_diagram.png")
fig.savefig(line_path, dpi=150, bbox_inches="tight")
print(f"Phase diagram saved -> {line_path}")

# --- Heatmap: which model is fastest at each (seq_len, block_size) ---
block_keys = [k for k in all_keys if k.startswith("Hybrid")]
block_sizes = sorted({int(k.split("b=")[1].split(",")[0]) for k in block_keys})

fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax2, use_state in zip(axes, [False, True]):
    label_suffix = "state=on" if use_state else "state=off"
    data = np.full((len(block_sizes), len(seq_lens)), np.nan)

    for bi, bs in enumerate(block_sizes):
        key = f"Hybrid(b={bs},{label_suffix})"
        for si, s in enumerate(seq_lens):
            if key in results.get(s, {}):
                baseline = results[s].get("Mamba", np.nan)
                hybrid   = results[s][key]
                data[bi, si] = hybrid / baseline   # ratio vs Mamba

    im = ax2.imshow(data, aspect="auto", cmap="RdYlGn_r", vmin=0.5, vmax=2.0)
    ax2.set_xticks(range(len(seq_lens)))
    ax2.set_xticklabels(seq_lens)
    ax2.set_yticks(range(len(block_sizes)))
    ax2.set_yticklabels(block_sizes)
    ax2.set_xlabel("Sequence Length")
    ax2.set_ylabel("Block Size")
    ax2.set_title(f"Hybrid / Mamba latency ratio ({label_suffix})\n<1 = Hybrid faster")
    plt.colorbar(im, ax=ax2)

heatmap_path = os.path.join(os.path.dirname(__file__), "longseq_heatmap.png")
fig2.savefig(heatmap_path, dpi=150, bbox_inches="tight")
print(f"Heatmap saved -> {heatmap_path}")
