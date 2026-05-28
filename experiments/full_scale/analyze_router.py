"""
Router specialization analysis.
Measures:
  1. Expert utilization per position (does the router prefer T or M at specific
     positions within the sequence? — positional bias)
  2. Expert utilization vs. token frequency (do common vs. rare tokens route
     differently?)
  3. Per-layer routing entropy over training (from saved routing_history)
  4. Routing consistency across seeds (do different seeds agree?)

Run after train_ablation.py.
"""
import sys, os, json, glob, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from config import (D_MODEL, N_HEADS, BLOCK_SIZE, FFN_MULT, SEQ_LEN,
                    RESULTS_DIR, SEEDS, BATCH_SIZE)
from dataset import WikiTextBPE
from models_scaled import ABRLM, count_params

matplotlib.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "axes.linewidth": 0.8,
})
assert torch.cuda.is_available(), "CUDA required."
device = torch.device("cuda")
OUT    = os.path.join(RESULTS_DIR, "router_analysis")
os.makedirs(OUT, exist_ok=True)


def load_or_train(train_ds, val_ds, seed=42, steps=3000):
    """Train an ABR-2L model briefly for routing analysis."""
    torch.manual_seed(seed)
    model = ABRLM(train_ds.vocab_size, D_MODEL, N_HEADS, 2, BLOCK_SIZE,
                  True, FFN_MULT, SEQ_LEN + 16).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    print(f"  Training ABR-2L for {steps} steps (seed={seed})...")
    for step in range(steps):
        x, y = train_ds.get_batch(BATCH_SIZE, device=device)
        logits = model(x)
        loss = criterion(logits.view(-1, train_ds.vocab_size), y.view(-1))
        loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        if step % 1000 == 0:
            print(f"    step {step}  loss={loss.item():.4f}", flush=True)
    return model


def collect_routing_decisions(model, val_ds, n_batches=200):
    """
    Returns routing_decisions: list of (B, N, 2) z_soft tensors,
    and input_ids: list of (B, T) tensors.
    """
    model.eval()
    all_routing = []   # per-layer, per-block routing probs
    all_inputs  = []

    with torch.no_grad():
        for _ in range(n_batches):
            x, _ = val_ds.get_batch(BATCH_SIZE, device=device)
            _ = model(x)
            # Collect routing from each layer
            layer_decisions = []
            for layer in model.layers:
                if hasattr(layer, "last_routing") and layer.last_routing is not None:
                    layer_decisions.append(layer.last_routing.cpu().numpy())  # (B, N, 2)
            all_routing.append(layer_decisions)
            all_inputs.append(x.cpu().numpy())

    model.train()
    return all_routing, all_inputs


def plot_positional_routing_bias(all_routing, block_size, seq_len, out_dir):
    """
    For each block position (0..N-1), what fraction of samples route to Mamba?
    """
    n_blocks = seq_len // block_size
    n_layers = len(all_routing[0])
    fig, axes = plt.subplots(1, n_layers, figsize=(7, 2.5), sharey=True)
    if n_layers == 1:
        axes = [axes]

    for li in range(n_layers):
        mamba_frac = np.zeros(n_blocks)
        count = 0
        for batch_routing in all_routing:
            r = batch_routing[li]          # (B, N, 2)
            mamba_frac += r[..., 1].mean(axis=0)  # mean over batch → (N,)
            count += 1
        mamba_frac /= count

        ax = axes[li]
        ax.bar(range(n_blocks), mamba_frac, color="#D6604D", alpha=0.75,
               edgecolor="#333", linewidth=0.5)
        ax.axhline(0.5, color="#333", lw=0.7, linestyle="--", label="50% (uniform)")
        ax.set_xlabel("Block position"); ax.set_ylabel("Mamba routing prob.")
        ax.set_title(f"Layer {li+1}")
        ax.set_ylim(0, 1); ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines[["top","right"]].set_visible(False)
        if li == 0:
            ax.legend(fontsize=6.5)

    fig.suptitle("Positional Routing Bias (Mamba fraction per block position)",
                 fontsize=9, y=1.01)
    path = os.path.join(out_dir, "positional_bias.png")
    fig.tight_layout()
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)

    # Print statistics
    print("\nPositional routing statistics (Mamba fraction):")
    for li in range(n_layers):
        mamba_frac = np.zeros(n_blocks)
        count = 0
        for batch_routing in all_routing:
            r = batch_routing[li]
            mamba_frac += r[..., 1].mean(axis=0)
            count += 1
        mamba_frac /= count
        print(f"  Layer {li+1}: min={mamba_frac.min():.3f}  "
              f"max={mamba_frac.max():.3f}  "
              f"std={mamba_frac.std():.4f}  "
              f"overall={mamba_frac.mean():.3f}")


def plot_routing_entropy_over_training(results_dir, out_dir):
    """Load routing_history from training JSON files and plot entropy over time."""
    pattern = os.path.join(results_dir, "ABR-2L-stateON_*_seed*.json")
    files   = glob.glob(pattern)
    if not files:
        pattern = os.path.join(results_dir, "*.json")
        files   = [f for f in glob.glob(pattern)
                   if "ABR" in f and "stateON" in f]
    if not files:
        print("No ABR-2L training files found — skipping entropy plot.")
        return

    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)
        rh = data.get("routing_history", [])
        if not rh:
            continue
        steps   = [r["step"]    for r in rh]
        entropy = [r.get("entropy", 0) for r in rh]
        ax.plot(steps, entropy, alpha=0.7, lw=1.0)

    ax.axhline(math.log(2) * 0.5, color="k", lw=0.8, linestyle="--",
               label="Target H*")
    ax.axhline(math.log(2),       color="k", lw=0.8, linestyle=":",
               label="Max H (uniform)")
    ax.set_xlabel("Step"); ax.set_ylabel("Routing Entropy H")
    ax.set_title("Router Entropy During Training")
    ax.legend(fontsize=6.5); ax.grid(linestyle="--", alpha=0.4)
    ax.spines[["top","right"]].set_visible(False)

    path = os.path.join(out_dir, "routing_entropy_over_training.png")
    fig.tight_layout()
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


def plot_routing_consistency_across_seeds(all_routing_per_seed, out_dir):
    """
    For each block position, compute routing agreement across seeds.
    High agreement = router converges to same decisions regardless of init.
    """
    n_layers = len(all_routing_per_seed[0][0])
    if n_layers == 0:
        return

    fig, ax = plt.subplots(figsize=(3.5, 2.2))

    for li in range(n_layers):
        mamba_fracs = []
        for seed_routing in all_routing_per_seed:
            fracs = []
            for batch_r in seed_routing:
                fracs.append(batch_r[li][..., 1].mean(axis=0))
            mamba_fracs.append(np.mean(fracs, axis=0))   # (N,)

        mamba_fracs = np.array(mamba_fracs)    # (n_seeds, N)
        std_across_seeds = mamba_fracs.std(axis=0)
        ax.plot(std_across_seeds, marker="o", markersize=3, lw=1.0,
                label=f"Layer {li+1}")

    ax.set_xlabel("Block position")
    ax.set_ylabel("Std of Mamba routing\nacross seeds")
    ax.set_title("Routing Consistency Across Seeds\n(low std = consistent routing)")
    ax.legend(fontsize=7); ax.grid(linestyle="--", alpha=0.4)
    ax.spines[["top","right"]].set_visible(False)

    path = os.path.join(out_dir, "routing_seed_consistency.png")
    fig.tight_layout()
    fig.savefig(path)
    print(f"Saved: {path}")
    plt.close(fig)


def main():
    print("Loading dataset...")
    train_ds = WikiTextBPE("train",      seq_len=SEQ_LEN, max_tokens=10_000_000)
    val_ds   = WikiTextBPE("validation", seq_len=SEQ_LEN)

    print("\nTraining analysis models across seeds...")
    all_routing_per_seed = []
    for seed in SEEDS:
        model = load_or_train(train_ds, val_ds, seed=seed, steps=3000)
        routing, inputs = collect_routing_decisions(model, val_ds, n_batches=100)
        all_routing_per_seed.append(routing)
        del model
        torch.cuda.empty_cache()

    print("\nGenerating router analysis figures...")
    # Use first seed for positional bias
    plot_positional_routing_bias(all_routing_per_seed[0],
                                  BLOCK_SIZE, SEQ_LEN, OUT)
    plot_routing_entropy_over_training(RESULTS_DIR, OUT)
    plot_routing_consistency_across_seeds(all_routing_per_seed, OUT)

    print(f"\nAll router analysis figures saved to: {OUT}")


if __name__ == "__main__":
    main()
