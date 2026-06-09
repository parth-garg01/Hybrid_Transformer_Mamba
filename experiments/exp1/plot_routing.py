"""
Routing heatmap visualisation.

Shows which blocks each layer routes to Transformer vs Mamba for a batch
of real WikiText-103 sequences. This is the key interpretability figure
the paper needs — it answers 'does the router learn anything meaningful?'

Run:
    python experiments/exp1/plot_routing.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from models.multi_layer_hybrid import MultiLayerHybridModel
from models.lm_wrapper         import LMWrapper
from training.lm_dataset       import WikiTextDataset

device = torch.device("cpu")

print("Loading data...", flush=True)
train_ds = WikiTextDataset(split="train", seq_len=128, max_chars=500_000)
val_ds   = WikiTextDataset(split="validation", seq_len=128,
                           stoi=train_ds.stoi, itos=train_ds.itos,
                           max_chars=None)
vocab_size = train_ds.vocab_size

# ── Train a small model briefly so routing is non-random ────────────────────
import torch.nn as nn, torch.optim as optim
from models.losses import load_balancing_loss, entropy_loss

base  = MultiLayerHybridModel(d_model=128, n_heads=4, block_size=8, num_layers=2)
model = LMWrapper(base, vocab_size).to(device)
opt   = optim.Adam(model.parameters(), lr=1e-3)
crit  = nn.CrossEntropyLoss()

print("Quick training (100 steps) to get non-random routing...", flush=True)
model.train()
for step in range(100):
    x, y = train_ds.get_batch(batch_size=32)
    out  = model(x)
    logits = out[0] if isinstance(out, tuple) else out
    ce  = crit(logits.view(-1, vocab_size), y.view(-1))
    aux = torch.tensor(0.0)
    if hasattr(model.model, "last_routing") and model.model.last_routing is not None:
        z   = model.model.last_routing
        aux = 0.01 * load_balancing_loss(z) - 0.001 * entropy_loss(z)
    (ce + aux).backward()
    opt.step(); opt.zero_grad()

# ── Collect routing decisions for a batch ────────────────────────────────────
model.eval()
BATCH = 8
x, _ = val_ds.get_batch(batch_size=BATCH)

routing_per_layer = []   # list of (B, num_blocks) integer tensors
with torch.no_grad():
    _ = model(x)
    for layer in base.layers:
        if hasattr(layer, "last_routing"):
            # last_routing: (B, num_blocks, 2) — prob of each expert
            r = layer.last_routing
            decisions = r.argmax(dim=-1).cpu()   # (B, num_blocks)  0=T 1=M
            routing_per_layer.append(decisions)

num_layers  = len(routing_per_layer)
num_blocks  = routing_per_layer[0].shape[1]

fig, axes = plt.subplots(1, num_layers, figsize=(14, 5))
if num_layers == 1:
    axes = [axes]

cmap = plt.cm.get_cmap("RdBu", 2)   # 0=red(Transformer), 1=blue(Mamba)

for li, (ax, decisions) in enumerate(zip(axes, routing_per_layer)):
    data = decisions.numpy().astype(float)   # (B, num_blocks)
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=-0.5, vmax=1.5,
                   interpolation="nearest")
    ax.set_title(f"Layer {li + 1} — Routing Decisions", fontsize=12)
    ax.set_xlabel("Block index (position along sequence)")
    ax.set_ylabel("Batch sample")
    ax.set_xticks(range(num_blocks))
    ax.set_yticks(range(BATCH))

# Shared legend
t_patch = mpatches.Patch(color=cmap(0.0), label="Transformer (Expert 0)")
m_patch = mpatches.Patch(color=cmap(1.0), label="Mamba (Expert 1)")
fig.legend(handles=[t_patch, m_patch], loc="lower center",
           ncol=2, fontsize=11, bbox_to_anchor=(0.5, -0.05))

# Compute and print expert utilisation
print("\nRouting utilisation per layer:")
for li, decisions in enumerate(routing_per_layer):
    t_pct = (decisions == 0).float().mean().item()
    m_pct = (decisions == 1).float().mean().item()
    print(f"  Layer {li+1}: Transformer={t_pct:.1%}  Mamba={m_pct:.1%}")

plt.tight_layout()
save_path = os.path.join(os.path.dirname(__file__), "routing_decisions.png")
plt.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"\nHeatmap saved -> {save_path}")
