import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import torch
import torch.nn as nn
import torch.optim as optim

from models.multi_layer_hybrid import MultiLayerHybridModel
from models.jamba_baseline     import JambaBaseline
from models.transformer_only   import TransformerOnly
from models.mamba_only         import MambaOnly
from models.soft_hybrid        import SoftHybridModel
from models.hybrid_model       import DynamicHybridModel
from models.lm_wrapper         import LMWrapper
from models.losses             import load_balancing_loss, entropy_loss
from training.lm_dataset       import WikiTextDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Dataset ──────────────────────────────────────────────────────────────────
print("\nLoading WikiText-103 (10M char subset)...")
train_dataset = WikiTextDataset(
    split="train", seq_len=128,
    dataset_name="wikitext-103-raw-v1",
    max_chars=10_000_000,
)
val_dataset = WikiTextDataset(
    split="validation", seq_len=128,
    stoi=train_dataset.stoi,
    itos=train_dataset.itos,
    dataset_name="wikitext-103-raw-v1",
    max_chars=None,
)
vocab_size = train_dataset.vocab_size
print(f"Vocab size: {vocab_size}\n")

# ── Model ─────────────────────────────────────────────────────────────────────
base_model = MultiLayerHybridModel(
    d_model=256, n_heads=8, block_size=8,
    num_layers=2, use_cross_block_state=True,
)
model     = LMWrapper(base_model, vocab_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, steps=60):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(steps):
            x, y = val_dataset.get_batch(batch_size=32)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            total += criterion(logits.view(-1, vocab_size), y.view(-1)).item()
    model.train()
    return total / steps

# ── Training loop ─────────────────────────────────────────────────────────────
STEPS       = 500
LOG_EVERY   = 100
curve_steps, curve_train, curve_val = [], [], []

print("Training MultiLayerHybridModel (2-layer, d=256, WikiText-103)...\n")

for step in range(STEPS):
    x, y = train_dataset.get_batch(batch_size=32)
    x, y = x.to(device), y.to(device)

    logits = model(x)
    if isinstance(logits, tuple):
        logits = logits[0]

    ce_loss = criterion(logits.view(-1, vocab_size), y.view(-1))

    # Auxiliary routing losses (load balancing + entropy)
    aux_loss = torch.tensor(0.0, device=device)
    if hasattr(model.model, "last_routing") and model.model.last_routing is not None:
        z_soft   = model.model.last_routing
        aux_loss = 0.01 * load_balancing_loss(z_soft) - 0.001 * entropy_loss(z_soft)

    loss = ce_loss + aux_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % LOG_EVERY == 0:
        val_loss = evaluate(model)
        curve_steps.append(step)
        curve_train.append(ce_loss.item())
        curve_val.append(val_loss)

        stats = ""
        if hasattr(model.model, "get_routing_stats"):
            r = model.model.get_routing_stats()
            if r:
                stats = (f"  Routing T:{r['transformer_pct']:.1%} "
                         f"M:{r['mamba_pct']:.1%} H:{r['entropy']:.4f}")

        print(f"Step {step:4d} | "
              f"Train {ce_loss.item():.4f} (PPL {torch.exp(ce_loss).item():.2f}) | "
              f"Val {val_loss:.4f} (PPL {torch.exp(torch.tensor(val_loss)).item():.2f})"
              f"{stats}", flush=True)

# ── Save training curves ──────────────────────────────────────────────────────
curves_path = os.path.join(os.path.dirname(__file__), "training_curves.json")
with open(curves_path, "w") as f:
    json.dump({"steps": curve_steps, "train_loss": curve_train,
               "val_loss": curve_val}, f, indent=2)
print(f"\nCurves saved -> {curves_path}")
