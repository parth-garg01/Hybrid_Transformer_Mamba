"""
Ablation: Fixed decay vs Adaptive entropy-controlled temperature scheduling.

Quantitatively compares the two temperature modes on the same model and
dataset, producing a side-by-side loss curve and temperature trace.

Run:
    python experiments/exp1/ablation_temperature.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from models.multi_layer_hybrid import MultiLayerHybridModel
from models.lm_wrapper         import LMWrapper
from models.losses             import load_balancing_loss, entropy_loss
from training.lm_dataset       import WikiTextDataset

device   = torch.device("cpu")
crit     = nn.CrossEntropyLoss()
STEPS    = 300
EVAL_S   = 40
D_MODEL  = 128
N_HEADS  = 4
BLOCK_SZ = 8

print("Loading dataset...", flush=True)
train_ds = WikiTextDataset(split="train",      seq_len=128, max_chars=3_000_000)
val_ds   = WikiTextDataset(split="validation", seq_len=128,
                           stoi=train_ds.stoi, itos=train_ds.itos, max_chars=None)
V = train_ds.vocab_size


def evaluate(model):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(EVAL_S):
            x, y = val_ds.get_batch(batch_size=16)
            out   = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            total += crit(logits.view(-1, V), y.view(-1)).item()
    model.train()
    return total / EVAL_S


def run(temp_mode, label):
    torch.manual_seed(42)
    base  = MultiLayerHybridModel(d_model=D_MODEL, n_heads=N_HEADS,
                                   block_size=BLOCK_SZ, num_layers=2)
    # Set temperature_mode on all routers
    for layer in base.layers:
        layer.router.temperature_mode = temp_mode

    model = LMWrapper(base, V).to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-3)

    ce_hist, temp_hist = [], []

    for step in range(STEPS):
        x, y = train_ds.get_batch(batch_size=32)
        out   = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        ce  = crit(logits.view(-1, V), y.view(-1))
        aux = torch.tensor(0.0)
        if hasattr(model.model, "last_routing") and model.model.last_routing is not None:
            z   = model.model.last_routing
            aux = 0.01 * load_balancing_loss(z) - 0.001 * entropy_loss(z)
        (ce + aux).backward()
        opt.step(); opt.zero_grad()

        ce_hist.append(ce.item())
        temp_hist.append(base.layers[0].router.temperature)

    val_loss = evaluate(model)
    ppl      = torch.exp(torch.tensor(val_loss)).item()

    print(f"\n[{label}]")
    print(f"  Val Loss : {val_loss:.4f}  PPL {ppl:.2f}")
    print(f"  Temp range: {min(temp_hist):.4f} - {max(temp_hist):.4f}")
    return ce_hist, temp_hist, val_loss


print("\n=== Ablation: temperature scheduling ===\n")
ce_fixed, temp_fixed, vl_fixed = run("fixed_decay", "Fixed decay (×0.999 per step)")
ce_adapt, temp_adapt, vl_adapt = run("adaptive",    "Adaptive (entropy-controlled)")

improvement = (vl_fixed - vl_adapt) / vl_fixed * 100
print(f"\nAdaptive vs Fixed: val loss {vl_fixed:.4f} -> {vl_adapt:.4f} "
      f"({'better' if vl_adapt < vl_fixed else 'worse'} by {abs(improvement):.1f}%)")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

steps = list(range(STEPS))
ax1.plot(steps, ce_fixed, alpha=0.6, label=f"Fixed decay  (val={vl_fixed:.4f})")
ax1.plot(steps, ce_adapt, alpha=0.6, label=f"Adaptive     (val={vl_adapt:.4f})")
ax1.set_xlabel("Training Step");  ax1.set_ylabel("CE Loss")
ax1.set_title("Training Loss: Fixed vs Adaptive Temperature")
ax1.legend();  ax1.grid(True, alpha=0.3)

ax2.plot(steps, temp_fixed, label="Fixed decay")
ax2.plot(steps, temp_adapt, label="Adaptive")
ax2.axhline(y=0.1, color="k", linestyle="--", alpha=0.3, label="Min temp (0.1)")
ax2.axhline(y=2.0, color="k", linestyle=":",  alpha=0.3, label="Max temp (2.0)")
ax2.set_xlabel("Training Step");  ax2.set_ylabel("Temperature")
ax2.set_title("Temperature Schedule Comparison")
ax2.legend();  ax2.grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(os.path.dirname(__file__), "ablation_temperature.png")
plt.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"Plot saved -> {save_path}")

results = {"fixed": {"val_loss": vl_fixed, "ce_curve": ce_fixed, "temp_curve": temp_fixed},
           "adaptive": {"val_loss": vl_adapt, "ce_curve": ce_adapt, "temp_curve": temp_adapt}}
with open(os.path.join(os.path.dirname(__file__), "ablation_temperature.json"), "w") as f:
    json.dump(results, f)
