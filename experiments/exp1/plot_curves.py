"""
Plot training curves saved by train_lm.py.

Run after training:
    python experiments/exp1/plot_curves.py
"""
import json, os, sys
import matplotlib.pyplot as plt

curves_path = os.path.join(os.path.dirname(__file__),
                           "../../training/training_curves.json")
curves_path = os.path.normpath(curves_path)

with open(curves_path) as f:
    data = json.load(f)

steps      = data["steps"]
train_loss = data["train_loss"]
val_loss   = data["val_loss"]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(steps, train_loss, marker="o", label="Train Loss", linewidth=2)
ax.plot(steps, val_loss,   marker="s", label="Val Loss",   linewidth=2)
ax.set_xlabel("Training Step", fontsize=12)
ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
ax.set_title("MultiLayerHybridModel — Training Curves\n"
             "(WikiText-103, d=256, 2 layers, block=8)", fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

save_path = os.path.join(os.path.dirname(__file__), "training_curves.png")
plt.tight_layout()
plt.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"Saved -> {save_path}")
