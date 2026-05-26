import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
import torch.optim as optim

from models.hybrid_model import DynamicHybridModel
from models.transformer_only import TransformerOnly
from models.mamba_only import MambaOnly
from models.soft_hybrid import SoftHybridModel
from models.lm_wrapper import LMWrapper
from models.losses import load_balancing_loss, entropy_loss
from training.language_model import TextDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# LOAD DATA
# -------------------------
from training.lm_dataset import WikiTextDataset

train_dataset = WikiTextDataset(split="train", seq_len=128)

val_dataset = WikiTextDataset(
    split="validation",
    seq_len=128,
    stoi=train_dataset.stoi,
    itos=train_dataset.itos
)

vocab_size = train_dataset.vocab_size

# -------------------------
# CHOOSE MODEL
# -------------------------
base_model = DynamicHybridModel(block_size=8)

model = LMWrapper(base_model, vocab_size).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


def evaluate(model, dataset, steps=50):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for _ in range(steps):
            x, y = dataset.get_batch(batch_size=32)
            x, y = x.to(device), y.to(device)

            logits = model(x)
           
            
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            total_loss += loss.item()

    model.train()
    return total_loss / steps
# -------------------------
# TRAIN LOOP
# -------------------------
for step in range(300):
    x, y = train_dataset.get_batch(batch_size=32)
    x, y = x.to(device), y.to(device)

    logits = model(x)
    if step % 200 == 0:
        if hasattr(model.model, "get_routing_stats"):
            stats = model.model.get_routing_stats()
            print("Routing:", stats)
    ce_loss = criterion(logits.view(-1, vocab_size), y.view(-1))

    # Auxiliary routing losses (only for hybrid model)
    aux_loss = torch.tensor(0.0, device=device)
    if hasattr(model.model, "last_routing"):
        z_soft = model.model.last_routing
        aux_loss = 0.01 * load_balancing_loss(z_soft) - 0.001 * entropy_loss(z_soft)

    loss = ce_loss + aux_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        val_loss = evaluate(model, val_dataset)

        print(f"Step {step}")
        print(f"Train Loss: {loss.item():.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Train PPL:  {torch.exp(loss).item():.2f}")
        print(f"Val PPL:    {torch.exp(torch.tensor(val_loss)).item():.2f}")