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
from training.language_model import TextDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# LOAD DATA
# -------------------------
with open("data/sample.txt", "r") as f:
    text = f.read()

dataset = TextDataset(text)
vocab_size = dataset.tokenizer.vocab_size

# -------------------------
# CHOOSE MODEL
# -------------------------
base_model = SoftHybridModel()

model = LMWrapper(base_model, vocab_size).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# -------------------------
# TRAIN LOOP
# -------------------------
for step in range(300):
    x, y = dataset.get_batch()
    x, y = x.to(device), y.to(device)

    logits = model(x)

    loss = criterion(logits.view(-1, vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")