import torch
import torch.nn.functional as F

from models.hybrid_model import DynamicHybridModel
from models.losses import load_balancing_loss, entropy_loss

# -------------------------
# DEVICE
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# MODEL
# -------------------------
model = DynamicHybridModel(d_model=128).to(device)

# -------------------------
# INPUT
# -------------------------
x = torch.randn(2, 6, 128).to(device)

# -------------------------
# FORWARD
# -------------------------
y, z_soft, indices = model(x)

# -------------------------
# FAKE TARGET (for testing)
# -------------------------
target = torch.randn_like(y)

# -------------------------
# LOSSES
# -------------------------
task_loss = F.mse_loss(y, target)

lb_loss = load_balancing_loss(z_soft)
ent_loss = entropy_loss(z_soft)

# weights
lambda_lb = 0.01
lambda_ent = 0.001

loss = task_loss + lambda_lb * lb_loss + lambda_ent * ent_loss

# -------------------------
# PRINT
# -------------------------
print("\n--- LOSSES ---")
print("Task Loss:", task_loss.item())
print("Load Balance Loss:", lb_loss.item())
print("Entropy Loss:", ent_loss.item())
print("Total Loss:", loss.item())

# -------------------------
# ROUTING STATS
# -------------------------
print("\n--- ROUTING ---")
print("Block routing:", indices)
print("Transformer %:", (indices == 0).float().mean().item())
print("Mamba %:", (indices == 1).float().mean().item())
print("Output shape:", y.shape)