import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import time

from models.hybrid_model import DynamicHybridModel
from models.transformer_only import TransformerOnly
from models.mamba_only import MambaOnly
from models.soft_hybrid import SoftHybridModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.randn(16, 64, 128).to(device)

models = {
    "Transformer": TransformerOnly().to(device),
    "Mamba": MambaOnly().to(device),
    "Soft Hybrid": SoftHybridModel().to(device),
    "Sparse Hybrid (b=2)": DynamicHybridModel(block_size=2).to(device),
    "Sparse Hybrid (b=8)": DynamicHybridModel(block_size=8).to(device),
    "Sparse Hybrid (b=16)": DynamicHybridModel(block_size=16).to(device),
}

for name, model in models.items():
    torch.cuda.synchronize()
    start = time.time()

    for _ in range(50):
        _ = model(x)

    torch.cuda.synchronize()
    end = time.time()

    print(f"{name}: {(end - start):.4f} sec")

    # 🔥 ADD THIS PART
    if "Sparse" in name:
        y, _, indices = model(x)
        print("  Routing % Transformer:", (indices == 0).float().mean().item())
        print("  Routing % Mamba:", (indices == 1).float().mean().item())