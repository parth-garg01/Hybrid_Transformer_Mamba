import torch
import torch.nn as nn

from .transformer import SimpleTransformerBlock
from .mamba import MambaBlock


class SoftHybridModel(nn.Module):
    def __init__(self, d_model=128, n_heads=4):
        super().__init__()

        self.transformer = SimpleTransformerBlock(d_model, n_heads)
        self.mamba = MambaBlock(d_model)

        self.router = nn.Linear(d_model, 1)

    def forward(self, x):
        y_T = self.transformer(x)
        y_M = self.mamba(x)

        p = torch.sigmoid(self.router(x))

        y = p * y_T + (1 - p) * y_M

        return y