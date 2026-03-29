import torch.nn as nn
from .mamba import MambaBlock


class MambaOnly(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.mamba = MambaBlock(d_model)

    def forward(self, x):
        return self.mamba(x)