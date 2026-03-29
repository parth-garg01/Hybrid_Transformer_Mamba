import torch
import torch.nn as nn

from mamba_ssm import Mamba


class MambaBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.mamba = Mamba(
            d_model=d_model,
            d_state=16,   # state size (can tune later)
            d_conv=4,     # convolution kernel
            expand=2      # expansion factor
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """

        y = self.mamba(x)

        # residual connection (VERY IMPORTANT for stability)
        return self.norm(x + y)