import torch.nn as nn
from .transformer import SimpleTransformerBlock


class TransformerOnly(nn.Module):
    def __init__(self, d_model=128, n_heads=4):
        super().__init__()
        self.transformer = SimpleTransformerBlock(d_model, n_heads)

    def forward(self, x):
        return self.transformer(x)