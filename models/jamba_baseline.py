import torch.nn as nn

from .transformer import SimpleTransformerBlock
from .mamba import MambaBlock


class JambaBaseline(nn.Module):
    """
    Fixed interleaving of Mamba and Transformer layers — no learned routing.

    Inspired by Jamba (AI21 Labs, 2024): interleave SSM layers with periodic
    attention layers at a fixed ratio. This is the strongest static-hybrid
    baseline to compare against adaptive routing.

    Layout with mamba_ratio=3, num_layers=4:
        Layer 0: Mamba
        Layer 1: Mamba
        Layer 2: Mamba
        Layer 3: Transformer   (every mamba_ratio+1 layers)

    Unlike DynamicHybridModel, the routing is predetermined — every token
    goes through the same expert at each layer. This makes the model fully
    parallelisable but inflexible.
    """

    def __init__(self, d_model=256, n_heads=8, num_layers=4, mamba_ratio=3):
        super().__init__()

        blocks = []
        block_types = []
        for i in range(num_layers):
            if (i + 1) % (mamba_ratio + 1) == 0:
                blocks.append(SimpleTransformerBlock(d_model, n_heads))
                block_types.append("transformer")
            else:
                blocks.append(MambaBlock(d_model))
                block_types.append("mamba")

        self.blocks      = nn.ModuleList(blocks)
        self._types      = block_types
        self.norm        = nn.LayerNorm(d_model)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def architecture_summary(self):
        return " -> ".join(self._types)
