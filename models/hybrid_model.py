import torch
import torch.nn as nn

from .transformer import SimpleTransformerBlock
from .mamba import MambaBlock
from .router import GumbelRouter


class DynamicHybridModel(nn.Module):
    def __init__(self, d_model=128, n_heads=4, block_size=2):
        super().__init__()

        self.d_model = d_model
        self.block_size = block_size

        self.transformer = SimpleTransformerBlock(d_model, n_heads)
        self.mamba = MambaBlock(d_model)

        # Router now works on block representations
        self.router = GumbelRouter(d_model)

    def forward(self, x):
        """
        x: (B, T, D)
        """

        B, T, D = x.shape
        assert T % self.block_size == 0

        num_blocks = T // self.block_size

        # -------------------------
        # 1. BLOCK CREATION
        # -------------------------
        x_blocks = x.view(B, num_blocks, self.block_size, D)

        # -------------------------
        # 2. BLOCK REPRESENTATION
        # -------------------------
        block_repr = x_blocks.mean(dim=2)

    
        # -------------------------
        # 3. ROUTING (PER BLOCK)
        # -------------------------
        z, z_soft, indices = self.router(block_repr)  # (B, num_blocks)

        # -------------------------
        # 4. FLATTEN BLOCKS
        # -------------------------
        x_blocks_flat = x_blocks.view(B * num_blocks, self.block_size, D)
        indices_flat = indices.view(B * num_blocks)

        # -------------------------
        # 5. SPLIT BLOCKS
        # -------------------------
        t_mask = indices_flat == 0
        m_mask = indices_flat == 1

        x_T = x_blocks_flat[t_mask]  # (N_T, block_size, D)
        x_M = x_blocks_flat[m_mask]  # (N_M, block_size, D)

        # -------------------------
        # 6. PROCESS BLOCKS
        # -------------------------
        y_T = torch.zeros_like(x_T)
        y_M = torch.zeros_like(x_M)

        if x_T.shape[0] > 0:
            y_T = self.transformer(x_T)

        if x_M.shape[0] > 0:
            y_M = self.mamba(x_M)

        # -------------------------
        # 7. RECONSTRUCT BLOCKS
        # -------------------------
        y_blocks_flat = torch.zeros_like(x_blocks_flat)

        y_blocks_flat[t_mask] = y_T
        y_blocks_flat[m_mask] = y_M

        # -------------------------
        # 8. RESHAPE BACK
        # -------------------------
        y_blocks = y_blocks_flat.view(B, num_blocks, self.block_size, D)
        y = y_blocks.view(B, T, D)

        return y, z_soft, indices