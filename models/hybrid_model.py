import torch
import torch.nn as nn

from .transformer import SimpleTransformerBlock
from .mamba import MambaBlock
from .router import GumbelRouter


class DynamicHybridModel(nn.Module):
    def __init__(self, d_model=128, n_heads=4, block_size=2, use_cross_block_state=True):
        super().__init__()

        self.d_model = d_model
        self.block_size = block_size
        self.use_cross_block_state = use_cross_block_state

        self.transformer = SimpleTransformerBlock(d_model, n_heads)
        self.mamba = MambaBlock(d_model)
        self.router = GumbelRouter(d_model)

    def forward(self, x):
        """
        x: (B, T, D)

        Novel contributions vs baseline:
          - Blocks processed sequentially so Mamba hidden state can persist
            across non-contiguous Mamba-routed blocks (cross-block state).
          - When use_cross_block_state=False, behaves like the original batched
            implementation (blocks processed independently, no state threading).
        """
        B, T, D = x.shape
        assert T % self.block_size == 0, \
            f"Sequence length {T} must be divisible by block_size {self.block_size}"

        num_blocks = T // self.block_size
        x_blocks = x.view(B, num_blocks, self.block_size, D)

        # Block representations for routing (mean pooling)
        block_repr = x_blocks.mean(dim=2)   # (B, num_blocks, D)

        z, z_soft, indices = self.router(block_repr)
        self.last_routing = z_soft.detach()

        if self.use_cross_block_state:
            y = self._forward_sequential(x, x_blocks, indices, B, T, D, num_blocks)
        else:
            y = self._forward_batched(x_blocks, indices, B, T, D, num_blocks)

        return y, z_soft, indices

    # ------------------------------------------------------------------
    # Sequential forward: passes Mamba hidden state across blocks
    # ------------------------------------------------------------------
    def _forward_sequential(self, x, x_blocks, indices, B, T, D, num_blocks):
        y_list = []
        mamba_hidden = None   # (1, B, d_inner) — persists across Mamba blocks

        for i in range(num_blocks):
            block = x_blocks[:, i]    # (B, block_size, D)
            route = indices[:, i]     # (B,)

            t_mask = (route == 0)
            m_mask = (route == 1)

            y_block = x.new_zeros(B, self.block_size, D)

            if t_mask.any():
                y_block[t_mask] = self.transformer(block[t_mask])

            if m_mask.any():
                # Slice state for only the Mamba-routed samples in this batch
                h_in = mamba_hidden[:, m_mask, :] if mamba_hidden is not None else None
                y_m, h_out = self.mamba(block[m_mask], hidden=h_in, return_hidden=True)
                y_block[m_mask] = y_m

                # Write updated state back into the per-sample state buffer
                if mamba_hidden is None:
                    mamba_hidden = x.new_zeros(h_out.shape[0], B, h_out.shape[-1])
                new_hidden = mamba_hidden.clone()
                new_hidden[:, m_mask, :] = h_out
                mamba_hidden = new_hidden

            y_list.append(y_block)

        return torch.stack(y_list, dim=1).view(B, T, D)

    # ------------------------------------------------------------------
    # Batched forward: original implementation, blocks are independent
    # ------------------------------------------------------------------
    def _forward_batched(self, x_blocks, indices, B, T, D, num_blocks):
        x_blocks_flat = x_blocks.reshape(B * num_blocks, self.block_size, D)
        indices_flat = indices.reshape(B * num_blocks)

        t_mask = (indices_flat == 0)
        m_mask = (indices_flat == 1)

        y_blocks_flat = torch.zeros_like(x_blocks_flat)

        if t_mask.any():
            y_blocks_flat[t_mask] = self.transformer(x_blocks_flat[t_mask])

        if m_mask.any():
            y_blocks_flat[m_mask] = self.mamba(x_blocks_flat[m_mask])

        return y_blocks_flat.view(B, num_blocks, self.block_size, D).view(B, T, D)

    def get_routing_stats(self):
        if not hasattr(self, "last_routing"):
            return None

        probs = self.last_routing   # (B, num_blocks, 2)
        avg = probs.float().mean(dim=(0, 1))

        return {
            "transformer_pct": avg[0].item(),
            "mamba_pct":        avg[1].item(),
            "entropy":          (-avg * torch.log(avg + 1e-9)).sum().item(),
        }
