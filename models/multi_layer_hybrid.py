import torch
import torch.nn as nn

from .hybrid_model import DynamicHybridModel


class MultiLayerHybridModel(nn.Module):
    """
    Stacks N independent DynamicHybridModel layers with a shared final LayerNorm.

    Each layer has its own router and its own Transformer + Mamba expert,
    so routing decisions are learned independently at each depth. Cross-block
    SSM state is threaded within each layer independently.

    This is the proper multi-layer extension of the single-layer design and is
    required to compete with multi-layer baselines like Jamba.
    """

    def __init__(self, d_model=256, n_heads=8, block_size=8,
                 num_layers=2, use_cross_block_state=True):
        super().__init__()
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            DynamicHybridModel(
                d_model=d_model,
                n_heads=n_heads,
                block_size=block_size,
                use_cross_block_state=use_cross_block_state,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        last_z_soft = None
        last_indices = None

        for layer in self.layers:
            x, z_soft, indices = layer(x)
            last_z_soft  = z_soft
            last_indices = indices

        x = self.norm(x)

        # Expose last layer's routing for aux losses and stats
        self.last_routing = last_z_soft.detach() if last_z_soft is not None else None

        return x, last_z_soft, last_indices

    def get_routing_stats(self):
        """Average routing statistics across all layers."""
        all_stats = [layer.get_routing_stats() for layer in self.layers]
        all_stats = [s for s in all_stats if s is not None]

        if not all_stats:
            return None

        return {
            "transformer_pct": sum(s["transformer_pct"] for s in all_stats) / len(all_stats),
            "mamba_pct":        sum(s["mamba_pct"]        for s in all_stats) / len(all_stats),
            "entropy":          sum(s["entropy"]           for s in all_stats) / len(all_stats),
        }
