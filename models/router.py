import torch
import torch.nn as nn
import torch.nn.functional as F


class GumbelRouter(nn.Module):
    """
    Gumbel Straight-Through Router

    Routes each token to one of K experts (default: 2)
    Expert 0 -> Transformer
    Expert 1 -> Mamba (SSM)
    """

    def __init__(self, d_model, num_experts=2, temperature=1.0):
        super().__init__()

        self.num_experts = num_experts
        self.temperature = temperature

        # Routing projection
        self.linear = nn.Linear(d_model, num_experts)

    def sample_gumbel(self, shape, eps=1e-10, device="cpu"):
        """
        Sample Gumbel noise: -log(-log(U))
        """
        U = torch.rand(shape, device=device)
        U = U.clamp(min=eps, max=1.0 - eps)
        return -torch.log(-torch.log(U))

    def gumbel_softmax(self, logits):
        """
        Apply Gumbel-Softmax
        """
        g = self.sample_gumbel(logits.shape, device=logits.device)
        y = logits + g
        return F.softmax(y / self.temperature, dim=-1)

    def forward(self, x):
        """
        x: (B, T, D)

        Returns:
            z       : (B, T, K) -> straight-through routing
            z_soft  : (B, T, K) -> probabilities
            indices : (B, T)    -> selected expert
        """

        # Step 1: logits
        logits = self.linear(x)  # (B, T, K)

        # Step 2: soft routing
        z_soft = self.gumbel_softmax(logits)

        # Step 3: hard routing
        indices = torch.argmax(z_soft, dim=-1)  # (B, T)

        z_hard = F.one_hot(indices, num_classes=self.num_experts).float()

        # Step 4: Straight-Through Estimator
        z = z_hard + (z_soft - z_soft.detach())

        return z, z_soft, indices