import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GumbelRouter(nn.Module):
    """
    Gumbel Straight-Through Router with Adaptive Temperature Scheduling.

    Expert 0 -> Transformer
    Expert 1 -> Mamba (SSM)

    Temperature adaptation:
      - If routing entropy drops below target (expert collapse risk), temperature
        is warmed up so the router explores more.
      - If entropy is healthy, temperature is slowly cooled for sharper decisions.
    This replaces the fixed-decay schedule with a closed-loop feedback controller.
    """

    def __init__(self, d_model, num_experts=2, temperature=1.0):
        super().__init__()

        self.num_experts = num_experts
        self.temperature = temperature

        # Target entropy: half of maximum entropy (log K / 2)
        self._target_entropy = 0.5 * math.log(num_experts)

        self.linear = nn.Linear(d_model, num_experts)

    def sample_gumbel(self, shape, eps=1e-10, device="cpu"):
        U = torch.rand(shape, device=device)
        U = U.clamp(min=eps, max=1.0 - eps)
        return -torch.log(-torch.log(U))

    def gumbel_softmax(self, logits):
        g = self.sample_gumbel(logits.shape, device=logits.device)
        return F.softmax((logits + g) / self.temperature, dim=-1)

    def forward(self, x):
        """
        x: (B, T, D)  — block representations

        Returns:
            z       : (B, T, K) — straight-through routing weights
            z_soft  : (B, T, K) — soft probabilities (used for aux losses)
            indices : (B, T)    — hard expert assignment
        """
        logits = self.linear(x)
        z_soft = self.gumbel_softmax(logits)

        if self.training:
            with torch.no_grad():
                avg_probs = z_soft.mean(dim=(0, 1))   # (K,)
                entropy = -(avg_probs * torch.log(avg_probs + 1e-9)).sum().item()

                if entropy < self._target_entropy:
                    # Router is collapsing toward one expert — increase temperature
                    self.temperature = min(self.temperature * 1.005, 2.0)
                else:
                    # Entropy is healthy — slowly cool toward sharper decisions
                    self.temperature = max(self.temperature * 0.999, 0.1)

        indices = torch.argmax(z_soft, dim=-1)
        z_hard = F.one_hot(indices, num_classes=self.num_experts).float()
        z = z_hard + (z_soft - z_soft.detach())   # straight-through estimator

        return z, z_soft, indices
