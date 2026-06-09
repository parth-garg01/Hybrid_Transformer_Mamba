import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GumbelRouter(nn.Module):
    """
    Gumbel Straight-Through Router with Adaptive Temperature and Deterministic Eval.

    Training : Gumbel-Softmax with adaptive entropy-controlled temperature.
    Inference: Plain softmax — fully deterministic, no Gumbel noise.

    Expert 0 -> Transformer
    Expert 1 -> Mamba (SSM)

    temperature_mode:
        'adaptive'   — closed-loop feedback: warms up when entropy < target,
                       cools when entropy is healthy. Prevents collapse without
                       a hand-tuned schedule.
        'fixed_decay'— classic exponential decay (* 0.999 per step) for ablation.
    """

    def __init__(self, d_model, num_experts=2, temperature=1.0,
                 temperature_mode="adaptive"):
        super().__init__()

        self.num_experts      = num_experts
        self.temperature      = temperature
        self.temperature_mode = temperature_mode
        self._target_entropy  = 0.5 * math.log(num_experts)

        self.linear = nn.Linear(d_model, num_experts)

    # ------------------------------------------------------------------
    def sample_gumbel(self, shape, eps=1e-10, device="cpu"):
        U = torch.rand(shape, device=device)
        U = U.clamp(min=eps, max=1.0 - eps)
        return -torch.log(-torch.log(U))

    def gumbel_softmax(self, logits):
        g = self.sample_gumbel(logits.shape, device=logits.device)
        return F.softmax((logits + g) / self.temperature, dim=-1)

    # ------------------------------------------------------------------
    def forward(self, x):
        """
        x: (B, T, D)  — block representations

        Returns:
            z       : (B, T, K) — straight-through routing weights
            z_soft  : (B, T, K) — soft probabilities (for aux losses / stats)
            indices : (B, T)    — hard expert assignment
        """
        logits = self.linear(x)

        if self.training:
            # Stochastic routing with Gumbel exploration
            z_soft = self.gumbel_softmax(logits)

            with torch.no_grad():
                avg_probs = z_soft.mean(dim=(0, 1))   # (K,)
                entropy   = -(avg_probs * torch.log(avg_probs + 1e-9)).sum().item()

                if self.temperature_mode == "adaptive":
                    if entropy < self._target_entropy:
                        # Collapse risk — warm up so router explores
                        self.temperature = min(self.temperature * 1.005, 2.0)
                    else:
                        # Healthy entropy — cool for sharper decisions
                        self.temperature = max(self.temperature * 0.999, 0.1)
                else:  # fixed_decay
                    self.temperature = max(self.temperature * 0.999, 0.1)
        else:
            # Deterministic routing at inference — no Gumbel noise
            z_soft = F.softmax(logits / self.temperature, dim=-1)

        indices = torch.argmax(z_soft, dim=-1)
        z_hard  = F.one_hot(indices, num_classes=self.num_experts).float()
        z       = z_hard + (z_soft - z_soft.detach())   # straight-through

        return z, z_soft, indices
