import torch


def load_balancing_loss(z_soft):
    """
    Encourages equal usage of experts
    z_soft: (B, T, K)
    """
    B, T, K = z_soft.shape

    P = z_soft.mean(dim=(0, 1))  # (K,)

    loss = K * torch.sum(P ** 2)

    return loss


def entropy_loss(z_soft):
    """
    Encourages exploration (high entropy)
    """
    eps = 1e-9
    entropy = -torch.sum(z_soft * torch.log(z_soft + eps), dim=-1)

    return entropy.mean()