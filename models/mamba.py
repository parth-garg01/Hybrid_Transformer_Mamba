import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


class _SimpleMambaFallback(nn.Module):
    """Pure-PyTorch SSM approximation with cross-block hidden state support."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        d_inner = int(expand * d_model)
        self.d_inner = d_inner
        self.in_proj = nn.Linear(d_model, d_inner * 2)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv,
                                padding=d_conv - 1, groups=d_inner)
        # GRU hidden state is carried across blocks when cross-block state is enabled
        self.gru = nn.GRU(d_inner, d_inner, batch_first=True)
        self.out_proj = nn.Linear(d_inner, d_model)

    def forward(self, x, hidden=None):
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)

        x_in = x_in.transpose(1, 2)
        x_in = self.conv1d(x_in)[..., :L]
        x_in = F.silu(x_in).transpose(1, 2)

        x_out, new_hidden = self.gru(x_in, hidden)   # new_hidden: (1, B, d_inner)
        return self.out_proj(x_out * F.silu(z)), new_hidden


class MambaBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        if MAMBA_AVAILABLE:
            self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        else:
            self.mamba = _SimpleMambaFallback(d_model, d_state=16, d_conv=4, expand=2)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, hidden=None, return_hidden=False):
        """
        x             : (B, L, D)
        hidden        : (1, B, d_inner) — cross-block SSM state (fallback only)
        return_hidden : if True, returns (output, new_hidden); else returns output
        """
        if MAMBA_AVAILABLE:
            y = self.mamba(x)
            result = self.norm(x + y)
            return (result, None) if return_hidden else result
        else:
            y, new_hidden = self.mamba(x, hidden)
            result = self.norm(x + y)
            return (result, new_hidden) if return_hidden else result