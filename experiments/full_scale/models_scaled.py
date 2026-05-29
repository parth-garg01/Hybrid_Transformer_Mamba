"""
Scaled models (~20M params) for full-scale experiments.
All models share the same embedding + LM head; only the backbone differs.
Supports: TransformerOnly, MambaOnly, ABR (Adaptive Block Routing).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Try importing real Mamba; fall back to GRU-based stub ────────────────────
try:
    from mamba_ssm import Mamba as MambaSSM
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Basic building blocks
# ─────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # Flash attention when available, else manual
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class FFN(nn.Module):
    def __init__(self, d_model, mult=4):
        super().__init__()
        self.fc1  = nn.Linear(d_model, d_model * mult, bias=False)
        self.fc2  = nn.Linear(d_model * mult, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model * mult, bias=False)

    def forward(self, x):
        return self.fc2(F.silu(self.fc1(x)) * self.gate(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_mult=4):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn  = CausalSelfAttention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model)
        self.ffn   = FFN(d_model, ffn_mult)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class MambaBlock(nn.Module):
    """Wraps real Mamba SSM or a GRU fallback (CPU/no-CUDA)."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = RMSNorm(d_model)
        if MAMBA_AVAILABLE:
            self.ssm = MambaSSM(d_model=d_model, d_state=d_state,
                                d_conv=d_conv, expand=expand)
            self._use_real = True
        else:
            d_inner = d_model * expand
            self.in_proj  = nn.Linear(d_model, d_inner * 2, bias=False)
            self.conv1d   = nn.Conv1d(d_inner, d_inner, d_conv,
                                      padding=d_conv - 1, groups=d_inner)
            self.gru      = nn.GRU(d_inner, d_inner, batch_first=True)
            self.out_proj = nn.Linear(d_inner, d_model, bias=False)
            self._use_real = False
            self.d_inner   = d_inner

    def forward(self, x):
        residual = x
        x = self.norm(x)
        if self._use_real:
            return residual + self.ssm(x)
        else:
            B, T, _ = x.shape
            xz = self.in_proj(x)
            xi, z = xz.chunk(2, dim=-1)
            xi = self.conv1d(xi.transpose(1, 2))[:, :, :T].transpose(1, 2)
            xi = F.silu(xi)
            xi_out, _ = self.gru(xi)
            return residual + self.out_proj(xi_out * F.silu(z))


# ─────────────────────────────────────────────────────────────────────────────
# Gumbel-Softmax Router
# ─────────────────────────────────────────────────────────────────────────────

class GumbelRouter(nn.Module):
    def __init__(self, d_model, num_experts=2, init_temp=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.temperature = init_temp
        self._target_H   = 0.5 * math.log(num_experts)
        self.linear      = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        # x: (B, block_size, D) → pool → (B, D)
        summary = x.mean(dim=1)
        logits  = self.linear(summary)          # (B, 2)

        if self.training:
            g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
            z_soft = F.softmax((logits + g) / self.temperature, dim=-1)
            with torch.no_grad():
                avg_p   = z_soft.mean(0)
                H       = -(avg_p * (avg_p + 1e-9).log()).sum().item()
                if H < self._target_H:
                    self.temperature = min(self.temperature * 1.005, 2.0)
                else:
                    self.temperature = max(self.temperature * 0.999, 0.1)
        else:
            z_soft = F.softmax(logits / self.temperature, dim=-1)

        indices = z_soft.argmax(dim=-1)
        z_hard  = F.one_hot(indices, self.num_experts).float()
        z       = z_hard + (z_soft - z_soft.detach())
        return z, z_soft, indices


# ─────────────────────────────────────────────────────────────────────────────
# Single ABR Routing Layer
# ─────────────────────────────────────────────────────────────────────────────

class ABRLayer(nn.Module):
    def __init__(self, d_model, n_heads, block_size, use_cross_block_state=True,
                 ffn_mult=4):
        super().__init__()
        self.block_size           = block_size
        self.use_cross_block_state = use_cross_block_state
        self.router      = GumbelRouter(d_model)
        self.transformer = TransformerBlock(d_model, n_heads, ffn_mult)
        self.mamba       = MambaBlock(d_model)
        self.last_routing = None

    def forward(self, x):
        B, T, D = x.shape
        b = self.block_size
        pad = (b - T % b) % b
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        _, Tp, _ = x.shape
        N = Tp // b

        blocks = x.view(B, N, b, D)
        z_list, idx_list = [], []
        for i in range(N):
            z, z_soft, idx = self.router(blocks[:, i])
            z_list.append(z_soft)
            idx_list.append(idx)

        z_soft_all = torch.stack(z_list, dim=1)   # (B, N, 2)
        indices    = torch.stack(idx_list, dim=1)  # (B, N)
        self.last_routing = z_soft_all

        if self.use_cross_block_state:
            out = self._forward_cbsc(blocks, indices, B, N, b, D)
        else:
            out = self._forward_batched(blocks, indices, B, N, b, D)

        out = out.view(B, Tp, D)[:, :T, :]
        return out, z_soft_all, indices

    def _forward_cbsc(self, blocks, indices, B, N, b, D):
        """
        Two-pass vectorized CBSC (fast, no per-block Python loop on the GPU).

        Pass 1 — run every block through its expert in one large batched call.
        Pass 2 — for each Mamba block that directly follows another Mamba block,
                  inject the predecessor's last-token output into position 0 of
                  its input, then re-run Mamba for those blocks only.

        This keeps GPU utilisation high (two big CUDA calls instead of N small
        sequential ones) while correctly propagating the recurrent context that
        defines CBSC across block boundaries.
        """
        flat   = blocks.view(B * N, b, D)   # (B*N, b, D)
        r_flat = indices.view(B * N)        # (B*N,)
        t_flat = (r_flat == 0)
        m_flat = (r_flat == 1)

        # ── Pass 1: full parallel forward ────────────────────────────────────
        y_flat = flat.clone()
        if t_flat.any():
            y_flat[t_flat] = self.transformer(flat[t_flat])
        if m_flat.any():
            y_flat[m_flat] = self.mamba(flat[m_flat])

        # ── Find consecutive Mamba→Mamba block pairs that need CBSC ─────────
        m_route   = (indices == 1)                  # (B, N) bool
        prev_mamba = torch.zeros_like(m_route)
        prev_mamba[:, 1:] = m_route[:, :-1]
        cbsc_mask = m_route & prev_mamba            # (B, N)

        if cbsc_mask.any():
            # Last token of each block, pass-1 output: (B, N, D)
            last_tok = y_flat.view(B, N, b, D)[:, :, -1, :]

            # Predecessor's last token shifted by 1: (B, N, D)
            prev_last = torch.zeros_like(last_tok)
            prev_last[:, 1:, :] = last_tok[:, :-1, :]

            # Build CBSC-augmented inputs: add predecessor last token to pos-0
            inp2 = flat.view(B, N, b, D).clone()
            cbsc_exp = cbsc_mask.unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1)
            prev_exp  = prev_last.unsqueeze(2)                 # (B, N, 1, D)
            inp2[:, :, :1, :] = inp2[:, :, :1, :] + cbsc_exp * prev_exp

            # ── Pass 2: re-run only the CBSC Mamba blocks ───────────────────
            cbsc_flat = cbsc_mask.view(B * N)
            inp2_flat = inp2.view(B * N, b, D)
            y_flat2   = y_flat.clone()
            y_flat2[cbsc_flat] = self.mamba(inp2_flat[cbsc_flat])
            return y_flat2.view(B, N * b, D)

        return y_flat.view(B, N * b, D)

    def _forward_batched(self, blocks, indices, B, N, b, D):
        flat = blocks.view(B * N, b, D)
        t_out = self.transformer(flat)
        m_out = self.mamba(flat)
        route_flat = indices.view(B * N)
        sel = route_flat.float().unsqueeze(-1).unsqueeze(-1)
        out = (1 - sel) * t_out + sel * m_out
        return out.view(B, N, b, D).view(B, N * b, D)


# ─────────────────────────────────────────────────────────────────────────────
# Full Models
# ─────────────────────────────────────────────────────────────────────────────

class BaseModel(nn.Module):
    """Shared embedding + positional + LM head."""
    def __init__(self, vocab_size, d_model, max_seq_len=1024):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.norm    = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight   # weight tying
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def encode(self, x):
        B, T = x.shape
        pos  = torch.arange(T, device=x.device)
        return self.embed(x) + self.pos_emb(pos)

    def decode(self, h):
        return self.lm_head(self.norm(h))


class TransformerLM(BaseModel):
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=4,
                 ffn_mult=4, max_seq_len=1024):
        super().__init__(vocab_size, d_model, max_seq_len)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_mult)
            for _ in range(n_layers)])

    def forward(self, x):
        h = self.encode(x)
        for layer in self.layers:
            h = layer(h)
        return self.decode(h)


class MambaLM(BaseModel):
    def __init__(self, vocab_size, d_model=256, n_layers=4, max_seq_len=1024):
        super().__init__(vocab_size, d_model, max_seq_len)
        self.layers = nn.ModuleList([MambaBlock(d_model) for _ in range(n_layers)])

    def forward(self, x):
        h = self.encode(x)
        for layer in self.layers:
            h = layer(h)
        return self.decode(h)


class ABRLM(BaseModel):
    """
    Multi-layer Adaptive Block Routing Language Model.
    use_routing=False → pure Transformer (for routing-off ablation)
    use_cross_block_state → CBSC on/off
    """
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=2,
                 block_size=8, use_cross_block_state=True, ffn_mult=4,
                 max_seq_len=1024):
        super().__init__(vocab_size, d_model, max_seq_len)
        self.layers = nn.ModuleList([
            ABRLayer(d_model, n_heads, block_size, use_cross_block_state, ffn_mult)
            for _ in range(n_layers)])
        self.last_routing = None

    def forward(self, x):
        h = self.encode(x)
        last_z = None
        for layer in self.layers:
            h, z_soft, _ = layer(h)
            last_z = z_soft
        self.last_routing = last_z.detach() if last_z is not None else None
        return self.decode(h)

    def get_routing_stats(self):
        if self.last_routing is None:
            return {}
        z = self.last_routing
        t_pct = (z[..., 0].mean()).item()
        m_pct = (z[..., 1].mean()).item()
        avg   = z.mean(dim=(0, 1))
        H     = -(avg * (avg + 1e-9).log()).sum().item()
        return {"transformer_pct": t_pct, "mamba_pct": m_pct, "entropy": H}


def build_model(cfg_name, vocab_size, d_model=256, n_heads=8, block_size=8,
                ffn_mult=4, max_seq_len=1024):
    """Factory that builds the right model from an ABLATION_CONFIGS entry."""
    label, n_layers, use_routing, use_cbsc, expert_type = cfg_name

    if expert_type == "transformer":
        return TransformerLM(vocab_size, d_model, n_heads, n_layers, ffn_mult, max_seq_len)
    elif expert_type == "mamba":
        return MambaLM(vocab_size, d_model, n_layers, max_seq_len)
    else:
        return ABRLM(vocab_size, d_model, n_heads, n_layers, block_size,
                     use_cbsc, ffn_mult, max_seq_len)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
