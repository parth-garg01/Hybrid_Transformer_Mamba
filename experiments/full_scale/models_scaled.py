"""
Scaled models (~20M params) for full-scale experiments.
All models share the same embedding + LM head; only the backbone differs.
Supports: TransformerOnly, MambaOnly, Jamba-style (1:1, 1:3), ABR (Adaptive Block
Routing) with cross-block state continuity (CBSC).

Public API (used by config.py / the training loop; keep these names stable):
    MAMBA_BACKEND            "mamba_ssm" or "gru_fallback"
    require_real_mamba(allow_fallback)
    build_model(cfg, vocab_size, **kw)   cfg = legacy tuple or dict (see build_model)
    model.routing_probs                  list of non-detached z_soft (B, N, E), one per ABR layer
    model.last_routing                   detached z_soft of the last ABR layer (stats only)
    model.update_router_temperatures()   call ONCE per optimizer step
    model.get_routing_stats()            overall + per-layer stats incl. tau
    count_params(model)

Expert index convention everywhere: 0 = Transformer (attention), 1 = Mamba.

`legacy=True` reproduces the original (pre-fix) code exactly: LegacyGumbelRouter +
LegacyABRLayer below keep the old maths (bugs B1-B5 included).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba as MambaSSM
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False

try:  # Mamba-2 (needed for gated state passing on the real backend)
    from mamba_ssm import Mamba2 as Mamba2SSM
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    MAMBA2_AVAILABLE = True
except Exception:  # ImportError, or triton missing
    MAMBA2_AVAILABLE = False

MAMBA_BACKEND = "mamba_ssm" if MAMBA_AVAILABLE else "gru_fallback"


def require_real_mamba(allow_fallback: bool = False) -> str:
    """Refuse to continue on the GRU stand-in unless explicitly allowed.

    Returns the backend name. Training/benchmark scripts must call this before
    building any Mamba-containing model and record the returned value.
    """
    if MAMBA_BACKEND != "mamba_ssm" and not allow_fallback:
        raise RuntimeError(
            "Real Mamba (mamba_ssm) is not available: Mamba layers would silently be "
            "replaced by a GRU stand-in. Results from the stand-in are NOT Mamba results. "
            "Install mamba_ssm + causal-conv1d on a CUDA machine, or pass "
            "--allow-gru-fallback for debugging only (the run is then non-reportable).")
    return MAMBA_BACKEND


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
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

    def forward_queries(self, h_q, q_pos, h_kv):
        """Attention for a subset of query positions over a full key/value sequence.

        h_q   : (B, L, C)  normed inputs at the query positions
        q_pos : (B, L)     absolute position of each query (long)
        h_kv  : (B, T, C)  normed inputs of the whole layer input (keys/values)
        Query at position p attends to keys at positions <= p (causal).
        """
        B, L, C = h_q.shape
        T = h_kv.shape[1]
        W = self.qkv.weight
        q = F.linear(h_q, W[:C])
        k, v = F.linear(h_kv, W[C:]).split(C, dim=2)
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        key_pos = torch.arange(T, device=h_q.device)
        mask = key_pos.view(1, 1, 1, T) <= q_pos.view(B, 1, L, 1)   # (B, 1, L, T)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(B, L, C)
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

    def forward_routed(self, x, tok_mask):
        """P1.2 efficient path. Queries only at routed tokens, keys/values from ALL
        tokens of the layer input (causal). FFN/norms run on routed tokens only.

        x        : (B, T, D) full layer input
        tok_mask : (B, T) bool, True where the token's block is routed to attention
        returns  : (B, T, D) block outputs at routed tokens, exactly 0 elsewhere
        """
        B, T, D = x.shape
        counts = tok_mask.sum(1)                                     # (B,)
        L = int(counts.max().item())
        if L == 0:
            return x.new_zeros(B, T, D)
        # stable sort puts routed positions first, in their original order
        order = torch.sort((~tok_mask).to(torch.int8), dim=1, stable=True).indices
        q_pos = order[:, :L]                                         # (B, L)
        valid = torch.arange(L, device=x.device).view(1, L) < counts.view(B, 1)
        q_pos = torch.where(valid, q_pos, torch.zeros_like(q_pos))   # pads -> pos 0 (no NaN)
        idx = q_pos.unsqueeze(-1).expand(B, L, D)
        x_q = x.gather(1, idx)
        h_kv = self.norm1(x)
        h_q = h_kv.gather(1, idx)
        y = x_q + self.attn.forward_queries(h_q, q_pos, h_kv)
        y = y + self.ffn(self.norm2(y))
        y = y * valid.unsqueeze(-1).to(y.dtype)
        return x.new_zeros(B, T, D).scatter_add(1, idx, y)

    def forward_routed_reference(self, x, tok_mask):
        """Simple reference: full causal block over the whole sequence, then select."""
        y = self.forward(x)
        return torch.where(tok_mask.unsqueeze(-1), y, torch.zeros_like(y))


class MambaBlock(nn.Module):
    """Wraps real Mamba SSM or a GRU fallback (CPU/no-CUDA). Unchanged legacy block
    (used by MambaLM, JambaLM and LegacyABRLayer)."""
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
        B, T, _ = x.shape
        xz = self.in_proj(x)
        xi, z = xz.chunk(2, dim=-1)
        xi = self.conv1d(xi.transpose(1, 2))[:, :, :T].transpose(1, 2)
        xi = F.silu(xi)
        xi_out, _ = self.gru(xi)
        return residual + self.out_proj(xi_out * F.silu(z))


# ─────────────────────────────────────────────────────────────────────────────
# Mamba expert with explicit state interface (P1.3)
# ─────────────────────────────────────────────────────────────────────────────

class GRUStateGate(nn.Module):
    """h_init = sigmoid(W_g h_prev + b_g) * h_prev for the GRU stand-in.
    h_prev: (1, B, d_inner). W_g: d_inner x d_inner (zero init), b_g: d_inner (+4 init).
    Params: d_inner^2 + d_inner."""
    def __init__(self, d_inner, bias_init=4.0):
        super().__init__()
        self.lin = nn.Linear(d_inner, d_inner, bias=True)
        nn.init.zeros_(self.lin.weight)
        nn.init.constant_(self.lin.bias, bias_init)

    def forward(self, h):
        return torch.sigmoid(self.lin(h)) * h


class Mamba2StateGate(nn.Module):
    """Gate for the Mamba-2 SSM state h_prev: (B, nheads, headdim, d_state).

    A full linear map over the flattened state (nheads*headdim*d_state)^2 would be
    ~1e9 params at our size, so W_g is structured: one d_state x d_state matrix per
    head, shared over headdim:  pre[b,h,p,m] = sum_n h_prev[b,h,p,n] W[h,n,m] + b[h,p,m].
    W: (nheads, d_state, d_state) zero init; b: (nheads, headdim, d_state) = +4.
    Params: nheads * d_state * (d_state + headdim).
    """
    def __init__(self, nheads, headdim, d_state, bias_init=4.0):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(nheads, d_state, d_state))
        self.bias = nn.Parameter(torch.full((nheads, headdim, d_state), float(bias_init)))

    def forward(self, h):
        pre = torch.einsum("bhpn,hnm->bhpm", h, self.weight.to(h.dtype)) + self.bias.to(h.dtype)
        return torch.sigmoid(pre) * h


class MambaExpert(nn.Module):
    """Pre-norm residual Mamba layer with an explicit recurrent-state interface:

        forward(x, init_state=None) -> (y, final_state)

    x, y: (B, L, d_model). y includes the residual (y = x + mixer(norm(x))).
    Backends (chosen at construction):
      "gru"    : GRU stand-in (same parameter names as MambaBlock's fallback).
                 state = (conv_state (B, d_inner, d_conv-1), h (1, B, d_inner)).
      "mamba1" : mamba_ssm.Mamba. Does NOT support init_state / final_state
                 (returns final_state=None); fine for cbsc none/state (packed run).
      "mamba2" : mamba_ssm.Mamba2 params, forward re-implemented with
                 mamba_chunk_scan_combined so states can be passed in and out.
                 state = (conv_state (B, conv_dim, d_conv-1),
                          ssm_state  (B, nheads, headdim, d_state)),
                 conv_dim = d_inner + 2*ngroups*d_state.
                 # UNVERIFIED: needs Kaggle GPU check
    The conv state is passed through ungated; only the recurrent part is gated.
    """
    def __init__(self, d_model, d_state=None, d_conv=4, expand=2, backend=None,
                 mamba_version=1):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.d_conv = d_conv
        if backend is None:
            if not MAMBA_AVAILABLE:
                backend = "gru"
            elif mamba_version == 2:
                backend = "mamba2"
            else:
                backend = "mamba1"
        self.backend = backend
        if backend == "mamba1":
            self.ssm = MambaSSM(d_model=d_model, d_state=d_state or 16,
                                d_conv=d_conv, expand=expand)
        elif backend == "mamba2":
            if not MAMBA2_AVAILABLE:
                raise RuntimeError("mamba_version=2 requested but mamba_ssm.Mamba2 / "
                                   "mamba_chunk_scan_combined could not be imported.")
            self.ssm = Mamba2SSM(d_model=d_model, d_state=d_state or 64,
                                 d_conv=d_conv, expand=expand)
        elif backend == "gru":
            d_inner = d_model * expand
            self.in_proj  = nn.Linear(d_model, d_inner * 2, bias=False)
            self.conv1d   = nn.Conv1d(d_inner, d_inner, d_conv,
                                      padding=d_conv - 1, groups=d_inner)
            self.gru      = nn.GRU(d_inner, d_inner, batch_first=True)
            self.out_proj = nn.Linear(d_inner, d_model, bias=False)
            self.d_inner  = d_inner
        else:
            raise ValueError(f"unknown Mamba backend {backend!r}")

    @property
    def supports_state(self):
        return self.backend in ("gru", "mamba2")

    def make_state_gate(self, bias_init=4.0):
        if self.backend == "gru":
            return GRUStateGate(self.d_inner, bias_init)
        if self.backend == "mamba2":
            m = self.ssm
            return Mamba2StateGate(m.nheads, m.headdim, m.d_state, bias_init)
        raise ValueError("gated_state CBSC needs a state-passing backend "
                         "(GRU stand-in or Mamba-2); Mamba-1 cannot return its state.")

    def forward(self, x, init_state=None):
        h = self.norm(x)
        if self.backend == "gru":
            y, st = self._forward_gru(h, init_state)
        elif self.backend == "mamba2":
            y, st = self._forward_mamba2(h, init_state)
        else:
            if init_state is not None:
                raise NotImplementedError("Mamba-1 backend cannot take an initial state")
            y, st = self.ssm(h), None
        return x + y, st

    def _causal_conv(self, xi_t, conv_prev, weight, bias):
        """xi_t: (B, C, L). Causal depthwise conv continuing from conv_prev (B, C, K-1).
        Returns (out (B, C, L), new conv state (B, C, K-1))."""
        K = self.d_conv
        if conv_prev is None:
            conv_prev = xi_t.new_zeros(xi_t.shape[0], xi_t.shape[1], K - 1)
        xin = torch.cat([conv_prev, xi_t], dim=-1)
        out = F.conv1d(xin, weight, bias, groups=xi_t.shape[1])
        return out, xin[..., xin.shape[-1] - (K - 1):]

    def _forward_gru(self, x, init_state):
        conv_prev, h0 = init_state if init_state is not None else (None, None)
        xz = self.in_proj(x)
        xi, z = xz.chunk(2, dim=-1)
        xi_c, conv_new = self._causal_conv(xi.transpose(1, 2), conv_prev,
                                           self.conv1d.weight, self.conv1d.bias)
        xi = F.silu(xi_c.transpose(1, 2))
        xi_out, h_new = self.gru(xi, h0)
        return self.out_proj(xi_out * F.silu(z)), (conv_new, h_new)

    def _forward_mamba2(self, u, init_state):
        # UNVERIFIED: needs Kaggle GPU check. Mirrors mamba_ssm Mamba2.forward
        # (non-fused path) but threads conv_state and ssm_state through explicitly.
        m = self.ssm
        conv_prev, ssm_prev = init_state if init_state is not None else (None, None)
        Bsz, L, _ = u.shape
        zxbcdt = m.in_proj(u)
        A = -torch.exp(m.A_log.float())
        d_mlp = (zxbcdt.shape[-1] - 2 * m.d_ssm - 2 * m.ngroups * m.d_state - m.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt, [d_mlp, d_mlp, m.d_ssm, m.d_ssm + 2 * m.ngroups * m.d_state, m.nheads],
            dim=-1)
        xBC_c, conv_new = self._causal_conv(xBC.transpose(1, 2), conv_prev,
                                            m.conv1d.weight, m.conv1d.bias)
        xBC = F.silu(xBC_c.transpose(1, 2))
        xs, Bm, Cm = torch.split(
            xBC, [m.d_ssm, m.ngroups * m.d_state, m.ngroups * m.d_state], dim=-1)
        H, P, G, N = m.nheads, m.headdim, m.ngroups, m.d_state
        dt_kw = {} if tuple(m.dt_limit) == (0.0, float("inf")) else {"dt_limit": m.dt_limit}
        D = m.D.view(H, P) if m.D_has_hdim else m.D
        y, ssm_new = mamba_chunk_scan_combined(
            xs.reshape(Bsz, L, H, P), dt, A,
            Bm.reshape(Bsz, L, G, N), Cm.reshape(Bsz, L, G, N),
            chunk_size=m.chunk_size, D=D,
            z=None if m.rmsnorm else z.reshape(Bsz, L, H, P),
            dt_bias=m.dt_bias, dt_softplus=True,
            initial_states=ssm_prev, return_final_states=True, **dt_kw)
        y = y.reshape(Bsz, L, H * P)
        if m.rmsnorm:
            y = m.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        return m.out_proj(y), (conv_new, ssm_new)


# ─────────────────────────────────────────────────────────────────────────────
# Gumbel-Softmax Routers
# ─────────────────────────────────────────────────────────────────────────────

class LegacyGumbelRouter(nn.Module):
    """Original router (temperature updated inside forward; bug B5). legacy=True only."""
    def __init__(self, d_model, num_experts=2, init_temp=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.temperature = init_temp
        self._target_H   = 0.5 * math.log(num_experts)
        self.linear      = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        summary = x.mean(dim=1)
        logits  = self.linear(summary)

        if self.training:
            g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
            z_soft = F.softmax((logits + g) / self.temperature, dim=-1)
            with torch.no_grad():
                avg_p = z_soft.mean(0)
                H     = -(avg_p * (avg_p + 1e-9).log()).sum().item()
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

    def update_temperature(self, H=None):  # legacy updates inside forward
        return self.temperature


class GumbelRouter(nn.Module):
    """Gumbel-softmax router over per-block summaries (P1.1, P1.4).

    forward(summary (B, N, D)) -> z (B, N, E) straight-through one-hot,
                                  z_soft (B, N, E), indices (B, N)
    The temperature is NOT changed in forward. On each training forward the
    entropy of the batch-average routing distribution is recorded; the training
    loop calls update_temperature() once per optimizer step to apply the schedule.
    """
    def __init__(self, d_model, num_experts=2, init_temp=1.0, tau_min=0.1, tau_max=2.0):
        super().__init__()
        self.num_experts = num_experts
        self.temperature = init_temp
        self.tau_min, self.tau_max = tau_min, tau_max
        self._target_H   = 0.5 * math.log(num_experts)
        self._last_H     = None
        self.linear      = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, summary, force_indices=None):
        logits = self.linear(summary)
        if self.training:
            g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
            z_soft = F.softmax((logits + g) / self.temperature, dim=-1)
            with torch.no_grad():
                avg_p = z_soft.reshape(-1, self.num_experts).mean(0)
                self._last_H = -(avg_p * (avg_p + 1e-9).log()).sum().item()
        else:
            z_soft = F.softmax(logits / self.temperature, dim=-1)
        indices = z_soft.argmax(dim=-1) if force_indices is None else force_indices
        z_hard  = F.one_hot(indices, self.num_experts).to(z_soft.dtype)
        z       = z_hard + (z_soft - z_soft.detach())
        return z, z_soft, indices

    def update_temperature(self, H=None):
        """Apply one step of the temperature schedule. Uses H if given, else the
        entropy recorded on the last training forward (consumed: a second call
        without a new training forward is a no-op). Returns the new tau."""
        if H is None:
            H, self._last_H = self._last_H, None
        if H is None:
            return self.temperature
        if H < self._target_H:
            self.temperature = min(self.temperature * 1.005, self.tau_max)
        else:
            self.temperature = max(self.temperature * 0.999, self.tau_min)
        return self.temperature


# ─────────────────────────────────────────────────────────────────────────────
# Legacy ABR layer (exact original behaviour; legacy=True)
# ─────────────────────────────────────────────────────────────────────────────

class LegacyABRLayer(nn.Module):
    """Original ABRLayer, unchanged maths (bugs B1-B3 and B5 included)."""
    def __init__(self, d_model, n_heads, block_size, use_cross_block_state=True,
                 use_gated_cbsc=False, ffn_mult=4):
        super().__init__()
        self.block_size            = block_size
        self.use_cross_block_state = use_cross_block_state
        self.use_gated_cbsc        = use_gated_cbsc
        self.router      = LegacyGumbelRouter(d_model)
        self.transformer = TransformerBlock(d_model, n_heads, ffn_mult)
        self.mamba       = MambaBlock(d_model)
        self.last_routing = None
        self.last_indices = None

        if use_gated_cbsc:
            # Learned per-dimension gate on the CBSC injection.
            # Zero-init weight + zero bias -> sigmoid(0)=0.5 at step 0.
            self.cbsc_gate = nn.Linear(d_model, d_model, bias=True)
            nn.init.zeros_(self.cbsc_gate.weight)
            nn.init.zeros_(self.cbsc_gate.bias)

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

        z_soft_all = torch.stack(z_list, dim=1)
        indices    = torch.stack(idx_list, dim=1)
        self.last_routing = z_soft_all
        self.last_indices = indices

        if self.use_cross_block_state:
            out = self._forward_cbsc(blocks, indices, B, N, b, D)
        else:
            out = self._forward_batched(blocks, indices, B, N, b, D)

        return out.view(B, Tp, D)[:, :T, :], z_soft_all, indices

    def _forward_cbsc(self, blocks, indices, B, N, b, D):
        flat   = blocks.view(B * N, b, D)
        r_flat = indices.view(B * N)

        y_flat = flat.clone()
        t_mask = (r_flat == 0)
        m_mask = (r_flat == 1)
        if t_mask.any():
            y_flat[t_mask] = self.transformer(flat[t_mask])
        if m_mask.any():
            y_flat[m_mask] = self.mamba(flat[m_mask])

        m_route    = (indices == 1)
        prev_mamba = torch.zeros_like(m_route)
        prev_mamba[:, 1:] = m_route[:, :-1]
        cbsc_mask  = m_route & prev_mamba

        if cbsc_mask.any():
            last_tok  = y_flat.view(B, N, b, D)[:, :, -1, :]
            prev_last = torch.zeros_like(last_tok)
            prev_last[:, 1:, :] = last_tok[:, :-1, :]

            inp2     = flat.view(B, N, b, D).clone()
            cbsc_exp = cbsc_mask.unsqueeze(-1).unsqueeze(-1)

            if self.use_gated_cbsc:
                gate      = torch.sigmoid(self.cbsc_gate(prev_last))
                injection = (gate * prev_last).unsqueeze(2)
            else:
                injection = prev_last.unsqueeze(2)

            inp2[:, :, :1, :] = inp2[:, :, :1, :] + cbsc_exp * injection

            cbsc_flat = cbsc_mask.view(B * N)
            inp2_flat = inp2.view(B * N, b, D)
            y_flat2   = y_flat.clone()
            y_flat2[cbsc_flat] = self.mamba(inp2_flat[cbsc_flat])
            return y_flat2.view(B, N * b, D)

        return y_flat.view(B, N * b, D)

    def _forward_batched(self, blocks, indices, B, N, b, D):
        flat       = blocks.view(B * N, b, D)
        t_out      = self.transformer(flat)
        m_out      = self.mamba(flat)
        route_flat = indices.view(B * N)
        sel        = route_flat.float().unsqueeze(-1).unsqueeze(-1)
        out        = (1 - sel) * t_out + sel * m_out
        return out.view(B, N * b, D)


# ─────────────────────────────────────────────────────────────────────────────
# Fixed ABR layer
# ─────────────────────────────────────────────────────────────────────────────

CBSC_MODES    = ("none", "state", "gated_state", "legacy_inject")
ROUTING_MODES = ("learned", "random", "alternate")


class ABRLayer(nn.Module):
    """Adaptive Block Routing layer (fixed version).

    The sequence (B, T, D) is cut into N blocks of size b. Each block goes to
    the Transformer expert (index 0) or the Mamba expert (index 1).

    routing_mode
      "learned"   : Gumbel router. The chosen expert's block output is multiplied
                    by the straight-through weight z[..., chosen] (== 1 in forward,
                    gradient flows into the router; P1.1).
      "random"    : each block -> attention with prob attn_frac (no gradient).
      "alternate" : blocks T, M, T, M, ...
    router_input (causality): the router only sees information available at the
      block's first position, so routing never depends on future tokens.
      "first_token"     = layer input at the block's first token (default);
      "prev_block_mean" = mean of the previous block (zeros for block 0).
    Transformer expert (P1.2): queries from routed blocks, keys/values from ALL
      earlier tokens of the layer input. attn_impl "gather" (efficient) or
      "reference" (full causal attention, then select).
    cbsc_mode (P1.3)
      "none"          : each Mamba block starts from zero state.
      "state"         : per sample, the Mamba-routed blocks are concatenated in
                        order (right-padded) and run as one stream, so state flows
                        from each Mamba block to the next regardless of the
                        Transformer blocks in between.
      "gated_state"   : block-by-block loop; the recurrent state handed to the next
                        Mamba block is sigmoid(W_g h + b_g) * h, b_g init +4.
      "legacy_inject" : old behaviour (adjacent M->M blocks only, previous block's
                        last output vector added to the next block's first input).

    Test hook: set `force_routing` to a LongTensor of shape (N,) or (B, N) to fix
    the routing decision (the STE weight / gradient path is kept).
    """
    def __init__(self, d_model, n_heads, block_size, cbsc_mode="state",
                 routing_mode="learned", attn_frac=0.5, ffn_mult=4,
                 router_input="first_token", attn_impl="gather",
                 mamba_version=1, mamba_backend=None, gate_bias_init=4.0):
        super().__init__()
        if cbsc_mode not in CBSC_MODES:
            raise ValueError(f"cbsc_mode must be one of {CBSC_MODES}, got {cbsc_mode!r}")
        if routing_mode not in ROUTING_MODES:
            raise ValueError(f"routing_mode must be one of {ROUTING_MODES}, got {routing_mode!r}")
        if router_input not in ("first_token", "prev_block_mean"):
            raise ValueError(f"bad router_input {router_input!r}")
        if attn_impl not in ("gather", "reference"):
            raise ValueError(f"bad attn_impl {attn_impl!r}")
        self.block_size   = block_size
        self.cbsc_mode    = cbsc_mode
        self.routing_mode = routing_mode
        self.attn_frac    = attn_frac
        self.router_input = router_input
        self.attn_impl    = attn_impl
        self.router = GumbelRouter(d_model) if routing_mode == "learned" else None
        self.transformer = TransformerBlock(d_model, n_heads, ffn_mult)
        self.mamba = MambaExpert(d_model, backend=mamba_backend, mamba_version=mamba_version)
        if cbsc_mode == "gated_state":
            self.cbsc_gate = self.mamba.make_state_gate(gate_bias_init)
        self.force_routing = None
        self.last_routing = None
        self.last_indices = None

    # ── routing ──────────────────────────────────────────────────────────────
    def _route(self, blocks):
        B, N, b, D = blocks.shape
        dev = blocks.device
        forced = None
        if self.force_routing is not None:
            forced = torch.as_tensor(self.force_routing, device=dev).long()
            forced = forced.expand(B, N) if forced.dim() == 1 else forced
        if self.routing_mode == "learned":
            if self.router_input == "first_token":
                summary = blocks[:, :, 0, :]
            else:
                summary = torch.cat([blocks.new_zeros(B, 1, D),
                                     blocks[:, :-1].mean(dim=2)], dim=1)
            z, z_soft, idx = self.router(summary, forced)
            w = z.gather(-1, idx.unsqueeze(-1)).squeeze(-1)          # (B, N), == 1
            return idx, w, z_soft
        if forced is not None:
            idx = forced
        elif self.routing_mode == "random":
            idx = (torch.rand(B, N, device=dev) >= self.attn_frac).long()
        else:  # alternate: T, M, T, M, ...
            idx = (torch.arange(N, device=dev) % 2).expand(B, N)
        z_soft = F.one_hot(idx, 2).to(blocks.dtype)
        return idx, None, z_soft

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, x):
        """x: (B, T, D) -> (out (B, T, D), z_soft (B, N, E), indices (B, N))."""
        B, T, D = x.shape
        b = self.block_size
        pad = (b - T % b) % b
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        Tp = x.shape[1]
        N = Tp // b
        blocks = x.view(B, N, b, D)

        idx, w, z_soft = self._route(blocks)
        self.last_routing = z_soft
        self.last_indices = idx.detach()
        t_blk, m_blk = idx == 0, idx == 1

        out = blocks.new_zeros(B, N, b, D)
        if bool(t_blk.any()):
            tok_mask = t_blk.repeat_interleave(b, dim=1)              # (B, Tp)
            if self.attn_impl == "reference":
                yt = self.transformer.forward_routed_reference(x, tok_mask)
            else:
                yt = self.transformer.forward_routed(x, tok_mask)
            out = out + yt.view(B, N, b, D)
        if bool(m_blk.any()):
            out = out + self._mamba_path(blocks, m_blk)

        if w is not None:
            out = out * w.view(B, N, 1, 1)
        return out.view(B, Tp, D)[:, :T, :], z_soft, idx

    # ── Mamba expert paths ───────────────────────────────────────────────────
    def _mamba_path(self, blocks, m_blk):
        """Returns (B, N, b, D) with Mamba outputs at Mamba-routed blocks, 0 elsewhere."""
        B, N, b, D = blocks.shape
        mode = self.cbsc_mode
        if mode == "none":
            out = blocks.new_zeros(B, N, b, D)
            y, _ = self.mamba(blocks[m_blk])                          # (Nm, b, D)
            out[m_blk] = y
            return out
        if mode == "legacy_inject":
            return self._mamba_legacy_inject(blocks, m_blk)

        # Pack Mamba blocks of each sample to the front, in order (stable sort).
        counts = m_blk.sum(1)                                         # (B,)
        C = int(counts.max().item())
        order = torch.sort((~m_blk).to(torch.int8), dim=1, stable=True).indices[:, :C]
        valid = torch.arange(C, device=blocks.device).view(1, C) < counts.view(B, 1)
        gidx = order.view(B, C, 1, 1).expand(B, C, b, D)
        packed = blocks.gather(1, gidx) * valid.view(B, C, 1, 1).to(blocks.dtype)

        if mode == "state":
            # Right padding is causal-safe: real blocks never see the pad blocks.
            y, _ = self.mamba(packed.reshape(B, C * b, D))
            y = y.view(B, C, b, D)
        else:  # gated_state: explicit block-by-block state passing
            ys, state = [], None
            for j in range(C):
                if state is not None:
                    conv_s, rec_s = state
                    state = (conv_s, self.cbsc_gate(rec_s))
                yj, state = self.mamba(packed[:, j], state)
                ys.append(yj)
            y = torch.stack(ys, dim=1)
        y = y * valid.view(B, C, 1, 1).to(y.dtype)
        return blocks.new_zeros(B, N, b, D).scatter_add(1, gidx, y)

    def _mamba_legacy_inject(self, blocks, m_blk):
        """Old CBSC: pass 1 isolated blocks; for adjacent M->M pairs add the previous
        block's pass-1 last output to the next block's first input, re-run."""
        B, N, b, D = blocks.shape
        y1 = blocks.new_zeros(B, N, b, D)
        y1[m_blk] = self.mamba(blocks[m_blk])[0]
        prev_m = torch.zeros_like(m_blk)
        prev_m[:, 1:] = m_blk[:, :-1]
        cbsc = m_blk & prev_m
        if not bool(cbsc.any()):
            return y1
        prev_last = torch.cat([blocks.new_zeros(B, 1, D), y1[:, :-1, -1, :]], dim=1)
        inj = torch.zeros_like(blocks)
        inj[:, :, 0, :] = prev_last * cbsc.unsqueeze(-1).to(blocks.dtype)
        inp2 = blocks + inj
        y2 = y1.clone()
        y2[cbsc] = self.mamba(inp2[cbsc])[0]
        return y2


# ─────────────────────────────────────────────────────────────────────────────
# Full Models
# ─────────────────────────────────────────────────────────────────────────────

class BaseModel(nn.Module):
    """Shared embedding + positional + LM head (weight-tied).
    Note: _init_weights runs before subclasses create their layers, so it only
    touches the embedding/head (original behaviour, kept for reproducibility)."""
    def __init__(self, vocab_size, d_model, max_seq_len=1024):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.norm    = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
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

    def backbone(self, h):
        """Hidden states (B, T, D) after embedding -> hidden states before final norm."""
        for layer in self.layers:
            h = layer(h)
        return h

    def forward(self, x):
        return self.decode(self.backbone(self.encode(x)))


class TransformerLM(BaseModel):
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=4,
                 ffn_mult=4, max_seq_len=1024):
        super().__init__(vocab_size, d_model, max_seq_len)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_mult) for _ in range(n_layers)])


class MambaLM(BaseModel):
    def __init__(self, vocab_size, d_model=256, n_layers=4, max_seq_len=1024):
        super().__init__(vocab_size, d_model, max_seq_len)
        self.layers = nn.ModuleList([MambaBlock(d_model) for _ in range(n_layers)])


JAMBA_PATTERNS = {"jamba-1:1": 2, "jamba-1:3": 4}   # attention every k-th layer


class JambaLM(BaseModel):
    """
    Jamba-style fixed interleaved Transformer-Mamba model (no routing, no CBSC).
    pattern "jamba-1:1": T-M-T-M ...   pattern "jamba-1:3": T-M-M-M ...
    """
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=4,
                 ffn_mult=4, max_seq_len=1024, pattern="jamba-1:1"):
        super().__init__(vocab_size, d_model, max_seq_len)
        if pattern not in JAMBA_PATTERNS:
            raise ValueError(f"pattern must be one of {list(JAMBA_PATTERNS)}, got {pattern!r}")
        self.pattern = pattern
        k = JAMBA_PATTERNS[pattern]
        layers = []
        for i in range(n_layers):
            if i % k == 0:
                layers.append(TransformerBlock(d_model, n_heads, ffn_mult))
            else:
                layers.append(MambaBlock(d_model))
        self.layers = nn.ModuleList(layers)


class ABRLM(BaseModel):
    """
    Multi-layer Adaptive Block Routing Language Model.

    legacy=True  : original code (LegacyABRLayer); use_cross_block_state /
                   use_gated_cbsc select the old variants.
    legacy=False : fixed ABRLayer; cbsc_mode / routing_mode as in ABRLayer. If
                   cbsc_mode is None it is derived from the old flags
                   (gated -> "gated_state", cbsc -> "state", else "none").
    mamba_version (real backend only): None -> 2 for gated_state (state passing
                   needs Mamba-2), else 1.
    """
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=2,
                 block_size=8, use_cross_block_state=True, use_gated_cbsc=False,
                 ffn_mult=4, max_seq_len=1024, *, legacy=False, cbsc_mode=None,
                 routing_mode="learned", attn_frac=0.5, router_input="first_token",
                 attn_impl="gather", mamba_version=None, mamba_backend=None,
                 gate_bias_init=4.0):
        super().__init__(vocab_size, d_model, max_seq_len)
        self.legacy = legacy
        if legacy:
            self.cbsc_mode, self.routing_mode = "legacy", "learned"
            self.layers = nn.ModuleList([
                LegacyABRLayer(d_model, n_heads, block_size,
                               use_cross_block_state, use_gated_cbsc, ffn_mult)
                for _ in range(n_layers)])
        else:
            if cbsc_mode is None:
                cbsc_mode = ("gated_state" if use_gated_cbsc else
                             "state" if use_cross_block_state else "none")
            if mamba_version is None:
                mamba_version = 2 if cbsc_mode == "gated_state" else 1
            self.cbsc_mode, self.routing_mode = cbsc_mode, routing_mode
            self.layers = nn.ModuleList([
                ABRLayer(d_model, n_heads, block_size, cbsc_mode=cbsc_mode,
                         routing_mode=routing_mode, attn_frac=attn_frac,
                         ffn_mult=ffn_mult, router_input=router_input,
                         attn_impl=attn_impl, mamba_version=mamba_version,
                         mamba_backend=mamba_backend, gate_bias_init=gate_bias_init)
                for _ in range(n_layers)])
        self.last_routing  = None
        self.routing_probs = []

    def backbone(self, h):
        probs = []
        for layer in self.layers:
            h, z_soft, _ = layer(h)
            probs.append(z_soft)
        self.routing_probs = probs                      # NOT detached (aux losses)
        self.last_routing = probs[-1].detach() if probs else None
        return h

    def routers(self):
        return [l.router for l in self.layers if getattr(l, "router", None) is not None]

    def update_router_temperatures(self):
        """Call once per optimizer step. Returns the per-layer taus. No-op for
        legacy (it updates inside forward) and for random/alternate routing."""
        if self.legacy:
            return [r.temperature for r in self.routers()]
        return [r.update_temperature() for r in self.routers()]

    @staticmethod
    def _stats(z, idx):
        avg = z.mean(dim=(0, 1))
        H = -(avg * (avg + 1e-9).log()).sum().item()
        Hb = -(z * (z + 1e-9).log()).sum(-1).mean().item()
        s = {"transformer_pct": z[..., 0].mean().item(),
             "mamba_pct":       z[..., 1].mean().item(),
             "entropy":         H,
             "block_entropy":   Hb}
        if idx is not None:
            s["transformer_frac_hard"] = (idx == 0).float().mean().item()
        return s

    def get_routing_stats(self):
        """Overall stats over ALL layers (soft-prob means; entropy of the overall
        average distribution; hard attention fraction), plus "per_layer" (list of
        dicts with the same keys, "layer" and "tau") and "tau" (list per layer).
        NB: the old code reported the last layer only; that is per_layer[-1]."""
        zs = [l.last_routing for l in self.layers if l.last_routing is not None]
        if not zs:
            return {}
        per_layer = []
        for i, l in enumerate(self.layers):
            if l.last_routing is None:
                continue
            s = self._stats(l.last_routing.detach(), l.last_indices)
            s["layer"] = i
            r = getattr(l, "router", None)
            s["tau"] = r.temperature if r is not None else None
            per_layer.append(s)
        z_all = torch.cat([z.detach().reshape(-1, z.shape[-1]) for z in zs], 0).unsqueeze(0)
        idx_all = None
        if all(l.last_indices is not None for l in self.layers):
            idx_all = torch.cat([l.last_indices.reshape(-1) for l in self.layers]).unsqueeze(0)
        out = self._stats(z_all, idx_all)
        out["per_layer"] = per_layer
        out["tau"] = [s["tau"] for s in per_layer]
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

_ARCH_KEYS = ("d_model", "n_heads", "block_size", "ffn_mult", "max_seq_len")
_ABR_KEYS  = ("router_input", "attn_impl", "mamba_version", "gate_bias_init")


def build_model(cfg, vocab_size, d_model=256, n_heads=8, block_size=8,
                ffn_mult=4, max_seq_len=1024, **kw):
    """Build a model from a config entry.

    cfg may be
      * the legacy tuple (label, n_layers, use_routing, use_cbsc, expert_type) with
        expert_type in {"transformer","mamba","jamba","hybrid","hybrid-gated"}.
        Built with the FIXED code by default ("hybrid" -> cbsc "state"/"none",
        "hybrid-gated" -> "gated_state"); pass legacy=True to reproduce the
        original code exactly.
      * a dict:
        {"label": str, "n_layers": int (default 4),
         "expert_type": "transformer"|"mamba"|"jamba-1:1"|"jamba-1:3"|"abr" (default "abr"),
         "routing_mode": "learned"|"random"|"alternate" (default "learned"),
         "cbsc_mode": "none"|"state"|"gated_state"|"legacy_inject" (default "state"),
         "legacy": bool (default False), "attn_frac": float (default 0.5)}
        Optional: d_model, n_heads, block_size, ffn_mult, max_seq_len (override the
        keyword args), router_input, attn_impl, mamba_version, gate_bias_init.
        "jamba" is accepted as "jamba-1:1"; "hybrid"/"hybrid-gated" as "abr".
        With legacy=True: cbsc_mode "none" -> old state=OFF, "state" -> old
        state=ON, "gated_state" -> old hybrid-gated. Unknown keys are ignored.
    Extra keyword args (n_layers, expert_type, routing_mode, cbsc_mode, legacy,
    attn_frac, router_input, ...) are defaults; dict values override them.
    """
    arch = dict(d_model=d_model, n_heads=n_heads, block_size=block_size,
                ffn_mult=ffn_mult, max_seq_len=max_seq_len)
    opts = dict(n_layers=4, expert_type="abr", routing_mode="learned",
                cbsc_mode="state", legacy=False, attn_frac=0.5)
    abr_extra = {}
    for k, v in kw.items():
        if k in opts:
            opts[k] = v
        elif k in _ABR_KEYS:
            abr_extra[k] = v
        else:
            raise TypeError(f"build_model got unexpected keyword {k!r}")

    if isinstance(cfg, (tuple, list)):
        label, n_layers, use_routing, use_cbsc, expert_type = cfg
        opts["n_layers"] = n_layers
        if expert_type == "jamba":
            opts["expert_type"] = "jamba-1:1"
        elif expert_type in ("hybrid", "hybrid-gated"):
            opts["expert_type"] = "abr"
            opts["cbsc_mode"] = ("gated_state" if expert_type == "hybrid-gated" else
                                 "state" if use_cbsc else "none")
        else:
            opts["expert_type"] = expert_type
    elif isinstance(cfg, dict):
        for k, v in cfg.items():
            if k in opts:
                opts[k] = v
            elif k in _ARCH_KEYS:
                arch[k] = v
            elif k in _ABR_KEYS:
                abr_extra[k] = v
        et = opts["expert_type"]
        if et == "jamba":
            opts["expert_type"] = "jamba-1:1"
        elif et in ("hybrid", "hybrid-gated"):
            opts["expert_type"] = "abr"
            if et == "hybrid-gated" and "cbsc_mode" not in cfg:
                opts["cbsc_mode"] = "gated_state"
    else:
        raise TypeError(f"cfg must be a tuple or dict, got {type(cfg).__name__}")

    et = opts["expert_type"]
    n_layers = opts["n_layers"]
    if et == "transformer":
        return TransformerLM(vocab_size, arch["d_model"], arch["n_heads"], n_layers,
                             arch["ffn_mult"], arch["max_seq_len"])
    if et == "mamba":
        return MambaLM(vocab_size, arch["d_model"], n_layers, arch["max_seq_len"])
    if et in JAMBA_PATTERNS:
        return JambaLM(vocab_size, arch["d_model"], arch["n_heads"], n_layers,
                       arch["ffn_mult"], arch["max_seq_len"], pattern=et)
    if et != "abr":
        raise ValueError(f"unknown expert_type {et!r}")

    cbsc = opts["cbsc_mode"]
    if cbsc not in CBSC_MODES:
        raise ValueError(f"cbsc_mode must be one of {CBSC_MODES}, got {cbsc!r}")
    if opts["routing_mode"] not in ROUTING_MODES:
        raise ValueError(f"routing_mode must be one of {ROUTING_MODES}")
    if opts["legacy"]:
        if opts["routing_mode"] != "learned":
            raise ValueError("legacy=True only supports routing_mode='learned'")
        if cbsc == "legacy_inject":
            raise ValueError("use legacy=True OR cbsc_mode='legacy_inject', not both")
        return ABRLM(vocab_size, arch["d_model"], arch["n_heads"], n_layers,
                     arch["block_size"], use_cross_block_state=(cbsc != "none"),
                     use_gated_cbsc=(cbsc == "gated_state"), ffn_mult=arch["ffn_mult"],
                     max_seq_len=arch["max_seq_len"], legacy=True)
    return ABRLM(vocab_size, arch["d_model"], arch["n_heads"], n_layers,
                 arch["block_size"], ffn_mult=arch["ffn_mult"],
                 max_seq_len=arch["max_seq_len"], legacy=False, cbsc_mode=cbsc,
                 routing_mode=opts["routing_mode"], attn_frac=opts["attn_frac"],
                 **abr_extra)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
