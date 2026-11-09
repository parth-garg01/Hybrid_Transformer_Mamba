"""Phase 1 model tests (P1.1-P1.4). CPU only, GRU stand-in for Mamba, tiny models.

Run: ~/.venvs/abr/bin/python -m pytest tests/test_models.py -q
"""
import math
import types

import pytest
import torch
import torch.nn.functional as F

import models_scaled as ms
from models_scaled import build_model, count_params

torch.set_num_threads(4)

VOCAB, D, H, BS, T = 50, 32, 4, 8, 64
ARCH = dict(d_model=D, n_heads=H, block_size=BS, ffn_mult=2, max_seq_len=128)

# ─────────────────────────────────────────────────────────────────────────────
# Reference: the ORIGINAL code, copied verbatim from
#   git show fixes:experiments/full_scale/models_scaled.py   (lines 6-381)
# Executed into its own namespace so it cannot be affected by edits to the module.
# ─────────────────────────────────────────────────────────────────────────────
ORIG_SRC = r'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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


# ─────────────────────────────────────────────────────────────────────────────
# Single ABR Routing Layer
# ─────────────────────────────────────────────────────────────────────────────

class ABRLayer(nn.Module):
    def __init__(self, d_model, n_heads, block_size, use_cross_block_state=True,
                 use_gated_cbsc=False, ffn_mult=4):
        super().__init__()
        self.block_size            = block_size
        self.use_cross_block_state = use_cross_block_state
        self.use_gated_cbsc        = use_gated_cbsc
        self.router      = GumbelRouter(d_model)
        self.transformer = TransformerBlock(d_model, n_heads, ffn_mult)
        self.mamba       = MambaBlock(d_model)
        self.last_routing = None

        if use_gated_cbsc:
            # Learned per-dimension gate on the CBSC injection.
            # Zero-init weight + zero bias → sigmoid(0)=0.5 at step 0,
            # letting the model learn from a neutral starting point rather
            # than a fixed full-injection (additive) or no-injection (zero).
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

        if self.use_cross_block_state:
            out = self._forward_cbsc(blocks, indices, B, N, b, D)
        else:
            out = self._forward_batched(blocks, indices, B, N, b, D)

        return out.view(B, Tp, D)[:, :T, :], z_soft_all, indices

    def _forward_cbsc(self, blocks, indices, B, N, b, D):
        """
        Two-pass vectorised CBSC.

        Pass 1: run every block through its assigned expert in batch.
        Pass 2: for consecutive Mamba→Mamba pairs, inject the predecessor's
                last-token hidden vector into position 0 of the successor's
                input (optionally through a learned sigmoid gate), then
                re-run Mamba for those blocks only.
        """
        flat   = blocks.view(B * N, b, D)
        r_flat = indices.view(B * N)

        # ── Pass 1 ───────────────────────────────────────────────────────────
        y_flat = flat.clone()
        t_mask = (r_flat == 0)
        m_mask = (r_flat == 1)
        if t_mask.any():
            y_flat[t_mask] = self.transformer(flat[t_mask])
        if m_mask.any():
            y_flat[m_mask] = self.mamba(flat[m_mask])

        # ── Identify consecutive Mamba→Mamba pairs ───────────────────────────
        m_route    = (indices == 1)
        prev_mamba = torch.zeros_like(m_route)
        prev_mamba[:, 1:] = m_route[:, :-1]
        cbsc_mask  = m_route & prev_mamba

        if cbsc_mask.any():
            last_tok  = y_flat.view(B, N, b, D)[:, :, -1, :]   # (B, N, D)
            prev_last = torch.zeros_like(last_tok)
            prev_last[:, 1:, :] = last_tok[:, :-1, :]           # shifted by 1

            inp2     = flat.view(B, N, b, D).clone()
            cbsc_exp = cbsc_mask.unsqueeze(-1).unsqueeze(-1)     # (B, N, 1, 1)

            if self.use_gated_cbsc:
                # Per-dimension sigmoid gate learned from the predecessor state.
                # Allows the model to selectively pass or block each feature.
                gate      = torch.sigmoid(self.cbsc_gate(prev_last))   # (B, N, D)
                injection = (gate * prev_last).unsqueeze(2)            # (B, N, 1, D)
            else:
                injection = prev_last.unsqueeze(2)                     # (B, N, 1, D)

            inp2[:, :, :1, :] = inp2[:, :, :1, :] + cbsc_exp * injection

            # ── Pass 2: re-run only CBSC blocks ──────────────────────────────
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
# Full Models
# ─────────────────────────────────────────────────────────────────────────────

class BaseModel(nn.Module):
    """Shared embedding + positional + LM head (weight-tied)."""
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


class TransformerLM(BaseModel):
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=4,
                 ffn_mult=4, max_seq_len=1024):
        super().__init__(vocab_size, d_model, max_seq_len)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_mult) for _ in range(n_layers)])

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


class JambaLM(BaseModel):
    """
    Jamba-style fixed interleaved Transformer-Mamba model.
    Layer pattern for n_layers=4: T-M-T-M (no routing, no CBSC).
    Serves as a strong hybrid baseline: shows whether dynamic routing
    provides any benefit over fixed interleaving.
    """
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=4,
                 ffn_mult=4, max_seq_len=1024):
        super().__init__(vocab_size, d_model, max_seq_len)
        layers = []
        for i in range(n_layers):
            if i % 2 == 0:
                layers.append(TransformerBlock(d_model, n_heads, ffn_mult))
            else:
                layers.append(MambaBlock(d_model))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        h = self.encode(x)
        for layer in self.layers:
            h = layer(h)
        return self.decode(h)


class ABRLM(BaseModel):
    """
    Multi-layer Adaptive Block Routing Language Model.
    use_cross_block_state : additive CBSC on/off
    use_gated_cbsc        : replace additive injection with learned sigmoid gate
    """
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=2,
                 block_size=8, use_cross_block_state=True, use_gated_cbsc=False,
                 ffn_mult=4, max_seq_len=1024):
        super().__init__(vocab_size, d_model, max_seq_len)
        self.layers = nn.ModuleList([
            ABRLayer(d_model, n_heads, block_size,
                     use_cross_block_state, use_gated_cbsc, ffn_mult)
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
        z     = self.last_routing
        avg   = z.mean(dim=(0, 1))
        H     = -(avg * (avg + 1e-9).log()).sum().item()
        return {
            "transformer_pct": z[..., 0].mean().item(),
            "mamba_pct":        z[..., 1].mean().item(),
            "entropy":          H,
        }
'''


def _orig_namespace():
    ns = {"__name__": "orig_models_scaled"}
    exec(compile(ORIG_SRC, "orig_models_scaled", "exec"), ns)
    return ns


ORIG = _orig_namespace()


def orig_build_model(cfg_tuple, vocab_size, d_model, n_heads, block_size, ffn_mult,
                     max_seq_len):
    """Original build_model (copied from the same commit)."""
    label, n_layers, use_routing, use_cbsc, expert_type = cfg_tuple
    O = ORIG
    if expert_type == "transformer":
        return O["TransformerLM"](vocab_size, d_model, n_heads, n_layers, ffn_mult, max_seq_len)
    elif expert_type == "mamba":
        return O["MambaLM"](vocab_size, d_model, n_layers, max_seq_len)
    elif expert_type == "jamba":
        return O["JambaLM"](vocab_size, d_model, n_heads, n_layers, ffn_mult, max_seq_len)
    elif expert_type == "hybrid-gated":
        return O["ABRLM"](vocab_size, d_model, n_heads, n_layers, block_size,
                          use_cross_block_state=True, use_gated_cbsc=True,
                          ffn_mult=ffn_mult, max_seq_len=max_seq_len)
    else:
        return O["ABRLM"](vocab_size, d_model, n_heads, n_layers, block_size,
                          use_cross_block_state=use_cbsc, use_gated_cbsc=False,
                          ffn_mult=ffn_mult, max_seq_len=max_seq_len)


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def make(seed=0, **cfg):
    torch.manual_seed(seed)
    c = {"n_layers": 2, "expert_type": "abr"}
    c.update(cfg)
    return build_model(c, VOCAB, **ARCH)


def tokens(B=3, T_=T, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (B, T_), generator=g)


def ce_loss(model, x):
    logits = model(x)
    return F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), x[:, 1:].reshape(-1))


def abr_layers(model):
    return [l for l in model.layers if isinstance(l, (ms.ABRLayer, ms.LegacyABRLayer))]


NEW_CBSC = ["none", "state", "gated_state"]

LEAK_VARIANTS = {
    "abr-none":            dict(cbsc_mode="none"),
    "abr-state":           dict(cbsc_mode="state"),
    "abr-gated":           dict(cbsc_mode="gated_state"),
    "abr-legacy_inject":   dict(cbsc_mode="legacy_inject"),
    "abr-state-prevmean":  dict(cbsc_mode="state", router_input="prev_block_mean"),
    "abr-state-refattn":   dict(cbsc_mode="state", attn_impl="reference"),
    "random-state":        dict(cbsc_mode="state", routing_mode="random"),
    "random-none":         dict(cbsc_mode="none", routing_mode="random"),
    "alternate-state":     dict(cbsc_mode="state", routing_mode="alternate"),
    "alternate-gated":     dict(cbsc_mode="gated_state", routing_mode="alternate"),
    "legacy-off":          dict(legacy=True, cbsc_mode="none"),
    "legacy-on":           dict(legacy=True, cbsc_mode="state"),
    "legacy-gated":        dict(legacy=True, cbsc_mode="gated_state"),
    "jamba-1:1":           dict(expert_type="jamba-1:1", n_layers=4),
    "jamba-1:3":           dict(expert_type="jamba-1:3", n_layers=4),
    "transformer":         dict(expert_type="transformer"),
    "mamba":               dict(expert_type="mamba"),
}


# ─────────────────────────────────────────────────────────────────────────────
# P1.1 router gradient
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cbsc", NEW_CBSC)
def test_router_receives_gradient(cbsc):
    model = make(cbsc_mode=cbsc, routing_mode="learned")
    model.train()
    ce_loss(model, tokens()).backward()
    for i, layer in enumerate(model.layers):
        g = layer.router.linear.weight.grad
        assert g is not None, f"layer {i}: router grad is None"
        assert g.abs().sum().item() > 0, f"layer {i}: router grad is zero"


def test_legacy_router_has_no_gradient():
    """Documents bug B1 in legacy mode (kept on purpose for reproducibility)."""
    model = make(legacy=True, cbsc_mode="state")
    model.train()
    ce_loss(model, tokens()).backward()
    for layer in model.layers:
        g = layer.router.linear.weight.grad
        assert g is None or g.abs().sum().item() == 0


@pytest.mark.parametrize("cbsc", NEW_CBSC + ["legacy_inject"])
@pytest.mark.parametrize("train", [False, True])
def test_hard_routing_forward_unchanged(cbsc, train):
    """Multiplying by the STE weight z[..., chosen] does not change the forward."""
    model = make(cbsc_mode=cbsc)
    model.train(train)
    x = tokens()
    torch.manual_seed(5)
    y_ste = model(x)
    for layer in model.layers:
        orig = layer._route

        def hard(self, blocks, _orig=orig):
            idx, w, z_soft = _orig(blocks)
            return idx, None, z_soft          # pure hard routing, no multiply
        layer._route = types.MethodType(hard, layer)
    torch.manual_seed(5)
    y_hard = model(x)
    assert torch.equal(y_ste, y_hard)


# ─────────────────────────────────────────────────────────────────────────────
# P1.2 causality and cross-block context
# ─────────────────────────────────────────────────────────────────────────────

def _check_no_leak(model, T_):
    x = tokens(B=2, T_=T_, seed=3)
    h = model.encode(x).detach().requires_grad_(True)
    out = model.decode(model.backbone(h))
    for t in range(T_):
        (g,) = torch.autograd.grad(out[:, t].sum(), h, retain_graph=True)
        fut = g[:, t + 1:].abs().max().item() if t + 1 < T_ else 0.0
        assert fut == 0.0, f"position {t} has gradient {fut} from the future"
        assert g[:, t].abs().max().item() > 0


@pytest.mark.parametrize("name", list(LEAK_VARIANTS))
def test_no_future_leakage(name):
    model = make(**LEAK_VARIANTS[name])
    model.train()                     # Gumbel noise / random routing active
    _check_no_leak(model, T)


@pytest.mark.parametrize("name", ["abr-state", "abr-gated", "legacy-on"])
def test_no_future_leakage_ragged_length(name):
    """T not a multiple of block_size (right padding inside the layer)."""
    model = make(**LEAK_VARIANTS[name])
    model.eval()
    _check_no_leak(model, T - 3)


def _layer_input_grad(layer, h, out_pos):
    out, _, _ = layer(h)
    (g,) = torch.autograd.grad(out[:, out_pos].sum(), h)
    return g


@pytest.mark.parametrize("cbsc", NEW_CBSC)
def test_cross_block_dependency(cbsc):
    """20 random routing patterns (forced). Within ONE layer, the last token of block
    k>0 depends on block 0 exactly when the semantics say it should:
      * block k is Transformer-routed (attends to all earlier tokens), or
      * block k and block 0 are both Mamba-routed and CBSC carries state.
    Otherwise the gradient is exactly zero (no hidden path)."""
    torch.manual_seed(0)
    layer = ms.ABRLayer(D, H, BS, cbsc_mode=cbsc, ffn_mult=2)
    layer.train()
    N = T // BS
    gen = torch.Generator().manual_seed(123)
    n_dep = 0
    for _ in range(20):
        pat = torch.randint(0, 2, (N,), generator=gen)
        layer.force_routing = pat
        h = torch.randn(2, T, D, requires_grad=True)
        for k in range(1, N):
            g = _layer_input_grad(layer, h, (k + 1) * BS - 1)
            dep0 = g[:, :BS].abs().max().item()
            expect = bool(pat[k] == 0) or (bool(pat[k] == 1) and bool(pat[0] == 1)
                                            and cbsc != "none")
            if expect:
                assert dep0 > 0, f"pattern {pat.tolist()} block {k}: no dependency on block 0"
                n_dep += 1
            else:
                assert dep0 == 0.0, f"pattern {pat.tolist()} block {k}: unexpected path"
    assert n_dep > 0


def test_cross_block_dependency_full_model():
    """In the full 2-layer model with learned routing, a token in the last block
    depends on block 0 for 20 random forced patterns that contain a Transformer
    block after block 0 (PRD P1.2 acceptance)."""
    model = make(cbsc_mode="state")
    model.train()
    N = T // BS
    gen = torch.Generator().manual_seed(7)
    done = 0
    while done < 20:
        pats = [torch.randint(0, 2, (N,), generator=gen) for _ in model.layers]
        if not any(bool((p[1:] == 0).any()) for p in pats):
            continue
        for l, p in zip(model.layers, pats):
            l.force_routing = p
        x = tokens(B=2, seed=done)
        h = model.encode(x).detach().requires_grad_(True)
        out = model.decode(model.backbone(h))
        # PRD statement: SOME token in a block k>0 depends on block 0.
        dep_any = False
        for k in range(1, N):
            (gk,) = torch.autograd.grad(out[:, (k + 1) * BS - 1].sum(), h, retain_graph=True)
            if gk[:, :BS].abs().max().item() > 0:
                dep_any = True
                break
        assert dep_any, f"patterns {[p.tolist() for p in pats]}"
        done += 1


@pytest.mark.parametrize("seed", range(5))
def test_reference_attention_matches_gather(seed):
    """Efficient gathered-query attention == full causal attention then select."""
    torch.manual_seed(seed)
    blk = ms.TransformerBlock(D, H, 2)
    x = torch.randn(3, T, D)
    tok = (torch.rand(3, T // BS) < 0.5).repeat_interleave(BS, dim=1)
    tok[0] = False                     # a sample with no attention blocks
    a = blk.forward_routed(x, tok)
    b = blk.forward_routed_reference(x, tok)
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()


# ─────────────────────────────────────────────────────────────────────────────
# P1.3 true CBSC
# ─────────────────────────────────────────────────────────────────────────────

def test_cbsc_equivalence():
    """All blocks -> Mamba with cbsc 'state' == the Mamba expert over the whole sequence."""
    torch.manual_seed(0)
    layer = ms.ABRLayer(D, H, BS, cbsc_mode="state", ffn_mult=2)
    layer.train()
    layer.force_routing = torch.ones(T // BS, dtype=torch.long)
    x = torch.randn(2, T, D)
    out, _, _ = layer(x)
    ref, _ = layer.mamba(x)
    assert torch.allclose(out, ref, atol=1e-4), (out - ref).abs().max()


def test_gated_open_gate_equals_state():
    """gated_state with the gate forced fully open (bias +50) == plain state CBSC."""
    torch.manual_seed(0)
    st = ms.ABRLayer(D, H, BS, cbsc_mode="state", ffn_mult=2)
    torch.manual_seed(0)
    gt = ms.ABRLayer(D, H, BS, cbsc_mode="gated_state", ffn_mult=2)
    missing = gt.load_state_dict(st.state_dict(), strict=False)
    assert all(k.startswith("cbsc_gate") for k in missing.missing_keys)
    with torch.no_grad():
        gt.cbsc_gate.lin.bias.fill_(50.0)
    pat = torch.tensor([1, 0, 1, 1, 0, 0, 1, 1])
    st.force_routing = gt.force_routing = pat
    st.eval(); gt.eval()
    x = torch.randn(2, T, D)
    a, _, _ = st(x)
    b, _, _ = gt(x)
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()


def test_gate_init_mostly_open():
    torch.manual_seed(0)
    layer = ms.ABRLayer(D, H, BS, cbsc_mode="gated_state", ffn_mult=2)
    g = layer.cbsc_gate
    assert torch.all(g.lin.weight == 0)
    assert torch.allclose(g.lin.bias, torch.full_like(g.lin.bias, 4.0))
    h = torch.randn(1, 2, layer.mamba.d_inner)
    assert torch.allclose(g(h), torch.sigmoid(torch.tensor(4.0)) * h)


@pytest.mark.parametrize("cbsc", ["state", "gated_state"])
def test_cbsc_skips_transformer_blocks(cbsc):
    """Routing M-T-M: block 3 (index 2) depends on block 1 (index 0) via the Mamba
    state, although block 2 (index 1) is attention. With cbsc 'none' it does not."""
    torch.manual_seed(0)
    T3 = 3 * BS
    pat = torch.tensor([1, 0, 1])
    results = {}
    for mode in (cbsc, "none"):
        torch.manual_seed(0)
        layer = ms.ABRLayer(D, H, BS, cbsc_mode=mode, ffn_mult=2)
        layer.eval()
        layer.force_routing = pat
        h = torch.randn(2, T3, D, generator=torch.Generator().manual_seed(1),
                        requires_grad=True)
        out, _, _ = layer(h)
        (g,) = torch.autograd.grad(out[:, 2 * BS:].sum(), h)
        results[mode] = g[:, :BS].abs().max().item()
        if mode == cbsc:
            # verify through the state explicitly
            with torch.no_grad():
                y0, s0 = layer.mamba(h[:, :BS])
                if mode == "gated_state":
                    s0 = (s0[0], layer.cbsc_gate(s0[1]))
                y2, _ = layer.mamba(h[:, 2 * BS:], s0)
                assert torch.allclose(out[:, 2 * BS:], y2, atol=1e-5)
                assert torch.allclose(out[:, :BS], y0, atol=1e-5)
    assert results[cbsc] > 0
    assert results["none"] == 0.0


def test_mamba_expert_state_continuation():
    """Running the expert on [a; b] == running on a, then on b from a's final state."""
    torch.manual_seed(0)
    e = ms.MambaExpert(D)
    x = torch.randn(2, 20, D)
    full, sf = e(x)
    a, sa = e(x[:, :7])
    b, sb = e(x[:, 7:], sa)
    assert torch.allclose(torch.cat([a, b], 1), full, atol=1e-5)
    assert torch.allclose(sb[1], sf[1], atol=1e-5)
    assert torch.allclose(sb[0], sf[0], atol=1e-6)


def test_mamba_expert_matches_legacy_block():
    """MambaExpert (GRU) with zero state computes the same function as MambaBlock."""
    torch.manual_seed(0)
    old = ms.MambaBlock(D)
    torch.manual_seed(0)
    new = ms.MambaExpert(D)
    new.load_state_dict(old.state_dict())
    x = torch.randn(2, 20, D)
    assert torch.allclose(old(x), new(x)[0], atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# P1.4 aux losses + temperature
# ─────────────────────────────────────────────────────────────────────────────

def test_temperature_updates_once_per_step():
    model = make(cbsc_mode="state")
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    x = tokens()
    tau0 = [r.temperature for r in model.routers()]
    for _ in range(3):                # several forwards (e.g. grad accumulation)
        ce_loss(model, x).backward()
    assert [r.temperature for r in model.routers()] == tau0, "forward changed tau"
    history = [tau0]
    for step in range(4):
        ce_loss(model, x).backward()
        opt.step(); opt.zero_grad()
        model.update_router_temperatures()
        history.append([r.temperature for r in model.routers()])
        # a second call in the same step must not change tau again
        model.update_router_temperatures()
        assert [r.temperature for r in model.routers()] == history[-1]
    for prev, cur in zip(history, history[1:]):
        for a, b in zip(prev, cur):
            assert b != a                                  # changed exactly once
            assert b in (min(a * 1.005, 2.0), max(a * 0.999, 0.1))
    # eval forwards record nothing
    model.eval()
    model(x)
    before = [r.temperature for r in model.routers()]
    model.update_router_temperatures()
    assert [r.temperature for r in model.routers()] == before


def test_aux_loss_gradient_path():
    """Aux losses from model.routing_probs alone reach every router."""
    model = make(cbsc_mode="gated_state")
    model.train()
    model(tokens())
    assert len(model.routing_probs) == len(model.layers)
    assert all(z.requires_grad for z in model.routing_probs)
    assert not model.last_routing.requires_grad
    aux = 0.0
    for z in model.routing_probs:
        avg = z.mean(dim=(0, 1))
        lb = (avg * avg).sum() * z.shape[-1]
        ent = -(avg * (avg + 1e-9).log()).sum()
        aux = aux + lb - 0.01 * ent
    (aux / len(model.routing_probs)).backward()
    for i, l in enumerate(model.layers):
        g = l.router.linear.weight.grad
        assert g is not None and g.abs().sum().item() > 0, f"layer {i}"


def test_routing_stats():
    model = make(cbsc_mode="state", n_layers=3)
    model.train()
    model(tokens())
    s = model.get_routing_stats()
    for k in ("transformer_pct", "mamba_pct", "entropy", "per_layer", "tau",
              "transformer_frac_hard"):
        assert k in s
    assert len(s["per_layer"]) == 3 and len(s["tau"]) == 3
    assert all(p["tau"] == 1.0 for p in s["per_layer"])
    assert abs(s["transformer_pct"] + s["mamba_pct"] - 1) < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# Baseline routing modes, Jamba, factory, backend
# ─────────────────────────────────────────────────────────────────────────────

def test_random_routing_fraction():
    model = make(cbsc_mode="state", routing_mode="random", attn_frac=0.25, n_layers=1)
    model.train()
    fr = []
    for s in range(30):
        model(tokens(B=4, seed=s))
        fr.append((model.layers[0].last_indices == 0).float().mean().item())
    assert abs(sum(fr) / len(fr) - 0.25) < 0.06
    assert len(model.routers()) == 0


def test_alternate_routing_pattern():
    model = make(cbsc_mode="state", routing_mode="alternate", n_layers=1)
    model(tokens())
    idx = model.layers[0].last_indices
    assert idx[0].tolist() == [0, 1] * (T // BS // 2)


def test_jamba_patterns():
    m11 = make(expert_type="jamba-1:1", n_layers=4)
    m13 = make(expert_type="jamba-1:3", n_layers=4)
    kinds = lambda m: ["T" if isinstance(l, ms.TransformerBlock) else "M" for l in m.layers]
    assert kinds(m11) == list("TMTM")
    assert kinds(m13) == list("TMMM")
    with pytest.raises(ValueError):
        ms.JambaLM(VOCAB, D, H, 4, pattern="jamba-2:1")


def test_build_model_formats():
    m = build_model({"label": "x"}, VOCAB, **ARCH)
    assert isinstance(m, ms.ABRLM) and len(m.layers) == 4
    assert m.cbsc_mode == "state" and m.routing_mode == "learned" and not m.legacy
    t = build_model(("ABR-2L-state=ON (Ours)", 2, True, True, "hybrid"), VOCAB, **ARCH)
    assert t.cbsc_mode == "state" and not t.legacy
    t = build_model(("ABR-2L-state=OFF", 2, True, False, "hybrid"), VOCAB, **ARCH)
    assert t.cbsc_mode == "none"
    t = build_model(("G", 2, True, True, "hybrid-gated"), VOCAB, **ARCH)
    assert t.cbsc_mode == "gated_state"
    t = build_model(("G", 2, True, True, "hybrid-gated"), VOCAB, legacy=True, **ARCH)
    assert t.legacy and all(l.use_gated_cbsc for l in t.layers)
    j = build_model(("J", 4, False, False, "jamba"), VOCAB, **ARCH)
    assert isinstance(j, ms.JambaLM) and j.pattern == "jamba-1:1"
    d = build_model({"expert_type": "abr", "block_size": 16, "n_layers": 1,
                     "cbsc_mode": "none", "routing_mode": "random", "attn_frac": 0.3,
                     "seeds": [1, 2]}, VOCAB, **ARCH)
    assert d.layers[0].block_size == 16 and d.layers[0].attn_frac == 0.3
    with pytest.raises(ValueError):
        build_model({"cbsc_mode": "bogus"}, VOCAB, **ARCH)
    with pytest.raises(ValueError):
        build_model({"routing_mode": "bogus"}, VOCAB, **ARCH)
    with pytest.raises(ValueError):
        build_model({"expert_type": "bogus"}, VOCAB, **ARCH)


def test_backend_flags():
    assert ms.MAMBA_BACKEND in ("mamba_ssm", "gru_fallback")
    if ms.MAMBA_BACKEND == "gru_fallback":
        with pytest.raises(RuntimeError):
            ms.require_real_mamba(allow_fallback=False)
        assert ms.require_real_mamba(allow_fallback=True) == "gru_fallback"
    else:
        assert ms.require_real_mamba(False) == "mamba_ssm"


# ─────────────────────────────────────────────────────────────────────────────
# Legacy reproduction
# ─────────────────────────────────────────────────────────────────────────────

LEGACY_TUPLES = [
    ("ABR-1L-state=OFF",       1, True,  False, "hybrid"),
    ("ABR-2L-state=ON (Ours)", 2, True,  True,  "hybrid"),
    ("ABR-2L-CBSC-Gated",      2, True,  True,  "hybrid-gated"),
    ("Transformer-Only",       2, False, False, "transformer"),
    ("Mamba-Only",             2, False, False, "mamba"),
    ("Jamba-Style-4L",         4, False, False, "jamba"),
]


@pytest.mark.parametrize("cfg", LEGACY_TUPLES, ids=[c[0] for c in LEGACY_TUPLES])
def test_legacy_matches_original(cfg):
    kw = dict(d_model=D, n_heads=H, block_size=BS, ffn_mult=2, max_seq_len=128)
    torch.manual_seed(42)
    old = orig_build_model(cfg, VOCAB, **kw)
    torch.manual_seed(42)
    new = build_model(cfg, VOCAB, legacy=True, **kw)
    # identical initialisation from the same seed (same RNG consumption order)
    so, sn = old.state_dict(), new.state_dict()
    assert list(so) == list(sn)
    for k in so:
        assert torch.equal(so[k], sn[k]), k
    opt_o = torch.optim.AdamW(old.parameters(), lr=1e-3)
    opt_n = torch.optim.AdamW(new.parameters(), lr=1e-3)
    for step in range(3):             # train mode: Gumbel noise + in-forward tau updates
        x = tokens(B=2, T_=T - 5, seed=step)
        torch.manual_seed(100 + step)
        lo = ce_loss(old, x)
        torch.manual_seed(100 + step)
        ln = ce_loss(new, x)
        assert torch.equal(lo, ln), (step, lo.item(), ln.item())
        lo.backward(); ln.backward()
        opt_o.step(); opt_n.step(); opt_o.zero_grad(); opt_n.zero_grad()
        if hasattr(new, "update_router_temperatures"):
            new.update_router_temperatures()       # must be a no-op in legacy mode
    if isinstance(new, ms.ABRLM):
        assert [l.router.temperature for l in old.layers] == \
               [l.router.temperature for l in new.layers]
        assert old.get_routing_stats()["entropy"] == \
               new.get_routing_stats()["per_layer"][-1]["entropy"]
    old.eval(); new.eval()
    x = tokens(B=2)
    assert torch.equal(old(x), new(x))


# ─────────────────────────────────────────────────────────────────────────────
# Parameter counts
# ─────────────────────────────────────────────────────────────────────────────

def test_param_counts():
    n_layers = 2
    d_inner = 2 * D
    legacy_off = count_params(make(legacy=True, cbsc_mode="none"))
    legacy_gated = count_params(make(legacy=True, cbsc_mode="gated_state"))
    new_none = count_params(make(cbsc_mode="none"))
    new_state = count_params(make(cbsc_mode="state"))
    new_inject = count_params(make(cbsc_mode="legacy_inject"))
    new_gated = count_params(make(cbsc_mode="gated_state"))
    rnd = count_params(make(cbsc_mode="state", routing_mode="random"))
    alt = count_params(make(cbsc_mode="state", routing_mode="alternate"))
    # fixing the code adds no parameters except the state gate
    assert new_none == new_state == new_inject == legacy_off
    assert legacy_gated - legacy_off == n_layers * (D * D + D)
    assert new_gated - new_state == n_layers * (d_inner * d_inner + d_inner)
    # random / alternate have no router (2 x d_model weights per layer)
    assert new_state - rnd == n_layers * 2 * D == new_state - alt
    # Jamba: 1:1 has 2 T + 2 M layers, 1:3 has 1 T + 3 M layers
    tr = count_params(make(expert_type="transformer", n_layers=4))
    mb = count_params(make(expert_type="mamba", n_layers=4))
    j11 = count_params(make(expert_type="jamba-1:1", n_layers=4))
    j13 = count_params(make(expert_type="jamba-1:3", n_layers=4))
    shared = VOCAB * D + 128 * D + D          # tied embed/head + pos + final norm
    per_t, per_m = (tr - shared) // 4, (mb - shared) // 4
    assert j11 == shared + 2 * per_t + 2 * per_m
    assert j13 == shared + 1 * per_t + 3 * per_m
    # ABR layer = router + one T block + one M block
    assert new_state == shared + n_layers * (2 * D + per_t + per_m)


# ─────────────────────────────────────────────────────────────────────────────
# Real-Mamba checks (skipped locally; run on the Kaggle GPU)
# ─────────────────────────────────────────────────────────────────────────────

needs_mamba2 = pytest.mark.skipif(not (ms.MAMBA2_AVAILABLE and torch.cuda.is_available()),
                                  reason="needs mamba_ssm Mamba-2 + CUDA (Kaggle GPU check)")
needs_mamba1 = pytest.mark.skipif(not (ms.MAMBA_AVAILABLE and torch.cuda.is_available()),
                                  reason="needs mamba_ssm + CUDA (Kaggle GPU check)")


@needs_mamba2
def test_gpu_mamba2_manual_forward_matches_library():
    torch.manual_seed(0)
    e = ms.MambaExpert(256, backend="mamba2").cuda()
    x = torch.randn(2, 96, 256, device="cuda")
    ours, _ = e(x)
    lib = x + e.ssm(e.norm(x))
    assert torch.allclose(ours, lib, atol=1e-3), (ours - lib).abs().max()


@needs_mamba2
def test_gpu_mamba2_state_continuation_and_grad():
    torch.manual_seed(0)
    e = ms.MambaExpert(256, backend="mamba2").cuda()
    x = torch.randn(2, 96, 256, device="cuda", requires_grad=True)
    full, _ = e(x)
    a, sa = e(x[:, :40])
    b, _ = e(x[:, 40:], sa)
    assert torch.allclose(torch.cat([a, b], 1), full, atol=1e-3)
    (g,) = torch.autograd.grad(b.sum(), x)
    assert g[:, :40].abs().sum() > 0          # gradient flows through initial_states


@needs_mamba1
@pytest.mark.parametrize("version", [1, 2])
def test_gpu_cbsc_equivalence(version):
    if version == 2 and not ms.MAMBA2_AVAILABLE:
        pytest.skip("no Mamba-2")
    torch.manual_seed(0)
    layer = ms.ABRLayer(256, 8, 32, cbsc_mode="state", mamba_version=version).cuda()
    layer.force_routing = torch.ones(8, dtype=torch.long)
    x = torch.randn(2, 256, 256, device="cuda")
    out, _, _ = layer(x)
    ref, _ = layer.mamba(x)
    assert torch.allclose(out, ref, atol=1e-4), (out - ref).abs().max()
