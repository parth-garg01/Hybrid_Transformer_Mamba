"""
Central configuration for all full-scale experiments.
Target: ~20M params, WikiText-103, GPT-2 BPE tokenizer.

Model configs are dicts consumed by ``models_scaled.build_model``:
    label, n_layers, expert_type, routing_mode, cbsc_mode, legacy, attn_frac
See prd.md §3 (P2.2) for the experiment list and §0.3 for the seed budget.
"""

# ── Model ─────────────────────────────────────────────────────────────────────
D_MODEL      = 256       # hidden dim
N_HEADS      = 8
BLOCK_SIZE   = 32        # tokens per routing block (default; see BLOCK_SIZE_SWEEP)
FFN_MULT     = 4         # FFN hidden = D_MODEL * FFN_MULT

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET      = "wikitext"
DATASET_NAME = "wikitext-103-raw-v1"
TOKENIZER    = "gpt2"    # BPE, vocab ≈ 50K (+1 pad token)
TRAIN_MAX_TOKENS        = None         # Phase 2: the full train split
LEGACY_TRAIN_MAX_TOKENS = 50_000_000   # what the old scripts used

# ── Training ──────────────────────────────────────────────────────────────────
TRAIN_STEPS  = 60_000    # PRD P2.1: every model gets the same 60k-step budget
LEGACY_TRAIN_STEPS = 30_000   # old train_ablation.py budget (train_abr_long used 60k)
BATCH_SIZE   = 8
SEQ_LEN      = 256
LR           = 3e-4
LR_WARMUP    = 500
WEIGHT_DECAY = 0.1
GRAD_CLIP    = 1.0
EVAL_EVERY   = 1_000     # cheap periodic val estimate every N steps
EVAL_BATCHES = 50        # fixed batches (fixed generator) for the periodic estimate
EVAL_SEED    = 1234      # generator seed for those fixed batches (same for all runs)
EVAL_STEPS   = 200       # legacy: 200 random val batches
CKPT_EVERY   = 2_000     # PRD §0.3: checkpoint at least every 2k steps

# ── Full evaluation (PRD P1.5) ────────────────────────────────────────────────
FULL_EVAL_SEQ_LEN    = 256
FULL_EVAL_STRIDE     = 128
FULL_EVAL_BATCH_SIZE = 8

# ── Seeds (PRD §0.3 / P2.3) ───────────────────────────────────────────────────
SEEDS_KEY   = [42, 123, 7, 2024, 31337]
SEEDS_OTHER = [42, 123, 7]
SEEDS       = SEEDS_OTHER     # backward-compat alias (analyze_router.py, legacy)

# ── Auxiliary loss weights ────────────────────────────────────────────────────
LAMBDA_LB  = 0.01
LAMBDA_ENT = 0.001

# ── Weights & Biases ──────────────────────────────────────────────────────────
WANDB_PROJECT = "abr-cbsc"
WANDB_ENTITY  = "parth-garg10012007"   # owner's personal account, not the team entity

# ── Output ────────────────────────────────────────────────────────────────────
RESULTS_DIR        = "results"
PHASE2_RESULTS_DIR = "results/phase2"


def _cfg(label, n_layers, expert_type, routing_mode="learned", cbsc_mode="none",
         attn_frac=0.5, legacy=False):
    return {"label": label, "n_layers": n_layers, "expert_type": expert_type,
            "routing_mode": routing_mode, "cbsc_mode": cbsc_mode,
            "legacy": legacy, "attn_frac": attn_frac}


# ── Phase 2 configs (PRD §3 P2.2) ─────────────────────────────────────────────
# attn_frac: fraction of blocks/layers that use attention. It drives
# routing_mode="random" (expected fraction; override with --attn-frac to match
# the learned router) and is informational elsewhere.
# Alternate-Blocks and Random-Routing use true state CBSC ("state"), so that
# learned-vs-random differs from ABR+CBSC only in the routing.
PHASE2_CONFIGS = [
    _cfg("Transformer-Only", 4, "transformer", attn_frac=1.0),
    _cfg("Mamba-Only",       4, "mamba",       attn_frac=0.0),
    _cfg("Jamba-1:1",        4, "jamba-1:1",   attn_frac=0.5),    # T-M-T-M
    _cfg("Jamba-1:3",        4, "jamba-1:3",   attn_frac=0.25),   # T-M-M-M
]
for _L in (2, 4):
    PHASE2_CONFIGS += [
        _cfg(f"Alternate-Blocks-{_L}L", _L, "abr", "alternate", "state"),
        _cfg(f"Random-Routing-{_L}L",   _L, "abr", "random",    "state"),
        _cfg(f"ABR-{_L}L-noCBSC",       _L, "abr", "learned",   "none"),
        _cfg(f"ABR-{_L}L-CBSC",         _L, "abr", "learned",   "state"),
        _cfg(f"ABR-{_L}L-gatedCBSC",    _L, "abr", "learned",   "gated_state"),
    ]
del _L

# Key comparisons get 5 seeds (PRD §0.3): ABR+CBSC, ABR+gated CBSC, ABR no-CBSC,
# Random-Routing, Jamba-1:3, Transformer-Only. Everything else gets 3.
KEY_CONFIGS = {"Transformer-Only", "Jamba-1:3"} | {
    n for L in (2, 4) for n in (f"Random-Routing-{L}L", f"ABR-{L}L-noCBSC",
                                f"ABR-{L}L-CBSC", f"ABR-{L}L-gatedCBSC")}

# Key pairs for Welch tests (PRD P2.3): (a, b, description). Positive t: a > b.
KEY_PAIRS = []
for _L in (2, 4):
    KEY_PAIRS += [
        (f"ABR-{_L}L-CBSC",      f"ABR-{_L}L-noCBSC",      f"CBSC vs no-CBSC ({_L}L)"),
        (f"ABR-{_L}L-gatedCBSC", f"ABR-{_L}L-CBSC",        f"gated vs plain CBSC ({_L}L)"),
        (f"ABR-{_L}L-CBSC",      f"Random-Routing-{_L}L",  f"learned vs random routing ({_L}L)"),
    ]
del _L
KEY_PAIRS += [
    ("ABR-4L-CBSC",      "Jamba-1:3",        "ABR+CBSC vs Jamba-1:3 (4L)"),
    ("ABR-4L-gatedCBSC", "Jamba-1:3",        "ABR+gated CBSC vs Jamba-1:3 (4L)"),
    ("ABR-4L-CBSC",      "Transformer-Only", "ABR+CBSC vs Transformer (4L)"),
]

# ── Block-size sweep (PRD P2.4): best ABR variant, 3 seeds ───────────────────
BLOCK_SIZE_SWEEP  = [8, 16, 32, 64]
BLOCK_SWEEP_SEEDS = SEEDS_OTHER

# ── Legacy configs: the 9 old ABLATION_CONFIGS, reproducing today's code ──────
# Old tuple: (label, n_layers, use_routing, use_cross_block_state, expert_type)
LEGACY_TUPLES = [
    ("Transformer-Only",        4, False, False, "transformer"),
    ("Mamba-Only",              4, False, False, "mamba"),
    ("Jamba-Style-4L",          4, False, False, "jamba"),
    ("ABR-1L-state=OFF",        1, True,  False, "hybrid"),
    ("ABR-1L-state=ON",         1, True,  True,  "hybrid"),
    ("ABR-2L-state=OFF",        2, True,  False, "hybrid"),
    ("ABR-2L-state=ON (Ours)",  2, True,  True,  "hybrid"),
    ("ABR-2L-CBSC-Gated",       2, True,  True,  "hybrid-gated"),
    ("ABR-4L-CBSC-Gated",       4, True,  True,  "hybrid-gated"),
]
# Deprecated alias kept so old scripts (Colab_Run.py) still import.
ABLATION_CONFIGS = LEGACY_TUPLES


def legacy_tuple_to_dict(t):
    """Map an old config tuple to the new dict format with legacy=True.

    hybrid + CBSC off -> abr / none; hybrid + CBSC on -> abr / legacy_inject
    (old additive output injection); hybrid-gated -> abr / gated_state with
    legacy=True, which the model must map to the old gated output injection;
    jamba -> jamba-1:1 (old JambaLM was T-M-T-M).
    """
    label, n_layers, _use_routing, use_cbsc, et = t
    if et == "transformer":
        return _cfg(label, n_layers, "transformer", attn_frac=1.0, legacy=True)
    if et == "mamba":
        return _cfg(label, n_layers, "mamba", attn_frac=0.0, legacy=True)
    if et == "jamba":
        return _cfg(label, n_layers, "jamba-1:1", attn_frac=0.5, legacy=True)
    if et == "hybrid-gated":
        return _cfg(label, n_layers, "abr", "learned", "gated_state", legacy=True)
    return _cfg(label, n_layers, "abr", "learned",
                "legacy_inject" if use_cbsc else "none", legacy=True)


LEGACY_CONFIGS = [legacy_tuple_to_dict(t) for t in LEGACY_TUPLES]
LEGACY_LONG_LABELS = ["ABR-2L-state=OFF", "ABR-2L-state=ON (Ours)",
                      "ABR-2L-CBSC-Gated", "ABR-4L-CBSC-Gated"]


def get_config(label, legacy=False):
    pool = LEGACY_CONFIGS if legacy else PHASE2_CONFIGS
    for c in pool:
        if c["label"] == label:
            return dict(c)
    raise KeyError(f"unknown config {label!r}; choose from {[c['label'] for c in pool]}")


def seeds_for(label):
    return list(SEEDS_KEY if label in KEY_CONFIGS else SEEDS_OTHER)


def contains_mamba(cfg):
    return cfg["expert_type"] != "transformer"


# ── Context-length benchmark (PRD P2.5) ───────────────────────────────────────
CONTEXT_LENGTHS    = [256, 512, 1024, 2048, 4096]
BENCH_BATCH_SIZE   = 8
BENCH_WARMUP_ITERS = 10
BENCH_ITERS        = 50
