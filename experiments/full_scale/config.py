"""
Central configuration for all full-scale experiments.
Target: ~20M params, WikiText-103, GPT-2 BPE tokenizer, CUDA required.
"""

# ── Model ─────────────────────────────────────────────────────────────────────
D_MODEL      = 256       # hidden dim  (512 → ~50M params if VRAM allows)
N_HEADS      = 8
BLOCK_SIZE   = 8         # tokens per routing block
FFN_MULT     = 4         # FFN hidden = D_MODEL * FFN_MULT

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET      = "wikitext"
DATASET_NAME = "wikitext-103-raw-v1"
TOKENIZER    = "gpt2"    # BPE, vocab ≈ 50K

# ── Training ──────────────────────────────────────────────────────────────────
TRAIN_STEPS  = 10_000
BATCH_SIZE   = 32
SEQ_LEN      = 256       # default; overridden in context-length benchmark
LR           = 3e-4
LR_WARMUP    = 500
WEIGHT_DECAY = 0.1
GRAD_CLIP    = 1.0
EVAL_EVERY   = 500
EVAL_STEPS   = 100

# ── Ablation seeds ────────────────────────────────────────────────────────────
SEEDS        = [42, 123, 7]

# ── Ablation configs to sweep ─────────────────────────────────────────────────
# Each entry: (label, n_layers, use_routing, use_cross_block_state, expert_type)
# expert_type: "hybrid" | "transformer" | "mamba"
ABLATION_CONFIGS = [
    # Baselines
    ("Transformer-Only",        4, False, False, "transformer"),
    ("Mamba-Only",              4, False, False, "mamba"),
    # Routing ablation
    ("ABR-1L-state=OFF",        1, True,  False, "hybrid"),
    ("ABR-1L-state=ON",         1, True,  True,  "hybrid"),
    ("ABR-2L-state=OFF",        2, True,  False, "hybrid"),
    ("ABR-2L-state=ON (Ours)",  2, True,  True,  "hybrid"),
]

# ── Context lengths to benchmark ──────────────────────────────────────────────
CONTEXT_LENGTHS = [128, 256, 512, 1024]

# ── Efficiency benchmark ──────────────────────────────────────────────────────
BENCH_BATCH_SIZE  = 8
BENCH_WARMUP_ITERS = 10
BENCH_ITERS        = 50

# ── Auxiliary loss weights ────────────────────────────────────────────────────
LAMBDA_LB  = 0.01
LAMBDA_ENT = 0.001

# ── Output ────────────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
