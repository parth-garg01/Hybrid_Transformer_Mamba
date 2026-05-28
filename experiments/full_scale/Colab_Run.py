# ============================================================
# Run this file as a Colab notebook (copy cells into Colab)
# or paste the entire file into a Colab code cell.
#
# BEFORE RUNNING:
#   Runtime > Change runtime type > GPU (T4 or better)
# ============================================================

# ── CELL 1: Install dependencies ─────────────────────────────────────────────
"""
!pip install -q datasets transformers mamba-ssm causal-conv1d
!git clone https://github.com/parth-garg01/Hybrid_Transformer_Mamba.git
%cd Hybrid_Transformer_Mamba
"""

# ── CELL 2: Verify CUDA ───────────────────────────────────────────────────────
"""
import torch
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory/1e9,1), "GB")
"""

# ── CELL 3: Run full ablation (all configs × 3 seeds) ────────────────────────
# Estimated time on T4: ~4-6 hours for 10k steps × 6 configs × 3 seeds
# For a quick test, edit config.py: TRAIN_STEPS = 1000, SEEDS = [42]
"""
%cd experiments/full_scale
!python train_ablation.py
"""

# ── CELL 4: Run context-length + efficiency benchmark ────────────────────────
# Estimated time: ~1-2 hours
"""
!python bench_context.py
"""

# ── CELL 5: Router specialization analysis ───────────────────────────────────
"""
!python analyze_router.py
"""

# ── CELL 6: Generate all figures ─────────────────────────────────────────────
"""
!python plot_results.py
"""

# ── CELL 7: Download results ──────────────────────────────────────────────────
"""
import shutil
shutil.make_archive("full_scale_results", "zip", "results")
from google.colab import files
files.download("full_scale_results.zip")
"""

# ── CELL 8: Quick sanity-check run (2k steps, 1 seed, small model) ───────────
# Use this to verify everything works before committing to the full run.
"""
import subprocess, sys
# Temporarily patch config for quick test
quick_cfg = open("config.py").read()
quick_cfg = quick_cfg.replace("TRAIN_STEPS  = 10_000", "TRAIN_STEPS  = 500")
quick_cfg = quick_cfg.replace("SEEDS        = [42, 123, 7]", "SEEDS        = [42]")
open("config_quick.py","w").write(quick_cfg)

import importlib, config
# Run a single config end-to-end
import os; os.makedirs("results", exist_ok=True)
from config import ABLATION_CONFIGS
from dataset import WikiTextBPE
from train_ablation import train_one

train_ds = WikiTextBPE("train",      seq_len=256, max_tokens=5_000_000)
val_ds   = WikiTextBPE("validation", seq_len=256)

cfg = ABLATION_CONFIGS[-1]   # ABR-2L state=ON
print(f"Quick test: {cfg[0]}")
r = train_one(cfg, 42, train_ds, val_ds)
print(f"Done. Val PPL = {r['val_ppl']:.2f}")
"""
