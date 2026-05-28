# Full-Scale Experiments — Requires CUDA GPU

## Why you can't run this locally
Your Windows machine has `torch 2.11.0+cpu` — no CUDA.
These experiments need a GPU. Use **Google Colab (free T4)** or **Kaggle (free T4/P100)**.

## Estimated runtime on a T4 GPU (16 GB VRAM)
| Script | Config | Time |
|---|---|---|
| `train_ablation.py` | 6 configs × 3 seeds × 10k steps | ~5-7 hrs |
| `bench_context.py` | 4 ctx lengths × 3 models | ~1-2 hrs |
| `analyze_router.py` | 3 seeds × 3k steps | ~30 min |
| **Total** | | **~7-10 hrs** |

For a quick proof-of-concept, set in `config.py`:
```python
TRAIN_STEPS = 1000
SEEDS       = [42]
```
This runs in ~30 minutes and gives you real data, just with higher variance.

## What each script produces

### `train_ablation.py` → `results/ablation_aggregate.csv`
Full ablation table with mean ± std across seeds:

| Model | Params | PPL mean | PPL std |
|---|---|---|---|
| Transformer-Only | 20M | x.xx | x.xx |
| Mamba-Only | 20M | x.xx | x.xx |
| ABR-1L-state=OFF | 10M | x.xx | x.xx |
| ABR-1L-state=ON | 10M | x.xx | x.xx |
| ABR-2L-state=OFF | 20M | x.xx | x.xx |
| ABR-2L-state=ON (Ours) | 20M | x.xx | x.xx |

This directly addresses the reviewer concern: **isolates CBSC from layer depth**.

### `bench_context.py` → `results/context_length_benchmark.json`
Context-length table the paper needs:

| Context | Model | PPL | tok/s | Mem (GB) | Latency (ms) | GFLOPs |
|---|---|---|---|---|---|---|
| 128 | Transformer | x | x | x | x | x |
| 128 | Mamba | x | x | x | x | x |
| 128 | ABR-2L | x | x | x | x | x |
| 256 | ... | | | | | |
| 512 | ... | | | | | |
| 1024 | ... | | | | | |

### `analyze_router.py` → `results/router_analysis/`
- `positional_bias.png` — does the router prefer T or M at specific sequence positions?
- `routing_entropy_over_training.png` — router health during training
- `routing_seed_consistency.png` — do different seeds agree on routing decisions?

### `plot_results.py` → `results/figures/`
IEEE-quality paper figures from all results above.

## How to run on Colab

1. Open [colab.research.google.com](https://colab.research.google.com)
2. Runtime → Change runtime type → **T4 GPU**
3. Run these cells in order:

```python
# Cell 1 — Install
!pip install -q datasets transformers mamba-ssm causal-conv1d
!git clone https://github.com/parth-garg01/Hybrid_Transformer_Mamba.git
%cd Hybrid_Transformer_Mamba/experiments/full_scale

# Cell 2 — Verify GPU
import torch; print(torch.cuda.get_device_name(0))

# Cell 3 — Full ablation
!python train_ablation.py

# Cell 4 — Context + efficiency
!python bench_context.py

# Cell 5 — Router analysis
!python analyze_router.py

# Cell 6 — Figures
!python plot_results.py

# Cell 7 — Download
import shutil; shutil.make_archive("results", "zip", "results")
from google.colab import files; files.download("results.zip")
```

## Model scale (~20M params)
With `D_MODEL=256`, `N_HEADS=8`, GPT-2 BPE (50K vocab):
- Embedding: 50K × 256 = 12.8M
- 4× Transformer layers: ~8M
- **Total ≈ 20M** ✓

To scale to ~50M: set `D_MODEL=512` in `config.py`.
