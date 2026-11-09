# Project Log: ABR + CBSC Fixes

Plan: see `prd.md`. This file is the single source of truth for progress (PRD §0.1).

## Status board
| Task | Status | Owner | Last update |
|------|--------|-------|-------------|
| P1.0 Tests + project.md | in progress | lead | 2026-11-09 |
| P1.0 Provenance, CI, signing (PRD §0.4) | todo | agent: provenance | |
| P1.1 Router gradient | todo | agent: model | |
| P1.2 Cross-block context | todo | agent: model | |
| P1.3 True CBSC | todo | agent: model | |
| P1.4 Aux losses + temperature | todo | agent: model | |
| P1.5 Full evaluation + stats | done | agent: train-eval | 2026-11-09 |
| P1.6 Smoke runs | todo | lead | |
| Kaggle pipeline + Mamba compatibility check | todo | agent: kaggle | |
| P2.1–P2.6 Fair experiments | todo | | |
| P3.1 Attention-budget dial | todo | | |
| P3.2 Scale up | todo | | |
| P3.3 Broader evaluation | todo | | |
| P3.4 Router analysis | todo | | |
| Phase 4 Paper update (local only) | todo | | |

## Key results so far
None yet. No reportable experiment has been run under the new pipeline.

## Decisions and deviations from the PRD
- 2026-11-09 — The numbers in the current paper have no surviving raw logs (the original RTX 2050 machine is gone). They are marked unverifiable and will not be reused (PRD §0.4 G).
- 2026-11-09 — Compute split: everything involving Mamba runs on Kaggle; everything else runs locally on the Apple M5 Mac (PRD §0.3).
- 2026-11-09 — The paper (`paper/`) is kept locally only and excluded via `.gitignore`. An earlier commit that briefly contained it was removed from the `fixes` branch before any experiment work began. The commit may still be reachable on GitHub by its exact hash until GitHub purges it.
- 2026-11-09 — Weights & Biases is required for all reportable runs. Kaggle runs log offline and are synced locally (PRD §0.4 C).
- 2026-11-09 — Work is parallelised across agents in separate git worktrees (PRD §0.5).
- 2026-11-09 — Local dev environment: `~/.venvs/abr` (Python 3.13, PyTorch with MPS). Kaggle CLI: `~/.venvs/kaggle`.

## Log
### 2026-11-09 — P1.5 + P1.4 (training loop): unified train/eval pipeline
- **Done:**
  - New `train.py`: one entry point per (config, seed[, block size]). It calls `provenance.start_run` with the per-split dataset SHA-256 hashes, `param_count`, `train_tokens` and `mamba_backend`, and refuses Mamba models without real `mamba_ssm` unless `--allow-gru-fallback` is passed (such runs are marked non-reportable). It logs to W&B (project `abr-cbsc`, entity `parth-garg10012007`; offline files go in `<out>/<run_id>/wandb`), tees all output to `<out>/<run_id>/train.log` and ends with `finish_run`.
  - Aux losses (load balancing + entropy) are now averaged over every layer's non-detached `routing_probs` (B4). `model.update_router_temperatures()` is called exactly once per optimizer step (B5, training-loop side).
  - Checkpoint/resume every 2,000 steps: model, optimizer, step, torch/numpy/random/CUDA/MPS and data-generator RNG states, router temperatures, curves and the W&B id. `--max-hours` checkpoints and exits cleanly. Rerunning the same command resumes, keeping one `run_id` and one log. A resume is refused if the config, the dataset hash or the git commit changed.
  - `evaluation.evaluate_full`: deterministic sliding window (256/128) over the whole split, scoring each token exactly once. It runs at the end of every run on the full validation and test splits (B6). The periodic estimate uses 50 fixed val batches from a fixed generator, the same for every run.
  - `stats.py`: `summarize` (ddof=1, 95% t-interval), `welch` (t, p, Welch df, Cohen's d, Hedges' g), Holm-adjusted `compare_table` over `config.KEY_PAIRS`, and `aggregate()` writing `summary.csv` + `key_pairs.csv` (B8). `config.py` now has `PHASE2_CONFIGS` (14 configs), `LEGACY_CONFIGS` (the old 9, as dicts with `legacy=True`), the seed lists, `KEY_CONFIGS`, `BLOCK_SIZE_SWEEP` and `TRAIN_STEPS=60_000`. `dataset.py` caches tokenized splits (train/validation/test) in `~/.cache/abr/wikitext103_<split>_gpt2.pt` and samples batches from a seeded `torch.Generator`.
  - `train_ablation.py` is now a sweep driver around `train.main` and `train_abr_long.py` a legacy shortcut. Both take `--legacy` to reproduce the old runs. `bench_context.py` benchmarks any config (forward and train, tok/s and peak memory, lengths 256–4096), with `--routing-fractions LABEL=f` / `--routing-fractions-from <results dir>`.
- **Files changed:** `experiments/full_scale/{config.py,dataset.py,evaluation.py,stats.py,train.py,train_ablation.py,train_abr_long.py,bench_context.py}`, `tests/test_train_eval.py`
- **Results:** `tests/test_train_eval.py`: 21 passed in about 1.3 s on CPU (fake model and fake provenance). An extra check against the real `provenance.py` from `p1/provenance` (fake model, interrupted and resumed run) wrote `result.json` + `train.log` with the expected meta. No experiment numbers yet.
- **Problems / notes:**
  - Phase 2 run count with this config list: 10 key configs × 5 seeds + 4 other configs × 3 seeds + block-size sweep (3 extra sizes × 3 seeds) = 71 runs of 60k steps. That is probably more than 100 GPU-hours on a T4, i.e. several weeks of Kaggle quota. Transformer-Only (5 runs) can run locally. The lead should confirm the budget before Phase 2 starts.
  - **Not decided by me:** Alternate-Blocks and Random-Routing are listed at both 2L and 4L, with true state CBSC, so that "learned vs random" differs from ABR+CBSC only in the routing. The PRD only says "ABR variants at 2L and 4L".
  - Phase 2 trains on the full WikiText-103 train split (`TRAIN_MAX_TOKENS=None`). The old scripts used the first 50M tokens, which `--legacy` still does. 60k × 8 × 256 ≈ 123M tokens, about one epoch.
  - `--legacy` does not call `update_router_temperatures()`, on the assumption that the legacy model still updates τ inside forward (B5 reproduction). The legacy "gated" config is encoded as `cbsc_mode="gated_state", legacy=True`, and `models_scaled` must map that to the old gated output injection.
  - `start_utc`/`duration_s` in `meta` cover only the final session of a resumed run. The total over all sessions is in `results.wall_time_s` and `results.sessions`.
  - `.gitignore` contains `*.log`. For the raw logs to be committed with results (PRD §0.4 C), `results/**/train.log` must be un-ignored.
- **Next:** after merge, integration test with the real `models_scaled` (P1.1–P1.4), then the P1.6 smoke runs with `train.py --steps 500`.
### 2026-11-09 — Setup
- **Done:**
  - Wrote the LaTeX version of the paper (local only).
  - Reviewed the code and confirmed bugs B1–B8 (PRD §1) by running gradient tests on CPU.
  - Wrote the PRD and committed it as a signed pre-registration (`96f0153`, Verified on GitHub).
  - Set up SSH commit signing and Kaggle CLI access.
- **Files changed:** `prd.md`, `.gitignore`, `project.md`
- **Results:** none (setup only).
- **Problems / notes:**
  - Measured local training speed on the M5 (MPS): Transformer-4L 183 ms/step, ABR-2L with the GRU stand-in 283 ms/step (batch 8, T=256).
- **Next:** parallel Phase 1 work by agents (provenance/CI, model fixes, training/eval, Kaggle pipeline).
