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
| P1.5 Full evaluation + stats | todo | agent: train-eval | |
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
