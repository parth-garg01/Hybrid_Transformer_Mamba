# Project Log: ABR + CBSC Fixes

Plan: see `prd.md`. This file is the single source of truth for progress (PRD §0.1).

## Status board
| Task | Status | Owner | Last update |
|------|--------|-------|-------------|
| P1.0 Tests + project.md | in progress | lead | 2026-11-09 |
| P1.0 Provenance, CI, signing (PRD §0.4) | done | agent: provenance | 2026-11-09 |
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

### 2026-11-09 — P1.0 Provenance, CI (PRD §0.4)
- **Done:**
  - Added `experiments/full_scale/provenance.py` with the agreed API: `start_run`, `dataset_sha256`, `set_wandb_url`, `finish_run`, `RunLogger`, plus `DirtyWorkingTreeError`/`ProvenanceError` and `REQUIRED_META_KEYS`. `start_run` refuses to start on uncommitted changes to tracked files unless `debug=True` (then `reportable: false`), and records git commit/branch/describe/tags, command line, UTC times, host, GPU (name, driver), CUDA/cuDNN/torch/`mamba_ssm`/`causal-conv1d` versions, full `pip freeze`, Kaggle env info and the W&B URL. `finish_run` writes `<out>/<run_id>/result.json` atomically, copies the raw log unmodified to `train.log`, and never overwrites an existing result.
  - Secrets are never recorded: env vars with KEY/TOKEN/SECRET/PASSWORD in the name are skipped, secret-looking CLI values and config keys are redacted, and URL credentials are stripped from `pip freeze`.
  - Added `scripts/make_manifest.py` (writes `results/MANIFEST.sha256` in `sha256sum` format) and `scripts/verify_provenance.py` (required keys, reportable, commit exists, not dirty, run_id matches dir, manifest match; passes with "0 results" when there are none).
  - Added CI (`.github/workflows/tests.yml`: Ubuntu, Python 3.11, CPU torch, `pytest -q` over all of `tests/`, then the verifier), `requirements-dev.txt`, `requirements-lock.txt` (local dev lock; Kaggle env is recorded per run), `REPRODUCE.md` skeleton and `CITATION.cff`.
- **Files changed:** `experiments/full_scale/provenance.py`, `scripts/make_manifest.py`, `scripts/verify_provenance.py`, `tests/test_provenance.py`, `.github/workflows/tests.yml`, `requirements-dev.txt`, `requirements-lock.txt`, `REPRODUCE.md`, `CITATION.cff`, `project.md`
- **Results:** `tests/test_provenance.py`: 22 passed locally (macOS, Python 3.13, torch 2.15). `python scripts/verify_provenance.py results`: 0 results, PASS. No experiment results (tooling only).
- **Problems / notes:**
  - `.gitignore` ignores `*.log` and `*.png`, so `results/<run>/train.log` and any figures under `results/` would not be committed and the verifier would fail on a fresh checkout. The lead should add `!results/**/train.log` (and `!results/**/*.png` if figures go there). `make_manifest.py` warns about git-ignored files.
  - Code uploaded to Kaggle without `.git` cannot answer `git status`. Added `python experiments/full_scale/provenance.py snapshot`, which writes `git_snapshot.json` (commit info plus SHA-256 of every tracked file); `start_run` uses it when there is no `.git` and refuses if any tracked file differs. The Kaggle launcher should use it. `git_snapshot.json` should probably be git-ignored.
  - CI checks out with `fetch-depth: 0` so the verifier can find result commits; results must only reference pushed commits.
  - Deviations: `extra` keys are merged into the top level of `meta` (they may not overwrite recorded fields, except `mamba_backend`). Extra meta keys beyond the agreed list: `schema_version`, `debug`, `git_tags`, `git_source`, `git_changed_files`, `git_untracked_files`, `python_executable`, `cwd`, `start_unix`, `cudnn_version`, `gpu_count`, `gpu_driver`, `causal_conv1d_version`. `kaggle` is always a dict with `on_kaggle` (false locally). The verifier treats non-reportable runs as warnings unless `--strict`.
- **Next:** training loop (agent: train-eval) calls `start_run`/`finish_run` and passes `dataset_sha256` and `param_count` via `extra`; Kaggle launcher writes the git snapshot and sets `KAGGLE_KERNEL_URL`.
