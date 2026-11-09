# Project Log: ABR + CBSC Fixes

Plan: see `prd.md`. This file is the single source of truth for progress (PRD §0.1).

## Status board
| Task | Status | Owner | Last update |
|------|--------|-------|-------------|
| P1.0 Tests + project.md | in progress | lead | 2026-11-09 |
| P1.0 Provenance, CI, signing (PRD §0.4) | done | agent: provenance | 2026-11-09 |
| P1.1 Router gradient | done | agent: model | 2026-11-09 |
| P1.2 Cross-block context | done | agent: model | 2026-11-09 |
| P1.3 True CBSC | in progress (CPU done; real Mamba-2 gated path unverified, needs Kaggle GPU check) | agent: model | 2026-11-09 |
| P1.4 Aux losses + temperature | done (model side; training loop must call the new API) | agent: model | 2026-11-09 |
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
- 2026-11-09 — (agent: model) **Router input made causal.** The old router summarised a block by the mean of all its tokens, so the routing decision (and, with the P1.1 straight-through weight, the gradient) for position t depended on later tokens in the same block. The fixed router uses the layer input at the block's **first token** (`router_input="first_token"`, default; `"prev_block_mean"` also available). Legacy mode keeps the old mean (its gradient is zero because of argmax, but the hard decision still uses future tokens of the block; legacy results therefore carry a small within-block look-ahead).
- 2026-11-09 — (agent: model) Gated CBSC gate size. GRU stand-in: full `d_inner x d_inner` gate on the GRU state `(1, B, d_inner)`. Real Mamba-2: the state is `(B, nheads, headdim, d_state)`; a full linear map over the flattened state would have ~1e9 parameters, so `W_g` is one `d_state x d_state` matrix per head (shared over headdim), with bias `(nheads, headdim, d_state)`. `W_g` is zero-initialised and the bias is +4 (gate = sigmoid(4) ~ 0.982, nearly open). Only the recurrent state is gated; the conv state is carried ungated.
- 2026-11-09 — (agent: model) On the real backend, `cbsc_mode="gated_state"` uses Mamba-2 (state passing needs `mamba_chunk_scan_combined`), while the other modes default to Mamba-1 (`mamba_ssm.Mamba`). This confounds gated vs plain CBSC. **Recommendation:** pass `"mamba_version": 2` for all ABR variants (and ideally the Mamba/Jamba baselines) in Phase 2. Lead to decide before pre-registered runs start.
- 2026-11-09 — (agent: model) Legacy tuples passed to `build_model` now build the **fixed** model by default (PRD: new behaviour is the default); `legacy=True` reproduces the old code bit-for-bit.
- 2026-11-09 — (agent: model) P1.1–P1.4 landed as one code commit because all four changes live in one file (`models_scaled.py`) and depend on each other. The commit message lists each task.
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

### 2026-11-09 — P1.1, P1.2, P1.3, P1.4: model fixes (agent: model)
- **Done:**
  - **P1.1:** The new `GumbelRouter` returns a straight-through `z`. The chosen expert's block output is multiplied by `z[..., chosen]` (exactly 1 in forward), so the router gets a gradient. The old code is kept as `LegacyGumbelRouter` + `LegacyABRLayer` (`legacy=True`).
  - **P1.2:** The Transformer expert takes queries from attention-routed blocks only, with keys/values from **all** earlier tokens of the layer input. The efficient path (`attn_impl="gather"`) gathers the routed queries and applies a causal `attn_mask` in SDPA. The reference path (`attn_impl="reference"`) runs full causal attention, then selects. Both are tested to be equal. The router input was made causal (see Decisions).
  - **P1.3:** New `MambaExpert.forward(x, init_state=None) -> (y, final_state)`.
    - `cbsc_mode="state"` packs each sample's Mamba blocks in order (right-padded), runs the expert once, and scatters back.
    - `"gated_state"` loops block by block with `h_init = sigmoid(W_g h_prev + b_g) * h_prev`, bias +4.
    - `"none"` starts each Mamba block from zero state.
    - `"legacy_inject"` is the old injection.
    - The real Mamba-2 path is written with `mamba_chunk_scan_combined(initial_states=..., return_final_states=True)` and is marked `# UNVERIFIED: needs Kaggle GPU check`.
  - **P1.4:**
    - `model.routing_probs` holds the non-detached `z_soft` of every ABR layer. `model.last_routing` is the detached z_soft of the last layer.
    - The temperature update moved to `router.update_temperature()`, which uses the entropy recorded on the last training forward and is consumed once. The training loop calls `model.update_router_temperatures()` once per optimizer step.
    - `get_routing_stats()` gives overall stats (all layers), `per_layer` stats and `tau`.
  - **Also:**
    - Routing modes `learned` / `random` (`attn_frac`) / `alternate`.
    - `JambaLM(pattern="jamba-1:1"|"jamba-1:3")`.
    - `MAMBA_BACKEND` and `require_real_mamba(allow_fallback)`.
    - `build_model` accepts both legacy tuples and the dict format.
- **Files changed:** `experiments/full_scale/models_scaled.py`, `tests/test_models.py`, `tests/conftest.py`
- **Results:**
  - `pytest tests/test_models.py -q`: **63 passed, 4 skipped in ~10 s** (CPU, GRU stand-in; the 4 skipped are GPU-only real-Mamba checks `test_gpu_*`, to be run on Kaggle). The tests cover router gradient (none/state/gated), STE forward unchanged, no future leakage (17 variants incl. legacy, random, alternate, Jamba, ragged T), cross-block dependency (20 forced patterns per mode, exact expected paths), gather == reference attention, CBSC equivalence (all-Mamba state mode == expert over the whole sequence), gate fully open == state, M-T-M state skip, temperature once per step, aux-loss-only router gradient, legacy bit-exact vs the original code (`fixes` copy embedded in the test; 3 training steps + eval), and parameter counts.
  - A mutation check confirmed that the leakage test catches a non-causal router and unmasked attention.
  - **Parameter overhead of the gate:**
    - GRU stand-in: `d_inner^2 + d_inner` per layer (262,656 at d_model=256).
    - Mamba-2: `nheads*d_state*(d_state+headdim)` per layer (65,536 at d_model=256, headdim=64, d_state=64; unverified).
    - Legacy gate: `d_model^2 + d_model` (65,792).
  - The fix adds no other parameters. Random/alternate routing has no router (-2·d_model per layer).
  - **Full-size parameter counts** (vocab 50257, d_model 256, **GRU stand-in, so the Mamba-containing numbers differ on real Mamba**):

    | Model | Params |
    |-------|--------|
    | Transformer-4L | 17.13M |
    | Mamba-4L | 20.82M |
    | Jamba-1:1 | 18.98M |
    | Jamba-1:3 | 19.90M |
    | ABR-4L state | 25.02M |
    | ABR-4L gated | 26.07M |
- **Problems / notes:**
  - Within one layer, a Mamba block preceded only by Transformer blocks gets no context from them; with cbsc none it gets no cross-block context at all. Context mixes across layers via the Transformer blocks. This is the PRD's optional "Mamba gets a summary of intervening Transformer blocks" ablation; it is not implemented.
  - The Mamba-1 backend cannot take or return state, so on real hardware `gated_state` requires Mamba-2.
  - The training loops (`train_ablation.py`, `train_abr_long.py`) still use `model.last_routing` for aux losses and must be switched to `model.routing_probs` + `model.update_router_temperatures()` (owned by the train-eval agent).
- **Next:** Kaggle GPU check of the real-Mamba paths. It must confirm:
  1. `import mamba_ssm.Mamba2` and `mamba_chunk_scan_combined` work on the assigned GPU.
  2. `MambaExpert(backend="mamba2")` over `[a; b]` equals `a`, then `b` from `a`'s final state (tol ~1e-3 in fp32). Run `test_mamba_expert_state_continuation` with the mamba2 backend.
  3. `test_cbsc_equivalence` with real Mamba-1 and Mamba-2.
  4. The manual Mamba-2 forward with `init_state=None` matches `Mamba2.forward`.
  5. Gradients flow through `initial_states`.
