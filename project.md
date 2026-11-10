# Project Log: ABR + CBSC Fixes

Plan: see `prd.md`. This file is the single source of truth for progress (PRD §0.1).

## Status board
| Task | Status | Owner | Last update |
|------|--------|-------|-------------|
| P1.0 Tests + project.md | done | lead | 2026-11-09 |
| P1.0 Provenance, CI, signing (PRD §0.4) | done | agent: provenance | 2026-11-09 |
| P1.1 Router gradient | done | agent: model | 2026-11-09 |
| P1.2 Cross-block context | done | agent: model | 2026-11-09 |
| P1.3 True CBSC | in progress (CPU done; real Mamba-2 gated path unverified, needs Kaggle GPU check) | agent: model | 2026-11-09 |
| P1.4 Aux losses + temperature | done (model side; training loop must call the new API) | agent: model | 2026-11-09 |
| P1.5 Full evaluation + stats | done | agent: train-eval | 2026-11-09 |
| P1.6 Smoke runs | todo | lead | |
| Kaggle pipeline + Mamba compatibility check | done | agent: kaggle | 2026-11-10 |
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
- 2026-11-09 — **Phase 2 plan trimmed to 46 runs (pre-registered, before any Phase 2 run).** The original budget (71+ runs) did not fit Kaggle's free quota. New plan: 6 key 4L configs × 5 seeds = 30; Mamba-Only, Jamba-1:1, Alternate-Blocks-4L × 2 seeds = 6; ABR-2L-CBSC vs noCBSC × 2 seeds = 4 (exploratory); block sweep {8,16,64} × 2 seeds = 6. Dropped from the default plan: ABR-2L-gatedCBSC, Random-Routing-2L, Alternate-Blocks-2L. Key pairs reduced to the 4L pairs plus one exploratory 2L pair. Source of truth: `SEED_COUNT`, `KEY_PAIRS`, `BLOCK_SWEEP_*` in `config.py`.
- 2026-11-09 — Kaggle runs use the GPU T4 x2 accelerator with two jobs per session (one per GPU). Only the owner's Kaggle account is used; no second account.
- 2026-11-09 — GitHub Actions CI is not used (owner's decision). The workflow file stays in the repo but does not run. Instead, the full test suite (`~/.venvs/abr/bin/python -m pytest -q`) is run locally before each commit, and the pass count goes into the log entry. GPU tests run on Kaggle.
- 2026-11-09 — **P2.1 parameter matching (pre-registered, before any Phase 2 run).** The earlier "ABR-4L 25M vs Transformer 17M" came from the GRU stand-in. With the real Mamba layers (counted analytically: Mamba-1 mixer 437,760, Mamba-2 431,768, embeddings 13.13M), ABR-4L is 19.05M and Transformer-4L is 17.32M (+10%). Baselines are now matched **by depth** to within ±5% of ABR-4L: Transformer 6L 19.42M (+1.9%), Mamba 13L 18.74M (−1.6%), Jamba-1:1 8L 19.05M (0%), Jamba-1:3 10L 19.30M (+1.3%). Random/Alternate routing share ABR's layout. The matched baselines get **more** compute per token than ABR-4L, which works against ABR (the conservative choice). The new GPU test `test_gpu_phase2_param_match` checks this on Kaggle with the real backend.
- 2026-11-10 — Run orchestration: **packed Kaggle sessions, no cloud agent** (owner's decision). Each T4 x2 session runs about 11.5 h of runs back to back in two GPU lanes. The Kaggle key never leaves the laptop. The owner opens the laptop about once a day; `queue.py advance` then downloads the finished session and launches the next one.
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

### 2026-11-09 — Lead: merge of provenance, model and train-eval branches
- **Done:**
  - Merged `p1/provenance`, `p1/model` and `p1/train-eval` into `fixes`. The `project.md` conflicts were resolved by keeping every entry, and each status row took the version with progress.
  - Fixed an integration bug: `config.LEGACY_CONFIGS` encodes old "CBSC on" as `cbsc_mode="legacy_inject"` with `legacy=True`, which `build_model` rejected. `build_model` now accepts it.
  - Added `test_legacy_configs_all_build`, which builds every legacy config through the real model code.
  - Un-ignored `results/**/*.log` and `results/**/*.png` so logs and figures are committed with results (PRD §0.4 C), and ignored the generated `git_snapshot.json`.
- **Files changed:** `experiments/full_scale/models_scaled.py`, `tests/test_models.py`, `.gitignore`, `project.md`
- **Results:**
  - Full suite: 107 passed, 4 skipped (GPU-only real-Mamba tests, to run on Kaggle).
  - Smoke check: every Phase 2 config (10 configs incl. all baselines and 2L/4L ABR) trained 6 steps on synthetic tokens, with the GRU stand-in on MPS, debug mode and W&B disabled. All configs ran train → periodic eval → full val/test eval → `result.json` with provenance meta (commit, dataset hashes, `reportable=false`). Legacy configs also run.
  - On the ABR runs the router's routing fraction and entropy now move during training, and tau updates once per step.
  - These smoke outputs are not results and were not kept.
- **Problems / notes:**
  - **Parameter mismatch to resolve before Phase 2 (P2.1).** On the stand-in, ABR-4L (~25M) is much larger than Transformer-4L (~17M), because each ABR layer holds both experts. Matching within ±5% needs a size adjustment (e.g. smaller d_model for ABR, or more layers for the baselines). To decide with real-Mamba counts from Kaggle.
  - The Mamba-2 recommendation from the model agent (use `mamba_version: 2` for all ABR variants) waits on the Kaggle compatibility check.
  - The Phase 2 plan is 71 runs, about 100+ T4 GPU-hours. The budget needs the owner's sign-off.
- **Next:** merge the Kaggle pipeline and its GPU check, run the GPU tests on Kaggle, then the P1.6 smoke runs on real Mamba.


### 2026-11-09 — Lead: token cache dataset
- Built the WikiText-103 GPT-2 token cache locally (`dataset.py`, commit 8c30683): train 117,920,140 tokens, validation and test splits, vocab 50,258.
- Uploaded it as the **private** Kaggle dataset `parthgarg101/abr-wikitext103-gpt2-tokens`, so Kaggle sessions don't re-tokenize. SHA-256:
  - train `7896d3b921ae6741135a54aba6e1f060e7306e7e58a7110cac8618d235af2f6d`
  - validation `a6091ca4a7f0bd4caae7e4d7eb895cf01a1891bf32e1aa64587c26c877e222ea`
  - test `84bb27f5bcebc4b63e77725bee2e0d318f79cab1ba34b10e9413612630b29b64`
- Kaggle kernels must check these hashes before training (recorded per run in `kaggle_env.json`).


### 2026-11-09 — Kaggle pipeline + Mamba compatibility check (agent: kaggle)
- **Done:**
  - Built `scripts/kaggle/launch.py`: packages a commit with `git archive` (plus `COMMIT`, and `git_snapshot.json` from `provenance.py snapshot` when the module is in the commit), uploads it as a new version of the private dataset `parthgarg101/abr-code` (commit sha in the version notes), pushes a private GPU + internet script kernel `abr-<name>`, waits with backoff, downloads, and writes `launch.json` (commit, archive SHA-256, dataset version, kernel slug/version, timestamps, run/lane plan). It detects phone-verification / GPU refusals (push response, pulled-back kernel metadata, `no_gpu` in `kaggle_env.json`).
  - Kernel runner (`runner_template.py`): verifies the snapshot's commit, mounts and SHA-256-checks the token cache (`ABR_CACHE_DIR`), installs mamba-ssm/causal-conv1d, sets `WANDB_MODE=offline`, `KAGGLE_KERNEL_URL`, `CODE_COMMIT`, tees all output, writes `kaggle_env.json`. Named runs run one per GPU (T4 x2), each lane back to back, with `{max_hours}` = remaining lane budget, own `output/<run>/run.log` and `run_status.json`, and resume by copying the previous session's `output/<run>/` back.
  - `queue.py`: queue from `config.planned_runs()` (+ block sweep), key configs first; packed sessions (2 lanes x 11.5 h, longest-job-first from `scripts/kaggle/run_costs.json`); `advance` = one idempotent command (check, selective download, settle, launch next).
  - `compat_check.py` and `time_train.py`; ran both on Kaggle (2 GPU kernels in total).
  - Created the private dataset `parthgarg101/abr-mamba-wheels` (sm_75 wheels built by the T4x2 kernel + Triton cache; SHA256SUMS inside: causal_conv1d `c13ca5ad…edb9`, mamba_ssm `33994f83…cccc`, triton_cache.tar.gz `775d5338…4216`).
- **Files changed:** `scripts/kaggle/{launch.py,runner_template.py,compat_check.py,time_train.py,queue.py,run_costs.json,README.md}`, `tests/test_kaggle_launcher.py`, `results/kaggle/compat_check/`, `results/kaggle/gputest_t4x2/` (logs/JSON only), `project.md`
- **Results:**
  - Environment (`results/kaggle/compat_check/output/kaggle_env.json`): Kaggle "GPU T4 x2" (2 x Tesla T4, sm_75, 15 GB), driver 580.178.04, CUDA 13.0, torch 2.11.0+cu130, Python 3.13, triton 3.6.0, **mamba-ssm 2.3.2.post1, causal-conv1d 1.7.0** (no release wheels exist for this combo; built from source), transformers 4.57.6.
  - Compatibility (`results/kaggle/compat_check/output/compat_check.json`), all checks OK:

    | Check | Result on T4 |
    |---|---|
    | Mamba-1 fwd+bwd (d=256, L=256, B=8) | OK in fp32, fp16 autocast, bf16 autocast (bf16 not native on T4, emulated) |
    | Mamba-1 fused vs reference path | max abs diff 0.0 |
    | Mamba-2 fwd+bwd | OK in fp32/fp16/bf16 (first call JIT-compiles Triton for ~17 min in this check) |
    | `mamba_chunk_scan_combined` + `initial_states` + `return_final_states` | OK; vs naive fp32 recurrence: y 2.0e-5, final state 9.7e-6, grad(initial_states) 1.9e-6 |
    | Gradient through a gated state hand-off | non-zero to gate and to the previous block's input |
    | Chunked vs single (kernel, L=512) | split at 256: 0.0; split at 200: 5.4e-5; 16 blocks of 32: 3.3e-5 |
    | Chunked vs single (full Mamba2 layer, blocks of 32) | 1.2e-6 when SSM **and conv** state are carried; 1.74 if the conv state is dropped |
    | Mamba-1 initial state | none: `selective_scan_fn(..., return_last_state=False)` has no initial-state input; only `Mamba.step(conv_state, ssm_state)` (single-token inference) |
    | Layer throughput fwd+bwd, fp32 | Mamba-1 ~0.50-0.56 M tok/s, Mamba-2 ~0.51-0.52 M tok/s at L=256/1024/4096 (fp16: Mamba-1 1.1-1.3 M, Mamba-2 0.52 M) |
  - GPU tests on T4 x2 (`results/kaggle/gputest_t4x2/output/pytest/`): all 5 `test_gpu_*` pass (`test_gpu_mamba2_manual_forward_matches_library`, `test_gpu_mamba2_state_continuation_and_grad`, `test_gpu_cbsc_equivalence[1]`, `[2]`, `test_gpu_phase2_param_match`). Full suite there: 106 passed, 54 failed. 53 failures are CPU tests in `test_models.py` that assume the GRU stand-in, but `mamba_ssm` is importable on Kaggle so they build real Mamba on CPU tensors; 1 is `test_provenance::test_kaggle_detection` seeing real Kaggle env vars. All 53 launcher tests passed on Kaggle too.
  - Training speed on T4 (fp32, B=8, T=256, two runs in parallel on T4 x2; `results/kaggle/gputest_t4x2/output/time_*/timing.json`): Transformer-Only 6L **11.05 steps/s** (1.5 h per 60k steps), Mamba-Only 13L **9.40** (1.8 h), ABR-4L-CBSC **7.92** (2.1 h), ABR-4L-gatedCBSC **3.75** (4.4 h; plus ~5 min Triton JIT). Final full eval not included (est. a few min).
  - Plan estimate (`queue.py plan` logic, `run_costs.json`): 46 runs = 105-119 lane-hours = **5-6 packed T4x2 sessions** of 11.5 h (5 if the sweep uses ABR-4L-CBSC, 6 if ABR-4L-gatedCBSC), i.e. ~58-69 session-hours, about 2-2.3 weeks at 30 h/week if Kaggle counts session hours.
  - **GPU time used: ~127 min** (compat check 100 min, of which 75 min was the all-arch source build; T4x2 test 27 min).
- **Problems / notes:**
  - No release wheels for torch 2.11/CUDA 13/cp313; the stock source build takes 75 min. Fixed: build only sm_75 (13 min) and install from `abr-mamba-wheels` (expected ~1 min; not yet exercised by a run). Kaggle strips `+` from uploaded file names; the runner restores it.
  - Kaggle P100 is retired; the default GPU shape is T4 x2, and CLI 1.7.4.5 cannot choose a shape anyway.
  - `kaggle kernels output` of the T4x2 test took > 1 h (1 GB of checkpoints plus thousands of Triton cache files) and timed out. The Triton cache now lives outside `/kaggle/working`, and the queue downloads only small files.
  - The first launch's `timeout_s` was recorded but not sent to Kaggle (fixed: `-t` is now passed).
  - The code archive is deterministic only without `git_snapshot.json` (the snapshot has a timestamp); its SHA-256 is still recorded.
  - Big binaries of the T4x2 test (`output/wheels/*.whl`, `triton_cache.tar.gz`, `*.pt`) are not committed; they stay in the Kaggle kernel version and the wheels dataset.
- **Decisions:**
  - Gated CBSC on real Mamba must use Mamba-2: run each Mamba block through the layer's own ops (in_proj, causal conv, `mamba_chunk_scan_combined` with `initial_states`/`return_final_states`, norm, out_proj), carrying the SSM state `(B, nheads, headdim, d_state)` = `(B, 8, 64, 64)` per layer **and** the conv state `(B, conv_dim, d_conv-1)` = `(B, 640, 3)`; gate only the SSM state. Mamba-1 cannot take an initial state; plain CBSC with Mamba-1 has to use the concatenate-the-Mamba-blocks approach.
- **Next:** lead merges `p1/kaggle`; fix the CPU tests to force the GRU stand-in when run on a GPU box; `queue.py init --commit <tag>` and `queue.py advance` to start Phase 2; update `run_costs.json` after the first real session.


### 2026-11-10 — Lead: merged Kaggle pipeline, Phase 2 launched
- Merged `p1/kaggle` (c343f17). Compat check on 2× T4: Mamba-1 and Mamba-2 forward/backward OK; Mamba-2 initial-state passing OK. All 5 `test_gpu_*` pass on Kaggle, including the param-match test.
- Fixed the CPU tests that failed on Kaggle (they built real Mamba because `mamba_ssm` is installed); they now force the GRU stand-in (1b0a5b2). Local: 161 passed, 5 skipped.
- Measured T4 speed (fp32, B=8, T=256, two runs in parallel): Transformer 6L 11.05 steps/s (1.5 h/run), Mamba 13L 9.40 (1.8 h), ABR-4L-CBSC 7.92 (2.1 h), ABR-4L-gatedCBSC 3.75 (4.4 h).
- Queue initialised at commit 1b0a5b2 with the 40 main runs; the 6 block-sweep runs are added after the main results pick the best 4L ABR variant (rule fixed in prd.md). The estimate is 5 packed sessions.
- **Session q001 launched** 2026-11-09T23:26Z: https://www.kaggle.com/code/parthgarg101/abr-q001 (10 runs across 2 lanes).

### 2026-11-10 — Lead: session q001 finished; queue bug; q002 discarded
- **q001** (11:07Z, complete): 9 runs finished at 60k steps; 1 was cut off by the budget and resumes next session (Transformer-Only s123, 0.70 h done). Outputs are in `results/kaggle/queue/q001/output/`. Measured hours per run: ABR-4L 2.2–2.3, gated 4.2, Jamba-1:3 10L 2.0, Transformer 6L 1.7.
- **Bug:** `queue.py` settled runs from `<session>/<run>/`, but the runner writes `<session>/output/<run>/`. Every q001 run therefore looked unstarted, and `advance` launched **q002 at 11:09Z with the same 10 runs from scratch**. Fixed in `_settle`; the test now uses the real layout, and it fails on the old code.
- **q002** is marked `discarded` in `queue.json` and was cancelled in the Kaggle UI. None of its outputs are used; it is a duplicate launch, not a re-run picked for a better result. The queue state was rebuilt from q001's outputs with the fixed code.
- Mounted wheel binaries are excluded from downloads and git.
- **Early numbers, 1–2 seeds each, not yet analysed** (val PPL): Jamba-1:3 10L 70.4–70.9; Transformer 6L 78.4; ABR-4L-CBSC 104.0; ABR-4L-gatedCBSC 104.5; ABR-4L-noCBSC 110.2–113.6; Random-Routing-4L 110.3–110.4.
