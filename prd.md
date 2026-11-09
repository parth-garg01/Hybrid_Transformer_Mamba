# PRD: Fixing and Strengthening ABR + CBSC (Hybrid Transformer–Mamba)

**Owner:** Parth Garg
**Repo:** `parth-garg01/Hybrid_Transformer_Mamba`
**Paper source:** `paper/main.tex` (IEEE conference format, compiles to 8 pages). **The paper is kept locally only and is never committed or pushed** (`paper/` is in `.gitignore`).
**Goal:** Make the code correct, run fair experiments, and strengthen the contribution so the paper can be submitted to a Q3 journal (after Phase 2) or a Q2 journal (after Phase 3).

---

## 0. Ground rules (read first)

### 0.1 Progress log: `project.md` (mandatory)

Before writing any code, **create `project.md` in the repo root** using the template in Section 8.
Then **update it after every task you finish** (not only at the end of a phase). Each entry must say:

- the date,
- the task ID from this PRD (for example `P1.2`),
- what was done, in 2–5 plain sentences,
- the files that changed,
- the results or numbers produced (with the path to the results file),
- any problem found, and anything decided differently from this PRD (with the reason),
- what comes next.

Also keep the **Status board** at the top of `project.md` in sync: each task is `todo`, `in progress`, `done` or `blocked`.
`project.md` is the single source of truth for progress. Anyone (human or AI) picking up the work should be able to read it and continue without asking questions.

### 0.2 Other rules

- Follow the git rules in the global `~/.claude/CLAUDE.md`. Commit as `Parth Garg <parth.garg2024@vitstudent.ac.in>` and never add AI co-author lines. Make one commit per finished task, with the task ID in the message (for example `P1.1: give router a gradient path`).
- **Keep the original behaviour reproducible.** Add a `legacy=True` option (or a `--legacy` flag) that reproduces the code exactly as it is today, so the numbers in the current paper can still be regenerated. All new behaviour is the default.
- No silent fallbacks in experiments. `models_scaled.py` silently swaps Mamba for a GRU when `mamba_ssm` is missing. Training and benchmark scripts must **refuse to run** if real Mamba is unavailable, unless a `--allow-gru-fallback` flag is passed. Every result file must record which one was used.
- Every result JSON must record: git commit hash, full config, seed, parameter count, train tokens seen, wall-clock time, GPU name, and the Mamba backend in use.
- Don't change the paper text until Phase 4.

### 0.3 Where things run (compute plan)

The owner no longer has the RTX 2050 laptop. Work is split by one rule:

**Anything that uses Mamba runs on Kaggle. Everything else runs locally on the Mac.**

| Runs on **Kaggle** (NVIDIA GPU, official `mamba_ssm`) | Runs **locally** (Apple M5 Mac, PyTorch `mps`/CPU) |
|---|---|
| Any training, evaluation or benchmark of a model that contains a Mamba layer: Mamba-Only, Jamba-1:1, Jamba-1:3, Alternate-Blocks, Random-Routing, all ABR variants | All code writing and editing |
| Phase 1 GPU checks: `test_cbsc_equivalence` with real Mamba, P1.6 smoke runs | All unit tests in `tests/` (using the GRU stand-in for Mamba; these check wiring and gradients, not quality) |
| Throughput/memory benchmarks of Mamba-containing models (P2.5) | Transformer-Only training, evaluation and benchmarks |
| | Aggregating results, statistics, tables, figures, router analysis (P3.4) on downloaded outputs, and the paper (Phase 4) |

Rules:
- **Never report numbers from the GRU stand-in as Mamba results.** Locally, Mamba-containing models may only be run for debugging and tests.
- For throughput comparisons, benchmark **all** models, including Transformer-Only, **on the same Kaggle GPU**, so the speed numbers are comparable. Local Transformer timings are for development only.
- **Kaggle workflow:**
  - Code is uploaded as a private Kaggle dataset or notebook and launched with `kaggle kernels push`, with GPU and internet enabled. The account must be phone-verified.
  - `mamba_ssm` and `causal-conv1d` are installed at the start of each kernel, and the installed versions are logged.
  - Outputs are fetched with `kaggle kernels output` into `results/` locally.
  - The API token lives in `~/.kaggle/kaggle.json` (chmod 600). Never print it, commit it, or copy it into notebooks or logs.
- **Kaggle limits:** about 30 GPU hours per week, and about 12 hours per session.
  - Every run must checkpoint, at least every 2,000 steps, and resume across sessions.
  - Log the GPU hours used per run in `project.md`, and keep a running weekly total.
- **GPU type:** Kaggle offers T4 or P100.
  - The first Kaggle job is a short **compatibility check**: does Mamba-1, and Mamba-2 (needed for gated CBSC state passing), work on the assigned GPU?
  - If Mamba-2 kernels fail, use the block-by-block state-passing fallback from P1.3 and record the decision.
- **Seed budget**, to fit the weekly quota:
  - 5 seeds for the key comparisons: ABR+CBSC, ABR+gated CBSC, ABR no-CBSC, Random-Routing, Jamba-1:3 and Transformer-Only.
  - 3 seeds for the other baselines and the block-size sweep.
  - Record any change to this in `project.md`.
- Phase 3 scale-up (125M–350M) needs A100-class GPUs and does not fit Kaggle's free tier. Decide the compute source (Colab Pro, academic credits, or a reduced scale) before starting P3.2.

### 0.4 Research integrity and provenance (mandatory, applies to every task)

**Goal:** any reviewer, editor or reader must be able to verify, from independent timestamped records, that:
- the code existed when we say it did,
- every reported number came from that code,
- nothing was hand-edited or cherry-picked.

**A. Git history is the audit trail**
- **Signed commits.** Every commit is signed with an SSH signing key registered on the `parth-garg01` GitHub account, so GitHub shows **Verified** on each one. Set this up in P1.0 (`git config gpg.format ssh`, `user.signingkey`, `commit.gpgsign true`, and register the key on GitHub as a *signing* key).
- **Small, frequent commits.** One per task, with honest messages that describe what changed and why.
- **Never rewrite history.** No `--force` pushes, no rebasing or amending of pushed commits, and no deleting branches that results depend on. Mistakes are fixed with new commits, and the mistake stays visible.
- **Push regularly.** GitHub's push timestamps are an independent record.
- **Tag every milestone** (`m1-phase1-fixed`, `m2-phase2-results`, …), and record the tag in `project.md`.

**B. The plan is fixed before the results exist (pre-registration)**
- Commit and push this PRD, including the full list of models, seeds, metrics and statistical tests, **before** any Phase 2 run starts. This timestamped commit is the pre-registration.
- If the plan changes later, record the change and the reason in `project.md` under *Decisions and deviations*, **before** looking at the affected results where possible.
- **Report every run that was started.** Failed, crashed or bad runs are listed too, with the reason. No run is silently dropped or re-seeded to get a better number.

**C. Every result is traceable to exact code, data and environment**
- **Clean code only.** Training and evaluation scripts **refuse to start** if the git working tree has uncommitted changes (overridable only with `--debug`, which marks the run as non-reportable).
- **What each result JSON records:**
  - the git commit hash and tag,
  - the full config, seed and command line,
  - start and end UTC timestamps,
  - the GPU model, driver, and CUDA/PyTorch/`mamba_ssm` versions,
  - a full `pip freeze`,
  - the **SHA-256 of the tokenized dataset**,
  - the Kaggle kernel URL and version number.
- **Keep raw logs.** Save complete stdout/stderr per run (`results/<run>/train.log`) next to the JSON. Don't trim or edit them.
- **Integrity manifest.** `results/MANIFEST.sha256` lists the SHA-256 of every result file. It is regenerated by a script and committed with the results.
- **Kaggle versions as third-party evidence.** Kaggle keeps an immutable, timestamped copy of each kernel version with its code, logs and outputs. Never delete kernel versions. Make the kernels public when the paper is submitted.
- **Weights & Biases (required).** Log every reportable run to W&B (project `abr-cbsc`; entity = the owner's W&B account). Log the config, the git commit, the metrics curves, router stats and the final results. Record the W&B run URL in each result JSON.
  - Local runs log online.
  - Kaggle runs log with `WANDB_MODE=offline` into the kernel output. After downloading the output, sync it locally with `wandb sync`, so the W&B API key never has to be stored on Kaggle. The run's own timestamps are kept in the logged files.
  - The W&B API key lives only in the owner's local `~/.netrc` (created by `wandb login`). Never print, commit or upload it.

**D. No hand-typed numbers in the paper**
- Every number, table and figure in the paper is generated by scripts (`scripts/make_tables.py`, `scripts/make_figures.py`, which **are** committed) from the committed result files.
- The scripts write tables to `paper/generated/*.tex` (local only, like the rest of `paper/`), which the paper pulls in with `\input`.
- Reviewers can rerun the scripts and get the same PDF content.

**E. Anyone can reproduce it**
- Pin the environment: `requirements-lock.txt` and the exact Kaggle docker image or version.
- Deterministic seeding (`torch.manual_seed`, `numpy`, `random`, and `torch.use_deterministic_algorithms` where supported). Document any nondeterminism that remains, such as CUDA kernels.
- A `REPRODUCE.md` with the exact commands to:
  - (a) run the tests,
  - (b) reproduce one main result end to end,
  - (c) regenerate every table and figure from saved results.
- **Continuous integration.** A GitHub Actions workflow runs the CPU unit tests on every push, giving an independent, public record that the tests passed at each commit.

**F. Archive and release**
- At submission, create a GitHub release and archive it on **Zenodo**, which gives an immutable, citable snapshot with a DOI. Cite the DOI in the paper.
- Upload trained checkpoints, with SHA-256 hashes, to Hugging Face Hub or Zenodo.
- Post the paper to arXiv when submitting. This gives a public, timestamped record of the work.
- Add `CITATION.cff` and keep the `LICENSE`.

**G. Honesty about the old results and about tools**
- The current paper's numbers have **no surviving raw logs** (the original machine is gone). Mark them in `project.md` as unverifiable. **Do not reuse them** in any new submission; replace every number with results from this pipeline.
- **Disclose AI assistance.** If AI coding or writing tools were used, disclose this as the target journal's policy requires, usually in the acknowledgements or methods. Most publishers (IEEE, Springer, Elsevier) now require it, and an undisclosed use that is discovered later is a much bigger authenticity problem than a disclosed one.
- Record in `project.md` which tools were used for which parts of the work.

**Acceptance for 0.4:**
- `scripts/verify_provenance.py` checks that every result file has the required metadata, that its commit exists in the repo, and that its hash matches the manifest.
- It runs in CI and must pass before any milestone tag is created.

---

### 0.5 Parallel work with multiple agents

Work may be split across several AI agents (or people) working at the same time. Rules:
- Each agent works in its own git worktree and branch (`p1/<task>`), and is merged into `fixes` by the lead after its tests pass.
- Each agent owns specific files. Shared interfaces (for example the provenance/run-metadata API) are agreed in this PRD or in `project.md` **before** parallel work starts.
- Every agent writes its own entries in `project.md`. The lead resolves merge conflicts in `project.md` by keeping all entries.
- All commits follow 0.2 and 0.4: signed, the owner's identity, no AI co-author lines.

## 1. Background: what is wrong today

These were confirmed by running the code (CPU, GRU fallback) and checking gradients.

| # | Problem | Where | Effect |
|---|---------|-------|--------|
| B1 | **Router gets no gradient.** Expert choice uses a hard `argmax` (`indices`), and the straight-through output `z` is never used. The routing probabilities passed to the aux losses are `.detach()`ed. | `experiments/full_scale/models_scaled.py:139-142`, `:180-188`, `:368` | Router weights never update (`router.linear.weight.grad is None`). Routing is pure Gumbel noise during training. Entropy ≈ ln 2 is caused by this, not by "slow convergence". |
| B2 | **Blocks are processed in isolation.** Each 32-token block goes through attention or Mamba on its own (`blocks.view(B*N, b, D)`). | `models_scaled.py:197-259` | A token cannot see anything before its block (measured gradient is exactly 0). Effective context is 32 tokens vs 256 for the baselines. This is the most likely cause of the big perplexity gap. |
| B3 | **CBSC is not what the paper says.** The code adds the previous block's last *output* vector (size `d_model`) to the next block's first *input* token. It does this only for directly adjacent Mamba→Mamba blocks, one hop only (pass 2 uses pass-1 outputs). It is not Mamba's hidden state. | `models_scaled.py:197-250` | Paper Eq. (5) and "regardless of how many Transformer blocks intervene" do not match the code. The gate is `d_model×d_model`, not `d_inner`. Zero-init gives 0.5 injection, not "same as baseline". |
| B4 | **Aux losses have no effect** (input detached). They are also applied to the last layer only. | `models_scaled.py:368`, `train_ablation.py:134-137` | λ_lb and λ_ent do nothing. |
| B5 | **Temperature updates per block, not per step.** `GumbelRouter.forward` is called N times per layer per step, and each call changes `self.temperature`. | `models_scaled.py:126-135`, `:181-184` | The schedule is 8× (×layers) faster than described. |
| B6 | **Evaluation uses 200 random windows**, not the full validation split. | `dataset.py:get_batch`, `train_ablation.py:48-57` | The paper's claim is wrong, and the numbers are noisy. |
| B7 | **Throughput benchmark** runs ABR-2L, but the paper names ABR-4L-CBSC-Gated. The figure shows only 2 models. | `bench_context.py` | Claim and figure don't match. |
| B8 | `np.std` uses population std (ddof=0) with 3 seeds. | `train_ablation.py:224` | Standard deviations are understated. |

---

## 2. Phase 1: Fix the code

**Target:** about 1–2 weeks. **Exit criterion:** all Phase 1 tests in `tests/` pass on CPU (GRU fallback) and on GPU (real Mamba).

### P1.0 Set up tests and `project.md`
- Create `project.md` (Section 8 template).
- Create `tests/` with `pytest`. All tests must run on CPU in under 2 minutes with tiny models (`d_model=32`, `T=64`, `block_size=8`).

### P1.1 Give the router a real gradient path (fixes B1)
- In `ABRLayer`, the output of each block must be multiplied by the straight-through weight of its chosen expert:
  `y_block = z[..., chosen] * expert_chosen(x_block)`.
  In the forward pass `z[..., chosen] == 1`, so outputs are unchanged. In the backward pass, gradient flows into the router through `z_soft`.
- Return `z` (not only `z_soft`) from the router to the layer, and use it.
- Optional (`router_mode="soft_topk"`): document the alternative of mixing both experts' outputs weighted by `z_soft` during warm-up, then switching to hard routing.
- **Acceptance:**
  - Test `test_router_receives_gradient`: after one backward pass, every layer's `router.linear.weight.grad` is non-zero, for all variants (OFF, CBSC, gated).
  - Test: with a fixed seed, outputs in eval mode equal the hard-routed outputs (no change in what the model computes, only in gradients).

### P1.2 Let every block see all earlier text (fixes B2)
- **Transformer expert:** queries come only from the Transformer-routed blocks, but keys and values come from **all earlier tokens of the same layer input**, with a causal mask.
  Implementation: compute K, V over the full sequence `(B, T, D)`, gather Q for the routed positions, and use a block-causal mask with `F.scaled_dot_product_attention` and `attn_mask`.
  Keep a simple reference version (full causal attention over the whole sequence, then select routed blocks) for correctness tests.
- **Mamba expert:** see P1.3. With true CBSC, Mamba blocks get context from all earlier Mamba-routed blocks.
- Optionally add a variant where the Mamba expert also gets a summary of intervening Transformer blocks. Report it as a separate ablation.
- **Acceptance:**
  - Test `test_cross_block_dependency`: for 20 random routing patterns, the output at a token in block k > 0 has non-zero gradient with respect to some input token in block 0.
  - Test `test_no_future_leakage`: the output at position t has **exactly zero** gradient with respect to every input position > t (this must hold for every variant, including legacy).

### P1.3 Make CBSC carry real Mamba state across the whole sequence (fixes B3)
- **Additive/plain CBSC:** for each sequence, run the Mamba expert over the **concatenation of all its Mamba-routed blocks, in order**, then scatter outputs back to their positions. This is the same as carrying Mamba's recurrent state from each Mamba block to the next Mamba block, whatever Transformer blocks sit in between.
  Use `mamba_ssm` variable-length/packed sequences, or pad per batch with a mask. With the GRU fallback, carry the GRU hidden state between blocks.
- **Gated CBSC:** gate the state handed from one Mamba block to the next:
  `h_init = sigmoid(W_g h_prev + b_g) ⊙ h_prev`.
  Use Mamba-2's chunked scan (`initial_states` / `return_final_states`) to pass and receive states block by block.
  Gate size must match the real state size (`d_inner × d_state` per head, or whatever the Mamba version exposes). Document the exact shapes in `project.md`.
  Initialise so that the gate starts **fully open** (bias large positive, e.g. +4), so step 0 equals plain CBSC. Or keep 0.5 but describe it honestly; record the choice.
- Keep the old behaviour behind `cbsc_mode="legacy_inject"`.
- **Acceptance:**
  - Test `test_cbsc_equivalence`: with every block routed to Mamba and CBSC on, the ABR layer output equals a plain Mamba layer run over the whole sequence (tolerance 1e-4).
  - Test `test_cbsc_skips_transformer_blocks`: with routing M-T-M, output in the third block depends on input in the first block.
  - Record the exact parameter overhead of the gate in `project.md`.

### P1.4 Turn the aux losses back on, for every layer (fixes B4, B5)
- Collect `z_soft` from **every** ABR layer (not detached). Compute load-balancing and entropy losses per layer and average them.
- Move the temperature update out of `GumbelRouter.forward` into an explicit `router.update_temperature(H)` called **once per training step per layer** by the training loop. Log τ per layer.
- **Acceptance:**
  - Test: the total loss has a gradient path to the router from the aux terms alone (zero out CE and check the gradient is non-zero).
  - Test: temperature changes at most once per `optimizer.step()`.

### P1.5 Evaluation and stats fixes (fixes B6, B8)
- Add `evaluate_full(model, split)`: deterministic sliding-window evaluation over the **entire** split (window 256, stride 128, only score the last 128 tokens of each window except the first). Report token-level cross-entropy and perplexity.
- Report the **test** split too (only once, at the end of each run).
- Use `np.std(..., ddof=1)`, and report 95% confidence intervals and Welch t-tests.
- **Acceptance:** two evaluation runs on the same checkpoint give identical numbers.

### P1.6 Smoke run
- Train each variant for 500 steps on GPU and record loss curves in `project.md`.
- Confirm that router entropy now **moves away from ln 2**, or explain why not.

---

## 3. Phase 2: Fair experiments (target: enough for a Q3 journal)

### P2.1 Matched comparison
- All models trained for the **same number of tokens** and the same steps, LR schedule and batch size.
- Match **parameter count** within ±5% and report **FLOPs per token** for each model (`fvcore` or analytic). If a model is larger, say so in the table.
- Everyone gets 60k steps (or more). No model is compared at a different budget.

### P2.2 Models to run

| Name | Description |
|------|-------------|
| Transformer-Only | 4 layers |
| Mamba-Only | 4 layers |
| Jamba-1:1 | fixed T-M-T-M |
| Jamba-1:3 | fixed T-M-M-M |
| Alternate-Blocks | ABR layer with fixed alternating block assignment (T,M,T,M…) |
| Random-Routing | ABR layer with random assignment (same expected attention fraction as the learned router) |
| ABR (no CBSC) | learned routing, fixed code |
| ABR + CBSC | learned routing, true state carry |
| ABR + gated CBSC | as above with gate |

Run 2-layer and 4-layer versions of the ABR variants.

### P2.3 Seeds and statistics
- **5 seeds** per configuration: `{42, 123, 7, 2024, 31337}`.
- Report mean ± std (ddof=1), 95% CI, and Welch t-tests for the key pairs (CBSC vs no-CBSC, gated vs plain, ABR vs Jamba, learned vs random routing).

### P2.4 Block-size sweep
- Block sizes **8, 16, 32, 64** for the best ABR variant (3 seeds is fine here). Plot perplexity and throughput against block size.

### P2.5 Throughput and memory (fixes B7)
- Benchmark **every** model in P2.2 at sequence lengths 256, 512, 1k, 2k, 4k (as far as memory allows): tokens/sec and peak memory, forward-only and training.
- For ABR, use the **routing fractions the trained model actually produces**, and report them next to the numbers.

### P2.6 Outputs
- `results/phase2/*.json` (one per run), `results/phase2/summary.csv`, and scripts that regenerate every table and figure from those files: `scripts/make_tables.py` and `scripts/make_figures.py`.
- Hardware note: the RTX 2050 (4 GB) is enough for the 15–20M models. Log the GPU and time per run in `project.md`.

---

## 4. Phase 3: Make it stronger (target: Q2 journal)

### P3.1 Attention-budget "dial"
- Add `attn_budget ∈ [0, 1]`: per sequence, only the top-`ceil(budget × N)` blocks by router score go to attention; the rest go to Mamba. This is like Mixture-of-Depths capacity. Train with the budget fixed, or sampled per batch so that one model supports several budgets.
- Produce a **quality-vs-speed curve**: perplexity and tokens/sec for budgets 0, 0.125, 0.25, 0.5, 0.75 and 1.0. Compare against Jamba-style models with matching attention fractions.
- This is the main selling point of the stronger paper. Present it as the headline figure.

### P3.2 Scale up
- Train **125M** and, if budget allows, **350M** parameter versions of the best ABR model and the main baselines (Transformer, Mamba, Jamba-1:3).
- Data: 2–5B tokens from FineWeb-Edu or SlimPajama (log the exact subset and tokenizer).
- Compute: Colab/Kaggle A100, or academic credits. Add checkpoint/resume support (already partly present) and log GPU-hours in `project.md`.

### P3.3 Broader evaluation
- **Recall/memory:** MQAR (multi-query associative recall) and passkey retrieval at several lengths.
- **Long documents:** PG-19 perplexity at 2k–8k context.
- **Downstream:** `lm-evaluation-harness` zero-shot on HellaSwag, PIQA, ARC-Easy, ARC-Challenge, WinoGrande and LAMBADA.
- Same seeds and reporting rules as Phase 2.

### P3.4 What the router learned
- Routing fraction by part of speech (spaCy tags), by token frequency, by token surprisal (loss), and by position in the sequence.
- Routing for each layer, and how it changes over training.
- Show 3–5 example passages colored by routing decision.
- Save plots to `results/phase3/router_analysis/`.

---

## 5. Phase 4: Update the paper

Do this after the experiments are finished. The paper stays local only; never commit or push anything under `paper/`.

- Make the method section (`paper/main.tex`, Sec. III) describe the **fixed** code exactly: equations, state shapes, gate initialization, aux losses on all layers, temperature updated per step.
- Replace the "router needs more training" explanation with evidence from Phases 2–3.
- Replace all tables and figures with ones regenerated by `scripts/make_tables.py` and `scripts/make_figures.py`.
- Fix reference [3] (MambaByte: J. Wang, T. Gangavarapu, J. N. Yan, A. M. Rush, 2024).
- Add related work: Block-State Transformer, Block-Recurrent Transformer, Recurrent Memory Transformer, MoE-Mamba, Samba, Hymba, TransMamba, and work on when content-based routing works in hybrid models.
- Use one symbol for block size everywhere (`b`).
- Check that it compiles with pdfLaTeX (Overleaf) with no errors.

---

## 6. Order of work and milestones

| Milestone | Tasks | Done when |
|-----------|-------|-----------|
| M1 | P1.0–P1.6 | All Phase 1 tests pass; smoke runs logged |
| M2 | P2.1–P2.6 | Phase 2 tables and figures regenerate from saved results |
| M3 | P3.1 | Quality-vs-speed curve produced |
| M4 | P3.2–P3.4 | Scaled models trained and evaluated; router analysis done |
| M5 | Phase 4 | Paper updated and compiles |

Don't start Phase 2 runs until M1 is done. The long runs are expensive, and they are wasted if the model is still wrong.

## 7. Risks

- **Router still collapses or stays random after the fix.** Try a soft-mixing warm-up (P1.1 option), a lower τ floor, or a larger λ_lb, and log each attempt in `project.md`. If learned routing never beats Random-Routing, report that honestly. The attention-budget dial and the correct CBSC are still contributions.
- **Packed Mamba is slow or awkward.** Fall back to looping over blocks with explicit state passing (Mamba-2 `initial_states`). It's slower but correct; record the throughput impact.
- **GPU memory.** Use gradient accumulation and bf16, and record the effective batch size.
- **Compute for Phase 3.** If 350M isn't affordable, 125M plus the budget-dial curve is still a solid Q2 submission. Note the decision in `project.md`.

---

## 8. `project.md` template

Create this file at the repo root before starting, and keep it updated after every task.

```markdown
# Project Log: ABR + CBSC Fixes

## Status board
| Task | Status | Last update |
|------|--------|-------------|
| P1.0 Tests + project.md | todo | |
| P1.1 Router gradient | todo | |
| P1.2 Cross-block context | todo | |
| P1.3 True CBSC | todo | |
| P1.4 Aux losses + temperature | todo | |
| P1.5 Full evaluation + stats | todo | |
| P1.6 Smoke runs | todo | |
| P2.1–P2.6 Fair experiments | todo | |
| P3.1 Attention-budget dial | todo | |
| P3.2 Scale up | todo | |
| P3.3 Broader evaluation | todo | |
| P3.4 Router analysis | todo | |
| Phase 4 Paper update | todo | |

## Key results so far
(Short table of the latest headline numbers, with links to result files.)

## Decisions and deviations from the PRD
(Date — decision — reason.)

## Log
### YYYY-MM-DD — <Task ID>: <short title>
- **Done:**
- **Files changed:**
- **Results:**
- **Problems / notes:**
- **Next:**
```
