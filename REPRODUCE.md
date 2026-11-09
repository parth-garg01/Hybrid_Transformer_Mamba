# Reproducing the results

This file gives the exact commands to (a) run the tests, (b) reproduce a main result end to
end, and (c) regenerate every table and figure from the saved results (PRD section 0.4 E).
Sections marked **TBD** depend on scripts that do not exist yet; they are filled in as
the scripts land.

## 0. Environment

| Where | What runs there | Environment record |
|---|---|---|
| Local (macOS, Apple Silicon, PyTorch `mps`/CPU) | code, CPU unit tests, Transformer-Only runs, aggregation, tables, figures | `requirements-lock.txt` (local dev lock) |
| Kaggle (NVIDIA T4/P100, official `mamba_ssm`) | every training, evaluation and benchmark of a model with a Mamba layer | recorded per run in `result.json` (`meta.pip_freeze`, `meta.kaggle`, CUDA/torch/`mamba_ssm` versions, GPU name and driver) |

Local setup:

```bash
python3 -m venv ~/.venvs/abr
~/.venvs/abr/bin/pip install -r requirements-lock.txt
```

`requirements-lock.txt` is the local dev lock only. The Kaggle environment is not pinned by
this file; each Kaggle run records its own full `pip freeze` and versions in its result.

Results produced with the GRU stand-in for Mamba (`meta.mamba_backend != "mamba_ssm"`) are
for debugging only and are never reported as Mamba results.

## 1. Run the tests

CPU only; Mamba layers use the GRU stand-in, so these check wiring and gradients, not quality.

```bash
python -m pytest -q                              # all unit tests in tests/
python scripts/verify_provenance.py results      # provenance of every committed result
```

CI (`.github/workflows/tests.yml`) runs the same two commands on every push and pull
request (Ubuntu, Python 3.11, CPU PyTorch from the PyTorch CPU wheel index,
`requirements-dev.txt`).

## 2. How a run is recorded (provenance)

Every training/evaluation script uses `experiments/full_scale/provenance.py`:

1. `start_run(config, debug=...)` refuses to start if tracked files have uncommitted
   changes. `--debug` overrides this and marks the run `reportable: false`.
2. The run records the git commit/branch/describe, the command line, UTC timestamps, host,
   GPU, CUDA/PyTorch/`mamba_ssm` versions, the full `pip freeze`, the SHA-256 of the
   tokenized dataset, Kaggle kernel info and the W&B run URL.
3. `finish_run(...)` writes `results/<run_id>/result.json` and copies the complete raw log
   to `results/<run_id>/train.log`, unmodified.

After adding results, regenerate the integrity manifest and verify before committing:

```bash
python scripts/make_manifest.py results          # writes results/MANIFEST.sha256
python scripts/verify_provenance.py results      # must print PASS
git add results && git commit -m "Pn.m: add results for ..."
```

The manifest uses the `sha256sum` format, so it can also be checked independently with
`cd results && shasum -a 256 -c MANIFEST.sha256`.

## 3. Launching a run

### 3a. Locally (non-Mamba models only, e.g. Transformer-Only)

**TBD** (training entry point and flags are being rewritten in Phase 1). The pattern:

```bash
git status                       # must be clean, or the script refuses to start
python experiments/full_scale/<train_script>.py --config <name> --seed 42
# results land in results/<run_id>/{result.json,train.log}
```

Local runs log to W&B online (project `abr-cbsc`).

### 3b. On Kaggle (any model with a Mamba layer)

**TBD** (the Kaggle launcher is being written in Phase 1). The workflow (PRD section 0.3):

1. Commit everything. If the code is uploaded without its `.git` directory, write a git
   snapshot first so the run can still prove which commit it came from (and is refused if
   any tracked file was changed in the copy):
   ```bash
   python experiments/full_scale/provenance.py snapshot      # writes git_snapshot.json
   ```
2. Upload the code as a private Kaggle dataset/notebook and launch with
   `kaggle kernels push` (GPU and internet on). The kernel installs `mamba_ssm` and
   `causal-conv1d` at start, and the launcher sets `KAGGLE_KERNEL_URL` so it is recorded.
3. The kernel logs to W&B with `WANDB_MODE=offline` (no API key on Kaggle).
4. Fetch outputs: `kaggle kernels output <user>/<kernel> -p results/`.
5. Sync W&B locally: `wandb sync <path to offline run dir>`.
6. Run `make_manifest.py` and `verify_provenance.py`, then commit the results.

Never delete Kaggle kernel versions; they are the third-party record of each run.

## 4. Reproduce one main result end to end

**TBD** (after Phase 2). This will name one main result (for example ABR+CBSC, seed 42),
the exact commit/tag, the Kaggle command, and the expected numbers with tolerance.

## 5. Regenerate every table and figure from saved results

**TBD**: `scripts/make_tables.py` and `scripts/make_figures.py` do not exist yet.
Planned commands:

```bash
python scripts/verify_provenance.py results
python scripts/make_tables.py results       # writes paper/generated/*.tex (local only)
python scripts/make_figures.py results
```

## 6. Known sources of nondeterminism

**TBD**: document which CUDA kernels (including `mamba_ssm` selective scan) are
nondeterministic, and the run-to-run variance observed with fixed seeds.
