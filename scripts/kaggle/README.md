# Kaggle GPU pipeline

Rule (PRD §0.3): anything that runs a Mamba layer runs on Kaggle with the official
`mamba_ssm`; everything else runs locally. This folder holds the launcher that ships
an exact commit of this repo to a Kaggle GPU kernel.

| File | What it is |
|---|---|
| `launch.py` | CLI: package a commit, upload it, push the kernel, wait, download |
| `runner_template.py` | The kernel script (placeholders filled in by `launch.py`) |
| `compat_check.py` | Mamba-1/Mamba-2 GPU compatibility check (runs on Kaggle) |
| `time_train.py` | steps/s of a short `train.py` run (real schedule, `--stop-after-steps`) |
| `queue.py` | Job queue for the Phase 2 runs: 2 runs per T4 x2 session, resume across sessions |

Prerequisites: Kaggle CLI at `~/.venvs/kaggle/bin/kaggle`, authenticated through
`~/.kaggle/kaggle.json` (chmod 600; never print, copy or commit it). The account must be
**phone-verified** or Kaggle refuses GPU and internet.

## Launch

```bash
git commit ...   # the launcher packages a commit, never the working tree
python scripts/kaggle/launch.py --name compat-check --commit HEAD \
    --cmd "python scripts/kaggle/compat_check.py" \
    --gpu t4 --wait --download results/kaggle/compat_check
```

* `--name` becomes the kernel slug `parthgarg101/abr-<name>` (lowercase, digits, `-`).
* `--commit` is any sha or ref. `git archive <sha>` is packed into `abr-code.tar.gz`
  (deterministic, so its SHA-256 identifies the code) and uploaded as a new version of the
  **private** dataset `parthgarg101/abr-code` (created on first use). The version notes
  contain the sha. The launcher waits until that version is ready before pushing.
* `--cmd` runs with bash inside `/kaggle/working/repo`. Write outputs to
  `$ABR_OUTPUT_DIR` (= `/kaggle/working/output`), which is what gets downloaded.
* `--gpu t4|t4x2|p100` is written to `kernel-metadata.json` as `machine_shape`. Kaggle's
  `NvidiaTeslaT4` shape *is* "GPU T4 x2" and is the server default; P100 is retired (the
  server substitutes T4 x2). Kaggle CLI 1.7.4.5 (the newest that installs on the CLI venv's
  Python 3.9) does not send `machine_shape` at all, so every GPU kernel gets the default.
  The real GPU(s) are logged in `kaggle_env.json`.
* `--timeout SECONDS` caps the kernel run time (passed to `kaggle kernels push -t`).
* `--cache-dataset parthgarg101/abr-wikitext103-gpt2-tokens` mounts the token cache, exports
  `ABR_CACHE_DIR`, records each file's SHA-256 in `kaggle_env.json["token_cache"]` and refuses
  to run if a hash differs from the local `~/.cache/abr` copy (`--cache-local`).
* `--wheels-dataset OWNER/SLUG` mounts prebuilt mamba/causal-conv1d wheels (and Triton cache).
* `--dry-run --keep-staging DIR` builds the archive and kernel folder without network.

The launcher prints the kernel URL and writes `results/kaggle/<name>/launch.json` with the
commit, archive SHA-256, dataset version, kernel slug, kernel version and UTC timestamps
(launched, pushed, finished, downloaded) and the final status.

## Two runs per session (T4 x2)

Give several commands; each becomes a named run with its own GPU:

```bash
python scripts/kaggle/launch.py --name pair1 --commit <sha> --gpu t4x2 \
    --run ABR-4L-CBSC_s42  --cmd "python experiments/full_scale/train.py --config ABR-4L-CBSC --seed 42 --out \$ABR_OUTPUT_DIR --device cuda --wandb offline --max-hours 11" \
    --run Jamba-1_3_s42    --cmd "python experiments/full_scale/train.py --config Jamba-1:3 --seed 42 --out \$ABR_OUTPUT_DIR --device cuda --wandb offline --max-hours 11"
# or: --cmds-file runs.json   with [{"name": "...", "cmd": "..."}, ...]
```

Run *i* gets `CUDA_VISIBLE_DEVICES=i % n_gpus` (extra runs queue up behind the first two),
`ABR_OUTPUT_DIR=/kaggle/working/output/<run>/`, `WANDB_DIR=<that>/wandb`, its own
`output/<run>/run.log`, and `output/<run>/run_status.json` (state `done` / `failed` /
`preempted` / `running`, exit code, session count, history). `ABR_SESSION_DEADLINE`
(epoch s) is exported; at 11.5 h the runner sends SIGTERM so the run can checkpoint.
A single unnamed `--cmd` keeps the simple layout (`output/run.log`).

**Resume:** pass `--resume-from <previous kernel>`. Before a named run starts, the runner
copies the previous session's `output/<run>/` (checkpoints, logs, W&B files) back into the
new `output/<run>/`, moves the old `run.log` to `previous_sessions/run.session<N>.log`, and
sets `ABR_RESUMED=1`. `train.py` then resumes from
`$ABR_OUTPUT_DIR/checkpoints/<run_name>/ckpt.pt` by itself.

## Job queue (packed sessions)

```bash
python scripts/kaggle/queue.py init --commit <sha> [--sweep <LABEL>] [--exclude-non-mamba]
python scripts/kaggle/queue.py plan       # next session's lane plan + estimated session count
python scripts/kaggle/queue.py advance    # THE command: run it whenever (cron, by hand)
python scripts/kaggle/queue.py status
python scripts/kaggle/queue.py add-sweep <LABEL>   # once the sweep config is chosen
python scripts/kaggle/queue.py retry <run_id>
```

* `init` builds the queue from `config.planned_runs()` (40 runs), ordered key 5-seed configs
  first, seed-major (every key config gets seed 42 before any gets seed 123), then the 2-seed
  configs, then the 6 block-sweep runs. Defaults: `--cache-dataset
  parthgarg101/abr-wikitext103-gpt2-tokens` (expected SHA-256s read from `~/.cache/abr` and
  stored in the queue), `--wheels-dataset parthgarg101/abr-mamba-wheels`. All sessions use
  the queue's commit.
* **Packed sessions.** Each session is one T4 x2 kernel `abr-qNNN` with two GPU lanes. A lane
  runs its runs back to back until `--session-hours` (11.5) is used up. A run's
  `--max-hours` is the lane's remaining budget when it starts, so the run cut off at the end
  checkpoints cleanly; a lane does not start a run with < 20 min left (that run stays
  pending, untouched). The plan takes partial runs first, then the queue order, until both
  lanes are full, and splits them longest-job-first using `run_costs.json` (steps/s measured
  on Kaggle). Partial runs go first in their lane. The plan is stored in the session's
  `launch.json` (`session_plan`) and in `queue.json`.
* **`advance`** is idempotent: it checks the active session; if the kernel finished and is not
  downloaded yet it downloads it (small files only: logs, `run_status.json`, `result.json`,
  `kaggle_env.json`, W&B files; checkpoints, wheels and the Triton cache stay on Kaggle) and
  settles each run; then, if fewer than `--max-sessions` (default 1) sessions run and runs
  remain, it launches the next packed session. Running it again while a session runs does
  nothing.
* Settling a run: **done** if exit 0 and a `result.json` exists; **pending** with
  `resume_from = that kernel` if it stopped on its budget or was killed (the next session
  copies its `output/<run>/` back, and `train.py` resumes from its checkpoint); a non-zero exit
  (or a kernel error before it started) counts as a failure, and after `--max-failures` (2)
  it is **failed**. Failed runs stay listed (PRD §0.4 B).
* State: `results/kaggle/queue.json`; per-session outputs in `results/kaggle/queue/qNNN/`.

## Re-attach, wait and download later

```bash
python scripts/kaggle/launch.py --name compat-check --attach --wait --download results/kaggle/compat_check
```

`--attach` does not push anything; it polls `kaggle kernels status` with backoff (30 s,
×1.5, max 5 min) and then runs `kaggle kernels output`. Plain CLI equivalent:
`kaggle kernels status parthgarg101/abr-<name>` and
`kaggle kernels output parthgarg101/abr-<name> -p results/kaggle/<name>`.

## Resume a long run in a new session

Sessions stop at about 12 h. Training must checkpoint (at least every 2,000 steps) into
`$ABR_OUTPUT_DIR`. To continue, launch a new kernel with the previous kernel's output
attached:

```bash
python scripts/kaggle/launch.py --name run-a-s2 --commit <same sha> \
    --resume-from parthgarg101/abr-run-a-s1 --cmd "python ... --resume-dir \$ABR_RESUME_DIRS"
```

The previous output is mounted read-only under `/kaggle/input/...`; the runner exports
the paths in `ABR_RESUME_DIRS` (colon-separated). Use the **same commit** for every session
of a run.

## What the kernel does

1. Finds `abr-code.tar.gz` (or an auto-extracted copy) under `/kaggle/input`, extracts it to
   `/kaggle/working/repo` and **refuses to run** if its `COMMIT` file differs from the
   commit the launcher packaged.
2. Logs `nvidia-smi`, torch/CUDA/cxx11abi.
3. Installs `causal-conv1d` 1.7.0 and `mamba-ssm` 2.3.2.post1 (the versions the compat
   check validated), trying in order:
   1. wheels found under `/kaggle/input` whose local version matches this environment, e.g.
      `mamba_ssm-2.3.2.post1+cu130torch2.11sm75-cp313-cp313-linux_x86_64.whl` (attach a
      wheels dataset, see below): seconds;
   2. GitHub release wheels matching torch/CUDA/Python/cxx11abi (none exist for Kaggle's
      torch 2.11 + CUDA 13 + Python 3.13 as of 2026-11-09);
   3. a source build of the pinned sdists **for the running GPU's arch only** (setup.py is
      patched to emit one `-gencode`), saved to `output/wheels/`: about 10 min on a T4;
   4. stock `pip install --no-build-isolation` (all 9 archs): **75 min** on Kaggle.
   Then `datasets`, `transformers<5`, `wandb`, and `pip_freeze.txt`.
   The Triton cache (`TRITON_CACHE_DIR`) is restored from any mounted `triton_cache.tar.gz`
   and saved to `output/triton_cache.tar.gz` (Mamba-2 kernels JIT-compile for minutes on a T4).
   Make the install fast for every later session by uploading `output/wheels/*.whl` and
   `output/triton_cache.tar.gz` of one finished kernel as a private dataset and attaching it
   with `--wheels-dataset <owner>/<slug>`.
4. Sets `WANDB_MODE=offline`, `WANDB_DIR=$ABR_OUTPUT_DIR/wandb`, `KAGGLE_KERNEL_URL`,
   `CODE_COMMIT`, `KAGGLE_DATASET_VERSION`, `ABR_OUTPUT_DIR`, `ABR_RESUME_DIRS`
   (and `KAGGLE_KERNEL_VERSION` only if the platform exposes it, which it normally does not).
5. Runs the command; all output (install logs included) goes to `output/run.log`.
6. Writes `output/kaggle_env.json`: GPU name, compute capability, driver, CUDA, torch,
   mamba_ssm/causal_conv1d versions and install method, Python, timestamps, dataset version,
   commit, exit code. The kernel exits with the command's exit code.
7. Deletes `/kaggle/working/repo` so the outputs hold results only (the code is in the
   dataset version).

After download, sync W&B locally: `wandb sync results/kaggle/<name>/wandb/offline-run-*`.

## Notes for the provenance agent and the lead

* **There is no `.git` inside the kernel.** The commit is passed explicitly: the archive has
  `repo/COMMIT` (full sha) and `repo/COMMIT_REF`, and the runner exports `CODE_COMMIT`.
  Provenance code on Kaggle must read `CODE_COMMIT` (or `COMMIT`) instead of calling git.
* If the packaged commit contains `experiments/full_scale/provenance.py`, the launcher runs
  `python experiments/full_scale/provenance.py snapshot` in a temporary clone of exactly that
  commit and adds the resulting `git_snapshot.json` to the archive root (`repo/git_snapshot.json`).
  `start_run` can then verify the SHA-256 of every tracked file on Kaggle. If the module is
  absent, only `COMMIT` is shipped. The runner records `git_snapshot_present` and its SHA-256.
* The kernel version number is not visible inside the kernel. It is recorded locally in
  `results/kaggle/<name>/launch.json` (`kernel_version`), which should be merged into result
  JSONs after download. `KAGGLE_KERNEL_URL` is set inside the kernel.
* Kaggle keeps every kernel version (code, log, output). Never delete kernel versions or
  dataset versions that results depend on.

## Quota

* About **30 GPU hours per week** (resets weekly) and **12 h per session**. Check the
  remaining quota at https://www.kaggle.com/settings.
* Log GPU time per run and a weekly running total in `project.md`. The run time is
  `finished_utc - started_utc` in `kaggle_env.json`.
* T4 x2 runs two jobs per session, so 30 GPU h/week is about 30 h of wall time for 2 runs
  if Kaggle counts session hours rather than per-GPU hours. Check the quota meter on the
  settings page after the first T4 x2 session to confirm.
* Install cost per session: about 1 min from the wheels dataset; 13 min building for sm_75
  only; 75 min with the stock all-arch build. Check `kaggle_env.json["mamba_install"]["method"]`.
* `kaggle kernels output` downloads everything (1 GB and > 1 h for a session with checkpoints
  and a Triton cache directory). The queue downloads selectively (about 20 s).

## Errors

* **GPU/internet refused, or the kernel ran without a GPU:** the account is not
  phone-verified. Log in as the owner, go to https://www.kaggle.com/settings, verify the phone
  number, and relaunch. The launcher detects this from the push response, from the kernel
  metadata pulled back after the push, and from `no_gpu` in `kaggle_env.json`.
* **GPU quota exceeded:** wait for the weekly reset.
* **`code snapshot commit ... != expected`:** the dataset version used by the kernel isn't the
  one just uploaded (a race). Relaunch.
* **Dataset is public:** the launcher refuses to upload code to a public `abr-code`.
