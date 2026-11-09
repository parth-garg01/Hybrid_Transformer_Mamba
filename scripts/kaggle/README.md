# Kaggle GPU pipeline

Rule (PRD §0.3): anything that runs a Mamba layer runs on Kaggle with the official
`mamba_ssm`; everything else runs locally. This folder holds the launcher that ships
an exact commit of this repo to a Kaggle GPU kernel.

| File | What it is |
|---|---|
| `launch.py` | CLI: package a commit, upload it, push the kernel, wait, download |
| `runner_template.py` | The kernel script (placeholders filled in by `launch.py`) |
| `compat_check.py` | Mamba-1/Mamba-2 GPU compatibility check (runs on Kaggle) |

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
* `--gpu t4|p100` is written to `kernel-metadata.json` as `machine_shape`, but Kaggle CLI
  1.7.4.5 (the newest that installs on the CLI venv's Python 3.9) does not send it, so
  Kaggle assigns its default accelerator. The real GPU is logged in `kaggle_env.json`.
* `--timeout SECONDS` caps the kernel run time.
* `--dry-run --keep-staging DIR` builds the archive and kernel folder without network.

The launcher prints the kernel URL and writes `results/kaggle/<name>/launch.json` with the
commit, archive SHA-256, dataset version, kernel slug, kernel version and UTC timestamps
(launched, pushed, finished, downloaded) and the final status.

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
3. Installs `causal-conv1d` and `mamba-ssm`: looks up the newest GitHub release wheel that
   matches the preinstalled torch major.minor, CUDA major, Python tag and cxx11abi, and
   `pip install --no-deps`s it. If no wheel matches, falls back to
   `pip install --no-build-isolation` (which itself tries the release wheel, then compiles;
   compiling is slow). Then installs `datasets`, `transformers<5`, `wandb`, and saves
   `pip_freeze.txt`.
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
* Install takes a few minutes per kernel (prebuilt wheels). A source build can take 30+ min:
  if `kaggle_env.json` shows `"method": "pip_no_build_isolation"`, check `run.log`.

## Errors

* **GPU/internet refused, or the kernel ran without a GPU:** the account is not
  phone-verified. Log in as the owner, go to https://www.kaggle.com/settings, verify the phone
  number, and relaunch. The launcher detects this from the push response, from the kernel
  metadata pulled back after the push, and from `no_gpu` in `kaggle_env.json`.
* **GPU quota exceeded:** wait for the weekly reset.
* **`code snapshot commit ... != expected`:** the dataset version used by the kernel isn't the
  one just uploaded (a race). Relaunch.
* **Dataset is public:** the launcher refuses to upload code to a public `abr-code`.
