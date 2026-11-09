#!/usr/bin/env python3
"""Launch a command from a committed snapshot of this repo on a Kaggle GPU.

Usage:
    launch.py --name <short> --commit <sha-or-ref> --cmd "<command run inside repo>"
              [--gpu t4|t4x2|p100] [--wait] [--download results/kaggle/<name>]
    launch.py --name <short> --commit <sha> --gpu t4x2 \
              --run a --cmd "<cmd A>" --run b --cmd "<cmd B>"   # one process per GPU
    launch.py --name <short> --commit <sha> --cmds-file runs.json   # [{"name":..,"cmd":..}]
    launch.py --name <short> --attach [--wait] [--download DIR]   # re-attach to a pushed kernel

What it does (see scripts/kaggle/README.md):
  1. `git archive <commit>` (never the working tree), adds a COMMIT file, packs it as
     abr-code.tar.gz and uploads it as a new version of the PRIVATE dataset
     <user>/abr-code (created on first use). The commit sha goes into the version notes.
  2. Generates a PRIVATE script kernel <user>/abr-<name> (GPU + internet on, dataset
     source abr-code) whose runner installs mamba-ssm/causal-conv1d, runs the command,
     and tees everything to /kaggle/working/output/run.log.
  3. Optionally waits for completion and downloads the outputs.
  4. Records commit, dataset version, kernel slug/version and timestamps in
     results/kaggle/<name>/launch.json.

Only the standard library is used. All Kaggle calls go through the `kaggle` CLI (and a
small helper in the CLI's own venv to read the dataset version number), so the
launcher can be tested with the CLI mocked.

Never prints or reads the Kaggle API token.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _P
# This folder contains queue.py, which would shadow the stdlib `queue` module for
# anything imported later; drop the script directory from sys.path first.
_sys.path[:] = [p for p in _sys.path if _P(p or ".").resolve() != _P(__file__).resolve().parent]

import argparse
import datetime as _dt
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER_TEMPLATE = HERE / "runner_template.py"

DEFAULT_USER = "parthgarg101"
DATASET_SLUG = "abr-code"
ARCHIVE_NAME = "abr-code.tar.gz"
KERNEL_PREFIX = "abr-"
DEFAULT_KAGGLE_BIN = os.path.expanduser("~/.venvs/kaggle/bin/kaggle")

# Kaggle's GPU machine shape "NvidiaTeslaT4" is the "GPU T4 x2" accelerator and is the
# server default; "NvidiaTeslaP100" is retired (the server substitutes T4 x2). CLI 1.7.x
# does not send machine_shape at all, so every GPU kernel gets the default T4 x2. We still
# write the field newer CLIs understand, and record what was actually assigned at run time.
MACHINE_SHAPES = {"t4": "NvidiaTeslaT4", "t4x2": "NvidiaTeslaT4", "p100": "NvidiaTeslaP100"}
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")

PHONE_HELP = (
    "\n*** ACTION NEEDED BY THE OWNER ***\n"
    "Kaggle refused GPU and/or internet for this kernel. This happens when the account\n"
    "is not phone-verified. Fix: log in to https://www.kaggle.com as {user}, open\n"
    "Settings (https://www.kaggle.com/settings) -> 'Phone verification', verify your\n"
    "phone number, then re-run this launcher with the same arguments.\n"
)

TERMINAL_OK = {"complete"}
TERMINAL_BAD = {"error", "cancelrequested", "cancelacknowledged", "cancelled", "canceled"}


class LaunchError(RuntimeError):
    pass


def utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------------------
# Naming / metadata
# --------------------------------------------------------------------------------------
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,40}[a-z0-9]$|^[a-z0-9]$")


def validate_name(name: str) -> str:
    """Kernel short names must already be slugs, so the title resolves to the id."""
    if not _SLUG_RE.match(name or ""):
        raise LaunchError(
            f"--name {name!r} must be lowercase letters, digits and '-', 1-42 chars, "
            "not starting/ending with '-'. Use e.g. 'compat-check'."
        )
    return name


def kernel_slug(name: str) -> str:
    return KERNEL_PREFIX + validate_name(name)


def kernel_ref(user: str, name: str) -> str:
    return f"{user}/{kernel_slug(name)}"


def kernel_url(user: str, name: str) -> str:
    return f"https://www.kaggle.com/code/{user}/{kernel_slug(name)}"


def dataset_ref(user: str) -> str:
    return f"{user}/{DATASET_SLUG}"


def make_kernel_metadata(user: str, name: str, gpu: str = "t4",
                         code_file: str = "runner.py",
                         kernel_sources: list[str] | None = None,
                         extra_datasets: list[str] | None = None) -> dict:
    slug = kernel_slug(name)
    if gpu not in MACHINE_SHAPES:
        raise LaunchError(f"--gpu must be one of {sorted(MACHINE_SHAPES)}")
    return {
        "id": f"{user}/{slug}",
        "title": slug,  # slugify(title) == slug, avoids Kaggle renaming surprises
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [dataset_ref(user)] + [d for d in (extra_datasets or [])
                                                   if d != dataset_ref(user)],
        "competition_sources": [],
        "kernel_sources": list(kernel_sources or []),
        "model_sources": [],
        "machine_shape": MACHINE_SHAPES[gpu],
    }


def make_dataset_metadata(user: str) -> dict:
    return {
        "title": DATASET_SLUG,
        "id": dataset_ref(user),
        "licenses": [{"name": "CC0-1.0"}],
    }


def version_notes(sha: str, ref: str, name: str) -> str:
    return f"commit {sha} (ref {ref}) for kernel {KERNEL_PREFIX}{name}"


# --------------------------------------------------------------------------------------
# Packaging (git archive of a commit, never the working tree)
# --------------------------------------------------------------------------------------
def _git(repo: Path, *args: str, binary: bool = False):
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True)
    if r.returncode != 0:
        raise LaunchError(f"git {' '.join(args)} failed: {r.stderr.decode(errors='replace').strip()}")
    return r.stdout if binary else r.stdout.decode().strip()


def resolve_commit(repo: Path, ref: str) -> str:
    return _git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")


PROVENANCE_SCRIPT = "experiments/full_scale/provenance.py"
SNAPSHOT_NAME = "git_snapshot.json"


def commit_has_file(repo: Path, sha: str, path: str) -> bool:
    r = subprocess.run(["git", "-C", str(repo), "cat-file", "-e", f"{sha}:{path}"],
                       capture_output=True)
    return r.returncode == 0


def make_git_snapshot(repo: Path, sha: str, python: str = sys.executable) -> bytes | None:
    """If the commit ships the provenance module, run its `snapshot` mode in a temporary
    checkout of exactly that commit and return git_snapshot.json (else None).

    The snapshot records the commit and the SHA-256 of every tracked file, so on Kaggle
    (no .git) `start_run` can refuse to run modified code. Without the module we fall
    back to the plain COMMIT file, which is always written.
    """
    if not commit_has_file(repo, sha, PROVENANCE_SCRIPT):
        return None
    with tempfile.TemporaryDirectory(prefix="abr-snap-") as td:
        co = Path(td) / "checkout"
        _git(repo, "clone", "--quiet", "--shared", "--no-checkout", str(repo), str(co))
        _git(co, "checkout", "--quiet", "--detach", sha)
        r = subprocess.run([python, PROVENANCE_SCRIPT, "snapshot"], cwd=co,
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise LaunchError(f"provenance snapshot failed:\n{r.stdout}\n{r.stderr}")
        out = co / SNAPSHOT_NAME
        if not out.exists():
            found = [p for p in co.rglob(SNAPSHOT_NAME) if ".git" not in p.parts]
            if not found:
                raise LaunchError(f"provenance snapshot did not produce {SNAPSHOT_NAME}:\n{r.stdout}")
            out = found[0]
        data = out.read_bytes()
        snap = json.loads(data)
        if sha not in json.dumps(snap):
            raise LaunchError(f"{SNAPSHOT_NAME} does not mention commit {sha}")
        return data


def build_code_archive(repo: Path, ref: str, out_dir: Path) -> dict:
    """Write out_dir/abr-code.tar.gz containing `git archive <sha>` plus a COMMIT file.

    Members live under `repo/`. The archive is deterministic for a given commit
    (fixed mtimes, sorted members, gzip mtime 0), so its sha256 identifies the code.
    """
    sha = resolve_commit(repo, ref)
    raw = _git(repo, "archive", "--format=tar", sha, binary=True)
    commit_time = _git(repo, "show", "-s", "--format=%ct", sha)
    mtime = int(commit_time)

    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / ARCHIVE_NAME
    src = tarfile.open(fileobj=io.BytesIO(raw), mode="r:")
    members = sorted(src.getmembers(), key=lambda m: m.name)
    n_files = 0
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:", format=tarfile.PAX_FORMAT) as dst:
        for m in members:
            if m.name in ("pax_global_header",) or m.type == tarfile.XGLTYPE:
                continue
            data = src.extractfile(m) if m.isfile() else None
            m2 = tarfile.TarInfo("repo/" + (m.name[2:] if m.name.startswith("./") else m.name))
            m2.type, m2.mode, m2.size, m2.linkname = m.type, m.mode, m.size, m.linkname
            m2.mtime, m2.uid, m2.gid, m2.uname, m2.gname = mtime, 0, 0, "", ""
            dst.addfile(m2, data)
            n_files += int(m.isfile())
        extra = [("COMMIT", (sha + "\n").encode()), ("COMMIT_REF", (ref + "\n").encode())]
        snap = make_git_snapshot(repo, sha)
        if snap is not None:
            extra.append((SNAPSHOT_NAME, snap))
        for fname, b in extra:
            ti = tarfile.TarInfo("repo/" + fname)
            ti.size, ti.mtime, ti.mode = len(b), mtime, 0o644
            dst.addfile(ti, io.BytesIO(b))
    with open(archive_path, "wb") as f:
        with gzip.GzipFile(fileobj=f, mode="wb", mtime=0, filename="") as gz:
            gz.write(buf.getvalue())
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    return {"commit": sha, "ref": ref, "archive": str(archive_path),
            "archive_sha256": digest, "n_files": n_files, "git_snapshot": snap is not None}


def normalize_runs(cmds, names=None) -> list[dict]:
    """[cmd] -> [{'name': None, 'cmd': cmd}] (single-run layout: output/run.log);
    several cmds -> named runs, each with its own GPU slot and output/<name>/ directory."""
    if isinstance(cmds, str):
        cmds = [cmds]
    runs = []
    for i, c in enumerate(cmds):
        if isinstance(c, dict):
            runs.append({k: v for k, v in c.items() if k in ("name", "cmd", "lane")})
            runs[-1].setdefault("name", None)
        else:
            runs.append({"name": (names[i] if names and i < len(names) else None), "cmd": c})
    if not runs or any(not r["cmd"] for r in runs):
        raise LaunchError("need at least one non-empty command")
    if len(runs) > 1 or runs[0]["name"]:
        for i, r in enumerate(runs):
            r["name"] = r["name"] or f"run{i}"
            if not RUN_NAME_RE.match(r["name"]):
                raise LaunchError(f"bad run name {r['name']!r} (letters, digits, . _ -)")
        if len({r["name"] for r in runs}) != len(runs):
            raise LaunchError("run names must be unique")
    return runs


def load_cmds_file(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list) or not all(isinstance(d, dict) and "cmd" in d for d in data):
        raise LaunchError(f"{path}: expected a JSON list of {{'name': ..., 'cmd': ...}}")
    return data


def render_runner(template: str, *, cmd, commit: str, user: str, name: str,
                  dataset_version, gpu: str, resume_sources: list[str] | None = None,
                  token_cache: dict | None = None, session_hours: float = 11.5) -> str:
    runs = cmd if isinstance(cmd, list) and cmd and isinstance(cmd[0], dict) else normalize_runs(cmd)
    values = {
        "__ABR_RUNS__": runs,
        "__ABR_COMMIT__": commit,
        "__ABR_KERNEL_URL__": kernel_url(user, name),
        "__ABR_KERNEL_REF__": kernel_ref(user, name),
        "__ABR_DATASET_REF__": dataset_ref(user),
        "__ABR_DATASET_VERSION__": dataset_version,
        "__ABR_GPU_REQUESTED__": gpu,
        "__ABR_RESUME_SOURCES__": list(resume_sources or []),
        "__ABR_TOKEN_CACHE__": token_cache,
        "__ABR_SESSION_HOURS__": float(session_hours),
    }
    out = template
    for key, val in values.items():
        token = f'"{key}"'
        if token not in out:
            raise LaunchError(f"runner template is missing placeholder {token}")
        out = out.replace(token, repr(val))  # valid Python literal (str/int/None/list)
    return out


TOKEN_CACHE_GLOB = "wikitext103_*_gpt2.pt"
DEFAULT_LOCAL_CACHE = Path(os.path.expanduser("~/.cache/abr"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def token_cache_spec(ref: str | None, local_dir: Path | None = DEFAULT_LOCAL_CACHE) -> dict | None:
    """{'ref': <user/slug>, 'expected_sha256': {file: sha}} for a token-cache dataset.

    The expected hashes come from the local cache the dataset was uploaded from; the
    runner refuses to start if the mounted files differ. No local files -> the runner
    only records the hashes it sees.
    """
    if not ref:
        return None
    if "/" not in ref:
        raise LaunchError(f"--cache-dataset must be <owner>/<slug>, got {ref!r}")
    exp = {}
    if local_dir and Path(local_dir).is_dir():
        for f in sorted(Path(local_dir).glob(TOKEN_CACHE_GLOB)):
            exp[f.name] = sha256_file(f)
    return {"ref": ref, "expected_sha256": exp}


def build_kernel_dir(out_dir: Path, *, user: str, name: str, cmd, commit: str,
                     dataset_version, gpu: str, kernel_sources: list[str] | None = None,
                     token_cache: dict | None = None, wheels_dataset: str | None = None,
                     session_hours: float = 11.5) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    extra = ([token_cache["ref"]] if token_cache else []) + ([wheels_dataset] if wheels_dataset else [])
    meta = make_kernel_metadata(user, name, gpu, kernel_sources=kernel_sources, extra_datasets=extra)
    runner = render_runner(RUNNER_TEMPLATE.read_text(), cmd=cmd, commit=commit, user=user,
                           name=name, dataset_version=dataset_version, gpu=gpu,
                           resume_sources=kernel_sources, token_cache=token_cache,
                           session_hours=session_hours)
    (out_dir / meta["code_file"]).write_text(runner)
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out_dir


# --------------------------------------------------------------------------------------
# Kaggle CLI wrapper (mocked in tests)
# --------------------------------------------------------------------------------------
_PHONE_PAT = re.compile(r"phone|verif", re.I)
_GPU_REFUSED_PAT = re.compile(
    r"(gpu|accelerator|internet).{0,80}(not (allowed|available|permitted|enabled)|"
    r"denied|refus|quota|disabled|requires?)|"
    r"(quota|limit).{0,40}(gpu|accelerator)", re.I)


def classify_error(text: str) -> str | None:
    if not text:
        return None
    if _PHONE_PAT.search(text):
        return "phone_verification"
    if re.search(r"quota", text, re.I) and re.search(r"gpu|accelerator", text, re.I):
        return "gpu_quota"
    if _GPU_REFUSED_PAT.search(text):
        return "gpu_or_internet_refused"
    if re.search(r"401|unauthori[sz]ed|403|forbidden", text, re.I):
        return "auth"
    return None


class KaggleCLI:
    def __init__(self, binary: str = DEFAULT_KAGGLE_BIN, user: str = DEFAULT_USER):
        self.binary = binary
        self.user = user
        bin_dir = Path(binary).parent
        self.python = str(bin_dir / "python")

    def run(self, *args: str, check: bool = True, timeout: int = 1800) -> subprocess.CompletedProcess:
        env = dict(os.environ, PYTHONWARNINGS="ignore")
        r = subprocess.run([self.binary, *args], capture_output=True, text=True,
                           env=env, timeout=timeout)
        r.stdout = _strip_noise(r.stdout)
        r.stderr = _strip_noise(r.stderr)
        if check and r.returncode != 0:
            raise LaunchError(f"kaggle {' '.join(args)} failed (exit {r.returncode}):\n"
                              f"{r.stdout}\n{r.stderr}".strip())
        return r

    # datasets -----------------------------------------------------------------------
    def dataset_info(self, ref: str) -> dict | None:
        """Return {'version': int, 'is_private': bool} or None if it doesn't exist."""
        owner, slug = ref.split("/")
        code = (
            "import json,sys\n"
            "from kaggle.api.kaggle_api_extended import KaggleApi\n"
            "from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest\n"
            "api=KaggleApi(); api.authenticate()\n"
            "r=ApiGetDatasetRequest(); r.owner_slug=sys.argv[1]; r.dataset_slug=sys.argv[2]\n"
            "try:\n"
            "    with api.build_kaggle_client() as k:\n"
            "        d=k.datasets.dataset_api_client.get_dataset(r)\n"
            "    print(json.dumps({'version': d.current_version_number, 'is_private': d.is_private}))\n"
            "except Exception as e:\n"
            "    s=str(e)\n"
            # Kaggle answers 403 (not 404) for a dataset that doesn't exist yet; if it
            # exists but isn't ours, the subsequent `datasets create` fails loudly.
            "    print(json.dumps({'error': s[:500], 'missing': any(c in s for c in ('404', '403', 'Not Found'))}))\n"
        )
        r = subprocess.run([self.python, "-W", "ignore", "-c", code, owner, slug],
                           capture_output=True, text=True, timeout=300)
        lines = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
        if not lines:
            raise LaunchError(f"could not query dataset {ref}: {r.stderr[-800:]}")
        info = json.loads(lines[-1])
        if info.get("missing"):
            return None
        if "error" in info:
            raise LaunchError(f"could not query dataset {ref}: {info['error']}")
        return info

    def dataset_status(self, ref: str) -> str:
        r = self.run("datasets", "status", ref, check=False)
        return (r.stdout or r.stderr).strip().splitlines()[-1].strip() if (r.stdout or r.stderr) else ""

    def dataset_create(self, folder: Path) -> str:
        # private by default (no --public)
        return self.run("datasets", "create", "-p", str(folder)).stdout

    def dataset_version(self, folder: Path, notes: str) -> str:
        return self.run("datasets", "version", "-p", str(folder), "-m", notes).stdout

    # kernels ------------------------------------------------------------------------
    def kernel_push(self, folder: Path, timeout: int | None = None) -> subprocess.CompletedProcess:
        extra = ["-t", str(int(timeout))] if timeout else []
        return self.run("kernels", "push", "-p", str(folder), *extra, check=False)

    def kernel_status(self, ref: str) -> tuple[str, str]:
        r = self.run("kernels", "status", ref, check=False)
        return parse_status(r.stdout + "\n" + r.stderr)

    def kernel_pull_metadata(self, ref: str, folder: Path) -> dict:
        self.run("kernels", "pull", ref, "-p", str(folder), "-m")
        return json.loads((folder / "kernel-metadata.json").read_text())

    def kernel_output(self, ref: str, folder: Path) -> str:
        folder.mkdir(parents=True, exist_ok=True)
        return self.run("kernels", "output", ref, "-p", str(folder), "-o", timeout=3600).stdout


def _strip_noise(s: str) -> str:
    return "\n".join(ln for ln in (s or "").splitlines()
                     if "NotOpenSSLWarning" not in ln and "warnings.warn(" not in ln)


def parse_status(text: str) -> tuple[str, str]:
    """'x has status "KernelWorkerStatus.COMPLETE"' -> ('complete', failure_message)."""
    m = re.search(r'has status "([^"]+)"', text)
    status = m.group(1).split(".")[-1].lower() if m else "unknown"
    fm = re.search(r'Failure message: "(.*)"', text, re.S)
    return status, (fm.group(1).strip() if fm else "")


def parse_push_output(text: str) -> dict:
    m = re.search(r"Kernel version (\d+) successfully pushed", text)
    url = re.search(r"(https://www\.kaggle\.com/\S+)", text)
    err = re.search(r"Kernel push error: (.*)", text)
    return {
        "version": int(m.group(1)) if m else None,
        "url": url.group(1).rstrip(".") if url else None,
        "error": err.group(1).strip() if err else None,
    }


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def upload_code(cli: KaggleCLI, pkg: dict, staging: Path, name: str,
                poll: float = 15.0, max_wait: float = 1800.0) -> dict:
    """Create or version the private code dataset and wait until the new version is ready."""
    ref = dataset_ref(cli.user)
    (staging / "dataset-metadata.json").write_text(json.dumps(make_dataset_metadata(cli.user), indent=2))
    before = cli.dataset_info(ref)
    notes = version_notes(pkg["commit"], pkg["ref"], name)
    if before is None:
        print(f"[launch] creating private dataset {ref}")
        out = cli.dataset_create(staging)
        prev_version = 0
    else:
        if not before.get("is_private", True):
            raise LaunchError(f"dataset {ref} is PUBLIC; refusing to upload code to it.")
        prev_version = int(before["version"])
        print(f"[launch] adding version to {ref} (current v{prev_version})")
        out = cli.dataset_version(staging, notes)
    if re.search(r"error", out, re.I) and not re.search(r"being created", out, re.I):
        raise LaunchError(f"dataset upload failed:\n{out}")
    t0 = time.time()
    while True:
        info = cli.dataset_info(ref)
        status = cli.dataset_status(ref)
        if info and int(info["version"]) > prev_version and status.lower().endswith("ready"):
            break
        if time.time() - t0 > max_wait:
            raise LaunchError(f"dataset {ref} did not become ready (status={status!r}, info={info})")
        time.sleep(poll)
    if before is None:
        # `datasets create` cannot take notes; the commit is also inside the archive.
        print("[launch] note: first dataset version has no version notes; commit is in COMMIT file")
    return {"ref": ref, "version": int(info["version"]), "notes": notes,
            "is_private": info.get("is_private")}


def push_kernel(cli: KaggleCLI, kdir: Path, user: str, name: str, timeout: int | None = None) -> dict:
    r = cli.kernel_push(kdir, timeout) if timeout else cli.kernel_push(kdir)
    text = (r.stdout + "\n" + r.stderr).strip()
    print(text)
    parsed = parse_push_output(text)
    if r.returncode != 0 or parsed["error"] or parsed["version"] is None and not parsed["url"]:
        kind = classify_error(text)
        if kind in ("phone_verification", "gpu_or_internet_refused"):
            print(PHONE_HELP.format(user=user), file=sys.stderr)
        elif kind == "gpu_quota":
            print("\n*** Kaggle GPU quota exhausted for this week. Wait for the weekly reset "
                  "(see https://www.kaggle.com/settings -> Accelerator quota).\n", file=sys.stderr)
        raise LaunchError(f"kernel push failed ({kind or 'unknown'}): {parsed['error'] or text}")
    return parsed


def verify_server_settings(cli: KaggleCLI, ref: str, user: str) -> dict:
    """Pull back the kernel metadata; Kaggle may silently drop GPU/internet if unverified."""
    with tempfile.TemporaryDirectory() as td:
        try:
            meta = cli.kernel_pull_metadata(ref, Path(td))
        except LaunchError as e:
            print(f"[launch] warning: could not pull kernel metadata to verify settings: {e}")
            return {}
    got = {k: meta.get(k) for k in ("enable_gpu", "enable_internet", "is_private")}
    if got.get("enable_gpu") is False or got.get("enable_internet") is False:
        print(PHONE_HELP.format(user=user), file=sys.stderr)
        raise LaunchError(f"Kaggle accepted the kernel but disabled GPU/internet: {got}")
    if got.get("is_private") is False:
        print("[launch] WARNING: kernel is public on the server!", file=sys.stderr)
    return got


def wait_for_kernel(cli: KaggleCLI, ref: str, *, initial: float = 30.0, max_interval: float = 300.0,
                    max_total: float = 13 * 3600, sleep=time.sleep) -> tuple[str, str]:
    """Poll `kaggle kernels status` with exponential backoff until terminal.

    Right after a push, the status endpoint can still report the previous version's
    terminal state, so a terminal state is only trusted once a non-terminal state was
    seen or after it has been reported 3 times in a row.
    """
    interval, waited, seen_running, terminal_streak = initial, 0.0, False, 0
    last = None
    while True:
        try:
            status, msg = cli.kernel_status(ref)
        except Exception as e:  # network hiccup: keep polling
            status, msg = "unknown", str(e)
        if status != last:
            print(f"[launch] {utcnow()} {ref}: {status}" + (f" ({msg})" if msg else ""))
            last = status
        if status in TERMINAL_OK | TERMINAL_BAD:
            terminal_streak += 1
            if seen_running or terminal_streak >= 3:
                return status, msg
        elif status in ("queued", "running"):
            seen_running = True
            terminal_streak = 0
        if waited > max_total:
            return "timeout", f"gave up after {waited:.0f}s"
        sleep(interval)
        waited += interval
        interval = min(max_interval, interval * 1.5)


def write_launch_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def download_outputs(cli: KaggleCLI, ref: str, dest: Path, user: str) -> dict:
    print(f"[launch] downloading outputs of {ref} -> {dest}")
    cli.kernel_output(ref, dest)
    env_files = list(dest.rglob("kaggle_env.json"))
    summary = {"dest": str(dest), "files": sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file())}
    if env_files:
        env = json.loads(env_files[0].read_text())
        summary["gpu"] = env.get("gpu_name")
        if env.get("no_gpu"):
            print(PHONE_HELP.format(user=user), file=sys.stderr)
    return summary


def launch_session(*, repo: Path, cli: "KaggleCLI", name: str, commit: str, runs: list[dict],
                   gpu: str = "t4", resume_from: list[str] | None = None, timeout: int | None = None,
                   rec_path: Path | None = None, staging: Path | None = None,
                   dry_run: bool = False, token_cache: dict | None = None,
                   wheels_dataset: str | None = None, session_hours: float = 11.5,
                   plan: dict | None = None) -> dict:
    """Package `commit`, upload it, push kernel abr-<name> and write launch.json. Returns the record."""
    user = cli.user
    ref = kernel_ref(user, name)
    rec_path = rec_path or (repo / "results" / "kaggle" / name / "launch.json")
    staging_root = staging or Path(tempfile.mkdtemp(prefix="abr-kaggle-"))
    ds_dir, k_dir = staging_root / "dataset", staging_root / "kernel"
    pkg = build_code_archive(repo, commit, ds_dir)
    print(f"[launch] packaged commit {pkg['commit']} ({pkg['n_files']} files, "
          f"sha256 {pkg['archive_sha256'][:16]}..., git_snapshot={pkg['git_snapshot']})")
    if gpu == "p100":
        print("[launch] warning: Kaggle retired P100; the session will run on the default GPU (T4 x2).")
    record = {
        "name": name, "commit": pkg["commit"], "commit_ref": commit,
        "archive_sha256": pkg["archive_sha256"], "git_snapshot": pkg["git_snapshot"],
        "runs": runs, "cmd": runs[0]["cmd"] if len(runs) == 1 else None,
        "gpu_requested": gpu, "kaggle_user": user,
        "kernel_ref": ref, "kernel_url": kernel_url(user, name),
        "launched_utc": utcnow(), "resume_from": list(resume_from or []),
        "token_cache": token_cache, "wheels_dataset": wheels_dataset,
        "session_hours": session_hours, "session_plan": plan,
    }
    if timeout:
        record["timeout_s"] = timeout
    if dry_run:
        build_kernel_dir(k_dir, user=user, name=name, cmd=runs, commit=pkg["commit"],
                         dataset_version=None, gpu=gpu, kernel_sources=resume_from,
                         token_cache=token_cache, wheels_dataset=wheels_dataset,
                         session_hours=session_hours)
        record["staging"] = str(staging_root)
        print(f"[launch] dry run: staging in {staging_root}")
        return record
    ds = upload_code(cli, pkg, ds_dir, name)
    record["dataset"] = ds
    print(f"[launch] dataset {ds['ref']} v{ds['version']} ready")
    build_kernel_dir(k_dir, user=user, name=name, cmd=runs, commit=pkg["commit"],
                     dataset_version=ds["version"], gpu=gpu, kernel_sources=resume_from,
                     token_cache=token_cache, wheels_dataset=wheels_dataset,
                     session_hours=session_hours)
    pushed = push_kernel(cli, k_dir, user, name, timeout=timeout)
    record["kernel_version"] = pushed["version"]
    record["kernel_version_url"] = pushed["url"]
    record["pushed_utc"] = utcnow()
    record["server_settings"] = verify_server_settings(cli, ref, user)
    write_launch_record(rec_path, record)
    print(f"[launch] kernel: {record['kernel_url']} (version {pushed['version']})")
    print(f"[launch] record: {rec_path}")
    return record


def main(argv=None) -> int:
    try:  # progress lines must show up promptly even when redirected to a file
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="short kernel name; kernel slug is abr-<name>")
    p.add_argument("--commit", help="git sha or ref to package (required unless --attach)")
    p.add_argument("--cmd", action="append", default=[],
                   help="shell command run inside /kaggle/working/repo; repeat for parallel runs "
                        "(one per GPU, CUDA_VISIBLE_DEVICES=0,1,...)")
    p.add_argument("--run", action="append", default=[], metavar="NAME",
                   help="name for the matching --cmd (output goes to output/<NAME>/)")
    p.add_argument("--cmds-file", help='JSON list of {"name": ..., "cmd": ...}')
    p.add_argument("--gpu", choices=sorted(MACHINE_SHAPES), default="t4",
                   help="requested GPU (Kaggle's GPU shape is T4 x2; the real one is logged)")
    p.add_argument("--wait", action="store_true", help="poll until the kernel finishes")
    p.add_argument("--download", metavar="DIR",
                   help="download kernel outputs into DIR (launch.json is then written there too)")
    p.add_argument("--attach", action="store_true",
                   help="don't push; wait/download an already pushed kernel (uses launch.json)")
    p.add_argument("--resume-from", action="append", default=[], metavar="KERNEL",
                   help="attach a previous kernel's output as input (e.g. parthgarg101/abr-run-s1); "
                        "each named run's previous output/<run>/ is copied back before it starts")
    p.add_argument("--timeout", type=int, default=None, help="kernel run-time limit in seconds")
    p.add_argument("--session-hours", type=float, default=11.5,
                   help="named runs stop starting / get SIGTERM after this (Kaggle limit is 12 h); "
                        "'{max_hours}' in a command is replaced by the lane's remaining hours")
    p.add_argument("--cache-dataset", metavar="OWNER/SLUG",
                   help="private dataset with wikitext103_*_gpt2.pt token caches; mounted and "
                        "exported as ABR_CACHE_DIR, SHA-256 checked against --cache-local")
    p.add_argument("--cache-local", default=str(DEFAULT_LOCAL_CACHE),
                   help="local cache dir the dataset was uploaded from (expected hashes)")
    p.add_argument("--wheels-dataset", metavar="OWNER/SLUG",
                   help="private dataset with prebuilt mamba_ssm/causal_conv1d wheels (+ Triton cache)")
    p.add_argument("--repo", default=None, help="repo root (default: git toplevel of this file)")
    p.add_argument("--user", default=DEFAULT_USER)
    p.add_argument("--kaggle-bin", default=DEFAULT_KAGGLE_BIN)
    p.add_argument("--dry-run", action="store_true", help="package + generate kernel dir only, no network")
    p.add_argument("--keep-staging", metavar="DIR", help="write the staging dirs here instead of a temp dir")
    a = p.parse_args(argv)

    try:
        validate_name(a.name)
        repo = Path(a.repo) if a.repo else Path(_git(HERE, "rev-parse", "--show-toplevel"))
        dest = None
        if a.download:
            dest = Path(a.download)
            if not dest.is_absolute():
                dest = repo / dest
        rec_path = (dest or repo / "results" / "kaggle" / a.name) / "launch.json"
        cli = KaggleCLI(a.kaggle_bin, a.user)
        ref = kernel_ref(a.user, a.name)

        if a.attach:
            record = json.loads(rec_path.read_text()) if rec_path.exists() else {"kernel_ref": ref}
        else:
            if a.cmds_file:
                runs = normalize_runs(load_cmds_file(Path(a.cmds_file)))
            else:
                runs = normalize_runs(a.cmd, a.run)
            if not a.commit:
                p.error("--commit is required unless --attach")
            record = launch_session(
                repo=repo, cli=cli, name=a.name, commit=a.commit, runs=runs, gpu=a.gpu,
                resume_from=a.resume_from, timeout=a.timeout, rec_path=rec_path,
                token_cache=token_cache_spec(a.cache_dataset, Path(a.cache_local)),
                wheels_dataset=a.wheels_dataset, session_hours=a.session_hours,
                staging=Path(a.keep_staging) if a.keep_staging else None, dry_run=a.dry_run)
            if a.dry_run:
                print(json.dumps(record, indent=2))
                return 0

        if a.wait:
            status, msg = wait_for_kernel(cli, ref)
            record["final_status"], record["failure_message"] = status, msg
            record["finished_utc"] = utcnow()
            write_launch_record(rec_path, record)
            if msg and classify_error(msg) in ("phone_verification", "gpu_or_internet_refused"):
                print(PHONE_HELP.format(user=a.user), file=sys.stderr)
        if dest is not None:
            record["download"] = download_outputs(cli, ref, dest, a.user)
            record["downloaded_utc"] = utcnow()
            write_launch_record(rec_path, record)
        print(f"[launch] kernel URL: {kernel_url(a.user, a.name)}")
        if record.get("final_status") and record["final_status"] != "complete":
            print(f"[launch] kernel finished with status {record['final_status']}: "
                  f"{record.get('failure_message', '')}", file=sys.stderr)
            return 2
        return 0
    except LaunchError as e:
        print(f"[launch] ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
