"""Run provenance: record exactly which code, data and environment produced a result.

Implements PRD section 0.4 C. Typical use in a training script::

    from provenance import start_run, set_wandb_url, finish_run, dataset_sha256, RunLogger

    meta = start_run(cfg, debug=args.debug,
                     extra={"dataset_sha256": dataset_sha256(train_ids)})
    run_dir = os.path.join(RESULTS_DIR, meta["run_id"])
    with RunLogger(os.path.join(run_dir, "train.log")) as log:
        ...train...
        set_wandb_url(meta, wandb.run.url)
    finish_run(meta, results, RESULTS_DIR, log_path=log.path)

``start_run`` refuses to start when tracked files have uncommitted changes
(``DirtyWorkingTreeError``) unless ``debug=True``; debug runs are marked
``reportable = False``.

When the code runs from a copy without a ``.git`` directory (for example a Kaggle
dataset upload), the launcher should write a git snapshot first with
``python experiments/full_scale/provenance.py snapshot <repo_root>/git_snapshot.json``
(or ``write_git_snapshot``). ``start_run`` then reads the commit from the snapshot and
re-hashes every tracked file listed in it, so an edited copy is still refused.
The snapshot path can also be given with the ``PROVENANCE_GIT_SNAPSHOT`` env var.

Secrets are never recorded: env vars whose names contain KEY, TOKEN, SECRET or
PASSWORD are skipped, secret-looking command-line values and config keys are
redacted, and credentials embedded in URLs in ``pip freeze`` are stripped.
"""
from __future__ import annotations

import copy
import datetime as _dt
import hashlib
import io
import json
import os
import platform as _platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any

__all__ = [
    "DirtyWorkingTreeError",
    "ProvenanceError",
    "REQUIRED_META_KEYS",
    "RunLogger",
    "dataset_sha256",
    "finish_run",
    "is_secret_name",
    "set_wandb_url",
    "start_run",
    "write_git_snapshot",
]

SCHEMA_VERSION = 1

#: Keys every result.json ``meta`` must contain (checked by scripts/verify_provenance.py).
REQUIRED_META_KEYS = (
    "run_id", "reportable", "git_commit", "git_branch", "git_describe", "git_dirty",
    "command_line", "start_utc", "end_utc", "duration_s",
    "hostname", "platform", "python_version", "torch_version", "cuda_version", "gpu_name",
    "mamba_backend", "mamba_ssm_version", "pip_freeze", "kaggle", "wandb_url", "config",
)

SNAPSHOT_FILENAME = "git_snapshot.json"
SNAPSHOT_ENV = "PROVENANCE_GIT_SNAPSHOT"

# Env vars that indicate the process runs inside a Kaggle kernel.
_KAGGLE_MARKERS = ("KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE", "KAGGLE_DATA_PROXY_PROJECT")
_SECRET_WORDS = ("KEY", "TOKEN", "SECRET", "PASSWORD")
# Config keys are matched more narrowly: "train_tokens" or "eos_token" are not secrets.
_SECRET_CONFIG_RE = re.compile(
    r"(secret|password|passwd|credential|apikey|api_key|access_key|private_key|"
    r"auth_token|access_token|api_token|hf_token|wandb_key|kaggle_key)", re.IGNORECASE)
_URL_CRED_RE = re.compile(r"(://)[^/@\s]+@")
REDACTED = "<redacted>"


class ProvenanceError(RuntimeError):
    """Provenance could not be established (for example: not a git repo, no snapshot)."""


class DirtyWorkingTreeError(ProvenanceError):
    """Tracked files have uncommitted changes; reportable runs need a clean tree."""


# --------------------------------------------------------------------------- secrets

def is_secret_name(name: str) -> bool:
    """True if an env var name looks like it holds a secret (KEY/TOKEN/SECRET/PASSWORD)."""
    up = name.upper()
    return any(w in up for w in _SECRET_WORDS)


def _redact_config(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: (REDACTED if isinstance(k, str) and _SECRET_CONFIG_RE.search(k)
                    else _redact_config(v)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_redact_config(v) for v in obj]
    return obj


def _redact_argv(argv: list[str]) -> list[str]:
    out: list[str] = []
    redact_next = False
    for arg in argv:
        if redact_next:
            out.append(REDACTED)
            redact_next = False
            continue
        if arg.startswith("-"):
            name, eq, _ = arg.partition("=")
            if is_secret_name(name.lstrip("-").replace("-", "_")):
                if eq:
                    out.append(f"{name}={REDACTED}")
                else:
                    out.append(arg)
                    redact_next = True
                continue
        out.append(_URL_CRED_RE.sub(r"\1" + REDACTED + "@", arg))
    return out


# --------------------------------------------------------------------------- git

def _git(repo_root: str, *args: str) -> str | None:
    try:
        p = subprocess.run(["git", "-C", repo_root, *args], capture_output=True,
                           text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return p.stdout.strip() if p.returncode == 0 else None


def _find_repo_root(repo_root: str | None) -> str:
    if repo_root is not None:
        return os.path.abspath(repo_root)
    here = os.path.dirname(os.path.abspath(__file__))
    top = _git(here, "rev-parse", "--show-toplevel")
    return top if top else os.path.abspath(os.path.join(here, "..", ".."))


def _is_git_repo(repo_root: str) -> bool:
    return _git(repo_root, "rev-parse", "--is-inside-work-tree") == "true"


def _git_status(repo_root: str) -> tuple[list[str], list[str]]:
    """Return (changed tracked files, untracked files). Ignored files are not listed."""
    try:
        p = subprocess.run(["git", "-C", repo_root, "status", "--porcelain=v1", "-z",
                            "--untracked-files=all"], capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise ProvenanceError(f"git status failed: {e}") from e
    if p.returncode != 0:
        raise ProvenanceError(f"git status failed: {p.stderr.decode(errors='replace')}")
    changed, untracked = [], []
    entries = p.stdout.decode("utf-8", errors="replace").split("\0")
    i = 0
    while i < len(entries):
        e = entries[i]
        i += 1
        if not e:
            continue
        code, path = e[:2], e[3:]
        if code == "??":
            untracked.append(path)
        else:
            changed.append(path)
            if code[0] in "RC":  # rename/copy: next entry is the original path
                i += 1
    return sorted(changed), sorted(untracked)


def _git_info(repo_root: str) -> dict:
    commit = _git(repo_root, "rev-parse", "HEAD")
    if commit is None:
        raise ProvenanceError(f"{repo_root}: no commits yet (git rev-parse HEAD failed)")
    branch = _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    tags = _git(repo_root, "tag", "--points-at", "HEAD")
    return {
        "git_commit": commit,
        "git_branch": branch,
        "git_describe": _git(repo_root, "describe", "--tags", "--always", "--dirty"),
        "git_tags": tags.split() if tags else [],
    }


def _sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _tracked_files(repo_root: str) -> list[str]:
    out = _git(repo_root, "ls-files", "-z")
    return sorted(p for p in (out or "").split("\0") if p)


def write_git_snapshot(repo_root: str | None = None, out_path: str | None = None) -> str:
    """Record commit info and the SHA-256 of every tracked file, for runs from a copy
    of the code without ``.git`` (e.g. a Kaggle dataset). Requires a clean tree."""
    root = _find_repo_root(repo_root)
    if not _is_git_repo(root):
        raise ProvenanceError(f"{root} is not a git repository")
    changed, _ = _git_status(root)
    if changed:
        raise DirtyWorkingTreeError(
            "refusing to snapshot a dirty tree; commit these first: " + ", ".join(changed))
    snap = _git_info(root)
    snap["git_dirty"] = False
    snap["created_utc"] = _utc_iso()
    snap["files"] = {p: _sha256_file(os.path.join(root, p)) for p in _tracked_files(root)
                     if os.path.isfile(os.path.join(root, p))}
    out_path = out_path or os.path.join(root, SNAPSHOT_FILENAME)
    _atomic_write_json(out_path, snap)
    return out_path


def _git_from_snapshot(repo_root: str, snap_path: str) -> tuple[dict, list[str]]:
    with open(snap_path) as f:
        snap = json.load(f)
    changed = []
    for rel, digest in sorted(snap.get("files", {}).items()):
        p = os.path.join(repo_root, rel)
        if not os.path.isfile(p) or _sha256_file(p) != digest:
            changed.append(rel)
    info = {k: snap.get(k) for k in ("git_commit", "git_branch", "git_describe", "git_tags")}
    info["git_source"] = "snapshot"
    info["git_snapshot_created_utc"] = snap.get("created_utc")
    return info, changed


# --------------------------------------------------------------------------- environment

def _utc_iso(ts: float | None = None) -> str:
    d = _dt.datetime.fromtimestamp(time.time() if ts is None else ts, tz=_dt.timezone.utc)
    return d.isoformat(timespec="seconds").replace("+00:00", "Z")


def _utc_compact(ts: float) -> str:
    return _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _slug(s: Any) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "-", str(s)).strip("-") or "run"


def _torch_info() -> dict:
    info = {"torch_version": None, "cuda_version": None, "cudnn_version": None,
            "gpu_name": "cpu", "gpu_count": 0}
    try:
        import torch
    except Exception:  # torch missing or broken: still record the rest
        return info
    info["torch_version"] = torch.__version__
    info["cuda_version"] = torch.version.cuda
    try:
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["cudnn_version"] = torch.backends.cudnn.version()
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            info["gpu_name"] = "apple-mps"
    except Exception:
        pass
    return info


def _gpu_driver() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return p.stdout.strip().splitlines()[0] if p.returncode == 0 and p.stdout.strip() else None


def _dist_version(name: str) -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version
        return version(name)
    except PackageNotFoundError:
        return None
    except Exception:
        return None


def _mamba_info(config: dict, extra: dict) -> dict:
    backend = extra.get("mamba_backend", config.get("mamba_backend"))
    if backend is None:
        try:
            import mamba_ssm  # noqa: F401
            backend = "mamba_ssm"
        except Exception:
            backend = "unavailable"
    return {
        "mamba_backend": backend,
        "mamba_ssm_version": _dist_version("mamba_ssm") or _dist_version("mamba-ssm"),
        "causal_conv1d_version": _dist_version("causal_conv1d") or _dist_version("causal-conv1d"),
    }


def _pip_freeze() -> list[str]:
    try:
        p = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"],
                           capture_output=True, text=True, timeout=120)
        lines = p.stdout.splitlines() if p.returncode == 0 else []
    except (OSError, subprocess.TimeoutExpired):
        lines = []
    if not lines:  # pip unavailable: fall back to installed distributions
        from importlib.metadata import distributions
        lines = sorted({f"{d.metadata['Name']}=={d.version}" for d in distributions()
                        if d.metadata["Name"]}, key=str.lower)
    return [_URL_CRED_RE.sub(r"\1" + REDACTED + "@", ln) for ln in lines if ln.strip()]


def _kaggle_info() -> dict:
    env = {k: v for k, v in os.environ.items()
           if k.startswith("KAGGLE_") and not is_secret_name(k)}
    on_kaggle = any(m in os.environ for m in _KAGGLE_MARKERS)
    return {
        "on_kaggle": on_kaggle,
        "markers_present": [m for m in _KAGGLE_MARKERS if m in os.environ],
        "kernel_url": os.environ.get("KAGGLE_KERNEL_URL"),
        "kernel_version": os.environ.get("KAGGLE_KERNEL_VERSION"),
        "env": dict(sorted(env.items())),
    }


# --------------------------------------------------------------------------- public API

def start_run(config: dict, *, debug: bool = False, repo_root: str | None = None,
              extra: dict | None = None) -> dict:
    """Collect run metadata. Raises ``DirtyWorkingTreeError`` if tracked files have
    uncommitted changes, unless ``debug=True`` (then ``reportable`` is False).

    ``config`` should contain ``label`` (or ``name``/``model``) and ``seed``; they form
    ``run_id = <label>_seed<seed>_<YYYYmmddTHHMMSSZ>``. Keys in ``extra`` (for example
    ``dataset_sha256``, ``param_count``, ``mamba_backend``) are added to the top level of
    ``meta``; they may not overwrite the recorded fields.
    """
    if not isinstance(config, dict):
        raise TypeError("config must be a dict")
    extra = dict(extra or {})
    t0 = time.time()
    root = _find_repo_root(repo_root)

    untracked: list[str] = []
    snap_path = os.environ.get(SNAPSHOT_ENV) or os.path.join(root, SNAPSHOT_FILENAME)
    if _is_git_repo(root):
        changed, untracked = _git_status(root)
        git = _git_info(root)
        git["git_source"] = "git"
    elif os.path.isfile(snap_path):
        git, changed = _git_from_snapshot(root, snap_path)
    elif debug:
        changed = []
        git = {"git_commit": None, "git_branch": None, "git_describe": None,
               "git_tags": [], "git_source": "none"}
    else:
        raise ProvenanceError(
            f"{root} is not a git repository and no {SNAPSHOT_FILENAME} was found; "
            "run `provenance.py snapshot` before uploading the code, or pass debug=True")

    if changed and not debug:
        raise DirtyWorkingTreeError(
            "refusing to start a reportable run with uncommitted changes to tracked files "
            f"(commit them, or pass --debug for a non-reportable run): {', '.join(changed)}")

    label = config.get("label") or config.get("name") or config.get("model") or "run"
    seed = config.get("seed", "NA")
    meta: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": f"{_slug(label)}_seed{_slug(seed)}_{_utc_compact(t0)}",
        "reportable": not debug,
        "debug": bool(debug),
        **git,
        "git_dirty": bool(changed),
        "git_changed_files": changed,
        "git_untracked_files": untracked,
        "command_line": shlex.join(_redact_argv(list(sys.argv))),
        "python_executable": sys.executable,
        "cwd": os.getcwd(),
        "start_utc": _utc_iso(t0),
        "start_unix": t0,
        "hostname": socket.gethostname(),
        "platform": _platform.platform(),
        "python_version": sys.version.split()[0],
        **_torch_info(),
        "gpu_driver": _gpu_driver(),
        **_mamba_info(config, extra),
        "pip_freeze": _pip_freeze(),
        "kaggle": _kaggle_info(),
        "wandb_url": None,
        "config": _redact_config(copy.deepcopy(config)),
    }
    for k, v in extra.items():
        if k in meta and k != "mamba_backend":
            raise ValueError(f"extra key {k!r} would overwrite a recorded meta field")
        meta[k] = v
    return meta


def dataset_sha256(data: Any) -> str:
    """SHA-256 of a tokenized dataset.

    - torch tensor: the bytes of ``data.detach().cpu().to(int64).contiguous()`` (integer
      and bool tensors are cast to int64; float tensors are hashed in their own dtype),
    - numpy array: same rule, so a tensor and its ``.numpy()`` give the same hash,
    - ``bytes``/``bytearray``/``memoryview``: hashed directly,
    - ``str``/``os.PathLike``: the file contents, streamed.
    """
    h = hashlib.sha256()
    if isinstance(data, (bytes, bytearray, memoryview)):
        h.update(bytes(data))
        return h.hexdigest()
    if isinstance(data, (str, os.PathLike)):
        return _sha256_file(os.fspath(data))
    try:
        import torch
        if isinstance(data, torch.Tensor):
            t = data.detach().cpu()
            if not (t.is_floating_point() or t.is_complex()):
                t = t.to(torch.int64)
            data = t.contiguous().numpy()
    except ImportError:
        pass
    try:
        import numpy as np
    except ImportError as e:
        raise TypeError(f"cannot hash object of type {type(data).__name__}") from e
    if isinstance(data, np.ndarray):
        arr = data
        if arr.dtype.kind in "biu":
            arr = arr.astype(np.int64, copy=False)
        h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()
    raise TypeError(f"dataset_sha256: unsupported type {type(data).__name__}")


def set_wandb_url(meta: dict, url: str) -> None:
    """Record the W&B run URL (call once the W&B run exists)."""
    meta["wandb_url"] = None if url is None else str(url)


def _json_default(o: Any) -> Any:
    if hasattr(o, "tolist"):  # numpy arrays/scalars, torch tensors
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, os.PathLike):
        return os.fspath(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _atomic_write_bytes(path: str, data: bytes) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-", suffix="-" + os.path.basename(path))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _atomic_write_json(path: str, obj: Any) -> None:
    text = json.dumps(obj, indent=2, sort_keys=False, default=_json_default, allow_nan=True)
    _atomic_write_bytes(path, (text + "\n").encode("utf-8"))


def finish_run(meta: dict, results: dict, out_dir: str, log_path: str | None = None) -> str:
    """Write ``<out_dir>/<run_id>/result.json`` atomically and copy the raw log to
    ``<out_dir>/<run_id>/train.log`` unmodified. Returns the result.json path.
    Refuses to overwrite an existing result.json."""
    run_dir = os.path.join(out_dir, meta["run_id"])
    os.makedirs(run_dir, exist_ok=True)
    result_path = os.path.join(run_dir, "result.json")
    if os.path.exists(result_path):
        raise FileExistsError(f"{result_path} already exists; results are never overwritten")

    if log_path is not None:
        dest = os.path.join(run_dir, "train.log")
        if not (os.path.exists(dest) and os.path.samefile(log_path, dest)):
            fd, tmp = tempfile.mkstemp(dir=run_dir, prefix=".tmp-", suffix="-train.log")
            os.close(fd)
            shutil.copyfile(log_path, tmp)
            os.replace(tmp, dest)

    end = time.time()
    m = dict(meta)
    m["end_utc"] = _utc_iso(end)
    start = m.get("start_unix")
    m["duration_s"] = round(end - start, 3) if isinstance(start, (int, float)) else None
    _atomic_write_json(result_path, {"meta": m, "results": results})
    return result_path


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self._streams:
            st.flush()

    def isatty(self):
        return False

    @property
    def encoding(self):
        return "utf-8"


class RunLogger:
    """Tee ``sys.stdout`` and ``sys.stderr`` into a log file (appended, line-buffered).

    Only Python-level writes are captured; output written directly to file descriptor
    1/2 by C extensions is not (Kaggle's own kernel log still has it)::

        with RunLogger("results/<run_id>/train.log") as log:
            ...
    """

    def __init__(self, path: str, mode: str = "a"):
        self.path = path
        self.mode = mode
        self._file = None
        self._old = None

    def __enter__(self) -> "RunLogger":
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._file = open(self.path, self.mode, encoding="utf-8", buffering=1)
        self._old = (sys.stdout, sys.stderr)
        sys.stdout = _Tee(self._old[0], self._file)
        sys.stderr = _Tee(self._old[1], self._file)
        return self

    def __exit__(self, *exc) -> None:
        sys.stdout, sys.stderr = self._old
        self._file.flush()
        self._file.close()
        self._file = None


def _main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Provenance helpers")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot", help="write git_snapshot.json for a code copy without .git")
    s.add_argument("out", nargs="?", default=None)
    s.add_argument("--repo-root", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "snapshot":
        print(write_git_snapshot(args.repo_root, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
