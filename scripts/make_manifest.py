#!/usr/bin/env python
"""Write <results_dir>/MANIFEST.sha256 (PRD section 0.4 C).

One line per file under results_dir, except the manifest itself, sorted by path,
in the format of `sha256sum` / `shasum -a 256` (so `shasum -a 256 -c` also works)::

    <sha256>  relative/path

OS clutter (.DS_Store, Thumbs.db) and in-progress temp files (.tmp-*) are skipped.
Files that git would ignore are still hashed but reported, because they would be
missing from the commit and the verifier would then fail in CI.

Usage: python scripts/make_manifest.py [results_dir]   (default: results)
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile

MANIFEST_NAME = "MANIFEST.sha256"
SKIP_NAMES = {".DS_Store", "Thumbs.db", MANIFEST_NAME}
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints"}


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def list_result_files(results_dir: str) -> list[str]:
    """Relative POSIX paths of every file to be covered by the manifest, sorted."""
    out = []
    for dirpath, dirnames, filenames in os.walk(results_dir):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in filenames:
            if name in SKIP_NAMES or name.startswith(".tmp-"):
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, results_dir).replace(os.sep, "/")
            out.append(rel)
    return sorted(out)


def compute_manifest(results_dir: str) -> dict[str, str]:
    return {rel: sha256_file(os.path.join(results_dir, rel))
            for rel in list_result_files(results_dir)}


def parse_manifest(path: str) -> dict[str, str]:
    entries = {}
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            digest, sep, rel = line.partition("  ")
            if not sep or len(digest) != 64:
                raise ValueError(f"{path}:{n}: malformed manifest line: {line!r}")
            entries[rel.lstrip("*")] = digest.lower()
    return entries


def git_ignored(results_dir: str, rels: list[str]) -> list[str]:
    if not rels:
        return []
    try:
        p = subprocess.run(["git", "-C", results_dir, "check-ignore", "--stdin", "-z"],
                           input="\0".join(rels).encode(), capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if p.returncode not in (0, 1):  # 128: not a git repo
        return []
    return sorted(x for x in p.stdout.decode().split("\0") if x)


def write_manifest(results_dir: str) -> tuple[str, int]:
    entries = compute_manifest(results_dir)
    text = "".join(f"{d}  {rel}\n" for rel, d in sorted(entries.items()))
    path = os.path.join(results_dir, MANIFEST_NAME)
    fd, tmp = tempfile.mkstemp(dir=results_dir, prefix=".tmp-")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)
    return path, len(entries)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results_dir", nargs="?", default="results")
    args = ap.parse_args(argv)
    if not os.path.isdir(args.results_dir):
        print(f"error: {args.results_dir} is not a directory", file=sys.stderr)
        return 2
    path, n = write_manifest(args.results_dir)
    print(f"wrote {path} ({n} files)")
    ignored = git_ignored(args.results_dir, list_result_files(args.results_dir))
    if ignored:
        print("WARNING: these files are git-ignored, so they will not be committed and "
              "verify_provenance.py will fail on a fresh checkout. Fix .gitignore:",
              file=sys.stderr)
        for rel in ignored:
            print(f"  {rel}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
