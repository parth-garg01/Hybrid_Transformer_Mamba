#!/usr/bin/env python
"""Verify the provenance of every result under a results directory (PRD section 0.4).

For every `result.json` found (recursively):
  - all required meta keys are present (see provenance.REQUIRED_META_KEYS),
  - `reportable` is true; non-reportable (debug) runs are listed as NON-REPORTABLE
    (a failure with --strict, otherwise a warning, and their git checks are warnings),
  - `git_commit` exists in this repository (`git cat-file -e <commit>^{commit}`),
  - `git_dirty` is false,
  - the run directory name equals `run_id`.
And for the directory as a whole, every file matches `MANIFEST.sha256`: no missing,
extra or changed files.

Exits 0 when everything passes (including "0 results" or no results dir), 1 otherwise.

Usage: python scripts/verify_provenance.py [results_dir] [--repo PATH] [--strict]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_DEFAULT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO_DEFAULT, "experiments", "full_scale"))

from make_manifest import MANIFEST_NAME, compute_manifest, parse_manifest  # noqa: E402
from provenance import REQUIRED_META_KEYS  # noqa: E402


def commit_exists(repo: str, commit: str) -> bool:
    try:
        p = subprocess.run(["git", "-C", repo, "cat-file", "-e", f"{commit}^{{commit}}"],
                           capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return p.returncode == 0


def find_results(results_dir: str) -> list[str]:
    found = []
    for dirpath, dirnames, filenames in os.walk(results_dir):
        dirnames.sort()
        if "result.json" in filenames:
            found.append(os.path.join(dirpath, "result.json"))
    return sorted(found)


def check_result(path: str, repo: str, strict: bool) -> tuple[list[str], list[str], bool]:
    """Return (failures, warnings, reportable) for one result.json."""
    fails, warns = [], []
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return [f"cannot read JSON: {e}"], [], False
    meta = doc.get("meta") if isinstance(doc, dict) else None
    if not isinstance(meta, dict):
        return ["no 'meta' object"], [], False
    if "results" not in doc:
        fails.append("no 'results' object")

    missing = [k for k in REQUIRED_META_KEYS if k not in meta]
    if missing:
        fails.append("missing meta keys: " + ", ".join(missing))

    reportable = meta.get("reportable") is True
    # Git problems on a debug run are expected; they only fail reportable runs.
    git_bucket = fails if (reportable or strict) else warns
    if not reportable:
        (fails if strict else warns).append("NON-REPORTABLE run (debug mode or reportable != true)")

    commit = meta.get("git_commit")
    if not commit:
        git_bucket.append("git_commit is empty")
    elif not commit_exists(repo, str(commit)):
        git_bucket.append(f"git_commit {commit} does not exist in {repo} "
                          "(unpushed/unfetched commit, or shallow clone?)")
    if meta.get("git_dirty") is not False:
        git_bucket.append(f"git_dirty is {meta.get('git_dirty')!r} (must be false)")

    run_dir = os.path.basename(os.path.dirname(path))
    if meta.get("run_id") and meta["run_id"] != run_dir:
        fails.append(f"run_id {meta['run_id']!r} does not match directory {run_dir!r}")
    if not os.path.isfile(os.path.join(os.path.dirname(path), "train.log")):
        warns.append("no train.log next to result.json")
    return fails, warns, reportable


def check_manifest(results_dir: str) -> list[str]:
    actual = compute_manifest(results_dir)
    mpath = os.path.join(results_dir, MANIFEST_NAME)
    if not os.path.isfile(mpath):
        return [f"{MANIFEST_NAME} missing ({len(actual)} files unlisted); "
                "run scripts/make_manifest.py"] if actual else []
    try:
        listed = parse_manifest(mpath)
    except ValueError as e:
        return [str(e)]
    fails = []
    for rel in sorted(set(listed) - set(actual)):
        fails.append(f"listed in manifest but missing: {rel}")
    for rel in sorted(set(actual) - set(listed)):
        fails.append(f"not in manifest: {rel}")
    for rel in sorted(set(actual) & set(listed)):
        if actual[rel] != listed[rel]:
            fails.append(f"HASH MISMATCH (file changed after manifest): {rel}")
    return fails


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify result provenance (PRD 0.4).")
    ap.add_argument("results_dir", nargs="?", default="results")
    ap.add_argument("--repo", default=REPO_DEFAULT, help="git repo the commits must exist in")
    ap.add_argument("--strict", action="store_true",
                    help="treat non-reportable runs as failures")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.results_dir):
        print(f"verify_provenance: {args.results_dir!r} does not exist: 0 results. PASS")
        return 0

    results = find_results(args.results_dir)
    n_fail = n_nonrep = 0
    for path in results:
        rel = os.path.relpath(path, args.results_dir)
        fails, warns, reportable = check_result(path, args.repo, args.strict)
        n_nonrep += not reportable
        status = "FAIL" if fails else ("OK (non-reportable)" if not reportable else "OK")
        print(f"[{status}] {rel}")
        for msg in fails:
            print(f"    FAIL: {msg}")
        for msg in warns:
            print(f"    warn: {msg}")
        n_fail += bool(fails)

    manifest_fails = check_manifest(args.results_dir)
    print(f"[{'FAIL' if manifest_fails else 'OK'}] {MANIFEST_NAME}")
    for msg in manifest_fails:
        print(f"    FAIL: {msg}")

    ok = n_fail == 0 and not manifest_fails
    print(f"verify_provenance: {len(results)} results, {n_fail} failed, "
          f"{n_nonrep} non-reportable, manifest {'ok' if not manifest_fails else 'FAILED'}. "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
