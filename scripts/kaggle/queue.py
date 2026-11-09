#!/usr/bin/env python3
"""Simple Kaggle job queue for the Phase 2 runs.

Slots = sessions x GPUs: each Kaggle session is one T4 x2 kernel running up to two
training processes in parallel (one per GPU). Only `--max-sessions` kernels run at once
(default 1; raise it only if Kaggle allows more concurrent GPU sessions on the account).

State lives in results/kaggle/queue.json:
  runs[run_id]      config, seed, block_size, cmd, status (pending/running/done/failed),
                    current kernel, list of sessions, resume_from, failures
  sessions[name]    kernel ref/version, runs, status, timestamps, download dir

    queue.py init --commit <sha> [--sweep LABEL] [--exclude-non-mamba]
    queue.py add-sweep LABEL         # the 6 block-size-sweep runs (8/16/64 x 2 seeds)
    queue.py status
    queue.py step  [--max-sessions 1] [--dry-run]   # one scheduling pass
    queue.py watch [--interval 900]                 # step until everything is done/failed
    queue.py retry <run_id>                         # failed -> pending

A step (1) polls running sessions; for finished ones it downloads the outputs and marks
each run done (a result.json exists), failed (non-zero exit, after --max-failures), or
pending again with resume_from = that kernel, so the next session copies the run's
output dir (checkpoints) back and train.py resumes; then (2) fills free slots with
pending runs, key 5-seed configs first. All sessions of the queue use the same commit.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _P
# This folder contains queue.py, which would shadow the stdlib `queue` module for
# anything imported later; drop the script directory from sys.path first.
_sys.path[:] = [p for p in _sys.path if _P(p or ".").resolve() != _P(__file__).resolve().parent]

import argparse
import importlib.util
import json
import os
import re
import shlex
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("abr_kaggle_launch", HERE / "launch.py")
L = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("abr_kaggle_launch", L)
_spec.loader.exec_module(L)

STATE_REL = Path("results") / "kaggle" / "queue.json"
SESSIONS_REL = Path("results") / "kaggle" / "queue"
TRAIN = "experiments/full_scale/train.py"


def load_config(repo: Path):
    path = repo / "experiments" / "full_scale" / "config.py"
    spec = importlib.util.spec_from_file_location("abr_full_scale_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_id_for(label: str, seed: int, block_size: int | None) -> str:
    """Same naming as train.run_name_for (without the abr-only block suffix rule)."""
    slug = re.sub(r"[^A-Za-z0-9.-]+", "_", label).strip("_")
    rid = f"{slug}_s{seed}"
    if block_size is not None:
        rid += f"_b{block_size}"
    return rid


def train_cmd(label: str, seed: int, block_size: int | None, max_hours: float) -> str:
    parts = ["python", TRAIN, "--config", label, "--seed", str(seed),
             "--out", "$ABR_OUTPUT_DIR", "--device", "cuda", "--wandb", "offline",
             "--max-hours", str(max_hours)]
    if block_size is not None:
        parts += ["--block-size", str(block_size)]
    return " ".join(p if p == "$ABR_OUTPUT_DIR" else shlex.quote(p) for p in parts)


def order_runs(runs: list[tuple], key_configs: set[str]) -> list[tuple]:
    """Key (5-seed) configs first; within a group, seed-major so every config gets its first
    seed before any gets its second; block-sweep runs last. Stable otherwise."""
    cfg_order = {}
    for lab, _, _ in runs:
        cfg_order.setdefault(lab, len(cfg_order))
    seed_rank = {}
    for lab, seed, _ in runs:
        seed_rank.setdefault((lab, seed), sum(1 for k in seed_rank if k[0] == lab))

    def k(r):
        lab, seed, bs = r
        return (bs is not None, lab not in key_configs, seed_rank[(lab, seed)], cfg_order[lab],
                bs or 0)
    return sorted(runs, key=k)


def new_run(label, seed, block_size, max_hours):
    return {"config": label, "seed": seed, "block_size": block_size,
            "cmd": train_cmd(label, seed, block_size, max_hours), "status": "pending",
            "kernel": None, "sessions": [], "resume_from": None, "failures": 0,
            "updated_utc": L.utcnow()}


def sweep_runs(C, label):
    if label not in {c["label"] for c in C.PHASE2_CONFIGS}:
        raise SystemExit(f"unknown sweep config {label!r}")
    return [(label, s, b) for b in C.BLOCK_SWEEP_TRAIN for s in C.BLOCK_SWEEP_SEEDS]


class Queue:
    def __init__(self, repo: Path, cli=None):
        self.repo = repo
        self.path = repo / STATE_REL
        self.cli = cli
        self.state = json.loads(self.path.read_text()) if self.path.exists() else None

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state, indent=2) + "\n")
        os.replace(tmp, self.path)

    # ---- building ----------------------------------------------------------------
    def init(self, commit: str, sweep: str | None = None, exclude_non_mamba: bool = False,
             max_hours: float = 11.0, force: bool = False):
        if self.state and not force:
            raise SystemExit(f"{self.path} exists; use --force to overwrite")
        sha = L.resolve_commit(self.repo, commit)
        C = load_config(self.repo)
        planned = list(C.planned_runs())
        if exclude_non_mamba:
            planned = [r for r in planned if C.contains_mamba(C.get_config(r[0]))]
        if sweep:
            planned += sweep_runs(C, sweep)
        ordered = order_runs(planned, set(C.KEY_CONFIGS))
        self.state = {"version": 1, "commit": sha, "user": self.cli.user if self.cli else L.DEFAULT_USER,
                      "max_hours": max_hours, "created_utc": L.utcnow(), "next_session": 1,
                      "order": [], "runs": {}, "sessions": {}}
        for lab, seed, bs in ordered:
            self._add(lab, seed, bs)
        self.save()
        return self.state

    def _add(self, lab, seed, bs):
        rid = run_id_for(lab, seed, bs)
        if rid in self.state["runs"]:
            return False
        self.state["runs"][rid] = new_run(lab, seed, bs, self.state["max_hours"])
        self.state["order"].append(rid)
        return True

    def add_sweep(self, label: str):
        C = load_config(self.repo)
        added = [run_id_for(*r) for r in sweep_runs(C, label) if self._add(*r)]
        self.save()
        return added

    def retry(self, rid: str):
        r = self.state["runs"][rid]
        r.update(status="pending", failures=0, updated_utc=L.utcnow())
        self.save()

    # ---- scheduling --------------------------------------------------------------
    def counts(self):
        out = {}
        for r in self.state["runs"].values():
            out[r["status"]] = out.get(r["status"], 0) + 1
        return out

    def poll(self, max_failures: int = 2):
        for name, s in self.state["sessions"].items():
            if s["status"] != "running":
                continue
            status, msg = self.cli.kernel_status(s["kernel_ref"])
            if status not in L.TERMINAL_OK | L.TERMINAL_BAD:
                continue
            s.update(kernel_status=status, failure_message=msg, finished_utc=L.utcnow())
            dest = self.repo / SESSIONS_REL / name
            try:
                s["download"] = L.download_outputs(self.cli, s["kernel_ref"], dest, self.cli.user)
                s["download_dir"] = str(dest.relative_to(self.repo))
            except L.LaunchError as e:
                s["download_error"] = str(e)
                self.save()
                continue  # retry the download next step
            for rid in s["runs"]:
                self._settle(rid, s, dest, max_failures)
            s["status"] = "downloaded"
            self.save()

    def _settle(self, rid, s, dest: Path, max_failures):
        r = self.state["runs"][rid]
        rdir = dest / rid
        st = {}
        if (rdir / "run_status.json").exists():
            st = json.loads((rdir / "run_status.json").read_text())
        has_result = any(rdir.rglob("result.json")) if rdir.exists() else False
        if rdir.exists():
            r["resume_from"] = s["kernel_ref"]  # newest output that holds this run's checkpoints
        r["last_exit_code"] = st.get("exit_code")
        r["last_state"] = st.get("state")
        if has_result and st.get("exit_code") == 0:
            r["status"] = "done"
        elif st.get("state") == "failed" or (not st and s.get("kernel_status") != "complete"):
            r["failures"] += 1
            r["status"] = "failed" if r["failures"] >= max_failures else "pending"
        else:  # incomplete (budget exit), preempted by the runner deadline, or killed at 12 h
            r["status"] = "pending"
        r["kernel"] = None
        r["updated_utc"] = L.utcnow()

    def launch_free_slots(self, max_sessions=1, gpus_per_session=2, gpu="t4x2",
                          timeout=None, dry_run=False):
        launched = []
        running = sum(1 for s in self.state["sessions"].values() if s["status"] == "running")
        pending = [rid for rid in self.state["order"] if self.state["runs"][rid]["status"] == "pending"]
        while running < max_sessions and pending:
            group, pending = pending[:gpus_per_session], pending[gpus_per_session:]
            name = f"q{self.state['next_session']:03d}"
            resume = sorted({self.state["runs"][g]["resume_from"] for g in group
                             if self.state["runs"][g]["resume_from"]})
            runs = [{"name": g, "cmd": self.state["runs"][g]["cmd"]} for g in group]
            rec = L.launch_session(repo=self.repo, cli=self.cli, name=name,
                                   commit=self.state["commit"], runs=runs, gpu=gpu,
                                   resume_from=resume, timeout=timeout, dry_run=dry_run,
                                   rec_path=self.repo / SESSIONS_REL / name / "launch.json")
            if dry_run:
                print(f"[queue] dry run: would launch {name} with {group} (resume {resume})")
                launched.append(name)
                running += 1
                continue
            self.state["next_session"] += 1
            self.state["sessions"][name] = {
                "kernel_ref": rec["kernel_ref"], "kernel_url": rec["kernel_url"],
                "kernel_version": rec.get("kernel_version"),
                "dataset_version": rec.get("dataset", {}).get("version"),
                "runs": group, "resume_from": resume, "status": "running",
                "launched_utc": rec["launched_utc"]}
            for g in group:
                r = self.state["runs"][g]
                r.update(status="running", kernel=rec["kernel_ref"], updated_utc=L.utcnow())
                r["sessions"].append(name)
            self.save()
            launched.append(name)
            running += 1
        return launched

    def step(self, **kw):
        self.poll(max_failures=kw.pop("max_failures", 2))
        return self.launch_free_slots(**kw)

    def finished(self):
        return all(r["status"] in ("done", "failed") for r in self.state["runs"].values())


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--user", default=L.DEFAULT_USER)
    ap.add_argument("--kaggle-bin", default=L.DEFAULT_KAGGLE_BIN)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("init")
    p.add_argument("--commit", required=True)
    p.add_argument("--sweep", metavar="LABEL", help="also add the block-size sweep for this config")
    p.add_argument("--exclude-non-mamba", action="store_true",
                   help="leave Transformer-Only out (PRD §0.3 lets it run locally)")
    p.add_argument("--max-hours", type=float, default=11.0,
                   help="train.py --max-hours per session (session limit is 12 h)")
    p.add_argument("--force", action="store_true")
    p = sp.add_parser("add-sweep")
    p.add_argument("label")
    sp.add_parser("status")
    p = sp.add_parser("retry")
    p.add_argument("run_id")
    for name in ("step", "watch"):
        p = sp.add_parser(name)
        p.add_argument("--max-sessions", type=int, default=1,
                       help="concurrent Kaggle GPU sessions (default 1)")
        p.add_argument("--gpus-per-session", type=int, default=2)
        p.add_argument("--max-failures", type=int, default=2)
        p.add_argument("--timeout", type=int, default=None, help="kernel run-time limit (s)")
        p.add_argument("--dry-run", action="store_true")
        if name == "watch":
            p.add_argument("--interval", type=int, default=900)
    a = ap.parse_args(argv)
    repo = Path(a.repo) if a.repo else Path(L._git(HERE, "rev-parse", "--show-toplevel"))
    q = Queue(repo, L.KaggleCLI(a.kaggle_bin, a.user))
    try:
        if a.cmd == "init":
            st = q.init(a.commit, a.sweep, a.exclude_non_mamba, a.max_hours, a.force)
            print(f"[queue] {len(st['runs'])} runs, commit {st['commit']} -> {q.path}")
            return 0
        if q.state is None:
            raise SystemExit(f"no queue at {q.path}; run `queue.py init --commit <sha>` first")
        if a.cmd == "add-sweep":
            print(f"[queue] added {q.add_sweep(a.label)}")
        elif a.cmd == "retry":
            q.retry(a.run_id)
        elif a.cmd == "status":
            print(json.dumps(q.counts()))
            for rid in q.state["order"]:
                r = q.state["runs"][rid]
                print(f"{r['status']:8s} {rid:40s} sessions={','.join(r['sessions']) or '-'}"
                      f" resume={r['resume_from'] or '-'}")
        else:
            kw = dict(max_sessions=a.max_sessions, gpus_per_session=a.gpus_per_session,
                      timeout=a.timeout, dry_run=a.dry_run, max_failures=a.max_failures)
            while True:
                launched = q.step(**dict(kw))
                print(f"[queue] {L.utcnow()} launched {launched or 'nothing'}; {q.counts()}")
                if a.cmd == "step" or q.finished() or a.dry_run:
                    break
                time.sleep(a.interval)
        return 0
    except L.LaunchError as e:
        print(f"[queue] ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
