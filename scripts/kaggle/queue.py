#!/usr/bin/env python3
"""Kaggle job queue for the Phase 2 runs, with packed sessions.

Each Kaggle session is one T4 x2 kernel with two GPU *lanes*. A lane runs a list of runs
back to back until the session budget (`--session-hours`, default 11.5 h, under Kaggle's
12 h limit) is used up. Every run gets `--max-hours` = its lane's remaining budget when it
starts, so a run cut off at the end checkpoints cleanly; the next session resumes partial
runs first, then starts new ones. Runs are picked in priority order (partial runs, then
key 5-seed configs seed-major, then 2-seed configs, then the block sweep) until the two
lanes are full, and split over the lanes longest-job-first using the per-config cost
estimates in scripts/kaggle/run_costs.json (measured steps/s on Kaggle T4).

State: results/kaggle/queue.json (runs: status pending/running/done/failed, sessions,
hours done, resume source). Session outputs and launch.json (with the lane plan) go to
results/kaggle/queue/qNNN/.

    queue.py init --commit <sha> [--sweep LABEL] [--exclude-non-mamba]
                  [--cache-dataset parthgarg101/abr-wikitext103-gpt2-tokens] [--wheels-dataset ..]
    queue.py add-sweep LABEL         # the 6 block-size-sweep runs (8/16/64 x 2 seeds)
    queue.py plan                    # print the next session plan and the session count
    queue.py advance                 # check/download the active session, launch the next one
    queue.py status
    queue.py retry <run_id>          # failed -> pending

`advance` is idempotent: it only downloads sessions whose kernel has finished and that are
not downloaded yet, and only launches when fewer than --max-sessions (default 1) run.
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
COSTS_PATH = HERE / "run_costs.json"
MAX_HOURS_TOKEN = "{max_hours}"   # replaced by the runner with the lane's remaining hours


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


def train_cmd(label: str, seed: int, block_size: int | None, max_hours=None) -> str:
    parts = ["python", TRAIN, "--config", label, "--seed", str(seed),
             "--out", "$ABR_OUTPUT_DIR", "--device", "cuda", "--wandb", "offline",
             "--max-hours", MAX_HOURS_TOKEN]
    if block_size is not None:
        parts += ["--block-size", str(block_size)]
    return " ".join(p if p in ("$ABR_OUTPUT_DIR", MAX_HOURS_TOKEN) else shlex.quote(p) for p in parts)


def load_costs(path: Path = COSTS_PATH) -> dict:
    return json.loads(Path(path).read_text())


def est_total_hours(costs: dict, label: str, block_size=None) -> float:
    sps = costs["steps_per_s"].get(label, {}).get("value") if isinstance(
        costs["steps_per_s"].get(label), dict) else costs["steps_per_s"].get(label)
    if sps is None:
        sps = costs["default_steps_per_s"]
    return costs["total_steps"] / sps / 3600.0 + costs.get("per_run_overhead_hours", 0.0)


def pack_lanes(items: list[tuple[str, float]], n_lanes: int) -> list[list[tuple[str, float]]]:
    """Longest-processing-time-first: biggest job to the least loaded lane."""
    lanes = [[] for _ in range(n_lanes)]
    loads = [0.0] * n_lanes
    for rid, h in sorted(items, key=lambda x: -x[1]):
        i = loads.index(min(loads))
        lanes[i].append((rid, h))
        loads[i] += h
    return lanes


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


def new_run(label, seed, block_size, max_hours=None):
    return {"config": label, "seed": seed, "block_size": block_size,
            "cmd": train_cmd(label, seed, block_size), "status": "pending",
            "kernel": None, "sessions": [], "resume_from": None, "failures": 0,
            "hours_done": 0.0, "updated_utc": L.utcnow()}


def sweep_runs(C, label):
    if label not in {c["label"] for c in C.PHASE2_CONFIGS}:
        raise SystemExit(f"unknown sweep config {label!r}")
    return [(label, s, b) for b in C.BLOCK_SWEEP_TRAIN for s in C.BLOCK_SWEEP_SEEDS]


def _hours(h):
    from datetime import datetime
    try:
        f = "%Y-%m-%dT%H:%M:%SZ"
        return (datetime.strptime(h["finished_utc"], f) - datetime.strptime(h["started_utc"], f)).total_seconds() / 3600
    except Exception:
        return 0.0


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
             max_hours: float = 11.0, force: bool = False, token_cache: dict | None = None,
             wheels_dataset: str | None = None):
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
                      "token_cache": token_cache, "wheels_dataset": wheels_dataset, "order": [], "runs": {}, "sessions": {}}
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
        hist = st.get("history", [])
        r["hours_done"] = round(sum(_hours(h) for h in hist), 3)
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

    def remaining_hours(self, rid, costs):
        r = self.state["runs"][rid]
        total = est_total_hours(costs, r["config"], r["block_size"])
        return max(total - r.get("hours_done", 0.0), costs.get("min_resume_hours", 0.25))

    def plan_session(self, costs, session_hours=11.5, lanes=2, pending=None):
        """Pick runs (partial first, then queue order) until `lanes` x budget is covered,
        then pack them over the lanes longest-first; partial runs go first in their lane."""
        if pending is None:
            pending = [rid for rid in self.state["order"] if self.state["runs"][rid]["status"] == "pending"]
        partial = [r for r in pending if self.state["runs"][r].get("resume_from")]
        fresh = [r for r in pending if not self.state["runs"][r].get("resume_from")]
        overhead = costs.get("session_overhead_hours", 0.0)
        capacity = lanes * max(session_hours - overhead, 0.1)
        picked, total = [], 0.0
        for rid in partial + fresh:
            if total >= capacity:
                break
            h = self.remaining_hours(rid, costs)
            picked.append((rid, h))
            total += h
        packed = pack_lanes(picked, lanes)
        rank = {rid: i for i, rid in enumerate(partial + fresh)}
        out = []
        for lane in packed:
            # partial runs first (in priority order), then fresh runs longest-first
            lane.sort(key=lambda x: (x[0] not in partial, rank[x[0]] if x[0] in partial else -x[1]))
            out.append([{"run": rid, "est_hours": round(h, 2)} for rid, h in lane])
        return {"session_hours": session_hours, "overhead_hours": overhead, "lanes": out,
                "est_load_hours": [round(sum(x["est_hours"] for x in l), 2) for l in out]}

    def simulate(self, costs, session_hours=11.5, lanes=2):
        """Estimated number of sessions to finish everything not done (no failures)."""
        rem = {rid: self.remaining_hours(rid, costs) for rid in self.state["order"]
               if self.state["runs"][rid]["status"] in ("pending", "running")}
        partial = {rid for rid in rem if self.state["runs"][rid].get("resume_from")}
        n, budget = 0, max(session_hours - costs.get("session_overhead_hours", 0.0), 0.1)
        while rem and n < 1000:
            n += 1
            order = [r for r in self.state["order"] if r in rem]
            order = [r for r in order if r in partial] + [r for r in order if r not in partial]
            picked, total = [], 0.0
            for rid in order:
                if total >= lanes * budget:
                    break
                picked.append((rid, rem[rid]))
                total += rem[rid]
            for lane in pack_lanes(picked, lanes):
                lane.sort(key=lambda x: (x[0] not in partial, -x[1]))
                t = 0.0
                for rid, h in lane:
                    left = budget - t
                    if left < costs.get("min_start_hours", 1 / 3):
                        break
                    used = min(h, left)
                    t += used
                    rem[rid] -= used
                    if rem[rid] <= 1e-6:
                        del rem[rid]
                    else:
                        partial.add(rid)
                        rem[rid] += costs.get("per_run_overhead_hours", 0.0)
        return n

    def launch_next(self, *, costs, max_sessions=1, session_hours=11.5, lanes=2, gpu="t4x2",
                    timeout=None, dry_run=False):
        running = sum(1 for s in self.state["sessions"].values() if s["status"] == "running")
        launched = []
        while running < max_sessions:
            plan = self.plan_session(costs, session_hours, lanes)
            if not any(plan["lanes"]):
                break
            name = f"q{self.state['next_session']:03d}"
            group = [x["run"] for lane in plan["lanes"] for x in lane]
            resume = sorted({self.state["runs"][g]["resume_from"] for g in group
                             if self.state["runs"][g]["resume_from"]})
            runs = [{"name": x["run"], "cmd": self.state["runs"][x["run"]]["cmd"], "lane": i}
                    for i, lane in enumerate(plan["lanes"]) for x in lane]
            plan["name"] = name
            if dry_run:
                print(json.dumps(plan, indent=2))
                return [name]
            rec = L.launch_session(repo=self.repo, cli=self.cli, name=name,
                                   commit=self.state["commit"], runs=runs, gpu=gpu,
                                   resume_from=resume, timeout=timeout,
                                   rec_path=self.repo / SESSIONS_REL / name / "launch.json",
                                   token_cache=self.state.get("token_cache"),
                                   wheels_dataset=self.state.get("wheels_dataset"),
                                   session_hours=session_hours, plan=plan)
            self.state["next_session"] += 1
            self.state["sessions"][name] = {
                "kernel_ref": rec["kernel_ref"], "kernel_url": rec["kernel_url"],
                "kernel_version": rec.get("kernel_version"),
                "dataset_version": rec.get("dataset", {}).get("version"),
                "runs": group, "plan": plan, "resume_from": resume, "status": "running",
                "launched_utc": rec["launched_utc"]}
            for g in group:
                r = self.state["runs"][g]
                r.update(status="running", kernel=rec["kernel_ref"], updated_utc=L.utcnow())
                r["sessions"].append(name)
            self.save()
            launched.append(name)
            running += 1
        return launched

    def advance(self, *, max_failures=2, **kw):
        self.poll(max_failures=max_failures)
        return self.launch_next(**kw)

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
    p.add_argument("--cache-dataset", metavar="OWNER/SLUG",
                   help="token-cache dataset to mount (ABR_CACHE_DIR); expected SHA-256s are "
                        "taken from --cache-local now and stored in the queue")
    p.add_argument("--cache-local", default=str(L.DEFAULT_LOCAL_CACHE))
    p.add_argument("--wheels-dataset", metavar="OWNER/SLUG",
                   help="prebuilt mamba/causal-conv1d wheels dataset (fast installs)")
    p.add_argument("--force", action="store_true")
    p = sp.add_parser("add-sweep")
    p.add_argument("label")
    sp.add_parser("status")
    p = sp.add_parser("retry")
    p.add_argument("run_id")
    for name in ("advance", "plan", "watch"):
        p = sp.add_parser(name)
        p.add_argument("--max-sessions", type=int, default=1,
                       help="concurrent Kaggle GPU sessions (default 1)")
        p.add_argument("--session-hours", type=float, default=11.5)
        p.add_argument("--lanes", type=int, default=2, help="GPUs per session")
        p.add_argument("--max-failures", type=int, default=2)
        p.add_argument("--timeout", type=int, default=None, help="kernel run-time limit (s)")
        p.add_argument("--costs", default=str(COSTS_PATH))
        if name == "watch":
            p.add_argument("--interval", type=int, default=1800)
    a = ap.parse_args(argv)
    repo = Path(a.repo) if a.repo else Path(L._git(HERE, "rev-parse", "--show-toplevel"))
    q = Queue(repo, L.KaggleCLI(a.kaggle_bin, a.user))
    try:
        if a.cmd == "init":
            st = q.init(a.commit, a.sweep, a.exclude_non_mamba, None, a.force,
                        L.token_cache_spec(a.cache_dataset, Path(a.cache_local)), a.wheels_dataset)
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
            costs = load_costs(Path(a.costs))
            kw = dict(costs=costs, max_sessions=a.max_sessions, session_hours=a.session_hours,
                      lanes=a.lanes, timeout=a.timeout)
            if a.cmd == "plan":
                q.launch_next(dry_run=True, **kw)
                print(f"[queue] estimated sessions to finish: "
                      f"{q.simulate(costs, a.session_hours, a.lanes)}")
                return 0
            while True:
                launched = q.advance(max_failures=a.max_failures, **kw)
                print(f"[queue] {L.utcnow()} launched {launched or 'nothing'}; {q.counts()}")
                if a.cmd == "advance" or q.finished():
                    break
                time.sleep(a.interval)
        return 0
    except L.LaunchError as e:
        print(f"[queue] ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
