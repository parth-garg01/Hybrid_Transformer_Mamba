"""Unit tests for scripts/kaggle/launch.py: metadata, packaging, CLI handling (no network)."""
import ast
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("kaggle_launch", ROOT / "scripts" / "kaggle" / "launch.py")
launch = importlib.util.module_from_spec(_spec)
sys.modules["kaggle_launch"] = launch
_spec.loader.exec_module(launch)


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q")
    git(r, "config", "user.email", "t@example.com")
    git(r, "config", "user.name", "t")
    git(r, "config", "commit.gpgsign", "false")
    (r / "a.py").write_text("print('v1')\n")
    (r / ".hidden").write_text("x\n")
    (r / "sub").mkdir()
    (r / "sub" / "b.txt").write_text("b\n")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "c1")
    return r


def members(archive):
    with tarfile.open(archive) as t:
        return {m.name: (t.extractfile(m).read() if m.isfile() else None) for m in t.getmembers()}


# ---------------------------------------------------------------- metadata
def test_kernel_metadata_flags():
    m = launch.make_kernel_metadata("u", "compat-check", "t4")
    assert m["id"] == "u/abr-compat-check"
    assert m["title"] == "abr-compat-check"
    assert m["is_private"] is True and m["enable_gpu"] is True and m["enable_internet"] is True
    assert m["dataset_sources"] == ["u/abr-code"]
    assert m["kernel_type"] == "script" and m["language"] == "python"
    assert m["machine_shape"] == "NvidiaTeslaT4"
    assert launch.make_kernel_metadata("u", "x", "p100")["machine_shape"] == "NvidiaTeslaP100"


def test_kernel_metadata_resume_sources():
    m = launch.make_kernel_metadata("u", "run-s2", kernel_sources=["u/abr-run-s1"])
    assert m["kernel_sources"] == ["u/abr-run-s1"]


@pytest.mark.parametrize("bad", ["", "Compat", "a_b", "-x", "x-", "a" * 50, "a b"])
def test_invalid_names_rejected(bad):
    with pytest.raises(launch.LaunchError):
        launch.validate_name(bad)


def test_bad_gpu_rejected():
    with pytest.raises(launch.LaunchError):
        launch.make_kernel_metadata("u", "x", "a100")


def test_dataset_metadata_and_notes():
    assert launch.make_dataset_metadata("u")["id"] == "u/abr-code"
    n = launch.version_notes("abc123", "HEAD", "foo")
    assert "abc123" in n and "abr-foo" in n


def test_render_runner_is_valid_python_with_values():
    tpl = launch.RUNNER_TEMPLATE.read_text()
    cmd = "python x.py --a 'q' \"dq\" $HOME"
    out = launch.render_runner(tpl, cmd=cmd, commit="deadbeef", user="u", name="n",
                               dataset_version=None, gpu="t4", resume_sources=["u/abr-old"])
    tree = ast.parse(out)
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                consts[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    assert consts["RUNS"] == [{"name": None, "cmd": cmd}]
    assert consts["EXPECTED_COMMIT"] == "deadbeef"
    assert consts["DATASET_VERSION"] is None
    assert consts["KERNEL_URL"] == "https://www.kaggle.com/code/u/abr-n"
    assert consts["RESUME_SOURCES"] == ["u/abr-old"]
    assert "__ABR_" not in out


def test_render_runner_missing_placeholder():
    with pytest.raises(launch.LaunchError):
        launch.render_runner("x = 1\n", cmd="c", commit="s", user="u", name="n",
                             dataset_version=1, gpu="t4")


def test_build_kernel_dir(tmp_path):
    d = launch.build_kernel_dir(tmp_path / "k", user="u", name="n", cmd="echo hi",
                                commit="abc", dataset_version=7, gpu="t4")
    meta = json.loads((d / "kernel-metadata.json").read_text())
    runner = (d / meta["code_file"]).read_text()
    assert "DATASET_VERSION = 7" in runner
    compile(runner, "runner.py", "exec")


# ---------------------------------------------------------------- packaging
def test_archive_is_committed_tree_not_working_tree(repo, tmp_path):
    sha1 = git(repo, "rev-parse", "HEAD")
    (repo / "a.py").write_text("print('dirty')\n")          # uncommitted modification
    (repo / "untracked.py").write_text("x\n")               # untracked file
    pkg = launch.build_code_archive(repo, "HEAD", tmp_path / "ds")
    assert pkg["commit"] == sha1
    m = members(pkg["archive"])
    assert m["repo/a.py"] == b"print('v1')\n"
    assert "repo/untracked.py" not in m
    assert m["repo/.hidden"] == b"x\n"                       # dotfiles keep their names
    assert m["repo/sub/b.txt"] == b"b\n"
    assert m["repo/COMMIT"].decode().strip() == sha1
    assert m["repo/COMMIT_REF"].decode().strip() == "HEAD"
    assert "repo/git_snapshot.json" not in m                 # no provenance module -> fallback
    assert pkg["git_snapshot"] is False


def test_archive_specific_older_commit(repo, tmp_path):
    sha1 = git(repo, "rev-parse", "HEAD")
    (repo / "a.py").write_text("print('v2')\n")
    git(repo, "commit", "-qam", "c2")
    pkg = launch.build_code_archive(repo, sha1[:8], tmp_path / "ds")
    assert pkg["commit"] == sha1
    assert members(pkg["archive"])["repo/a.py"] == b"print('v1')\n"


def test_archive_deterministic(repo, tmp_path):
    a = launch.build_code_archive(repo, "HEAD", tmp_path / "1")
    b = launch.build_code_archive(repo, "HEAD", tmp_path / "2")
    assert a["archive_sha256"] == b["archive_sha256"]


def test_bad_ref(repo, tmp_path):
    with pytest.raises(launch.LaunchError):
        launch.build_code_archive(repo, "no-such-ref", tmp_path / "ds")


def test_provenance_snapshot_included(repo, tmp_path):
    prov = repo / "experiments" / "full_scale"
    prov.mkdir(parents=True)
    (prov / "provenance.py").write_text(
        "import json, subprocess, sys\n"
        "assert sys.argv[1] == 'snapshot'\n"
        "sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()\n"
        "json.dump({'commit': sha, 'files': {}}, open('git_snapshot.json', 'w'))\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "add provenance")
    sha = git(repo, "rev-parse", "HEAD")
    (repo / "a.py").write_text("dirty\n")
    pkg = launch.build_code_archive(repo, "HEAD", tmp_path / "ds")
    m = members(pkg["archive"])
    assert pkg["git_snapshot"] is True
    assert json.loads(m["repo/git_snapshot.json"])["commit"] == sha
    assert m["repo/COMMIT"].decode().strip() == sha
    assert not (repo / "git_snapshot.json").exists()        # ran in a temp checkout


# ---------------------------------------------------------------- CLI parsing / errors
def test_parse_push_output():
    out = ("Kernel version 3 successfully pushed.  Please check progress at "
           "https://www.kaggle.com/code/u/abr-n")
    p = launch.parse_push_output(out)
    assert p == {"version": 3, "url": "https://www.kaggle.com/code/u/abr-n", "error": None}
    assert launch.parse_push_output("Kernel push error: Nope")["error"] == "Nope"


def test_parse_status():
    assert launch.parse_status('u/k has status "KernelWorkerStatus.RUNNING"') == ("running", "")
    s, m = launch.parse_status('u/k has status "KernelWorkerStatus.ERROR"\nFailure message: "boom"')
    assert (s, m) == ("error", "boom")


@pytest.mark.parametrize("text,kind", [
    ("Phone verification is required to use GPU", "phone_verification"),
    ("Your account must be verified to enable internet", "phone_verification"),
    ("GPU quota exceeded for this week", "gpu_quota"),
    ("Accelerator not available for this user", "gpu_or_internet_refused"),
    ("401 Unauthorized", "auth"),
    ("something else", None),
])
def test_classify_error(text, kind):
    assert launch.classify_error(text) == kind


class FakeCLI:
    def __init__(self, push_text="Kernel version 1 successfully pushed.  Please check progress at "
                                 "https://www.kaggle.com/code/u/abr-n", push_rc=0, existing=None,
                 statuses=None, server_meta=None):
        self.user = "u"
        self.calls = []
        self.push_text, self.push_rc = push_text, push_rc
        self.existing = existing
        self.version = existing["version"] if existing else 0
        self.statuses = list(statuses or [])
        self.server_meta = server_meta or {"enable_gpu": True, "enable_internet": True, "is_private": True}

    def dataset_info(self, ref):
        if self.existing is None and self.version == 0:
            return None
        return {"version": self.version, "is_private": True}

    def dataset_status(self, ref):
        return "ready"

    def dataset_create(self, folder):
        self.calls.append(("create", folder))
        self.version = 1
        return "Your private Dataset is being created. Please check progress at ..."

    def dataset_version(self, folder, notes):
        self.calls.append(("version", folder, notes))
        self.version += 1
        return "Dataset version is being created. Please check progress at ..."

    def kernel_push(self, folder, timeout=None):
        self.calls.append(("push", folder))
        return SimpleNamespace(returncode=self.push_rc, stdout=self.push_text, stderr="")

    def kernel_status(self, ref):
        return self.statuses.pop(0)

    def kernel_pull_metadata(self, ref, folder):
        return self.server_meta


def test_upload_creates_then_versions(repo, tmp_path):
    pkg = launch.build_code_archive(repo, "HEAD", tmp_path / "ds")
    cli = FakeCLI()
    ds = launch.upload_code(cli, pkg, tmp_path / "ds", "n", poll=0)
    assert cli.calls[0][0] == "create" and ds["version"] == 1
    meta = json.loads((tmp_path / "ds" / "dataset-metadata.json").read_text())
    assert meta["id"] == "u/abr-code"
    cli2 = FakeCLI(existing={"version": 4})
    ds2 = launch.upload_code(cli2, pkg, tmp_path / "ds", "n", poll=0)
    assert cli2.calls[0][0] == "version" and pkg["commit"] in cli2.calls[0][2]
    assert ds2["version"] == 5


def test_upload_refuses_public_dataset(repo, tmp_path):
    pkg = launch.build_code_archive(repo, "HEAD", tmp_path / "ds")
    cli = FakeCLI(existing={"version": 1})
    cli.dataset_info = lambda ref: {"version": 1, "is_private": False}
    with pytest.raises(launch.LaunchError, match="PUBLIC"):
        launch.upload_code(cli, pkg, tmp_path / "ds", "n", poll=0)


def test_push_phone_verification_message(tmp_path, capsys):
    cli = FakeCLI(push_text="Kernel push error: Phone verification required to enable GPU", push_rc=0)
    with pytest.raises(launch.LaunchError, match="phone_verification"):
        launch.push_kernel(cli, tmp_path, "u", "n")
    assert "verify your" in capsys.readouterr().err


def test_push_ok():
    assert launch.push_kernel(FakeCLI(), Path("."), "u", "n")["version"] == 1


def test_server_silently_disabled_gpu(capsys):
    cli = FakeCLI(server_meta={"enable_gpu": False, "enable_internet": True, "is_private": True})
    with pytest.raises(launch.LaunchError):
        launch.verify_server_settings(cli, "u/abr-n", "u")
    assert "Phone verification" in capsys.readouterr().err


def test_wait_backoff_and_stale_status():
    # stale 'complete' from a previous version must not end the wait early
    cli = FakeCLI(statuses=[("complete", ""), ("queued", ""), ("running", ""), ("running", ""),
                            ("complete", "")])
    sleeps = []
    status, _ = launch.wait_for_kernel(cli, "u/abr-n", initial=10, max_interval=20, sleep=sleeps.append)
    assert status == "complete"
    assert sleeps == [10, 15, 20, 20]


def test_wait_error():
    cli = FakeCLI(statuses=[("running", ""), ("error", "OOM")])
    assert launch.wait_for_kernel(cli, "u/abr-n", sleep=lambda s: None) == ("error", "OOM")


def test_main_end_to_end_mocked(repo, tmp_path, monkeypatch):
    fake = FakeCLI(statuses=[("running", ""), ("complete", "")])

    def fake_output(ref, folder):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "kaggle_env.json").write_text(json.dumps({"gpu_name": "Tesla T4"}))
        return ""
    fake.kernel_output = fake_output
    monkeypatch.setattr(launch, "KaggleCLI", lambda *a, **k: fake)
    monkeypatch.setattr(launch.time, "sleep", lambda s: None)
    rc = launch.main(["--name", "t1", "--commit", "HEAD", "--cmd", "echo hi", "--repo", str(repo),
                      "--user", "u", "--wait", "--download", "results/kaggle/t1"])
    assert rc == 0
    rec = json.loads((repo / "results/kaggle/t1/launch.json").read_text())
    assert rec["commit"] == git(repo, "rev-parse", "HEAD")
    assert rec["dataset"]["version"] == 1
    assert rec["kernel_version"] == 1 and rec["kernel_ref"] == "u/abr-t1"
    assert rec["final_status"] == "complete" and rec["download"]["gpu"] == "Tesla T4"
    for k in ("launched_utc", "pushed_utc", "finished_utc", "downloaded_utc"):
        assert rec[k].endswith("Z")


# ---------------------------------------------------------------- multi-run (T4 x2)
def test_normalize_runs():
    assert launch.normalize_runs("echo a") == [{"name": None, "cmd": "echo a"}]
    r = launch.normalize_runs(["echo a", "echo b"])
    assert [x["name"] for x in r] == ["run0", "run1"]
    r = launch.normalize_runs(["echo a", "echo b"], ["A_s1", "B_s2"])
    assert [x["name"] for x in r] == ["A_s1", "B_s2"]
    with pytest.raises(launch.LaunchError):
        launch.normalize_runs(["a", "b"], ["x", "x"])
    with pytest.raises(launch.LaunchError):
        launch.normalize_runs(["a"], ["bad name"])
    with pytest.raises(launch.LaunchError):
        launch.normalize_runs([])


def test_t4x2_shape():
    assert launch.make_kernel_metadata("u", "x", "t4x2")["machine_shape"] == "NvidiaTeslaT4"


def test_cmds_file(tmp_path):
    f = tmp_path / "runs.json"
    f.write_text(json.dumps([{"name": "a", "cmd": "echo a"}, {"name": "b", "cmd": "echo b"}]))
    assert [r["name"] for r in launch.normalize_runs(launch.load_cmds_file(f))] == ["a", "b"]
    f.write_text(json.dumps({"cmd": "x"}))
    with pytest.raises(launch.LaunchError):
        launch.load_cmds_file(f)


def _run_runner(tmp_path, repo, runs, resume_sources=None, prev=None, extra_env=None):
    """Execute the generated runner locally against a fake /kaggle tree (no installs)."""
    pkg = launch.build_code_archive(repo, "HEAD", tmp_path / "ds")
    inp = tmp_path / "input"
    (inp / "abr-code").mkdir(parents=True)
    (inp / "abr-code" / "abr-code.tar.gz").write_bytes(Path(pkg["archive"]).read_bytes())
    if prev:
        for rel, content in prev.items():
            f = inp / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(content)
    kdir = launch.build_kernel_dir(tmp_path / "k", user="u", name="t", cmd=runs,
                                   commit=pkg["commit"], dataset_version=3, gpu="t4x2",
                                   kernel_sources=resume_sources)
    work = tmp_path / "work"
    env = dict(__import__("os").environ, ABR_RUNNER_WORK=str(work), ABR_RUNNER_INPUT=str(inp),
               ABR_RUNNER_SKIP_INSTALL="1", **(extra_env or {}))
    r = subprocess.run([sys.executable, str(kdir / "runner.py")], env=env, capture_output=True,
                       text=True, timeout=300)
    return r, work / "output"


def test_runner_named_runs_and_resume(repo, tmp_path):
    runs = launch.normalize_runs([
        'echo "A $ABR_RUN_NAME resumed=$ABR_RESUMED"; cat $ABR_OUTPUT_DIR/ckpt.txt; '
        'echo step2 > $ABR_OUTPUT_DIR/ckpt.txt',
        "echo B; exit 4"], ["A_s1", "B_s1"])
    prev = {"abr-old/A_s1/run_status.json": json.dumps({"sessions": 1, "state": "running"}),
            "abr-old/A_s1/ckpt.txt": "step1\n",
            "abr-old/A_s1/run.log": "old log\n"}
    r, out = _run_runner(tmp_path, repo, runs, ["u/abr-old"], prev)
    assert r.returncode == 1, r.stdout + r.stderr          # B failed -> kernel exit 1
    a = json.loads((out / "A_s1" / "run_status.json").read_text())
    b = json.loads((out / "B_s1" / "run_status.json").read_text())
    assert a["state"] == "done" and a["exit_code"] == 0 and a["sessions"] == 2
    assert b["state"] == "failed" and b["exit_code"] == 4 and b["sessions"] == 1
    log_a = (out / "A_s1" / "run.log").read_text()
    assert "A A_s1 resumed=1" in log_a and "step1" in log_a
    assert (out / "A_s1" / "ckpt.txt").read_text().strip() == "step2"
    assert (out / "A_s1" / "previous_sessions" / "run.session1.log").read_text() == "old log\n"
    env = json.loads((out / "kaggle_env.json").read_text())
    assert env["run_exit_codes"] == {"A_s1": 0, "B_s1": 4}
    assert not (tmp_path / "work" / "repo").exists()


def test_runner_single_run_layout(repo, tmp_path):
    r, out = _run_runner(tmp_path, repo, launch.normalize_runs("echo hi $CODE_COMMIT"))
    assert r.returncode == 0, r.stdout + r.stderr
    env = json.loads((out / "kaggle_env.json").read_text())
    assert env["cmd_exit_code"] == 0 and env["commit"] == env["commit_expected"]
    assert "hi " + env["commit"] in (out / "run.log").read_text()


# ---------------------------------------------------------------- queue
_qspec = importlib.util.spec_from_file_location("kaggle_queue", ROOT / "scripts" / "kaggle" / "queue.py")
kq = importlib.util.module_from_spec(_qspec)
_qspec.loader.exec_module(kq)

FAKE_CONFIG = '''
SEEDS_KEY = [42, 123, 7, 2024, 31337]
PHASE2_CONFIGS = [{"label": l, "expert_type": e} for l, e in [
    ("Transformer-Only", "transformer"), ("Mamba-Only", "mamba"), ("Jamba-1:3", "jamba-1:3"),
    ("ABR-4L-CBSC", "abr")]]
SEED_COUNT = {"Transformer-Only": 5, "Mamba-Only": 2, "Jamba-1:3": 5, "ABR-4L-CBSC": 5}
KEY_CONFIGS = {k for k, n in SEED_COUNT.items() if n == 5}
BLOCK_SWEEP_TRAIN = [8, 16, 64]
BLOCK_SWEEP_SEEDS = SEEDS_KEY[:2]
def get_config(label):
    return [c for c in PHASE2_CONFIGS if c["label"] == label][0]
def contains_mamba(cfg):
    return cfg["expert_type"] != "transformer"
def planned_runs():
    return [(c["label"], s, None) for c in PHASE2_CONFIGS for s in SEEDS_KEY[:SEED_COUNT[c["label"]]]]
'''


@pytest.fixture
def qrepo(repo):
    d = repo / "experiments" / "full_scale"
    d.mkdir(parents=True)
    (d / "config.py").write_text(FAKE_CONFIG)
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "cfg")
    return repo


def test_queue_order_and_sweep(qrepo):
    q = kq.Queue(qrepo, FakeCLI())
    st = q.init("HEAD", sweep="ABR-4L-CBSC")
    order = st["order"]
    assert len(order) == 5 + 2 + 5 + 5 + 6
    assert all(not r.startswith("Mamba-Only") for r in order[:15])  # key configs first
    assert order[:3] == ["Transformer-Only_s42", "Jamba-1_3_s42", "ABR-4L-CBSC_s42"]
    assert order[-6:] == [f"ABR-4L-CBSC_s{s}_b{b}" for s in (42, 123) for b in (8, 16, 64)]
    assert order[15:17] == ["Mamba-Only_s42", "Mamba-Only_s123"]
    r = st["runs"]["ABR-4L-CBSC_s42_b16"]
    assert "--block-size 16" in r["cmd"] and "--config ABR-4L-CBSC" in r["cmd"]
    assert "--out $ABR_OUTPUT_DIR" in r["cmd"] and st["commit"] == git(qrepo, "rev-parse", "HEAD")
    assert json.loads((qrepo / "results/kaggle/queue.json").read_text())["order"] == order
    assert q.add_sweep("ABR-4L-CBSC") == []                 # idempotent


def test_queue_exclude_non_mamba(qrepo):
    st = kq.Queue(qrepo, FakeCLI()).init("HEAD", exclude_non_mamba=True)
    assert not any(r.startswith("Transformer") for r in st["order"])


COSTS = {"total_steps": 60000, "default_steps_per_s": 2.0, "per_run_overhead_hours": 0.0,
         "session_overhead_hours": 0.5, "min_start_hours": 0.34,
         "steps_per_s": {"Transformer-Only": 4.0, "Jamba-1:3": 2.0, "ABR-4L-CBSC": 1.0,
                         "Mamba-Only": 2.0}}
# est hours: Transformer 4.17, Jamba 8.33, ABR 16.67, Mamba 8.33


def test_pack_lanes_lpt():
    lanes = kq.pack_lanes([("a", 5), ("b", 4), ("c", 3), ("d", 3), ("e", 1)], 2)
    loads = sorted(sum(h for _, h in l) for l in lanes)
    assert loads == [8, 8]


def test_plan_session_partial_first_and_budget(qrepo):
    q = kq.Queue(qrepo, FakeCLI())
    q.init("HEAD")
    q.state["runs"]["ABR-4L-CBSC_s123"].update(resume_from="u/abr-q001", hours_done=10.0)
    plan = q.plan_session(COSTS, session_hours=11.5, lanes=2)
    first = [lane[0]["run"] for lane in plan["lanes"]]
    assert "ABR-4L-CBSC_s123" in first                     # partial run resumes first
    flat = [x["run"] for lane in plan["lanes"] for x in lane]
    assert flat.count("ABR-4L-CBSC_s123") == 1
    picked = sum(plan["est_load_hours"])
    assert picked >= 2 * 11.0                               # both lanes filled
    assert picked - max(x["est_hours"] for l in plan["lanes"] for x in l) < 2 * 11.0
    assert abs(plan["est_load_hours"][0] - plan["est_load_hours"][1]) <= 17
    assert plan["lanes"][0][0]["est_hours"] > 0


def test_simulate_session_count(qrepo):
    q = kq.Queue(qrepo, FakeCLI())
    q.init("HEAD")
    total = sum(q.remaining_hours(r, COSTS) for r in q.state["order"])
    n = q.simulate(COSTS, 11.5, 2)
    assert n >= total / (2 * 11.0)
    assert n <= total / (2 * 11.0) + 3


def test_queue_advance_launch_settle_resume(qrepo, monkeypatch):
    cli = FakeCLI()
    q = kq.Queue(qrepo, cli)
    q.init("HEAD")
    monkeypatch.setattr(kq.L.time, "sleep", lambda s: None)
    kw = dict(costs=COSTS, max_sessions=1, session_hours=11.5, lanes=2)
    assert q.advance(**kw) == ["q001"]
    s = q.state["sessions"]["q001"]
    rec = json.loads((qrepo / "results/kaggle/queue/q001/launch.json").read_text())
    assert rec["session_plan"]["lanes"] == s["plan"]["lanes"] and rec["session_hours"] == 11.5
    lanes = {r["name"]: r["lane"] for r in rec["runs"]}
    assert set(lanes.values()) == {0, 1}
    assert all("{max_hours}" in r["cmd"] for r in rec["runs"])
    group = s["runs"]
    assert all(q.state["runs"][g]["status"] == "running" for g in group)
    # idempotent while the kernel runs
    cli.statuses = [("running", "")] * 3
    assert q.advance(**kw) == [] and q.advance(**kw) == []
    done, partial, never = group[0], group[1], group[-1]

    def fake_output(ref, folder):
        a, b = folder / done, folder / partial
        (a / "x").mkdir(parents=True)
        (a / "x" / "result.json").write_text("{}")
        hist = [{"started_utc": "2026-11-10T00:00:00Z", "finished_utc": "2026-11-10T04:00:00Z"}]
        (a / "run_status.json").write_text(json.dumps({"exit_code": 0, "state": "done", "history": hist}))
        b.mkdir(parents=True)
        (b / "run_status.json").write_text(json.dumps({"exit_code": 0, "state": "done", "history": hist}))
        return ""            # `never` was skipped by the runner: no output dir
    cli.kernel_output = fake_output
    cli.statuses = [("complete", "")]
    launched = q.advance(**kw)
    runs = q.state["runs"]
    assert runs[done]["status"] == "done"
    assert runs[partial]["resume_from"] == "u/abr-q001" and runs[partial]["hours_done"] == 4.0
    assert runs[never]["resume_from"] is None and runs[never]["failures"] == 0
    assert launched == ["q002"] and q.state["sessions"]["q001"]["status"] == "downloaded"
    plan2 = q.state["sessions"]["q002"]["plan"]
    assert partial in [lane[0]["run"] for lane in plan2["lanes"]]
    assert q.state["sessions"]["q002"]["resume_from"] == ["u/abr-q001"]
    # downloading again is a no-op
    cli.statuses = [("running", "")]
    assert q.advance(**kw) == []


def test_queue_failures(qrepo):
    q = kq.Queue(qrepo, FakeCLI())
    q.init("HEAD")
    sess = {"kernel_ref": "u/abr-q009", "kernel_status": "complete"}
    d = qrepo / "out"
    (d / "Mamba-Only_s42").mkdir(parents=True)
    (d / "Mamba-Only_s42" / "run_status.json").write_text(json.dumps({"exit_code": 1, "state": "failed"}))
    q._settle("Mamba-Only_s42", sess, d, max_failures=2)
    assert q.state["runs"]["Mamba-Only_s42"]["status"] == "pending"
    q._settle("Mamba-Only_s42", sess, d, max_failures=2)
    assert q.state["runs"]["Mamba-Only_s42"]["status"] == "failed"
    q.retry("Mamba-Only_s42")
    assert q.state["runs"]["Mamba-Only_s42"]["status"] == "pending"
    # kernel errored before writing run_status (e.g. install failure) counts as a failure
    q._settle("Jamba-1_3_s42", {"kernel_ref": "u/abr-q009", "kernel_status": "error"}, d, 2)
    assert q.state["runs"]["Jamba-1_3_s42"]["failures"] == 1


# ---------------------------------------------------------------- token cache
def test_token_cache_spec_and_metadata(tmp_path):
    (tmp_path / "wikitext103_train_gpt2.pt").write_bytes(b"abc")
    (tmp_path / "other.pt").write_bytes(b"x")
    spec = launch.token_cache_spec("u/abr-tokens", tmp_path)
    import hashlib
    assert spec == {"ref": "u/abr-tokens",
                    "expected_sha256": {"wikitext103_train_gpt2.pt": hashlib.sha256(b"abc").hexdigest()}}
    assert launch.token_cache_spec(None) is None
    with pytest.raises(launch.LaunchError):
        launch.token_cache_spec("noslash", tmp_path)
    m = launch.make_kernel_metadata("u", "n", extra_datasets=["u/abr-tokens"])
    assert m["dataset_sources"] == ["u/abr-code", "u/abr-tokens"]


def _cache_prev(content=b"abc"):
    return {"abr-tokens/wikitext103_train_gpt2.pt": content.decode()}


def test_runner_token_cache_verified(repo, tmp_path, monkeypatch):
    import hashlib
    cache = {"ref": "u/abr-tokens",
             "expected_sha256": {"wikitext103_train_gpt2.pt": hashlib.sha256(b"abc").hexdigest()}}
    orig = launch.build_kernel_dir
    monkeypatch.setattr(launch, "build_kernel_dir", lambda *a, **k: orig(*a, token_cache=cache, **k))
    r, out = _run_runner(tmp_path, repo, launch.normalize_runs("echo cache=$ABR_CACHE_DIR"),
                         prev=_cache_prev())
    assert r.returncode == 0, r.stdout + r.stderr
    env = json.loads((out / "kaggle_env.json").read_text())
    assert env["token_cache"]["verified"] is True
    assert env["token_cache"]["dir"].endswith("abr-tokens")
    assert "cache=" + env["token_cache"]["dir"] in (out / "run.log").read_text()


def test_runner_token_cache_mismatch_refuses(repo, tmp_path, monkeypatch):
    cache = {"ref": "u/abr-tokens", "expected_sha256": {"wikitext103_train_gpt2.pt": "0" * 64}}
    orig = launch.build_kernel_dir
    monkeypatch.setattr(launch, "build_kernel_dir", lambda *a, **k: orig(*a, token_cache=cache, **k))
    r, out = _run_runner(tmp_path, repo, launch.normalize_runs("echo SHOULD_NOT_RUN"),
                         prev=_cache_prev())
    assert r.returncode == 1
    env = json.loads((out / "kaggle_env.json").read_text())
    assert "mismatch" in env["runner_error"] and "cmd_exit_code" not in env


def test_patch_setup_for_arch(tmp_path):
    src = launch.RUNNER_TEMPLATE.read_text()
    ns = {}
    exec("import re\n" + src[src.index("def patch_setup_for_arch"):src.index("def pypi_sdist")], ns)
    setup = tmp_path / "setup.py"
    setup.write_text(
        "cc_flag = []\n"
        "if True:\n"
        '    cc_flag.append("-gencode")\n'
        '    cc_flag.append("arch=compute_75,code=sm_75")\n'
        '    cc_flag.append("-gencode")\n'
        '    cc_flag.append("arch=compute_80,code=sm_80")\n'
        "    if True:\n"
        '        cc_flag.append("-gencode")\n'
        '        cc_flag.append("arch=compute_90,code=sm_90")\n')
    assert ns["patch_setup_for_arch"](str(setup), "75") == ["75", "80", "90"]
    g = {}
    exec(setup.read_text(), g)
    assert g["cc_flag"] == ["-gencode", "arch=compute_75,code=sm_75"]
    with pytest.raises(RuntimeError):
        ns["patch_setup_for_arch"](str(setup), "60")


def test_kernel_dir_extra_datasets(tmp_path):
    d = launch.build_kernel_dir(tmp_path / "k", user="u", name="n", cmd="echo", commit="c",
                                dataset_version=1, gpu="t4x2",
                                token_cache={"ref": "u/tok", "expected_sha256": {}},
                                wheels_dataset="u/whl")
    meta = json.loads((d / "kernel-metadata.json").read_text())
    assert meta["dataset_sources"] == ["u/abr-code", "u/tok", "u/whl"]


def test_runner_lanes_max_hours_and_skip(repo, tmp_path):
    runs = [{"name": "a", "cmd": "echo hours={max_hours} lane=$ABR_MAX_HOURS", "lane": 1},
            {"name": "b", "cmd": "echo b", "lane": 1}]
    r, out = _run_runner(tmp_path, repo, launch.normalize_runs(runs),
                         extra_env={"ABR_SESSION_BUDGET_S": "7200"})
    assert r.returncode == 0, r.stdout + r.stderr
    a = json.loads((out / "a" / "run_status.json").read_text())
    assert a["lane"] == 1 and 1.9 < a["max_hours"] <= 2.0
    assert f"hours={a['max_hours']:.3f}" in (out / "a" / "run.log").read_text()
    # budget too small to start anything: runs are skipped, no output dirs, exit 0
    r, out = _run_runner(tmp_path / "2", repo, launch.normalize_runs(runs),
                         extra_env={"ABR_SESSION_BUDGET_S": "600"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert not (out / "a").exists() and not (out / "b").exists()
    assert json.loads((out / "kaggle_env.json").read_text())["skipped_runs"] == ["a", "b"]
