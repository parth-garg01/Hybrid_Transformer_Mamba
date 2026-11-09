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
    assert consts["CMD"] == cmd
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

    def kernel_push(self, folder):
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
