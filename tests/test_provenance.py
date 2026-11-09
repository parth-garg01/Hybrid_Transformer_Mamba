"""Tests for experiments/full_scale/provenance.py and the manifest/verifier scripts.

All git work happens in throwaway repos under tmp_path, with a local test identity and
`commit.gpgsign false` set in that repo only (never globally).
"""
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "experiments", "full_scale"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import provenance as prov  # noqa: E402


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


make_manifest = _load_script("make_manifest")
verify = _load_script("verify_provenance")

CONFIG = {"label": "ABR-2L", "seed": 42, "d_model": 256, "train_tokens": 1000}


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    git(r, "config", "user.name", "Test User")
    git(r, "config", "user.email", "test@example.com")
    git(r, "config", "commit.gpgsign", "false")
    git(r, "config", "tag.gpgsign", "false")
    (r / "train.py").write_text("print('hi')\n")
    (r / ".gitignore").write_text("*.ignored\n")
    git(r, "add", ".")
    git(r, "commit", "-q", "-m", "init")
    return r


def test_clean_tree_meta_fields(repo):
    meta = prov.start_run(CONFIG, repo_root=str(repo), extra={"dataset_sha256": "ab" * 32})
    for k in prov.REQUIRED_META_KEYS:
        if k not in ("end_utc", "duration_s"):
            assert k in meta, k
    assert meta["reportable"] is True
    assert meta["git_dirty"] is False
    assert meta["git_commit"] == git(repo, "rev-parse", "HEAD")
    assert meta["git_branch"] == "main"
    assert meta["run_id"].startswith("ABR-2L_seed42_")
    assert meta["run_id"].endswith("Z")
    assert meta["start_utc"].endswith("Z")
    assert meta["torch_version"] == torch.__version__
    assert meta["gpu_name"] in ("cpu", "apple-mps") or torch.cuda.is_available()
    assert isinstance(meta["pip_freeze"], list) and meta["pip_freeze"]
    assert any(line.lower().startswith("torch") for line in meta["pip_freeze"])
    assert meta["wandb_url"] is None
    assert meta["config"] == CONFIG and meta["config"] is not CONFIG
    assert meta["dataset_sha256"] == "ab" * 32
    assert isinstance(meta["kaggle"], dict) and "on_kaggle" in meta["kaggle"]
    json.dumps(meta)  # must be serialisable


def test_describe_uses_tag(repo):
    git(repo, "tag", "m0-test")  # lightweight tag: nothing to sign
    meta = prov.start_run(CONFIG, repo_root=str(repo))
    assert meta["git_describe"] == "m0-test"
    assert meta["git_tags"] == ["m0-test"]


def test_dirty_tree_refused(repo):
    (repo / "train.py").write_text("print('edited')\n")
    with pytest.raises(prov.DirtyWorkingTreeError) as e:
        prov.start_run(CONFIG, repo_root=str(repo))
    assert "train.py" in str(e.value)


def test_staged_change_refused(repo):
    (repo / "new.py").write_text("x = 1\n")
    git(repo, "add", "new.py")
    with pytest.raises(prov.DirtyWorkingTreeError):
        prov.start_run(CONFIG, repo_root=str(repo))


def test_untracked_and_ignored_allowed(repo):
    (repo / "scratch.txt").write_text("notes\n")
    (repo / "big.ignored").write_text("data\n")
    meta = prov.start_run(CONFIG, repo_root=str(repo))
    assert meta["reportable"] is True and meta["git_dirty"] is False
    assert meta["git_untracked_files"] == ["scratch.txt"]


def test_debug_mode_on_dirty_tree(repo):
    (repo / "train.py").write_text("print('edited')\n")
    meta = prov.start_run(CONFIG, repo_root=str(repo), debug=True)
    assert meta["reportable"] is False
    assert meta["git_dirty"] is True
    assert meta["git_changed_files"] == ["train.py"]


def test_debug_mode_on_clean_tree_is_still_non_reportable(repo):
    meta = prov.start_run(CONFIG, repo_root=str(repo), debug=True)
    assert meta["reportable"] is False


def test_not_a_repo_refused_unless_debug(tmp_path):
    d = tmp_path / "nogit"
    d.mkdir()
    with pytest.raises(prov.ProvenanceError):
        prov.start_run(CONFIG, repo_root=str(d))
    assert prov.start_run(CONFIG, repo_root=str(d), debug=True)["git_commit"] is None


def test_git_snapshot_for_copy_without_git(repo, tmp_path):
    snap = prov.write_git_snapshot(str(repo))
    copy = tmp_path / "kaggle_copy"
    copy.mkdir()
    for name in ("train.py", ".gitignore"):
        (copy / name).write_bytes((repo / name).read_bytes())
    (copy / prov.SNAPSHOT_FILENAME).write_bytes(open(snap, "rb").read())
    meta = prov.start_run(CONFIG, repo_root=str(copy))
    assert meta["git_commit"] == git(repo, "rev-parse", "HEAD")
    assert meta["git_source"] == "snapshot" and meta["reportable"] is True
    (copy / "train.py").write_text("print('edited on kaggle')\n")
    with pytest.raises(prov.DirtyWorkingTreeError):
        prov.start_run(CONFIG, repo_root=str(copy))


def test_mamba_backend_from_config_or_extra(repo):
    meta = prov.start_run({**CONFIG, "mamba_backend": "gru-fallback"}, repo_root=str(repo))
    assert meta["mamba_backend"] == "gru-fallback"
    meta = prov.start_run(CONFIG, repo_root=str(repo), extra={"mamba_backend": "mamba_ssm"})
    assert meta["mamba_backend"] == "mamba_ssm"


def test_extra_cannot_overwrite_recorded_fields(repo):
    with pytest.raises(ValueError):
        prov.start_run(CONFIG, repo_root=str(repo), extra={"git_commit": "deadbeef"})


def test_kaggle_detection(repo, monkeypatch):
    for name in list(os.environ):            # on a real Kaggle box, start clean
        if name.startswith("KAGGLE_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    monkeypatch.setenv("KAGGLE_KERNEL_URL", "https://www.kaggle.com/code/u/k?scriptVersionId=1")
    meta = prov.start_run(CONFIG, repo_root=str(repo))
    k = meta["kaggle"]
    assert k["on_kaggle"] is True
    assert k["markers_present"] == ["KAGGLE_KERNEL_RUN_TYPE"]
    assert k["kernel_url"].startswith("https://www.kaggle.com/")


def test_secrets_never_recorded(repo, monkeypatch, tmp_path):
    secret = "s3cr3t-value-that-must-not-leak-9137"
    for name in ("KAGGLE_KEY", "KAGGLE_USER_SECRETS_TOKEN", "WANDB_API_KEY",
                 "MY_PASSWORD", "HF_TOKEN", "KAGGLE_SECRET_THING"):
        monkeypatch.setenv(name, secret)
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")  # pretend we are on Kaggle
    monkeypatch.setattr(sys, "argv", ["train.py", "--wandb-api-key", secret,
                                      f"--hf_token={secret}", "--seed", "42"])
    cfg = {**CONFIG, "wandb_api_key": secret, "nested": {"password": secret}}
    meta = prov.start_run(cfg, repo_root=str(repo))
    out = prov.finish_run(meta, {"val_ppl": 1.0}, str(tmp_path / "results"))
    text = open(out).read()
    assert secret not in text
    assert secret not in json.dumps(meta)
    assert "KAGGLE_KEY" not in meta["kaggle"]["env"]
    assert "--seed 42" in meta["command_line"]
    assert meta["config"]["train_tokens"] == 1000  # not mistaken for a secret


def test_dataset_sha256_stability(tmp_path):
    t = torch.arange(1000, dtype=torch.int32).reshape(10, 100)
    h = prov.dataset_sha256(t)
    assert len(h) == 64
    assert h == prov.dataset_sha256(t.clone())
    assert h == prov.dataset_sha256(t.to(torch.int64))         # dtype-independent
    assert h == prov.dataset_sha256(t.t().contiguous().t())    # layout-independent
    assert h == prov.dataset_sha256(t.numpy())                 # numpy == torch
    assert h == prov.dataset_sha256(t.to(torch.int64).numpy().tobytes())
    t2 = t.clone()
    t2[3, 7] += 1
    assert prov.dataset_sha256(t2) != h
    f = tmp_path / "tokens.bin"
    f.write_bytes(t.to(torch.int64).numpy().tobytes())
    assert prov.dataset_sha256(str(f)) == h
    assert prov.dataset_sha256(f) == h
    # known value: sha256 of int64 [0, 1, 2]
    import hashlib
    assert prov.dataset_sha256(np.array([0, 1, 2])) == hashlib.sha256(
        np.array([0, 1, 2], dtype=np.int64).tobytes()).hexdigest()
    with pytest.raises(TypeError):
        prov.dataset_sha256(object())


def test_finish_run_output(repo, tmp_path):
    meta = prov.start_run(CONFIG, repo_root=str(repo))
    prov.set_wandb_url(meta, "https://wandb.ai/owner/abr-cbsc/runs/abc123")
    log = tmp_path / "raw.log"
    raw = b"step 1 loss 5.0\r\nstep 2 loss 4.0\n\xff trailing bytes kept"
    log.write_bytes(raw)
    out_dir = tmp_path / "results"
    path = prov.finish_run(meta, {"val_ppl": np.float32(12.5), "curve": np.arange(3)},
                           str(out_dir), log_path=str(log))
    assert path == os.path.join(str(out_dir), meta["run_id"], "result.json")
    doc = json.load(open(path))
    assert set(doc) == {"meta", "results"}
    assert doc["results"] == {"val_ppl": 12.5, "curve": [0, 1, 2]}
    assert doc["meta"]["wandb_url"].endswith("abc123")
    assert doc["meta"]["end_utc"].endswith("Z") and doc["meta"]["duration_s"] >= 0
    assert "end_utc" not in meta  # caller's dict not mutated
    assert (out_dir / meta["run_id"] / "train.log").read_bytes() == raw
    assert not [p for p in os.listdir(out_dir / meta["run_id"]) if p.startswith(".tmp-")]
    with pytest.raises(FileExistsError):
        prov.finish_run(meta, {}, str(out_dir))


def test_run_logger_tees_and_finish_run_accepts_same_path(repo, tmp_path, capsys):
    meta = prov.start_run(CONFIG, repo_root=str(repo))
    log_path = tmp_path / "results" / meta["run_id"] / "train.log"
    with prov.RunLogger(str(log_path)) as lg:
        print("hello stdout")
        print("hello stderr", file=sys.stderr)
    assert "hello stdout" in capsys.readouterr().out
    prov.finish_run(meta, {}, str(tmp_path / "results"), log_path=lg.path)
    text = log_path.read_text()
    assert "hello stdout" in text and "hello stderr" in text


def _make_results(repo, tmp_path, debug=False):
    if debug:
        (repo / "train.py").write_text("print('dirty')\n")
    meta = prov.start_run(CONFIG, repo_root=str(repo), debug=debug)
    log = tmp_path / "raw.log"
    log.write_text("training log\n")
    results = tmp_path / "results"
    prov.finish_run(meta, {"val_ppl": 10.0}, str(results), log_path=str(log))
    return results, meta


def test_verify_passes_on_clean_results(repo, tmp_path, capsys):
    results, meta = _make_results(repo, tmp_path)
    assert make_manifest.main([str(results)]) == 0
    manifest = (results / "MANIFEST.sha256").read_text().splitlines()
    assert [ln.split("  ", 1)[1] for ln in manifest] == sorted(
        [f"{meta['run_id']}/result.json", f"{meta['run_id']}/train.log"])
    assert verify.main([str(results), "--repo", str(repo)]) == 0
    assert "PASS" in capsys.readouterr().out


def test_verify_detects_tampered_file(repo, tmp_path, capsys):
    results, meta = _make_results(repo, tmp_path)
    make_manifest.main([str(results)])
    rj = results / meta["run_id"] / "result.json"
    rj.write_text(rj.read_text().replace('"val_ppl": 10.0', '"val_ppl": 9.0'))
    assert verify.main([str(results), "--repo", str(repo)]) == 1
    out = capsys.readouterr().out
    assert "HASH MISMATCH" in out and "result.json" in out


def test_verify_detects_extra_and_missing_files(repo, tmp_path, capsys):
    results, meta = _make_results(repo, tmp_path)
    make_manifest.main([str(results)])
    (results / "sneaky.json").write_text("{}")
    os.remove(results / meta["run_id"] / "train.log")
    assert verify.main([str(results), "--repo", str(repo)]) == 1
    out = capsys.readouterr().out
    assert "not in manifest: sneaky.json" in out
    assert "listed in manifest but missing" in out


def test_verify_detects_unknown_commit_and_missing_keys(repo, tmp_path, capsys):
    results, meta = _make_results(repo, tmp_path)
    rj = results / meta["run_id"] / "result.json"
    doc = json.loads(rj.read_text())
    doc["meta"]["git_commit"] = "0" * 40
    del doc["meta"]["pip_freeze"]
    rj.write_text(json.dumps(doc))
    make_manifest.main([str(results)])
    assert verify.main([str(results), "--repo", str(repo)]) == 1
    out = capsys.readouterr().out
    assert "does not exist" in out and "pip_freeze" in out


def test_verify_non_reportable(repo, tmp_path, capsys):
    results, _ = _make_results(repo, tmp_path, debug=True)
    make_manifest.main([str(results)])
    assert verify.main([str(results), "--repo", str(repo)]) == 0
    assert "NON-REPORTABLE" in capsys.readouterr().out
    assert verify.main([str(results), "--repo", str(repo), "--strict"]) == 1


def test_verify_no_results(tmp_path):
    p = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "verify_provenance.py"),
                        str(tmp_path / "does_not_exist")], capture_output=True, text=True)
    assert p.returncode == 0 and "0 results" in p.stdout
    empty = tmp_path / "empty"
    empty.mkdir()
    p = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "verify_provenance.py"),
                        str(empty)], capture_output=True, text=True)
    assert p.returncode == 0 and "0 results" in p.stdout
