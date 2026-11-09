"""Tests for the training/evaluation pipeline (PRD P1.4 training loop, P1.5).

CPU-only and fast. ``models_scaled`` and ``provenance`` are replaced by tiny
fakes through ``sys.modules`` so these tests do not depend on the real model
or provenance code (they are integration-tested after merging).
"""
import hashlib
import json
import math
import os
import sys
import types

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as sps

_FS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", "full_scale"))
if _FS not in sys.path:
    sys.path.insert(0, _FS)

import config as C            # noqa: E402
import evaluation as E        # noqa: E402
import stats as S             # noqa: E402
import train as TR            # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────────────────────────────────────

EVENTS = []   # "fwd" (training-mode forward) / "upd" (temperature update)


class FakeRouter(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear = nn.Linear(d, 2, bias=False)
        self.temperature = 1.0

    def update_temperature(self):
        self.temperature = max(0.1, self.temperature * 0.97)


class FakeABR(nn.Module):
    """Tiny block-routed LM: per-layer router -> soft gate on a residual MLP."""

    def __init__(self, vocab, n_layers=2, d=16, block=8):
        super().__init__()
        self.block = block
        self.emb = nn.Embedding(vocab, d)
        self.routers = nn.ModuleList([FakeRouter(d) for _ in range(n_layers)])
        self.mix = nn.ModuleList([nn.Linear(d, d) for _ in range(n_layers)])
        self.head = nn.Linear(d, vocab)
        self.routing_probs = []

    def forward(self, x):
        h = self.emb(x)
        B, T, D = h.shape
        N = T // self.block
        self.routing_probs = []
        for r, lin in zip(self.routers, self.mix):
            # causal-ish summary: cumulative mean, so later tokens see earlier ones
            ctx = h.cumsum(1) / torch.arange(1, T + 1).view(1, T, 1)
            logits = r.linear(h[:, :N * self.block].reshape(B, N, self.block, D).mean(2))
            if self.training:
                g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
                logits = logits + g
            z = F.softmax(logits / r.temperature, dim=-1)
            self.routing_probs.append(z)
            gate = z[..., 0].repeat_interleave(self.block, dim=1)
            if gate.shape[1] < T:
                gate = F.pad(gate, (0, T - gate.shape[1]), value=1.0)
            h = h + torch.tanh(lin(ctx)) * gate.unsqueeze(-1)
        if self.training:
            EVENTS.append("fwd")
        return self.head(h)

    def update_router_temperatures(self):
        EVENTS.append("upd")
        for r in self.routers:
            r.update_temperature()

    def get_routing_stats(self):
        z = self.routing_probs[-1].detach()
        avg = z.reshape(-1, 2).mean(0)
        return {"transformer_pct": float(avg[0]), "mamba_pct": float(avg[1]),
                "entropy": float(-(avg * avg.log()).sum())}


def make_fake_models():
    m = types.ModuleType("models_scaled")
    m.MAMBA_BACKEND = "gru_fallback"
    m.built = []

    def require_real_mamba(allow_fallback=False):
        if not allow_fallback:
            raise RuntimeError("real mamba_ssm required")

    def build_model(cfg, vocab_size, d_model=256, n_heads=8, block_size=8,
                    ffn_mult=4, max_seq_len=1024):
        m.built.append(dict(cfg))
        return FakeABR(vocab_size, n_layers=cfg["n_layers"], block=block_size)

    m.require_real_mamba = require_real_mamba
    m.build_model = build_model
    m.count_params = lambda model: sum(p.numel() for p in model.parameters())
    return m


def make_fake_provenance():
    p = types.ModuleType("provenance")
    p.calls = {"start": [], "finish": []}
    counter = {"n": 0}

    def start_run(config, *, debug=False, extra=None):
        counter["n"] += 1
        assert "label" in config and "seed" in config
        meta = {"run_id": f"{config['label']}_seed{config['seed']}_{counter['n']}",
                "reportable": not debug, "debug": debug, "git_commit": "abc123",
                "config": config, "wandb_url": None}
        for k, v in (extra or {}).items():
            if k in meta and k != "mamba_backend":
                raise ValueError(k)
            meta[k] = v
        p.calls["start"].append(meta)
        return meta

    def dataset_sha256(t):
        return hashlib.sha256(t.detach().cpu().to(torch.int64).numpy().tobytes()).hexdigest()

    def set_wandb_url(meta, url):
        meta["wandb_url"] = url

    def finish_run(meta, results, out_dir, log_path=None):
        d = os.path.join(out_dir, meta["run_id"])
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "result.json")
        if os.path.exists(path):
            raise FileExistsError(path)
        with open(path, "w") as f:
            json.dump({"meta": meta, "results": results}, f)
        p.calls["finish"].append((meta, results, out_dir, log_path))
        return path

    p.start_run, p.dataset_sha256 = start_run, dataset_sha256
    p.set_wandb_url, p.finish_run = set_wandb_url, finish_run
    p.RunLogger = TR.TeeLog
    return p


VOCAB = 50


def fake_splits(max_train_tokens=None):
    g = torch.Generator().manual_seed(0)
    return {"train": torch.randint(VOCAB, (6000,), generator=g),
            "validation": torch.randint(VOCAB, (1300,), generator=g),
            "test": torch.randint(VOCAB, (1100,), generator=g)}, VOCAB


@pytest.fixture
def env(monkeypatch):
    ms, prov = make_fake_models(), make_fake_provenance()
    monkeypatch.setitem(sys.modules, "models_scaled", ms)
    monkeypatch.setitem(sys.modules, "provenance", prov)
    monkeypatch.setattr(TR, "load_splits", fake_splits)
    EVENTS.clear()
    return ms, prov


def _argv(out, steps=6, extra=()):
    return ["--config", "ABR-2L-CBSC", "--seed", "0", "--steps", str(steps),
            "--out", str(out), "--debug", "--allow-gru-fallback", "--device", "cpu",
            "--wandb", "disabled", "--batch-size", "4", "--seq-len", "32",
            "--block-size", "8", "--ckpt-every", "2", "--eval-every", "2",
            "--eval-batches", "3", "--log-every", "1", *extra]


def _ckpt(out):
    path = os.path.join(out, "checkpoints", "ABR-2L-CBSC_s0_b8", "ckpt.pt")
    return torch.load(path, map_location="cpu", weights_only=False)


# ─────────────────────────────────────────────────────────────────────────────
# evaluate_full
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n,seq,stride", [(1000, 64, 32), (1001, 64, 32), (257, 256, 128),
                                          (100, 256, 128), (999, 50, 50), (777, 64, 17)])
def test_sliding_windows_score_every_token_once(n, seq, stride):
    counts = np.zeros(n, dtype=int)          # counts[j]: times target j is scored
    for s, L, f in E.sliding_windows(n, seq, stride):
        assert s >= 0 and s + L + 1 <= n and 0 <= f < L
        assert L == min(seq, n - 1)
        counts[s + 1 + f:s + 1 + L] += 1
    assert counts[0] == 0 and (counts[1:] == 1).all()


def test_evaluate_full_bigram_exact_and_covers_all_tokens():
    torch.manual_seed(0)
    model = nn.Embedding(VOCAB, VOCAB)               # logits depend only on current token
    data = torch.randint(VOCAB, (1234,))
    r = E.evaluate_full(model, data, seq_len=64, stride=32, batch_size=3)
    ref = F.cross_entropy(model(data[:-1]), data[1:]).item()
    assert r["n_tokens"] == len(data) - 1
    assert r["loss"] == pytest.approx(ref, rel=1e-6)
    assert r["ppl"] == pytest.approx(math.exp(ref), rel=1e-6)


def test_evaluate_full_deterministic():
    torch.manual_seed(1)
    model = FakeABR(VOCAB, block=8).train()          # train mode on purpose: eval must switch
    data = torch.randint(VOCAB, (900,))
    a = E.evaluate_full(model, data, seq_len=64, stride=32, batch_size=4)
    b = E.evaluate_full(model, data, seq_len=64, stride=32, batch_size=4)
    assert a == b
    assert model.training                            # mode restored


# ─────────────────────────────────────────────────────────────────────────────
# stats
# ─────────────────────────────────────────────────────────────────────────────

def test_summarize_matches_scipy():
    x = [3.1, 2.7, 3.6, 3.3, 2.9]
    s = S.summarize(x)
    assert s["n"] == 5
    assert s["std"] == pytest.approx(np.std(x, ddof=1))
    lo, hi = sps.t.interval(0.95, 4, loc=np.mean(x), scale=sps.sem(x))
    assert (s["ci_low"], s["ci_high"]) == (pytest.approx(lo), pytest.approx(hi))
    assert math.isnan(S.summarize([1.0])["std"])


def test_welch_matches_scipy():
    a, b = [10.2, 11.1, 9.8, 10.5, 10.9], [11.5, 12.3, 11.9]
    w = S.welch(a, b)
    ref = sps.ttest_ind(a, b, equal_var=False)
    assert w["t"] == pytest.approx(ref.statistic)
    assert w["p"] == pytest.approx(ref.pvalue)
    va, vb = np.var(a, ddof=1) / 5, np.var(b, ddof=1) / 3
    assert w["df"] == pytest.approx((va + vb) ** 2 / (va ** 2 / 4 + vb ** 2 / 2))
    sp = math.sqrt((4 * np.var(a, ddof=1) + 2 * np.var(b, ddof=1)) / 6)
    d = (np.mean(a) - np.mean(b)) / sp
    assert w["cohen_d"] == pytest.approx(d)
    assert w["hedges_g"] == pytest.approx(d * (1 - 3 / (4 * 8 - 9)))


def test_compare_table_and_holm():
    vals = {"A": [1.0, 1.1, 0.9], "B": [2.0, 2.1, 1.9, 2.2], "C": [1.0, 1.2, 1.1]}
    rows = S.compare_table(vals, pairs=[("A", "B", "a-b"), ("A", "C", "a-c"),
                                        ("A", "missing", "a-?")])
    assert [r["comparison"] for r in rows] == ["a-b", "a-c", "a-?"]
    assert math.isnan(rows[2]["p"]) and rows[2]["n_b"] == 0
    p = [rows[0]["p"], rows[1]["p"]]
    lo, hi = sorted(p)
    assert sorted([rows[0]["p_holm"], rows[1]["p_holm"]]) == \
        [pytest.approx(min(1, 2 * lo)), pytest.approx(max(min(1, 2 * lo), min(1, hi)))]
    # every key pair names real configs
    labels = {c["label"] for c in C.PHASE2_CONFIGS}
    assert all(a in labels and b in labels for a, b, _ in C.KEY_PAIRS)


# ─────────────────────────────────────────────────────────────────────────────
# config
# ─────────────────────────────────────────────────────────────────────────────

def test_configs():
    labels = [c["label"] for c in C.PHASE2_CONFIGS]
    assert len(labels) == len(set(labels))
    keys = {"label", "n_layers", "expert_type", "routing_mode", "cbsc_mode", "legacy",
            "attn_frac"}
    for c in C.PHASE2_CONFIGS + C.LEGACY_CONFIGS:
        assert set(c) == keys
    assert len(C.LEGACY_CONFIGS) == 9 and all(c["legacy"] for c in C.LEGACY_CONFIGS)
    assert not any(c["legacy"] for c in C.PHASE2_CONFIGS)
    assert C.seeds_for("ABR-4L-CBSC") == C.SEEDS_KEY
    assert C.seeds_for("Mamba-Only") == C.SEEDS_OTHER
    assert C.KEY_CONFIGS <= set(labels)


# ─────────────────────────────────────────────────────────────────────────────
# aux losses (B4)
# ─────────────────────────────────────────────────────────────────────────────

def test_aux_loss_uses_all_layers():
    zs = [torch.randn(4, 6, 2, requires_grad=True) for _ in range(3)]
    probs = [F.softmax(z, -1) for z in zs]
    aux, parts = TR.aux_losses(probs, lambda_lb=0.01, lambda_ent=0.001)
    lb = torch.stack([TR.load_balancing_loss(p) for p in probs]).mean()
    ent = torch.stack([TR.entropy_loss(p) for p in probs]).mean()
    assert aux.item() == pytest.approx((0.01 * lb - 0.001 * ent).item())
    assert parts["lb_layers"].shape == (3,)
    aux.backward()
    for z in zs:
        assert z.grad is not None and z.grad.abs().sum() > 0
    assert TR.aux_losses([], 0.01, 0.001) == (None, {})


def test_aux_loss_reaches_every_router_from_aux_alone():
    torch.manual_seed(0)
    model = FakeABR(VOCAB, n_layers=3, block=8).train()
    model(torch.randint(VOCAB, (2, 32)))
    aux, _ = TR.aux_losses(model.routing_probs, 0.01, 0.001)   # CE deliberately excluded
    aux.backward()
    for r in model.routers:
        assert r.linear.weight.grad is not None and r.linear.weight.grad.abs().sum() > 0


# ─────────────────────────────────────────────────────────────────────────────
# training smoke + checkpoint/resume
# ─────────────────────────────────────────────────────────────────────────────

def test_train_smoke_end_to_end(env, tmp_path):
    ms, prov = env
    out = tmp_path / "a"
    assert TR.main(_argv(out)) == TR.DONE
    # temperature updated exactly once per optimizer step, right after it
    assert EVENTS.count("upd") == 6
    assert EVENTS == ["fwd", "upd"] * 6
    meta, results, out_dir, log_path = prov.calls["finish"][0]
    assert results["steps"] == 6 and results["tokens_seen"] == 6 * 4 * 32
    assert results["val_n_tokens"] == 1299 and results["test_n_tokens"] == 1099
    assert results["curves"]["step"] == [2, 4, 6]
    assert results["mamba_backend"] == "gru_fallback" and results["reportable"] is False
    assert meta["reportable"] is False
    assert set(meta["dataset_sha256"]) == {"train", "validation", "test"}
    assert meta["param_count"] == results["params"] and meta["train_tokens"] == 768
    assert len(results["routing_history"]) == 3 and "tau" in results["routing_history"][0]
    assert log_path == os.path.join(str(out), meta["run_id"], "train.log")
    assert "session start" in open(log_path).read()
    assert os.path.exists(os.path.join(str(out), meta["run_id"], "result.json"))
    ck = _ckpt(out)
    assert ck["finished"] and ck["step"] == 6
    # running again does not retrain
    EVENTS.clear()
    assert TR.main(_argv(out)) == TR.DONE and EVENTS == []


def test_resume_reproduces_uninterrupted_run(env, tmp_path):
    ms, prov = env
    TR.main(_argv(tmp_path / "full"))
    full = _ckpt(tmp_path / "full")

    EVENTS.clear()
    out = tmp_path / "split"
    assert TR.main(_argv(out, extra=["--stop-after-steps", "3"])) == TR.INCOMPLETE
    mid = _ckpt(out)
    assert mid["step"] == 3 and not mid["finished"]
    assert TR.main(_argv(out)) == TR.DONE
    assert EVENTS.count("upd") == 6
    res = _ckpt(out)

    for k, v in full["model"].items():
        assert torch.equal(v, res["model"][k]), k
    assert full["router_temps"] == res["router_temps"]
    assert full["curves"] == res["curves"]
    a, b = prov.calls["finish"][0][1], prov.calls["finish"][1][1]
    assert a["val_loss"] == b["val_loss"] and a["test_loss"] == b["test_loss"]
    assert b["n_sessions"] == 2
    # one run_id across both sessions, one log with both sessions in it
    assert res["run_id"] == mid["run_id"]
    log = open(prov.calls["finish"][1][3]).read()
    assert log.count("session start") == 2


def test_resume_refuses_changed_config(env, tmp_path):
    out = tmp_path / "c"
    TR.main(_argv(out, extra=["--stop-after-steps", "2"]))
    with pytest.raises(SystemExit):
        TR.main(_argv(out, steps=8))


def test_requires_real_mamba_without_flag(env, tmp_path):
    argv = [a for a in _argv(tmp_path / "m") if a != "--allow-gru-fallback"]
    with pytest.raises(RuntimeError):
        TR.main(argv)


def test_legacy_mode_runs_without_temperature_calls(env, tmp_path):
    ms, prov = env
    argv = _argv(tmp_path / "l", steps=2)
    argv[argv.index("ABR-2L-CBSC")] = "ABR-2L-state=ON (Ours)"
    argv += ["--legacy"]
    assert TR.main(argv) == TR.DONE
    assert ms.built[-1]["legacy"] and ms.built[-1]["cbsc_mode"] == "legacy_inject"
    assert "upd" not in EVENTS
    assert "legacy_val_loss_random200" in prov.calls["finish"][0][1]


# ─────────────────────────────────────────────────────────────────────────────
# wrappers and benchmark
# ─────────────────────────────────────────────────────────────────────────────

def test_sweep_wrapper_runs_and_aggregates(env, tmp_path):
    import train_ablation
    import train_abr_long
    ms, prov = env
    out = tmp_path / "sweep"
    argv = _argv(out, steps=2)[4:]          # drop --config/--seed; keep the overrides
    argv[argv.index("--out") + 1] = str(out)
    assert train_ablation.main(["--configs", "ABR-2L-CBSC", "ABR-2L-noCBSC",
                                "--seeds", "1", "2", *argv])
    assert len(prov.calls["finish"]) == 4
    assert (out / "summary.csv").exists() and (out / "key_pairs.csv").exists()
    # legacy plans
    a = argparse_ns(legacy=True)
    assert len(train_ablation.plan(a)) == 9 * 3
    b = argparse_ns(block_size_sweep="ABR-4L-CBSC")
    assert sorted({blk for _, _, blk in train_ablation.plan(b)}) == C.BLOCK_SIZE_SWEEP
    assert callable(train_abr_long.main)


def argparse_ns(**kw):
    base = {"legacy": False, "configs": None, "seeds": None, "block_size_sweep": None}
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_bench_context_with_routing_fraction(env, tmp_path):
    import bench_context
    ms, prov = env
    rows = bench_context.main(["--configs", "Transformer-Only", "ABR-2L-CBSC",
                               "--lengths", "32", "--iters", "1", "--warmup", "0",
                               "--batch-size", "2", "--block-size", "8", "--device", "cpu",
                               "--allow-gru-fallback", "--debug",
                               "--routing-fractions", "ABR-2L-CBSC=0.3",
                               "--out", str(tmp_path / "bench")])
    assert {(r["config"], r["mode"]) for r in rows} == {
        ("Transformer-Only", "forward"), ("Transformer-Only", "train"),
        ("ABR-2L-CBSC", "forward"), ("ABR-2L-CBSC", "train")}
    assert all(r["status"] == "ok" and r["tokens_per_s"] > 0 for r in rows)
    abr = [c for c in ms.built if c["label"] == "ABR-2L-CBSC"][0]
    assert abr["routing_mode"] == "random" and abr["attn_frac"] == 0.3
