"""
Unified training entry point for Phase 2 (and legacy reproduction).

One invocation = one (config, seed[, block size]) run:

    python train.py --config ABR-4L-CBSC --seed 42
    python train.py --config Transformer-Only --seed 7 --device mps --wandb online
    python train.py --config ABR-2L-CBSC --seed 42 --block-size 16 --max-hours 11.5
    python train.py --legacy --config "ABR-2L-state=ON (Ours)" --seed 42 --out results/legacy

Running the same command again resumes from ``<out>/checkpoints/<run_name>/ckpt.pt``
(model, optimizer, step, all RNG states incl. the data generator, curves, W&B id).
With ``--max-hours`` the run checkpoints and exits cleanly before the budget
is used up (Kaggle 12 h sessions). Finished runs are not retrained.

Outputs (PRD §0.4 C):
    <out>/<run_id>/train.log       full stdout/stderr of every session (appended)
    <out>/<run_id>/result.json     via provenance.finish_run (never overwritten)
    <out>/<run_id>/wandb/          offline W&B files (``wandb sync`` them later)
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS

import argparse
import datetime as _dt
import importlib
import math
import random
import re
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import config as C                                    # noqa: E402
from dataset import load_tokens, get_batch, batch_at  # noqa: E402
from evaluation import evaluate_full                  # noqa: E402


# Imported lazily so tests can substitute fakes through sys.modules.
def _ms():
    return importlib.import_module("models_scaled")


def _prov():
    return importlib.import_module("provenance")


DONE, INCOMPLETE = "done", "incomplete"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train one config/seed (PRD Phase 2).",
                                allow_abbrev=False)
    p.add_argument("--config", required=True, help="label from config.PHASE2_CONFIGS "
                   "(or LEGACY_CONFIGS with --legacy)")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=None,
                   help=f"default {C.TRAIN_STEPS} ({C.LEGACY_TRAIN_STEPS} with --legacy)")
    p.add_argument("--block-size", type=int, default=C.BLOCK_SIZE)
    p.add_argument("--attn-frac", type=float, default=None,
                   help="override attn_frac (e.g. Random-Routing matched to a learned router)")
    p.add_argument("--out", default=C.PHASE2_RESULTS_DIR)
    p.add_argument("--debug", action="store_true",
                   help="allow a dirty git tree; the run is marked non-reportable")
    p.add_argument("--allow-gru-fallback", action="store_true",
                   help="allow the GRU stand-in for Mamba (non-reportable)")
    p.add_argument("--legacy", action="store_true",
                   help="reproduce the old pipeline (old configs, last-layer detached aux, "
                        "200 random val batches, global RNG for batches)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--wandb", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--max-hours", type=float, default=None,
                   help="wall-clock budget for THIS session; checkpoint and exit before it")
    # Development / test overrides (not for reportable runs).
    S = argparse.SUPPRESS
    p.add_argument("--batch-size", type=int, default=C.BATCH_SIZE, help=S)
    p.add_argument("--seq-len", type=int, default=C.SEQ_LEN, help=S)
    p.add_argument("--ckpt-every", type=int, default=C.CKPT_EVERY, help=S)
    p.add_argument("--eval-every", type=int, default=C.EVAL_EVERY, help=S)
    p.add_argument("--eval-batches", type=int, default=C.EVAL_BATCHES, help=S)
    p.add_argument("--log-every", type=int, default=50, help=S)
    p.add_argument("--max-train-tokens", type=int, default=None, help=S)
    p.add_argument("--exit-margin-min", type=float, default=15.0, help=S)
    p.add_argument("--stop-after-steps", type=int, default=None, help=S)
    return p.parse_args(argv)


def resolve_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("--device mps requested but MPS is not available")
    return torch.device(name)


def run_name_for(cfg, seed, block_size, legacy):
    slug = re.sub(r"[^A-Za-z0-9.-]+", "_", cfg["label"]).strip("_")
    name = f"{slug}_s{seed}"
    if cfg["expert_type"] == "abr":
        name += f"_b{block_size}"
    if legacy:
        name += "_legacy"
    return name


# ─────────────────────────────────────────────────────────────────────────────
# Losses, schedule, helpers
# ─────────────────────────────────────────────────────────────────────────────

def cosine_lr(step, warmup, total, base_lr):
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    if total <= warmup:
        return base_lr
    progress = (step - warmup) / (total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def _avg_routing(z):
    return z.reshape(-1, z.shape[-1]).mean(dim=0)


def load_balancing_loss(z_soft):
    """E * sum_e p_e^2 of the batch-average routing distribution (1 when uniform)."""
    avg = _avg_routing(z_soft)
    return (avg * avg).sum() * z_soft.shape[-1]


def entropy_loss(z_soft):
    """Entropy of the batch-average routing distribution (maximised via -λ_ent)."""
    avg = _avg_routing(z_soft)
    return -(avg * (avg + 1e-9).log()).sum()


def aux_losses(routing_probs, lambda_lb=C.LAMBDA_LB, lambda_ent=C.LAMBDA_ENT):
    """Aux loss averaged over ALL layers' (non-detached) routing probs (fixes B4).

    Returns (aux_tensor_or_None, parts) where parts holds detached per-layer and
    mean values for logging.
    """
    zs = [z for z in (routing_probs or []) if z is not None]
    if not zs:
        return None, {}
    lbs = torch.stack([load_balancing_loss(z) for z in zs])
    ents = torch.stack([entropy_loss(z) for z in zs])
    lb, ent = lbs.mean(), ents.mean()
    aux = lambda_lb * lb - lambda_ent * ent
    return aux, {"lb": lb.detach(), "ent": ent.detach(),
                 "lb_layers": lbs.detach(), "ent_layers": ents.detach()}


def router_temperatures(model):
    """{module_name: tau} for every submodule with a numeric ``temperature``."""
    out = {}
    for name, m in model.named_modules():
        if m is model or not hasattr(m, "temperature"):
            continue
        t = getattr(m, "temperature")
        if isinstance(t, torch.Tensor) and t.numel() == 1:
            out[name] = float(t.item())
        elif isinstance(t, (int, float)) and not isinstance(t, bool):
            out[name] = float(t)
    return out


def set_router_temperatures(model, temps):
    for name, m in model.named_modules():
        if name in temps and hasattr(m, "temperature"):
            t = getattr(m, "temperature")
            if isinstance(t, torch.Tensor):
                with torch.no_grad():
                    t.fill_(temps[name])
            else:
                m.temperature = float(temps[name])


def _jsonable(x):
    if isinstance(x, torch.Tensor):
        return x.item() if x.numel() == 1 else x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _flatten_numeric(d, prefix):
    out = {}
    for k, v in d.items():
        key = f"{prefix}/{k}"
        if isinstance(v, dict):
            out.update(_flatten_numeric(v, key))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = v
        elif isinstance(v, list) and all(isinstance(e, (int, float)) for e in v):
            out.update({f"{key}/{i}": e for i, e in enumerate(v)})
    return out


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False


def rng_state(device, gen):
    st = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
          "random": random.getstate(), "data": gen.get_state() if gen is not None else None}
    if device.type == "cuda":
        st["cuda"] = torch.cuda.get_rng_state_all()
    if device.type == "mps":
        st["mps"] = torch.mps.get_rng_state()
    return st


def set_rng_state(st, device, gen):
    torch.set_rng_state(st["torch"])
    np.random.set_state(st["numpy"])
    random.setstate(st["random"])
    if gen is not None and st.get("data") is not None:
        gen.set_state(st["data"])
    if device.type == "cuda" and "cuda" in st:
        torch.cuda.set_rng_state_all(st["cuda"])
    if device.type == "mps" and "mps" in st:
        torch.mps.set_rng_state(st["mps"])


def _save_atomic(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _commit_of(meta):
    if not isinstance(meta, dict):
        return None
    for k in ("git_commit", "commit"):
        if meta.get(k):
            return meta[k]
    g = meta.get("git")
    if isinstance(g, dict):
        return g.get("commit")
    return None


class _Tee:
    def __init__(self, stream, fh):
        self._s, self._fh = stream, fh

    def write(self, s):
        self._s.write(s)
        self._fh.write(s)
        return len(s)

    def flush(self):
        self._s.flush()
        self._fh.flush()

    def isatty(self):
        return False

    def fileno(self):
        return self._s.fileno()

    def __getattr__(self, name):
        return getattr(self._s, name)


class TeeLog:
    """Tee sys.stdout and sys.stderr into an (appended) log file."""

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._fh = open(self.path, "a", buffering=1)
        self._old = (sys.stdout, sys.stderr)
        sys.stdout, sys.stderr = _Tee(sys.stdout, self._fh), _Tee(sys.stderr, self._fh)
        return self

    def __exit__(self, *exc):
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout, sys.stderr = self._old
        self._fh.close()
        return False


def _utcnow():
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_splits(max_train_tokens=None):
    """{"train","validation","test"} -> 1-D LongTensor, plus vocab size."""
    train, vocab = load_tokens("train", max_train_tokens)
    val, _ = load_tokens("validation")
    test, _ = load_tokens("test")
    return {"train": train, "validation": val, "test": test}, vocab


@torch.no_grad()
def estimate_val(model, val, val_ix, seq_len, vocab, device):
    """Cheap val estimate on fixed windows (same windows for every run)."""
    was = model.training
    model.eval()
    tot = 0.0
    for row in val_ix:
        x, y = batch_at(val, row.tolist(), seq_len, device)
        tot += F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1)).item()
    model.train(was)
    return tot / len(val_ix)


@torch.no_grad()
def legacy_evaluate(model, val, n_steps, batch_size, seq_len, vocab, device):
    """Old evaluate(): n_steps random batches drawn from the GLOBAL torch RNG."""
    model.eval()
    tot = 0.0
    for _ in range(n_steps):
        x, y = get_batch(val, batch_size, seq_len, device)
        tot += F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1)).item()
    model.train()
    return tot / n_steps


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_run_config(args, cfg, steps):
    """Everything that defines the training trajectory (checked on resume)."""
    return {
        "model_cfg": cfg, "seed": args.seed, "steps": steps,
        "block_size": args.block_size, "batch_size": args.batch_size,
        "seq_len": args.seq_len, "d_model": C.D_MODEL, "n_heads": C.N_HEADS,
        "ffn_mult": C.FFN_MULT, "lr": C.LR, "lr_warmup": C.LR_WARMUP,
        "weight_decay": C.WEIGHT_DECAY, "betas": [0.9, 0.95], "grad_clip": C.GRAD_CLIP,
        "lambda_lb": C.LAMBDA_LB, "lambda_ent": C.LAMBDA_ENT,
        "eval_every": args.eval_every, "eval_batches": args.eval_batches,
        "eval_seed": C.EVAL_SEED,
        "full_eval": {"seq_len": C.FULL_EVAL_SEQ_LEN, "stride": C.FULL_EVAL_STRIDE},
        "max_train_tokens": args.max_train_tokens, "legacy": args.legacy,
        "dataset": f"{C.DATASET}/{C.DATASET_NAME}", "tokenizer": C.TOKENIZER,
    }


def main(argv=None):
    args = parse_args(argv)
    cfg = C.get_config(args.config, legacy=args.legacy)
    if args.attn_frac is not None:
        cfg["attn_frac"] = args.attn_frac
    if args.max_train_tokens is None:
        args.max_train_tokens = (C.LEGACY_TRAIN_MAX_TOKENS if args.legacy
                                 else C.TRAIN_MAX_TOKENS)
    steps = args.steps or (C.LEGACY_TRAIN_STEPS if args.legacy else C.TRAIN_STEPS)
    device = resolve_device(args.device)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    run_name = run_name_for(cfg, args.seed, args.block_size, args.legacy)
    cpath = os.path.join(out, "checkpoints", run_name, "ckpt.pt")
    run_config = build_run_config(args, cfg, steps)

    ckpt = torch.load(cpath, map_location="cpu", weights_only=False) \
        if os.path.exists(cpath) else None
    if ckpt is not None:
        if ckpt.get("finished"):
            print(f"[train] {run_name} already finished -> {ckpt.get('result_path')}")
            return DONE
        if ckpt["run_config"] != run_config:
            diff = sorted(k for k in set(ckpt["run_config"]) | set(run_config)
                          if ckpt["run_config"].get(k) != run_config.get(k))
            raise SystemExit(f"[train] checkpoint {cpath} was made with a different "
                             f"config (differs in {diff}); refusing to resume")

    ms, prov = _ms(), _prov()

    backend = None
    if C.contains_mamba(cfg):
        ms.require_real_mamba(allow_fallback=args.allow_gru_fallback)
        backend = ms.MAMBA_BACKEND

    data, vocab = load_splits(args.max_train_tokens)
    hashes = {k: prov.dataset_sha256(v) for k, v in data.items()}
    if ckpt is not None and ckpt["dataset_sha256"] != hashes:
        raise SystemExit("[train] dataset hash differs from the checkpoint; refusing to resume")

    debug = args.debug or bool(ckpt and ckpt.get("debug"))
    reportable = (not debug) and backend != "gru_fallback"

    t_session = time.time()
    seed_everything(args.seed)
    T = args.seq_len
    model = ms.build_model(cfg, vocab, d_model=C.D_MODEL, n_heads=C.N_HEADS,
                           block_size=args.block_size, ffn_mult=C.FFN_MULT,
                           max_seq_len=max(T, C.FULL_EVAL_SEQ_LEN) + 16).to(device)
    params = ms.count_params(model)

    # Provenance: label/seed in the config form run_id; only agreed keys go
    # top-level in `extra`, everything else is namespaced under "train_meta".
    extra = {
        "dataset_sha256": hashes, "param_count": params,
        "train_tokens": steps * args.batch_size * T, "mamba_backend": backend,
        "train_meta": {
            "run_name": run_name, "device": str(device), "legacy": args.legacy,
            "dataset_n_tokens": {k: int(len(v)) for k, v in data.items()},
            "max_hours": args.max_hours,
            "resumed_from_step": int(ckpt["step"]) if ckpt is not None else None,
            "previous_sessions": ckpt["sessions"] if ckpt is not None else [],
        },
    }
    meta = prov.start_run({"label": cfg["label"], "seed": args.seed, **run_config},
                          debug=args.debug, extra=extra)
    if not reportable:
        meta["reportable"] = False      # GRU stand-in or a debug session in this run

    if ckpt is not None:
        old, new = ckpt.get("git_commit"), _commit_of(meta)
        if old and new and old != new and not args.debug:
            raise SystemExit(f"[train] code changed since this run started ({old[:10]} -> "
                             f"{new[:10]}); rerun with --debug or start a fresh --out")
        meta["run_id"] = ckpt["run_id"]          # keep one run_id across sessions
    run_id = meta.get("run_id") or run_name
    meta["run_id"] = run_id
    log_path = os.path.join(out, run_id, "train.log")
    Logger = getattr(prov, "RunLogger", None) or TeeLog

    with Logger(log_path):
        print(f"\n===== session start {_utcnow()} | run_id={run_id} | run={run_name} | "
              f"device={device} | mamba_backend={backend} | reportable={reportable} =====",
              flush=True)
        try:
            status, result_payload = _train(args, cfg, steps, device, run_config, data,
                                            vocab, hashes, ckpt, cpath, meta, run_id,
                                            run_name, out, backend, reportable, debug,
                                            prov, model, params, t_session)
        except BaseException:
            traceback.print_exc()
            raise
        print(f"===== session end {_utcnow()} | status={status} =====", flush=True)

    if status == DONE:
        results, final_state = result_payload
        rpath = os.path.join(out, run_id, "result.json")
        if os.path.exists(rpath):     # crashed after finish_run on a previous session
            print(f"[train] {rpath} already exists; not overwriting")
            result_path = rpath
        else:
            result_path = prov.finish_run(meta, results, out, log_path)
        final_state["result_path"] = str(result_path)
        _save_atomic(final_state, cpath)
        print(f"[train] result -> {result_path}")
    return status


def _train(args, cfg, steps, device, run_config, data, vocab, hashes, ckpt, cpath, meta,
           run_id, run_name, out, backend, reportable, debug, prov, model, params,
           t_session):
    B, T = args.batch_size, args.seq_len
    train, val = data["train"], data["validation"]
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY,
                            betas=(0.9, 0.95))
    gen = None if args.legacy else torch.Generator().manual_seed(args.seed)
    vg = torch.Generator().manual_seed(C.EVAL_SEED)
    val_ix = torch.randint(len(val) - T - 1, (args.eval_batches, B), generator=vg)

    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        set_router_temperatures(model, ckpt.get("router_temps", {}))
        set_rng_state(ckpt["rng"], device, gen)
        start_step = int(ckpt["step"])
        curves, routing_history = ckpt["curves"], ckpt["routing_history"]
        elapsed_base, sessions = float(ckpt["elapsed_s"]), list(ckpt["sessions"])
        interval = dict(ckpt["interval"])
        wandb_id = ckpt.get("wandb_id")
        print(f"[train] resumed from step {start_step}", flush=True)
    else:
        start_step, elapsed_base, sessions = 0, 0.0, []
        curves = {"step": [], "train_loss": [], "val_loss_est": [], "lr": [], "tokens": []}
        routing_history = []
        interval = {"sum": 0.0, "n": 0}
        wandb_id = None
    sessions.append({"start_utc": _utcnow(), "start_step": start_step,
                     "device": str(device), "git_commit": _commit_of(meta)})
    print(f"[train] {cfg['label']} seed={args.seed} params={params:,} steps={steps} "
          f"B={B} T={T} block={args.block_size}", flush=True)

    # ── W&B ──────────────────────────────────────────────────────────────────
    run_dir = os.path.join(out, run_id)
    wb, wandb_url = None, None
    if args.wandb != "disabled":
        import wandb
        os.makedirs(run_dir, exist_ok=True)
        wandb_id = wandb_id or wandb.util.generate_id()
        wb = wandb.init(project=C.WANDB_PROJECT, entity=C.WANDB_ENTITY, name=run_id, id=wandb_id, resume="allow",
                        mode=args.wandb, dir=run_dir,
                        config={**run_config, "run_id": run_id, "run_name": run_name,
                                "params": params, "mamba_backend": backend,
                                "git_commit": _commit_of(meta), "dataset_sha256": hashes,
                                "reportable": reportable})
        if args.wandb == "online" and getattr(wb, "url", None):
            wandb_url = wb.url
            prov.set_wandb_url(meta, wandb_url)

    def wlog(d, step):
        if wb is not None:
            wb.log(d, step=step)

    def snapshot(step, finished=False):
        return {"step": step, "model": model.state_dict(), "optimizer": opt.state_dict(),
                "router_temps": router_temperatures(model), "rng": rng_state(device, gen),
                "curves": curves, "routing_history": routing_history,
                "interval": interval, "elapsed_s": elapsed_base + time.time() - t_session,
                "sessions": sessions, "run_config": run_config, "run_id": run_id,
                "run_name": run_name, "wandb_id": wandb_id, "dataset_sha256": hashes,
                "git_commit": (ckpt or {}).get("git_commit") or _commit_of(meta),
                "debug": debug, "finished": finished}

    if ckpt is None:
        _save_atomic(snapshot(0), cpath)        # pins run_id / wandb id from step 0

    budget_s = None
    if args.max_hours is not None:
        budget_s = max(0.0, args.max_hours * 3600 - args.exit_margin_min * 60)
    has_temp_update = hasattr(model, "update_router_temperatures")
    model.train()

    for step in range(start_step + 1, steps + 1):
        lr = cosine_lr(step, C.LR_WARMUP, steps, C.LR)
        for pg in opt.param_groups:
            pg["lr"] = lr
        x, y = get_batch(train, B, T, device, generator=gen)
        logits = model(x)
        ce = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))

        if args.legacy:
            z = getattr(model, "last_routing", None)
            if z is None and getattr(model, "routing_probs", None):
                z = model.routing_probs[-1]
            aux, parts = None, {}
            if z is not None:
                z = z.detach()          # old behaviour: no gradient (bug B4)
                aux = C.LAMBDA_LB * load_balancing_loss(z) - C.LAMBDA_ENT * entropy_loss(z)
        else:
            aux, parts = aux_losses(getattr(model, "routing_probs", None),
                                    C.LAMBDA_LB, C.LAMBDA_ENT)
        loss = ce if aux is None else ce + aux
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
        opt.step()
        opt.zero_grad(set_to_none=True)
        # Exactly once per optimizer step (fixes B5). Legacy models update in forward.
        if has_temp_update and not args.legacy:
            model.update_router_temperatures()

        ce_val = ce.item()
        interval["sum"] += ce_val
        interval["n"] += 1
        tokens = step * B * T

        if step % args.log_every == 0:
            d = {"train/loss": ce_val, "lr": lr, "tokens_seen": tokens}
            if aux is not None:
                d["train/aux"] = float(aux.item())
            for k in ("lb", "ent"):
                if k in parts:
                    d[f"train/aux_{k}"] = float(parts[k].item())
            d.update({f"tau/{k}": v for k, v in router_temperatures(model).items()})
            wlog(d, step)

        if step % args.eval_every == 0 or step == steps:
            rs = _jsonable(model.get_routing_stats()) if hasattr(model, "get_routing_stats") else {}
            taus = router_temperatures(model)
            if args.legacy:
                v = legacy_evaluate(model, val, C.EVAL_STEPS, B, T, vocab, device)
            else:
                v = estimate_val(model, val, val_ix, T, vocab, device)
            tr = interval["sum"] / max(interval["n"], 1)
            interval = {"sum": 0.0, "n": 0}
            for k, val_ in (("step", step), ("train_loss", tr), ("val_loss_est", v),
                            ("lr", lr), ("tokens", tokens)):
                curves[k].append(val_)
            entry = {"step": step, **(rs if isinstance(rs, dict) else {"stats": rs}),
                     "tau": taus}
            if parts:
                entry["aux_lb_layers"] = parts["lb_layers"].tolist()
                entry["aux_ent_layers"] = parts["ent_layers"].tolist()
            routing_history.append(entry)
            wlog({"train/loss_avg": tr, "val/loss_est": v,
                  "val/ppl_est": math.exp(min(v, 20)), "tokens_seen": tokens,
                  **(_flatten_numeric(rs, "routing") if isinstance(rs, dict) else {})}, step)
            r_str = ""
            if isinstance(rs, dict) and rs:
                r_str = " | " + " ".join(f"{k}={v_:.3f}" for k, v_ in rs.items()
                                          if isinstance(v_, (int, float)))
            print(f"  [{cfg['label']} s={args.seed}] step={step:6d} lr={lr:.2e} "
                  f"train={tr:.4f} val~={v:.4f} (ppl {math.exp(min(v, 20)):.1f}){r_str}",
                  flush=True)

        session_s = time.time() - t_session
        stop = ((args.stop_after_steps is not None and step - start_step >= args.stop_after_steps)
                or (budget_s is not None and session_s >= budget_s))
        if step % args.ckpt_every == 0 and not (stop and step < steps):
            _save_atomic(snapshot(step), cpath)
        if stop and step < steps:
            print(f"[train] session budget reached at step {step}; checkpoint saved. "
                  f"Run the same command again to resume.", flush=True)
            sessions[-1].update({"end_utc": _utcnow(), "end_step": step})
            _save_atomic(snapshot(step), cpath)
            if wb is not None:
                wb.finish()
            return INCOMPLETE, None

    # ── Final full evaluation (PRD P1.5): full val AND test, once ────────────
    fe = dict(seq_len=C.FULL_EVAL_SEQ_LEN, stride=C.FULL_EVAL_STRIDE,
              batch_size=C.FULL_EVAL_BATCH_SIZE, device=device)
    print("[train] full evaluation on validation and test splits ...", flush=True)
    val_full = evaluate_full(model, data["validation"], **fe)
    test_full = evaluate_full(model, data["test"], **fe)
    elapsed = elapsed_base + time.time() - t_session
    sessions[-1].update({"end_utc": _utcnow(), "end_step": steps})
    print(f"[train] FINAL val loss={val_full['loss']:.4f} ppl={val_full['ppl']:.2f} | "
          f"test loss={test_full['loss']:.4f} ppl={test_full['ppl']:.2f} | "
          f"wall={elapsed / 3600:.2f} h", flush=True)

    rs_final = _jsonable(model.get_routing_stats()) if hasattr(model, "get_routing_stats") else {}
    results = {
        "label": cfg["label"], "seed": args.seed, "config": cfg, "run_name": run_name,
        "block_size": args.block_size, "steps": steps, "legacy": args.legacy,
        "params": params, "tokens_seen": steps * B * T,
        "val_loss": val_full["loss"], "val_ppl": val_full["ppl"],
        "val_n_tokens": val_full["n_tokens"],
        "test_loss": test_full["loss"], "test_ppl": test_full["ppl"],
        "test_n_tokens": test_full["n_tokens"],
        "final_train_loss": curves["train_loss"][-1] if curves["train_loss"] else None,
        "final_val_loss_est": curves["val_loss_est"][-1] if curves["val_loss_est"] else None,
        "curves": curves, "routing_history": _jsonable(routing_history),
        "final_routing_stats": rs_final,
        "wall_time_s": elapsed,
        "gpu_hours": elapsed / 3600 if device.type == "cuda" else 0.0,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "n_sessions": len(sessions), "sessions": sessions,
        "mamba_backend": backend, "reportable": reportable, "debug": debug,
        "dataset_sha256": hashes, "wandb_url": wandb_url,
        "wandb_dir": run_dir if args.wandb == "offline" else None,
    }
    if args.legacy:
        results["legacy_val_loss_random200"] = results["final_val_loss_est"]
    if wb is not None:
        wb.summary.update({k: results[k] for k in ("val_loss", "val_ppl", "test_loss",
                                                    "test_ppl", "params", "tokens_seen",
                                                    "wall_time_s", "gpu_hours")})
        wb.finish()
    return DONE, (results, snapshot(steps, finished=True))


if __name__ == "__main__":
    main()
