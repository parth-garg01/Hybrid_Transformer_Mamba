"""
Full ablation training script.
Trains every config in ABLATION_CONFIGS × SEEDS.
Saves per-run JSON and final aggregate CSV.

Run on Colab:
    python train_ablation.py
"""
import sys, os, json, time, math, csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import numpy as np

from config import (ABLATION_CONFIGS, SEEDS, TRAIN_STEPS, BATCH_SIZE, SEQ_LEN,
                    LR, LR_WARMUP, WEIGHT_DECAY, GRAD_CLIP, EVAL_EVERY,
                    EVAL_STEPS, LAMBDA_LB, LAMBDA_ENT, D_MODEL, N_HEADS,
                    BLOCK_SIZE, FFN_MULT, RESULTS_DIR)
from dataset import WikiTextBPE
from models_scaled import build_model, count_params

assert torch.cuda.is_available(), "CUDA required. Run on Colab/Kaggle with GPU."
device = torch.device("cuda")
os.makedirs(RESULTS_DIR, exist_ok=True)


def cosine_lr(step, warmup, total, base_lr):
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    if total <= warmup:
        return base_lr
    progress = (step - warmup) / (total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def load_balancing_loss(z_soft):
    """Encourage uniform expert utilization."""
    avg = z_soft.mean(dim=(0, 1))
    return (avg * avg).sum() * z_soft.shape[-1]


def entropy_loss(z_soft):
    avg = z_soft.mean(dim=(0, 1))
    return -(avg * (avg + 1e-9).log()).sum()


def evaluate(model, val_ds, criterion, n_steps, seq_len):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(n_steps):
            x, y = val_ds.get_batch(BATCH_SIZE, device=device)
            logits = model(x)
            total += criterion(logits.view(-1, val_ds.vocab_size), y.view(-1)).item()
    model.train()
    return total / n_steps


def run_label(label):
    return label.replace(' ', '_').replace('=', '').replace('(', '').replace(')', '')


def save_checkpoint(label, seed, step, model, optimizer, curve_steps, curve_train,
                    curve_val, routing_history, elapsed_so_far):
    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "curve_steps": curve_steps,
        "curve_train": curve_train,
        "curve_val": curve_val,
        "routing_history": routing_history,
        "elapsed_so_far": elapsed_so_far,
    }
    path = os.path.join(RESULTS_DIR, f"{run_label(label)}_seed{seed}_ckpt.pt")
    torch.save(ckpt, path)


def load_checkpoint(label, seed, model, optimizer):
    path = os.path.join(RESULTS_DIR, f"{run_label(label)}_seed{seed}_ckpt.pt")
    if not os.path.exists(path):
        return 0, [], [], [], [], 0.0
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"  Resumed from checkpoint at step {ckpt['step']}", flush=True)
    return (ckpt["step"], ckpt["curve_steps"], ckpt["curve_train"],
            ckpt["curve_val"], ckpt["routing_history"], ckpt["elapsed_so_far"])


CKPT_EVERY = 2_000   # save checkpoint every N steps


def train_one(cfg_tuple, seed, train_ds, val_ds):
    label = cfg_tuple[0]

    # Skip if final result already exists (allows re-running after partial crash)
    fname = f"{run_label(label)}_seed{seed}.json"
    fpath = os.path.join(RESULTS_DIR, fname)
    if os.path.exists(fpath):
        print(f"  [SKIP] {label} seed={seed} — result already exists.", flush=True)
        with open(fpath) as f:
            return json.load(f)

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model(cfg_tuple, train_ds.vocab_size,
                        d_model=D_MODEL, n_heads=N_HEADS,
                        block_size=BLOCK_SIZE, ffn_mult=FFN_MULT,
                        max_seq_len=SEQ_LEN + 16).to(device)

    params = count_params(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                   weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))

    # Resume from checkpoint if one exists
    start_step, curve_steps, curve_train, curve_val, routing_history, elapsed_base = \
        load_checkpoint(label, seed, model, optimizer)

    t0 = time.time()

    for step in range(start_step + 1, TRAIN_STEPS + 1):
        lr = cosine_lr(step, LR_WARMUP, TRAIN_STEPS, LR)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = train_ds.get_batch(BATCH_SIZE, device=device)
        logits = model(x)
        ce = criterion(logits.view(-1, train_ds.vocab_size), y.view(-1))

        aux = torch.tensor(0.0, device=device)
        if hasattr(model, "last_routing") and model.last_routing is not None:
            z = model.last_routing
            aux = LAMBDA_LB * load_balancing_loss(z) - LAMBDA_ENT * entropy_loss(z)

        (ce + aux).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % EVAL_EVERY == 0 or step == TRAIN_STEPS:
            val_loss = evaluate(model, val_ds, criterion, EVAL_STEPS, SEQ_LEN)
            curve_steps.append(step)
            curve_train.append(ce.item())
            curve_val.append(val_loss)

            rs = {}
            if hasattr(model, "get_routing_stats"):
                rs = model.get_routing_stats()
                routing_history.append({"step": step, **rs})

            ppl_t = math.exp(min(ce.item(), 20))
            ppl_v = math.exp(min(val_loss, 20))
            r_str = (f"  T:{rs.get('transformer_pct', 0):.1%} "
                     f"M:{rs.get('mamba_pct', 0):.1%} "
                     f"H:{rs.get('entropy', 0):.3f}") if rs else ""
            print(f"  [{label} seed={seed}] step={step:5d} | "
                  f"train={ce.item():.4f} PPL={ppl_t:.1f} | "
                  f"val={val_loss:.4f} PPL={ppl_v:.1f}{r_str}", flush=True)

        # Periodic checkpoint so training can resume after interruption
        if step % CKPT_EVERY == 0:
            save_checkpoint(label, seed, step, model, optimizer,
                            curve_steps, curve_train, curve_val, routing_history,
                            elapsed_base + time.time() - t0)

    elapsed = elapsed_base + time.time() - t0
    final_val = curve_val[-1]
    final_ppl = math.exp(min(final_val, 20))

    result = {
        "label": label, "seed": seed, "params": params,
        "val_loss": final_val, "val_ppl": final_ppl,
        "time_s": elapsed,
        "curve_steps": curve_steps,
        "curve_train":  curve_train,
        "curve_val":    curve_val,
        "routing_history": routing_history,
    }

    with open(fpath, "w") as f:
        json.dump(result, f)

    # Clean up checkpoint once final result is saved
    ckpt_path = os.path.join(RESULTS_DIR, f"{run_label(label)}_seed{seed}_ckpt.pt")
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    return result


def main():
    print("Loading dataset (WikiText-103, GPT-2 BPE)...")
    train_ds = WikiTextBPE("train",      seq_len=SEQ_LEN, max_tokens=50_000_000)
    val_ds   = WikiTextBPE("validation", seq_len=SEQ_LEN)

    all_results = []
    for cfg in ABLATION_CONFIGS:
        for seed in SEEDS:
            print(f"\n{'='*60}")
            print(f"Config: {cfg[0]}  |  seed={seed}")
            print(f"{'='*60}")
            r = train_one(cfg, seed, train_ds, val_ds)
            all_results.append(r)

    # ── Aggregate: mean ± std across seeds ───────────────────────────────────
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in all_results:
        grouped[r["label"]].append(r)

    agg_rows = []
    for label, runs in grouped.items():
        ppls   = [r["val_ppl"]  for r in runs]
        losses = [r["val_loss"] for r in runs]
        params = runs[0]["params"]
        agg_rows.append({
            "Model":        label,
            "Params":       params,
            "Val_PPL_mean": round(float(np.mean(ppls)),   3),
            "Val_PPL_std":  round(float(np.std(ppls)),    3),
            "Val_Loss_mean":round(float(np.mean(losses)), 4),
            "Val_Loss_std": round(float(np.std(losses)),  4),
        })

    csv_path = os.path.join(RESULTS_DIR, "ablation_aggregate.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=agg_rows[0].keys())
        w.writeheader(); w.writerows(agg_rows)

    print(f"\n\n{'='*60}")
    print("ABLATION RESULTS  (mean ± std across seeds)")
    print(f"{'='*60}")
    print(f"{'Model':<30} {'Params':>10} {'PPL mean':>10} {'PPL std':>8}")
    print("-" * 60)
    for row in agg_rows:
        print(f"{row['Model']:<30} {row['Params']:>10,} "
              f"{row['Val_PPL_mean']:>10.2f} ±{row['Val_PPL_std']:>7.2f}")
    print(f"\nSaved -> {csv_path}")

    # ── Welch t-tests for key CBSC comparisons ───────────────────────────────
    try:
        from scipy import stats as scipy_stats
        print(f"\n{'='*60}")
        print("SIGNIFICANCE TESTS  (Welch two-sample t-test, two-tailed)")
        print(f"{'='*60}")
        pairs = [
            ("ABR-2L-state=OFF",   "ABR-2L-state=ON (Ours)", "CBSC additive vs OFF (2L)"),
            ("ABR-2L-state=ON (Ours)", "ABR-2L-CBSC-Gated",  "Gated vs additive CBSC"),
            ("ABR-1L-state=OFF",   "ABR-1L-state=ON",        "CBSC additive vs OFF (1L)"),
            ("Jamba-Style-4L",     "ABR-2L-state=ON (Ours)", "ABR vs Jamba-style"),
        ]
        for label_a, label_b, desc in pairs:
            ppls_a = [r["val_ppl"] for r in grouped.get(label_a, [])]
            ppls_b = [r["val_ppl"] for r in grouped.get(label_b, [])]
            if len(ppls_a) >= 2 and len(ppls_b) >= 2:
                t, p = scipy_stats.ttest_ind(ppls_a, ppls_b, equal_var=False)
                sig  = "**" if p < 0.05 else ("*" if p < 0.10 else "ns")
                print(f"  {desc:<40} t={t:+.3f}  p={p:.4f}  {sig}")
    except ImportError:
        print("  (scipy not available — skipping significance tests)")
    # ── End significance tests ────────────────────────────────────────────────


if __name__ == "__main__":
    main()
