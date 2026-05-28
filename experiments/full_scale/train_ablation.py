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
        return base_lr * step / warmup
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


def train_one(cfg_tuple, seed, train_ds, val_ds):
    label = cfg_tuple[0]
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

    curve_steps, curve_train, curve_val = [], [], []
    routing_history = []
    t0 = time.time()

    for step in range(1, TRAIN_STEPS + 1):
        # learning rate schedule
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

            # routing stats
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

    elapsed = time.time() - t0
    final_val  = curve_val[-1]
    final_ppl  = math.exp(min(final_val, 20))

    result = {
        "label": label, "seed": seed, "params": params,
        "val_loss": final_val, "val_ppl": final_ppl,
        "time_s": elapsed,
        "curve_steps": curve_steps,
        "curve_train":  curve_train,
        "curve_val":    curve_val,
        "routing_history": routing_history,
    }

    fname = f"{label.replace(' ', '_').replace('=', '').replace('(', '').replace(')', '')}_seed{seed}.json"
    with open(os.path.join(RESULTS_DIR, fname), "w") as f:
        json.dump(result, f)

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


if __name__ == "__main__":
    main()
