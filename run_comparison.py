"""
Full model comparison on WikiText-103 (10M char subset).
Compares: Transformer-Only, Mamba-Only, Soft Hybrid, Jamba Baseline,
          Sparse Hybrid (state=off), MultiLayer Hybrid (state=on).
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.optim as optim

from models.transformer_only   import TransformerOnly
from models.mamba_only         import MambaOnly
from models.soft_hybrid        import SoftHybridModel
from models.hybrid_model       import DynamicHybridModel
from models.multi_layer_hybrid import MultiLayerHybridModel
from models.jamba_baseline     import JambaBaseline
from models.lm_wrapper         import LMWrapper
from models.losses             import load_balancing_loss, entropy_loss
from training.lm_dataset       import WikiTextDataset

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
STEPS     = 200
EVAL_S    = 50
D         = 128
NH        = 4

print(f"Device: {device}")
print("Loading WikiText-103 (10M chars)...", flush=True)

train_ds = WikiTextDataset(split="train",      seq_len=128, max_chars=10_000_000)
val_ds   = WikiTextDataset(split="validation", seq_len=128,
                           stoi=train_ds.stoi, itos=train_ds.itos, max_chars=None)
V = train_ds.vocab_size
print(f"Vocab: {V}\n")

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def evaluate(model):
    model.eval()
    tot = 0.0
    with torch.no_grad():
        for _ in range(EVAL_S):
            x, y = val_ds.get_batch(batch_size=16)
            x, y = x.to(device), y.to(device)
            out  = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            tot += criterion(logits.view(-1, V), y.view(-1)).item()
    model.train()
    return tot / EVAL_S

def run(base_model, label, use_aux=False):
    torch.manual_seed(42)
    model = LMWrapper(base_model, V).to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    params = count_params(model)
    t0     = time.time()
    ce_buf = []

    for step in range(STEPS):
        x, y = train_ds.get_batch(batch_size=32)
        x, y = x.to(device), y.to(device)
        out    = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        ce  = criterion(logits.view(-1, V), y.view(-1))
        aux = torch.tensor(0.0, device=device)
        if use_aux and hasattr(model.model, "last_routing") \
                   and model.model.last_routing is not None:
            z   = model.model.last_routing
            aux = 0.01 * load_balancing_loss(z) - 0.001 * entropy_loss(z)
        (ce + aux).backward()
        opt.step(); opt.zero_grad()
        ce_buf.append(ce.item())

    elapsed    = time.time() - t0
    train_loss = sum(ce_buf[-50:]) / 50
    val_loss   = evaluate(model)
    ppl_t      = torch.exp(torch.tensor(train_loss)).item()
    ppl_v      = torch.exp(torch.tensor(val_loss)).item()

    routing_str = ""
    if hasattr(model.model, "get_routing_stats"):
        r = model.model.get_routing_stats()
        if r:
            routing_str = (f"  Routing T:{r['transformer_pct']:.1%} "
                           f"M:{r['mamba_pct']:.1%} H:{r['entropy']:.4f}")

    print(f"\n[{label}]", flush=True)
    print(f"  Params     : {params:,}", flush=True)
    print(f"  Train Loss : {train_loss:.4f}  PPL {ppl_t:.2f}", flush=True)
    print(f"  Val   Loss : {val_loss:.4f}  PPL {ppl_v:.2f}", flush=True)
    print(f"  Time       : {elapsed:.1f}s", flush=True)
    if routing_str:
        print(routing_str, flush=True)
    return val_loss, ppl_v, elapsed, params

rows = []
rows.append(("1. Transformer-Only (d=256)",
    *run(TransformerOnly(d_model=D, n_heads=NH), "Transformer-Only")))

rows.append(("2. Mamba-Only (d=256, CPU fallback)",
    *run(MambaOnly(d_model=D), "Mamba-Only (CPU fallback)")))

rows.append(("3. Soft Hybrid (d=256)",
    *run(SoftHybridModel(d_model=D, n_heads=NH), "Soft Hybrid")))

rows.append(("4. Jamba Baseline (4L, d=256, 3:1 M:T)",
    *run(JambaBaseline(d_model=D, n_heads=NH, num_layers=4, mamba_ratio=3),
         "Jamba Baseline (fixed interleaving)")))

rows.append(("5. Sparse Hybrid b=8 state=OFF 1L (baseline)",
    *run(DynamicHybridModel(d_model=D, n_heads=NH, block_size=8,
                             use_cross_block_state=False),
         "Sparse Hybrid 1L state=OFF — baseline", use_aux=True)))

rows.append(("6. MultiLayer Hybrid 2L state=ON (NOVEL)",
    *run(MultiLayerHybridModel(d_model=D, n_heads=NH, block_size=8,
                                num_layers=2, use_cross_block_state=True),
         "MultiLayer Hybrid 2L state=ON — NOVEL", use_aux=True)))

print("\n" + "=" * 76, flush=True)
print("FINAL COMPARISON TABLE", flush=True)
print(f"Dataset: WikiText-103 (10M chars) | seq=128 | {STEPS} steps | d={D}", flush=True)
print("=" * 76, flush=True)
print(f"{'Model':<46} {'ValLoss':>8} {'ValPPL':>8} {'Time':>7} {'Params':>10}", flush=True)
print("-" * 76, flush=True)
best = min(r[1] for r in rows)
for name, vl, ppl, tt, p in rows:
    flag = " <-- BEST" if abs(vl - best) < 1e-6 else ""
    print(f"{name:<46} {vl:>8.4f} {ppl:>8.2f} {tt:>6.0f}s {p:>10,}{flag}", flush=True)
print("=" * 76, flush=True)

out = os.path.join(os.path.dirname(__file__), "comparison_results.json")
with open(out, "w") as f:
    json.dump([{"model": n, "val_loss": vl, "val_ppl": ppl,
                "time_s": tt, "params": p}
               for n, vl, ppl, tt, p in rows], f, indent=2)
print(f"Results saved -> {out}", flush=True)
