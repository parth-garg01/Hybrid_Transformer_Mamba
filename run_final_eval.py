import sys, os, torch, torch.nn as nn, torch.optim as optim, time
sys.path.insert(0, os.path.dirname(__file__))

from models.hybrid_model import DynamicHybridModel
from models.transformer_only import TransformerOnly
from models.mamba_only import MambaOnly
from models.soft_hybrid import SoftHybridModel
from models.lm_wrapper import LMWrapper
from models.losses import load_balancing_loss, entropy_loss
from training.lm_dataset import WikiTextDataset

device = torch.device("cpu")
criterion = nn.CrossEntropyLoss()
STEPS     = 500
EVAL_S    = 80

train_ds = WikiTextDataset(split="train", seq_len=128)
val_ds   = WikiTextDataset(split="validation", seq_len=128,
                           stoi=train_ds.stoi, itos=train_ds.itos)
vocab_size = train_ds.vocab_size
print(f"Vocab: {vocab_size} chars | Dataset: WikiText-2 | Steps: {STEPS}\n", flush=True)

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def evaluate(model):
    model.eval()
    total = 0
    with torch.no_grad():
        for _ in range(EVAL_S):
            x, y = val_ds.get_batch(batch_size=16)
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            total += criterion(logits.view(-1, vocab_size), y.view(-1)).item()
    model.train()
    return total / EVAL_S

def run(base_model, label, use_aux=False):
    torch.manual_seed(42)
    model = LMWrapper(base_model, vocab_size).to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    params = count_params(model)
    t0 = time.time()
    ce_hist = []

    for step in range(STEPS):
        x, y = train_ds.get_batch(batch_size=32)
        out    = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        ce  = criterion(logits.view(-1, vocab_size), y.view(-1))
        aux = torch.tensor(0.0)
        if use_aux and hasattr(model.model, "last_routing"):
            z   = model.model.last_routing
            aux = 0.01 * load_balancing_loss(z) - 0.001 * entropy_loss(z)
        loss = ce + aux
        opt.zero_grad(); loss.backward(); opt.step()
        ce_hist.append(ce.item())

    elapsed = time.time() - t0
    train_loss = sum(ce_hist[-50:]) / 50
    val_loss   = evaluate(model)
    ppl_t = torch.exp(torch.tensor(train_loss)).item()
    ppl_v = torch.exp(torch.tensor(val_loss)).item()

    routing_str = ""
    if hasattr(model.model, "get_routing_stats"):
        r = model.model.get_routing_stats()
        if r:
            routing_str = f"  Routing -> T:{r['transformer_pct']:.1%}  M:{r['mamba_pct']:.1%}  H:{r['entropy']:.4f}"

    print(f"[{label}]", flush=True)
    print(f"  Params     : {params:,}", flush=True)
    print(f"  Train Loss : {train_loss:.4f}  PPL {ppl_t:.2f}", flush=True)
    print(f"  Val   Loss : {val_loss:.4f}  PPL {ppl_v:.2f}", flush=True)
    print(f"  Train Time : {elapsed:.1f}s", flush=True)
    if routing_str:
        print(routing_str, flush=True)
    print(flush=True)
    return val_loss, ppl_v, elapsed, params

rows = []

rows.append(("Transformer-Only",
    *run(TransformerOnly(), "Transformer-Only")))

rows.append(("Mamba-Only (fallback)",
    *run(MambaOnly(), "Mamba-Only (CPU fallback)")))

rows.append(("Soft Hybrid",
    *run(SoftHybridModel(), "Soft Hybrid")))

rows.append(("Sparse Hybrid b=8 state=OFF (baseline)",
    *run(DynamicHybridModel(block_size=8, use_cross_block_state=False),
         "Sparse Hybrid b=8 state=OFF — baseline", use_aux=True)))

rows.append(("Sparse Hybrid b=8 state=ON  (novel)",
    *run(DynamicHybridModel(block_size=8, use_cross_block_state=True),
         "Sparse Hybrid b=8 state=ON  — NOVEL", use_aux=True)))

rows.append(("Sparse Hybrid b=16 state=ON (novel)",
    *run(DynamicHybridModel(block_size=16, use_cross_block_state=True),
         "Sparse Hybrid b=16 state=ON — NOVEL", use_aux=True)))

print("=" * 72, flush=True)
print("FINAL RESULTS TABLE  (WikiText-2 | seq=128 | 500 steps | CPU)", flush=True)
print("=" * 72, flush=True)
print(f"{'Model':<42} {'ValLoss':>8} {'ValPPL':>8} {'Time(s)':>8} {'Params':>10}", flush=True)
print("-" * 72, flush=True)
best_vl = min(r[1] for r in rows)
for name, vl, ppl, tt, p in rows:
    marker = " <-- BEST" if abs(vl - best_vl) < 1e-6 else ""
    print(f"{name:<42} {vl:>8.4f} {ppl:>8.2f} {tt:>8.1f} {p:>10,}{marker}", flush=True)
print("=" * 72, flush=True)
