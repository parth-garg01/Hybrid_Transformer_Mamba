"""
Full-split evaluation (PRD P1.5, fixes B6).

Deterministic sliding window over the ENTIRE token stream. Target token j
(predicted from tokens < j) for j = 1 .. N-1 is scored exactly once:
    - windows start at 0, stride, 2*stride, ...; every window has the full
      ``seq_len`` inputs (the last one is right-aligned to the end of the data);
    - the first window scores all its targets, every later window scores only
      the targets not scored by an earlier window (normally the last
      ``stride`` tokens, so each scored token sees >= seq_len - stride context).
"""
import math
import torch
import torch.nn.functional as F


def sliding_windows(n_tokens, seq_len, stride):
    """List of (start, length, score_from) for a stream of ``n_tokens`` tokens.

    The window feeds data[start:start+length] and predicts
    data[start+1:start+length+1]; offsets >= score_from within it are scored.
    """
    if stride <= 0 or stride > seq_len:
        raise ValueError("need 0 < stride <= seq_len")
    n_targets = n_tokens - 1
    if n_targets <= 0:
        return []
    if n_targets <= seq_len:
        return [(0, n_targets, 0)]
    starts = list(range(0, n_targets - seq_len + 1, stride))
    if starts[-1] + seq_len < n_targets:
        starts.append(n_targets - seq_len)
    out, scored_end = [], 0          # targets with index < scored_end+1 are done
    for s in starts:
        end = s + seq_len            # this window covers targets s+1 .. end
        out.append((s, seq_len, scored_end - s))
        scored_end = end
    return out


@torch.no_grad()
def evaluate_full(model, data_tensor, seq_len=256, stride=128, batch_size=8,
                  device="cpu"):
    """Token-weighted mean CE and perplexity over the whole ``data_tensor``."""
    was_training = model.training
    model.eval()
    data = data_tensor.cpu()
    windows = sliding_windows(len(data), seq_len, stride)
    total, count = 0.0, 0
    try:
        for i in range(0, len(windows), batch_size):
            chunk = windows[i:i + batch_size]
            L = chunk[0][1]
            x = torch.stack([data[s:s + L] for s, _, _ in chunk]).to(device)
            y = torch.stack([data[s + 1:s + L + 1] for s, _, _ in chunk]).to(device)
            mask = torch.zeros(len(chunk), L, dtype=torch.bool)
            for r, (_, _, f) in enumerate(chunk):
                mask[r, f:] = True
            mask = mask.to(device)
            logits = model(x)
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                 y.reshape(-1), reduction="none").view(len(chunk), L)
            total += float(ce[mask].cpu().double().sum().item())
            count += int(mask.sum().item())
    finally:
        model.train(was_training)
    loss = total / max(count, 1)
    return {"loss": loss, "ppl": math.exp(min(loss, 50.0)), "n_tokens": count}
