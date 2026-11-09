"""
WikiText-103 with the GPT-2 BPE tokenizer.

Tokenization is deterministic (fixed dataset config, fixed split order, fixed
chunking) and cached to disk as a 1-D LongTensor, so every run and every
session uses byte-identical tokens. The training script hashes the tensor with
``provenance.dataset_sha256``.

Cache: ``$ABR_CACHE_DIR`` or ``~/.cache/abr``, file
``wikitext103_<split>_gpt2.pt``. The cache always holds the FULL split;
``max_tokens`` slices its prefix, which equals what the old chunked loader
produced when it stopped early.
"""
import os
import torch

SPLITS = ("train", "validation", "test")
_CHUNK = 5_000   # articles per tokenizer call (keeps peak RAM low)


def cache_dir():
    return os.environ.get("ABR_CACHE_DIR",
                          os.path.join(os.path.expanduser("~"), ".cache", "abr"))


def cache_path(split):
    return os.path.join(cache_dir(), f"wikitext103_{split}_gpt2.pt")


def _tokenizer():
    from transformers import GPT2TokenizerFast
    local = os.path.join(os.path.expanduser("~"), "gpt2_tokenizer")
    path = local if os.path.isdir(local) else "gpt2"
    tok = GPT2TokenizerFast.from_pretrained(path, local_files_only=os.path.isdir(local))
    tok.add_special_tokens({"pad_token": "<|pad|>"})
    return tok


def _tokenize_split(split):
    from datasets import load_dataset
    tok = _tokenizer()
    raw = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    texts = [t for t in raw["text"] if t.strip()]
    ids = []
    for i in range(0, len(texts), _CHUNK):
        ids.extend(tok.encode("\n".join(texts[i:i + _CHUNK]), add_special_tokens=False))
    return torch.tensor(ids, dtype=torch.long), len(tok)


def load_tokens(split, max_tokens=None):
    """Return (tokens: LongTensor[N], vocab_size). Builds the cache if missing."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    path = cache_path(split)
    if os.path.exists(path):
        blob = torch.load(path, map_location="cpu")
    else:
        tokens, vocab = _tokenize_split(split)
        blob = {"tokens": tokens, "vocab_size": vocab, "split": split,
                "dataset": "wikitext-103-raw-v1", "tokenizer": "gpt2+pad"}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        torch.save(blob, tmp)
        os.replace(tmp, path)
    tokens = blob["tokens"]
    if max_tokens:
        tokens = tokens[:max_tokens]
    return tokens.contiguous(), int(blob["vocab_size"])


def get_batch(data, batch_size, seq_len, device="cpu", generator=None):
    """Random (x, y) windows from a 1-D token tensor.

    Pass a seeded CPU ``torch.Generator`` for reproducible sampling; with
    ``generator=None`` the global torch RNG is used (legacy behaviour).
    """
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + seq_len] for i in ix])
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix])
    return x.to(device), y.to(device)


def batch_at(data, ix, seq_len, device="cpu"):
    """(x, y) windows at fixed start indices (used for the fixed val estimate)."""
    x = torch.stack([data[i:i + seq_len] for i in ix])
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix])
    return x.to(device), y.to(device)


class WikiTextBPE:
    """Backward-compatible wrapper used by the older scripts."""

    def __init__(self, split="train", seq_len=256, max_tokens=None):
        self.seq_len = seq_len
        self.split = split
        self.data, self.vocab_size = load_tokens(split, max_tokens)
        print(f"  [{split}] {len(self.data):,} tokens | vocab={self.vocab_size}")

    def get_batch(self, batch_size, device="cuda", generator=None):
        return get_batch(self.data, batch_size, self.seq_len, device, generator)
