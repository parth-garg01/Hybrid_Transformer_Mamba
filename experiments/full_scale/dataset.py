"""
WikiText-103 dataset with GPT-2 BPE tokenizer.
Returns (input_ids, target_ids) batches for language modeling.
"""
import torch
from datasets import load_dataset
from transformers import GPT2TokenizerFast


class WikiTextBPE:
    def __init__(self, split="train", seq_len=256, max_tokens=None):
        self.seq_len = seq_len
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        tok.add_special_tokens({"pad_token": "<|pad|>"})
        self.vocab_size = len(tok)

        raw = load_dataset("wikitext", "wikitext-103-raw-v1",
                           split=split, trust_remote_code=True)
        text = "\n".join(t for t in raw["text"] if t.strip())

        ids = tok.encode(text, add_special_tokens=False)
        if max_tokens:
            ids = ids[:max_tokens]
        self.data = torch.tensor(ids, dtype=torch.long)
        print(f"  [{split}] {len(self.data):,} tokens | vocab={self.vocab_size}")

    def get_batch(self, batch_size, device="cuda"):
        ix = torch.randint(len(self.data) - self.seq_len - 1, (batch_size,))
        x = torch.stack([self.data[i:i + self.seq_len]     for i in ix])
        y = torch.stack([self.data[i + 1:i + self.seq_len + 1] for i in ix])
        return x.to(device), y.to(device)
