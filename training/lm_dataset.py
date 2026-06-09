from datasets import load_dataset
import torch


class WikiTextDataset:
    """
    Character-level dataset loader supporting WikiText-2 and WikiText-103.

    max_chars caps the number of characters used (useful for WikiText-103
    on CPU where loading the full 103M-char corpus is slow). Set to None
    to use the complete split.
    """

    def __init__(self, split="train", seq_len=64, stoi=None, itos=None,
                 dataset_name="wikitext-103-raw-v1", max_chars=10_000_000):

        dataset = load_dataset("wikitext", dataset_name)
        text    = " ".join(dataset[split]["text"])

        if max_chars is not None and len(text) > max_chars:
            text = text[:max_chars]

        if stoi is None:
            chars     = sorted(set(text))
            self.stoi = {ch: i for i, ch in enumerate(chars)}
            self.itos = {i: ch for ch, i in self.stoi.items()}
        else:
            self.stoi = stoi
            self.itos = itos

        self.vocab_size = len(self.stoi)
        self.data       = torch.tensor(
            [self.stoi.get(ch, 0) for ch in text],
            dtype=torch.long,
        )
        self.seq_len = seq_len

        print(f"  [{split}] {len(text):,} chars | vocab={self.vocab_size} | "
              f"dataset={dataset_name}", flush=True)

    def get_batch(self, batch_size=16):
        ix = torch.randint(0, len(self.data) - self.seq_len - 1, (batch_size,))
        x  = torch.stack([self.data[i : i + self.seq_len]     for i in ix])
        y  = torch.stack([self.data[i + 1 : i + self.seq_len + 1] for i in ix])
        return x, y
