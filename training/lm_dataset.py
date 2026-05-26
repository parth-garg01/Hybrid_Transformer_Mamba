from datasets import load_dataset
import torch


class WikiTextDataset:
    def __init__(self, split="train", seq_len=64, stoi=None, itos=None):
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        text = " ".join(dataset[split]["text"])

        # 🔥 THIS IS WHERE YOUR CODE GOES
        if stoi is None:
            chars = sorted(list(set(text)))
            self.stoi = {ch: i for i, ch in enumerate(chars)}
            self.itos = {i: ch for ch, i in self.stoi.items()}
        else:
            self.stoi = stoi
            self.itos = itos

        self.vocab_size = len(self.stoi)

        self.data = torch.tensor(
            [self.stoi.get(ch, 0) for ch in text],  # OOV handled
            dtype=torch.long
        )

        self.seq_len = seq_len

    def get_batch(self, batch_size=16):
        ix = torch.randint(0, len(self.data) - self.seq_len - 1, (batch_size,))
        x = torch.stack([self.data[i:i+self.seq_len] for i in ix])
        y = torch.stack([self.data[i+1:i+self.seq_len+1] for i in ix])
        return x, y