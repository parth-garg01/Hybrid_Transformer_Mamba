import torch

class SimpleTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, tokens):
        return ''.join([self.itos[t] for t in tokens])


class TextDataset:
    def __init__(self, text, seq_len=32):
        self.tokenizer = SimpleTokenizer(text)
        self.data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        self.seq_len = seq_len

    def get_batch(self, batch_size=16):
        ix = torch.randint(0, len(self.data) - self.seq_len - 1, (batch_size,))
        x = torch.stack([self.data[i:i+self.seq_len] for i in ix])
        y = torch.stack([self.data[i+1:i+self.seq_len+1] for i in ix])
        return x, y