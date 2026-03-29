import torch
import torch.nn as nn

class LMWrapper(nn.Module):
    def __init__(self, base_model, vocab_size, d_model=128):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, d_model)
        self.model = base_model
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embed(x)  # (B, T, D)

        out = self.model(x)

        # 🔥 Handle different model outputs
        if isinstance(out, tuple):
            y = out[0]   # hybrid model
        else:
            y = out      # baseline models

        logits = self.head(y)  # (B, T, vocab)
        return logits