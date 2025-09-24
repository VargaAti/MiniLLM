# mini_llm.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# Beállítások
vocab = list("abcdefghijklmnopqrstuvwxyz ")
vocab_size = len(vocab)
char_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_char = {i: ch for ch, i in char_to_idx.items()}
device = "cuda" if torch.cuda.is_available() else "cpu"

# Segédfüggvények
def encode(text):
    return [char_to_idx[c] for c in text.lower() if c in char_to_idx]

def decode(indices):
    return ''.join([idx_to_char[i] for i in indices])

# Kis tanuló adat
train_data = "hello hello hello world world world gpt gpt gpt ai ai ai".lower()

# Modell paraméterek
block_size = 8
embedding_dim = 32
hidden_dim = 64

# Egyszerű Transformer blokk (csak self-attention + FFN)
class MiniTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.attn = nn.MultiheadAttention(embedding_dim, num_heads=2, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )

    def forward(self, x):
        B, T = x.size()
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = token_emb + pos_emb
        attn_out, _ = self.attn(x, x, x)
        logits = self.ff(attn_out)
        return logits

# Modell létrehozása
model = MiniTransformer().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Tréning ciklus
def get_batch():
    x = []
    y = []
    for _ in range(32):  # batch size
        start = random.randint(0, len(train_data) - block_size - 1)
        chunk = train_data[start:start+block_size+1]
        x.append(encode(chunk[:-1]))
        y.append(encode(chunk[1:]))
    return torch.tensor(x, dtype=torch.long).to(device), torch.tensor(y, dtype=torch.long).to(device)

print("Tréning indul...")
for step in range(1000):
    xb, yb = get_batch()
    logits = model(xb)
    loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

# Generálás
def generate(start_text, length=100):
    model.eval()
    context = torch.tensor([encode(start_text)], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(length):
            if context.size(1) > block_size:
                context = context[:, -block_size:]
            logits = model(context)
            next_token_logits = logits[0, -1]
            probs = F.softmax(next_token_logits, dim=0)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat([context, next_token.unsqueeze(0)], dim=1)

        return decode(context[0].tolist())

print("\n📤 Generált szöveg:")
print(generate("hel"))
