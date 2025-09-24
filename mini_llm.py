import torch
import torch.nn as nn
import torch.nn.functional as F
import random

vocab = list("abcdefghijklmnopqrstuvwxyz !?.:,") + ["<|user|>", "<|bot|>"]  # Plusz speciális tokenek
vocab_size = len(vocab)
char_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_char = {i: ch for ch, i in char_to_idx.items()}
device = "cuda" if torch.cuda.is_available() else "cpu"

def encode(text):
    # Egyszerű karakter szintű kódolás, speciális tokenek helyett szóközöket is kezeljük
    # Pl. user és bot tokeneket külön kezelhetjük a beszélgetésnél
    tokens = []
    words = text.split()
    for w in words:
        if w == "<|user|>" or w == "<|bot|>":
            tokens.append(char_to_idx[w])
        else:
            tokens.extend([char_to_idx[c] for c in w if c in char_to_idx])
            tokens.append(char_to_idx[' '])  # szóköz a szavak között
    return tokens

def decode(indices):
    return ''.join([idx_to_char[i] for i in indices]).replace('  ', ' ')

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.ln1(x + attn_out)
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        return x

class MiniChatTransformer(nn.Module):
    def __init__(self, vocab_size, block_size=64, embedding_dim=64, num_layers=4, num_heads=4, hidden_dim=256):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.layers = nn.ModuleList([TransformerBlock(embedding_dim, num_heads, hidden_dim) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)
        self.block_size = block_size

    def forward(self, x):
        B, T = x.size()
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(torch.arange(T, device=x.device))
        x = token_emb + pos_emb
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# Használat:

model = MiniChatTransformer(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Egy egyszerű beszélgetési adat példa
train_data = [
    "<|user|> hello <|bot|> hi there! how can i help you? ",
    "<|user|> what is your name? <|bot|> i am a mini transformer bot. ",
    "<|user|> tell me a joke <|bot|> why did the chicken cross the road? to get to the other side! ",
    "<|user|> how are you? <|bot|> i am just a bunch of code, but thanks for asking! ",
    "<|user|> what can you do? <|bot|> i can chat with you and tell jokes. ",
    "<|user|> goodbye <|bot|> bye! have a nice day! ",
    "<|user|> what is ai? <|bot|> ai stands for artificial intelligence. it means machines that can learn. ",
    "<|user|> can you help me with math? <|bot|> sure! ask me a math question. ",
    "<|user|> what is 2 plus 2? <|bot|> 2 plus 2 is 4. ",
    "<|user|> thank you <|bot|> you are welcome! ",
]


def get_batch(batch_size=16):
    x, y = [], []
    for _ in range(batch_size):
        text = random.choice(train_data)
        if len(text) < 10:
            continue
        start = random.randint(0, max(0, len(text) - 65))
        chunk = text[start:start+65]
        enc = encode(chunk)
        if len(enc) < 2:
            continue
        x.append(enc[:-1])
        y.append(enc[1:])
    x = [torch.tensor(seq, dtype=torch.long) for seq in x]
    y = [torch.tensor(seq, dtype=torch.long) for seq in y]
    x = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0)
    y = nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=0)
    return x.to(device), y.to(device)


# Tréning
for step in range(1000):
    xb, yb = get_batch()
    logits = model(xb)
    logits = logits.view(-1, vocab_size)
    yb = yb.view(-1)
    loss = F.cross_entropy(logits, yb, ignore_index=0)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(f"Step {step} Loss: {loss.item():.4f}")

# Generálás egyszerűen:
def generate(prompt, length=100):
    model.eval()
    context = torch.tensor([encode(prompt)], dtype=torch.long).to(device)
    with torch.no_grad():
        for _ in range(length):
            if context.size(1) > model.block_size:
                context = context[:, -model.block_size:]
            logits = model(context)
            next_token_logits = logits[0, -1]
            probs = F.softmax(next_token_logits, dim=0)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat([context, next_token.unsqueeze(0)], dim=1)
    return decode(context[0].tolist())

print(generate("<|user|> hello bot <|bot|>"))
