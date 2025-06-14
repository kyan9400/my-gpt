# train.py

import torch
from tokenizer import build_vocab, encode
from model import GPT

# === Hyperparameters ===
block_size = 8
batch_size = 32
embed_size = 64
n_heads = 4
n_layers = 2
train_ratio = 0.9
max_iters = 1000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === Load and preprocess data ===
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

stoi, itos = build_vocab(text)
vocab_size = len(stoi)
data = torch.tensor(encode(text, stoi), dtype=torch.long)
n = int(len(data) * train_ratio)
train_data = data[:n]
val_data = data[n:]

# === Data loader ===
def get_batch(split):
    source = train_data if split == 'train' else val_data
    ix = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i:i+block_size] for i in ix])
    y = torch.stack([source[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# === Instantiate model ===
model = GPT(vocab_size, block_size, embed_size, n_layers, n_heads).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# === Training loop ===
for step in range(max_iters):
    model.train()
    x, y = get_batch('train')
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % eval_interval == 0 or step == max_iters - 1:
        model.eval()
        with torch.no_grad():
            val_x, val_y = get_batch('val')
            _, val_loss = model(val_x, val_y)
        print(f"[{step:04d}] train loss: {loss.item():.4f}, val loss: {val_loss.item():.4f}")

# === Save the trained model ===
torch.save(model.state_dict(), "model.pt")
print("✅ Model saved to model.pt")
