# generate.py

import torch
from model import GPT
from tokenizer import build_vocab, encode, decode

block_size = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === Load text and vocab ===
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

stoi, itos = build_vocab(text)
vocab_size = len(stoi)

# === Load model ===
model = GPT(vocab_size, block_size).to(device)
model.load_state_dict(torch.load("model.pt"))
model.eval()

# === Generation function ===
def generate(model, idx, max_new_tokens, temperature=1.0):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)
    return idx

# === Prompt and generation ===
start_prompt = "to be"
start_ids = torch.tensor([encode(start_prompt.lower(), stoi)], dtype=torch.long).to(device)

out = generate(model, start_ids, max_new_tokens=100, temperature=1.0)
print(decode(out[0].tolist(), itos))