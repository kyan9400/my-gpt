# tokenizer.py

def build_vocab(text):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

def encode(text, stoi):
    return [stoi[c] for c in text if c in stoi]  # Skip unknown chars

def decode(indices, itos):
    return ''.join([itos.get(i, '?') for i in indices])  # '?' fallback
