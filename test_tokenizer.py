from tokenizer import build_vocab, encode, decode

# Step 1: Load and lowercase the text
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()  # all lowercased

# Step 2: Build vocab from that same text
stoi, itos = build_vocab(text)

# Step 3: Try a lowercase sample from that same vocab
sample = "to be"
encoded = encode(sample, stoi)
decoded = decode(encoded, itos)

print("✅ Vocab size:", len(stoi))
print("📝 Sample:", sample)
print("📦 Encoded:", encoded)
print("🔄 Decoded:", decoded)
