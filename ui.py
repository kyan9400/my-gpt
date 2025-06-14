# ui.py

import torch
import gradio as gr
from model import GPT
from tokenizer import build_vocab, encode, decode

# === Config ===
block_size = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === Load data and vocab ===
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

stoi, itos = build_vocab(text)
vocab_size = len(stoi)

# === Load model ===
model = GPT(vocab_size, block_size).to(device)
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()

# === Generation function ===
def generate(prompt, max_new_tokens, temperature):
    model.eval()
    prompt = prompt.strip().lower()

    if not prompt:
        prompt = "the"

    encoded_prompt = encode(prompt, stoi)
    if not encoded_prompt:
        encoded_prompt = encode("the", stoi)

    input_ids = torch.tensor([encoded_prompt], dtype=torch.long).to(device)

    for _ in range(max_new_tokens):
        idx_cond = input_ids[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat((input_ids, next_id), dim=1)

    return decode(input_ids[0].tolist(), itos)

# === Gradio UI with Copy Button ===
with gr.Blocks(title="Mini-GPT Text Generator") as interface:
    gr.Markdown("### Type a prompt and generate text using your own GPT model.")

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(lines=2, label="Prompt", placeholder="Enter your prompt here...")
            max_tokens = gr.Slider(10, 200, value=100, step=1, label="Max New Tokens")
            temperature = gr.Slider(0.5, 1.5, value=1.0, step=0.1, label="Temperature")
            submit_btn = gr.Button("Submit", variant="primary")
        with gr.Column():
            output = gr.Textbox(label="Output", lines=10)
            copy_btn = gr.Button("📋 Copy to Clipboard")

    submit_btn.click(fn=generate, inputs=[prompt, max_tokens, temperature], outputs=output)

    copy_btn.click(
        None,
        inputs=[],
        outputs=[],
        js="() => { navigator.clipboard.writeText(document.querySelector('textarea[aria-label=Output]').value); }"
    )

interface.launch()