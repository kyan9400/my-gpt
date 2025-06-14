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
print(f"✅ Vocab size: {vocab_size}")

# === Load model ===
model = GPT(vocab_size, block_size).to(device)
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()

# === Sampling presets ===
presets = {
    "🎯 Precise": {"temperature": 0.7, "top_k": 20, "top_p": 1.0},
    "⚖️ Balanced": {"temperature": 1.0, "top_k": 40, "top_p": 0.95},
    "🎨 Creative": {"temperature": 1.2, "top_k": 0, "top_p": 0.9},
}

# === Sampling functions ===
def top_k_sampling(logits, k):
    k = min(k, logits.size(-1))
    v, i = torch.topk(logits, k)
    probs = torch.softmax(v, dim=-1)
    idx = torch.multinomial(probs, num_samples=1)
    return i.gather(-1, idx)

def top_p_sampling(logits, p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    cutoff = cumulative_probs > p
    cutoff[..., 1:] = cutoff[..., :-1].clone()
    cutoff[..., 0] = False
    sorted_logits[cutoff] = float('-inf')
    probs = torch.softmax(sorted_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return sorted_indices.gather(-1, next_token)

# === Token/word/char count ===
def get_prompt_stats(prompt):
    tokens = encode(prompt.strip().lower(), stoi)
    word_count = len(prompt.strip().split())
    char_count = len(prompt)
    return f"🧮 Tokens: {len(tokens)} | ✍️ Words: {word_count} | 🔤 Chars: {char_count}"

# === Text generation ===
def generate(prompt, max_new_tokens, temperature, top_k, top_p, multiple):
    prompt = prompt.strip().lower()
    if not prompt:
        prompt = "the"
    encoded_prompt = encode(prompt, stoi)
    if not encoded_prompt:
        encoded_prompt = encode("the", stoi)

    results = []
    num_outputs = 3 if multiple else 1

    for _ in range(num_outputs):
        input_ids = torch.tensor([encoded_prompt], dtype=torch.long).to(device)
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k > 0:
                next_id = top_k_sampling(logits, top_k)
            elif top_p < 1.0:
                next_id = top_p_sampling(logits, top_p)
            else:
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat((input_ids, next_id), dim=1)

        results.append(decode(input_ids[0].tolist(), itos))

    return "\n\n---\n\n".join(results)

# === Save Output to file ===
def save_output(text):
    filepath = "generated_output.txt"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    return filepath

# === Apply preset ===
def apply_preset(preset_label):
    preset = presets[preset_label]
    return preset["temperature"], preset["top_k"], preset["top_p"]

# === Clear All function ===
def clear_all():
    return (
        "", "", "⚖️ Balanced", 100, 1.0, 40, 0.95, False,
        "🧮 Tokens: 0 | ✍️ Words: 0 | 🔤 Chars: 0", None
    )

# === Gradio UI ===
with gr.Blocks(title="Mini-GPT Text Generator", theme=gr.themes.Base()) as interface:
    # Custom CSS for dark mode toggle support
    gr.HTML("""
    <style>
        html.dark body {
            background-color: #1e1e1e;
            color: #e0e0e0;
        }
        html.dark input, html.dark textarea, html.dark select {
            background-color: #2e2e2e;
            color: #fff;
            border: 1px solid #555;
        }
        html.dark .gr-button {
            background-color: #444 !important;
            color: #fff !important;
        }
    </style>
    """)

    gr.Markdown("## 🤖 Mini-GPT Text Generator with Advanced UI")

    with gr.Row():
        with gr.Column():
            preset_choice = gr.Dropdown(
                choices=list(presets.keys()), value="⚖️ Balanced", label="Sampling Preset"
            )
            prompt = gr.Textbox(lines=2, label="Prompt", placeholder="Enter your prompt here...")
            stats = gr.Markdown("🧮 Tokens: 0 | ✍️ Words: 0 | 🔤 Chars: 0")
            prompt.change(fn=get_prompt_stats, inputs=prompt, outputs=stats)

            max_tokens = gr.Slider(10, 200, value=100, step=1, label="Max New Tokens")
            temperature = gr.Slider(0.5, 1.5, value=1.0, step=0.1, label="Temperature")
            top_k = gr.Slider(0, 100, value=40, step=1, label="Top-k (0 = disabled)")
            top_p = gr.Slider(0.5, 1.0, value=0.95, step=0.01, label="Top-p (1.0 = disabled)")
            multiple_outputs = gr.Checkbox(label="Generate 3 outputs")
            submit_btn = gr.Button("▶️ Submit", variant="primary")
            clear_btn = gr.Button("🗑️ Clear All")
            theme_toggle = gr.Button("🌓 Toggle Theme")

        with gr.Column():
            output = gr.Textbox(label="Output", lines=15)
            copy_btn = gr.Button("📋 Copy to Clipboard")
            save_btn = gr.Button("💾 Save Output")
            download_link = gr.File(label="Download Output")

    # Events
    preset_choice.change(fn=apply_preset, inputs=preset_choice, outputs=[temperature, top_k, top_p])
    submit_btn.click(fn=generate, inputs=[prompt, max_tokens, temperature, top_k, top_p, multiple_outputs], outputs=output)
    clear_btn.click(fn=clear_all, inputs=[], outputs=[
        prompt, output, preset_choice, max_tokens, temperature, top_k, top_p,
        multiple_outputs, stats, download_link
    ])
    copy_btn.click(
        None,
        inputs=[],
        outputs=[],
        js="() => { navigator.clipboard.writeText(document.querySelector('textarea[aria-label=Output]').value); }"
    )
    save_btn.click(fn=save_output, inputs=output, outputs=download_link)
    theme_toggle.click(
        None,
        inputs=[],
        outputs=[],
        js="""
        () => {
            const html = document.querySelector('html');
            html.classList.toggle('dark');
        }
        """
    )

interface.launch()
