# 🧠 Mini GPT Text Generator

A tiny GPT-style character-level language model built with PyTorch and Gradio for interactive text generation.

## 🚀 Features

- Transformer-based GPT model (miniature)
- Character-level training on any `.txt` file
- Gradio UI with:
  - Prompt input and live output
  - Sampling presets (Precise / Balanced / Creative)
  - Token, word, and character counters
  - Top-k and top-p sampling
  - Generate 3 variations at once
  - 📋 Copy to clipboard
  - 💾 Save output to file
  - 🗑️ Clear All button

## 📂 Project Structure

```
my-gpt/
├── model.py           # GPT model definition
├── tokenizer.py       # Vocabulary and encoding logic
├── train.py           # Training script
├── generate.py        # Simple text generator CLI
├── ui.py              # Gradio web interface
├── input.txt          # Training text
├── model.pt           # Trained model weights
└── README.md
```

## 📦 Installation

1. Clone the repo:

```bash
git clone https://github.com/kyan9400/my-gpt.git
cd my-gpt
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, create one with:

```bash
pip freeze > requirements.txt
```

## 🏋️ Train the Model

Make sure you have an `input.txt` file in your folder with your training text.

```bash
python train.py
```

This trains the GPT model and saves it as `model.pt`.

## 💬 Run the Web UI

```bash
python ui.py
```

Then open in browser: [http://127.0.0.1:7860](http://127.0.0.1:7860)

## ✍️ Example Prompts

- `to be or not to be`
- `in the beginning`
- `the quick brown`

## 📄 Save Output

Use the "💾 Save Output" button to download generated text as a `.txt` file.

## 🧪 Try Sampling Presets

| Preset     | Description          |
|------------|----------------------|
| 🎯 Precise  | Lower temp + top-k   |
| ⚖️ Balanced | Medium temp + top-k/p |
| 🎨 Creative | Higher temp + top-p  |

## 📘 License

MIT License. Do whatever you want, just don't train it to overthrow humanity 🤖
