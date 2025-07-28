# Fine-Tuned Text-to-SQL

This project demonstrates how to fine-tune a language model (configurable, Mistral-7B-v0.3 by default) on a consumer-grade GPU and build web and CLI applications on top of it. The goal is to showcase the complete workflow of a modern language model from fine-tuning to deployment, not to achieve state-of-the-art text-to-SQL performance.

## Features

* **QLoRA Fine-tuning:** Memory-efficient fine-tuning
* **Parameter-Efficient Training:** Using LoRA Adapters and quantization techniques
* **Application Development:** Gradio web applications, CLI tools, and Jupyter notebook interface
* **Model Evaluation:** Multi-metric assessment

![Text-to-SQL Web App](text-to-sql%20web%20app.png)

## Project Structure

```text
text-to-sql-mistral/
├── core.py                 # Core business logic 
├── app.py                  # Gradio web interface
├── chat.py                 # Interactive CLI 
├── notebook_chat.py        # Jupyter notebook interface
├── notebook_example.ipynb  # Notebook usage example
├── evaluate.py             # Evaluation suite
├── finetune.py             # QLoRA fine-tuning script
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── .gitattributes          # Git LFS configuration
```

### Core Architecture

The project uses a modular architecture with shared business logic:

* **`core.py`**
  * Model loading and authentication
  * SQL generation, validation, and execution
  * Example schemas and questions
  * SQL formatting utilities
* **Interface files**
  * `app.py` - Web interface using Gradio
  * `chat.py` - Command-line interface
  * `notebook_chat.py` - Jupyter notebook interface

## Technical Details

* **Base Model:** `mistralai/Mistral-7B-v0.3` (configurable)
* **Quantization:** 4-bit (default) or 8-bit (better accuracy)
* **Adapter Rank:** 16 (4-bit) or 32 (8-bit)  (configurable)
* **Target Modules:** Query/Value (4-bit) or Query/Key/Value/Output (8-bit)
* **Training Data:** `b-mc2/sql-create-context` dataset
* **Prompt Format:** Context → Question → SQL structure

## Setup

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Authenticate with Hugging Face:**

   ```bash
   export HUGGINGFACE_HUB_TOKEN="your_token_here"
   # OR
   huggingface-cli login
   # OR pass --hf_token "your_token" to any script
   ```

## Workflow

## Quick Usage

```bash
# Training (4-bit quantization - default, ~7GB VRAM)
python finetune.py --hf_token "your_token"

# Training (8-bit quantization - better accuracy, ~14GB VRAM)
python finetune.py --hf_token "your_token" --use_8bit

# Launch interfaces
python app.py  --hf_token "your_token" --share         # Web app
python chat.py  --hf_token "your_token"                # CLI interface  
python evaluate.py --hf_token "your_token" --skip_base # Evaluation metrics
```

**For Jupyter Notebooks (Kaggle, Colab, JupyterLab):**

```python
from notebook_chat import start_chat
chat = start_chat(hf_token="your_token")
```

See [notebook_example.ipynb](notebook_example.ipynb).

## Detailed Usage

For comprehensive documentation of all command-line options and examples, see [detailed_usage.md](detailed_usage.md).
