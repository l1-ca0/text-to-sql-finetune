# Fine-Tuned Text-to-SQL 

This project demonstrates how to fine-tune a language model (configurable, Mistral-7B-v0.3 by default) on a consumer-grade GPU and build web and CLI applications on top of it. The goal is to showcase the complete workflow of a modern language model from fine-tuning to deployment, not to achieve state-of-the-art text-to-SQL performance.

## Features

* **QLoRA Fine-tuning:** Memory-efficient fine-tuning
* **Parameter-Efficient Training:** Using LoRA Adapters and quantization techniques
* **Application Development:** Gradio web applications and CLI tools
* **Model Evaluation:** Multi-metric assessment

## Project Structure

```text
text-to-sql-mistral/
├── app.py              # Gradio web interface
├── chat.py             # Interactive CLI 
├── evaluate.py         # Evaluation suite
├── finetune.py         # QLoRA fine-tuning script
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── .gitattributes      # Git LFS configuration
```

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
python finetune.py  # Train the model (~3 hours on Nvidia Tesla P100 GPU)
python app.py       # Launch web app
python chat.py      # CLI interface  
python evaluate.py  # Evaluation metrics
```

## Detailed Usage

### Fine-Tuning (`finetune.py`)

Train a QLoRA adapter for text-to-SQL generation:

```bash
python finetune.py [OPTIONS]
```

**Options:**

* `--model_name` - Base model to fine-tune (default: `mistralai/Mistral-7B-v0.3`)
* `--dataset_name` - Training dataset (default: `b-mc2/sql-create-context`)
* `--output_dir` - Directory to save adapter (default: `./sql-mistral-adapter-final`)
* `--max_steps` - Number of training steps (default: `1500`)
* `--push_to_hub` - Push adapter to Hugging Face Hub
* `--hub_model_id` - Hub repository ID (required if `--push_to_hub`)
* `--hf_token` - Hugging Face authentication token

**Examples:**

```bash
# Custom configuration
python finetune.py --max_steps 2000 --output_dir "./my-adapter"

# Train and push to Hub
python finetune.py --push_to_hub --hub_model_id "username/my-sql-adapter"
```

### Web Interface (`app.py`)

Launch an interactive Gradio web application:

```bash
python app.py [OPTIONS]
```

**Options:**

* `--adapter_path` - Path to fine-tuned adapter (default: `./sql-mistral-adapter-final`)
* `--base_model` - Base model name (default: `mistralai/Mistral-7B-v0.3`)
* `--port` - Server port (default: `7860`)
* `--share` - Create public shareable link
* `--hf_token` - Hugging Face authentication token

**Examples:**

```bash
# Custom port and public sharing
python app.py --port 8080 --share

# Use different adapter
python app.py --adapter_path "./my-custom-adapter"
```

### CLI Chat (`chat.py`)

Interactive command-line interface with session management:

```bash
python chat.py [OPTIONS]
```

**Options:**

* `--adapter_path` - Path to fine-tuned adapter (default: `./sql-mistral-adapter-final`)
* `--base_model_name` - Base model name (default: `mistralai/Mistral-7B-v0.3`)
* `--temperature` - Generation creativity (default: `0.1`, range: `0.0-1.0`)
* `--load_session` - Load previous chat session from file
* `--hf_token` - Hugging Face authentication token

**Examples:**

```bash
# More creative responses
python chat.py --temperature 0.3

# Load previous session
python chat.py --load_session "my_session.json"
```

**Available Commands:**

* `help` - Show all commands
* `examples` - Display sample schemas
* `save <file>` - Save current session
* `load <file>` - Load session
* `validate` - Check SQL syntax
* `format` - Format SQL output
* `clear` - Clear current schema
* `history` - Show conversation history

### Evaluation (`evaluate.py`)

Comprehensive model evaluation with multiple metrics:

```bash
python evaluate.py [OPTIONS]
```

**Options:**

* `--adapter_path` - Path to fine-tuned adapter (default: `./sql-mistral-adapter-final`)
* `--base_model_name` - Base model name (default: `mistralai/Mistral-7B-v0.3`)
* `--sample_size` - Number of samples to evaluate (default: all)
* `--output_report` - Save detailed JSON report to file
* `--skip_base` - Skip base model evaluation (faster)
* `--hf_token` - Hugging Face authentication token

**Examples:**

```bash
# Quick evaluation (fine-tuned model only)
python evaluate.py --skip_base

# Sample evaluation with report
python evaluate.py --sample_size 100 --output_report "results.json"
```

**Metrics Provided:**

* Exact match accuracy
* Normalized match accuracy  
* SQL syntax validity rate
* Token overlap scores
* Component analysis (JOIN, WHERE, etc.)
* Error categorization

## Technical Details

* **Base Model:** `mistralai/Mistral-7B-v0.3` (configurable)
* **Adapter Rank:** 16 (configurable)
* **Target Modules:** Query and Value projection layers
* **Training Data:** `b-mc2/sql-create-context` dataset
* **Prompt Format:** Context → Question → SQL structure
