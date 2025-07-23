# Detailed Usage Guide

This guide provides comprehensive documentation for all scripts and their command-line options.

## Fine-Tuning (`finetune.py`)

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

## Web Interface (`app.py`)

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

## CLI Chat (`chat.py`)

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

## Evaluation (`evaluate.py`)

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
