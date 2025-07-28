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
* `--use_8bit` - Use 8-bit quantization for better accuracy (requires ~14GB VRAM)
* `--lora_rank` - LoRA rank override (default: 16 for 4-bit, 32 for 8-bit)
* `--push_to_hub` - Push adapter to Hugging Face Hub
* `--hub_model_id` - Hub repository ID (required if `--push_to_hub`)
* `--hf_token` - Hugging Face authentication token

**Examples:**

```bash
# Default 4-bit training (memory efficient)
python finetune.py --hf_token "your_token"

# 8-bit training for better accuracy
python finetune.py --hf_token "your_token" --use_8bit

# Custom configuration with 8-bit
python finetune.py --use_8bit --max_steps 2000 --lora_rank 64

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
* `--use_8bit` - Use 8-bit quantization for inference (requires more VRAM)
* `--hf_token` - Hugging Face authentication token

**Examples:**

```bash
# Default 4-bit inference
python app.py --hf_token "your_token" --share

# 8-bit inference for better accuracy
python app.py --hf_token "your_token" --use_8bit --port 8080

# Use different adapter
python app.py --adapter_path "./my-custom-adapter" --use_8bit
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
* `--use_8bit` - Use 8-bit quantization for inference (requires more VRAM)
* `--hf_token` - Hugging Face authentication token

**Examples:**

```bash
# Default 4-bit inference
python chat.py --hf_token "your_token"

# 8-bit inference with higher creativity
python chat.py --hf_token "your_token" --use_8bit --temperature 0.3

# Load previous session with 8-bit
python chat.py --load_session "my_session.json" --use_8bit
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

## Jupyter Notebook Interface (`notebook_chat.py`)

Interactive widget-based interface for Jupyter environments:

```python
from notebook_chat import start_chat

# Start with default 4-bit quantization
chat = start_chat(hf_token="your_token")

# Start with 8-bit quantization for better accuracy
chat = start_chat(
    hf_token="your_token",
    use_8bit=True,
    adapter_path="./my-adapter"
)
```

**Parameters:**

* `adapter_path` - Path to fine-tuned adapter (default: `./sql-mistral-adapter-final`)
* `base_model_name` - Base model name (default: `mistralai/Mistral-7B-v0.3`)
* `hf_token` - Hugging Face authentication token
* `use_8bit` - Use 8-bit quantization for better accuracy (default: `False`)

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
* `--use_8bit` - Use 8-bit quantization for evaluation
* `--hf_token` - Hugging Face authentication token

**Examples:**

```bash
# Quick evaluation with 4-bit quantization
python evaluate.py --skip_base --hf_token "your_token"

# 8-bit evaluation with detailed report
python evaluate.py --use_8bit --sample_size 100 --output_report "results.json"

# Compare 4-bit vs 8-bit performance
python evaluate.py --output_report "4bit_results.json"
python evaluate.py --use_8bit --output_report "8bit_results.json"
```

**Metrics Provided:**

* Exact match accuracy
* Normalized match accuracy
* SQL syntax validity rate
* Token overlap scores
* Component analysis (JOIN, WHERE, etc.)
* Error categorization

## Core Module (`core.py`)

The core module provides shared functionality used by all interfaces:

**Key Functions:**

* `authenticate_huggingface(token)` - Handle HF authentication
* `load_model_and_tokenizer(model_name, adapter_path, use_8bit)` - Load model with quantization
* `generate_sql(model, tokenizer, schema, question, temperature, max_tokens)` - Generate SQL
* `validate_sql_syntax(sql, schema)` - Validate SQL syntax
* `execute_sql_query(sql, schema)` - Execute SQL against schema
* `format_sql(sql)` - Format SQL for readability
* `get_example_schemas()` - Get example database schemas
* `get_example_questions()` - Get example questions by schema type
