"""
Fine-tuning script for Text-to-SQL using QLoRA (Quantized Low-Rank Adaptation).

This script fine-tunes the Mistral-7B model to generate SQL queries from natural language
questions and database schemas. It uses QLoRA for memory-efficient training, combining
4-bit quantization with LoRA adapters to enable training on consumer GPUs.

Key Features:
- Memory efficient and parameter efficient
- Modular: Saves only adapter weights 
- Configurable: Command-line arguments for all major parameters
"""

import torch
import argparse
import os
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
from huggingface_hub import login

def authenticate_huggingface(token=None):
    """Handle Hugging Face authentication."""
    if token:
        print("Using provided token for authentication...")
        login(token=token)
        return True
    
    # Check for token in environment variables
    hf_token = os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HF_TOKEN')
    if hf_token:
        print("Using token from environment variable...")
        login(token=hf_token)
        return True
    
    # Check if already logged in
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"Already authenticated as: {user_info['name']}")
        return True
    except Exception:
        pass
    
    # Try to login interactively
    try:
        print("No token found. Please login to Hugging Face...")
        login()
        return True
    except Exception as e:
        print(f"Authentication failed: {e}")
        print("\nPlease authenticate using one of these methods:")
        print("1. Set environment variable: export HUGGINGFACE_HUB_TOKEN='your_token'")
        print("2. Run: huggingface-cli login")
        print("3. Pass token: --hf_token 'your_token'")
        return False

def main():
    """
    Main function that orchestrates the fine-tuning process.
    
    This function handles:
    1. Command-line argument parsing
    2. Hugging Face authentication
    3. Dataset loading and preprocessing
    4. Model quantization and LoRA setup
    5. Training configuration and execution
    6. Model saving and optional Hub upload
    """
    # Parse command-line arguments for flexible configuration
    parser = argparse.ArgumentParser(description="Fine-tune a model for Text-to-SQL.")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-v0.3", 
                       help="Base model name from Hugging Face Hub")
    parser.add_argument("--dataset_name", type=str, default="b-mc2/sql-create-context", 
                       help="Dataset name from Hugging Face Hub")
    parser.add_argument("--output_dir", type=str, default="./sql-mistral-adapter-final", 
                       help="Output directory for the trained adapter")
    parser.add_argument("--max_steps", type=int, default=1500, 
                       help="Number of training steps (controls training duration)")
    parser.add_argument("--push_to_hub", action='store_true', 
                       help="Push the trained adapter to Hugging Face Hub")
    parser.add_argument("--hub_model_id", type=str, default=None, 
                       help="Repository ID on the Hub (required if push_to_hub is True)")
    parser.add_argument("--hf_token", type=str, default=None,
                       help="Hugging Face token for authentication")
    parser.add_argument("--use_8bit", action='store_true',
                       help="Use 8-bit quantization")
    parser.add_argument("--lora_rank", type=int, default=None,
                       help="LoRA rank (default: 16 for 4-bit, 32 for 8-bit)")
    args = parser.parse_args()
    
    # Authenticate with Hugging Face
    if not authenticate_huggingface(args.hf_token):
        print("Authentication failed. Exiting.")
        return

    # Load the text-to-SQL dataset
    # Dataset contains: context (schema), question (natural language), answer (SQL)
    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split="train")
    print(f"Dataset loaded with {len(dataset)} training examples")

    if args.use_8bit:
        print("Using 8-bit quantization for improved accuracy...")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,                    # 8-bit quantization
            llm_int8_threshold=6.0,               # Threshold for outlier detection
            llm_int8_has_fp16_weight=False,       # Use int8 weights
        )
        default_lora_rank = 32
        default_batch_size = 2
        default_grad_accum = 2
        default_lr = 1e-4
    else:
        print("Using 4-bit quantization for memory efficiency...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,                    # 4-bit quantization
            bnb_4bit_quant_type="nf4",            # Uses NormalFloat4 quantization
            bnb_4bit_compute_dtype=torch.float16, # Computation still in float16
            bnb_4bit_use_double_quant=True        # Further compression
        )
        default_lora_rank = 16
        default_batch_size = 1
        default_grad_accum = 4
        default_lr = 2e-4


    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  
    
    # Ensure tokenizer has required attributes for SFTTrainer
    if not hasattr(tokenizer, 'model_max_length') or tokenizer.model_max_length > 1024:
        tokenizer.model_max_length = 1024

    # LoRA configuration - adapt based on quantization and user preference
    lora_rank = args.lora_rank if args.lora_rank is not None else default_lora_rank
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"] if args.use_8bit else ["q_proj", "v_proj"]
    
    print(f"LoRA configuration: rank={lora_rank}, target_modules={target_modules}")
    lora_config = LoraConfig(
        r=lora_rank,                          # Rank of adaptation matrices
        lora_alpha=lora_rank * 2,             # Scaling factor (alpha/r = 2.0)
        lora_dropout=0.05,                    # Dropout for regularization
        target_modules=target_modules         # Adapt more modules for 8-bit
    )

    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
        
    '''
    Converts the dataset into a structured prompt format:
    CONTEXT: Database schema (CREATE TABLE statements)
    QUESTION: Natural language query
    SQL: Target SQL query to generate
    '''
    def formatting_prompts_func(example):
        # Format each example into the required text format
        text = f"### CONTEXT\n{example['context']}\n\n### QUESTION\n{example['question']}\n\n### SQL\n{example['answer']}"
        return {"text": text}

    # Training arguments optimized for the chosen quantization
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=default_batch_size,
        gradient_accumulation_steps=default_grad_accum,
        learning_rate=default_lr,
        max_steps=args.max_steps,
        logging_steps=10,
        fp16=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_strategy="checkpoint",
        dataloader_drop_last=True,
        group_by_length=True,
        report_to="none",  # Disable wandb logging
        save_steps=500,
        logging_first_step=True,
    )

    # Preprocess the dataset to add the text field that SFTTrainer expects
    print("Preprocessing dataset...")
    def preprocess_function(example):
        # Process single example, not batched
        text = f"### CONTEXT\n{example['context']}\n\n### QUESTION\n{example['question']}\n\n### SQL\n{example['answer']}"
        return {"text": text}
    
    # Apply preprocessing to create text field 
    dataset = dataset.map(preprocess_function, remove_columns=dataset.column_names)
    
    print("Initializing SFTTrainer...")
    
    # Based on https://huggingface.co/docs/trl/en/sft_trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        args=training_args,
    )
    print("SFTTrainer initialized successfully")

    print("Starting training...")
    trainer.train()
    print("Training complete.")
    
    trainer.save_model()
    print(f"Adapter saved to {args.output_dir}")

if __name__ == "__main__":
    main()