import torch
import json
import re
import sqlite3
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
from collections import defaultdict
import pandas as pd
from datetime import datetime
from huggingface_hub import login
import os

def format_prompt(sample):
    return f"### CONTEXT\n{sample['context']}\n\n### QUESTION\n{sample['question']}\n\n### SQL\n"

def normalize_sql(sql):
    """Normalize SQL for better comparison."""
    if not sql:
        return ""
    # Remove extra whitespace and normalize case
    sql = re.sub(r'\s+', ' ', sql.strip())
    # Remove trailing semicolon
    sql = sql.rstrip(';')
    return sql.lower()

def is_sql_syntactically_valid(sql):
    """Check if SQL is syntactically valid using sqlite3."""
    try:
        if not sql or not sql.strip():
            return False
        # Create in-memory database
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()
        # Try to explain the query (doesn't execute, just parses)
        cursor.execute(f"EXPLAIN QUERY PLAN {sql}")
        conn.close()
        return True
    except Exception:
        return False

def calculate_token_overlap(pred_tokens, true_tokens):
    """Calculate token-level overlap between predicted and true SQL."""
    pred_set = set(pred_tokens)
    true_set = set(true_tokens)
    
    if not true_set or len(true_set) == 0:
        return 0.0
    
    intersection = pred_set.intersection(true_set)
    return len(intersection) / len(true_set)

def extract_sql_components(sql):
    """Extract key SQL components for analysis."""
    sql_lower = sql.lower()
    components = {
        'has_select': 'select' in sql_lower,
        'has_where': 'where' in sql_lower,
        'has_join': any(join in sql_lower for join in ['join', 'inner join', 'left join', 'right join']),
        'has_group_by': 'group by' in sql_lower,
        'has_order_by': 'order by' in sql_lower,
        'has_having': 'having' in sql_lower,
        'has_subquery': '(' in sql_lower and 'select' in sql_lower,
        'has_aggregate': any(agg in sql_lower for agg in ['count', 'sum', 'avg', 'max', 'min'])
    }
    return components

def evaluate_model(model, tokenizer, dataset, model_name="Model", sample_size=None):
    """Enhanced evaluation with multiple metrics."""
    results = {
        'exact_match': 0,
        'normalized_match': 0,
        'syntactically_valid': 0,
        'token_overlap_scores': [],
        'errors': [],
        'predictions': [],
        'component_analysis': defaultdict(lambda: {'correct': 0, 'total': 0})
    }
    
    # Sample dataset if requested
    if sample_size and sample_size < len(dataset):
        dataset = dataset.select(range(sample_size))
    
    total = len(dataset)
    
    for i, sample in enumerate(tqdm(dataset, desc=f"Evaluating {model_name}")):
        prompt = format_prompt(sample)
        ground_truth_sql = sample['answer']
        
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")
            
            outputs = model.generate(
                **inputs, 
                max_new_tokens=150, 
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,  # Deterministic for evaluation
                temperature=0.1
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract predicted SQL
            predicted_sql = generated_text.split("### SQL\n")[-1].strip()
            
            # Clean up common generation artifacts
            predicted_sql = predicted_sql.split('\n')[0]  # Take first line only
            predicted_sql = predicted_sql.split('###')[0]  # Remove any following sections
            predicted_sql = predicted_sql.strip()
            
            # Normalize for comparison
            pred_normalized = normalize_sql(predicted_sql)
            true_normalized = normalize_sql(ground_truth_sql)
            
            # Exact match
            if predicted_sql.lower().strip() == ground_truth_sql.lower().strip():
                results['exact_match'] += 1
            
            # Normalized match
            if pred_normalized == true_normalized:
                results['normalized_match'] += 1
            
            # Syntax validation
            if is_sql_syntactically_valid(predicted_sql):
                results['syntactically_valid'] += 1
            
            # Token overlap
            pred_tokens = pred_normalized.split()
            true_tokens = true_normalized.split()
            overlap = calculate_token_overlap(pred_tokens, true_tokens)
            results['token_overlap_scores'].append(overlap)
            
            # Component analysis
            true_components = extract_sql_components(ground_truth_sql)
            pred_components = extract_sql_components(predicted_sql)
            
            for component, has_component in true_components.items():
                results['component_analysis'][component]['total'] += 1
                if has_component and pred_components.get(component, False):
                    results['component_analysis'][component]['correct'] += 1
            
            # Store prediction for detailed analysis
            results['predictions'].append({
                'index': i,
                'question': sample['question'],
                'context': sample['context'][:200] + "..." if len(sample['context']) > 200 else sample['context'],
                'ground_truth': ground_truth_sql,
                'predicted': predicted_sql,
                'exact_match': predicted_sql.lower().strip() == ground_truth_sql.lower().strip(),
                'normalized_match': pred_normalized == true_normalized,
                'syntactically_valid': is_sql_syntactically_valid(predicted_sql),
                'token_overlap': overlap
            })
            
        except Exception as e:
            results['errors'].append({
                'index': i,
                'error': str(e),
                'question': sample['question']
            })
            # Add empty prediction to maintain consistency
            results['predictions'].append({
                'index': i,
                'question': sample['question'],
                'context': sample['context'][:200] + "..." if len(sample['context']) > 200 else sample['context'],
                'ground_truth': sample['answer'],
                'predicted': "",
                'exact_match': False,
                'normalized_match': False,
                'syntactically_valid': False,
                'token_overlap': 0.0
            })
    
    # Calculate final metrics
    metrics = {
        'exact_match_accuracy': (results['exact_match'] / total) * 100,
        'normalized_match_accuracy': (results['normalized_match'] / total) * 100,
        'syntax_validity_rate': (results['syntactically_valid'] / total) * 100,
        'avg_token_overlap': sum(results['token_overlap_scores']) / len(results['token_overlap_scores']) * 100 if results['token_overlap_scores'] else 0,
        'error_rate': (len(results['errors']) / total) * 100,
        'total_samples': total
    }
    
    return metrics, results

def print_detailed_results(metrics, results, model_name):
    """Print comprehensive evaluation results."""
    print(f"\n{'='*50}")
    print(f"{model_name} Evaluation Results")
    print(f"{'='*50}")
    
    print(f"Overall Metrics:")
    print(f"  • Exact Match Accuracy:     {metrics['exact_match_accuracy']:.2f}%")
    print(f"  • Normalized Match Accuracy: {metrics['normalized_match_accuracy']:.2f}%")
    print(f"  • Syntax Validity Rate:     {metrics['syntax_validity_rate']:.2f}%")
    print(f"  • Average Token Overlap:    {metrics['avg_token_overlap']:.2f}%")
    print(f"  • Error Rate:               {metrics['error_rate']:.2f}%")
    print(f"  • Total Samples:            {metrics['total_samples']}")
    
    # Component analysis
    if results['component_analysis']:
        print(f"\nSQL Component Analysis:")
        for component, stats in results['component_analysis'].items():
            if stats['total'] > 0:
                accuracy = (stats['correct'] / stats['total']) * 100
                print(f"  • {component.replace('has_', '').replace('_', ' ').title()}: {accuracy:.1f}% ({stats['correct']}/{stats['total']})")
    
    # Error analysis
    if results['errors']:
        print(f"\nSample Errors:")
        for error in results['errors'][:3]:  # Show first 3 errors
            print(f"  • Sample {error['index']}: {error['error']}")
    
    print()

def save_detailed_report(base_metrics, base_results, ft_metrics, ft_results, output_file):
    """Save detailed evaluation report to JSON."""
    report = {
        'timestamp': datetime.now().isoformat(),
        'base_model': {
            'metrics': base_metrics,
            'sample_predictions': base_results['predictions'][:10]  # Save first 10 for space
        },
        'fine_tuned_model': {
            'metrics': ft_metrics,
            'sample_predictions': ft_results['predictions'][:10]
        },
        'comparison': {
            'improvement_exact_match': ft_metrics['exact_match_accuracy'] - base_metrics['exact_match_accuracy'],
            'improvement_normalized': ft_metrics['normalized_match_accuracy'] - base_metrics['normalized_match_accuracy'],
            'improvement_syntax': ft_metrics['syntax_validity_rate'] - base_metrics['syntax_validity_rate'],
            'improvement_token_overlap': ft_metrics['avg_token_overlap'] - base_metrics['avg_token_overlap']
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Detailed report saved to: {output_file}")

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
    
    print("Authentication required for gated model.")
    print("Please set HUGGINGFACE_HUB_TOKEN environment variable or use --hf_token argument")
    return False

def main():
    parser = argparse.ArgumentParser(description="Evaluate a Text-to-SQL model with comprehensive metrics.")
    parser.add_argument("--adapter_path", type=str, default="./sql-mistral-adapter-final", help="Path to the fine-tuned adapter.")
    parser.add_argument("--base_model_name", type=str, default="mistralai/Mistral-7B-v0.3", help="Base model name.")
    parser.add_argument("--sample_size", type=int, default=None, help="Number of samples to evaluate (default: all)")
    parser.add_argument("--output_report", type=str, default=None, help="Path to save detailed JSON report")
    parser.add_argument("--skip_base", action='store_true', help="Skip base model evaluation")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token for authentication")
    args = parser.parse_args()
    
    # Authenticate with Hugging Face
    if not authenticate_huggingface(args.hf_token):
        print("[ERROR] Authentication failed. Exiting.")
        return

    # Load dataset (only train split available, so we'll use a subset for evaluation)
    print("Loading dataset...")
    full_dataset = load_dataset("b-mc2/sql-create-context", split="train")
    
    # Use last 1000 samples as validation set (or all if dataset is smaller)
    dataset_size = len(full_dataset)
    eval_size = min(1000, dataset_size // 10)  # Use 10% or 1000 samples, whichever is smaller
    dataset = full_dataset.select(range(dataset_size - eval_size, dataset_size))
    
    print(f"Using {len(dataset)} samples from the end of training set for evaluation")
    
    if args.sample_size:
        print(f"Further limiting to sample size: {args.sample_size}")
        dataset = dataset.select(range(min(args.sample_size, len(dataset))))
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_metrics, base_results = None, None
    
    # Evaluate base model (optional)
    if not args.skip_base:
        print("\nLoading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name, 
            load_in_4bit=True, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        base_model.eval()
        
        base_metrics, base_results = evaluate_model(
            base_model, tokenizer, dataset, "Base Model", args.sample_size
        )
        print_detailed_results(base_metrics, base_results, "Base Model")
        
        # Free memory
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Evaluate fine-tuned model
    print("\nLoading fine-tuned model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name, 
        load_in_4bit=True, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    ft_model = PeftModel.from_pretrained(base_model, args.adapter_path)
    ft_model.eval()
    
    ft_metrics, ft_results = evaluate_model(
        ft_model, tokenizer, dataset, "Fine-Tuned Model", args.sample_size
    )
    print_detailed_results(ft_metrics, ft_results, "Fine-Tuned Model")

    # Comparison summary
    if base_metrics:
        print(f"\nImprovement Summary:")
        print(f"  • Exact Match:      {ft_metrics['exact_match_accuracy'] - base_metrics['exact_match_accuracy']:+.2f}%")
        print(f"  • Normalized Match: {ft_metrics['normalized_match_accuracy'] - base_metrics['normalized_match_accuracy']:+.2f}%")
        print(f"  • Syntax Validity:  {ft_metrics['syntax_validity_rate'] - base_metrics['syntax_validity_rate']:+.2f}%")
        print(f"  • Token Overlap:    {ft_metrics['avg_token_overlap'] - base_metrics['avg_token_overlap']:+.2f}%")

    # Save detailed report
    if args.output_report:
        if base_metrics:
            save_detailed_report(base_metrics, base_results, ft_metrics, ft_results, args.output_report)
        else:
            # Save only fine-tuned results
            report = {
                'timestamp': datetime.now().isoformat(),
                'fine_tuned_model': {
                    'metrics': ft_metrics,
                    'sample_predictions': ft_results['predictions'][:20]
                }
            }
            with open(args.output_report, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to: {args.output_report}")

if __name__ == "__main__":
    main()