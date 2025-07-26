"""
Gradio Web Interface for Text-to-SQL Generation.

This script provides a user-friendly web interface for generating SQL queries
from natural language questions and database schemas using a fine-tuned Mistral-7B model.

Features:
- Interactive web interface with real-time generation
- Example schemas and questions for quick testing
- SQL syntax validation and formatting
- Generation parameter controls
- Session history and export functionality
- Responsive design with error handling
"""

import gradio as gr
import torch
import json
import os
from datetime import datetime
import argparse
from core import (
    authenticate_huggingface, load_model_and_tokenizer, generate_sql,
    validate_sql_syntax, execute_sql_query, format_sql,
    get_example_schemas, get_example_questions
)

# --- Global Configuration ---
ADAPTER_PATH = "./sql-mistral-adapter-final"  # Path to your fine-tuned adapter
BASE_MODEL_NAME = "mistralai/Mistral-7B-v0.3"

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None

def load_model():
    """Load model and tokenizer using core module."""
    global model, tokenizer, device
    
    try:
        model, tokenizer, device = load_model_and_tokenizer(BASE_MODEL_NAME, ADAPTER_PATH)
        return True
    except Exception:
        return False

# SQL utilities are now imported from core module

# Example data functions are now imported from core module

# --- Core Generation Functions ---
def generate_sql_with_validation(schema, question, temperature=0.1, max_tokens=200, format_output=True):
    """Generate SQL query with enhanced parameters and validation."""
    global model, tokenizer
    
    if not model or not tokenizer:
        return "Error: Model not loaded. Please restart the application.", "", "", ""
    
    # Use core module for SQL generation
    sql_query = generate_sql(model, tokenizer, schema, question, temperature, max_tokens)
    
    if sql_query.startswith("Error") or sql_query.startswith("Please provide"):
        return sql_query, "", "", ""
    
    # Validate syntax with schema context
    is_valid, validation_msg = validate_sql_syntax(sql_query, schema)
    validation_status = f"✓ Valid SQL syntax" if is_valid else f"⚠ Warning: {validation_msg}"
    
    # Format SQL if requested
    formatted_sql = format_sql(sql_query) if format_output else sql_query
    
    # Execute SQL to show results
    query_results = execute_sql_query(sql_query, schema)
    
    return sql_query, formatted_sql, validation_status, query_results

# SQL execution function is now imported from core module

def load_example_schema(schema_name):
    """Load an example schema."""
    examples = get_example_schemas()
    return examples.get(schema_name, "")

def load_example_question(schema_name):
    """Load example questions for a schema."""
    questions = get_example_questions()
    schema_questions = questions.get(schema_name, [])
    return gr.Dropdown(choices=schema_questions, value=schema_questions[0] if schema_questions else "")

def apply_example_question(question):
    """Apply selected example question."""
    return question

def clear_all():
    """Clear all inputs and outputs."""
    return "", "", "", "", "", ""

def export_session(schema, question, sql, formatted_sql, validation):
    """Export current session to JSON."""
    session_data = {
        "timestamp": datetime.now().isoformat(),
        "schema": schema,
        "question": question,
        "generated_sql": sql,
        "formatted_sql": formatted_sql,
        "validation_status": validation
    }
    
    filename = f"sql_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        return f"Session exported to {filename}"
    except Exception as e:
        return f"Error exporting session: {e}"

# --- Create Enhanced Gradio Interface ---
def create_interface():
    """Create the enhanced Gradio interface."""
    
    with gr.Blocks(title="Text-to-SQL Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Text-to-SQL Generator")
        gr.Markdown("Generate SQL queries from natural language using a fine-tuned Mistral-7B model.")
        
        with gr.Row():
            with gr.Column(scale=2):
                # Schema input section
                with gr.Group():
                    gr.Markdown("### Database Schema")
                    schema_input = gr.Textbox(
                        lines=8,
                        label="Table Schema",
                        placeholder="Enter your CREATE TABLE statements here...",
                        info="Provide the database schema (CREATE TABLE statements)"
                    )
                    
                    # Example schema selector
                    with gr.Row():
                        example_schema_dropdown = gr.Dropdown(
                            choices=list(get_example_schemas().keys()),
                            label="Load Example Schema",
                            value=None
                        )
                        load_schema_btn = gr.Button("Load Schema", size="sm")
                
                # Question input section
                with gr.Group():
                    gr.Markdown("### Question")
                    question_input = gr.Textbox(
                        lines=2,
                        label="Natural Language Question",
                        placeholder="e.g., What is the average price of all products?",
                        info="Ask your question in natural language"
                    )
                    
                    # Example questions
                    example_questions_dropdown = gr.Dropdown(
                        choices=[],
                        label="Example Questions",
                        value=None,
                        visible=False
                    )
                    apply_question_btn = gr.Button("Use This Question", size="sm", visible=False)
                
                # Generation parameters
                with gr.Accordion("Advanced Settings", open=False):
                    temperature_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.1,
                        step=0.1,
                        label="Temperature",
                        info="Higher values make output more creative but less predictable"
                    )
                    max_tokens_slider = gr.Slider(
                        minimum=50,
                        maximum=300,
                        value=200,
                        step=50,
                        label="Max Tokens",
                        info="Maximum length of generated SQL"
                    )
                    format_checkbox = gr.Checkbox(
                        value=True,
                        label="Format SQL Output",
                        info="Apply basic formatting to the generated SQL"
                    )
                
                # Action buttons
                with gr.Row():
                    generate_btn = gr.Button("Generate SQL", variant="primary", size="lg")
                    clear_btn = gr.Button("Clear All", size="lg")
            
            with gr.Column(scale=2):
                # Output section
                with gr.Group():
                    gr.Markdown("### Generated SQL")
                    sql_output = gr.Code(
                        label="Raw SQL",
                        language="sql",
                        lines=5
                    )
                    
                    formatted_output = gr.Code(
                        label="Formatted SQL",
                        language="sql",
                        lines=8
                    )
                    
                    validation_output = gr.Textbox(
                        label="Validation Status",
                        interactive=False,
                        lines=2
                    )
                    
                    query_results = gr.Textbox(
                        label="Query Results",
                        interactive=False,
                        lines=10,
                        max_lines=20
                    )
                
                # Export section
                with gr.Group():
                    gr.Markdown("### Session Management")
                    export_btn = gr.Button("Export Session", size="sm")
                    export_status = gr.Textbox(
                        label="Export Status",
                        interactive=False,
                        lines=1
                    )
        
        # Event handlers
        load_schema_btn.click(
            fn=load_example_schema,
            inputs=[example_schema_dropdown],
            outputs=[schema_input]
        )
        
        example_schema_dropdown.change(
            fn=load_example_question,
            inputs=[example_schema_dropdown],
            outputs=[example_questions_dropdown]
        ).then(
            fn=lambda x: (gr.update(visible=True), gr.update(visible=True)),
            inputs=[example_schema_dropdown],
            outputs=[example_questions_dropdown, apply_question_btn]
        )
        
        apply_question_btn.click(
            fn=apply_example_question,
            inputs=[example_questions_dropdown],
            outputs=[question_input]
        )
        
        generate_btn.click(
            fn=generate_sql_with_validation,
            inputs=[schema_input, question_input, temperature_slider, max_tokens_slider, format_checkbox],
            outputs=[sql_output, formatted_output, validation_output, query_results]
        )
        
        clear_btn.click(
            fn=clear_all,
            outputs=[schema_input, question_input, sql_output, formatted_output, validation_output, query_results]
        )
        
        export_btn.click(
            fn=export_session,
            inputs=[schema_input, question_input, sql_output, formatted_output, validation_output],
            outputs=[export_status]
        )
        
        # Add some example usage information
        with gr.Accordion("Usage Tips", open=False):
            gr.Markdown("""
            ### How to Use:
            1. **Schema**: Enter your database schema (CREATE TABLE statements)
            2. **Question**: Ask your question in natural language
            3. **Generate**: Click "Generate SQL" to create the query
            4. **Validate**: Check the validation status for syntax correctness
            5. **Export**: Save your session for later reference
            
            ### Tips for Better Results:
            - Use descriptive table and column names
            - Include foreign key relationships in your schema
            - Be specific in your questions
            - Try the example schemas to get started
            - Adjust temperature for more creative vs. precise outputs
            """)
    
    return demo

def main():
    """Main function to load model and launch interface."""
    global ADAPTER_PATH, BASE_MODEL_NAME
    
    parser = argparse.ArgumentParser(description="Launch Text-to-SQL Gradio Interface")
    parser.add_argument("--adapter_path", type=str, default=ADAPTER_PATH, help="Path to fine-tuned adapter")
    parser.add_argument("--base_model", type=str, default=BASE_MODEL_NAME, help="Base model name")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token for authentication")
    args = parser.parse_args()
    
    # Update global paths
    ADAPTER_PATH = args.adapter_path
    BASE_MODEL_NAME = args.base_model
    
    # Authenticate with Hugging Face
    if not authenticate_huggingface(args.hf_token):
        print("[ERROR] Authentication failed. Please check your token and try again.")
        return
    
    # Load model
    if not load_model():
        print("Failed to load model. Exiting.")
        return
    
    # Create and launch interface
    demo = create_interface()
    
    print(f"\nStarting Gradio interface...")
    print(f"Server will be available at:")
    print(f"  Local: http://localhost:{args.port}")
    print(f"  Network: http://0.0.0.0:{args.port}")
    if args.share:
        print(f"  Public link will be generated...")
    
    demo.launch(
        share=args.share,
        server_port=args.port,
        server_name="0.0.0.0",  # Always bind to all interfaces
        inbrowser=False  # Don't auto-open browser
    )

if __name__ == "__main__":
    main()