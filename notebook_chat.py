"""
Notebook-compatible Text-to-SQL Chat Interface.

This version is designed to work in Jupyter notebooks (Kaggle, Colab, JupyterLab) 
with interactive widgets instead of command-line input.
"""

import torch
import json
import os
from datetime import datetime
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
from core import (
    authenticate_huggingface, load_model_and_tokenizer, generate_sql,
    validate_sql_syntax, execute_sql_query, format_sql,
    get_example_schemas, get_example_questions
)

class TextToSQLChat:
    def __init__(self, adapter_path="./sql-mistral-adapter-final", base_model_name="mistralai/Mistral-7B-v0.3", hf_token=None):
        self.adapter_path = adapter_path
        self.base_model_name = base_model_name
        self.model = None
        self.tokenizer = None
        self.current_schema = ""
        self.history = []
        
        # Authenticate with Hugging Face
        authenticate_huggingface(hf_token)
        
        # Load model
        self.load_model()
        
        # Create UI
        self.create_ui()
    
    def load_model(self):
        """Load the model and tokenizer."""
        try:
            self.model, self.tokenizer, self.device = load_model_and_tokenizer(self.base_model_name, self.adapter_path)
            print("[SUCCESS] Model loaded successfully!")
        except Exception as e:
            print(f"[ERROR] Error loading model: {e}")
            raise
    
    def create_ui(self):
        """Create the interactive UI."""
        # Schema input
        self.schema_input = widgets.Textarea(
            value='',
            placeholder='Enter your database schema (CREATE TABLE statements)...',
            description='Schema:',
            layout=widgets.Layout(width='100%', height='200px')
        )
        
        # Question input
        self.question_input = widgets.Text(
            value='',
            placeholder='Enter your question in natural language...',
            description='Question:',
            layout=widgets.Layout(width='100%')
        )
        
        # Generate button
        self.generate_btn = widgets.Button(
            description='Generate SQL',
            button_style='primary',
            layout=widgets.Layout(width='150px')
        )
        self.generate_btn.on_click(self.generate_sql)
        
        # Clear button
        self.clear_btn = widgets.Button(
            description='Clear Schema',
            button_style='warning',
            layout=widgets.Layout(width='150px')
        )
        self.clear_btn.on_click(self.clear_schema)
        
        # Examples button
        self.examples_btn = widgets.Button(
            description='Load Example',
            button_style='info',
            layout=widgets.Layout(width='150px')
        )
        self.examples_btn.on_click(self.load_example)
        
        # Example selector
        self.example_selector = widgets.Dropdown(
            options=['E-commerce Store', 'Library System', 'Employee Database'],
            description='Example:',
            layout=widgets.Layout(width='200px')
        )
        
        # Output area
        self.output = widgets.Output()
        
        # Layout
        button_box = widgets.HBox([self.generate_btn, self.clear_btn, self.examples_btn])
        example_box = widgets.HBox([self.example_selector])
        
        self.ui = widgets.VBox([
            widgets.HTML("<h2>Text-to-SQL Chat Interface</h2>"),
            self.schema_input,
            example_box,
            self.question_input,
            button_box,
            self.output
        ])
    
    # Example schemas are now imported from core module
    
    def load_example(self, b):
        """Load an example schema."""
        examples = get_example_schemas()
        selected = self.example_selector.value
        if selected in examples:
            self.schema_input.value = examples[selected]
            with self.output:
                clear_output()
                print(f"[SUCCESS] Loaded {selected} example schema")
    
    def clear_schema(self, b):
        """Clear the schema input."""
        self.schema_input.value = ""
        self.current_schema = ""
        with self.output:
            clear_output()
            print("[INFO] Schema cleared")
    
    # SQL utilities are now imported from core module
    
    def generate_sql(self, b):
        """Generate SQL from the inputs."""
        schema = self.schema_input.value.strip()
        question = self.question_input.value.strip()
        
        with self.output:
            clear_output()
            
            if not schema:
                print("[ERROR] Please provide a database schema first")
                return
            
            if not question:
                print("[ERROR] Please enter a question")
                return
            
            print("[INFO] Generating SQL...")
            
            # Generate SQL
            sql_query = generate_sql(self.model, self.tokenizer, schema, question)
            
            # Display result
            print("Generated SQL:")
            print("=" * 50)
            print(sql_query)
            print("=" * 50)
            
            # Validate syntax
            is_valid, validation_msg = validate_sql_syntax(sql_query, schema)
            if is_valid:
                print(f"[VALID] {validation_msg}")
            else:
                print(f"[WARNING] {validation_msg}")
            
            # Try to execute the query
            try:
                query_results = execute_sql_query(sql_query, schema)
                if query_results:
                    print("\nQuery Results:")
                    print("-" * 30)
                    print(query_results)
            except Exception as e:
                print(f"[INFO] Could not execute query: {e}")
            
            # Add to history
            self.history.append({
                'timestamp': datetime.now().isoformat(),
                'schema': schema,
                'question': question,
                'generated_sql': sql_query
            })
    
    def display(self):
        """Display the chat interface."""
        display(self.ui)

# Usage function for notebooks
def start_chat(adapter_path="./sql-mistral-adapter-final", base_model_name="mistralai/Mistral-7B-v0.3", hf_token=None):
    """
    Start the Text-to-SQL chat interface in Jupyter notebooks.
    
    Args:
        adapter_path: Path to the fine-tuned adapter
        base_model_name: Base model name
        hf_token: Hugging Face token for authentication
    """
    chat = TextToSQLChat(adapter_path, base_model_name, hf_token)
    chat.display()
    return chat