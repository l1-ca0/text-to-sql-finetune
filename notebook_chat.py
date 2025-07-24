"""
Notebook-compatible Text-to-SQL Chat Interface.

This version is designed to work in Jupyter notebooks (Kaggle, Colab, JupyterLab) 
with interactive widgets instead of command-line input.
"""

import torch
import json
import sqlite3
import re
import os
from datetime import datetime
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

class TextToSQLChat:
    def __init__(self, adapter_path="./sql-mistral-adapter-final", base_model_name="mistralai/Mistral-7B-v0.3", hf_token=None):
        self.adapter_path = adapter_path
        self.base_model_name = base_model_name
        self.model = None
        self.tokenizer = None
        self.current_schema = ""
        self.history = []
        
        # Authenticate with Hugging Face
        if hf_token:
            login(token=hf_token)
        
        # Load model
        self.load_model()
        
        # Create UI
        self.create_ui()
    
    def load_model(self):
        """Load the model and tokenizer."""
        print("Loading model and adapter...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Configure quantization if CUDA is available
            quantization_config = None
            if torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
            
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
            self.model.eval()
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
    
    def get_example_schemas(self):
        """Return example database schemas."""
        return {
            "E-commerce Store": """CREATE TABLE products (
    product_id INT PRIMARY KEY,
    name VARCHAR(100),
    price DECIMAL(10,2),
    category VARCHAR(50),
    stock_quantity INT
);

CREATE TABLE customers (
    customer_id INT PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    registration_date DATE
);

CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    order_date DATE,
    total_amount DECIMAL(10,2),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Sample data
INSERT INTO products VALUES 
(1, 'Laptop', 999.99, 'Electronics', 10),
(2, 'Mouse', 25.50, 'Electronics', 50),
(3, 'Desk Chair', 199.99, 'Furniture', 0);

INSERT INTO customers VALUES 
(1, 'John Smith', 'john@email.com', '2023-01-15'),
(2, 'Jane Doe', 'jane@email.com', '2023-02-20');

INSERT INTO orders VALUES 
(1, 1, '2023-03-01', 1025.49),
(2, 2, '2023-03-05', 199.99);""",
            
            "Library System": """CREATE TABLE books (
    book_id INT PRIMARY KEY,
    title VARCHAR(200),
    author VARCHAR(100),
    publication_year INT,
    genre VARCHAR(50)
);

CREATE TABLE members (
    member_id INT PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    join_date DATE
);

CREATE TABLE loans (
    loan_id INT PRIMARY KEY,
    book_id INT,
    member_id INT,
    loan_date DATE,
    return_date DATE,
    FOREIGN KEY (book_id) REFERENCES books(book_id),
    FOREIGN KEY (member_id) REFERENCES members(member_id)
);

-- Sample data
INSERT INTO books VALUES 
(1, 'The Great Gatsby', 'F. Scott Fitzgerald', 1925, 'Fiction'),
(2, 'Python Programming', 'John Doe', 2021, 'Technology');

INSERT INTO members VALUES 
(1, 'Alice Johnson', 'alice@email.com', '2022-01-15'),
(2, 'Bob Brown', 'bob@email.com', '2022-06-20');

INSERT INTO loans VALUES 
(1, 1, 1, '2023-03-01', '2023-03-15'),
(2, 2, 2, '2023-03-05', NULL);""",
            
            "Employee Database": """CREATE TABLE departments (
    department_id INT PRIMARY KEY,
    department_name VARCHAR(50),
    budget DECIMAL(12,2)
);

CREATE TABLE employees (
    employee_id INT PRIMARY KEY,
    name VARCHAR(100),
    department VARCHAR(50),
    salary DECIMAL(10,2),
    hire_date DATE,
    manager_id INT
);

-- Sample data
INSERT INTO departments VALUES 
(1, 'Engineering', 500000.00),
(2, 'Marketing', 200000.00);

INSERT INTO employees VALUES 
(1, 'John Manager', 'Engineering', 120000.00, '2020-01-15', NULL),
(2, 'Alice Developer', 'Engineering', 95000.00, '2021-03-20', 1);"""
        }
    
    def load_example(self, b):
        """Load an example schema."""
        examples = self.get_example_schemas()
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
    
    def validate_sql_syntax(self, sql, schema=None):
        """Validate SQL syntax using sqlite3."""
        try:
            if not sql or not sql.strip():
                return False, "Empty SQL query"
                
            conn = sqlite3.connect(':memory:')
            cursor = conn.cursor()
            
            # If schema is provided, create the tables first
            if schema:
                try:
                    statements = [stmt.strip() for stmt in schema.split(';') if stmt.strip()]
                    for statement in statements:
                        if statement.upper().startswith(('CREATE', 'INSERT')):
                            cursor.execute(statement)
                except Exception:
                    pass
            
            # Try to parse the SQL using EXPLAIN
            cursor.execute(f"EXPLAIN QUERY PLAN {sql}")
            conn.close()
            return True, "Valid SQL syntax"
        except Exception as e:
            return False, f"SQL syntax error: {str(e)}"
    
    def execute_sql_query(self, sql_query, schema):
        """Execute SQL query against the schema and return results."""
        try:
            conn = sqlite3.connect(':memory:')
            cursor = conn.cursor()
            
            # Create tables and insert data from schema
            statements = [stmt.strip() for stmt in schema.split(';') if stmt.strip()]
            for statement in statements:
                if statement:
                    cursor.execute(statement)
            
            # Execute the query
            cursor.execute(sql_query)
            results = cursor.fetchall()
            
            # Get column names
            column_names = [description[0] for description in cursor.description] if cursor.description else []
            
            conn.close()
            
            if not results:
                return "Query executed successfully but returned no results."
            
            # Format results as a simple table
            if len(results) == 1 and len(results[0]) == 1:
                return f"Result: {results[0][0]}"
            else:
                output = []
                if column_names:
                    header = " | ".join(column_names)
                    output.append(header)
                    output.append("-" * len(header))
                
                for row in results[:10]:  # Limit to first 10 rows
                    output.append(" | ".join(str(cell) if cell is not None else "NULL" for cell in row))
                
                if len(results) > 10:
                    output.append(f"... and {len(results) - 10} more rows")
                
                return "\n".join(output)
                
        except Exception as e:
            return f"Query execution error: {str(e)}"
    
    def generate_sql_query(self, schema, question, temperature=0.1):
        """Generate SQL query from natural language."""
        prompt = f"### CONTEXT\n{schema}\n\n### QUESTION\n{question}\n\n### SQL\n"
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.1
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            sql_query = response.split("### SQL\n")[-1].strip()
            
            # Clean up common artifacts
            sql_query = sql_query.split('\n')[0]  # Take first line
            sql_query = sql_query.split('###')[0]  # Remove following sections
            sql_query = sql_query.strip()
            
            return sql_query
        except Exception as e:
            return f"Error generating SQL: {e}"
    
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
            sql_query = self.generate_sql_query(schema, question)
            
            # Display result
            print("Generated SQL:")
            print("=" * 50)
            print(sql_query)
            print("=" * 50)
            
            # Validate syntax
            is_valid, validation_msg = self.validate_sql_syntax(sql_query, schema)
            if is_valid:
                print(f"[VALID] {validation_msg}")
            else:
                print(f"[WARNING] {validation_msg}")
            
            # Try to execute the query
            try:
                query_results = self.execute_sql_query(sql_query, schema)
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