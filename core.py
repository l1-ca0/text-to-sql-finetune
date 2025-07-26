"""
Core Text-to-SQL functionality module.

This module contains shared business logic for SQL generation, validation,
and model management used across different interfaces (web, CLI, notebook).
"""

import torch
import sqlite3
import re
import os
from datetime import datetime
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login


# --- Authentication ---
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
    
    print("[WARNING] Authentication required for gated model.")
    print("Please set HUGGINGFACE_HUB_TOKEN environment variable or use --hf_token argument")
    return False


# --- Model Loading & Management ---
def load_model_and_tokenizer(base_model_name, adapter_path):
    """Load model and tokenizer with proper error handling."""
    print("Loading model and tokenizer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

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
            base_model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
        print("Model loaded successfully.")
        return model, tokenizer, device
    except Exception as e:
        print(f"Error loading model: {e}")
        if "gated repo" in str(e).lower() or "unauthorized" in str(e).lower():
            print("[ERROR] This appears to be an authentication error.")
            print("Please ensure you have:")
            print("1. Requested access to the model on Hugging Face")
            print("2. Set your HF token: export HUGGINGFACE_HUB_TOKEN='your_token'")
            print("3. Or use: --hf_token 'your_token'")
        raise


# --- SQL Generation ---
def generate_sql(model, tokenizer, schema, question, temperature=0.1, max_tokens=200):
    """Generate SQL query from natural language question and schema."""
    if not schema.strip() or not question.strip():
        return "Please provide both schema and question."
    
    prompt = f"### CONTEXT\n{schema}\n\n### QUESTION\n{question}\n\n### SQL\n"
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract SQL query
        sql_query = response.split("### SQL\n")[-1].strip()
        
        # Clean up common artifacts
        if '\n' in sql_query:
            sql_query = sql_query.split('\n')[0]  # Take first line
        if '###' in sql_query:
            sql_query = sql_query.split('###')[0]  # Remove following sections
        sql_query = sql_query.strip()
        
        return sql_query if sql_query else "Could not generate SQL for the given input."
        
    except Exception as e:
        return f"Error generating SQL: {str(e)}"


# --- SQL Utilities ---
def validate_sql_syntax(sql, schema=None):
    """Validate SQL syntax using sqlite3 with optional schema context."""
    try:
        if not sql or not sql.strip():
            return False, "Empty SQL query"
            
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()
        
        # If schema is provided, create the tables first
        if schema:
            try:
                # Split schema into individual statements
                statements = [stmt.strip() for stmt in schema.split(';') if stmt.strip()]
                for statement in statements:
                    if statement.upper().startswith(('CREATE', 'INSERT')):
                        cursor.execute(statement)
            except Exception:
                # If schema creation fails, just do syntax validation
                pass
        
        # Try to parse the SQL using EXPLAIN (syntax check only)
        cursor.execute(f"EXPLAIN QUERY PLAN {sql}")
        conn.close()
        return True, "Valid SQL syntax"
    except Exception as e:
        # If EXPLAIN fails, try a simpler syntax check
        try:
            conn = sqlite3.connect(':memory:')
            cursor = conn.cursor()
            # Use sqlite3's built-in parser by preparing the statement
            try:
                conn.execute(f"EXPLAIN {sql}")
                conn.close()
                return True, "Valid SQL syntax (tables may not exist)"
            except sqlite3.OperationalError as parse_error:
                if "no such table" in str(parse_error).lower():
                    return True, "Valid SQL syntax (tables not found in validation DB)"
                else:
                    return False, f"SQL syntax error: {parse_error}"
        except Exception as e2:
            return False, f"SQL validation error: {str(e2)}"


def execute_sql_query(sql_query, schema):
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
        
        # Format results as a table
        if len(results) == 1 and len(results[0]) == 1:
            # Single value result
            return f"Result: {results[0][0]}"
        else:
            # Multiple rows/columns - format as table
            try:
                import pandas as pd
                df = pd.DataFrame(results, columns=column_names)
                return df.to_string(index=False, max_rows=20)
            except ImportError:
                # Fallback if pandas is not available
                output = []
                if column_names:
                    header = " | ".join(column_names)
                    output.append(header)
                    output.append("-" * len(header))
                
                for row in results[:20]:  # Limit to first 20 rows
                    output.append(" | ".join(str(cell) if cell is not None else "NULL" for cell in row))
                
                if len(results) > 20:
                    output.append(f"... and {len(results) - 20} more rows")
                
                return "\n".join(output)
            
    except Exception as e:
        return f"Query execution error: {str(e)}"


def format_sql(sql):
    """Basic SQL formatting for better readability."""
    if not sql:
        return sql
    
    # Add line breaks after major keywords
    keywords = ['SELECT', 'FROM', 'WHERE', 'JOIN', 'GROUP BY', 'ORDER BY', 'HAVING']
    formatted = sql
    
    for keyword in keywords:
        formatted = re.sub(f'\\b{keyword}\\b', f'\n{keyword}', formatted, flags=re.IGNORECASE)
    
    # Clean up extra whitespace
    formatted = re.sub(r'\n\s*\n', '\n', formatted)
    formatted = formatted.strip()
    
    return formatted


# --- Example Data ---
def get_example_data():
    """Return both schemas and questions from one function."""
    schemas = {
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
(3, 'Desk Chair', 199.99, 'Furniture', 0),
(4, 'Coffee Mug', 12.99, 'Kitchen', 25),
(5, 'Notebook', 8.99, 'Office', 100);

INSERT INTO customers VALUES 
(1, 'John Smith', 'john@email.com', '2023-01-15'),
(2, 'Jane Doe', 'jane@email.com', '2023-02-20'),
(3, 'Bob Wilson', 'bob@email.com', '2022-12-10');

INSERT INTO orders VALUES 
(1, 1, '2023-03-01', 1025.49),
(2, 2, '2023-03-05', 199.99),
(3, 1, '2023-03-10', 21.98);""",
        
        "Library System": """CREATE TABLE books (
    book_id INT PRIMARY KEY,
    title VARCHAR(200),
    author VARCHAR(100),
    isbn VARCHAR(20),
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
(1, 'The Great Gatsby', 'F. Scott Fitzgerald', '978-0-7432-7356-5', 1925, 'Fiction'),
(2, 'To Kill a Mockingbird', 'Harper Lee', '978-0-06-112008-4', 1960, 'Fiction'),
(3, 'Python Programming', 'John Doe', '978-1-234-56789-0', 2021, 'Technology'),
(4, 'Data Science Handbook', 'Jane Smith', '978-1-234-56790-6', 2022, 'Technology'),
(5, '1984', 'George Orwell', '978-0-452-28423-4', 1949, 'Fiction');

INSERT INTO members VALUES 
(1, 'Alice Johnson', 'alice@email.com', '2022-01-15'),
(2, 'Bob Brown', 'bob@email.com', '2022-06-20'),
(3, 'Carol Davis', 'carol@email.com', '2023-01-10');

INSERT INTO loans VALUES 
(1, 1, 1, '2023-03-01', '2023-03-15'),
(2, 3, 2, '2023-03-05', NULL),
(3, 4, 2, '2023-03-10', NULL),
(4, 2, 3, '2023-02-20', '2023-03-06');""",
        
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
    manager_id INT,
    FOREIGN KEY (manager_id) REFERENCES employees(employee_id)
);

-- Sample data
INSERT INTO departments VALUES 
(1, 'Engineering', 500000.00),
(2, 'Marketing', 200000.00),
(3, 'Sales', 300000.00),
(4, 'HR', 150000.00);

INSERT INTO employees VALUES 
(1, 'John Manager', 'Engineering', 120000.00, '2020-01-15', NULL),
(2, 'Alice Developer', 'Engineering', 95000.00, '2021-03-20', 1),
(3, 'Bob Developer', 'Engineering', 90000.00, '2021-06-10', 1),
(4, 'Carol Marketing', 'Marketing', 75000.00, '2022-01-05', NULL),
(5, 'Dave Sales', 'Sales', 80000.00, '2022-08-15', NULL),
(6, 'Eve HR', 'HR', 70000.00, '2023-01-10', NULL);"""
    }
    
    questions = {
        "E-commerce Store": [
            "What is the average price of all products?",
            "Show me all customers who registered in 2023",
            "Find the total sales amount for each customer",
            "Which products are out of stock?",
            "List the top 5 most expensive products"
        ],
        "Library System": [
            "How many books were published after 2020?",
            "Show all overdue books (not returned yet)",
            "Find members who have borrowed more than 3 books",
            "List all books by genre with their authors",
            "Which books are currently on loan?"
        ],
        "Employee Database": [
            "Show all employees in the Engineering department",
            "Find the highest paid employee in each department",
            "List employees hired in the last year",
            "Show the organizational hierarchy (managers and their reports)",
            "Calculate the average salary by department"
        ]
    }
    
    return {"schemas": schemas, "questions": questions}


def get_example_schemas():
    """Return example database schemas."""
    return get_example_data()["schemas"]


def get_example_questions():
    """Return example questions for each schema type."""
    return get_example_data()["questions"]