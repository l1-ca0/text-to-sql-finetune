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
import sqlite3
import re
import os
from datetime import datetime
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from huggingface_hub import login

# --- Global Configuration ---
ADAPTER_PATH = "./sql-mistral-adapter-final"  # Path to your fine-tuned adapter
BASE_MODEL_NAME = "mistralai/Mistral-7B-v0.3"

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None

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

def load_model_and_tokenizer():
    """Load model and tokenizer with proper error handling."""
    global model, tokenizer, device
    
    print("Loading model and tokenizer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        from transformers import BitsAndBytesConfig
        
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
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
            BASE_MODEL_NAME,
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        model.eval()
        print("Model loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        if "gated repo" in str(e).lower() or "unauthorized" in str(e).lower():
            print("[ERROR] This appears to be an authentication error.")
            print("Please ensure you have:")
            print("1. Requested access to the model on Hugging Face")
            print("2. Set your HF token: export HUGGINGFACE_HUB_TOKEN='your_token'")
            print("3. Or use: python app.py --hf_token 'your_token'")
        return False

def validate_sql_syntax(sql, schema=None):
    """Validate SQL syntax using sqlite3."""
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

def get_example_schemas():
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

def get_example_questions():
    """Return example questions for each schema type."""
    return {
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

# --- Core Generation Functions ---
def generate_sql(schema, question, temperature=0.1, max_tokens=200, format_output=True):
    """Generate SQL query with enhanced parameters and validation."""
    global model, tokenizer
    
    if not model or not tokenizer:
        return "Error: Model not loaded. Please restart the application.", "", "", ""
    
    if not schema.strip() or not question.strip():
        return "Please provide both schema and question.", "", "", ""
    
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
        
        if not sql_query:
            return "Could not generate SQL for the given input.", "", "", ""
        
        # Validate syntax with schema context
        is_valid, validation_msg = validate_sql_syntax(sql_query, schema)
        validation_status = f"✓ Valid SQL syntax" if is_valid else f"⚠ Warning: {validation_msg}"
        
        # Format SQL if requested
        formatted_sql = format_sql(sql_query) if format_output else sql_query
        
        # Execute SQL to show results
        query_results = execute_sql_query(sql_query, schema)
        
        return sql_query, formatted_sql, validation_status, query_results
        
    except Exception as e:
        return f"Error generating SQL: {str(e)}", "", "", ""

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
        
        # Safety check for empty results
        if len(results) == 0:
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
            fn=generate_sql,
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
    if not load_model_and_tokenizer():
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