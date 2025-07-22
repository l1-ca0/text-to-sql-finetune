"""
Interactive CLI chat interface for Text-to-SQL model.

This script provides a user-friendly command-line interface to interact with
the fine-tuned Mistral-7B model for generating SQL queries from natural language.

Features:
- Interactive schema input with validation
- Command history and shortcuts
- SQL syntax highlighting and validation
- Session management with save/load
- Multiple output formats
- Example schemas and questions
"""

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

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_colored(text, color=Colors.END):
    """Print colored text to terminal."""
    print(f"{color}{text}{Colors.END}")

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
        
        # Format results as a simple table
        if len(results) == 1 and len(results[0]) == 1:
            # Single value result
            return f"Result: {results[0][0]}"
        else:
            # Multiple rows/columns - format as simple table
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
            "Find members who have borrowed more than 1 book",
            "List all books by genre with their authors",
            "Which books are currently on loan?"
        ],
        "Employee Database": [
            "Show all employees in the Engineering department",
            "Find the highest paid employee in each department",
            "List employees hired after 2021",
            "Show the organizational hierarchy (managers and their reports)",
            "Calculate the average salary by department"
        ]
    }

def get_example_schemas():
    """Return example database schemas with sample data for user reference."""
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

def show_help():
    """Display help information."""
    help_text = f"""
{Colors.BOLD}Text-to-SQL Chat Interface Help{Colors.END}

{Colors.CYAN}Commands:{Colors.END}
  {Colors.GREEN}help{Colors.END} or {Colors.GREEN}?{Colors.END}           - Show this help message
  {Colors.GREEN}examples{Colors.END}          - Show example database schemas
  {Colors.GREEN}questions{Colors.END}         - Show example questions for current schema type
  {Colors.GREEN}clear{Colors.END}             - Clear the current schema
  {Colors.GREEN}history{Colors.END}           - Show conversation history
  {Colors.GREEN}save <filename>{Colors.END}   - Save current session
  {Colors.GREEN}load <filename>{Colors.END}   - Load previous session
  {Colors.GREEN}validate{Colors.END}          - Validate last generated SQL
  {Colors.GREEN}format{Colors.END}            - Format last generated SQL
  {Colors.GREEN}exit{Colors.END} or {Colors.GREEN}quit{Colors.END}      - Exit the program

{Colors.CYAN}Usage:{Colors.END}
1. Enter your database schema (CREATE TABLE statements)
2. Press Enter twice when finished with schema
3. Ask your question in natural language
4. Review the generated SQL query

{Colors.CYAN}Tips:{Colors.END}
• Use descriptive table and column names
• Include foreign key relationships in your schema
• Be specific in your questions
• Use 'examples' command to see sample schemas
"""
    print(help_text)

def show_examples():
    """Display example schemas."""
    examples = get_example_schemas()
    print(f"\n{Colors.BOLD}Example Database Schemas:{Colors.END}\n")
    
    for name, schema in examples.items():
        print(f"{Colors.CYAN}{name}:{Colors.END}")
        # Show only the CREATE TABLE parts for brevity
        schema_lines = schema.split('\n')
        create_section = []
        for line in schema_lines:
            if line.strip().startswith('--'):
                break
            create_section.append(line)
        print(f"{Colors.YELLOW}{chr(10).join(create_section[:15])}...{Colors.END}\n")

def show_example_questions():
    """Display example questions for schema types."""
    questions = get_example_questions()
    print(f"\n{Colors.BOLD}Example Questions by Schema Type:{Colors.END}\n")
    
    for schema_type, question_list in questions.items():
        print(f"{Colors.CYAN}{schema_type}:{Colors.END}")
        for i, question in enumerate(question_list, 1):
            print(f"  {i}. {question}")
        print()

class ChatSession:
    """Manages chat session state and history."""
    
    def __init__(self):
        self.history = []
        self.current_schema = ""
        self.last_sql = ""
    
    def add_interaction(self, schema, question, sql):
        """Add interaction to history."""
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'schema': schema,
            'question': question,
            'generated_sql': sql
        })
        self.last_sql = sql
    
    def save_session(self, filename):
        """Save session to file."""
        try:
            with open(filename, 'w') as f:
                json.dump({
                    'current_schema': self.current_schema,
                    'history': self.history
                }, f, indent=2)
            return True, f"Session saved to {filename}"
        except Exception as e:
            return False, f"Error saving session: {e}"
    
    def load_session(self, filename):
        """Load session from file."""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                self.current_schema = data.get('current_schema', '')
                self.history = data.get('history', [])
            return True, f"Session loaded from {filename}"
        except Exception as e:
            return False, f"Error loading session: {e}"
    
    def show_history(self):
        """Display conversation history."""
        if not self.history:
            print("No conversation history yet.")
            return
        
        print(f"\n{Colors.BOLD}Conversation History:{Colors.END}\n")
        for i, interaction in enumerate(self.history[-5:], 1):  # Show last 5
            print(f"{Colors.CYAN}[{i}] {interaction['timestamp'][:19]}{Colors.END}")
            print(f"Q: {interaction['question']}")
            print(f"SQL: {Colors.GREEN}{interaction['generated_sql']}{Colors.END}\n")

def generate_sql(model, tokenizer, schema, question, temperature=0.1):
    """Generate SQL with enhanced parameters."""
    prompt = f"### CONTEXT\n{schema}\n\n### QUESTION\n{question}\n\n### SQL\n"
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        sql_query = response.split("### SQL\n")[-1].strip()
        
        # Clean up common artifacts
        sql_query = sql_query.split('\n')[0]  # Take first line
        sql_query = sql_query.split('###')[0]  # Remove following sections
        sql_query = sql_query.strip()
        
        return sql_query
    except Exception as e:
        return f"Error generating SQL: {e}"

def authenticate_huggingface(token=None):
    """Handle Hugging Face authentication."""
    if token:
        print_colored("Using provided token for authentication...", Colors.CYAN)
        login(token=token)
        return True
    
    # Check for token in environment variables
    hf_token = os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HF_TOKEN')
    if hf_token:
        print_colored("Using token from environment variable...", Colors.CYAN)
        login(token=hf_token)
        return True
    
    # Check if already logged in
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print_colored(f"Already authenticated as: {user_info['name']}", Colors.GREEN)
        return True
    except Exception:
        pass
    
    print_colored("Authentication required for gated model.", Colors.YELLOW)
    print_colored("Please set HUGGINGFACE_HUB_TOKEN environment variable or use --hf_token argument", Colors.YELLOW)
    return False

def main():
    parser = argparse.ArgumentParser(description="Interactive chat with a Text-to-SQL model.")
    parser.add_argument("--adapter_path", type=str, default="./sql-mistral-adapter-final", help="Path to the fine-tuned adapter.")
    parser.add_argument("--base_model_name", type=str, default="mistralai/Mistral-7B-v0.3", help="Base model name.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature (0.0-1.0)")
    parser.add_argument("--load_session", type=str, help="Load previous session from file")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token for authentication")
    args = parser.parse_args()
    
    # Initialize session
    session = ChatSession()
    
    # Load previous session if specified
    if args.load_session:
        success, message = session.load_session(args.load_session)
        if success:
            print_colored(f"[SUCCESS] {message}", Colors.GREEN)
        else:
            print_colored(f"[ERROR] {message}", Colors.RED)
    
    # Authenticate with Hugging Face
    if not authenticate_huggingface(args.hf_token):
        print_colored("[ERROR] Authentication failed. Exiting.", Colors.RED)
        return
    
    # Load Models
    print_colored("Loading model and adapter...", Colors.CYAN)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print_colored(f"Using device: {device}", Colors.BLUE)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name,
            load_in_4bit=torch.cuda.is_available(),
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
        model.eval()
        print_colored("[SUCCESS] Model loaded successfully!", Colors.GREEN)
    except Exception as e:
        print_colored(f"[ERROR] Error loading model: {e}", Colors.RED)
        return
    
    # Welcome message
    print_colored(f"""
{Colors.BOLD}Welcome to Text-to-SQL Chat Interface!{Colors.END}

Type '{Colors.GREEN}help{Colors.END}' or '{Colors.GREEN}?{Colors.END}' for commands and usage instructions.
Type '{Colors.GREEN}examples{Colors.END}' to see sample database schemas.
Type '{Colors.GREEN}exit{Colors.END}' or '{Colors.GREEN}quit{Colors.END}' to end the session.
""", Colors.HEADER)

    # Main chat loop
    while True:
        try:
            # Schema input phase
            if not session.current_schema:
                print_colored("\nPlease provide your database schema:", Colors.CYAN)
                print_colored("(CREATE TABLE statements - press Enter twice when finished)", Colors.YELLOW)
                
                schema_lines = []
                empty_lines = 0
                
                while True:
                    line = input()
                    if line.strip() == "":
                        empty_lines += 1
                        if empty_lines >= 2:
                            break
                    else:
                        empty_lines = 0
                        schema_lines.append(line)
                
                schema = "\n".join(schema_lines).strip()
                
                if not schema:
                    print_colored("[ERROR] Schema cannot be empty. Please try again.", Colors.RED)
                    continue
                
                session.current_schema = schema
                print_colored("[SUCCESS] Schema saved!", Colors.GREEN)
            
            # Question input phase
            print_colored(f"\nCurrent schema: {len(session.current_schema)} characters", Colors.BLUE)
            question = input(f"{Colors.CYAN}Enter your question (or command): {Colors.END}")
            
            # Handle commands
            if question.lower() in ['exit', 'quit']:
                print_colored("Goodbye!", Colors.CYAN)
                break
            elif question.lower() in ['help', '?']:
                show_help()
                continue
            elif question.lower() == 'examples':
                show_examples()
                continue
            elif question.lower() == 'questions':
                show_example_questions()
                continue
            elif question.lower() == 'clear':
                session.current_schema = ""
                print_colored("Schema cleared!", Colors.YELLOW)
                continue
            elif question.lower() == 'history':
                session.show_history()
                continue
            elif question.lower().startswith('save '):
                filename = question[5:].strip()
                if filename:
                    success, message = session.save_session(filename)
                    color = Colors.GREEN if success else Colors.RED
                    prefix = "[SUCCESS]" if success else "[ERROR]"
                    print_colored(f"{prefix} {message}", color)
                else:
                    print_colored("[ERROR] Please specify a filename: save <filename>", Colors.RED)
                continue
            elif question.lower().startswith('load '):
                filename = question[5:].strip()
                if filename:
                    success, message = session.load_session(filename)
                    color = Colors.GREEN if success else Colors.RED
                    prefix = "[SUCCESS]" if success else "[ERROR]"
                    print_colored(f"{prefix} {message}", color)
                else:
                    print_colored("[ERROR] Please specify a filename: load <filename>", Colors.RED)
                continue
            elif question.lower() == 'validate':
                if session.last_sql:
                    is_valid, message = validate_sql_syntax(session.last_sql, session.current_schema)
                    color = Colors.GREEN if is_valid else Colors.RED
                    prefix = "[VALID]" if is_valid else "[INVALID]"
                    print_colored(f"{prefix} {message}", color)
                else:
                    print_colored("[ERROR] No SQL to validate. Generate a query first.", Colors.RED)
                continue
            elif question.lower() == 'format':
                if session.last_sql:
                    formatted = format_sql(session.last_sql)
                    print_colored(f"\nFormatted SQL:", Colors.CYAN)
                    print_colored(formatted, Colors.GREEN)
                else:
                    print_colored("[ERROR] No SQL to format. Generate a query first.", Colors.RED)
                continue
            
            if not question.strip():
                print_colored("[ERROR] Question cannot be empty.", Colors.RED)
                continue
            
            # Generate SQL
            print_colored("Generating SQL...", Colors.YELLOW)
            sql_query = generate_sql(model, tokenizer, session.current_schema, question, args.temperature)
            
            # Display result
            print_colored(f"\nGenerated SQL:", Colors.CYAN)
            print_colored(f"{Colors.GREEN}{sql_query}{Colors.END}")
            
            # Validate syntax with schema context
            is_valid, validation_msg = validate_sql_syntax(sql_query, session.current_schema)
            if is_valid:
                print_colored(f"[VALID] {validation_msg}", Colors.GREEN)
            else:
                print_colored(f"[WARNING] {validation_msg}", Colors.YELLOW)
            
            # Try to execute the query to show results
            try:
                query_results = execute_sql_query(sql_query, session.current_schema)
                if query_results:
                    print_colored(f"\nQuery Results:", Colors.CYAN)
                    print_colored(query_results, Colors.BLUE)
            except Exception as e:
                print_colored(f"[INFO] Could not execute query: {e}", Colors.YELLOW)
            
            # Add to session
            session.add_interaction(session.current_schema, question, sql_query)
            
            print_colored("-" * 50, Colors.BLUE)

        except KeyboardInterrupt:
            print_colored("\n\nExiting chat. Goodbye!", Colors.CYAN)
            break
        except Exception as e:
            print_colored(f"[ERROR] An error occurred: {e}", Colors.RED)
            print_colored("Please try again or type 'help' for assistance.", Colors.YELLOW)

if __name__ == "__main__":
    main()