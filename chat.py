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
import os
from datetime import datetime
import argparse
from core import (
    authenticate_huggingface, load_model_and_tokenizer, generate_sql,
    validate_sql_syntax, execute_sql_query, format_sql,
    get_example_schemas, get_example_questions
)

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

# SQL utilities are now imported from core module

# Example data functions are now imported from core module

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

# SQL generation and authentication functions are now imported from core module

def main():
    parser = argparse.ArgumentParser(description="Interactive chat with a Text-to-SQL model.")
    parser.add_argument("--adapter_path", type=str, default="./sql-mistral-adapter-final", help="Path to the fine-tuned adapter.")
    parser.add_argument("--base_model_name", type=str, default="mistralai/Mistral-7B-v0.3", help="Base model name.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature (0.0-1.0)")
    parser.add_argument("--load_session", type=str, help="Load previous session from file")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token for authentication")
    parser.add_argument("--use_8bit", action="store_true", help="Use 8-bit quantization for inference")
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
    
    try:
        model, tokenizer, device = load_model_and_tokenizer(args.base_model_name, args.adapter_path, args.use_8bit)
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