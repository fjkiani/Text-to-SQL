"""CLI entry point. Run with: uv run cli (or python -m src.cli)"""

import argparse
import os
import sys
import time

from dotenv import load_dotenv

from src.utils import load_db, print_table_schema, format_results_as_table
from src.agent import TextToSQLAgent, DEFAULT_MODEL


# Available models for the --model flag
AVAILABLE_MODELS = {
    "gpt-oss-120b": "accounts/fireworks/models/gpt-oss-120b",
    "deepseek-v4-flash": "accounts/fireworks/models/deepseek-v4-flash",
    "glm-5p2": "accounts/fireworks/models/glm-5p2",
    "kimi-k2p6": "accounts/fireworks/models/kimi-k2p6",
    "deepseek-v4-pro": "accounts/fireworks/models/deepseek-v4-pro",
}


def _print_banner(model_name: str, db_path: str):
    """Print the welcome banner."""
    print()
    print("=" * 70)
    print("  Agentic Text-to-SQL CLI")
    print("  Powered by Fireworks AI open-source models")
    print("=" * 70)
    print(f"  Model: {model_name}")
    print(f"  Database: {db_path}")
    print()
    print("  Type a question in natural language and press Enter.")
    print("  Commands:  exit/quit  |  help  |  schema  |  clear  |  history")
    print("=" * 70)
    print()


def _print_help():
    """Print available commands."""
    print()
    print("Available commands:")
    print("  <question>   Ask a natural language question about your data")
    print("  schema       Show the database schema (all tables and columns)")
    print("  clear        Clear conversation history (start fresh)")
    print("  history      Show conversation history from this session")
    print("  help         Show this help message")
    print("  exit/quit    Exit the CLI")
    print()
    print("Available models (use --model flag at startup):")
    for name, model_id in AVAILABLE_MODELS.items():
        marker = " (default)" if name == "gpt-oss-120b" else ""
        print(f"  {name:<20s} {model_id}{marker}")
    print()


def _print_response(response):
    """Print the agent's response in a formatted way."""
    print()
    print("-" * 70)

    # Show SQL
    if response.sql:
        print(f"SQL:")
        print(f"  {response.sql}")
        print()

    # Show execution status
    if response.success:
        print(f"Results ({len(response.results)} rows, {response.attempts} attempt(s), {response.latency:.2f}s):")
        print()
        if response.results and response.columns:
            table = format_results_as_table(response.results, response.columns, max_rows=20)
            for line in table.split("\n"):
                print(f"  {line}")
        else:
            print("  (no results returned)")
    else:
        print(f"Error: {response.error}")
        print(f"  (attempted {response.attempts} time(s), {response.latency:.2f}s)")

    # Show summary
    if response.summary:
        print()
        print("Summary:")
        # Indent the summary for readability
        for line in response.summary.split("\n"):
            print(f"  {line}")

    print("-" * 70)
    print()


def main():
    """Main CLI entry point — interactive REPL for text-to-SQL queries."""
    parser = argparse.ArgumentParser(
        description="Agentic Text-to-SQL CLI powered by Fireworks AI"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-oss-120b",
        help=(
            "Model to use. Options: gpt-oss-120b (default), "
            "deepseek-v4-flash, glm-5p2, kimi-k2p6, deepseek-v4-pro. "
            "You can also pass a full model ID (accounts/fireworks/models/...)."
        ),
    )
    parser.add_argument(
        "--db",
        type=str,
        default="data/Chinook.db",
        help="Path to the SQLite database file (default: data/Chinook.db)",
    )
    args = parser.parse_args()

    # Load .env if present
    load_dotenv()

    # Resolve model ID
    model_id = AVAILABLE_MODELS.get(args.model, args.model)

    # Check API key
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        print("Error: FIREWORKS_API_KEY environment variable is not set.")
        print("Set it with: export FIREWORKS_API_KEY=<your-key>")
        print("Or create a .env file with: FIREWORKS_API_KEY=<your-key>")
        sys.exit(1)

    # Load database
    try:
        conn = load_db(args.db)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading database: {e}")
        sys.exit(1)

    # Create agent
    try:
        agent = TextToSQLAgent(conn, model=model_id, api_key=api_key)
    except Exception as e:
        print(f"Error initializing agent: {e}")
        sys.exit(1)

    # Print banner
    _print_banner(args.model, args.db)

    # REPL loop
    while True:
        try:
            user_input = input("sql> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print("Goodbye!")
            break

        if not user_input:
            continue

        # Handle commands
        lower = user_input.lower()
        if lower in ("exit", "quit"):
            print("Goodbye!")
            break
        elif lower == "help":
            _print_help()
            continue
        elif lower == "schema":
            print_table_schema(conn)
            continue
        elif lower == "clear":
            agent.reset()
            print("Conversation history cleared.")
            print()
            continue
        elif lower == "history":
            history = agent.get_history()
            print(f"\nConversation history ({len(history)} messages):")
            print("-" * 50)
            for msg in history:
                role = msg.get("role", "?")
                content = msg.get("content", "")
                if content:
                    # Truncate long messages
                    display = content[:100] + "..." if len(content) > 100 else content
                    print(f"  [{role}] {display}")
                elif msg.get("tool_calls"):
                    print(f"  [{role}] (tool calls: {[tc['function']['name'] for tc in msg['tool_calls']]})")
            print("-" * 50)
            print()
            continue

        # Process as a question
        try:
            response = agent.ask(user_input)
            _print_response(response)
        except Exception as e:
            print(f"\nError processing question: {e}")
            print()

    # Cleanup
    conn.close()


if __name__ == "__main__":
    main()
