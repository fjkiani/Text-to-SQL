"""Agent logic for text-to-SQL conversion using a tool-calling ReAct loop."""

import json
import os
import time
import sqlite3
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

from src.utils import get_ddl_schema, get_sample_rows, execute_sql


@dataclass
class AgentResponse:
    """Structured response from the agent for a single question."""

    sql: str
    results: list
    columns: list
    summary: str
    latency: float
    attempts: int
    success: bool
    error: Optional[str]
    raw_messages: list = field(default_factory=list)


# Default model — fastest + cheapest on Fireworks serverless
DEFAULT_MODEL = "accounts/fireworks/models/gpt-oss-120b"

# Tool definitions exposed to the model via function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": (
                "Execute a SQL query against the SQLite database. "
                "Returns the query results as JSON, or an error message if the query fails."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "The SQL query to execute. Only SELECT queries are allowed.",
                    }
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sample_rows",
            "description": (
                "Get sample rows from a table to understand what the data looks like. "
                "Use this when the schema alone is ambiguous about column values."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "The name of the table to sample.",
                    }
                },
                "required": ["table_name"],
            },
        },
    },
]


def _build_system_prompt(ddl_schema: str) -> str:
    """Build the system prompt with injected DDL schema and SQL generation rules."""
    return f"""You are a text-to-SQL agent. You have access to a SQLite database with this schema:

{ddl_schema}

Use the run_sql tool to execute SQL queries. Use get_sample_rows to inspect data when the schema is ambiguous.

## SQL Generation Rules
1. Select only the columns the user asks for — do not add extra columns like IDs unless explicitly requested.
2. Use clear, descriptive column aliases (e.g., "TotalSales", "CustomerCount", "GenreName").
3. When the user asks for "names" of people, combine FirstName and LastName into a single column (e.g., "FirstName || ' ' || LastName AS Name").
4. Use ROUND(x, 2) for all monetary values to ensure clean output.
5. Use LEFT JOIN when the question implies all rows from the left table (e.g., "how many tracks in EACH playlist" — playlists with 0 tracks should still appear).
6. For "top N" questions, use ORDER BY ... DESC LIMIT N.
7. For ranking questions, use RANK() or ROW_NUMBER() window functions.
8. For date filtering, use strftime() (e.g., strftime('%Y', InvoiceDate) = '2021').
9. For "most" or "highest" questions, use ORDER BY ... DESC LIMIT 1.

## Workflow
1. Generate SQL that answers the user's question.
2. Call run_sql to execute it.
3. If the query fails, read the error message, fix the SQL, and try again.
4. Once you have results, provide a concise natural-language summary of what the data shows.
5. Do not include markdown code fences in your final summary — just describe the results.

Remember: you are helping a non-technical user understand their data. Be clear and concise."""


class TextToSQLAgent:
    """
    Agentic text-to-SQL converter using a tool-calling ReAct loop.

    The agent injects the database DDL schema into the system prompt, exposes
    run_sql and get_sample_rows tools to the model, and implements a self-healing
    loop where SQL execution errors are fed back to the model for correction.

    Conversation history is maintained across turns to support follow-up questions.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: str = "https://api.fireworks.ai/inference/v1",
        max_retries: int = 3,
        temperature: float = 0.0,
        max_tokens: int = 2000,
    ):
        self.conn = conn
        self.model = model
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize OpenAI-compatible client for Fireworks
        self.client = OpenAI(
            api_key=api_key or os.environ.get("FIREWORKS_API_KEY"),
            base_url=base_url,
        )

        # Build system prompt with DDL schema
        ddl = get_ddl_schema(conn)
        self.system_prompt = _build_system_prompt(ddl)

        # Conversation history (persists across turns for follow-up questions)
        self.messages: list[dict] = [{"role": "system", "content": self.system_prompt}]

    def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool call and return the result as a string for the model."""
        if tool_name == "run_sql":
            sql = arguments.get("sql", "")
            success, results, columns = execute_sql(self.conn, sql)
            if success:
                # Return results as JSON (truncate if very large)
                result_json = json.dumps(results[:50], default=str)
                if len(results) > 50:
                    result_json += f"\n... ({len(results)} total rows, showing first 50)"
                return result_json
            else:
                return results  # error message string

        elif tool_name == "get_sample_rows":
            table_name = arguments.get("table_name", "")
            try:
                samples = get_sample_rows(self.conn, table_name, n=3)
                return json.dumps(samples, default=str)
            except Exception as e:
                return f"Error getting sample rows: {e}"

        return f"Unknown tool: {tool_name}"

    def _extract_sql_from_tool_call(self, tool_call) -> str:
        """Extract the SQL string from a run_sql tool call."""
        try:
            args = json.loads(tool_call.function.arguments)
            return args.get("sql", "")
        except (json.JSONDecodeError, AttributeError):
            return ""

    def ask(self, question: str) -> AgentResponse:
        """
        Process a natural language question through the agentic loop.

        The loop:
        1. Add the user question to conversation history
        2. Call the model with tools
        3. If the model calls run_sql, execute it and feed results back
        4. If SQL fails, the error is fed back and the model self-corrects
        5. Repeat until the model produces a final text response or max_retries hit
        6. Return a structured AgentResponse

        Args:
            question: Natural language question from the user

        Returns:
            AgentResponse with SQL, results, summary, latency, and metadata
        """
        start_time = time.time()
        self.messages.append({"role": "user", "content": question})

        # Track the last successful SQL and results
        last_sql = ""
        last_results = []
        last_columns = []
        last_error = None
        attempts = 0
        tool_call_count = 0

        # Agentic loop: keep going until model produces final text or we hit limits
        while tool_call_count < self.max_retries * 2:  # allow multiple tool calls per retry
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=TOOLS,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
            except Exception as e:
                latency = time.time() - start_time
                return AgentResponse(
                    sql=last_sql,
                    results=last_results,
                    columns=last_columns,
                    summary="",
                    latency=latency,
                    attempts=attempts,
                    success=False,
                    error=f"API error: {e}",
                    raw_messages=self.messages.copy(),
                )

            choice = response.choices[0]
            msg = choice.message

            # If the model made tool calls, execute them and continue the loop
            if msg.tool_calls:
                # Add the assistant message with tool calls to history
                self.messages.append({
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                })

                # Execute each tool call
                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {}

                    # Track SQL attempts
                    if tool_name == "run_sql":
                        attempts += 1
                        sql = self._extract_sql_from_tool_call(tc)
                        if sql:
                            last_sql = sql

                    # Execute the tool
                    tool_result = self._execute_tool(tool_name, arguments)

                    # Track results/errors
                    if tool_name == "run_sql":
                        success, results, columns = execute_sql(self.conn, arguments.get("sql", ""))
                        if success:
                            last_results = results
                            last_columns = columns
                            last_error = None
                        else:
                            last_error = results  # error message

                    # Add tool result to conversation
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result,
                    })

                tool_call_count += 1
                continue

            # No tool calls — model produced a final text response
            else:
                summary = msg.content or ""
                self.messages.append({"role": "assistant", "content": summary})

                latency = time.time() - start_time
                return AgentResponse(
                    sql=last_sql,
                    results=last_results,
                    columns=last_columns,
                    summary=summary,
                    latency=latency,
                    attempts=attempts,
                    success=len(last_results) > 0 or attempts > 0,
                    error=last_error,
                    raw_messages=self.messages.copy(),
                )

        # Exhausted retries — return what we have
        latency = time.time() - start_time
        return AgentResponse(
            sql=last_sql,
            results=last_results,
            columns=last_columns,
            summary="",
            latency=latency,
            attempts=attempts,
            success=len(last_results) > 0,
            error=last_error or "Max retries exceeded without final response",
            raw_messages=self.messages.copy(),
        )

    def reset(self):
        """Clear conversation history (keeps the system prompt)."""
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def get_history(self) -> list[dict]:
        """Return the conversation history as a list of message dicts."""
        return self.messages.copy()
