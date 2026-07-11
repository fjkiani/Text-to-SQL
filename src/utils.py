import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pandas as pd


def load_db(db_path: str = "data/Chinook.db") -> sqlite3.Connection:
    """
    Load the SQLite database and return a connection.

    Args:
        db_path: Path to the SQLite database file. Defaults to "data/Chinook.db"

    Returns:
        sqlite3.Connection: Active database connection

    Raises:
        FileNotFoundError: If the database file doesn't exist
        sqlite3.Error: If there's an error connecting to the database
    """
    db_file = Path(db_path)
    if not db_file.exists():
        raise FileNotFoundError(
            f"Database file not found: {db_path}\n"
            "Please run setup.sh first to create the database."
        )

    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Error connecting to database: {e}")


def query_db(
    conn: sqlite3.Connection,
    query: str,
    params: Optional[tuple] = None,
    return_as_df: bool = True,
) -> List[Dict[str, Any]] | pd.DataFrame:
    """
    Execute a SQL query and return results as a pandas DataFrame or list of dictionaries.

    Args:
        conn: Active SQLite database connection
        query: SQL query string to execute
        params: Optional tuple of parameters for parameterized queries
        return_as_df: If True, return pandas DataFrame; if False, return list of dicts

    Returns:
        pd.DataFrame or List[Dict[str, Any]]: Query results

    Raises:
        sqlite3.Error: If there's an error executing the query
    """
    try:
        if return_as_df:
            return pd.read_sql_query(query, conn, params=params)
        else:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            columns = [description[0] for description in cursor.description]
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            return results
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Error executing query: {e}")


def get_schema(conn: sqlite3.Connection) -> Dict[str, List[Dict[str, str]]]:
    """
    Get the database schema including all tables and their columns.

    Args:
        conn: Active SQLite database connection

    Returns:
        Dict[str, List[Dict[str, str]]]: Dictionary mapping table names to their column info
    """
    schema = {}

    tables = query_db(
        conn,
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
        return_as_df=False,
    )

    for table in tables:
        table_name = table["name"]
        columns = query_db(conn, f"PRAGMA table_info({table_name})", return_as_df=False)
        schema[table_name] = columns

    return schema


def print_table_schema(
    conn: sqlite3.Connection, table_name: Optional[str] = None
) -> None:
    """
    Print a formatted view of the database schema.

    Args:
        conn: Active SQLite database connection
        table_name: Optional specific table name to display. If None, shows all tables.
    """
    schema = get_schema(conn)

    if table_name:
        if table_name not in schema:
            print(f"Error: Table '{table_name}' not found in database.")
            print(f"Available tables: {', '.join(schema.keys())}")
            return
        tables_to_print = {table_name: schema[table_name]}
    else:
        tables_to_print = schema

    print("\n" + "=" * 100)
    print(f"DATABASE SCHEMA - {len(schema)} tables")
    print("=" * 100)

    if not table_name:
        print("\nTables:")
        for i, tbl in enumerate(schema.keys(), 1):
            print(f"  {i}. {tbl}")
        print()

    for tbl_name, columns in tables_to_print.items():
        print("\n" + "-" * 100)
        print(f"Table: {tbl_name} ({len(columns)} columns)")
        print("-" * 100)
        print(f"{'Column':<30} {'Type':<20} {'Nullable':<12} {'PK':<5} {'Default':<15}")
        print("-" * 100)

        for col in columns:
            nullable = "NULL" if col["notnull"] == 0 else "NOT NULL"
            pk = "Y" if col["pk"] > 0 else ""
            default = str(col["dflt_value"]) if col["dflt_value"] is not None else ""
            print(
                f"{col['name']:<30} {col['type']:<20} {nullable:<12} {pk:<5} {default:<15}"
            )

    print("=" * 100 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# New functions for the agentic text-to-SQL system
# ──────────────────────────────────────────────────────────────────────────────

# SQL keywords that indicate destructive operations — blocked for safety
_DESTRUCTIVE_KEYWORDS = frozenset({
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
    "TRUNCATE", "REPLACE", "ATTACH", "DETACH", "PRAGMA",
    "VACUUM", "REINDEX",
})


def _is_destructive(sql: str) -> bool:
    """Check if a SQL statement contains destructive keywords outside of SELECT queries."""
    stripped = sql.strip().upper()
    # Allow SELECT queries (including WITH ... SELECT)
    if stripped.startswith("SELECT") or stripped.startswith("WITH"):
        return False
    # Check for destructive keywords at the start
    first_word = stripped.split()[0] if stripped.split() else ""
    return first_word in _DESTRUCTIVE_KEYWORDS


def get_ddl_schema(conn: sqlite3.Connection) -> str:
    """
    Get the full DDL schema (CREATE TABLE statements) from the database.

    This is used to inject the schema into the agent's system prompt so the
    model knows the exact table names, column names, types, and foreign key
    relationships without needing a tool call.

    Args:
        conn: Active SQLite database connection

    Returns:
        str: All CREATE TABLE statements joined by newlines
    """
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name, sql FROM sqlite_master "
        "WHERE type='table' AND sql IS NOT NULL ORDER BY name"
    )
    rows = cursor.fetchall()
    ddl_lines = [row[1].strip() + ";" for row in rows if row[1]]
    return "\n".join(ddl_lines)


def get_sample_rows(
    conn: sqlite3.Connection, table_name: str, n: int = 3
) -> List[Dict[str, Any]]:
    """
    Get n sample rows from a table as a list of dictionaries.

    This is exposed as a tool to the agent so it can inspect actual data
    values when the schema alone is ambiguous (e.g., understanding what
    values look like in a column).

    Args:
        conn: Active SQLite database connection
        table_name: Name of the table to sample
        n: Number of rows to return (default 3)

    Returns:
        List[Dict[str, Any]]: Sample rows as list of dicts

    Raises:
        sqlite3.Error: If the table doesn't exist or query fails
        ValueError: If table_name contains suspicious characters
    """
    # Sanitize table name — only allow alphanumeric and underscore
    if not table_name.replace("_", "").isalnum():
        raise ValueError(f"Invalid table name: {table_name}")

    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM [{table_name}] LIMIT {n}")
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    return [dict(zip(columns, row)) for row in rows]


def execute_sql(
    conn: sqlite3.Connection, sql: str
) -> Tuple[bool, Any, List[str]]:
    """
    Execute a SQL query safely and return structured results.

    This is the core function used by the agent's run_sql tool. It enforces
    read-only access by blocking destructive SQL statements, then executes
    the query and returns results in a structured format.

    Args:
        conn: Active SQLite database connection
        sql: SQL query string to execute

    Returns:
        Tuple of (success, results_or_error, columns):
        - On success: (True, [{"col": val, ...}, ...], ["col1", "col2", ...])
        - On error: (False, "error message string", [])
    """
    # Safety check: block destructive operations
    if _is_destructive(sql):
        return (
            False,
            f"Security: Destructive SQL operations are not allowed. "
            f"Only SELECT queries are permitted.",
            [],
        )

    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        results = [dict(zip(columns, row)) for row in rows]
        return (True, results, columns)
    except sqlite3.Error as e:
        return (False, f"SQLite error: {e}", [])
    except Exception as e:
        return (False, f"Unexpected error: {e}", [])


def format_results_as_table(
    results: List[Dict[str, Any]],
    columns: List[str],
    max_rows: int = 20,
) -> str:
    """
    Format query results as a readable text table for CLI display.

    Args:
        results: List of result rows as dictionaries
        columns: Column names to display
        max_rows: Maximum rows to show before truncating (default 20)

    Returns:
        str: Formatted table string
    """
    if not results:
        return "(no results)"

    # Truncate if needed
    display_rows = results[:max_rows]
    truncated = len(results) - max_rows

    # Calculate column widths
    widths = {}
    for col in columns:
        widths[col] = len(str(col))
        for row in display_rows:
            val = row.get(col, "")
            widths[col] = max(widths[col], len(str(val)))

    # Build separator and header
    sep = "+" + "+".join("-" * (w + 2) for w in [widths[c] for c in columns]) + "+"
    header = "|" + "|".join(f" {str(c):<{widths[c]}} " for c in columns) + "|"

    lines = [sep, header, sep]
    for row in display_rows:
        line = "|" + "|".join(
            f" {str(row.get(c, '')):<{widths[c]}} " for c in columns
        ) + "|"
        lines.append(line)
    lines.append(sep)

    if truncated > 0:
        lines.append(f"... {truncated} more rows")

    return "\n".join(lines)
