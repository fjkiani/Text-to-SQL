"""Multi-dialect SQL support for text-to-SQL.

Detects the SQL dialect from a database connection string and adjusts
the agent's SQL generation rules accordingly. Currently supports SQLite
and PostgreSQL, with MySQL stubbed for future expansion.

The key differences handled:
- Date functions: strftime() (SQLite) vs EXTRACT/DATE_TRUNC (PostgreSQL) vs DATE_FORMAT (MySQL)
- String concatenation: || (SQLite/PostgreSQL) vs CONCAT() (MySQL)
- Auto-increment: AUTOINCREMENT (SQLite) vs SERIAL (PostgreSQL) vs AUTO_INCREMENT (MySQL)
- Schema introspection: sqlite_master vs information_schema
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DialectConfig:
    """Configuration for a SQL dialect."""
    name: str
    display_name: str
    date_function: str  # e.g., "strftime('%Y', column)"
    date_year_example: str  # e.g., "strftime('%Y', InvoiceDate) = '2021'"
    string_concat_op: str  # "||" or "CONCAT(...)"
    string_concat_example: str
    limit_syntax: str  # "LIMIT n" or "TOP n"
    rules: list  # dialect-specific SQL generation rules


# ── Dialect configurations ────────────────────────────────────────────────────

DIALECTS = {
    "sqlite": DialectConfig(
        name="sqlite",
        display_name="SQLite",
        date_function="strftime()",
        date_year_example="strftime('%Y', InvoiceDate) = '2021'",
        string_concat_op="||",
        string_concat_example="FirstName || ' ' || LastName AS Name",
        limit_syntax="LIMIT n",
        rules=[
            "For date filtering, use strftime() (e.g., strftime('%Y', InvoiceDate) = '2021').",
            "For string concatenation, use the || operator (e.g., FirstName || ' ' || LastName).",
            "Use LIMIT n for restricting rows.",
        ],
    ),
    "postgresql": DialectConfig(
        name="postgresql",
        display_name="PostgreSQL",
        date_function="EXTRACT() / DATE_TRUNC()",
        date_year_example="EXTRACT(YEAR FROM InvoiceDate) = 2021",
        string_concat_op="||",
        string_concat_example="FirstName || ' ' || LastName AS Name",
        limit_syntax="LIMIT n",
        rules=[
            "For date filtering, use EXTRACT() (e.g., EXTRACT(YEAR FROM InvoiceDate) = 2021) or DATE_TRUNC().",
            "For string concatenation, use the || operator or CONCAT() function.",
            "Use LIMIT n for restricting rows.",
            "Use ILIKE for case-insensitive string matching (instead of LIKE).",
            "Use ::type for type casting (e.g., value::numeric).",
        ],
    ),
    "mysql": DialectConfig(
        name="mysql",
        display_name="MySQL",
        date_function="DATE_FORMAT() / YEAR()",
        date_year_example="YEAR(InvoiceDate) = 2021",
        string_concat_op="CONCAT()",
        string_concat_example="CONCAT(FirstName, ' ', LastName) AS Name",
        limit_syntax="LIMIT n",
        rules=[
            "For date filtering, use YEAR() (e.g., YEAR(InvoiceDate) = 2021) or DATE_FORMAT().",
            "For string concatenation, use CONCAT() function (e.g., CONCAT(FirstName, ' ', LastName)).",
            "Use LIMIT n for restricting rows.",
            "Use backticks for identifier quoting if needed.",
        ],
    ),
}


def detect_dialect(connection_string: str) -> str:
    """
    Detect the SQL dialect from a connection string or URI.

    Examples:
        "sqlite:///data/Chinook.db"     -> "sqlite"
        "postgresql://user:pass@host/db" -> "postgresql"
        "mysql://user:pass@host/db"      -> "mysql"
        "data/Chinook.db"                -> "sqlite" (default for file paths)

    Args:
        connection_string: Database connection string or file path

    Returns:
        str: Dialect name ("sqlite", "postgresql", or "mysql")
    """
    cs = connection_string.lower().strip()

    if cs.startswith("postgresql://") or cs.startswith("postgres://") or cs.startswith("psql://"):
        return "postgresql"
    if cs.startswith("mysql://") or cs.startswith("mariadb://"):
        return "mysql"
    # Default to SQLite for file paths and sqlite:// URIs
    return "sqlite"


def get_dialect_config(dialect: str) -> DialectConfig:
    """
    Get the DialectConfig for a named dialect.

    Args:
        dialect: Dialect name ("sqlite", "postgresql", "mysql")

    Returns:
        DialectConfig with dialect-specific SQL rules and syntax
    """
    return DIALECTS.get(dialect, DIALECTS["sqlite"])


def get_dialect_rules(dialect: str) -> list[str]:
    """Get the list of SQL generation rules for a specific dialect."""
    config = get_dialect_config(dialect)
    return config.rules


def get_dialect_specific_prompt(dialect: str) -> str:
    """
    Build a dialect-specific prompt section for the agent system prompt.

    This replaces the hardcoded SQLite-specific rules in the system prompt
    with rules appropriate for the target dialect.
    """
    config = get_dialect_config(dialect)
    lines = [f"## {config.display_name} SQL Dialect Rules"]
    for rule in config.rules:
        lines.append(f"- {rule}")
    return "\n".join(lines)
