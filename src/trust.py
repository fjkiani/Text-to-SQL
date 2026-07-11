"""Trust and validation layer for text-to-SQL.

Provides 5 production trust signals that are computed for every query:

1. SQL validation — Parses generated SQL, validates JOIN paths against the
   schema's foreign key graph, detects wrong-table joins and suspicious patterns.
2. Row count sanity — Compares result row counts against expected ranges based
   on question type (aggregation, lookup, ranking, etc.).
3. Confidence scoring — Combines all signals into a 0-100 score with a breakdown.
4. Summary verification — Cross-checks LLM summary claims against actual result data.
5. Provenance tracking — Records which few-shot patterns matched, whether the
   model deviated from known-good SQL, and the full decision trail.

Usage:
    from src.trust import TrustLayer

    trust = TrustLayer(conn)
    report = trust.analyze(
        question="What are the top 5 genres by sales?",
        sql="SELECT g.Name, SUM(il.UnitPrice * il.Quantity) FROM ...",
        results=[{"Genre": "Rock", "TotalSales": 826.65}, ...],
        columns=["Genre", "TotalSales"],
        summary="The top five genres are...",
        attempts=1,
        few_shot_patterns=["aggregation_join"],
    )
    print(report.confidence_score)  # 0-100
    print(report.flags)             # list of issues
"""

import re
import sqlite3
from dataclasses import dataclass, field
from typing import Optional


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ValidationFlag:
    """A single validation issue or observation."""
    severity: str  # "critical", "warning", "info", "ok"
    category: str  # "join_path", "row_count", "summary", "provenance"
    message: str
    detail: Optional[str] = None


@dataclass
class TrustReport:
    """Complete trust analysis for a single query."""
    confidence_score: int  # 0-100
    confidence_label: str  # "high", "medium", "low", "very_low"
    flags: list[ValidationFlag] = field(default_factory=list)
    join_validation: dict = field(default_factory=dict)
    row_count_check: dict = field(default_factory=dict)
    summary_check: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "confidence_score": self.confidence_score,
            "confidence_label": self.confidence_label,
            "flags": [
                {"severity": f.severity, "category": f.category, "message": f.message, "detail": f.detail}
                for f in self.flags
            ],
            "join_validation": self.join_validation,
            "row_count_check": self.row_count_check,
            "summary_check": self.summary_check,
            "provenance": self.provenance,
        }


# ── SQL parsing helpers ───────────────────────────────────────────────────────

def extract_tables(sql: str) -> list[str]:
    """Extract table names from a SQL query (FROM and JOIN clauses)."""
    # Match: FROM table [alias], JOIN table [alias]
    # Capture the table name (first word after FROM/JOIN), skip the alias
    patterns = [
        r'\bFROM\s+(\w+)(?:\s+(?:AS\s+)?\w+)?',
        r'\bJOIN\s+(\w+)(?:\s+(?:AS\s+)?\w+)?',
        r'\bINNER\s+JOIN\s+(\w+)(?:\s+(?:AS\s+)?\w+)?',
        r'\bLEFT\s+JOIN\s+(\w+)(?:\s+(?:AS\s+)?\w+)?',
        r'\bRIGHT\s+JOIN\s+(\w+)(?:\s+(?:AS\s+)?\w+)?',
        r'\bLEFT\s+OUTER\s+JOIN\s+(\w+)(?:\s+(?:AS\s+)?\w+)?',
        r'\bRIGHT\s+OUTER\s+JOIN\s+(\w+)(?:\s+(?:AS\s+)?\w+)?',
        r'\bFULL\s+OUTER\s+JOIN\s+(\w+)(?:\s+(?:AS\s+)?\w+)?',
        r'\bCROSS\s+JOIN\s+(\w+)(?:\s+(?:AS\s+)?\w+)?',
    ]
    tables = set()
    for pattern in patterns:
        for m in re.finditer(pattern, sql, re.IGNORECASE):
            word = m.group(1)
            # Skip SQL keywords that might be captured
            if word.upper() not in ("SELECT", "WHERE", "GROUP", "ORDER", "HAVING", "LIMIT",
                                     "AS", "ON", "AND", "OR", "LEFT", "RIGHT", "INNER", "JOIN",
                                     "OUTER", "FULL", "CROSS", "SET", "VALUES", "INTO"):
                tables.add(word)
    return list(tables)


def extract_join_conditions(sql: str) -> list[dict]:
    """Extract JOIN ON conditions to understand which columns are being joined."""
    # Match: ON table1.col1 = table2.col2 (with optional aliases)
    pattern = r'\bON\s+(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)'
    matches = re.findall(pattern, sql, re.IGNORECASE)
    joins = []
    for m in matches:
        joins.append({
            "left_alias": m[0],
            "left_col": m[1],
            "right_alias": m[2],
            "right_col": m[3],
        })
    return joins


def extract_join_types(sql: str) -> list[dict]:
    """Extract the type of each JOIN (INNER, LEFT, etc.)."""
    joins = []
    # Match JOIN clauses with their type
    pattern = r'((?:INNER|LEFT|RIGHT|FULL|CROSS)?\s*(?:OUTER)?\s*JOIN)\s+(\w+)(?:\s+(?:AS\s+)?(\w+))?'
    for m in re.finditer(pattern, sql, re.IGNORECASE):
        join_type = m.group(1).strip().upper() if m.group(1) else "JOIN"
        table = m.group(2)
        alias = m.group(3) if m.group(3) else table
        joins.append({"type": join_type, "table": table, "alias": alias})
    return joins


def extract_aliases(sql: str) -> dict[str, str]:
    """Extract table aliases (e.g., 'InvoiceLine il' -> {'il': 'InvoiceLine'})."""
    aliases = {}
    # Match: FROM table alias, JOIN table alias
    pattern = r'(?:FROM|JOIN)\s+(\w+)(?:\s+(?:AS\s+)?(\w+))?'
    for m in re.finditer(pattern, sql, re.IGNORECASE):
        table = m.group(1)
        alias = m.group(2) if m.group(2) else table
        if alias.upper() not in ("WHERE", "GROUP", "ORDER", "HAVING", "LIMIT", "ON", "LEFT", "RIGHT", "INNER", "JOIN"):
            aliases[alias] = table
    return aliases


# ── Schema FK graph ───────────────────────────────────────────────────────────

def build_fk_graph(conn: sqlite3.Connection) -> dict[str, list[dict]]:
    """
    Build a foreign key graph from the database schema.

    Uses PRAGMA foreign_key_list for reliable FK extraction, falling back
    to DDL parsing for databases that don't support the pragma.

    Returns:
        Dict mapping table_name -> list of {column, references_table, references_column}
    """
    fk_graph = {}

    # Get all table names
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
    table_names = [row[0] for row in cur.fetchall()]

    for table_name in table_names:
        fk_graph[table_name] = []

        # Method 1: PRAGMA foreign_key_list (most reliable for SQLite)
        try:
            cur.execute(f"PRAGMA foreign_key_list([{table_name}])")
            fk_rows = cur.fetchall()
            for fk_row in fk_rows:
                # PRAGMA foreign_key_list columns: id, seq, table, from, to, on_update, on_delete, match
                fk_graph[table_name].append({
                    "column": fk_row[3],           # "from" column
                    "references_table": fk_row[2],  # referenced table
                    "references_column": fk_row[4], # "to" column
                })
        except sqlite3.Error:
            pass

        # Method 2: DDL parsing fallback (for non-SQLite or if pragma returns empty)
        if not fk_graph[table_name]:
            cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name = ?", (table_name,))
            row = cur.fetchone()
            ddl = row[0] if row else ""
            # Parse FOREIGN KEY clauses with bracket-quoted or unquoted identifiers
            fk_matches = re.findall(
                r'FOREIGN\s+KEY\s+\[?(\w+)\]?\s+REFERENCES\s+\[?(\w+)\]?\s+\[?(\w+)\]?',
                ddl, re.IGNORECASE
            )
            for col, ref_table, ref_col in fk_matches:
                fk_graph[table_name].append({
                    "column": col,
                    "references_table": ref_table,
                    "references_column": ref_col,
                })

    return fk_graph


def get_valid_join_pairs(fk_graph: dict[str, list[dict]]) -> set[tuple[str, str]]:
    """Get all valid table pairs that can be joined via a foreign key."""
    pairs = set()
    for table, fks in fk_graph.items():
        for fk in fks:
            pair = tuple(sorted([table, fk["references_table"]]))
            pairs.add(pair)
    return pairs


def get_join_columns(fk_graph: dict[str, list[dict]], table_a: str, table_b: str) -> list[tuple[str, str]]:
    """Get the valid join columns between two tables."""
    joins = []
    for table, fks in fk_graph.items():
        for fk in fks:
            if (table == table_a and fk["references_table"] == table_b) or \
               (table == table_b and fk["references_table"] == table_a):
                if table == table_a:
                    joins.append((fk["column"], fk["references_column"]))
                else:
                    joins.append((fk["references_column"], fk["column"]))
    return joins


# ── Question type detection ───────────────────────────────────────────────────

def detect_question_type(question: str) -> str:
    """Detect the type of question to set expected row count ranges."""
    q = question.lower()

    # Aggregation with "top N" — expect N rows
    if re.search(r'top\s+\d+', q) or re.search(r'best\s+\d+', q):
        n_match = re.search(r'(?:top|best)\s+(\d+)', q)
        n = int(n_match.group(1)) if n_match else 5
        return f"top_n:{n}"

    # "Each" / "every" — expect multiple rows (one per entity)
    # Check this BEFORE count/total, since "how many X does each Y" is per_entity
    if re.search(r'\beach\b', q) or re.search(r'\bevery\b', q):
        return "per_entity"

    # Count questions — expect 1 row
    if re.search(r'\bhow many\b', q) or re.search(r'\bcount\b', q):
        return "count"

    # Total/sum questions — expect 1 row
    if re.search(r'\btotal\b', q) or re.search(r'\bsum\b', q) or re.search(r'\brevenue\b', q):
        return "single_aggregate"

    # "List all" — expect multiple rows
    if re.search(r'\blist all\b', q) or re.search(r'\bshow all\b', q):
        return "list_all"

    # "Which" / "who" — expect 1 or few rows
    if re.search(r'\bwhich\b', q) or re.search(r'\bwho\b', q):
        if re.search(r'\bmost\b', q) or re.search(r'\bhighest\b', q) or re.search(r'\blargest\b', q):
            return "single_result"
        return "lookup"

    # Average — expect 1 row or per-entity
    if re.search(r'\baverage\b', q) or re.search(r'\bavg\b', q):
        if re.search(r'\beach\b', q) or re.search(r'\bcountry\b', q):
            return "per_entity"
        return "single_aggregate"

    # Ranking — expect N rows
    if re.search(r'\brank\b', q) or re.search(r'\branking\b', q):
        return "ranking"

    # Default
    return "unknown"


def expected_row_range(question_type: str, question: str) -> tuple[int, int]:
    """Return (min_expected, max_expected) row count for a question type."""
    if question_type.startswith("top_n:"):
        n = int(question_type.split(":")[1])
        return (1, n)
    if question_type == "count":
        return (1, 1)
    if question_type == "single_aggregate":
        return (1, 1)
    if question_type == "single_result":
        return (1, 3)
    if question_type == "lookup":
        return (1, 50)
    if question_type == "per_entity":
        return (2, 500)
    if question_type == "list_all":
        return (1, 1000)
    if question_type == "ranking":
        return (1, 20)
    return (0, 10000)  # unknown — wide range


# ── Summary verification ──────────────────────────────────────────────────────

def verify_summary(summary: str, results: list[dict], columns: list[str]) -> dict:
    """
    Cross-check LLM summary claims against actual result data.

    Checks:
    - Does the summary mention the correct number of results?
    - Does the summary mention values that appear in the results?
    - Does the summary claim things not supported by the data?
    """
    checks = {"passed": [], "failed": [], "warnings": []}

    if not summary or not results:
        return checks

    summary_lower = summary.lower()

    # Check 1: Does the summary mention the right count?
    # Look for "N rows" or "N results" or "N genres" etc.
    count_mentions = re.findall(r'(\d+)\s+(?:rows?|results?|items?|entries|genres?|tracks?|albums?|customers?|artists?|playlists?)', summary_lower)
    if count_mentions:
        mentioned_count = int(count_mentions[0])
        actual_count = len(results)
        if mentioned_count == actual_count:
            checks["passed"].append(f"Summary count matches: {actual_count} results")
        elif abs(mentioned_count - actual_count) <= 1:
            checks["warnings"].append(f"Summary count slightly off: says {mentioned_count}, actual {actual_count}")
        else:
            checks["failed"].append(f"Summary count mismatch: says {mentioned_count}, actual {actual_count}")

    # Check 2: Do the values mentioned in the summary appear in the results?
    # Extract potential values from results (strings and numbers)
    result_values = set()
    for row in results[:10]:
        for col, val in row.items():
            if val is not None:
                result_values.add(str(val).lower().strip())
                if isinstance(val, (int, float)):
                    result_values.add(str(round(float(val), 2)))

    # Extract values mentioned in summary (numbers and quoted strings)
    summary_numbers = re.findall(r'\$?(\d+\.?\d*)', summary)
    mentioned_values = set()
    for num in summary_numbers:
        try:
            mentioned_values.add(str(round(float(num), 2)))
        except ValueError:
            pass

    # Check if mentioned numbers appear in results
    matched = 0
    unmatched = 0
    for val in mentioned_values:
        if val in result_values:
            matched += 1
        else:
            # Check if it's close to any result value
            try:
                val_f = float(val)
                close = any(abs(val_f - float(rv)) < 0.1 for rv in result_values if rv.replace('.', '').replace('-', '').isdigit())
                if close:
                    matched += 1
                else:
                    unmatched += 1
            except ValueError:
                unmatched += 1

    if matched > 0 and unmatched == 0:
        checks["passed"].append(f"All {matched} values mentioned in summary found in results")
    elif unmatched > 0 and matched > 0:
        checks["warnings"].append(f"Summary mentions {unmatched} value(s) not found in results (may be derived)")
    elif unmatched > 0 and matched == 0:
        checks["failed"].append(f"Summary mentions {unmatched} value(s) not found in results")

    # Check 3: Does the summary mention column names from the results?
    col_mentions = 0
    for col in columns:
        if col.lower() in summary_lower:
            col_mentions += 1
    if col_mentions > 0:
        checks["passed"].append(f"Summary references {col_mentions}/{len(columns)} result columns")
    elif columns:
        checks["warnings"].append("Summary doesn't mention any result column names")

    return checks


# ── Main trust layer ──────────────────────────────────────────────────────────

class TrustLayer:
    """
    Analyzes agent responses for trust signals.

    Combines SQL validation, row count sanity, summary verification,
    and provenance tracking into a confidence score.
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.fk_graph = build_fk_graph(conn)
        self.valid_pairs = get_valid_join_pairs(self.fk_graph)
        self.all_tables = set(self.fk_graph.keys())

    def analyze(
        self,
        question: str,
        sql: str,
        results: list[dict],
        columns: list[str],
        summary: str,
        attempts: int,
        few_shot_patterns: list[str] = None,
    ) -> TrustReport:
        """Run all trust checks and return a TrustReport."""
        flags: list[ValidationFlag] = []
        score = 100  # Start at 100 and deduct for issues

        # ── 1. SQL / JOIN validation ──
        join_validation = self._validate_joins(sql, flags)
        if join_validation["invalid_joins"]:
            score -= 30  # Wrong join path is critical
        if join_validation["unvalidated_joins"]:
            score -= 10  # Joins we can't verify are suspicious
        if join_validation["unknown_tables"]:
            score -= 15  # Tables not in schema

        # ── 2. Row count sanity ──
        row_count_check = self._check_row_count(question, results, flags)
        if row_count_check["status"] == "warning":
            score -= 10
        elif row_count_check["status"] == "critical":
            score -= 20

        # ── 3. Self-healing attempts ──
        if attempts > 1:
            deduction = (attempts - 1) * 10
            score -= deduction
            flags.append(ValidationFlag(
                severity="warning",
                category="self_healing",
                message=f"SQL required {attempts} attempts (self-healing triggered {attempts - 1} time(s))",
                detail=f"Confidence reduced by {deduction} points due to retries",
            ))
        elif attempts == 1:
            flags.append(ValidationFlag(
                severity="ok",
                category="self_healing",
                message="SQL succeeded on first attempt (no self-healing needed)",
            ))

        # ── 4. Summary verification ──
        summary_check = verify_summary(summary, results, columns)
        summary_issues = len(summary_check.get("failed", []))
        summary_warnings = len(summary_check.get("warnings", []))
        if summary_issues > 0:
            score -= 15
            for fail in summary_check["failed"]:
                flags.append(ValidationFlag(
                    severity="critical",
                    category="summary",
                    message=f"Summary verification failed: {fail}",
                ))
        if summary_warnings > 0:
            score -= 5
            for warn in summary_check["warnings"]:
                flags.append(ValidationFlag(
                    severity="warning",
                    category="summary",
                    message=f"Summary warning: {warn}",
                ))
        if summary_check.get("passed"):
            flags.append(ValidationFlag(
                severity="ok",
                category="summary",
                message=f"Summary verified: {len(summary_check['passed'])} check(s) passed",
            ))

        # ── 5. Provenance tracking ──
        provenance = self._track_provenance(sql, few_shot_patterns, flags)
        if provenance["matched_patterns"]:
            score += 5  # Bonus for matching known-good patterns (cap at 100)
        if provenance["deviated_from_pattern"]:
            score -= 10

        # Clamp score
        score = max(0, min(100, score))

        # Determine label
        if score >= 85:
            label = "high"
        elif score >= 65:
            label = "medium"
        elif score >= 40:
            label = "low"
        else:
            label = "very_low"

        return TrustReport(
            confidence_score=score,
            confidence_label=label,
            flags=flags,
            join_validation=join_validation,
            row_count_check=row_count_check,
            summary_check=summary_check,
            provenance=provenance,
        )

    def _validate_joins(self, sql: str, flags: list[ValidationFlag]) -> dict:
        """Validate that JOIN paths in the SQL follow the schema's FK graph."""
        result = {
            "tables_found": [],
            "joins_found": [],
            "invalid_joins": [],
            "unvalidated_joins": [],
            "unknown_tables": [],
            "join_types": [],
        }

        if not sql:
            return result

        # Extract tables and aliases
        tables = extract_tables(sql)
        aliases = extract_aliases(sql)
        join_conditions = extract_join_conditions(sql)
        join_types = extract_join_types(sql)

        result["tables_found"] = tables
        result["joins_found"] = join_conditions
        result["join_types"] = [{"type": j["type"], "table": j["table"]} for j in join_types]

        # Check for unknown tables
        for t in tables:
            if t not in self.all_tables:
                result["unknown_tables"].append(t)
                flags.append(ValidationFlag(
                    severity="critical",
                    category="join_path",
                    message=f"Table '{t}' not found in database schema",
                ))

        # Validate each JOIN condition against the FK graph
        for jc in join_conditions:
            left_table = aliases.get(jc["left_alias"], jc["left_alias"])
            right_table = aliases.get(jc["right_alias"], jc["right_alias"])

            # Skip if either table is unknown
            if left_table not in self.all_tables or right_table not in self.all_tables:
                result["unvalidated_joins"].append({
                    "left": f"{left_table}.{jc['left_col']}",
                    "right": f"{right_table}.{jc['right_col']}",
                    "reason": "unknown_table",
                })
                continue

            # Check if this join matches a known FK relationship
            valid_cols = get_join_columns(self.fk_graph, left_table, right_table)
            pair = tuple(sorted([left_table, right_table]))

            if pair in self.valid_pairs:
                # Check if the join columns match the FK columns
                col_match = any(
                    (jc["left_col"] == lc and jc["right_col"] == rc) or
                    (jc["left_col"] == rc and jc["right_col"] == lc)
                    for lc, rc in valid_cols
                )
                if col_match:
                    flags.append(ValidationFlag(
                        severity="ok",
                        category="join_path",
                        message=f"JOIN {left_table}.{jc['left_col']} = {right_table}.{jc['right_col']} follows FK relationship",
                    ))
                else:
                    result["invalid_joins"].append({
                        "left": f"{left_table}.{jc['left_col']}",
                        "right": f"{right_table}.{jc['right_col']}",
                        "valid_columns": [f"{lc}={rc}" for lc, rc in valid_cols],
                        "reason": "wrong_join_column",
                    })
                    flags.append(ValidationFlag(
                        severity="critical",
                        category="join_path",
                        message=f"JOIN {left_table}.{jc['left_col']} = {right_table}.{jc['right_col']} uses wrong columns",
                        detail=f"Valid join columns: {', '.join(f'{lc}={rc}' for lc, rc in valid_cols)}",
                    ))
            else:
                # Tables are not directly connected by FK — could be a multi-hop join or wrong path
                result["unvalidated_joins"].append({
                    "left": f"{left_table}.{jc['left_col']}",
                    "right": f"{right_table}.{jc['right_col']}",
                    "reason": "no_direct_fk",
                })
                flags.append(ValidationFlag(
                    severity="warning",
                    category="join_path",
                    message=f"JOIN {left_table} <-> {right_table}: no direct FK relationship",
                    detail="These tables are not directly connected by a foreign key. Verify the join path manually.",
                ))

        # Check for INNER JOIN where LEFT JOIN might be needed
        # (heuristic: if question contains "each" or "every", LEFT JOIN is expected)
        for jt in join_types:
            if "LEFT" not in jt["type"].upper() and "OUTER" not in jt["type"].upper():
                result["join_types"].append({"type": jt["type"], "table": jt["table"]})

        return result

    def _check_row_count(self, question: str, results: list[dict], flags: list[ValidationFlag]) -> dict:
        """Check if the result row count is reasonable for the question type."""
        q_type = detect_question_type(question)
        min_expected, max_expected = expected_row_range(q_type, question)
        actual = len(results)

        result = {
            "question_type": q_type,
            "expected_range": [min_expected, max_expected],
            "actual_count": actual,
            "status": "ok",
        }

        if actual == 0:
            result["status"] = "critical"
            flags.append(ValidationFlag(
                severity="critical",
                category="row_count",
                message="Query returned 0 rows — may indicate wrong table, wrong filter, or empty result",
            ))
        elif actual < min_expected:
            result["status"] = "warning"
            flags.append(ValidationFlag(
                severity="warning",
                category="row_count",
                message=f"Expected at least {min_expected} rows, got {actual} (question type: {q_type})",
                detail="Fewer rows than expected may indicate an overly restrictive filter or wrong join",
            ))
        elif actual > max_expected:
            result["status"] = "warning"
            flags.append(ValidationFlag(
                severity="warning",
                category="row_count",
                message=f"Expected at most {max_expected} rows, got {actual} (question type: {q_type})",
                detail="More rows than expected may indicate a missing GROUP BY, wrong join cardinality, or missing LIMIT",
            ))
        else:
            flags.append(ValidationFlag(
                severity="ok",
                category="row_count",
                message=f"Row count {actual} within expected range [{min_expected}, {max_expected}] for {q_type} questions",
            ))

        return result

    def _track_provenance(
        self,
        sql: str,
        few_shot_patterns: list[str] = None,
        flags: list[ValidationFlag] = None,
    ) -> dict:
        """Track which few-shot patterns influenced this query and whether it deviated."""
        from src.few_shot import FEW_SHOT_EXAMPLES

        provenance = {
            "matched_patterns": [],
            "matched_examples": [],
            "deviated_from_pattern": False,
            "pattern_similarity": {},
        }

        if not sql:
            return provenance

        sql_lower = sql.lower()

        # Check each few-shot example for similarity
        for ex in FEW_SHOT_EXAMPLES:
            ex_sql = ex["sql"].lower()

            # Simple pattern matching: check if key SQL fragments match
            # Extract key structural elements: tables joined, aggregation functions, GROUP BY columns
            ex_tables = set(extract_tables(ex_sql))
            sql_tables = set(extract_tables(sql_lower))

            table_overlap = ex_tables & sql_tables
            if len(table_overlap) >= 2:
                # Significant table overlap — this pattern likely influenced the query
                similarity = len(table_overlap) / max(len(ex_tables | sql_tables), 1)
                provenance["matched_patterns"].append(ex["pattern"])
                provenance["matched_examples"].append({
                    "pattern": ex["pattern"],
                    "question": ex["question"],
                    "similarity": round(similarity, 3),
                    "shared_tables": list(table_overlap),
                })
                provenance["pattern_similarity"][ex["pattern"]] = round(similarity, 3)

                # Check for deviation: same tables but different join type or missing key clause
                ex_has_left_join = "left join" in ex_sql
                sql_has_left_join = "left join" in sql_lower
                if ex_has_left_join and not sql_has_left_join:
                    provenance["deviated_from_pattern"] = True
                    if flags:
                        flags.append(ValidationFlag(
                            severity="warning",
                            category="provenance",
                            message=f"Query deviates from '{ex['pattern']}' pattern: example uses LEFT JOIN but generated SQL doesn't",
                            detail=f"Example question: {ex['question']}",
                        ))

                # Check if aggregation function matches
                ex_agg = re.findall(r'\b(SUM|COUNT|AVG|MIN|MAX)\s*\(', ex_sql)
                sql_agg = re.findall(r'\b(SUM|COUNT|AVG|MIN|MAX)\s*\(', sql_lower)
                if set(ex_agg) != set(sql_agg) and ex_agg:
                    if flags:
                        flags.append(ValidationFlag(
                            severity="info",
                            category="provenance",
                            message=f"Aggregation differs from '{ex['pattern']}' pattern: example uses {ex_agg}, SQL uses {sql_agg}",
                        ))

        if provenance["matched_patterns"] and flags:
            flags.append(ValidationFlag(
                severity="ok",
                category="provenance",
                message=f"Query matches {len(provenance['matched_patterns'])} known-good pattern(s): {', '.join(provenance['matched_patterns'])}",
            ))
        elif not provenance["matched_patterns"] and flags:
            flags.append(ValidationFlag(
                severity="info",
                category="provenance",
                message="No matching few-shot pattern found — this is a novel query type",
            ))

        return provenance
