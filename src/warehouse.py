"""
Warehouse-on-the-fly: turn uploaded structured files (CSV, etc.) into queryable
DuckDB tables, per tenant. The text-to-SQL agent can then query user data, not
just the static Chinook DB.

Each tenant gets its own DuckDB file under a shared warehouse dir. Tables are
registered by name and introspectable so the agent's schema graph can include them.
"""
import os
import re
import threading

_lock = threading.Lock()
WAREHOUSE_DIR = os.environ.get("WAREHOUSE_DIR", "/workspace/tessera_warehouse")


def _warehouse_path(tenant: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", tenant)
    os.makedirs(WAREHOUSE_DIR, exist_ok=True)
    return os.path.join(WAREHOUSE_DIR, f"{safe}.duckdb")


def get_warehouse_conn(tenant: str = "default"):
    """Return a DuckDB connection for the tenant's warehouse."""
    import duckdb
    return duckdb.connect(_warehouse_path(tenant))


def _infer_table_name(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    name = re.sub(r"[^A-Za-z0-9_]", "_", base)
    if name and name[0].isdigit():
        name = "t_" + name
    return name or "uploaded_table"


def register_csv(tenant: str, path: str, table_name: str = None) -> dict:
    """
    Load a CSV into the tenant's DuckDB warehouse as a table.
    Returns {table_name, row_count, columns:[{name,type}]}.
    Raises on parse failure (loud, no silent empty tables).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    table = table_name or _infer_table_name(path)
    with _lock:
        conn = get_warehouse_conn(tenant)
        try:
            conn.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_csv_auto(?)", [path])
            row_count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            cols = conn.execute(f"DESCRIBE {table}").fetchall()
        finally:
            conn.close()
    if row_count == 0:
        raise ValueError(f"CSV {path} produced 0 rows in table {table}")
    return {
        "table_name": table,
        "row_count": row_count,
        "columns": [{"name": c[0], "type": c[1]} for c in cols],
        "tenant": tenant,
    }


def list_tables(tenant: str = "default") -> list[dict]:
    """List all tables in the tenant's warehouse with row counts."""
    if not os.path.exists(_warehouse_path(tenant)):
        return []
    conn = get_warehouse_conn(tenant)
    try:
        tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
        out = []
        for t in tables:
            n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            cols = conn.execute(f"DESCRIBE {t}").fetchall()
            out.append({"table_name": t, "row_count": n,
                        "columns": [{"name": c[0], "type": c[1]} for c in cols]})
        return out
    finally:
        conn.close()


def query(tenant: str, sql: str) -> dict:
    """Run SQL against the tenant's warehouse. Returns {columns, rows, row_count}."""
    conn = get_warehouse_conn(tenant)
    try:
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        return {"columns": cols, "rows": rows, "row_count": len(rows)}
    finally:
        conn.close()


if __name__ == "__main__":
    # smoke test with a real CSV
    import csv
    with open("/tmp/sales.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region", "product", "revenue", "units"])
        w.writerow(["West", "Widget", 1200.50, 30])
        w.writerow(["East", "Widget", 980.00, 22])
        w.writerow(["West", "Gadget", 2100.75, 15])
        w.writerow(["East", "Gadget", 1750.20, 12])
    info = register_csv("acme", "/tmp/sales.csv")
    print("registered:", info["table_name"], info["row_count"], "rows")
    print("columns:", [(c["name"], c["type"]) for c in info["columns"]])
    res = query("acme", "SELECT region, SUM(revenue) as total FROM sales GROUP BY region ORDER BY total DESC")
    print("query result:", res["rows"])
    print("tables:", [t["table_name"] for t in list_tables("acme")])
