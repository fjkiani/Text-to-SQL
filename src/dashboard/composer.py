"""
Dashboard composer: turn a question + result set into a multi-panel dashboard.

Reuses and extends the chart-detection heuristics from the web layer
(`_detect_chart_type`) to auto-pick the right visualization per panel, then
composes a dashboard document that the UI renders and that persists to R2
under {tenant}/dashboards/{dashboard_id}.json.

No stub panels: every panel carries real data derived from the query results.
"""
import json
import os
import time
import uuid
from typing import Optional


# ── Chart-type detection (extends web._detect_chart_type) ────────────────────

def _col_types(results: list, columns: list) -> dict:
    """Classify each column as numeric or categorical from the actual values."""
    types = {}
    for col in columns:
        values = [r.get(col) for r in results if r.get(col) is not None]
        if not values:
            types[col] = "categorical"
            continue
        numeric = sum(1 for v in values if isinstance(v, (int, float)) and not isinstance(v, bool))
        types[col] = "numeric" if numeric > len(values) * 0.7 else "categorical"
    return types


def _is_temporal(values: list) -> bool:
    """Heuristic: a categorical column looks temporal if values parse as dates/years."""
    import re
    sample = [str(v) for v in values[:10] if v is not None]
    if not sample:
        return False
    pat = re.compile(r"^\d{4}(-\d{2})?(-\d{2})?([ T]\d{2}:\d{2})?")
    return sum(1 for s in sample if pat.match(s)) >= len(sample) * 0.7


def detect_panel(question: str, results: list, columns: list) -> Optional[dict]:
    """
    Pick the best single panel for a result set.

    Returns a panel dict {type, title, chart, data} or None if not chartable.
    Types: 'bar', 'line', 'table', 'metric'.
    """
    if not results or not columns:
        return None

    types = _col_types(results, columns)
    numeric = [c for c in columns if types[c] == "numeric"]
    categorical = [c for c in columns if types[c] == "categorical"]

    # Single-row, single-numeric -> big metric panel
    if len(results) == 1 and len(numeric) >= 1 and len(columns) <= 2:
        val_col = numeric[0]
        return {
            "type": "metric",
            "title": val_col,
            "chart": "metric",
            "data": {"value": results[0].get(val_col), "label": val_col},
        }

    # 1 categorical + 1 numeric -> bar (or line if temporal)
    if len(categorical) == 1 and len(numeric) == 1 and len(results) >= 2:
        label_col, value_col = categorical[0], numeric[0]
        labels = [str(r.get(label_col, "")) for r in results]
        values = [round(v, 2) if isinstance(v, float) else v for v in (r.get(value_col, 0) for r in results)]
        chart = "line" if _is_temporal(labels) else "bar"
        return {
            "type": chart,
            "title": f"{value_col} by {label_col}",
            "chart": chart,
            "data": {"labels": labels, "values": values, "label_col": label_col, "value_col": value_col},
        }

    # 1 categorical + >=2 numeric -> grouped bar (first two numerics)
    if len(categorical) == 1 and len(numeric) >= 2 and len(results) >= 2:
        label_col = categorical[0]
        series = numeric[:2]
        labels = [str(r.get(label_col, "")) for r in results]
        datasets = [
            {"name": s, "values": [round(r.get(s, 0), 2) if isinstance(r.get(s), float) else r.get(s, 0) for r in results]}
            for s in series
        ]
        return {
            "type": "bar",
            "title": f"{' & '.join(series)} by {label_col}",
            "chart": "bar",
            "data": {"labels": labels, "datasets": datasets, "label_col": label_col, "series": series},
        }

    # Fallback: table panel (always real data)
    return {
        "type": "table",
        "title": "Results",
        "chart": "table",
        "data": {"columns": columns, "rows": results[:100], "row_count": len(results)},
    }


def compose(question: str, results: list, columns: list, title: Optional[str] = None) -> dict:
    """
    Compose a dashboard from a question + result set.

    Returns a dashboard document:
      {id, title, question, created_at, panels:[...], source:{columns, row_count}}
    Every panel carries real data. Raises if results are empty (loud, no fake panels).
    """
    if not results:
        raise ValueError("compose() requires non-empty results — no fake panels")

    panel = detect_panel(question, results, columns)
    panels = [panel] if panel else []

    # Always include the raw table as a second panel when the primary is a chart,
    # so the underlying numbers are visible alongside the visualization.
    if panel and panel["type"] in ("bar", "line", "metric"):
        panels.append({
            "type": "table",
            "title": "Underlying data",
            "chart": "table",
            "data": {"columns": columns, "rows": results[:100], "row_count": len(results)},
        })

    return {
        "id": str(uuid.uuid4())[:12],
        "title": title or (question[:80] + ("…" if len(question) > 80 else "")),
        "question": question,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "panels": panels,
        "source": {"columns": columns, "row_count": len(results)},
    }


# ── R2 persistence ────────────────────────────────────────────────────────────

def _r2_client():
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def save_dashboard(dashboard: dict, tenant: str = "default", bucket: Optional[str] = None) -> dict:
    """Persist a dashboard document to R2 under {tenant}/dashboards/{id}.json."""
    bucket = bucket or os.environ.get("R2_BUCKET", "tessera-embeddings")
    key = f"{tenant}/dashboards/{dashboard['id']}.json"
    s3 = _r2_client()
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(dashboard, default=str).encode())
    return {"key": key, "id": dashboard["id"], "tenant": tenant}


def load_dashboard(dashboard_id: str, tenant: str = "default", bucket: Optional[str] = None) -> Optional[dict]:
    """Load a dashboard document from R2. Returns None if not found."""
    bucket = bucket or os.environ.get("R2_BUCKET", "tessera-embeddings")
    key = f"{tenant}/dashboards/{dashboard_id}.json"
    s3 = _r2_client()
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except Exception:
        return None


def list_dashboards(tenant: str = "default", bucket: Optional[str] = None) -> list[dict]:
    """List all dashboards for a tenant (id, title, created_at) from R2."""
    bucket = bucket or os.environ.get("R2_BUCKET", "tessera-embeddings")
    prefix = f"{tenant}/dashboards/"
    s3 = _r2_client()
    out = []
    try:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        for item in resp.get("Contents", []):
            obj = s3.get_object(Bucket=bucket, Key=item["Key"])
            d = json.loads(obj["Body"].read())
            out.append({"id": d.get("id"), "title": d.get("title"), "created_at": d.get("created_at"),
                        "panel_count": len(d.get("panels", []))})
    except Exception:
        pass
    return sorted(out, key=lambda x: x.get("created_at") or "", reverse=True)


if __name__ == "__main__":
    # smoke test with real-shaped data
    cols = ["Genre", "TotalSales"]
    rows = [{"Genre": "Rock", "TotalSales": 826.65}, {"Genre": "Latin", "TotalSales": 382.14},
            {"Genre": "Metal", "TotalSales": 261.36}, {"Genre": "Jazz", "TotalSales": 79.20}]
    dash = compose("Top genres by sales", rows, cols)
    print("panels:", [p["type"] for p in dash["panels"]])
    print("title:", dash["title"])
    metric = compose("Total revenue", [{"TotalRevenue": 2328.60}], ["TotalRevenue"])
    print("metric panel:", metric["panels"][0]["type"], metric["panels"][0]["data"])
