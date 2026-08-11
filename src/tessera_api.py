"""
Tessera platform API — the SaaS surface that ties the subsystems together.

Routes (all real, no stubs):
  POST /tessera/upload/doc       -> Unstructured parse + chunk + Arctic-embed + FAISS index (+R2 persist)
  POST /tessera/upload/csv       -> register CSV into the tenant's DuckDB warehouse
  GET  /tessera/warehouse/tables -> list the tenant's warehouse tables
  POST /tessera/ask              -> route: unstructured -> retrieval answer w/ citations;
                                    structured  -> text-to-SQL over warehouse (+ optional dashboard)
  POST /tessera/dashboard        -> compose + persist a dashboard from a question's results
  GET  /tessera/dashboards       -> list the tenant's dashboards
  GET  /tessera/dashboard/{id}   -> load a persisted dashboard

Tenancy: X-API-Key header (see src/auth.py). Every tenant's vectors, dashboards,
and warehouse are namespaced under their own key.
"""
import os
import shutil
import tempfile
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.auth import resolve_tenant
from src.ingest.service import ingest_file
from src.vector.store import VectorStore
from src.warehouse import register_csv, list_tables, query as warehouse_query
from src.retrieval import answer_with_retrieval
from src.dashboard import composer
from src.llm.client import chat

router = APIRouter(prefix="/tessera", tags=["tessera"])

# Uploads larger than this are rejected (bounded, not unlimited).
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class AskRequest(BaseModel):
    question: str
    mode: str = "auto"            # auto | retrieval | sql
    make_dashboard: bool = False
    k: int = 5


class DashboardRequest(BaseModel):
    question: str
    sql: str
    title: Optional[str] = None


def _save_upload(upload: UploadFile) -> str:
    """Persist an UploadFile to a temp path with its real extension. Returns path."""
    suffix = os.path.splitext(upload.filename or "upload")[1]
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    size = 0
    with os.fdopen(fd, "wb") as f:
        while True:
            block = upload.file.read(1024 * 1024)
            if not block:
                break
            size += len(block)
            if size > MAX_UPLOAD_BYTES:
                f.close()
                os.unlink(tmp)
                raise HTTPException(status_code=413, detail=f"Upload exceeds {MAX_UPLOAD_BYTES // (1024*1024)}MB limit")
            f.write(block)
    return tmp


# ── Document ingestion -> vector index ────────────────────────────────────────

@router.post("/upload/doc")
async def upload_doc(file: UploadFile = File(...), tenant: str = Depends(resolve_tenant)):
    """Parse + chunk + embed + index an unstructured document for this tenant."""
    tmp = _save_upload(file)
    try:
        ingested = ingest_file(tmp, tenant)
        vs = VectorStore(tenant=tenant)
        vs.load()  # merge with any existing persisted index
        added = vs.add(ingested["chunks"])
        saved = vs.save()
        return {
            "doc_id": ingested["doc_id"],
            "source": ingested["source"],
            "chunk_count": ingested["chunk_count"],
            "vectors_added": added,
            "index_count": vs.count(),
            "persisted": saved,
            "tenant": tenant,
        }
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ── CSV -> warehouse-on-the-fly ───────────────────────────────────────────────

@router.post("/upload/csv")
async def upload_csv(file: UploadFile = File(...), table_name: Optional[str] = None,
                     tenant: str = Depends(resolve_tenant)):
    """Register an uploaded CSV as a queryable DuckDB table for this tenant."""
    tmp = _save_upload(file)
    try:
        info = register_csv(tenant, tmp, table_name)
        return info
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


@router.get("/warehouse/tables")
async def warehouse_tables(tenant: str = Depends(resolve_tenant)):
    """List the tenant's warehouse tables with row counts and columns."""
    return {"tenant": tenant, "tables": list_tables(tenant)}


@router.get("/embed-health")
async def embed_health(probe: bool = False):
    """
    Which embedding backend is ACTUALLY serving, not which one is configured.

    ARCTIC_EMBED_URL still points at the Modal tessera-arctic-embed endpoint.
    `modal app list` reports that app "deployed" while the URL returns 404,
    because the workspace is over its spend limit -- so the configured value
    tells you nothing. This reports the latched failover state instead, and
    with ?probe=true actually embeds a token and reports the width it got
    back. Same lesson as the clearance bit: probe behaviour, not control
    plane. Unauthenticated on purpose: it exposes no tenant data.
    """
    from src.vector import store as vstore

    out = {
        "configured_arctic_url": vstore._ARCTIC_EMBED_URL or None,
        "arctic_latched_dead": vstore._remote_dead,
        "gemini_keys": len(vstore._gemini_keys()),
        "gemini_latched_dead": vstore._gemini_dead,
        "gemini_model": vstore._GEMINI_EMBED_MODEL,
        "gemini_dim": vstore._GEMINI_EMBED_DIM,
        "local_model": vstore._MODEL_NAME,
        "active_backend": vstore.embed_backend(),
        "space": vstore.embed_space(),
        "embed_calls": vstore._embed_calls,
        # With zero calls, active_backend is the CONFIGURED tier, not a
        # measured one -- a dead endpoint looks identical to a healthy one.
        # Call with ?probe=true to get an observed answer.
        "active_backend_observed": vstore._embed_calls > 0,
    }
    if probe:
        import time
        import numpy as np
        t0 = time.time()
        try:
            v = vstore.embed_texts(["clearance"])
            out["probe"] = {
                "ok": True, "dim": int(v.shape[1]),
                "l2_norm": round(float(np.linalg.norm(v[0])), 6),
                "ms": int((time.time() - t0) * 1000),
                # embed_texts may fail over mid-call, so re-read afterwards.
                "served_by": vstore.embed_backend(), "space": vstore.embed_space(),
            }
        except Exception as e:
            out["probe"] = {"ok": False, "error": f"{type(e).__name__}: {e}",
                            "ms": int((time.time() - t0) * 1000)}
        # The snapshot above was taken BEFORE the probe, so it would report the
        # pre-failover tier next to a probe result that contradicts it. Re-read.
        out.update({
            "arctic_latched_dead": vstore._remote_dead,
            "gemini_latched_dead": vstore._gemini_dead,
            "active_backend": vstore.embed_backend(),
            "space": vstore.embed_space(),
            "embed_calls": vstore._embed_calls,
            "active_backend_observed": vstore._embed_calls > 0,
        })
    return out


# ── Ask: retrieval and/or text-to-SQL ─────────────────────────────────────────

def _looks_structured(question: str, tenant: str) -> bool:
    """
    Heuristic router: decide whether a question targets the warehouse (SQL) or
    the document corpus (retrieval).

    Structured requires BOTH:
      - the tenant has warehouse tables, AND
      - the question shows explicit tabular/aggregate intent (group-by, top-N,
        count, per-X, ranking) — i.e. it reads like a query over rows/columns.

    Bare domain words ("revenue", "sales") alone do NOT force SQL, because a
    question like "What drove revenue growth in Q3?" is about the *narrative*
    in an uploaded document, not a table aggregation. When the tenant has
    indexed documents and the question lacks explicit aggregate structure, we
    prefer retrieval.
    """
    tables = list_tables(tenant)
    if not tables:
        return False
    q = question.lower()
    # Explicit aggregate/tabular intent — strong SQL signals.
    strong_sql_signals = [
        "how many", "count of", "total ", "sum of", "average", "avg ",
        "per ", " by ", "group by", "each ", "top ", "highest", "lowest",
        "rank", "max ", "min ", "breakdown", "list all", "which region",
        "which product", "grouped",
    ]
    if any(s in q for s in strong_sql_signals):
        return True
    # Weak domain words only route to SQL if the tenant has NO documents to
    # answer from — otherwise the question is almost certainly about the docs.
    weak_domain_words = ["revenue", "sales", "profit", "growth", "churn", "customers"]
    if any(w in q for w in weak_domain_words):
        try:
            vs = VectorStore(tenant=tenant)
            has_docs = vs.count() > 0 or vs.load()
        except Exception:
            has_docs = False
        return not has_docs
    return False


def _sql_answer(tenant: str, question: str) -> dict:
    """Generate SQL over the tenant's warehouse via the LLM, execute, return rows."""
    tables = list_tables(tenant)
    if not tables:
        raise ValueError(f"No warehouse tables for tenant '{tenant}'. Upload a CSV first.")
    schema_lines = []
    for t in tables:
        cols = ", ".join(f"{c['name']} {c['type']}" for c in t["columns"])
        schema_lines.append(f"TABLE {t['table_name']} ({cols})  -- {t['row_count']} rows")
    schema = "\n".join(schema_lines)
    messages = [
        {"role": "system", "content": (
            "You are a DuckDB SQL generator. Given the schema, write ONE valid DuckDB SELECT "
            "query that answers the question. Return ONLY the SQL, no markdown fences, no explanation."
        )},
        {"role": "user", "content": f"Schema:\n{schema}\n\nQuestion: {question}"},
    ]
    out = chat(messages, max_tokens=512)
    sql = out["text"].strip().strip("`")
    if sql.lower().startswith("sql"):
        sql = sql[3:].strip()
    result = warehouse_query(tenant, sql)
    return {"sql": sql, "columns": result["columns"], "rows": result["rows"],
            "row_count": result["row_count"], "backend": out["backend"]}


@router.post("/ask")
async def ask(req: AskRequest, tenant: str = Depends(resolve_tenant)):
    """
    Answer a question for this tenant.
    mode=auto routes to retrieval (unstructured) or text-to-SQL (structured).
    """
    mode = req.mode
    if mode == "auto":
        mode = "sql" if _looks_structured(req.question, tenant) else "retrieval"

    if mode == "retrieval":
        try:
            out = answer_with_retrieval(tenant, req.question, k=req.k)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return {"mode": "retrieval", "tenant": tenant, **out}

    # structured path
    try:
        out = _sql_answer(tenant, req.question)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SQL execution failed: {e}")

    response = {"mode": "sql", "tenant": tenant, **out}
    if req.make_dashboard and out["rows"]:
        dash = composer.compose(req.question, out["rows"], out["columns"])
        saved = composer.save_dashboard(dash, tenant)
        response["dashboard"] = {"id": dash["id"], "panels": [p["type"] for p in dash["panels"]],
                                 "persisted": saved}
    return response


# ── Dashboards ────────────────────────────────────────────────────────────────

@router.post("/dashboard")
async def make_dashboard(req: DashboardRequest, tenant: str = Depends(resolve_tenant)):
    """Compose + persist a dashboard from a SQL question over the warehouse."""
    try:
        result = warehouse_query(tenant, req.sql)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"SQL failed: {e}")
    if not result["rows"]:
        raise HTTPException(status_code=422, detail="Query returned no rows — cannot build a dashboard")
    dash = composer.compose(req.question, result["rows"], result["columns"], title=req.title)
    saved = composer.save_dashboard(dash, tenant)
    return {"dashboard": dash, "persisted": saved}


@router.get("/dashboards")
async def dashboards(tenant: str = Depends(resolve_tenant)):
    """List the tenant's persisted dashboards."""
    return {"tenant": tenant, "dashboards": composer.list_dashboards(tenant)}


@router.get("/dashboard/{dashboard_id}")
async def get_dashboard(dashboard_id: str, tenant: str = Depends(resolve_tenant)):
    """Load a persisted dashboard by id."""
    dash = composer.load_dashboard(dashboard_id, tenant)
    if not dash:
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return dash
