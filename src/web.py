"""FastAPI web application for the Agentic Text-to-SQL system.

Run with: uv run web (or python -m src.web)
Opens at http://localhost:8000
"""

import json
import os
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.utils import load_db, get_ddl_schema, get_sample_rows, get_schema
from src.agent import TextToSQLAgent, DEFAULT_MODEL
from src.cache import SemanticCache
from src.ratelimit import RateLimiter, UsageTracker


# ── Available models ──────────────────────────────────────────────────────────

AVAILABLE_MODELS = {
    "gpt-oss-120b": {
        "id": "accounts/fireworks/models/gpt-oss-120b",
        "name": "GPT-OSS 120B",
        "cost_in": 0.15,
        "cost_out": 0.60,
        "description": "Best balance of accuracy, latency, and cost",
    },
    "deepseek-v4-flash": {
        "id": "accounts/fireworks/models/deepseek-v4-flash",
        "name": "DeepSeek V4 Flash",
        "cost_in": 0.14,
        "cost_out": 0.28,
        "description": "Cheapest option, slightly lower accuracy",
    },
    "glm-5p2": {
        "id": "accounts/fireworks/models/glm-5p2",
        "name": "GLM 5.2",
        "cost_in": 1.40,
        "cost_out": 4.40,
        "description": "Strong accuracy, higher cost",
    },
    "kimi-k2p6": {
        "id": "accounts/fireworks/models/kimi-k2p6",
        "name": "Kimi K2.6",
        "cost_in": 0.95,
        "cost_out": 4.00,
        "description": "Reasoning model, good for complex queries",
    },
    "deepseek-v4-pro": {
        "id": "accounts/fireworks/models/deepseek-v4-pro",
        "name": "DeepSeek V4 Pro",
        "cost_in": 1.74,
        "cost_out": 3.48,
        "description": "Highest quality, highest latency (~16s)",
    },
}


# ── Session Manager ───────────────────────────────────────────────────────────

class SessionManager:
    """Manages in-memory agent sessions with conversation history."""

    def __init__(self, conn: sqlite3.Connection, api_key: str):
        self.conn = conn
        self.api_key = api_key
        self.sessions: dict[str, dict] = {}
        self.session_timeout = 1800  # 30 minutes

    def create_session(self, model: str = DEFAULT_MODEL) -> str:
        """Create a new session with its own agent instance."""
        session_id = str(uuid.uuid4())
        agent = TextToSQLAgent(
            conn=self.conn,
            model=model,
            api_key=self.api_key,
        )
        self.sessions[session_id] = {
            "agent": agent,
            "model": model,
            "created_at": time.time(),
            "last_active": time.time(),
        }
        return session_id

    def get_session(self, session_id: str) -> Optional[dict]:
        """Get a session by ID, creating one if it doesn't exist."""
        if session_id not in self.sessions:
            return None
        session = self.sessions[session_id]
        session["last_active"] = time.time()
        return session

    def get_or_create_session(self, session_id: Optional[str], model: str = DEFAULT_MODEL) -> str:
        """Get an existing session or create a new one."""
        if session_id and session_id in self.sessions:
            self.sessions[session_id]["last_active"] = time.time()
            return session_id
        return self.create_session(model)

    def clear_session(self, session_id: str) -> bool:
        """Clear conversation history for a session."""
        if session_id not in self.sessions:
            return False
        self.sessions[session_id]["agent"].reset()
        self.sessions[session_id]["last_active"] = time.time()
        return True

    def switch_model(self, session_id: str, model: str) -> bool:
        """Switch the model for an existing session, preserving conversation history."""
        if session_id not in self.sessions:
            return False
        old_agent = self.sessions[session_id]["agent"]
        new_agent = TextToSQLAgent(
            conn=self.conn,
            model=model,
            api_key=self.api_key,
        )
        # Preserve conversation history (system prompt will be the same)
        new_agent.messages = old_agent.messages
        self.sessions[session_id]["agent"] = new_agent
        self.sessions[session_id]["model"] = model
        self.sessions[session_id]["last_active"] = time.time()
        return True

    def get_history(self, session_id: str) -> Optional[list[dict]]:
        """Get conversation history for a session."""
        if session_id not in self.sessions:
            return None
        return self.sessions[session_id]["agent"].get_history()

    def cleanup_expired(self):
        """Remove sessions that have been inactive for longer than the timeout."""
        now = time.time()
        expired = [
            sid for sid, s in self.sessions.items()
            if now - s["last_active"] > self.session_timeout
        ]
        for sid in expired:
            del self.sessions[sid]


# ── Pydantic models for request bodies ────────────────────────────────────────

class QueryRequest(BaseModel):
    session_id: Optional[str] = None
    question: str
    model: Optional[str] = None


class SessionRequest(BaseModel):
    model: Optional[str] = DEFAULT_MODEL


# ── Chart detection ───────────────────────────────────────────────────────────

def _detect_chart_type(results: list, columns: list) -> Optional[dict]:
    """
    Auto-detect whether query results can be visualized as a chart.

    Detects:
    - Bar chart: 2 columns where one is categorical (label) and one is numeric (value)
    - Suitable for aggregation results (e.g., "Genre" + "TotalSales")

    Returns:
        dict with chart_type, labels, values, label_col, value_col if chartable,
        None otherwise.
    """
    if not results or not columns or len(columns) < 2:
        return None

    # Need exactly 2 columns for a simple bar chart
    if len(columns) > 2:
        return None

    # Need at least 2 rows for a meaningful chart
    if len(results) < 2:
        return None

    # Identify which column is categorical (label) and which is numeric (value)
    col_types = {}
    for col in columns:
        values = [r.get(col) for r in results]
        numeric_count = sum(1 for v in values if isinstance(v, (int, float)))
        col_types[col] = "numeric" if numeric_count > len(values) * 0.7 else "categorical"

    numeric_cols = [c for c in columns if col_types[c] == "numeric"]
    categorical_cols = [c for c in columns if col_types[c] == "categorical"]

    # Need exactly 1 numeric and 1 categorical for a bar chart
    if len(numeric_cols) != 1 or len(categorical_cols) != 1:
        return None

    label_col = categorical_cols[0]
    value_col = numeric_cols[0]

    # Build chart data
    labels = [str(r.get(label_col, "")) for r in results]
    values = [r.get(value_col, 0) for r in results]

    # Round floats for display
    values = [round(v, 2) if isinstance(v, float) else v for v in values]

    return {
        "chart_type": "bar",
        "label_col": label_col,
        "value_col": value_col,
        "labels": labels,
        "values": values,
    }


# ── App factory ───────────────────────────────────────────────────────────────

def create_app(db_path: str = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    load_dotenv()
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("FIREWORKS_API_KEY environment variable is not set")

    # Resolve DB path — check env var, then default locations
    if db_path is None:
        db_path = os.environ.get("DB_PATH", "data/Chinook.db")

    # Try multiple DB locations for deployment flexibility
    db_candidates = [
        db_path,
        os.path.join(os.path.dirname(__file__), "..", "data", "Chinook.db"),
        "/opt/data/Chinook.db",
        "data/Chinook.db",
    ]
    db_found = None
    for candidate in db_candidates:
        if os.path.exists(candidate):
            db_found = os.path.abspath(candidate)
            break

    if not db_found:
        # If no DB exists, download it via setup.sh
        import subprocess
        setup_script = os.path.join(os.path.dirname(__file__), "..", "setup.sh")
        if os.path.exists(setup_script):
            print(f"Database not found, running setup.sh...")
            subprocess.run(["bash", setup_script], check=True)
            db_found = os.path.abspath("data/Chinook.db")
        else:
            raise RuntimeError(f"Database not found at any of: {db_candidates}")

    print(f"Using database: {db_found}")

    # Load database with thread-safe connection
    conn = sqlite3.connect(db_found, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row

    session_manager = SessionManager(conn, api_key)

    # Production features: semantic cache, rate limiter, usage tracker
    cache = SemanticCache(ttl_seconds=3600, similarity_threshold=0.85)
    rate_limiter = RateLimiter(requests_per_minute=30, requests_per_hour=500)
    usage_tracker = UsageTracker()

    # Paths
    base_dir = Path(__file__).parent
    static_dir = base_dir / "static"
    static_dir.mkdir(exist_ok=True)

    app = FastAPI(title="Agentic Text-to-SQL", version="1.0.0")

    # Mount static files (for any additional assets)
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # ── Routes ─────────────────────────────────────────────────────────────

    @app.get("/health")
    async def health_check():
        """Health check endpoint for Render and monitoring."""
        return {"status": "ok", "service": "agentic-text-to-sql"}

    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Serve the main HTML UI."""
        index_path = static_dir / "index.html"
        if index_path.exists():
            return HTMLResponse(index_path.read_text())
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)

    @app.get("/api/models")
    async def list_models():
        """List available models."""
        return {"models": AVAILABLE_MODELS, "default": DEFAULT_MODEL}

    @app.get("/api/schema")
    async def get_schema_endpoint():
        """Return the database schema as structured JSON."""
        schema = get_schema(conn)
        ddl = get_ddl_schema(conn)
        # Format schema for frontend consumption
        tables = []
        for table_name, columns in schema.items():
            tables.append({
                "name": table_name,
                "columns": [
                    {
                        "name": col["name"],
                        "type": col["type"],
                        "nullable": col["notnull"] == 0,
                        "pk": col["pk"] > 0,
                    }
                    for col in columns
                ],
            })
        return {"tables": tables, "ddl": ddl}

    @app.get("/api/sample-rows/{table_name}")
    async def sample_rows(table_name: str):
        """Get sample rows from a table."""
        try:
            rows = get_sample_rows(conn, table_name, n=5)
            return {"table": table_name, "rows": rows}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/session/new")
    async def new_session(req: SessionRequest):
        """Create a new session."""
        model = req.model or DEFAULT_MODEL
        session_id = session_manager.create_session(model)
        return {"session_id": session_id, "model": model}

    @app.post("/api/session/{session_id}/clear")
    async def clear_session(session_id: str):
        """Clear conversation history for a session."""
        if not session_manager.clear_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        return {"status": "cleared", "session_id": session_id}

    @app.get("/api/session/{session_id}/history")
    async def session_history(session_id: str):
        """Get conversation history for a session."""
        history = session_manager.get_history(session_id)
        if history is None:
            raise HTTPException(status_code=404, detail="Session not found")
        # Filter out system prompt and tool messages for display
        display_history = []
        for msg in history:
            if msg.get("role") == "system":
                continue
            display_history.append({
                "role": msg.get("role"),
                "content": msg.get("content", ""),
            })
        return {"history": display_history}

    @app.post("/api/query")
    async def query(req: QueryRequest, request: Request):
        """Process a natural language question through the agent."""
        # Rate limiting — use client IP + session_id as the key
        client_ip = request.client.host if request.client else "unknown"
        rate_key = f"{client_ip}:{req.session_id or 'no-session'}"
        allowed, rate_info = rate_limiter.check(rate_key)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {rate_info.requests_this_minute}/{rate_info.limit_per_minute} per minute. Retry in {rate_info.retry_after}s.",
                headers={"Retry-After": str(rate_info.retry_after)},
            )

        # Semantic cache check — return cached response in <1ms if available
        cached = cache.get(req.question)
        if cached and cached.success:
            usage_tracker.record_query(
                session_id=req.session_id or "cached",
                model=req.model or DEFAULT_MODEL,
                latency=0.001,  # cache hit latency
                cache_hit=True,
                question=req.question,
            )
            return {
                "session_id": req.session_id or "",
                "sql": cached.sql,
                "results": cached.results,
                "columns": cached.columns,
                "summary": cached.summary,
                "latency": 0.001,
                "attempts": cached.attempts,
                "success": cached.success,
                "error": cached.error,
                "cache_hit": True,
                "chart": cached.chart or _detect_chart_type(cached.results, cached.columns),
                "trust": cached.trust,
            }

        # Get or create session
        model = req.model or DEFAULT_MODEL
        session_id = session_manager.get_or_create_session(req.session_id, model)

        # Switch model if different from current session
        if req.model and session_manager.sessions[session_id]["model"] != model:
            session_manager.switch_model(session_id, model)

        # Get the agent
        session = session_manager.get_session(session_id)
        if not session:
            session_id = session_manager.create_session(model)
            session = session_manager.get_session(session_id)

        agent = session["agent"]

        # Run the query
        try:
            response = agent.ask(req.question)

            # Cache the response for future queries
            cache.put(
                question=req.question,
                sql=response.sql,
                results=response.results,
                columns=response.columns,
                summary=response.summary,
                latency=response.latency,
                attempts=response.attempts,
                success=response.success,
                error=response.error,
                trust=response.trust,
                chart=_detect_chart_type(response.results, response.columns),
            )

            # Track usage
            usage_tracker.record_query(
                session_id=session_id,
                model=model,
                latency=response.latency,
                cache_hit=False,
                question=req.question,
            )

            return {
                "session_id": session_id,
                "sql": response.sql,
                "results": response.results,
                "columns": response.columns,
                "summary": response.summary,
                "latency": round(response.latency, 3),
                "attempts": response.attempts,
                "success": response.success,
                "error": response.error,
                "cache_hit": False,
                "chart": _detect_chart_type(response.results, response.columns),
                "trust": response.trust,
            }
        except Exception as e:
            return {
                "session_id": session_id,
                "sql": "",
                "results": [],
                "columns": [],
                "summary": "",
                "latency": 0,
                "attempts": 0,
                "success": False,
                "error": str(e),
                "cache_hit": False,
                "chart": None,
            }

    @app.get("/api/benchmark")
    async def benchmark_results():
        """Return benchmark results if available."""
        benchmark_path = Path("benchmark_results.json")
        if benchmark_path.exists():
            return json.loads(benchmark_path.read_text())
        return {"error": "No benchmark results found. Run: python -m src.benchmark"}

    @app.get("/api/eval")
    async def eval_results():
        """Return eval results if available."""
        eval_path = Path("eval_report.json")
        if eval_path.exists():
            return json.loads(eval_path.read_text())
        return {"error": "No eval results found. Run: python -m src.eval"}

    @app.get("/api/cache/metrics")
    async def cache_metrics():
        """Return semantic cache performance metrics."""
        return cache.get_metrics()

    @app.get("/api/usage")
    async def global_usage():
        """Return global usage metrics across all sessions."""
        return usage_tracker.get_global_usage()

    @app.get("/api/usage/{session_id}")
    async def session_usage(session_id: str):
        """Return usage metrics for a specific session."""
        usage = usage_tracker.get_session_usage(session_id)
        if usage is None:
            raise HTTPException(status_code=404, detail="Session usage not found")
        return usage

    @app.get("/api/rate-limit/{client_id}")
    async def rate_limit_status(client_id: str):
        """Return current rate limit status for a client."""
        return rate_limiter.get_status(client_id)

    return app


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    """Run the web server."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Agentic Text-to-SQL Web UI")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Port to bind (default: 8000 or $PORT)")
    parser.add_argument("--db", default="data/Chinook.db", help="Database path")
    args = parser.parse_args()

    # Use $PORT env var (set by Render and other cloud platforms) or fall back to 8000
    port = args.port or int(os.environ.get("PORT", 8000))

    load_dotenv()
    if not os.environ.get("FIREWORKS_API_KEY"):
        print("Error: FIREWORKS_API_KEY not set")
        print("Set it with: export FIREWORKS_API_KEY=<your-key>")
        print("Or create a .env file with: FIREWORKS_API_KEY=<your-key>")
        return

    app = create_app(db_path=args.db)
    print(f"\n  Agentic Text-to-SQL Web UI")
    print(f"  Open http://localhost:{port} in your browser\n")
    uvicorn.run(app, host=args.host, port=port)


if __name__ == "__main__":
    main()
