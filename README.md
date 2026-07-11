# Agentic Text-to-SQL — Fireworks AI Field Engineering Take-Home

An agentic text-to-SQL system that converts natural language questions to SQL, executes them against a database, and returns results with a natural-language summary. Built on Fireworks AI open-source models using a tool-calling ReAct architecture with a self-healing SQL correction loop. Available as both an **interactive CLI** and a **web-based BI tool**.

## Architecture

### Agentic Loop (Hybrid Tool-Calling)

The agent uses a **ReAct-style tool-calling loop** with OpenAI-compatible function calling:

1. **Schema injection**: The full DDL schema (CREATE TABLE statements) is injected into the system prompt upfront — no tool call needed to discover the schema, saving ~1s of latency per query.
2. **Tool calling**: The model has two tools:
   - `run_sql(sql)` — executes a SQL query and returns results as JSON, or an error message
   - `get_sample_rows(table_name)` — returns 3 sample rows from a table (for disambiguation)
3. **Self-healing**: If `run_sql` returns an error, the error message is fed back to the model as the tool result. The model reads the error, generates corrected SQL, and retries — up to 3 attempts. The user never sees the failed attempt.
4. **Conversation context**: Message history persists across turns, so follow-up questions like "what about just the top 3?" work without re-explaining the schema.
5. **Natural-language summary**: After successful execution, the model produces a concise summary of the results.

### Model Selection

Benchmarked 4 open-source models available on Fireworks serverless:

| Model | ID | Avg Latency | P50 Latency | Exec Accuracy | Data Match | Cost ($/M tok) |
|-------|-----|-------------|-------------|---------------|------------|-----------------|
| GPT-OSS 120B | `accounts/fireworks/models/gpt-oss-120b` | 2.23s | 2.00s | 10/10 | 9/10 | $0.15 in / $0.60 out |
| DeepSeek V4 Flash | `accounts/fireworks/models/deepseek-v4-flash` | 4.92s | 4.66s | 10/10 | 7/10 | $0.14 in / $0.28 out |
| GLM 5.2 | `accounts/fireworks/models/glm-5p2` | 3.74s | 3.92s | 10/10 | 8/10 | $1.40 in / $4.40 out |
| Kimi K2.6 | `accounts/fireworks/models/kimi-k2p6` | 3.52s | 3.25s | 10/10 | 8/10 | $0.95 in / $4.00 out |

**Default model**: `gpt-oss-120b` — best balance of accuracy (9/10 data match, highest of all models), latency (P50 2.00s, well under the 3s target and 3.5x faster than the customer's 7s baseline), and cost (~90% cheaper than GPT-5.4).

**Self-healing in action**: The benchmark demonstrated the self-healing loop working in practice. DeepSeek V4 Flash triggered self-correction on q_006 (3 attempts) and q_010 (2 attempts), and GLM 5.2 self-corrected on q_006 (2 attempts). GPT-OSS 120B needed zero retries — all 10 queries succeeded on the first attempt.

### Safety

- `execute_sql()` enforces **read-only access** — blocks INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, and other destructive operations
- Table names in `get_sample_rows()` are sanitized to prevent SQL injection

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (package manager)
- A Fireworks AI API key

### Installation

```bash
# 1. Download the Chinook database
./setup.sh

# 2. Install dependencies
uv sync

# 3. Set your API key
export FIREWORKS_API_KEY=<your-key>

# Or create a .env file:
echo "FIREWORKS_API_KEY=<your-key>" > .env
```

## Usage

### Interactive CLI

```bash
# Default model (gpt-oss-120b)
uv run cli

# Specify a model
uv run cli --model deepseek-v4-flash
uv run cli --model glm-5p2
uv run cli --model kimi-k2p6

# Specify a database
uv run cli --db data/Chinook.db
```

Or without uv:
```bash
python -m src.cli
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `<question>` | Ask a natural language question about your data |
| `schema` | Show the database schema (all tables and columns) |
| `clear` | Clear conversation history (start fresh) |
| `history` | Show conversation history from this session |
| `help` | Show available commands and models |
| `exit` / `quit` | Exit the CLI |

### Example Session

```
sql> What are the top 5 best-selling genres by total sales?

SQL:
  SELECT g.Name AS Genre, ROUND(SUM(il.UnitPrice * il.Quantity), 2) AS TotalSales
  FROM InvoiceLine il JOIN Track t ON il.TrackId = t.TrackId
  JOIN Genre g ON t.GenreId = g.GenreId
  GROUP BY g.GenreId, g.Name ORDER BY TotalSales DESC LIMIT 5;

Results (5 rows, 1 attempt(s), 2.58s):
  +----------------------+------------+
  | Genre                | TotalSales |
  +----------------------+------------+
  | Rock                 | 826.65     |
  | Latin                | 382.14     |
  | Metal                | 261.36     |
  | Alternative & Punk   | 241.56     |
  | TV Shows             | 93.53      |
  +----------------------+------------+

Summary:
  The five genres that generated the most revenue are:
  1. Rock – $826.65
  2. Latin – $382.14
  ...

sql> What about just the top 3?
(follow-up uses conversation context — no schema re-injection needed)
```

### Web UI

The web app provides a full BI tool interface with the same agent backend as the CLI:

```bash
# Start the web server
uv run web

# Or with options
uv run web --host 0.0.0.0 --port 8000 --db data/Chinook.db

# Or without uv
python -m src.web
```

Open `http://localhost:8000` in your browser.

#### Features

- **Chat tab**: Ask natural language questions, see generated SQL, results tables, and summaries with latency and attempt count badges. Conversation context persists across questions in the same session.
- **Benchmark tab**: Visual comparison of all 4 benchmarked models — accuracy bar chart, latency bar chart with 3s target line, and a detailed comparison table.
- **Eval tab**: Summary cards (execution accuracy, data match, P50 latency) and a per-question results table showing generated SQL, gold SQL, and match status.
- **Schema browser**: Sidebar with collapsible table accordions showing all columns with types and PK indicators. Click any table to view sample rows in a modal.
- **Model switcher**: Switch between all 5 available models from the header dropdown. Conversation history is preserved when switching models.
- **Session management**: In-memory sessions with 30-minute timeout. Create new sessions, clear history, or view conversation history.

#### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve the web UI (HTML) |
| `/api/models` | GET | List available models with cost info |
| `/api/schema` | GET | Database schema as structured JSON + DDL |
| `/api/sample-rows/{table}` | GET | Sample rows from a table |
| `/api/query` | POST | Process a natural language question |
| `/api/session/new` | POST | Create a new session |
| `/api/session/{id}/clear` | POST | Clear session history |
| `/api/session/{id}/history` | GET | Get session conversation history |
| `/api/benchmark` | GET | Benchmark results (from `benchmark_results.json`) |
| `/api/eval` | GET | Eval results (from `eval_report.json`) |

### Evaluation

Run the agent on all 10 dev questions and compare against gold answers:

```bash
uv run eval
# or: python -m src.eval
```

Outputs:
- `dev_answers.json` — generated SQL + human-readable answer for each question
- `eval_report.json` — detailed metrics (exec success, data match, latency, attempts per question)

### Multi-Model Benchmark

Run all 10 dev questions against 4 models and compare:

```bash
uv run benchmark
# or: python -m src.benchmark
```

Outputs:
- `benchmark_results.json` — per-model per-question results with comparison table

## Project Structure

```
.
├── README.md
├── setup.sh                    # Downloads Chinook database
├── pyproject.toml              # Dependencies + script entry points
├── requirements.txt            # pip dependencies (for Render deployment)
├── render.yaml                 # Render.com Blueprint (IaC deployment config)
├── uv.lock
├── .github/workflows/eval.yml  # CI eval pipeline (GitHub Actions)
├── src/
│   ├── __init__.py
│   ├── utils.py                # DB utilities: load, query, DDL extraction, safe execution
│   ├── agent.py                # TextToSQLAgent: ReAct loop, self-healing, conversation context
│   ├── cli.py                  # Interactive REPL CLI
│   ├── web.py                  # FastAPI web app (BI tool UI backend)
│   ├── benchmark.py            # Multi-model benchmark harness
│   ├── eval.py                 # Evaluation harness (produces dev_answers.json)
│   ├── ci_eval.py              # CI eval pipeline with regression detection
│   ├── cache.py                # Semantic cache (exact + fuzzy match, TTL, metrics)
│   ├── few_shot.py             # Few-shot prompt registry (8 verified SQL examples)
│   ├── ratelimit.py            # Rate limiting + usage tracking (sliding window)
│   ├── dialect.py              # Multi-dialect SQL support (SQLite, PostgreSQL, MySQL)
│   └── static/
│       └── index.html          # Full BI tool frontend (vanilla HTML/JS/CSS)
├── data/
│   ├── Chinook.db              # SQLite database (11 tables, digital music store)
│   ├── dev_questions.json      # 10 development questions
│   ├── dev_questions_with_answers.json  # Gold SQL + expected results
│   └── dev_answers_example.json
├── dev_answers.json            # System outputs for 10 dev questions (generated by eval)
├── eval_report.json            # Detailed evaluation metrics
├── benchmark_results.json      # Multi-model comparison results
└── EMAIL_TO_RAUL.md            # Customer email response
```

## Production Features

### Semantic Caching

Repeated or similar questions return cached results in <1ms without calling the LLM. The cache uses:
- **Exact match**: Normalized question hash (case/punctuation insensitive) — O(1) lookup
- **Fuzzy match**: Jaccard similarity on word sets for paraphrased questions (threshold: 0.85)
- **TTL**: Entries expire after 1 hour (configurable)
- **LRU eviction**: Oldest entries evicted when cache reaches capacity

```python
from src.cache import SemanticCache
cache = SemanticCache(ttl_seconds=3600, similarity_threshold=0.85)
hit = cache.get("What are the top 5 genres?")  # <1ms if cached
```

Cache metrics available at `GET /api/cache/metrics`.

### Few-Shot Prompt Registry

8 verified SQL examples covering 8 query patterns (aggregation joins, date filtering, window functions, HAVING clauses, LEFT JOINs, etc.) are injected into the system prompt. This teaches the model schema-specific patterns, pushing accuracy from 90% toward 95%+.

```python
from src.few_shot import build_few_shot_block
block = build_few_shot_block(max_examples=5)  # Inject into system prompt
```

### Rate Limiting & Usage Tracking

Sliding window rate limiter (30 req/min, 500 req/hour by default) with per-session usage tracking including cost estimates.

```python
from src.ratelimit import RateLimiter, UsageTracker
limiter = RateLimiter(requests_per_minute=30, requests_per_hour=500)
tracker = UsageTracker()
```

Rate limit status at `GET /api/rate-limit/{client_id}`. Usage metrics at `GET /api/usage` and `GET /api/usage/{session_id}`.

### Query Result Visualization

The backend auto-detects 2-column numeric/categorical results and returns chart data. The frontend renders colored bar charts for aggregation results (e.g., "Genre vs TotalSales").

### Multi-Dialect SQL Support

Dialect detection from connection strings with dialect-specific SQL rules injected into the system prompt:

```python
from src.dialect import detect_dialect, get_dialect_specific_prompt
detect_dialect("postgresql://user:pass@host/db")  # -> "postgresql"
get_dialect_specific_prompt("postgresql")  # -> EXTRACT(), ||, ILIKE, ::type rules
```

Supports SQLite (default), PostgreSQL, and MySQL.

### CI Evaluation Pipeline

Regression detection with baseline comparison, designed for GitHub Actions:

```bash
# Run eval with regression checks
python -m src.ci_eval --min-exec-accuracy 1.0 --min-match-accuracy 0.80 --max-p50-latency 3.0

# Update baseline after intentional prompt changes
python -m src.ci_eval --update-baseline
```

Exit codes: 0=pass, 1=accuracy regression, 2=latency regression, 3=eval error.

The included GitHub Actions workflow (`.github/workflows/eval.yml`) runs the CI eval on every push/PR that modifies `src/` or the dev questions.

## Deployment

### Render.com

The app is configured for deployment on Render via `render.yaml`:

```yaml
services:
  - type: web
    name: agentic-text-to-sql
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn src.web:create_app --factory --host 0.0.0.0 --port $PORT
    healthCheckPath: /health
    envVars:
      - key: FIREWORKS_API_KEY
        sync: false
```

To deploy:
1. Push the code to GitHub
2. Create a new Web Service on Render, connect the GitHub repo
3. Set the `FIREWORKS_API_KEY` environment variable
4. Render auto-detects `render.yaml` and deploys

Or use the Render API:
```bash
curl -X POST https://api.render.com/v1/services \
  -H "Authorization: Bearer $RENDER_API_KEY" \
  -H "Content-Type: application/json" \
  -d @render.yaml  # (converted to JSON)
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI (HTML) |
| `/health` | GET | Health check |
| `/api/models` | GET | Available models |
| `/api/schema` | GET | Database schema |
| `/api/sample-rows/{table}` | GET | Sample rows from a table |
| `/api/query` | POST | Process a natural language question |
| `/api/session/new` | POST | Create a new session |
| `/api/session/{id}/clear` | POST | Clear session history |
| `/api/session/{id}/history` | GET | Get conversation history |
| `/api/benchmark` | GET | Benchmark results |
| `/api/eval` | GET | Eval results |
| `/api/cache/metrics` | GET | Semantic cache metrics |
| `/api/usage` | GET | Global usage metrics |
| `/api/usage/{session_id}` | GET | Per-session usage |
| `/api/rate-limit/{client_id}` | GET | Rate limit status |

## Results

### Evaluation (gpt-oss-120b)

| Metric | Result |
|--------|--------|
| Execution accuracy | 10/10 (100%) |
| Data match accuracy | 9/10 (90%) |
| Average latency | 2.17s |
| P50 latency | 1.97s |
| Self-healing triggered | 0/10 (all SQL correct on first attempt) |

The one data mismatch (q_003) is a column projection difference: the agent combines `FirstName` and `LastName` into a single `Name` column per the system prompt rules, while the gold answer has them separate. The actual data values are correct.

### Key Findings

1. **Quality**: 100% execution accuracy, 90% data match. The self-healing loop was available but not needed — the schema injection + SQL generation rules produced valid SQL on the first attempt for all 10 questions.
2. **Latency**: P50 of 1.97s — well under the 3s target, and a 3.5x improvement over the customer's 7s baseline.
3. **Cost**: At ~$0.15/$0.60 per million tokens (gpt-oss-120b), the cost is approximately 90% lower than GPT-5.4, making the unit economics work at 30,000 queries/day.

## AI Assistance Disclosure

This implementation was built with assistance from Biomni (Phylo's AI research collaborator). The AI helped with code generation, testing, and the multi-model benchmark. All code was reviewed, tested, and validated against the provided dev questions. The architecture decisions, model selection rationale, and customer email were written based on the actual benchmark results.
