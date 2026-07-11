"""Multi-model benchmark: run all dev questions against multiple Fireworks models."""

import json
import os
import time
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.utils import load_db, execute_sql
from src.agent import TextToSQLAgent


# Models to benchmark (all available on Fireworks serverless)
BENCHMARK_MODELS = {
    "gpt-oss-120b": "accounts/fireworks/models/gpt-oss-120b",
    "deepseek-v4-flash": "accounts/fireworks/models/deepseek-v4-flash",
    "glm-5p2": "accounts/fireworks/models/glm-5p2",
    "kimi-k2p6": "accounts/fireworks/models/kimi-k2p6",
}


def run_benchmark(
    db_path: str = "data/Chinook.db",
    questions_path: str = "data/dev_questions_with_answers.json",
    output_path: str = "benchmark_results.json",
    models: dict | None = None,
):
    """
    Run all dev questions against multiple models and produce a comparison report.

    Args:
        db_path: Path to the SQLite database
        questions_path: Path to the dev questions with gold answers
        output_path: Where to write the benchmark results JSON
        models: Dict of {friendly_name: model_id} to benchmark
    """
    if models is None:
        models = BENCHMARK_MODELS

    load_dotenv()
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        print("Error: FIREWORKS_API_KEY not set")
        sys.exit(1)

    # Load questions
    with open(questions_path) as f:
        questions = json.load(f)

    results = {}

    for model_name, model_id in models.items():
        print(f"\n{'='*60}")
        print(f"Benchmarking: {model_name} ({model_id})")
        print(f"{'='*60}")

        conn = load_db(db_path)
        agent = TextToSQLAgent(conn, model=model_id, api_key=api_key)

        model_results = []
        for q in questions:
            qid = q["id"]
            question = q["question"]
            tier = q.get("tier", "?")
            gold_sql = q.get("gold_sql", "")

            print(f"  {qid} (tier {tier}): {question[:60]}...", end=" ", flush=True)

            # Run the agent
            agent.reset()
            t0 = time.time()
            try:
                resp = agent.ask(question)
                latency = time.time() - t0

                # Check if SQL executes
                exec_success = resp.success

                # Compare results against gold
                data_match = False
                if exec_success and resp.sql:
                    # Run gold SQL to get gold results
                    try:
                        cur = conn.cursor()
                        cur.execute(gold_sql)
                        gold_rows = cur.fetchall()
                        gold_cols = [d[0] for d in cur.description]
                        gold_results = [dict(zip(gold_cols, r)) for r in gold_rows]

                        # Value-based comparison
                        data_match = _compare_results(resp.results, gold_results)
                    except Exception:
                        data_match = False

                entry = {
                    "question_id": qid,
                    "question": question,
                    "tier": tier,
                    "generated_sql": resp.sql,
                    "exec_success": exec_success,
                    "data_match": data_match,
                    "latency": round(resp.latency, 3),
                    "attempts": resp.attempts,
                    "error": resp.error,
                    "result_count": len(resp.results),
                }
                model_results.append(entry)
                status = "MATCH" if data_match else ("EXEC_OK" if exec_success else "FAIL")
                print(f"{status} ({resp.latency:.2f}s, {resp.attempts} attempt(s))")

            except Exception as e:
                latency = time.time() - t0
                entry = {
                    "question_id": qid,
                    "question": question,
                    "tier": tier,
                    "generated_sql": "",
                    "exec_success": False,
                    "data_match": False,
                    "latency": round(latency, 3),
                    "attempts": 0,
                    "error": str(e),
                    "result_count": 0,
                }
                model_results.append(entry)
                print(f"ERROR ({latency:.2f}s): {str(e)[:60]}")

        conn.close()

        # Calculate summary stats
        exec_count = sum(1 for r in model_results if r["exec_success"])
        match_count = sum(1 for r in model_results if r["data_match"])
        latencies = [r["latency"] for r in model_results if r["latency"] > 0]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        p50_latency = sorted(latencies)[len(latencies) // 2] if latencies else 0

        results[model_name] = {
            "model_id": model_id,
            "questions": model_results,
            "summary": {
                "exec_success": f"{exec_count}/10",
                "data_match": f"{match_count}/10",
                "avg_latency": round(avg_latency, 3),
                "p50_latency": round(p50_latency, 3),
            },
        }

        print(f"  Summary: exec={exec_count}/10, match={match_count}/10, "
              f"avg={avg_latency:.2f}s, p50={p50_latency:.2f}s")

    # Write results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nBenchmark results written to {output_path}")

    # Print comparison table
    print(f"\n{'='*70}")
    print("BENCHMARK COMPARISON")
    print(f"{'='*70}")
    print(f"{'Model':<20s} {'Exec':>6s} {'Match':>6s} {'Avg Lat':>8s} {'P50 Lat':>8s}")
    print(f"{'-'*70}")
    for model_name, data in results.items():
        s = data["summary"]
        print(f"{model_name:<20s} {s['exec_success']:>6s} {s['data_match']:>6s} "
              f"{s['avg_latency']:>7.2f}s {s['p50_latency']:>7.2f}s")
    print(f"{'='*70}")


def _compare_results(generated: list[dict], gold: list[dict]) -> bool:
    """
    Compare two result sets using value-based matching with numeric rounding.

    Converts each row to a sorted tuple of string values (handles column name
    differences), rounds numeric values to 2 decimal places, and compares as sets.
    """
    if len(generated) != len(gold):
        return False

    def row_to_value_tuple(row: dict) -> tuple:
        vals = []
        for v in row.values():
            if isinstance(v, (int, float)):
                vals.append(str(round(float(v), 2)))
            else:
                vals.append(str(v))
        return tuple(sorted(vals))

    gen_set = set(row_to_value_tuple(r) for r in generated)
    gold_set = set(row_to_value_tuple(r) for r in gold)
    return gen_set == gold_set


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark multiple models on dev questions")
    parser.add_argument("--db", default="data/Chinook.db", help="Database path")
    parser.add_argument("--output", default="benchmark_results.json", help="Output file")
    args = parser.parse_args()
    run_benchmark(db_path=args.db, output_path=args.output)


if __name__ == "__main__":
    main()
