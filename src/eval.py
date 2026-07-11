"""Evaluation harness: run agent on dev questions and compare against gold answers."""

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from src.utils import load_db, execute_sql
from src.agent import TextToSQLAgent, DEFAULT_MODEL


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


def _format_answer_summary(results: list[dict], columns: list[str]) -> str:
    """Create a human-readable summary of results for dev_answers.json."""
    if not results:
        return "(no results)"

    parts = []
    for row in results[:10]:
        vals = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                vals.append(f"${val:.2f}" if "price" in col.lower() or "total" in col.lower() or "sales" in col.lower() or "spent" in col.lower() or "revenue" in col.lower() else str(val))
            else:
                vals.append(str(val))
        parts.append(", ".join(vals))

    summary = "; ".join(parts)
    if len(results) > 10:
        summary += f"; ... ({len(results)} total rows)"
    return summary


def run_eval(
    db_path: str = "data/Chinook.db",
    questions_path: str = "data/dev_questions_with_answers.json",
    answers_output: str = "dev_answers.json",
    report_output: str = "eval_report.json",
    model: str = DEFAULT_MODEL,
):
    """
    Run the agent on all 10 dev questions, compare against gold answers,
    and produce dev_answers.json + eval_report.json.

    Args:
        db_path: Path to the SQLite database
        questions_path: Path to dev questions with gold answers
        answers_output: Where to write dev_answers.json (required deliverable)
        report_output: Where to write eval_report.json (detailed metrics)
        model: Model ID to use for evaluation
    """
    load_dotenv()
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        print("Error: FIREWORKS_API_KEY not set")
        sys.exit(1)

    # Load gold questions
    with open(questions_path) as f:
        questions = json.load(f)

    conn = load_db(db_path)
    agent = TextToSQLAgent(conn, model=model, api_key=api_key)

    dev_answers = {}
    eval_results = []
    exec_count = 0
    match_count = 0
    latencies = []

    print(f"\nRunning evaluation with model: {model}")
    print(f"{'='*70}")

    for q in questions:
        qid = q["id"]
        question = q["question"]
        tier = q.get("tier", "?")
        gold_sql = q.get("gold_sql", "")
        gold_answer = q.get("gold_answer", "")

        print(f"  {qid} (tier {tier}): {question[:60]}...", end=" ", flush=True)

        agent.reset()
        try:
            resp = agent.ask(question)
            latencies.append(resp.latency)

            # Get gold results by running gold SQL
            gold_results = []
            try:
                cur = conn.cursor()
                cur.execute(gold_sql)
                gold_cols = [d[0] for d in cur.description]
                gold_rows = cur.fetchall()
                gold_results = [dict(zip(gold_cols, r)) for r in gold_rows]
            except Exception:
                pass

            # Compare
            data_match = _compare_results(resp.results, gold_results) if resp.success else False
            exec_ok = resp.success

            if exec_ok:
                exec_count += 1
            if data_match:
                match_count += 1

            # Build dev_answers.json entry
            answer_summary = _format_answer_summary(resp.results, resp.columns) if resp.success else f"Error: {resp.error}"
            dev_answers[qid] = {
                "sql": resp.sql,
                "answer": answer_summary,
            }

            # Build eval report entry
            eval_results.append({
                "question_id": qid,
                "question": question,
                "tier": tier,
                "generated_sql": resp.sql,
                "gold_sql": gold_sql,
                "exec_success": exec_ok,
                "data_match": data_match,
                "latency": round(resp.latency, 3),
                "attempts": resp.attempts,
                "error": resp.error,
                "generated_result_count": len(resp.results),
                "gold_result_count": len(gold_results),
                "gold_answer": gold_answer,
            })

            status = "MATCH" if data_match else ("EXEC_OK" if exec_ok else "FAIL")
            print(f"{status} ({resp.latency:.2f}s, {resp.attempts} attempt(s))")

        except Exception as e:
            latencies.append(0)
            dev_answers[qid] = {
                "sql": "",
                "answer": f"Error: {e}",
            }
            eval_results.append({
                "question_id": qid,
                "question": question,
                "tier": tier,
                "generated_sql": "",
                "gold_sql": gold_sql,
                "exec_success": False,
                "data_match": False,
                "latency": 0,
                "attempts": 0,
                "error": str(e),
                "generated_result_count": 0,
                "gold_result_count": 0,
                "gold_answer": gold_answer,
            })
            print(f"ERROR: {str(e)[:60]}")

    conn.close()

    # Calculate stats
    p50 = sorted(latencies)[len(latencies) // 2] if latencies else 0
    avg = sum(latencies) / len(latencies) if latencies else 0

    # Write dev_answers.json
    with open(answers_output, "w") as f:
        json.dump(dev_answers, f, indent=2, default=str)
    print(f"\ndev_answers.json written to {answers_output}")

    # Write eval_report.json
    report = {
        "model": model,
        "summary": {
            "total_questions": len(questions),
            "exec_success": exec_count,
            "data_match": match_count,
            "exec_rate": f"{exec_count}/{len(questions)}",
            "match_rate": f"{match_count}/{len(questions)}",
            "avg_latency": round(avg, 3),
            "p50_latency": round(p50, 3),
        },
        "questions": eval_results,
    }
    with open(report_output, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"eval_report.json written to {report_output}")

    # Print summary
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Model:           {model}")
    print(f"  Execution:       {exec_count}/{len(questions)} ({exec_count*100//len(questions)}%)")
    print(f"  Data Match:      {match_count}/{len(questions)} ({match_count*100//len(questions)}%)")
    print(f"  Avg Latency:     {avg:.2f}s")
    print(f"  P50 Latency:     {p50:.2f}s")
    print(f"{'='*70}")
    print()

    # Per-question breakdown
    print(f"{'ID':<8s} {'Tier':>4s} {'Exec':>5s} {'Match':>6s} {'Latency':>8s} {'Attempts':>9s} {'Notes'}")
    print(f"{'-'*70}")
    for r in eval_results:
        exec_s = "OK" if r["exec_success"] else "FAIL"
        match_s = "YES" if r["data_match"] else "no"
        notes = ""
        if not r["exec_success"]:
            notes = f"error: {r['error'][:30]}" if r.get("error") else ""
        elif not r["data_match"]:
            notes = f"got {r['generated_result_count']} rows, gold {r['gold_result_count']}"
        print(f"{r['question_id']:<8s} {str(r['tier']):>4s} {exec_s:>5s} {match_s:>6s} "
              f"{r['latency']:>7.2f}s {r['attempts']:>9d} {notes}")
    print(f"{'-'*70}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate agent on dev questions")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model ID to use")
    parser.add_argument("--db", default="data/Chinook.db", help="Database path")
    parser.add_argument("--answers-output", default="dev_answers.json", help="dev_answers.json path")
    parser.add_argument("--report-output", default="eval_report.json", help="eval_report.json path")
    args = parser.parse_args()
    run_eval(
        db_path=args.db,
        model=args.model,
        answers_output=args.answers_output,
        report_output=args.report_output,
    )


if __name__ == "__main__":
    main()
