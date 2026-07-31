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
from src.questions import load_questions


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


def _write_checkpoint(path, dev_answers, eval_results, trust_flags):
    """Write incremental progress so a long eval run is observable and resumable.

    Writes {completed, total-so-far, exec, match, per-question results} after each
    question. Best-effort: a checkpoint write failure never breaks the eval.
    """
    if not path:
        return
    try:
        exec_n = sum(1 for r in eval_results if r.get("exec_success"))
        match_n = sum(1 for r in eval_results if r.get("data_match"))
        payload = {
            "completed": len(eval_results),
            "exec": exec_n,
            "match": match_n,
            "exec_rate": round(exec_n / len(eval_results), 3) if eval_results else 0,
            "match_rate": round(match_n / len(eval_results), 3) if eval_results else 0,
            "results": eval_results,
            "trust_flags": trust_flags,
        }
        with open(path, "w") as f:
            json.dump(payload, f, default=str)
    except Exception:
        pass


def run_eval(
    db_path: str = "data/Chinook.db",
    questions_path: str = "data/dev_questions_with_answers.json",
    answers_output: str = "dev_answers.json",
    report_output: str = "eval_report.json",
    model: str = DEFAULT_MODEL,
    dataset: str = None,
    checkpoint_path: str = None,
):
    """
    Run the agent on a question set, compare against gold answers,
    and produce dev_answers.json + eval_report.json.

    Args:
        db_path: Path to the SQLite database
        questions_path: Path to dev questions with gold answers (used when dataset is None)
        answers_output: Where to write dev_answers.json (required deliverable)
        report_output: Where to write eval_report.json (detailed metrics)
        model: Model ID to use for evaluation
        dataset: "dev" | "groundtruth" | "all". When set, overrides questions_path
                 and loads via src.questions.load_questions.
    """
    load_dotenv()
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        print("Error: FIREWORKS_API_KEY not set")
        sys.exit(1)

    # Load gold questions
    if dataset:
        questions = load_questions(dataset)
    else:
        with open(questions_path) as f:
            questions = json.load(f)

    conn = load_db(db_path)
    agent = TextToSQLAgent(conn, model=model, api_key=api_key)

    # Default checkpoint path alongside the report (incremental progress file).
    if checkpoint_path is None:
        checkpoint_path = report_output.replace(".json", "") + "_checkpoint.json"

    dev_answers = {}
    eval_results = []
    trust_flags = {}  # question_id -> trust report (for trust_monitor)
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

            # Capture trust report for the trust monitor
            if getattr(resp, "trust", None):
                trust_flags[qid] = resp.trust

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
                "join_complexity": q.get("join_complexity"),
                "failure_modes": q.get("failure_modes") or [],
                "synthetic": q.get("synthetic"),
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

            # Incremental checkpoint: write progress after every question so the
            # run is observable and resumable (survives a crash without losing all work).
            _write_checkpoint(checkpoint_path, dev_answers, eval_results, trust_flags)

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
                "join_complexity": q.get("join_complexity"),
                "failure_modes": q.get("failure_modes") or [],
                "synthetic": q.get("synthetic"),
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

    # Write trust_flags.json (input to trust_monitor)
    trust_out = report_output.replace(".json", "") + "_trust_flags.json"
    with open(trust_out, "w") as f:
        json.dump(trust_flags, f, indent=2, default=str)
    print(f"trust flags written to {trust_out} ({len(trust_flags)} questions)")

    # Breakdowns by tier and join_complexity
    def _breakdown(key):
        groups = {}
        for r in eval_results:
            k = r.get(key)
            if k is None:
                continue
            g = groups.setdefault(k, {"total": 0, "exec": 0, "match": 0})
            g["total"] += 1
            g["exec"] += 1 if r["exec_success"] else 0
            g["match"] += 1 if r["data_match"] else 0
        return {
            str(k): {
                "total": v["total"],
                "exec_rate": f"{v['exec']}/{v['total']}",
                "match_rate": f"{v['match']}/{v['total']}",
                "match_pct": round(100 * v["match"] / v["total"], 1) if v["total"] else 0,
            }
            for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))
        }

    # Write eval_report.json
    from datetime import datetime, timezone
    report = {
        "model": model,
        "dataset": dataset or "custom",
        "run_at": datetime.now(timezone.utc).isoformat(),  # embed timestamp of the actual run
        "summary": {
            "total_questions": len(questions),
            "exec_success": exec_count,
            "data_match": match_count,
            "exec_rate": f"{exec_count}/{len(questions)}",
            "match_rate": f"{match_count}/{len(questions)}",
            "avg_latency": round(avg, 3),
            "p50_latency": round(p50, 3),
            "by_tier": _breakdown("tier"),
            "by_join_complexity": _breakdown("join_complexity"),
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
    parser.add_argument("--dataset", default=None, choices=["dev", "groundtruth", "all"],
                        help="Question set to load via src.questions (overrides default questions_path)")
    args = parser.parse_args()
    run_eval(
        db_path=args.db,
        model=args.model,
        answers_output=args.answers_output,
        report_output=args.report_output,
        dataset=args.dataset,
    )


if __name__ == "__main__":
    main()
