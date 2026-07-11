"""CI-integrated evaluation pipeline for text-to-SQL.

Runs the eval harness, compares results against a stored baseline, and
exits with non-zero status if accuracy regresses. Designed to run in CI
(GitHub Actions, etc.) on every prompt or model change.

Usage:
    # Run eval and compare against baseline
    python -m src.ci_eval

    # Update the baseline (after intentional prompt changes)
    python -m src.ci_eval --update-baseline

    # Set custom thresholds
    python -m src.ci_eval --min-exec-accuracy 1.0 --min-match-accuracy 0.85 --max-p50-latency 3.0

Exit codes:
    0 = All checks passed (no regression)
    1 = Accuracy regression detected
    2 = Latency regression detected
    3 = Eval execution error
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from src.eval import run_eval
from src.agent import DEFAULT_MODEL


BASELINE_PATH = "eval_baseline.json"
EVAL_REPORT_PATH = "eval_report.json"
DEV_ANSWERS_PATH = "dev_answers.json"

# Default regression thresholds
DEFAULT_MIN_EXEC_ACCURACY = 1.0   # 100% execution accuracy required
DEFAULT_MIN_MATCH_ACCURACY = 0.80  # 80% data match required (allows 2/10 mismatch)
DEFAULT_MAX_P50_LATENCY = 3.0     # P50 must stay under 3s
DEFAULT_MAX_LATENCY_REGRESSION = 0.5  # P50 can't increase by more than 0.5s vs baseline


def load_baseline(path: str = BASELINE_PATH) -> dict | None:
    """Load the stored baseline eval results."""
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def save_baseline(eval_report: dict, path: str = BASELINE_PATH) -> None:
    """Save current eval results as the new baseline."""
    baseline = {
        "summary": eval_report["summary"],
        "model": eval_report["model"],
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "questions": [
            {
                "question_id": q["question_id"],
                "exec_success": q["exec_success"],
                "data_match": q["data_match"],
                "latency": q["latency"],
                "attempts": q["attempts"],
            }
            for q in eval_report["questions"]
        ],
    }
    Path(path).write_text(json.dumps(baseline, indent=2))
    print(f"Baseline saved to {path}")


def check_regression(
    current: dict,
    baseline: dict | None,
    min_exec_accuracy: float,
    min_match_accuracy: float,
    max_p50_latency: float,
    max_latency_regression: float,
) -> list[dict]:
    """
    Compare current eval results against thresholds and baseline.

    Returns a list of issues found (empty list = all passed).
    """
    issues = []
    summary = current["summary"]
    total = summary["total_questions"]

    exec_rate = summary["exec_success"] / total
    match_rate = summary["data_match"] / total
    p50 = summary["p50_latency"]

    # Check absolute thresholds
    if exec_rate < min_exec_accuracy:
        issues.append({
            "type": "accuracy",
            "severity": "critical",
            "message": f"Execution accuracy regression: {exec_rate:.0%} < {min_exec_accuracy:.0%} threshold",
            "current": exec_rate,
            "threshold": min_exec_accuracy,
        })

    if match_rate < min_match_accuracy:
        issues.append({
            "type": "accuracy",
            "severity": "critical",
            "message": f"Data match accuracy regression: {match_rate:.0%} < {min_match_accuracy:.0%} threshold",
            "current": match_rate,
            "threshold": min_match_accuracy,
        })

    if p50 > max_p50_latency:
        issues.append({
            "type": "latency",
            "severity": "warning",
            "message": f"P50 latency exceeds target: {p50:.2f}s > {max_p50_latency:.2f}s threshold",
            "current": p50,
            "threshold": max_p50_latency,
        })

    # Compare against baseline if available
    if baseline:
        b_summary = baseline["summary"]
        b_match = b_summary["data_match"] / b_summary["total_questions"]
        b_p50 = b_summary["p50_latency"]

        if match_rate < b_match:
            issues.append({
                "type": "accuracy",
                "severity": "critical",
                "message": f"Data match regression vs baseline: {match_rate:.0%} < {b_match:.0%} baseline",
                "current": match_rate,
                "baseline": b_match,
            })

        latency_delta = p50 - b_p50
        if latency_delta > max_latency_regression:
            issues.append({
                "type": "latency",
                "severity": "warning",
                "message": f"P50 latency regression vs baseline: {p50:.2f}s vs {b_p50:.2f}s baseline (+{latency_delta:.2f}s)",
                "current": p50,
                "baseline": b_p50,
                "delta": round(latency_delta, 3),
            })

        # Per-question regression check
        baseline_questions = {q["question_id"]: q for q in baseline["questions"]}
        for q in current["questions"]:
            qid = q["question_id"]
            b_q = baseline_questions.get(qid)
            if b_q:
                if b_q["data_match"] and not q["data_match"]:
                    issues.append({
                        "type": "accuracy",
                        "severity": "critical",
                        "message": f"Question {qid} regressed: was matching, now not matching",
                        "question_id": qid,
                    })
                if b_q["exec_success"] and not q["exec_success"]:
                    issues.append({
                        "type": "accuracy",
                        "severity": "critical",
                        "message": f"Question {qid} regressed: was executing, now failing",
                        "question_id": qid,
                    })

    return issues


def print_ci_report(summary: dict, issues: list[dict], baseline: dict | None) -> None:
    """Print a CI-friendly report with clear pass/fail indicators."""
    print("\n" + "=" * 70)
    print("CI EVALUATION REPORT")
    print("=" * 70)

    print(f"\n  Model:          {summary.get('model', 'unknown')}")
    print(f"  Execution:      {summary['exec_rate']} ({summary['exec_success']}/{summary['total_questions']})")
    print(f"  Data Match:     {summary['match_rate']} ({summary['data_match']}/{summary['total_questions']})")
    print(f"  Avg Latency:    {summary['avg_latency']}s")
    print(f"  P50 Latency:    {summary['p50_latency']}s")

    if baseline:
        b = baseline["summary"]
        print(f"\n  Baseline comparison:")
        print(f"    Baseline match:  {b['match_rate']}")
        print(f"    Baseline P50:    {b['p50_latency']}s")

    if not issues:
        print(f"\n  RESULT: ALL CHECKS PASSED")
        print("=" * 70)
        return

    print(f"\n  ISSUES FOUND ({len(issues)}):")
    for issue in issues:
        icon = "CRITICAL" if issue["severity"] == "critical" else "WARNING"
        print(f"    [{icon}] {issue['message']}")
    print("=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CI evaluation pipeline with regression detection")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model ID to evaluate")
    parser.add_argument("--db", default="data/Chinook.db", help="Database path")
    parser.add_argument("--baseline", default=BASELINE_PATH, help="Baseline file path")
    parser.add_argument("--update-baseline", action="store_true", help="Save current results as new baseline")
    parser.add_argument("--min-exec-accuracy", type=float, default=DEFAULT_MIN_EXEC_ACCURACY,
                        help=f"Minimum execution accuracy (default: {DEFAULT_MIN_EXEC_ACCURACY})")
    parser.add_argument("--min-match-accuracy", type=float, default=DEFAULT_MIN_MATCH_ACCURACY,
                        help=f"Minimum data match accuracy (default: {DEFAULT_MIN_MATCH_ACCURACY})")
    parser.add_argument("--max-p50-latency", type=float, default=DEFAULT_MAX_P50_LATENCY,
                        help=f"Maximum P50 latency in seconds (default: {DEFAULT_MAX_P50_LATENCY})")
    parser.add_argument("--max-latency-regression", type=float, default=DEFAULT_MAX_LATENCY_REGRESSION,
                        help=f"Max P50 increase vs baseline (default: {DEFAULT_MAX_LATENCY_REGRESSION}s)")
    args = parser.parse_args()

    load_dotenv()
    if not os.environ.get("FIREWORKS_API_KEY"):
        print("Error: FIREWORKS_API_KEY not set")
        sys.exit(3)

    # Run the eval
    print("Running evaluation...")
    try:
        run_eval(
            db_path=args.db,
            model=args.model,
            answers_output=DEV_ANSWERS_PATH,
            report_output=EVAL_REPORT_PATH,
        )
    except Exception as e:
        print(f"Eval execution error: {e}")
        sys.exit(3)

    # Load results
    with open(EVAL_REPORT_PATH) as f:
        current_report = json.load(f)

    # Update baseline if requested
    if args.update_baseline:
        save_baseline(current_report, args.baseline)
        print("Baseline updated. Exiting without regression check.")
        sys.exit(0)

    # Load baseline
    baseline = load_baseline(args.baseline)
    if not baseline:
        print(f"\nNo baseline found at {args.baseline}. Run with --update-baseline to create one.")
        print("Skipping baseline comparison. Checking absolute thresholds only.")

    # Check for regressions
    issues = check_regression(
        current=current_report,
        baseline=baseline,
        min_exec_accuracy=args.min_exec_accuracy,
        min_match_accuracy=args.min_match_accuracy,
        max_p50_latency=args.max_p50_latency,
        max_latency_regression=args.max_latency_regression,
    )

    # Print report
    print_ci_report(current_report["summary"], issues, baseline)

    # Exit with appropriate code
    if any(i["type"] == "accuracy" and i["severity"] == "critical" for i in issues):
        sys.exit(1)
    if any(i["type"] == "latency" for i in issues):
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
