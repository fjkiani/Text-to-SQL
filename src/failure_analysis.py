"""
Failure analysis for the groundTruth eval: cluster eval failures by
join_complexity and failure_mode to target few-shot refinement.

Reads the groundTruth eval report and emits:
  - match rate by tier and by join_complexity
  - the failure clusters (join_complexity x failure_mode) with the lowest match rates
  - the specific failed questions in the worst clusters (for few-shot targeting)

Usage:
  python -m src.failure_analysis --report groundtruth_eval_report.json --out failure_analysis.json
"""
import json
import argparse
from collections import defaultdict


def analyze(report: dict) -> dict:
    questions = report.get("questions", [])
    if not questions:
        raise ValueError("report has no 'questions' entries")

    def rate(rows):
        n = len(rows)
        m = sum(1 for r in rows if r.get("data_match"))
        e = sum(1 for r in rows if r.get("exec_success"))
        return {"n": n, "exec": e, "match": m,
                "exec_rate": round(e / n, 3) if n else None,
                "match_rate": round(m / n, 3) if n else None}

    by_tier = defaultdict(list)
    by_jc = defaultdict(list)
    by_cluster = defaultdict(list)  # (jc, failure_mode)
    failed = []

    for r in questions:
        tier = r.get("tier")
        jc = r.get("join_complexity") or "L?"
        fms = r.get("failure_modes") or []
        by_tier[tier].append(r)
        by_jc[jc].append(r)
        for m in fms:
            by_cluster[(jc, m)].append(r)
        if not r.get("data_match"):
            failed.append(r)

    # Rank clusters by worst match rate (min n=3 to be meaningful)
    cluster_stats = []
    for (jc, m), rows in by_cluster.items():
        s = rate(rows)
        if s["n"] >= 3:
            cluster_stats.append({"join_complexity": jc, "failure_mode": m, **s})
    cluster_stats.sort(key=lambda x: (x["match_rate"] if x["match_rate"] is not None else 1))

    # Worst individual failures for few-shot targeting
    failed_detail = [
        {"question_id": r.get("question_id"), "question": r.get("question"),
         "tier": r.get("tier"), "join_complexity": r.get("join_complexity"),
         "failure_modes": r.get("failure_modes"),
         "generated_sql": r.get("generated_sql"), "gold_sql": r.get("gold_sql"),
         "error": r.get("error")}
        for r in failed
    ]

    return {
        "total": len(questions),
        "overall": rate(questions),
        "by_tier": {str(k): rate(v) for k, v in sorted(by_tier.items(), key=lambda kv: str(kv[0]))},
        "by_join_complexity": {k: rate(v) for k, v in sorted(by_jc.items())},
        "worst_clusters": cluster_stats[:15],
        "failed_count": len(failed),
        "failed_questions": failed_detail,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="groundtruth_eval_report.json")
    ap.add_argument("--out", default="failure_analysis.json")
    args = ap.parse_args()
    report = json.load(open(args.report))
    out = analyze(report)
    json.dump(out, open(args.out, "w"), indent=2)

    print(f"Total: {out['total']}  overall match: {out['overall']['match_rate']}  exec: {out['overall']['exec_rate']}")
    print("\nBy join_complexity:")
    for jc, s in out["by_join_complexity"].items():
        print(f"  {jc}: n={s['n']} match={s['match_rate']} exec={s['exec_rate']}")
    print("\nWorst clusters (n>=3):")
    for c in out["worst_clusters"][:10]:
        print(f"  {c['join_complexity']} {c['failure_mode']}: n={c['n']} match={c['match_rate']}")
    print(f"\nFailed: {out['failed_count']}  -> {args.out}")


if __name__ == "__main__":
    main()
