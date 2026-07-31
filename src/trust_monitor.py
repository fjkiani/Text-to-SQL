"""
Trust-layer monitoring: measure how well the trust layer's flags detect the
failure modes annotated in the groundTruth dataset.

For each eval question we know:
  - the annotated failure_modes (what could go wrong), e.g. JN, GA, NL, DT, HL, SI, NH, FL
  - whether the agent actually got the answer right (data_match)
  - the trust flags the trust layer raised (join_path, row_count, summary, ...)

Monitoring question: when the agent FAILS a question annotated with a given
failure mode, does the trust layer raise the corresponding flag category?
That is the recall of the trust layer as a failure detector. We also report
precision: of all times a category fired, how often was the question actually
wrong.

Mapping rationale (failure_mode -> trust category):
  JN (join error)        -> join_path      : wrong/unvalidated joins are exactly join_path flags
  GA (wrong aggregation) -> summary        : aggregation mismatches surface as summary-vs-data contradictions
  NL (nested logic)      -> provenance     : novel nested/EXISTS/window shapes diverge from known patterns
  DT (date handling)     -> summary        : date-bucket errors show up as wrong numbers in the summary
  HL (null/anti-join)    -> join_path      : LEFT JOIN anti-joins are join-shape signals
  SI (set operation)     -> provenance     : EXISTS/NOT EXISTS/IN set ops diverge from exemplar patterns
  NH (null handling)     -> row_count      : COALESCE/zero-row cases manifest as row-count anomalies
  FL (filter error)      -> row_count      : wrong WHERE filters change result cardinality

This mapping is a hypothesis, not ground truth. The report computes detection
metrics PER category so we can see where the mapping holds and where the trust
layer is blind. That is the point of the monitor: find the blind spots.
"""
import json
import argparse
from collections import defaultdict

FAILURE_MODE_TO_CATEGORY = {
    "JN": "join_path",
    "GA": "summary",
    "NL": "provenance",
    "DT": "summary",
    "HL": "join_path",
    "SI": "provenance",
    "NH": "row_count",
    "FL": "row_count",
}

ALL_CATEGORIES = ["join_path", "row_count", "summary", "provenance", "self_healing", "agent_control"]


def _flag_categories(trust: dict) -> set:
    """Extract the set of flag categories raised on a single question."""
    cats = set()
    for f in (trust or {}).get("flags", []):
        sev = f.get("severity")
        # count only real problems, not 'ok' informational flags
        if sev in ("warning", "critical"):
            cats.add(f.get("category"))
    return cats


def build_trust_report(eval_results: list[dict], trust_results: dict[str, dict]) -> dict:
    """
    eval_results: list of per-question dicts from eval_report.json["questions"]
                  (must include question_id, data_match, failure_modes).
    trust_results: {question_id: trust_dict} with trust_dict containing "flags".
    """
    # Per-category detection metrics
    cat_tp = defaultdict(int)  # category fired AND question was wrong
    cat_fp = defaultdict(int)  # category fired AND question was right
    cat_fn = defaultdict(int)  # question was wrong AND category did NOT fire (but was expected)
    # Per-failure-mode: did we detect actual failures of that mode?
    fm_total = defaultdict(int)
    fm_failed = defaultdict(int)
    fm_detected = defaultdict(int)

    n_with_trust = 0
    for r in eval_results:
        qid = r["question_id"]
        trust = trust_results.get(qid)
        if trust is None:
            continue
        n_with_trust += 1
        correct = bool(r.get("data_match"))
        fired = _flag_categories(trust)
        expected_cats = {FAILURE_MODE_TO_CATEGORY[m] for m in (r.get("failure_modes") or []) if m in FAILURE_MODE_TO_CATEGORY}

        # category-level precision/recall vs actual correctness
        for cat in ALL_CATEGORIES:
            if cat in fired and not correct:
                cat_tp[cat] += 1
            elif cat in fired and correct:
                cat_fp[cat] += 1
            elif cat not in fired and not correct and cat in expected_cats:
                cat_fn[cat] += 1

        # failure-mode-level: when a question with mode M actually failed, was the mapped category raised?
        for m in (r.get("failure_modes") or []):
            fm_total[m] += 1
            if not correct:
                fm_failed[m] += 1
                mapped = FAILURE_MODE_TO_CATEGORY.get(m)
                if mapped and mapped in fired:
                    fm_detected[m] += 1

    def pr(cat):
        tp, fp, fn = cat_tp[cat], cat_fp[cat], cat_fn[cat]
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        return {"tp": tp, "fp": fp, "fn": fn,
                "precision": round(precision, 3) if precision is not None else None,
                "recall": round(recall, 3) if recall is not None else None}

    by_category = {cat: pr(cat) for cat in ALL_CATEGORIES}
    by_failure_mode = {
        m: {
            "annotated": fm_total[m],
            "actually_failed": fm_failed[m],
            "detected_when_failed": fm_detected[m],
            "detection_recall": round(fm_detected[m] / fm_failed[m], 3) if fm_failed[m] else None,
            "mapped_category": FAILURE_MODE_TO_CATEGORY.get(m),
        }
        for m in sorted(fm_total)
    }

    # Blind spots: failure modes that fail often but are rarely detected
    blind_spots = [
        m for m, v in by_failure_mode.items()
        if v["actually_failed"] >= 3 and (v["detection_recall"] or 0) < 0.5
    ]

    return {
        "questions_with_trust": n_with_trust,
        "by_category": by_category,
        "by_failure_mode": by_failure_mode,
        "blind_spots": blind_spots,
        "mapping": FAILURE_MODE_TO_CATEGORY,
    }


def main():
    parser = argparse.ArgumentParser(description="Trust-layer failure-detection monitor")
    parser.add_argument("--eval", default="eval_report.json", help="eval_report.json path")
    parser.add_argument("--trust", default="trust_flags.json", help="{question_id: trust} JSON")
    parser.add_argument("--out", default="trust_monitor_report.json", help="output path")
    args = parser.parse_args()

    with open(args.eval) as f:
        eval_report = json.load(f)
    eval_results = eval_report.get("questions", eval_report if isinstance(eval_report, list) else [])

    with open(args.trust) as f:
        trust_results = json.load(f)

    report = build_trust_report(eval_results, trust_results)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    # Console summary
    print(f"Trust monitor: {report['questions_with_trust']} questions with trust data")
    print(f"{'Category':<16s} {'Prec':>6s} {'Recall':>7s}  (tp/fp/fn)")
    for cat, m in report["by_category"].items():
        p = f"{m['precision']}" if m['precision'] is not None else "  -"
        r = f"{m['recall']}" if m['recall'] is not None else "  -"
        print(f"  {cat:<14s} {p:>6s} {r:>7s}  ({m['tp']}/{m['fp']}/{m['fn']})")
    print("\nFailure-mode detection recall (when question actually failed):")
    for m, v in report["by_failure_mode"].items():
        dr = v["detection_recall"]
        print(f"  {m} ({v['mapped_category']}): {v['detected_when_failed']}/{v['actually_failed']} = {dr}")
    if report["blind_spots"]:
        print(f"\nBLIND SPOTS (fail often, rarely detected): {', '.join(report['blind_spots'])}")
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
