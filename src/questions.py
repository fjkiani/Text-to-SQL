"""
Shared question-set loader.

Normalizes the two on-disk schemas (dev_questions_with_answers.json and
groundTruth.json) into a single superset record shape so eval.py, benchmark.py,
and trust_monitor.py can consume either dataset interchangeably.

Superset record:
    id, question, tier, gold_sql, gold_answer, expected_result,
    evaluation, failure_modes, join_complexity, synthetic

Missing fields in the dev set are filled with None / defaults.
"""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.normpath(os.path.join(_HERE, "..", "data"))

DEV_PATH = os.path.join(_DATA, "dev_questions_with_answers.json")
GROUNDTRUTH_PATH = os.path.join(_DATA, "groundTruth.json")

_SUPERSET_FIELDS = (
    "id", "question", "tier", "gold_sql", "gold_answer", "expected_result",
    "evaluation", "failure_modes", "join_complexity", "synthetic",
)


def _normalize(rec: dict, default_synthetic: bool) -> dict:
    out = {k: rec.get(k) for k in _SUPERSET_FIELDS}
    out["tier"] = rec.get("tier", "?")
    out["evaluation"] = rec.get("evaluation", "sql_result_match")
    out["failure_modes"] = rec.get("failure_modes") or []
    out["join_complexity"] = rec.get("join_complexity")
    out["synthetic"] = rec.get("synthetic", default_synthetic)
    return out


def _load(path: str, default_synthetic: bool) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return [_normalize(r, default_synthetic) for r in data]


def load_questions(dataset: str = "all") -> list[dict]:
    """
    Load a question set.

    dataset: "dev" | "groundtruth" | "all"
      - "dev": the original 10 dev questions (synthetic=False)
      - "groundtruth": the 282-record groundTruth set
      - "all": dev + groundtruth, deduplicated by id
    """
    dataset = (dataset or "all").lower()
    if dataset == "dev":
        return _load(DEV_PATH, default_synthetic=False)
    if dataset == "groundtruth":
        return _load(GROUNDTRUTH_PATH, default_synthetic=True)
    if dataset == "all":
        seen = {}
        for rec in _load(DEV_PATH, False) + _load(GROUNDTRUTH_PATH, True):
            seen[rec["id"]] = rec
        return list(seen.values())
    raise ValueError(f"unknown dataset: {dataset!r} (expected dev|groundtruth|all)")


if __name__ == "__main__":
    import sys
    ds = sys.argv[1] if len(sys.argv) > 1 else "all"
    qs = load_questions(ds)
    from collections import Counter
    print(f"dataset={ds} count={len(qs)}")
    print("  tiers:", dict(Counter(q["tier"] for q in qs)))
    print("  synthetic:", sum(1 for q in qs if q["synthetic"]))
