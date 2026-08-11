"""
Zeta Clearance — L1 constrained ownership-edge extractor.

The LLM PROPOSES candidate ownership edges from parsed document chunks. It NEVER
computes indirect/cumulative ownership (that is the deterministic graph engine's
job). Every edge carries a page citation + confidence + verbatim evidence span so
a human (or the trust harness) can audit it.

Output conforms to specs/edges.json. Edges with confidence < 0.7 are flagged for
human review, not silently dropped.
"""
from __future__ import annotations

import hashlib
import json
import re

# Reuse the Tessera LLM client (Arctic-first, Fireworks fallback). Do NOT rebuild.
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
try:
    from src.llm.client import chat  # noqa: E402
except ImportError:  # running with src/ itself on the path
    from llm.client import chat  # noqa: E402

EXTRACTION_PROMPT = """You are a KYB ownership-extraction engine. Read the document chunks below and extract EVERY direct ownership relationship as JSON.

For EACH ownership fact, output an object with EXACTLY these fields:
- owner_id: canonical id of the owner (a person or an entity). Lowercase, spaces->underscores.
- owned_entity_id: canonical id of the entity being owned.
- direct_pct: the DIRECT ownership percentage on this edge ONLY (a number 0-100). Do NOT multiply across hops.
- page: the 1-indexed page number where this fact appears (use the chunk's page if given, else 1).
- confidence: your confidence 0.0-1.0 that this extraction is correct.
- owner_type: one of "person", "entity", "trust", "nominee", "unknown".
- evidence_text: the VERBATIM sentence/span from the document stating this ownership.

RULES:
- Output ONLY a JSON array of these objects. No prose, no markdown fences.
- Do NOT compute indirect or cumulative ownership. Direct edges only.
- If a percentage is ambiguous, lower the confidence.
- If no ownership facts exist, output [].

DOCUMENT CHUNKS:
{chunks}
"""


import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from canon import canonical_id as _canon

def _canon_unused(s: str) -> str:
    return "_".join(str(s).strip().lower().split())


def _extract_json_array(text: str) -> list:
    """
    Pull the ownership-edge JSON array out of model output.

    A greedy r"\[.*\]" is wrong here and caused a real production failure:
    reasoning models emit chain-of-thought containing bracketed tokens such as
    "[chunk c1 | page 1]", so the greedy span started inside prose and the parse
    died at char 1. This scanner instead walks the text, tracks bracket depth
    while respecting string literals and escapes, and returns the first BALANCED
    array that both parses and looks like edge records.
    """
    # Reasoning models wrap their scratchpad; drop it before scanning.
    cleaned = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"```(?:json)?", " ", cleaned)

    candidates: list[list] = []
    for start in (i for i, ch in enumerate(cleaned) if ch == "["):
        depth, in_str, esc = 0, False, False
        for j in range(start, len(cleaned)):
            c = cleaned[j]
            if esc:
                esc = False
                continue
            if c == "\\":
                esc = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(cleaned[start:j + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, list):
                        candidates.append(parsed)
                    break
        if candidates and candidates[-1] and isinstance(candidates[-1][0], dict) \
                and "owner_id" in candidates[-1][0]:
            return candidates[-1]  # first well-formed EDGE array wins
    for c in candidates:  # fall back to any parsed array (possibly empty)
        if all(isinstance(x, dict) for x in c):
            return c
    raise ValueError(
        f"no balanced edge JSON array in model output (len={len(text)}): {text[:300]}"
    )


def _validate_edge(e: dict, source_hash: str) -> dict:
    required = ["owner_id", "owned_entity_id", "direct_pct", "page", "confidence"]
    for k in required:
        if k not in e:
            raise ValueError(f"extracted edge missing field {k}: {e}")
    pct = float(e["direct_pct"])
    if not (0 <= pct <= 100):
        raise ValueError(f"direct_pct out of range: {pct}")
    return {
        "owner_id": _canon(e["owner_id"]),
        "owned_entity_id": _canon(e["owned_entity_id"]),
        "direct_pct": pct,
        "source_hash": source_hash,
        "page": int(e.get("page", 1)),
        "confidence": float(e["confidence"]),
        "owner_type": e.get("owner_type", "unknown"),
        "evidence_text": e.get("evidence_text", ""),
    }


def _render(c: dict) -> str:
    pages = c.get("metadata", {}).get("pages") or [1]
    return f"[page {pages[0]}] {c.get('text','')}"


def _batch(chunks: list[dict], budget: int) -> list[list[dict]]:
    """
    Group chunks into prompt-sized batches.

    One request per chunk is wasteful when a data room is 400 small chunks, and
    one request for everything blows the context window and the gateway timeout.
    Batch by rendered character budget: bounded prompt size, bounded request
    count. A single chunk larger than the budget still gets its own batch rather
    than being dropped.
    """
    batches: list[list[dict]] = []
    cur: list[dict] = []
    cur_len = 0
    for c in chunks:
        n = len(_render(c))
        if cur and cur_len + n > budget:
            batches.append(cur)
            cur, cur_len = [], 0
        cur.append(c)
        cur_len += n
    if cur:
        batches.append(cur)
    return batches


def _merge_edges(batched: list[list[dict]]) -> tuple[list[dict], list[dict]]:
    """
    Merge per-batch edge lists, keyed on (owner_id, owned_entity_id).

    Two batches can report the same relationship with DIFFERENT percentages —
    e.g. a stale cap table on page 2 and an amended one on page 9. Averaging
    them manufactures a number that appears in no document and can move a
    beneficiary across the 25% control threshold. So both observations are kept
    in an explicit conflict record and one is selected by a fixed rule.

    SELECTION RULE (deterministic, and it must be):
      1. higher confidence wins;
      2. on equal confidence, the LARGER direct_pct wins.

    Rule 2 is not cosmetic. An earlier version fell through to first-seen, and
    because batches complete concurrently the winner depended on which HTTP
    response landed first: the same documents yielded john_smith at 24.0% or
    16.0% of the top entity across runs. A clearance decision that changes with
    network timing is not auditable. Taking the maximum is also the fail-safe
    direction for KYB — when the record contradicts itself, over-report the
    stake so the graph escalates a possible controller rather than clearing
    them — and every conflict is emitted for human adjudication regardless.
    """
    best: dict[tuple[str, str], dict] = {}
    conflicts: dict[tuple[str, str], dict] = {}
    for edges in batched:
        for e in edges:
            k = (e["owner_id"], e["owned_entity_id"])
            prev = best.get(k)
            if prev is None:
                best[k] = e
                continue
            if abs(prev["direct_pct"] - e["direct_pct"]) > 1e-9:
                rec = conflicts.setdefault(
                    k,
                    {
                        "owner_id": e["owner_id"],
                        "owned_entity_id": e["owned_entity_id"],
                        "values": [],
                        "pages": [],
                    },
                )
                for src in (prev, e):
                    if src["direct_pct"] not in rec["values"]:
                        rec["values"].append(src["direct_pct"])
                    if src["page"] not in rec["pages"]:
                        rec["pages"].append(src["page"])
            # Strict, total ordering -> independent of arrival order.
            if (e["confidence"], e["direct_pct"]) > (prev["confidence"], prev["direct_pct"]):
                best[k] = e
    for rec in conflicts.values():
        rec["values"].sort()
        rec["pages"].sort()
    ordered = sorted(conflicts.values(), key=lambda r: (r["owned_entity_id"], r["owner_id"]))
    return list(best.values()), ordered


def _extract_one(batch: list[dict], source_hash: str) -> tuple[list[dict], dict]:
    prompt = EXTRACTION_PROMPT.format(chunks="\n".join(_render(c) for c in batch))
    out = chat([{"role": "user", "content": prompt}], max_tokens=1500)
    raw = _extract_json_array(out["text"])
    return [_validate_edge(e, source_hash) for e in raw], out


def extract_edges(
    chunks: list[dict],
    source_text: str | None = None,
    batch_chars: int | None = None,
    max_workers: int | None = None,
) -> dict:
    """Extract candidate ownership edges from parsed chunks.

    chunks: [{chunk_id, text, source, metadata{pages,...}}] from Tessera ingest.
    Large inputs are split into prompt-sized batches and extracted concurrently,
    so wall time scales with the largest batch rather than the whole data room.

    Returns {"edges", "source_hash", "low_confidence", "backend", "model",
             "batches", "conflicts"}.
    """
    if not chunks:
        raise ValueError("no chunks to extract from")
    # source_hash anchors every edge to the exact document bytes.
    if source_text is None:
        source_text = "\n".join(c.get("text", "") for c in chunks)
    source_hash = hashlib.sha256(source_text.encode()).hexdigest()

    budget = int(batch_chars or os.environ.get("ZETA_EXTRACT_BATCH_CHARS", "12000"))
    workers = int(max_workers or os.environ.get("ZETA_EXTRACT_WORKERS", "4"))
    batches = _batch(chunks, budget)

    results: list[list[dict]] = []
    meta: dict = {}
    errors: list[str] = []

    if len(batches) == 1:
        edges, out = _extract_one(batches[0], source_hash)
        results.append(edges)
        meta = out
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(workers, len(batches))) as pool:
            futs = [pool.submit(_extract_one, b, source_hash) for b in batches]
            # Collect in BATCH ORDER, not completion order. as_completed() would
            # make the merge input sequence depend on network timing; combined
            # with any order-sensitive tie-break that yields different clearance
            # results for identical documents.
            for i, f in enumerate(futs):
                try:
                    edges, out = f.result()
                except Exception as exc:  # one bad batch must not void the run
                    errors.append(f"batch{i}:{type(exc).__name__}:{str(exc)[:120]}")
                    continue
                results.append(edges)
                meta = meta or out
        # A silently-dropped batch is a silently-dropped owner, which is a false
        # clear. Fail loudly if nothing survived; report partials otherwise.
        if not results:
            raise RuntimeError(f"all {len(batches)} extraction batches failed: {errors}")

    edges, conflicts = _merge_edges(results)
    edges.sort(key=lambda e: (e["owned_entity_id"], -e["direct_pct"], e["owner_id"]))
    low_conf = [e["owner_id"] for e in edges if e["confidence"] < 0.7]
    result = {
        "edges": edges,
        "source_hash": source_hash,
        "low_confidence": low_conf,
        "backend": meta.get("backend"),
        "model": meta.get("model"),
        "batches": len(batches),
        "conflicts": conflicts,
    }
    if errors:
        result["failed_batches"] = errors
    return result


if __name__ == "__main__":
    # Offline deterministic check of validation + JSON parsing (no LLM call).
    sample = '[{"owner_id":"John Smith","owned_entity_id":"Cayman HoldCo","direct_pct":60,"page":3,"confidence":0.92,"owner_type":"person","evidence_text":"John Smith holds 60% of Cayman HoldCo."}]'
    edges = [_validate_edge(e, "deadbeef") for e in _extract_json_array(sample)]
    assert edges[0]["owner_id"] == "john_smith"
    assert edges[0]["direct_pct"] == 60.0
    print("extractor validation OK:", json.dumps(edges[0]))
