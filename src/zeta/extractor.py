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


def extract_edges(chunks: list[dict], source_text: str | None = None) -> dict:
    """Extract candidate ownership edges from parsed chunks.

    chunks: [{chunk_id, text, source, metadata{pages,...}}] from Tessera ingest.
    Returns {"edges": [...], "source_hash", "low_confidence": [owner_id,...], "backend"}.
    """
    if not chunks:
        raise ValueError("no chunks to extract from")
    # source_hash anchors every edge to the exact document bytes.
    if source_text is None:
        source_text = "\n".join(c.get("text", "") for c in chunks)
    source_hash = hashlib.sha256(source_text.encode()).hexdigest()

    # Render chunks with page markers so the model can cite pages.
    rendered = []
    for c in chunks:
        pages = c.get("metadata", {}).get("pages") or [1]
        rendered.append(f"[page {pages[0]}] {c.get('text','')}")
    prompt = EXTRACTION_PROMPT.format(chunks="\n".join(rendered))

    out = chat([{"role": "user", "content": prompt}], max_tokens=1500)
    raw = _extract_json_array(out["text"])
    edges = [_validate_edge(e, source_hash) for e in raw]
    low_conf = [e["owner_id"] for e in edges if e["confidence"] < 0.7]
    return {
        "edges": edges,
        "source_hash": source_hash,
        "low_confidence": low_conf,
        "backend": out.get("backend"),
    }


if __name__ == "__main__":
    # Offline deterministic check of validation + JSON parsing (no LLM call).
    sample = '[{"owner_id":"John Smith","owned_entity_id":"Cayman HoldCo","direct_pct":60,"page":3,"confidence":0.92,"owner_type":"person","evidence_text":"John Smith holds 60% of Cayman HoldCo."}]'
    edges = [_validate_edge(e, "deadbeef") for e in _extract_json_array(sample)]
    assert edges[0]["owner_id"] == "john_smith"
    assert edges[0]["direct_pct"] == 60.0
    print("extractor validation OK:", json.dumps(edges[0]))
