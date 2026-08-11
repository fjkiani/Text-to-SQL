"""
Shared entity canonicalization + resolution for Zeta Clearance.

Single source of truth: L1 extraction and the L2 UBO graph MUST canonicalize
identically, or the graph silently drops edges and reports "no UBO found" for an
entity that plainly has one. That is the most dangerous failure mode in KYB —
a false clear — so it is handled here rather than in either caller.

Two distinct operations, deliberately kept separate:

1. canonical_id(s)   -> stable id. Lowercase, strip parenthetical/bracketed
   annotations ("(individual)", "(natural person)"), strip punctuation, collapse
   separators. Does NOT strip legal-entity suffixes, because "Blue Harbour Trust"
   and "Blue Harbour Ltd" are different legal persons and must not collapse.

2. resolve_entity(query, known) -> exact match, else unique suffix-stripped
   match, else ambiguity. Never guesses between multiple candidates: an ambiguous
   reference returns None with the candidate list so the caller can flag for
   human review instead of picking one and clearing the wrong entity.
"""
import re

# Legal-form suffixes used ONLY for resolution, never for id construction.
LEGAL_SUFFIXES = {
    "ltd", "limited", "llc", "lllc", "inc", "incorporated", "corp", "corporation",
    "plc", "gmbh", "ag", "sa", "sas", "sarl", "bv", "nv", "pte", "pty", "co",
    "company", "lp", "llp", "kg", "oy", "ab", "as", "aps", "spa", "srl", "kk",
    "holdings", "holding", "group",
}

_PARENS = re.compile(r"[\(\[\{][^\)\]\}]*[\)\]\}]")
_NONWORD = re.compile(r"[^a-z0-9]+")


def canonical_id(s: str) -> str:
    """Stable canonical id. Deterministic and idempotent."""
    t = str(s).strip().lower()
    t = _PARENS.sub(" ", t)          # drop "(individual)", "(natural person)"
    t = _NONWORD.sub("_", t)         # punctuation -> separator
    return t.strip("_")


def resolution_key(s: str) -> str:
    """Suffix-stripped key used only to match aliases of the same legal person."""
    cid = canonical_id(s)
    parts = [p for p in cid.split("_") if p]
    while len(parts) > 1 and parts[-1] in LEGAL_SUFFIXES:
        parts.pop()
    return "_".join(parts)


def resolve_entity(query: str, known_ids) -> tuple[str | None, list[str]]:
    """
    Resolve a user/query entity reference against ids present in the graph.

    Returns (resolved_id, candidates).
      - exact canonical hit          -> (id, [id])
      - exactly one alias hit        -> (id, [id])
      - several alias hits           -> (None, [ids...])   caller must flag
      - none                         -> (None, [])
    """
    known = list(dict.fromkeys(known_ids))
    q = canonical_id(query)
    if q in known:
        return q, [q]
    qk = resolution_key(query)
    if not qk:
        return None, []
    cands = [k for k in known if resolution_key(k) == qk]
    if len(cands) == 1:
        return cands[0], cands
    return None, cands


if __name__ == "__main__":
    cases = [
        ("Maria Garcia (individual)", "maria_garcia"),
        ("John Smith (natural person)", "john_smith"),
        ("Cayman HoldCo Ltd.", "cayman_holdco_ltd"),
        ("ACME OPCO LTD", "acme_opco_ltd"),
        ("Blue Harbour Trust", "blue_harbour_trust"),
    ]
    for raw, want in cases:
        got = canonical_id(raw)
        assert got == want, f"{raw!r} -> {got!r} != {want!r}"
    known = ["acme_opco_ltd", "cayman_holdco_ltd", "maria_garcia", "john_smith", "blue_harbour_trust"]
    assert resolve_entity("acme_opco", known)[0] == "acme_opco_ltd"
    assert resolve_entity("Acme OpCo Ltd", known)[0] == "acme_opco_ltd"
    assert resolve_entity("Maria Garcia (individual)", known)[0] == "maria_garcia"
    # over-merge guard: Trust vs Ltd must NOT collapse
    assert canonical_id("Blue Harbour Trust") != canonical_id("Blue Harbour Ltd")
    # ambiguity must not silently pick
    amb = ["northwind_ltd", "northwind_llc"]
    rid, cands = resolve_entity("Northwind", amb)
    assert rid is None and set(cands) == set(amb), (rid, cands)
    print("CANON OK — aliasing fixed, over-merge blocked, ambiguity flagged")
