"""
Zeta Clearance — deterministic UBO graph engine.

The LLM PROPOSES candidate ownership edges (with page citations + confidence).
This engine DISPOSES: it builds the ownership graph and computes indirect
ownership DETERMINISTICALLY by multiplying percentages across hops. The LLM is
never the sole ownership calculator.

Canonical rule (from the product spec):
    Person A owns 60% of HoldCo; HoldCo owns 40% of OpCo
    -> A's indirect OpCo ownership = 60% x 40% = 24%.

Edges are directed owner -> owned with a direct_pct weight. A "natural person"
is an owner with owner_type == "person" (or an owner that is never itself owned
and is not flagged as an entity/trust/nominee).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

DEFAULT_THRESHOLD_PCT = 25.0
LOW_CONFIDENCE = 0.7
# Sum of direct percentages into an entity "should" be ~100 for a fully
# explained cap table; we tolerate rounding gaps below this before flagging.
EXPLAINED_TOTAL_TOLERANCE = 99.0


def indirect_ownership(path_percentages: list[float]) -> float:
    """Product of per-hop percentages, returned as a percentage rounded to 6dp.

    indirect_ownership([60, 40]) == 24.0
    """
    result = 1.0
    for pct in path_percentages:
        result *= pct / 100.0
    return round(result * 100.0, 6)


@dataclass
class Edge:
    owner_id: str
    owned_entity_id: str
    direct_pct: float
    source_hash: str = ""
    page: int = 0
    confidence: float = 1.0
    owner_type: str = "unknown"  # person | entity | trust | nominee | unknown
    evidence_text: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Edge":
        return cls(
            owner_id=_canon(d["owner_id"]),
            owned_entity_id=_canon(d["owned_entity_id"]),
            direct_pct=float(d["direct_pct"]),
            source_hash=d.get("source_hash", ""),
            page=int(d.get("page", 0)),
            confidence=float(d.get("confidence", 1.0)),
            owner_type=d.get("owner_type", "unknown"),
            evidence_text=d.get("evidence_text", ""),
        )


import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from canon import canonical_id as _canon, resolve_entity as _resolve_entity  # shared w/ L1


class UBOGraph:
    def __init__(self, edges: list[Edge], threshold_pct: float = DEFAULT_THRESHOLD_PCT):
        self.edges = edges
        self.threshold_pct = threshold_pct
        # adjacency: owned_entity -> list of (owner_id, direct_pct, edge)
        self._owners_of: dict[str, list[tuple[str, float, Edge]]] = {}
        # owner_type lookup
        self._type: dict[str, str] = {}
        self.flags: list[str] = []
        self._flag_set: set[str] = set()
        self._build()

    def _add_flag(self, flag: str) -> None:
        if flag not in self._flag_set:
            self._flag_set.add(flag)
            self.flags.append(flag)

    def _build(self) -> None:
        for e in self.edges:
            self._owners_of.setdefault(e.owned_entity_id, []).append(
                (e.owner_id, e.direct_pct, e)
            )
            # record the most specific owner_type seen
            if e.owner_type != "unknown" or e.owner_id not in self._type:
                self._type[e.owner_id] = e.owner_type
            self._type.setdefault(e.owned_entity_id, "entity")
            if e.confidence < LOW_CONFIDENCE:
                self._add_flag("low_confidence_edge")

        # --- deterministic structural flags ---
        # 1. conflicting_documents: same (owner, owned) with differing pct from
        #    different source hashes.
        seen: dict[tuple[str, str], dict[str, float]] = {}
        for e in self.edges:
            key = (e.owner_id, e.owned_entity_id)
            seen.setdefault(key, {})
            if e.source_hash in seen[key] and seen[key][e.source_hash] != e.direct_pct:
                pass
            seen[key][e.source_hash] = e.direct_pct
        for (owner, owned), by_src in seen.items():
            if len(set(by_src.values())) > 1:
                self._add_flag("conflicting_documents")

        # 2. ownership_over_100 / missing_share_total per owned entity.
        for owned, owners in self._owners_of.items():
            total = sum(p for _, p, _ in owners)
            if total > 100.000001:
                self._add_flag("ownership_over_100")
            # only flag missing total if the entity has SOME explained ownership
            # but it clearly does not add up (cap table gap)
            if owners and total < EXPLAINED_TOTAL_TOLERANCE:
                self._add_flag("missing_share_total")

        # 3. circular_ownership: a cycle in the owner->owned graph.
        if self._has_cycle():
            self._add_flag("circular_ownership")

        # 4. nominee_structure present anywhere.
        if any(t == "nominee" for t in self._type.values()):
            self._add_flag("nominee_structure")

    def _has_cycle(self) -> bool:
        # DFS over owner->owned edges
        graph: dict[str, list[str]] = {}
        for e in self.edges:
            graph.setdefault(e.owner_id, []).append(e.owned_entity_id)
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n: WHITE for n in graph}

        def dfs(u: str) -> bool:
            color[u] = GRAY
            for v in graph.get(u, []):
                if color.get(v, WHITE) == GRAY:
                    return True
                if color.get(v, WHITE) == WHITE and dfs(v):
                    return True
            color[u] = BLACK
            return False

        return any(color[n] == WHITE and dfs(n) for n in list(graph))

    def _is_owned(self, node: str) -> bool:
        return node in self._owners_of

    def _node_type(self, node: str) -> str:
        return self._type.get(node, "unknown")

    def _is_natural_person(self, node: str) -> bool:
        """A natural person = explicitly typed 'person', OR an owner that is
        never itself owned and is not an entity/trust/nominee."""
        t = self._node_type(node)
        if t == "person":
            return True
        if t in ("entity", "trust", "nominee"):
            return False
        # unknown type: treat as person iff it is a root owner (never owned)
        return not self._is_owned(node)

    def _trace_to_persons(
        self, entity: str
    ) -> tuple[dict[str, list[dict]], bool]:
        """Walk UP the ownership chain from `entity` to natural persons.

        Returns (person_paths, found_unresolved_corporate).
        person_paths: person_id -> list of {chain, pcts, path_pct}
        """
        person_paths: dict[str, list[dict]] = {}
        unresolved_corporate = False

        def walk(node: str, chain: list[str], pcts: list[float], visited: set[str]):
            nonlocal unresolved_corporate
            owners = self._owners_of.get(node, [])
            if not owners:
                # `node` is a root. If it is itself the entity we're tracing and
                # has no owners, nothing to do.
                return
            for owner_id, pct, edge in owners:
                if owner_id in visited:
                    # circular — already flagged globally; stop this path.
                    continue
                new_chain = [owner_id] + chain
                new_pcts = [pct] + pcts
                if self._is_natural_person(owner_id):
                    person_paths.setdefault(owner_id, []).append(
                        {
                            "chain": new_chain,
                            "pcts": new_pcts,
                            "path_pct": indirect_ownership(new_pcts),
                        }
                    )
                else:
                    # corporate/trust/nominee owner — recurse up.
                    if not self._is_owned(owner_id):
                        # a corporate owner with no explained owners = unresolved
                        unresolved_corporate = True
                    walk(owner_id, new_chain, new_pcts, visited | {owner_id})

        walk(entity, [entity], [], {entity})
        return person_paths, unresolved_corporate

    def determine(self, entity_id: str) -> dict:
        _req = entity_id
        entity_id = _canon(entity_id)
        _amb = False
        if entity_id not in self._owners_of:
            _rid, _cands = _resolve_entity(_req, list(self._owners_of.keys()))
            if _rid:
                entity_id = _rid
            elif len(_cands) > 1:
                _amb = True
        person_paths, unresolved = self._trace_to_persons(entity_id)
        if unresolved:
            self._add_flag("unresolved_corporate_owner")
        if not person_paths:
            self._add_flag("no_natural_person_found")

        ubos = []
        for person_id, paths in person_paths.items():
            aggregate = round(sum(p["path_pct"] for p in paths), 6)
            if aggregate >= self.threshold_pct:
                ubos.append(
                    {
                        "person_id": person_id,
                        "aggregate_pct": aggregate,
                        "paths": paths,
                    }
                )
        ubos.sort(key=lambda u: -u["aggregate_pct"])

        if _amb:
            self._flag("ambiguous_entity_reference")
        review_required = bool(self.flags)
        return {
            "entity_id": entity_id,
            "ubos": ubos,
            "flags": list(self.flags),
            "threshold_pct": self.threshold_pct,
            "review_required": review_required,
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }


def determine_ubos(edges: list[dict], entity_id: str, threshold_pct: float = DEFAULT_THRESHOLD_PCT) -> dict:
    """Functional entry point matching specs/ubo_result.json."""
    parsed = [Edge.from_dict(e) for e in edges]
    graph = UBOGraph(parsed, threshold_pct=threshold_pct)
    return graph.determine(entity_id)


if __name__ == "__main__":
    # Canonical smoke: Person A 60% of HoldCo, HoldCo 40% of OpCo -> 24%.
    edges = [
        {"owner_id": "Person A", "owned_entity_id": "HoldCo", "direct_pct": 60, "owner_type": "person", "page": 1, "confidence": 0.95, "source_hash": "s1"},
        {"owner_id": "HoldCo", "owned_entity_id": "OpCo", "direct_pct": 40, "owner_type": "entity", "page": 1, "confidence": 0.95, "source_hash": "s1"},
    ]
    assert indirect_ownership([60, 40]) == 24.0, indirect_ownership([60, 40])
    out = determine_ubos(edges, "OpCo", threshold_pct=20.0)
    print(json.dumps(out, indent=2))
