"""
Zeta Clearance — L1 Agentic Interrogator.

The synchronous missing-doc loop: given the current ownership edges and the set
of documents the applicant has provided, decide the NEXT missing document and
surface an applicant-facing request — then PAUSE until the applicant responds,
and RESUME exactly where it left off.

LangGraph's interrupt()/Command(resume=...) is the ideal primitive. When
langgraph is not installed (this env), we implement the SAME pause/resume
semantics with an explicit, serializable state machine so the behaviour is real
and testable — not a fake "agent" returning a static string.

State machine:
    ASSESS -> (gap found) -> AWAIT_DOC --(applicant uploads)--> ASSESS -> ... -> SATISFIED
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

# Documents we require for each UBO (natural person >= threshold) and for the
# entity itself.
REQUIRED_ENTITY_DOCS = {"incorporation", "cap_table"}
REQUIRED_UBO_DOCS = {"id_document"}  # passport / government ID per UBO


@dataclass
class InterrogatorState:
    entity_id: str
    have_docs: list[str] = field(default_factory=list)          # doc record_types present
    ubo_ids: list[str] = field(default_factory=list)            # known UBO person_ids
    ubo_docs: dict[str, list[str]] = field(default_factory=dict)  # person_id -> doc types present
    status: str = "assess"                                       # assess | await_doc | satisfied
    pending: dict | None = None                                  # the outstanding request
    history: list[dict] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def _pretty(person_id: str) -> str:
    return person_id.replace("_", " ").title()


def assess(state: InterrogatorState, ubo_result: dict | None = None) -> InterrogatorState:
    """One interrogation step. Mutates+returns state. If a gap is found, sets
    status='await_doc' and pending=request payload (the 'interrupt')."""
    # Refresh known UBOs from the deterministic engine result if provided.
    if ubo_result is not None:
        state.ubo_ids = [u["person_id"] for u in ubo_result.get("ubos", [])]

    # 1. Entity-level docs.
    for req in sorted(REQUIRED_ENTITY_DOCS):
        if req not in state.have_docs:
            state.status = "await_doc"
            state.pending = {
                "missing": req,
                "entity": state.entity_id,
                "reason": f"required corporate document '{req}' not yet provided",
                "message": f"To verify {state.entity_id.replace('_',' ').title()}, please upload its {req.replace('_',' ')}.",
            }
            state.history.append({"action": "request_doc", **state.pending})
            return state

    # 2. Per-UBO ID docs.
    for pid in state.ubo_ids:
        have = set(state.ubo_docs.get(pid, []))
        for req in sorted(REQUIRED_UBO_DOCS - have):
            state.status = "await_doc"
            state.pending = {
                "missing": req,
                "entity": pid,
                "reason": f"UBO >= threshold with no {req}",
                "message": f"I see {_pretty(pid)} is a beneficial owner (>= threshold). Please upload their {req.replace('_',' ')} (e.g. passport).",
            }
            state.history.append({"action": "request_doc", **state.pending})
            return state

    state.status = "satisfied"
    state.pending = None
    state.history.append({"action": "satisfied", "entity": state.entity_id})
    return state


def resume(state: InterrogatorState, uploaded: dict) -> InterrogatorState:
    """The 'Command(resume=...)': applicant uploaded a doc. Record it and clear
    the pending request, then re-assess."""
    if state.status != "await_doc" or not state.pending:
        raise ValueError("resume called but no document is outstanding")
    record_type = uploaded.get("record_type")
    subject = uploaded.get("subject_ref")  # tokenized subject ref, or entity_id
    if not record_type:
        raise ValueError("uploaded doc must include record_type")
    # entity-level vs ubo-level
    if state.pending["entity"] == state.entity_id:
        if record_type not in state.have_docs:
            state.have_docs.append(record_type)
    else:
        state.ubo_docs.setdefault(state.pending["entity"], [])
        if record_type not in state.ubo_docs[state.pending["entity"]]:
            state.ubo_docs[state.pending["entity"]].append(record_type)
    state.history.append({"action": "resume", "received": record_type, "for": state.pending["entity"]})
    state.status = "assess"
    state.pending = None
    return state


def run_until_pause(state: InterrogatorState, ubo_result: dict | None = None) -> InterrogatorState:
    """Drive assessment until we either pause for a doc or are satisfied."""
    return assess(state, ubo_result)


if __name__ == "__main__":
    # Real pause/resume walkthrough (no LLM needed — deterministic gap logic).
    st = InterrogatorState(entity_id="acme_cayman_ltd")
    ubo = {"ubos": [{"person_id": "john_smith", "aggregate_pct": 30.0, "paths": []}], "flags": [], "review_required": False}

    st = run_until_pause(st, ubo)
    assert st.status == "await_doc" and st.pending["missing"] == "cap_table"
    print("PAUSE 1:", st.pending["message"])

    st = resume(st, {"record_type": "cap_table"})
    st = run_until_pause(st, ubo)
    assert st.status == "await_doc" and st.pending["missing"] == "incorporation"
    print("PAUSE 2:", st.pending["message"])

    st = resume(st, {"record_type": "incorporation"})
    st = run_until_pause(st, ubo)
    assert st.status == "await_doc" and st.pending["missing"] == "id_document"
    print("PAUSE 3:", st.pending["message"])

    st = resume(st, {"record_type": "id_document", "subject_ref": "john_smith"})
    st = run_until_pause(st, ubo)
    assert st.status == "satisfied", st.status
    print("SATISFIED. history steps:", len(st.history))
    print("INTERROGATOR SMOKE OK")
