"""
Zeta Clearance engine API — the Python compute plane for institutional KYB.

Mounted on the live Tessera FastAPI service so the TypeScript backend
(artifacts/api-server) can call it over HTTP. This replaces the planned Modal
service: same contract, but hosted on infrastructure that is already running,
which removes the Modal spend-limit dependency entirely.

Layer coverage:
  L1  /zeta/ingest          parse a document into provenance-tagged chunks
  L1  /zeta/extract_edges   LLM proposes ownership edges (page + confidence)
  L1  /zeta/interrogate     agentic missing-document loop (pause/resume)
  L2  /zeta/ubo             DETERMINISTIC multi-hop UBO computation
  L2  /zeta/vault/store     encrypt PII, return token only
  L3  /zeta/attest          Canton attestation + W3C VC issuance
  L3  /zeta/verify          relying-party verification (no PII crosses)
  L4  /zeta/relay           Canton -> EVM oracle push + allowlist
  L4  /zeta/revoke          revoke attestation, propagate to chain
      /zeta/health

Auth: Authorization: Bearer $ZETA_ENGINE_TOKEN (enforced when the env var is set).

Design rule enforced here: the LLM only ever PROPOSES edges. Ownership
percentages are computed by ubo_graph.py. No raw PII is returned by any endpoint
that a relying party can reach.
"""
import base64
import os
import tempfile

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from src.zeta.canon import canonical_id
from src.zeta.interrogator import InterrogatorState, run_until_pause
from src.zeta.ubo_graph import determine_ubos

router = APIRouter(prefix="/zeta", tags=["zeta"])

ENGINE_TOKEN = os.environ.get("ZETA_ENGINE_TOKEN", "")

# Text formats we parse natively so the 512MB web dyno never loads `unstructured`.
LIGHT_EXT = {".txt", ".csv", ".tsv", ".md", ".json"}


def _auth(authorization: str | None) -> None:
    if ENGINE_TOKEN and authorization != f"Bearer {ENGINE_TOKEN}":
        raise HTTPException(status_code=401, detail="invalid engine token")


# ----------------------------- models -----------------------------
class IngestReq(BaseModel):
    filename: str = "doc.txt"
    doc_b64: str


class EdgesReq(BaseModel):
    chunks: list[dict]


class UboReq(BaseModel):
    entity_id: str
    edges: list[dict]
    threshold_pct: float = 25.0
    # Conflicting direct percentages found during extraction (same owner/owned
    # pair reported with different values on different pages). Passed through
    # from /extract_edges so the determination carries the contradiction
    # instead of quietly resolving it.
    conflicts: list[dict] = []


class InterrogateReq(BaseModel):
    entity_id: str
    edges: list[dict] = []
    have_docs: list[str] = []


class VaultReq(BaseModel):
    subject_id: str
    record_type: str = "identity_document"
    doc_b64: str


class AttestReq(BaseModel):
    legal_entity_name: str
    decision: str
    risk_tier: str
    ubo_verified: bool
    evidence_hash: str
    subject: str = "applicant"
    relying_parties: list[str] = []
    valid_days: int = 365


class VerifyReq(BaseModel):
    contract_id: str
    relying_party: str


class RelayReq(BaseModel):
    contract_id: str
    relying_party: str


class RevokeReq(BaseModel):
    """
    Revocation is an ISSUER action on the Canton ledger, not a relying-party
    action -- `CantonLedger.revoke` rejects anyone who is not `contract.issuer`.
    The old model reused VerifyReq and passed `relying_party` straight into
    `by_party`, so every call from an actual relying party raised an uncaught
    PermissionError and returned HTTP 500. `relying_party` is kept as a
    deprecated alias so existing callers do not break.
    """
    contract_id: str
    by_party: str | None = None
    relying_party: str | None = None

    def party(self) -> str:
        # /attest hardcodes issuer="zeta_issuer", so that is the only party
        # that can ever revoke what this service issued.
        return self.by_party or self.relying_party or "zeta_issuer"


class ClearedReq(BaseModel):
    entity_key: str


# ----------------------------- L1 -----------------------------
def _light_chunks(name: str, data: bytes) -> list[dict]:
    text = data.decode("utf-8", errors="replace")
    blocks, buf = [], []
    for line in text.splitlines():
        buf.append(line)
        if len(buf) >= 40:
            blocks.append("\n".join(buf))
            buf = []
    if buf:
        blocks.append("\n".join(buf))
    return [{"chunk_id": f"{name}:{i}", "page": i + 1, "text": b}
            for i, b in enumerate(blocks) if b.strip()]


@router.post("/ingest")
def zeta_ingest(req: IngestReq, authorization: str | None = Header(None)):
    _auth(authorization)
    data = base64.b64decode(req.doc_b64)
    ext = os.path.splitext(req.filename)[1].lower()
    if ext in LIGHT_EXT or not ext:
        chunks = _light_chunks(req.filename, data)
        mode = "native"
    else:
        # Heavy formats (pdf/docx) use Tessera ingest; imported lazily so the
        # web dyno only pays the memory cost when a binary doc actually arrives.
        from src.ingest.service import ingest_file
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tf:
            tf.write(data)
            tmp = tf.name
        try:
            out = ingest_file(tmp, tenant="zeta")
            chunks = out["chunks"]
            mode = "unstructured"
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    return {"filename": req.filename, "chunk_count": len(chunks),
            "chunks": chunks, "parse_mode": mode}


@router.post("/extract_edges")
def zeta_extract_edges(req: EdgesReq, authorization: str | None = Header(None)):
    _auth(authorization)
    from src.zeta.extractor import extract_edges
    try:
        return extract_edges(req.chunks)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"extraction failed: {e}")


@router.post("/interrogate")
def zeta_interrogate(req: InterrogateReq, authorization: str | None = Header(None)):
    _auth(authorization)
    ubo = determine_ubos(req.edges, req.entity_id) if req.edges else None
    st = run_until_pause(
        InterrogatorState(entity_id=canonical_id(req.entity_id), have_docs=list(req.have_docs)), ubo
    )
    if st.status == "await_doc" and st.pending:
        return {"action": "request_doc", "missing": st.pending,
                "message": st.pending.get("message", ""), "history": st.history}
    return {"action": "satisfied", "message": "All required documents present.",
            "history": st.history}


# ----------------------------- L2 -----------------------------
@router.post("/ubo")
def zeta_ubo(req: UboReq, authorization: str | None = Header(None)):
    _auth(authorization)
    out = determine_ubos(req.edges, req.entity_id, threshold_pct=req.threshold_pct)
    if req.conflicts:
        # The source documents contradict each other about a direct holding.
        # The graph computed a number from ONE of those readings; a human has to
        # decide which register governs before this can be attested. Never let a
        # contradicted cap table auto-clear.
        out.setdefault("flags", [])
        if "conflicting_ownership_records" not in out["flags"]:
            out["flags"].append("conflicting_ownership_records")
        out["review_required"] = True
        out["conflicts"] = req.conflicts
    return out


@router.post("/vault/store")
def zeta_vault_store(req: VaultReq, authorization: str | None = Header(None)):
    _auth(authorization)
    from src.zeta.vault import Vault
    rec = Vault().store_pii(
        base64.b64decode(req.doc_b64), req.record_type, canonical_id(req.subject_id)
    )
    # Only the token and evidence hash ever cross the vault boundary.
    return {"token": rec["token"], "evidence_hash": rec["evidence_hash"]}


# ----------------------------- L3 -----------------------------
_LEDGER = None
_ISSUER = None


def _ledger():
    global _LEDGER
    if _LEDGER is None:
        from src.zeta.ledger_adapter import CantonLedger
        _LEDGER = CantonLedger()
    return _LEDGER


def _issuer():
    global _ISSUER
    if _ISSUER is None:
        from src.zeta.vc_issue import Issuer
        _ISSUER = Issuer()
    return _ISSUER


@router.post("/attest")
def zeta_attest(req: AttestReq, authorization: str | None = Header(None)):
    _auth(authorization)
    import hashlib
    from datetime import datetime, timedelta, timezone
    entity_hash = hashlib.sha256(canonical_id(req.legal_entity_name).encode()).hexdigest()
    expires = (datetime.now(timezone.utc) + timedelta(days=req.valid_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {
        "legalEntityHash": entity_hash,
        "decision": req.decision,
        "riskTier": req.risk_tier,
        "uboVerified": req.ubo_verified,
        "expiresAt": expires,
        "evidenceHash": req.evidence_hash,
    }
    try:
        contract = _ledger().create_attestation(
            issuer="zeta_issuer", subject=req.subject,
            observers=req.relying_parties, payload=payload,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    cid = contract["contractId"]
    vc = _issuer().issue_kyb_credential(
        subject_did=f"did:zeta:{req.subject}", decision=req.decision,
        risk_tier=req.risk_tier, ubo_verified=req.ubo_verified,
        expires_at=expires, evidence_hash=req.evidence_hash,
        canton_contract_id=cid,
    )
    return {"contract_id": cid, "payload": payload, "credential": vc}


@router.post("/verify")
def zeta_verify(req: VerifyReq, authorization: str | None = Header(None)):
    _auth(authorization)
    try:
        return {"verified": True, "claim": _ledger().verify(req.contract_id, req.relying_party)}
    except Exception as e:
        return {"verified": False, "reason": str(e)}


def _entity_key(contract_id: str) -> str | None:
    """bytes32 oracle key for a contract, or None if the contract is unreadable."""
    from src.zeta.relayer import _to_bytes32
    try:
        return _to_bytes32(_ledger().get(contract_id)["payload"]["legalEntityHash"])
    except Exception:
        return None


def _tear_down_clearance(contract_id: str) -> dict:
    """
    Drop the on-chain clearance bit for a contract's entity.

    This is the half of revocation that was missing. `/revoke` used to touch
    only the Canton ledger, so `verify` and `relay` correctly started refusing
    while the EVM oracle entry posted by an earlier `/relay` stayed
    `revoked=False` -- `isCleared` remained true and a PermissionedPool gating
    on that bit would still have accepted a deposit from a revoked entity.
    Confirmed live before this fix: entity_key 0xdfb35610... stayed cleared
    after the ledger reported "attestation revoked".
    """
    key = _entity_key(contract_id)
    if not key:
        return {"entity_key": None, "oracle_revoked": False, "is_cleared": False}
    orc = _oracle()
    if key in orc.attestations:
        orc.revoke(key, by=orc.relayer)
        return {"entity_key": key, "oracle_revoked": True,
                "is_cleared": orc.is_cleared(key)}
    # Never relayed, so there is nothing on chain to tear down.
    return {"entity_key": key, "oracle_revoked": False, "is_cleared": False}


@router.post("/revoke")
def zeta_revoke(req: RevokeReq, authorization: str | None = Header(None)):
    _auth(authorization)
    # Read the entity key BEFORE revoking: it is derived from the contract
    # payload, which must still be readable.
    try:
        _ledger().revoke(req.contract_id, by_party=req.party())
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown contract {req.contract_id}")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    return {"revoked": True, "contract_id": req.contract_id,
            **_tear_down_clearance(req.contract_id)}


@router.post("/cleared")
def zeta_cleared(req: ClearedReq, authorization: str | None = Header(None)):
    """
    Read the oracle clearance bit for an entity key. This is the only thing a
    permissioned pool needs at deposit time, and it was previously readable
    only as a side-effect of POST /relay -- which refuses once the attestation
    is revoked, so there was no way to observe the stale-clearance bug from
    outside the process.
    """
    _auth(authorization)
    orc = _oracle()
    a = orc.attestations.get(req.entity_key)
    return {"entity_key": req.entity_key, "known": a is not None,
            "is_cleared": orc.is_cleared(req.entity_key),
            "revoked": bool(a and a["revoked"])}


# ----------------------------- L4 -----------------------------
_ORACLE = None


def _oracle():
    global _ORACLE
    if _ORACLE is None:
        from src.zeta.relayer import EVMOracle
        _ORACLE = EVMOracle()
    return _ORACLE


@router.post("/relay")
def zeta_relay(req: RelayReq, authorization: str | None = Header(None)):
    _auth(authorization)
    from src.zeta.relayer import relay
    try:
        key = relay(_ledger(), _oracle(), req.contract_id, req.relying_party)
        return {"relayed": True, "entity_key": key,
                "is_cleared": _oracle().is_cleared(key)}
    except Exception as e:
        # A relay attempt that the ledger refuses because the attestation is
        # revoked or expired is also the last chance to notice that a stale
        # clearance bit is still standing on chain. Tear it down rather than
        # only returning 400.
        detail = str(e)
        torn = {}
        if "revoked" in detail or "expired" in detail:
            torn = _tear_down_clearance(req.contract_id)
        raise HTTPException(status_code=400, detail=detail, headers=(
            {"X-Zeta-Clearance-Torn-Down": "1"} if torn.get("oracle_revoked") else None))


@router.get("/health")
def zeta_health():
    """
    Liveness + which build and which LLM tiers are actually armed.

    `commit` is here because a Render deploy triggered by an env-var change
    builds the tree as of that moment, so pushing code and setting env vars in
    quick succession can put a deploy labelled with the new SHA in front of the
    old tree. Without a build marker served by the process itself, the only way
    to tell was to probe for a behaviour change and guess.
    """
    import os as _o
    from src.llm.openrouter import _keys as _or_keys, _models as _or_models

    try:
        from src.llm.gemini import _keys as _g_keys, _models as _g_models
        gem = {"keys": len(_g_keys()), "models": len(_g_models())}
    except Exception:
        gem = {"keys": 0, "models": 0}

    tiers = ([f"gemini({gem['keys']}k/{gem['models']}m)"] if gem["keys"] else []) + [
        f"openrouter({len(_or_keys())}k/{len(_or_models())}m)"
    ]
    return {
        "status": "ok",
        "service": "zeta-clearance-engine",
        "commit": (_o.environ.get("RENDER_GIT_COMMIT") or "unknown")[:7],
        "llm_tiers": tiers,
        "llm_keys": len(_or_keys()),
        "llm_models": len(_or_models()),
        "features": ["batched_extraction", "conflict_detection", "deterministic_merge"],
        "layers": ["L1-ingest", "L1-interrogate", "L2-ubo", "L2-vault",
                   "L3-attest", "L3-verify", "L4-relay"],
    }
