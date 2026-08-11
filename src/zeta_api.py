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
    return determine_ubos(req.edges, req.entity_id, threshold_pct=req.threshold_pct)


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


@router.post("/revoke")
def zeta_revoke(req: VerifyReq, authorization: str | None = Header(None)):
    _auth(authorization)
    _ledger().revoke(req.contract_id, by_party=req.relying_party)
    return {"revoked": True, "contract_id": req.contract_id}


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
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/health")
def zeta_health():
    from src.llm.openrouter import _keys, _models
    return {"status": "ok", "service": "zeta-clearance-engine",
            "llm_keys": len(_keys()), "llm_models": len(_models()),
            "layers": ["L1-ingest", "L1-interrogate", "L2-ubo", "L2-vault",
                       "L3-attest", "L3-verify", "L4-relay"]}
