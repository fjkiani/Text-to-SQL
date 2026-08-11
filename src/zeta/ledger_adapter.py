"""
Zeta Clearance — L3 ledger adapter.

Mirrors daml/KyBAttestation.daml semantics 1:1 so L4 and the front-end can
integrate against a real interface today. This is the SWAP POINT for a live
Canton participant node (the .daml is the source of truth; this adapter
reproduces its party-visibility + choice semantics in-process).

PAYLOAD BOUNDARY (enforced): only the 6 fields of specs/attestation.json may be
stored. Any attempt to write a field outside that set raises — raw PII can never
reach the ledger through this path.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone

ALLOWED_FIELDS = {"legalEntityHash", "decision", "riskTier", "uboVerified", "expiresAt", "evidenceHash"}
DECISIONS = {"approved", "rejected", "review_required"}
RISK_TIERS = {"low", "medium", "high"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_payload(payload: dict) -> None:
    extra = set(payload) - ALLOWED_FIELDS
    if extra:
        raise ValueError(f"PII BOUNDARY VIOLATION: fields not allowed on ledger: {sorted(extra)}")
    missing = ALLOWED_FIELDS - set(payload)
    if missing:
        raise ValueError(f"missing required attestation fields: {sorted(missing)}")
    if payload["decision"] not in DECISIONS:
        raise ValueError(f"decision must be one of {sorted(DECISIONS)}")
    if payload["riskTier"] not in RISK_TIERS:
        raise ValueError(f"riskTier must be one of {sorted(RISK_TIERS)}")
    if not isinstance(payload["uboVerified"], bool):
        raise ValueError("uboVerified must be bool")


class CantonLedger:
    """In-process Canton-attestation semantics. Party-visibility enforced."""

    def __init__(self):
        self._contracts: dict[str, dict] = {}

    def create_attestation(self, issuer: str, subject: str, observers: list[str],
                           payload: dict) -> dict:
        _validate_payload(payload)
        cid = str(uuid.uuid4())
        contract = {
            "contractId": cid,
            "template": "KyBAttestation",
            "issuer": issuer,
            "subject": subject,
            "observers": list(observers),
            "payload": dict(payload),
            "revoked": False,
            "createdAt": _now(),
        }
        self._contracts[cid] = contract
        # Return only what a stakeholder may see.
        return {"contractId": cid, "payload": dict(payload), "createdAt": contract["createdAt"]}

    def _visible_to(self, contract: dict, party: str) -> bool:
        return party in ([contract["issuer"], contract["subject"]] + contract["observers"])

    def verify(self, contract_id: str, party: str) -> dict:
        c = self._contracts.get(contract_id)
        if not c:
            raise KeyError(f"unknown contract {contract_id}")
        if not self._visible_to(c, party):
            raise PermissionError(f"party {party} has no visibility on {contract_id}")
        if c["revoked"]:
            raise ValueError("attestation revoked")
        if c["payload"]["expiresAt"] <= _now():
            raise ValueError("attestation expired")
        return dict(c["payload"])

    def grant_observer(self, contract_id: str, by_party: str, new_observer: str) -> None:
        c = self._contracts.get(contract_id)
        if not c:
            raise KeyError(f"unknown contract {contract_id}")
        if by_party != c["subject"]:
            raise PermissionError("only the subject may grant observers")
        if new_observer not in c["observers"]:
            c["observers"].append(new_observer)

    def revoke(self, contract_id: str, by_party: str) -> None:
        c = self._contracts.get(contract_id)
        if not c:
            raise KeyError(f"unknown contract {contract_id}")
        if by_party != c["issuer"]:
            raise PermissionError("only the issuer may revoke")
        c["revoked"] = True

    def get(self, contract_id: str) -> dict:
        c = self._contracts.get(contract_id)
        if not c:
            raise KeyError(f"unknown contract {contract_id}")
        return dict(c)


if __name__ == "__main__":
    ledger = CantonLedger()
    payload = {
        "legalEntityHash": hashlib.sha256(b"acme_cayman_ltd:KY").hexdigest(),
        "decision": "approved",
        "riskTier": "low",
        "uboVerified": True,
        "expiresAt": "2027-08-11T00:00:00+00:00",
        "evidenceHash": hashlib.sha256(b"evidence-bundle").hexdigest(),
    }
    att = ledger.create_attestation("zeta_agent", "acme_fund", ["aave_arc_pool"], payload)
    print("created contract:", att["contractId"])
    print("payload fields:", sorted(att["payload"].keys()))
    # relying party verifies
    v = ledger.verify(att["contractId"], "aave_arc_pool")
    print("verified by relying party:", v["decision"], v["riskTier"], "uboVerified=", v["uboVerified"])
    # a stranger has no visibility
    try:
        ledger.verify(att["contractId"], "random_stranger")
        print("ERROR: stranger should not see")
    except PermissionError as e:
        print("stranger denied:", e)
    # PII boundary enforced
    try:
        ledger.create_attestation("z", "s", [], {**payload, "passportNumber": "X123"})
        print("ERROR: PII should have been rejected")
    except ValueError as e:
        print("PII rejected:", e)
    # revoke
    ledger.revoke(att["contractId"], "zeta_agent")
    try:
        ledger.verify(att["contractId"], "aave_arc_pool")
    except ValueError as e:
        print("post-revoke verify fails:", e)
    print("LEDGER ADAPTER SMOKE OK")
