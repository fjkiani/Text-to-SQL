"""
Zeta Clearance — L4 Canton->EVM oracle relayer.

Reads a Canton KyBAttestation (via the ledger adapter — the swap point for a
live Canton participant node) and relays the minimal claim to the EVM
AttestationOracle. Then a permissioned pool (Aave-Arc style) gates deposits on
the oracle's clearance bit — instant verification, zero raw PII.

The EVM side is modelled by an in-process `EVMOracle` that reproduces
AttestationOracle.sol semantics exactly (same isCleared logic), so the full
flow is testable today; `relayer.py` is the single place that would hold a
web3.py signer when a live EVM RPC is provided.
"""
from __future__ import annotations

import hashlib
import sys
import time
from dataclasses import dataclass, field

sys.path.insert(0, "/workspace/zeta/l3_ledger")
from ledger_adapter import CantonLedger  # noqa: E402

DECISION_MAP = {"rejected": 0, "review_required": 1, "approved": 2}
RISK_MAP = {"low": 0, "medium": 1, "high": 2}


def _to_bytes32(hex_or_text: str) -> str:
    s = hex_or_text.strip()
    if len(s) == 64 and all(c in "0123456789abcdefABCDEF" for c in s):
        return "0x" + s.lower()
    return "0x" + hashlib.sha256(s.encode()).hexdigest()


def _to_unix(iso_ts: str) -> int:
    from datetime import datetime
    return int(datetime.fromisoformat(iso_ts.replace("Z", "+00:00")).timestamp())


@dataclass
class EVMOracle:
    """In-process mirror of AttestationOracle.sol (same isCleared semantics)."""
    relayer: str = "zeta_relayer"
    attestations: dict = field(default_factory=dict)

    def post_attestation(self, legal_entity_hash, decision, risk_tier, ubo_verified, expires_at, evidence_hash, by):
        if by != self.relayer:
            raise PermissionError("not relayer")
        self.attestations[legal_entity_hash] = {
            "legalEntityHash": legal_entity_hash, "decision": decision, "riskTier": risk_tier,
            "uboVerified": ubo_verified, "expiresAt": expires_at, "evidenceHash": evidence_hash,
            "revoked": False, "updatedAt": int(time.time()),
        }

    def revoke(self, legal_entity_hash, by):
        if by != self.relayer:
            raise PermissionError("not relayer")
        self.attestations[legal_entity_hash]["revoked"] = True

    def is_cleared(self, legal_entity_hash) -> bool:
        a = self.attestations.get(legal_entity_hash)
        if not a:
            return False
        return (not a["revoked"]) and a["decision"] == 2 and a["uboVerified"] and a["expiresAt"] > int(time.time())


@dataclass
class PermissionedPool:
    """Mirror of PermissionedPool.sol — gates deposits on oracle.isCleared."""
    oracle: EVMOracle
    deposits: dict = field(default_factory=dict)

    def deposit(self, legal_entity_hash, amount) -> int:
        if not self.oracle.is_cleared(legal_entity_hash):
            raise PermissionError("KYB not cleared")
        self.deposits[legal_entity_hash] = self.deposits.get(legal_entity_hash, 0) + amount
        return self.deposits[legal_entity_hash]


def relay(canton: CantonLedger, oracle: EVMOracle, contract_id: str, relying_party: str) -> str:
    """Read the Canton attestation and post it to the EVM oracle. Returns the
    entity hash key used on-chain."""
    claim = canton.verify(contract_id, relying_party)  # raises if no visibility / revoked / expired
    leh = _to_bytes32(claim["legalEntityHash"])
    oracle.post_attestation(
        legal_entity_hash=leh,
        decision=DECISION_MAP[claim["decision"]],
        risk_tier=RISK_MAP[claim["riskTier"]],
        ubo_verified=claim["uboVerified"],
        expires_at=_to_unix(claim["expiresAt"]),
        evidence_hash=_to_bytes32(claim["evidenceHash"]),
        by=oracle.relayer,
    )
    return leh


if __name__ == "__main__":
    # Full L3 -> L4 flow: Canton attestation -> relay -> EVM oracle -> pool deposit.
    canton = CantonLedger()
    payload = {
        "legalEntityHash": hashlib.sha256(b"acme_cayman_ltd:KY").hexdigest(),
        "decision": "approved", "riskTier": "low", "uboVerified": True,
        "expiresAt": "2027-08-11T00:00:00+00:00",
        "evidenceHash": hashlib.sha256(b"evidence-bundle").hexdigest(),
    }
    att = canton.create_attestation("zeta_agent", "acme_fund", ["aave_arc_pool"], payload)

    oracle = EVMOracle(relayer="zeta_relayer")
    pool = PermissionedPool(oracle)

    leh = relay(canton, oracle, att["contractId"], "aave_arc_pool")
    print("relayed to EVM, entity key:", leh[:18], "...")
    print("oracle.isCleared:", oracle.is_cleared(leh))

    # The $50M trade — gated on the clearance bit, no raw PII touched.
    bal = pool.deposit(leh, 50_000_000)
    print("deposit accepted, pool balance:", bal)

    # A non-cleared entity is rejected.
    stranger = _to_bytes32("unknown_entity")
    try:
        pool.deposit(stranger, 1_000_000)
        print("ERROR: stranger should be rejected")
    except PermissionError as e:
        print("non-cleared entity rejected:", e)

    # Revocation on Canton propagates: relay a revoke -> pool rejects.
    canton.revoke(att["contractId"], "zeta_agent")
    oracle.revoke(leh, by="zeta_relayer")
    print("after revoke, isCleared:", oracle.is_cleared(leh))
    try:
        pool.deposit(leh, 1_000)
        print("ERROR: revoked entity should be rejected")
    except PermissionError as e:
        print("revoked entity rejected:", e)
    print("L4 RELAYER SMOKE OK")
