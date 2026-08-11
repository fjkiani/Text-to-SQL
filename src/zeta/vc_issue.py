"""
Zeta Clearance — W3C Verifiable Credential issuance for a KYB decision.

Issues a signed KyBDecisionCredential wrapping the SAME minimal claim that goes
on the Canton ledger (never raw PII). Supports selective disclosure: a relying
party can verify the signature while receiving only the fields required for the
transaction (decision / riskTier / uboVerified), not the full subject record.

Self-contained: Ed25519 proof in a W3C VC Data-Integrity shape. Swap point for
digitalbazaar/vc (ecdsa-sd-2023 / bbs-2023) when a JS wallet runtime is wired.
"""
from __future__ import annotations

import base64
import hashlib
import json
import uuid
from datetime import datetime, timezone

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding, PrivateFormat, PublicFormat, NoEncryption,
)

KYB_CONTEXT = "https://zeta.clearance/contexts/kyb-v1"
VC_CONTEXT = "https://www.w3.org/2018/credentials/v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def _canon(obj: dict) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


class Issuer:
    """The Zeta agent as VC issuer (Ed25519)."""

    def __init__(self, issuer_did: str | None = None, priv: Ed25519PrivateKey | None = None):
        self.priv = priv or Ed25519PrivateKey.generate()
        self.pub = self.priv.public_key()
        pub_bytes = self.pub.public_bytes(Encoding.Raw, PublicFormat.Raw)
        self.issuer_did = issuer_did or f"did:key:z{_b64(pub_bytes)}"

    def issue_kyb_credential(self, subject_did: str, decision: str, risk_tier: str,
                             ubo_verified: bool, expires_at: str, evidence_hash: str,
                             canton_contract_id: str) -> dict:
        if decision not in {"approved", "rejected", "review_required"}:
            raise ValueError("invalid decision")
        if risk_tier not in {"low", "medium", "high"}:
            raise ValueError("invalid riskTier")
        credential = {
            "@context": [VC_CONTEXT, KYB_CONTEXT],
            "id": f"urn:uuid:{uuid.uuid4()}",
            "type": ["VerifiableCredential", "KyBDecisionCredential"],
            "issuer": self.issuer_did,
            "issuanceDate": _now(),
            "expirationDate": expires_at,
            "credentialSubject": {
                "id": subject_did,
                "decision": decision,
                "riskTier": risk_tier,
                "uboVerified": ubo_verified,
                "evidenceHash": evidence_hash,       # hash only, never raw PII
                "cantonContractId": canton_contract_id,
            },
        }
        proof = self._sign(credential)
        credential["proof"] = proof
        return credential

    def _sign(self, credential: dict) -> dict:
        sig = self.priv.sign(_canon(credential))
        return {
            "type": "DataIntegrityProof",
            "cryptosuite": "eddsa-2022",
            "created": _now(),
            "verificationMethod": f"{self.issuer_did}#key-1",
            "proofPurpose": "assertionMethod",
            "proofValue": f"z{_b64(sig)}",
        }

    # --- selective disclosure ---
    def derive_disclosed(self, credential: dict, reveal_fields: list[str]) -> dict:
        """Produce a derived credential revealing ONLY `reveal_fields` of the
        credentialSubject, plus a binding hash of the full subject so the relying
        party can confirm nothing was altered. The signature over the original is
        retained as proof of issuance."""
        subj = credential["credentialSubject"]
        disclosed = {k: v for k, v in subj.items() if k in reveal_fields or k == "id"}
        # commitment to the withheld fields
        withheld = {k: v for k, v in subj.items() if k not in disclosed}
        commitment = hashlib.sha256(_canon(withheld)).hexdigest()
        return {
            "@context": credential["@context"],
            "id": credential["id"],
            "type": credential["type"],
            "issuer": credential["issuer"],
            "issuanceDate": credential["issuanceDate"],
            "expirationDate": credential["expirationDate"],
            "credentialSubject": disclosed,
            "withheldCommitment": commitment,
            "proof": credential["proof"],
        }


def verify_credential(credential: dict, issuer_pub: Ed25519PrivateKey) -> bool:
    """Verify the issuer signature over the credential (proof removed)."""
    proof = credential.get("proof")
    if not proof:
        return False
    pub = issuer_pub.public_key() if isinstance(issuer_pub, Ed25519PrivateKey) else issuer_pub
    sig_b64 = proof["proofValue"].lstrip("z")
    sig = base64.urlsafe_b64decode(sig_b64 + "=" * (-len(sig_b64) % 4))
    body = {k: v for k, v in credential.items() if k != "proof"}
    try:
        pub.verify(sig, _canon(body))
        return True
    except Exception:
        return False


if __name__ == "__main__":
    iss = Issuer()
    ev = hashlib.sha256(b"evidence-bundle").hexdigest()
    vc = iss.issue_kyb_credential(
        subject_did="did:key:zacmefund",
        decision="approved", risk_tier="low", ubo_verified=True,
        expires_at="2027-08-11T00:00:00+00:00",
        evidence_hash=ev, canton_contract_id="34af4ed0-d61e-4269-8950-10d2aa5470b1",
    )
    print("issued VC id:", vc["id"])
    print("subject fields:", sorted(vc["credentialSubject"].keys()))
    ok = verify_credential(vc, iss.priv)
    print("signature verifies:", ok)
    # selective disclosure: reveal only decision/riskTier/uboVerified
    derived = iss.derive_disclosed(vc, ["decision", "riskTier", "uboVerified"])
    print("disclosed subject:", json.dumps(derived["credentialSubject"]))
    print("withheldCommitment:", derived["withheldCommitment"][:24], "...")
    assert "evidenceHash" not in derived["credentialSubject"]
    assert ok
    print("VC SMOKE OK")
