"""
Zeta Clearance — L2 Zero-Knowledge Vault.

Raw PII (passports, tax returns, cap tables) goes IN encrypted. Only a token +
evidence hash ever leaves. Downstream layers (L3 Canton attestation, L4
liquidity) see tokens and hashes — NEVER raw PII.

API-compatible with databunker's token model (store -> token, token -> record),
so a live databunker service can be swapped in behind the same interface. This
implementation is self-contained: AES-256-GCM at rest, key from env, token-
indexed on local disk. It REALLY encrypts — no plaintext "vault".
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import uuid
from datetime import datetime, timezone
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

VAULT_DIR = Path(os.environ.get("ZETA_VAULT_DIR", "/workspace/zeta/l2_vault/store"))
RECORD_TYPES = {"passport", "tax_return", "incorporation", "cap_table", "trust_deed", "id_document", "other"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_key() -> bytes:
    """32-byte AES key from env ZETA_VAULT_KEY (hex or raw). Never from git.

    For local dev only, if unset, an ephemeral key is generated and persisted to
    the vault dir with 0600 perms — clearly NOT for production.
    """
    raw = os.environ.get("ZETA_VAULT_KEY", "").strip()
    if raw:
        try:
            key = bytes.fromhex(raw)
        except ValueError:
            key = hashlib.sha256(raw.encode()).digest()
        if len(key) != 32:
            key = hashlib.sha256(key).digest()
        return key
    VAULT_DIR.mkdir(parents=True, exist_ok=True)
    kf = VAULT_DIR / ".dev_key"
    if kf.exists():
        return bytes.fromhex(kf.read_text().strip())
    key = secrets.token_bytes(32)
    kf.write_text(key.hex())
    os.chmod(kf, 0o600)
    return key


class Vault:
    """Encrypted, token-indexed PII store."""

    def __init__(self, vault_dir: str | None = None):
        self.dir = Path(vault_dir) if vault_dir else VAULT_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        self.key = _load_key()
        self.aes = AESGCM(self.key)
        self._index_path = self.dir / "index.json"
        self._index = self._read_index()

    def _read_index(self) -> dict:
        if self._index_path.exists():
            return json.loads(self._index_path.read_text())
        return {}

    def _write_index(self) -> None:
        self._index_path.write_text(json.dumps(self._index, indent=2))

    def store_pii(self, doc_bytes: bytes, record_type: str, subject_ref: str,
                  expires_at: str | None = None) -> dict:
        """Encrypt + store raw PII. Returns a VaultToken (specs/vault_token.json).

        evidence_hash is computed on the PLAINTEXT bytes BEFORE encryption so it
        can anchor the on-ledger attestation without ever revealing the doc.
        """
        if record_type not in RECORD_TYPES:
            raise ValueError(f"record_type must be one of {sorted(RECORD_TYPES)}")
        if not doc_bytes:
            raise ValueError("empty document refused")

        evidence_hash = hashlib.sha256(doc_bytes).hexdigest()
        token = str(uuid.uuid4())
        nonce = secrets.token_bytes(12)
        ciphertext = self.aes.encrypt(nonce, doc_bytes, None)

        blob_path = self.dir / f"{token}.bin"
        blob_path.write_bytes(nonce + ciphertext)

        self._index[token] = {
            "token": token,
            "record_type": record_type,
            "subject_ref": subject_ref,
            "evidence_hash": evidence_hash,
            "stored_at": _now(),
            "expires_at": expires_at,
            "blob": blob_path.name,
        }
        self._write_index()
        # Only the token + hash leave the vault boundary.
        return {
            "token": token,
            "record_type": record_type,
            "subject_ref": subject_ref,
            "evidence_hash": evidence_hash,
            "stored_at": self._index[token]["stored_at"],
            "expires_at": expires_at,
        }

    def get_evidence_hash(self, token: str) -> str:
        """Return the evidence hash WITHOUT decrypting PII."""
        rec = self._index.get(token)
        if not rec:
            raise KeyError(f"unknown token {token}")
        return rec["evidence_hash"]

    def reveal(self, token: str, authz: str) -> bytes:
        """Decrypt raw PII. Compliance-officer path ONLY — requires an explicit
        authorization token (env ZETA_VAULT_REVEAL_AUTHZ). Never called by L3/L4."""
        expected = os.environ.get("ZETA_VAULT_REVEAL_AUTHZ", "")
        if not expected or authz != expected:
            raise PermissionError("reveal denied: missing/invalid authorization")
        rec = self._index.get(token)
        if not rec:
            raise KeyError(f"unknown token {token}")
        raw = (self.dir / rec["blob"]).read_bytes()
        nonce, ciphertext = raw[:12], raw[12:]
        return self.aes.decrypt(nonce, ciphertext, None)

    def delete(self, token: str) -> bool:
        """GDPR right-to-erasure."""
        rec = self._index.pop(token, None)
        if not rec:
            return False
        blob = self.dir / rec["blob"]
        if blob.exists():
            blob.unlink()
        self._write_index()
        return True

    def list_tokens(self) -> list[str]:
        return list(self._index.keys())


if __name__ == "__main__":
    os.environ.setdefault("ZETA_VAULT_REVEAL_AUTHZ", "compliance-test-authz")
    v = Vault(vault_dir="/tmp/zeta_vault_smoke")
    doc = b"PASSPORT: John Smith, DOB 1980-01-01, No. X1234567"
    tok = v.store_pii(doc, "passport", subject_ref="subj_" + hashlib.sha256(b"john_smith").hexdigest()[:12])
    print("token:", tok["token"])
    print("evidence_hash:", tok["evidence_hash"])
    # on-disk bytes must NOT be the plaintext
    blob = (Path("/tmp/zeta_vault_smoke") / (tok["token"] + ".bin")).read_bytes()
    print("on-disk (first 32B hex):", blob[:32].hex())
    assert doc not in blob, "PLAINTEXT LEAKED TO DISK"
    # hash retrievable without decryption
    assert v.get_evidence_hash(tok["token"]) == tok["evidence_hash"]
    # reveal with authz recovers exactly
    assert v.reveal(tok["token"], authz="compliance-test-authz") == doc
    # reveal without authz denied
    try:
        v.reveal(tok["token"], authz="wrong")
        raise AssertionError("reveal should have been denied")
    except PermissionError:
        pass
    print("VAULT SMOKE OK: encrypted at rest, token-only downstream, authz-gated reveal")
