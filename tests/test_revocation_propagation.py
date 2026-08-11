"""
Zeta L3->L4 revocation propagation regression test.

Defect this pins down (found live on build c6caf0a, production):

  POST /zeta/revoke touched only the Canton ledger. /verify and /relay then
  correctly refused, but the EVM oracle entry written by the earlier /relay
  kept revoked=False, so AttestationOracle.isCleared stayed TRUE. A
  PermissionedPool gating deposits on that bit would still have accepted money
  from an entity whose KYB attestation had been revoked -- the exact failure
  the L4 layer exists to prevent.

  Two smaller defects in the same handler:
    * revoke() passed `relying_party` into `by_party`, but the ledger only lets
      the ISSUER revoke -> uncaught PermissionError -> HTTP 500 for every
      relying-party caller.
    * the clearance bit was readable only as a side-effect of POST /relay,
      which refuses once revoked -- so the stale bit was unobservable from
      outside the process.

Run: python3 test_revocation_propagation.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, "/workspace/fireworks_sql_git")
os.environ.pop("ZETA_ENGINE_TOKEN", None)  # disable bearer gate for in-process calls

from fastapi.testclient import TestClient  # noqa: E402
from fastapi import FastAPI  # noqa: E402

from src.zeta_api import router  # noqa: E402
from src.zeta.relayer import PermissionedPool  # noqa: E402
import src.zeta_api as za  # noqa: E402

app = FastAPI()
app.include_router(router)
C = TestClient(app)

EVID = "a" * 64
FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def attest(name="Acme OpCo Ltd", parties=("aave_arc",)):
    r = C.post("/zeta/attest", json={
        "legal_entity_name": name, "decision": "approved", "risk_tier": "medium",
        "ubo_verified": True, "evidence_hash": EVID, "relying_parties": list(parties),
    })
    assert r.status_code == 200, r.text
    return r.json()


print("\n1. attest -> relay -> oracle clearance is up")
a = attest()
cid = a["contract_id"]
check("attest returns 'payload', not 'ledger_payload'",
      "payload" in a and "ledger_payload" not in a, str(sorted(a)))
check("ledger payload is exactly the 6 non-PII fields",
      sorted(a["payload"]) == ["decision", "evidenceHash", "expiresAt",
                               "legalEntityHash", "riskTier", "uboVerified"])
check("raw legal name never reaches the ledger", "Acme" not in str(a["payload"]))

rl = C.post("/zeta/relay", json={"contract_id": cid, "relying_party": "aave_arc"}).json()
key = rl["entity_key"]
check("relay clears the entity on chain", rl["is_cleared"] is True)

pool = PermissionedPool(za._oracle())
check("permissioned pool accepts a cleared deposit", pool.deposit(key, 50_000_000) == 50_000_000)

print("\n2. /cleared exposes the bit a pool actually gates on")
cl = C.post("/zeta/cleared", json={"entity_key": key}).json()
check("/cleared reports known + cleared", cl["known"] and cl["is_cleared"] and not cl["revoked"])
unk = C.post("/zeta/cleared", json={"entity_key": "0x" + "f" * 64}).json()
check("/cleared on an unknown key is not cleared", unk["known"] is False and unk["is_cleared"] is False)

print("\n3. revoke by a relying party is 403, not 500")
r = C.post("/zeta/revoke", json={"contract_id": cid, "by_party": "aave_arc"})
check("relying-party revoke -> 403", r.status_code == 403, f"got {r.status_code}")
check("403 body explains the ledger rule", "issuer" in r.text)
r404 = C.post("/zeta/revoke", json={"contract_id": "does-not-exist"})
check("unknown contract -> 404", r404.status_code == 404, f"got {r404.status_code}")
check("ledger still live after refused revoke",
      C.post("/zeta/verify", json={"contract_id": cid, "relying_party": "aave_arc"}).json()["verified"] is True)

print("\n4. THE BUG: issuer revoke must tear down the on-chain clearance bit")
rv = C.post("/zeta/revoke", json={"contract_id": cid, "by_party": "zeta_issuer"}).json()
check("revoke succeeds", rv["revoked"] is True)
check("revoke reports oracle propagation", rv["oracle_revoked"] is True, str(rv))
check("revoke reports is_cleared=False", rv["is_cleared"] is False)
check("oracle bit is actually down (read back via /cleared)",
      C.post("/zeta/cleared", json={"entity_key": key}).json()["is_cleared"] is False)
check("ledger verify is down", C.post("/zeta/verify",
      json={"contract_id": cid, "relying_party": "aave_arc"}).json()["verified"] is False)

deposited = True
try:
    pool.deposit(key, 1_000)
except PermissionError:
    deposited = False
check("POOL REJECTS the revoked entity's deposit", deposited is False)

print("\n5. a re-relay of an already-revoked attestation also tears the bit down")
b = attest(name="Beta Holdings Ltd")
kb = C.post("/zeta/relay", json={"contract_id": b["contract_id"], "relying_party": "aave_arc"}).json()["entity_key"]
check("beta cleared", C.post("/zeta/cleared", json={"entity_key": kb}).json()["is_cleared"] is True)
# Revoke straight on the ledger, simulating an out-of-band / older-build revoke
# that never propagated.
za._ledger().revoke(b["contract_id"], by_party="zeta_issuer")
check("stale bit is still up before re-relay (this is the hazard)",
      C.post("/zeta/cleared", json={"entity_key": kb}).json()["is_cleared"] is True)
rr = C.post("/zeta/relay", json={"contract_id": b["contract_id"], "relying_party": "aave_arc"})
check("re-relay of a revoked attestation -> 400", rr.status_code == 400, f"got {rr.status_code}")
check("re-relay tore the stale bit down",
      C.post("/zeta/cleared", json={"entity_key": kb}).json()["is_cleared"] is False)

print("\n6. revocation is scoped to one entity")
c = attest(name="Gamma Trading Ltd")
kc = C.post("/zeta/relay", json={"contract_id": c["contract_id"], "relying_party": "aave_arc"}).json()["entity_key"]
C.post("/zeta/revoke", json={"contract_id": cid, "by_party": "zeta_issuer"})
check("an unrelated entity stays cleared",
      C.post("/zeta/cleared", json={"entity_key": kc}).json()["is_cleared"] is True)
check("distinct entities get distinct oracle keys", len({key, kb, kc}) == 3)

print("\n7. double revoke is idempotent, not a crash")
d = C.post("/zeta/revoke", json={"contract_id": cid, "by_party": "zeta_issuer"})
check("second revoke still 200", d.status_code == 200, f"got {d.status_code}")
check("still not cleared", d.json()["is_cleared"] is False)

print("\n8. legacy callers using relying_party='zeta_issuer' keep working")
e = attest(name="Delta Capital Ltd")
ke = C.post("/zeta/relay", json={"contract_id": e["contract_id"], "relying_party": "aave_arc"}).json()["entity_key"]
leg = C.post("/zeta/revoke", json={"contract_id": e["contract_id"], "relying_party": "zeta_issuer"})
check("deprecated field still accepted", leg.status_code == 200, f"got {leg.status_code}")
check("and it propagates too", leg.json()["oracle_revoked"] is True)
check("bit down", C.post("/zeta/cleared", json={"entity_key": ke}).json()["is_cleared"] is False)

print("\n" + ("ALL REVOCATION-PROPAGATION TESTS PASS" if not FAILS
              else f"{len(FAILS)} FAILURES: {FAILS}"))
sys.exit(1 if FAILS else 0)
