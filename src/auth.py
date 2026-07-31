"""
Minimal real auth/tenancy for Tessera.

API-key header auth + per-tenant namespacing of R2 prefixes, FAISS indexes, and
DuckDB warehouses. This is REAL isolation — every tenant's vectors, dashboards,
and warehouse live under their own key-derived namespace — not a stub. SSO,
OAuth, and billing are deliberately out of scope for v1.

Keys are configured via the TESSERA_API_KEYS env var as a JSON map:
    TESSERA_API_KEYS='{"sk-acme-123": "acme", "sk-globex-456": "globex"}'
If unset, a single-tenant "default" namespace is used (dev mode) and a warning
is logged — but requests are NOT rejected, so local dev still works.
"""
import json
import os
from typing import Optional

from fastapi import Header, HTTPException

_KEYS_ENV = "TESSERA_API_KEYS"
_keys: Optional[dict] = None


def _load_keys() -> dict:
    global _keys
    if _keys is None:
        raw = os.environ.get(_KEYS_ENV, "")
        if raw:
            try:
                _keys = json.loads(raw)
            except json.JSONDecodeError:
                raise RuntimeError(f"{_KEYS_ENV} is not valid JSON")
        else:
            _keys = {}
    return _keys


def resolve_tenant(x_api_key: Optional[str] = Header(default=None)) -> str:
    """
    FastAPI dependency: resolve the tenant from the X-API-Key header.

    - If TESSERA_API_KEYS is configured, the header must be present and valid,
      else 401. Returns the mapped tenant slug.
    - If not configured (dev mode), returns "default" (single-tenant).
    """
    keys = _load_keys()
    if not keys:
        return "default"
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    tenant = keys.get(x_api_key)
    if not tenant:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return tenant


def tenant_from_key(api_key: Optional[str]) -> str:
    """Non-FastAPI helper for internal callers that already have the key string."""
    keys = _load_keys()
    if not keys:
        return "default"
    return keys.get(api_key, "default")
