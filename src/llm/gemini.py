"""
Google Gemini backend with model failover.

Why this is the primary path: the OpenRouter free tier is genuinely free but
genuinely unreliable for a compliance workload — its free inventory churns
(17 free models one day, 14 the next), the good models 429 under load, and the
one fast model on the list is a reasoning model that burns its whole token
budget on chain-of-thought before emitting JSON. Gemini Flash answers the same
ownership-extraction prompt in single-digit seconds, deterministically.

OpenRouter is kept as the fallback tier, not deleted: two independent providers
means a provider outage degrades latency instead of failing clearance runs.

Auth: this credential authenticates as an API key (`x-goog-api-key` header or
`?key=`), NOT as an OAuth bearer — `Authorization: Bearer` returns 401
UNAUTHENTICATED and `oauth2/tokeninfo` rejects it as invalid_token. Verified
against generativelanguage.googleapis.com.

Env:
  GEMINI_API_KEYS   comma-separated keys (rotated on 429/quota)
  GEMINI_API_KEY    single-key alias
  GEMINI_MODELS     comma-separated model ids, ordered best -> fallback
  GEMINI_TIMEOUT_S  per-request timeout (default 45)
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"

# Ordered fast-and-cheap -> stronger fallback. Flash-class first: the extraction
# prompt is a structured-output task, not a reasoning task, so paying pro-class
# latency buys nothing and risks the gateway budget.
DEFAULT_MODELS = [
    "gemini-2.5-flash",
    "gemini-3.6-flash",
    "gemini-2.5-flash-lite",
    "gemma-4-31b-it",
    "gemini-2.5-pro",
]

_COOLING: dict[str, float] = {}
_COOLDOWN_S = float(os.environ.get("GEMINI_COOLDOWN_S", "60"))
_LAST_GOOD: dict[str, str] = {}


def _keys() -> list[str]:
    raw = os.environ.get("GEMINI_API_KEYS", "") or os.environ.get("GEMINI_API_KEY", "")
    return [k.strip() for k in raw.split(",") if k.strip()]


def _models() -> list[str]:
    raw = os.environ.get("GEMINI_MODELS", "")
    got = [m.strip() for m in raw.split(",") if m.strip()]
    return got or DEFAULT_MODELS


def _available(keys: list[str]) -> list[str]:
    now = time.time()
    live = [k for k in keys if _COOLING.get(k, 0) < now]
    if not live:
        _COOLING.clear()
        return keys
    lg = _LAST_GOOD.get("key")
    if lg in live:
        i = live.index(lg)
        live = live[i:] + live[:i]
    return live


def _to_gemini(messages: list[dict]) -> tuple[list[dict], str | None]:
    """Translate OpenAI-style messages into Gemini contents + systemInstruction."""
    system_parts: list[str] = []
    contents: list[dict] = []
    for m in messages:
        role = m.get("role", "user")
        text = m.get("content", "")
        if role == "system":
            system_parts.append(text)
            continue
        contents.append(
            {"role": "model" if role == "assistant" else "user", "parts": [{"text": text}]}
        )
    return contents, ("\n\n".join(system_parts) if system_parts else None)


def _extract_text(out: dict) -> str:
    """
    Pull text out of a Gemini response.

    Defensive on purpose: a response can come back with no `parts` at all when
    the model is cut off by MAX_TOKENS or blocked by a safety filter. Returning
    "" there would look like a successful empty extraction — i.e. an entity with
    no owners, which in KYB is a false clear. Raise instead so the caller fails
    over to the next model.
    """
    cands = out.get("candidates") or []
    if not cands:
        fb = (out.get("promptFeedback") or {}).get("blockReason")
        raise RuntimeError(f"no candidates (blockReason={fb})")
    cand = cands[0]
    parts = (cand.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts)
    if not text.strip():
        raise RuntimeError(f"empty candidate (finishReason={cand.get('finishReason')})")
    return text


def chat_gemini(messages: list[dict], **kw) -> dict:
    """Returns {"text", "backend", "model", "key_index", "attempts"}."""
    keys = _keys()
    if not keys:
        raise RuntimeError("GEMINI_API_KEYS not set")

    models = _models()
    lg_model = _LAST_GOOD.get("model")
    if lg_model in models:
        models = [lg_model] + [m for m in models if m != lg_model]

    timeout = float(os.environ.get("GEMINI_TIMEOUT_S", "45"))
    contents, system = _to_gemini(messages)

    body: dict = {
        "contents": contents,
        "generationConfig": {
            "temperature": kw.get("temperature", 0.0),
            "maxOutputTokens": kw.get("max_tokens", 1024),
        },
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}

    attempts: list[str] = []
    for model in models:
        for key in _available(keys):
            ki = keys.index(key) + 1
            url = f"{API_ROOT}/{model}:generateContent"
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json", "x-goog-api-key": key},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    out = json.loads(r.read())
                text = _extract_text(out)
                _LAST_GOOD["key"], _LAST_GOOD["model"] = key, model
                usage = out.get("usageMetadata") or {}
                return {
                    "text": text,
                    "backend": "gemini",
                    "model": model,
                    "key_index": ki,
                    "attempts": attempts,
                    "tokens": usage.get("totalTokenCount"),
                }
            except urllib.error.HTTPError as e:
                code = e.code
                detail = e.read().decode(errors="replace")[:160]
                attempts.append(f"{model}|k{ki}:{code}")
                if code in (429, 402):  # quota -> park this key, rotate
                    _COOLING[key] = time.time() + _COOLDOWN_S
                    continue
                if code in (400, 404):  # model unusable for this account
                    break
                continue  # 401/403/5xx -> next key
            except Exception as e:  # timeout, empty candidate, blocked
                attempts.append(f"{model}|k{ki}:{type(e).__name__}")
                continue

    raise RuntimeError(f"Gemini exhausted all models/keys: {attempts}")


if __name__ == "__main__":
    out = chat_gemini([{"role": "user", "content": "Reply with exactly: ZETA_GEMINI_OK"}])
    print("backend:", out["backend"], "| model:", out["model"], "| key:", out["key_index"])
    print("text:", out["text"][:120])
    print("attempts:", out["attempts"])
