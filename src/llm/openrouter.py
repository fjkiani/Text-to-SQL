"""
OpenRouter backend with API-key rotation and free-model failover.

Why this exists: single-key/single-model LLM access is a single point of failure.
Free tiers rate-limit (429) and exhaust (402). This client treats both as routine
and rotates rather than failing the request:

  for model in MODELS:            # capability failover (big -> small)
      for key in KEYS:            # quota failover (rotate exhausted keys)
          try -> return
          429/402 -> mark key cooling, next key
          400/404 -> model unusable, break to next model

State is kept in-process so subsequent calls start from the last known-good
(key, model) pair instead of re-walking dead keys.

Env:
  OPENROUTER_KEYS    comma-separated sk-or-v1-... keys
  OPENROUTER_MODELS  comma-separated model ids (ordered best -> fallback)
"""
import json
import os
import time
import urllib.error
import urllib.request

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Ordered fastest-reliable -> slowest fallback. Latencies measured on this task
# (ownership-edge extraction, 3 keys, warm): nano-30b 8.9s, gpt-oss-20b 28.6s,
# gemma-4-26b 33.3s, nemotron-120b 112.8s. The 120B is last because a single
# call at that latency exceeds the Render gateway budget and 502s the request.
# nemotron-nano is DEMOTED despite being fastest (8.9s): it is a reasoning model
# and on the long extraction prompt it spends its whole token budget on
# chain-of-thought, returning truncated prose with no JSON array at all.
DEFAULT_MODELS = [
    "openai/gpt-oss-20b:free",
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "openrouter/free",
]

# Keys that returned 429/402 are parked here until cooldown expires.
_COOLING: dict[str, float] = {}
_COOLDOWN_S = float(os.environ.get("OPENROUTER_COOLDOWN_S", "60"))
_LAST_GOOD: dict[str, str] = {}


def _keys() -> list[str]:
    raw = os.environ.get("OPENROUTER_KEYS", "") or os.environ.get("OPENROUTER_API_KEY", "")
    return [k.strip() for k in raw.split(",") if k.strip().startswith("sk-or-")]


def _models() -> list[str]:
    raw = os.environ.get("OPENROUTER_MODELS", "")
    got = [m.strip() for m in raw.split(",") if m.strip()]
    return got or DEFAULT_MODELS


def _available(keys: list[str]) -> list[str]:
    now = time.time()
    live = [k for k in keys if _COOLING.get(k, 0) < now]
    if not live:  # every key cooling -> use them all anyway rather than hard-fail
        _COOLING.clear()
        return keys
    # start from last-good key so we don't re-walk the rotation each call
    lg = _LAST_GOOD.get("key")
    if lg in live:
        i = live.index(lg)
        live = live[i:] + live[:i]
    return live


def chat_openrouter(messages: list[dict], **kw) -> dict:
    """Returns {"text", "backend", "model", "key_index", "attempts"}."""
    keys = _keys()
    if not keys:
        raise RuntimeError("OPENROUTER_KEYS not set")
    models = _models()
    lg_model = _LAST_GOOD.get("model")
    if lg_model in models:  # prefer the model that worked last
        models = [lg_model] + [m for m in models if m != lg_model]

    attempts: list[str] = []
    for model in models:
        for key in _available(keys):
            payload = {
                "model": model,
                "messages": messages,
                "temperature": kw.get("temperature", 0.0),
                "max_tokens": kw.get("max_tokens", 1024),
            }
            req = urllib.request.Request(
                OPENROUTER_URL,
                data=json.dumps(payload).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {key}",
                    "HTTP-Referer": "https://openclaw.dev/zeta",
                    "X-Title": "Zeta Clearance KYB",
                },
                method="POST",
            )
            ki = keys.index(key) + 1
            try:
                with urllib.request.urlopen(req, timeout=kw.get("timeout", float(os.environ.get("OPENROUTER_TIMEOUT_S", "45")))) as r:
                    out = json.loads(r.read())
                text = (out.get("choices") or [{}])[0].get("message", {}).get("content", "")
                if not text.strip():
                    attempts.append(f"{model}|k{ki}:empty")
                    continue
                _LAST_GOOD["key"], _LAST_GOOD["model"] = key, model
                return {
                    "text": text, "backend": "openrouter", "model": model,
                    "key_index": ki, "attempts": attempts,
                }
            except urllib.error.HTTPError as e:
                code = e.code
                attempts.append(f"{model}|k{ki}:{code}")
                if code in (429, 402):          # quota -> rotate key
                    _COOLING[key] = time.time() + _COOLDOWN_S
                    continue
                if code in (400, 404):          # model bad -> next model
                    break
                continue                        # 5xx/401 -> next key
            except Exception as e:
                attempts.append(f"{model}|k{ki}:{type(e).__name__}")
                continue
    raise RuntimeError(f"all OpenRouter keys/models exhausted: {attempts}")


if __name__ == "__main__":
    out = chat_openrouter([{"role": "user", "content": "Reply with exactly: OPENROUTER_OK"}])
    print("model:", out["model"], "key:", out["key_index"])
    print("text:", out["text"][:120])
    print("attempts:", out["attempts"])
