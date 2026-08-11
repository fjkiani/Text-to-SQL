"""
LLM client with Arctic-first + Fireworks fallback.

Primary: Arctic-instruct self-hosted on Modal (OpenAI-compatible /v1/chat/completions).
Fallback: Fireworks gpt-oss-120b (the proven default from the text-to-SQL engine).

The fallback is the DESIGNED resilience path, not a silent swap: every response
reports which backend served it, and Arctic failures are logged. When the Modal
workspace is funded and Arctic-instruct deploys, set ARCTIC_LLM_URL and the client
prefers it automatically.
"""
import os
import json
import urllib.request
import urllib.error

ARCTIC_LLM_URL = os.environ.get("ARCTIC_LLM_URL", "")  # e.g. https://<modal>.modal.run/v1/chat/completions
FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
FIREWORKS_MODEL = os.environ.get("FIREWORKS_MODEL", "accounts/fireworks/models/gpt-oss-120b")
ARCTIC_TIMEOUT = float(os.environ.get("ARCTIC_LLM_TIMEOUT", "8"))  # cold-start tolerant but bounded


def _post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _chat_arctic(messages: list[dict], **kw) -> str:
    if not ARCTIC_LLM_URL:
        raise RuntimeError("ARCTIC_LLM_URL not set (Arctic-instruct not deployed)")
    payload = {
        "model": kw.get("model", "Snowflake/snowflake-arctic-instruct"),
        "messages": messages,
        "temperature": kw.get("temperature", 0.0),
        "max_tokens": kw.get("max_tokens", 1024),
    }
    out = _post(ARCTIC_LLM_URL, payload, {}, ARCTIC_TIMEOUT)
    return out["choices"][0]["message"]["content"]


def _chat_fireworks(messages: list[dict], **kw) -> str:
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("FIREWORKS_API_KEY not set")
    payload = {
        "model": kw.get("model", FIREWORKS_MODEL),
        "messages": messages,
        "temperature": kw.get("temperature", 0.0),
        "max_tokens": kw.get("max_tokens", 1024),
    }
    out = _post(FIREWORKS_URL, payload, {"Authorization": f"Bearer {api_key}"}, 60)
    return out["choices"][0]["message"]["content"]


def chat(messages: list[dict], **kw) -> dict:
    """
    Send a chat completion. Returns {"text": str, "backend": "arctic"|"fireworks"}.
    Tries Arctic first (if configured), falls back to Fireworks on any failure.
    """
    errors = {}
    if ARCTIC_LLM_URL:
        try:
            return {"text": _chat_arctic(messages, **kw), "backend": "arctic"}
        except Exception as e:
            errors["arctic"] = str(e)[:160]
            print(f"[llm.client] Arctic failed ({errors['arctic']}); trying OpenRouter")

    # OpenRouter: rotating keys + free-model failover. Primary resilient path.
    try:
        from openrouter import chat_openrouter
    except ImportError:
        from .openrouter import chat_openrouter
    try:
        out = chat_openrouter(messages, **kw)
        if errors:
            out["upstream_errors"] = errors
        return out
    except Exception as e:
        errors["openrouter"] = str(e)[:300]
        print(f"[llm.client] OpenRouter exhausted ({errors['openrouter'][:120]}); trying Fireworks")

    text = _chat_fireworks(messages, **kw)
    return {"text": text, "backend": "fireworks", "upstream_errors": errors}


if __name__ == "__main__":
    out = chat([{"role": "user", "content": "Reply with exactly: TESSERA_LLM_OK"}])
    print("backend:", out["backend"])
    print("text:", out["text"][:120])
    if "arctic_error" in out:
        print("arctic_error:", out["arctic_error"][:120])
