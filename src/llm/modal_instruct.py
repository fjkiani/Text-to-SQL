"""
Modal app: tessera-arctic-instruct

Serves Snowflake/snowflake-arctic-instruct (480B MoE, FP8) via vLLM behind an
OpenAI-compatible /v1/chat/completions endpoint. This is the reasoning engine
for the Tessera platform.

Hardware reality: Arctic-instruct is a 480B-param MoE that requires 8xH100 (or
8xA100-80GB) with FP8 quantization. First cold start downloads ~500GB of
checkpoint shards (20-30 min); the weights are cached on a Modal volume
thereafter. scaledown_window keeps a replica warm briefly after traffic, then
scales to zero to bound cost (~$25-40/hr while warm).

The Tessera LLM client (src/llm/client.py) treats this endpoint as primary and
falls back to Fireworks gpt-oss-120b when it is cold/unavailable — so the
product never hard-fails while Arctic warms up.

Deploy:  modal deploy src/llm/modal_instruct.py
Call:    POST /v1/chat/completions {"model": "...", "messages": [...]}
         (OpenAI-compatible; served by vLLM's built-in server)

NOTE: Requires a funded Modal workspace with 8xH100 quota. As of 2026-07-31 the
testing1235 workspace is over its spend limit, so this app is written and
API-correct but NOT yet deployed. Deploy the moment the workspace is funded.
"""
import modal

app = modal.App("tessera-arctic-instruct")

MODEL_NAME = "Snowflake/snowflake-arctic-instruct"
MODEL_DIR = "/models"
N_GPU = 8
GPU = f"H100:{N_GPU}"

# Persistent volume for the ~500GB checkpoint so cold starts after the first
# are fast (weights cached, not re-downloaded).
weights_volume = modal.Volume.from_name("arctic-instruct-weights", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.6.3",          # OpenAI-compatible server + FP8 + tensor parallel
        "transformers>=4.45",
        "huggingface_hub[hf_transfer]",
        "ray",                  # vLLM multi-GPU orchestration
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


def _download_weights():
    """Download the checkpoint to the persistent volume (first cold start only)."""
    import os
    from huggingface_hub import snapshot_download

    target = os.path.join(MODEL_DIR, MODEL_NAME.replace("/", "--"))
    if os.path.exists(os.path.join(target, "config.json")):
        print(f"[modal_instruct] weights already cached at {target}")
        return target
    print(f"[modal_instruct] downloading {MODEL_NAME} -> {target} (this takes 20-30 min)")
    snapshot_download(
        repo_id=MODEL_NAME,
        local_dir=target,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.pt", "*.bin"],  # prefer safetensors shards
    )
    weights_volume.commit()
    print("[modal_instruct] download complete, volume committed")
    return target


@app.function(
    image=image,
    gpu=GPU,
    volumes={MODEL_DIR: weights_volume},
    timeout=60 * 60 * 6,          # allow long first download + serve
    scaledown_window=15 * 60,     # stay warm 15 min after last request, then scale to 0
)
@modal.web_server(port=8000, startup_timeout=60 * 40)
def serve():
    """Launch vLLM's OpenAI-compatible server on 8 GPUs with FP8 + tensor parallelism."""
    import subprocess

    model_path = _download_weights()
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--served-model-name", MODEL_NAME,
        "--tensor-parallel-size", str(N_GPU),
        "--quantization", "fp8",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.92",
        "--port", "8000",
        "--host", "0.0.0.0",
        "--trust-remote-code",
    ]
    print("[modal_instruct] launching:", " ".join(cmd))
    subprocess.Popen(cmd)


@app.local_entrypoint()
def test():
    """Smoke-test the deployed endpoint (OpenAI-compatible chat completion)."""
    import json
    import urllib.request

    url = serve.web_url + "/v1/chat/completions"
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Reply with exactly: ARCTIC_OK"}],
        "temperature": 0.0,
        "max_tokens": 16,
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read())
    print("response:", out["choices"][0]["message"]["content"])
