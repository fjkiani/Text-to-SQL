"""
Modal app: tessera-arctic-embed

Serves Snowflake/snowflake-arctic-embed-l-v2.0 (568M params) as an HTTP embedding
endpoint. This is the embedding engine for the Tessera vector layer.

Deploy:  modal deploy src/llm/modal_embed.py
Call:    POST {"texts": ["...", "..."]}  ->  {"embeddings": [[...], ...], "dim": 1024}
"""
import modal

app = modal.App("tessera-arctic-embed")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("sentence-transformers>=2.7.0", "torch", "transformers", "numpy")
)

MODEL_NAME = "Snowflake/snowflake-arctic-embed-l-v2.0"


@app.cls(image=image, gpu="T4", scaledown_window=300)
class Embedder:
    @modal.enter()
    def load(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(MODEL_NAME)

    @modal.method()
    def embed(self, texts: list[str]) -> list[list[float]]:
        # arctic-embed models expect a query prefix for retrieval queries; for
        # symmetric document embedding we encode directly.
        emb = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return emb.tolist()


@app.function(image=image, gpu="T4", scaledown_window=300)
@modal.web_endpoint(method="POST")
def embed(item: dict):
    """HTTP endpoint. Body: {"texts": [...]}. Returns {"embeddings": [[...]], "dim": int}."""
    texts = item.get("texts") or []
    if not texts:
        return {"embeddings": [], "dim": 0}
    emb = Embedder().embed.remote(texts)
    dim = len(emb[0]) if emb else 0
    return {"embeddings": emb, "dim": dim}


@app.local_entrypoint()
def test():
    out = Embedder().embed.remote(["hello world", "text to sql"])
    print("dim:", len(out[0]), "norm check:", sum(x*x for x in out[0]) ** 0.5)
