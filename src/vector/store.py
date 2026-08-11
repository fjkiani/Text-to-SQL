import os, json, threading
import numpy as np

_MODEL_NAME = "Snowflake/snowflake-arctic-embed-m"
_model = None
_model_lock = threading.Lock()

# Remote-first embedding: when ARCTIC_EMBED_URL is set (the live Modal
# tessera-arctic-embed endpoint, 1024-dim l-v2.0), embed over HTTP so the web
# dyno never needs torch/sentence-transformers installed. Falls back to the
# local CPU model only when no remote endpoint is configured.
_ARCTIC_EMBED_URL = os.environ.get("ARCTIC_EMBED_URL", "").strip()

# The remote endpoint being *configured* is not the same as it being *alive*.
# The Modal workspace hosting tessera-arctic-embed hit its spend limit, so the
# control plane still reports the app "deployed" while the URL returns 404 --
# and `_embed_remote` had no except branch, so every embed raised HTTPError and
# took the request with it. Once a remote call fails we latch to local for the
# life of the process instead of paying the timeout on every subsequent call.
_remote_dead = False
_remote_lock = threading.Lock()

# Until an embed has actually been attempted in this process, _remote_dead is
# False only because nothing has tried yet -- so embed_backend() reports the
# CONFIGURED tier, which is exactly the control-plane lie that hid the dead
# Modal endpoint. Callers that report health must be able to tell "observed"
# from "assumed", so count real attempts.
_embed_calls = 0

def _get_model():
    global _model
    with _model_lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(_MODEL_NAME)
    return _model

def _embed_local(texts):
    m = _get_model()
    return m.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype("float32")

def _embed_remote(texts):
    import urllib.request
    payload = json.dumps({"texts": texts}).encode()
    req = urllib.request.Request(
        _ARCTIC_EMBED_URL, data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read())
    return np.asarray(out["embeddings"], dtype="float32")

# Hosted embedding tier. Replaces the dead Modal endpoint without dragging
# torch onto a 512MB dyno: gemini-embedding-001 accepts outputDimensionality,
# so it is pinned to the same 768 the local arctic-embed-m produces and drops
# into an existing index without a re-index. Same DIMENSION is not the same
# SPACE, which is why the store keys its guard on embed_space(), not on dim.
_GEMINI_EMBED_MODEL = os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-001")
_GEMINI_EMBED_DIM = int(os.environ.get("GEMINI_EMBED_DIM", "768"))
_gemini_dead = False

def _gemini_keys():
    raw = os.environ.get("GEMINI_API_KEYS") or os.environ.get("GEMINI_API_KEY") or ""
    return [k.strip() for k in raw.split(",") if k.strip()]

def _embed_gemini(texts):
    import urllib.request
    keys = _gemini_keys()
    if not keys:
        raise RuntimeError("no GEMINI_API_KEYS configured")
    body = json.dumps({"requests": [
        {"model": f"models/{_GEMINI_EMBED_MODEL}",
         "content": {"parts": [{"text": t}]},
         "outputDimensionality": _GEMINI_EMBED_DIM} for t in texts]}).encode()
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{_GEMINI_EMBED_MODEL}:batchEmbedContents")
    last = None
    for key in keys:  # rotate on a dead or throttled key, same policy as the LLM tier
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json", "x-goog-api-key": key})
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                out = json.loads(r.read())
            vecs = np.asarray([e["values"] for e in out["embeddings"]], dtype="float32")
            # batchEmbedContents does not L2-normalise. The index is
            # IndexFlatIP, so unnormalised vectors would make the "cosine"
            # scores length-biased -- longer chunks would win on magnitude.
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs / np.where(norms == 0, 1.0, norms)
        except Exception as e:
            last = e
    raise RuntimeError(f"all Gemini embedding keys failed: {last}")

def embed_texts(texts):
    global _remote_dead, _gemini_dead, _embed_calls
    _embed_calls += 1
    if _ARCTIC_EMBED_URL and not _remote_dead:
        try:
            return _embed_remote(texts)
        except Exception as e:
            with _remote_lock:
                _remote_dead = True
            print(f"[vector] arctic remote embed failed ({type(e).__name__}: {e}); "
                  f"falling through", flush=True)
    if _gemini_keys() and not _gemini_dead:
        try:
            return _embed_gemini(texts)
        except Exception as e:
            _gemini_dead = True
            print(f"[vector] gemini embed failed ({type(e).__name__}: {e}); "
                  f"falling back to local {_MODEL_NAME}", flush=True)
    return _embed_local(texts)

def embed_backend() -> str:
    if _ARCTIC_EMBED_URL and not _remote_dead:
        return "arctic-remote"
    if _gemini_keys() and not _gemini_dead:
        return "gemini"
    return "local"

def embed_space() -> str:
    """
    Identity of the vector space, not just its width.

    gemini-embedding-001 pinned to 768 and arctic-embed-m at 768 are the same
    shape and completely different geometries; an index that mixes them returns
    confident nonsense with no dimension error to catch it. Every write and
    every query compares this string.
    """
    b = embed_backend()
    if b == "arctic-remote":
        return "arctic-remote:snowflake-arctic-embed-l-v2.0"
    if b == "gemini":
        return f"gemini:{_GEMINI_EMBED_MODEL}:{_GEMINI_EMBED_DIM}"
    return f"local:{_MODEL_NAME}"

def _r2_client():
    import boto3
    return boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"], region_name="auto")

class VectorStore:
    def __init__(self, tenant="default", bucket=None):
        self.tenant = tenant
        self.bucket = bucket or os.environ.get("R2_BUCKET", "tessera-embeddings")
        self.dim = 768
        self.index = None
        self.meta = []
        # None until the first write; then pinned to the space that wrote it.
        self.space = None
        self._ensure_index()
    def _ensure_index(self):
        """
        Build the index, or REBUILD it when self.dim has changed.

        The old body was `if self.index is None`, which made the dimension
        adaptation in add() dead code: self.dim moved 768 -> 1024 while
        index.d stayed 768, and faiss.IndexFlatIP.add then failed a bare
        AssertionError with no message. Rebuilding on a dim change is only
        safe while the index is empty; add() refuses the switch otherwise.
        """
        import faiss
        if self.index is None or self.index.d != self.dim:
            self.index = faiss.IndexFlatIP(self.dim)
    @property
    def _index_key(self): return f"{self.tenant}/vector/index.faiss"
    @property
    def _meta_key(self): return f"{self.tenant}/vector/metadata.json"
    def add(self, chunks):
        if not chunks: return 0
        space = embed_space()
        if self.space and self.space != space and self.index is not None and self.index.ntotal:
            raise ValueError(
                f"embedding space changed '{self.space}' -> '{space}' but tenant "
                f"'{self.tenant}' already holds {self.index.ntotal} vectors; two spaces of the "
                f"same width are not comparable -- re-index the tenant instead of mixing them"
            )
        vecs = embed_texts([c["text"] for c in chunks])
        if vecs.shape[1] != self.dim:
            # Mixing 768-dim local vectors with 1024-dim remote ones in one
            # index is not a dimension error to paper over -- the inner
            # products would be meaningless. Adapt only while empty; otherwise
            # refuse loudly rather than corrupt an existing tenant index.
            if self.index is not None and self.index.ntotal:
                raise ValueError(
                    f"embedding dimension changed {self.dim} -> {vecs.shape[1]} "
                    f"({embed_backend()} backend) but tenant '{self.tenant}' already holds "
                    f"{self.index.ntotal} vectors; re-index the tenant instead of mixing spaces"
                )
            self.dim = vecs.shape[1]
            self._ensure_index()
        self.index.add(vecs)
        self.space = space
        for c in chunks:
            self.meta.append({"doc_id": c.get("doc_id"),"chunk_id": c.get("chunk_id"),
                "text": c.get("text"),"source": c.get("source"),"metadata": c.get("metadata",{})})
        return len(chunks)
    def search(self, query, k=5):
        if self.index.ntotal == 0: return []
        space = embed_space()
        if self.space and self.space != space:
            raise ValueError(
                f"tenant '{self.tenant}' was indexed with '{self.space}' but the active backend "
                f"is '{space}'; scores across two embedding spaces are meaningless -- re-index"
            )
        q = embed_texts([query])
        if q.shape[1] != self.index.d:
            raise ValueError(
                f"query embedded at {q.shape[1]}-dim ({embed_backend()} backend) but tenant "
                f"'{self.tenant}' index is {self.index.d}-dim; re-index before querying"
            )
        k = min(k, self.index.ntotal)
        scores, idxs = self.index.search(q, k)
        out = []
        for score, i in zip(scores[0], idxs[0]):
            if 0 <= i < len(self.meta):
                m = dict(self.meta[i]); m["score"] = float(score); out.append(m)
        return out
    def save(self):
        import faiss, tempfile
        s3 = _r2_client()
        with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as tf: tmp = tf.name
        try:
            faiss.write_index(self.index, tmp)
            with open(tmp,"rb") as f: arr = f.read()
        finally:
            try: os.unlink(tmp)
            except OSError: pass
        s3.put_object(Bucket=self.bucket, Key=self._index_key, Body=arr)
        s3.put_object(Bucket=self.bucket, Key=self._meta_key,
            Body=json.dumps({"tenant":self.tenant,"dim":self.dim,"space":self.space,
                             "count":len(self.meta),"meta":self.meta}).encode())
        return {"index_key":self._index_key,"meta_key":self._meta_key,"count":len(self.meta)}
    def load(self):
        import faiss, tempfile
        s3 = _r2_client()
        try:
            idx_obj = s3.get_object(Bucket=self.bucket, Key=self._index_key)
            meta_obj = s3.get_object(Bucket=self.bucket, Key=self._meta_key)
        except Exception:
            return False
        data = idx_obj["Body"].read()
        with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as tf:
            tf.write(data); tmp = tf.name
        try: self.index = faiss.read_index(tmp)
        finally:
            try: os.unlink(tmp)
            except OSError: pass
        payload = json.loads(meta_obj["Body"].read())
        self.dim = payload["dim"]; self.meta = payload["meta"]
        # Indexes written before space tracking carry no marker. Treat them as
        # unpinned rather than guessing which model produced them.
        self.space = payload.get("space")
        return True
    def count(self): return int(self.index.ntotal)

if __name__ == "__main__":
    vs = VectorStore(tenant="smoketest")
    n = vs.add([
        {"doc_id":"d1","chunk_id":"c1","text":"Total revenue grew 14% quarter over quarter.","source":"q3.docx"},
        {"doc_id":"d1","chunk_id":"c2","text":"Enterprise contracts drove most of the growth.","source":"q3.docx"},
        {"doc_id":"d2","chunk_id":"c1","text":"How many tracks are in the Rock genre?","source":"faq.txt"},
    ])
    print("added", n, "count", vs.count())
    for h in vs.search("revenue growth", k=2):
        print(f"  {h['score']:.3f} {h['source']} {h['text'][:50]}")
    saved = vs.save(); print("saved:", saved)
    vs2 = VectorStore(tenant="smoketest"); ok = vs2.load()
    print("reloaded:", ok, "count", vs2.count())
    print("search after reload:", [h["text"][:40] for h in vs2.search("revenue", k=1)])
