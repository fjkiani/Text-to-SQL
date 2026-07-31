import os, json, threading
import numpy as np

_MODEL_NAME = "Snowflake/snowflake-arctic-embed-m"
_model = None
_model_lock = threading.Lock()

def _get_model():
    global _model
    with _model_lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(_MODEL_NAME)
    return _model

def embed_texts(texts):
    m = _get_model()
    return m.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype("float32")

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
        self._ensure_index()
    def _ensure_index(self):
        import faiss
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dim)
    @property
    def _index_key(self): return f"{self.tenant}/vector/index.faiss"
    @property
    def _meta_key(self): return f"{self.tenant}/vector/metadata.json"
    def add(self, chunks):
        if not chunks: return 0
        vecs = embed_texts([c["text"] for c in chunks])
        if vecs.shape[1] != self.dim:
            self.dim = vecs.shape[1]; self._ensure_index()
        self.index.add(vecs)
        for c in chunks:
            self.meta.append({"doc_id": c.get("doc_id"),"chunk_id": c.get("chunk_id"),
                "text": c.get("text"),"source": c.get("source"),"metadata": c.get("metadata",{})})
        return len(chunks)
    def search(self, query, k=5):
        if self.index.ntotal == 0: return []
        q = embed_texts([query]); k = min(k, self.index.ntotal)
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
            Body=json.dumps({"tenant":self.tenant,"dim":self.dim,"count":len(self.meta),"meta":self.meta}).encode())
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
