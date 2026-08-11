"""
Tessera vector store: embedding failover, dimension safety, space identity.

Four defects, all live in production before this:

1. ARCTIC_EMBED_URL pointed at the Modal tessera-arctic-embed endpoint. The
   Modal workspace exceeded its spend limit, so `modal app list` still reports
   the app "deployed" while the URL returns 404. `_embed_remote` had no except
   branch, so every embed raised HTTPError and took the request with it.

2. add() adapted self.dim on a dimension change, but `_ensure_index` only built
   when `self.index is None` -- so the index stayed 768-dim while self.dim said
   1024 and faiss.IndexFlatIP.add raised a BARE AssertionError with no message.
   The adaptation branch was dead code.

3. Nothing stopped 768-dim local vectors and 1024-dim remote vectors from
   entering the same tenant index across a backend switch.

4. The subtle one. Replacing the dead Modal endpoint with gemini-embedding-001
   pinned to outputDimensionality=768 makes the new backend the SAME WIDTH as
   local arctic-embed-m, so a dimension check cannot see the switch at all --
   and inner products across two unrelated geometries are confident nonsense.
   The store therefore pins an embed_space() identity, not just a width.

Run: python3 test_vector_dim_fallback.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, "/workspace/fireworks_sql_git")

import numpy as np  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


os.environ["ARCTIC_EMBED_URL"] = "https://testing1235--tessera-arctic-embed-embed.modal.run"
os.environ.pop("GEMINI_API_KEYS", None)
os.environ.pop("GEMINI_API_KEY", None)
import src.vector.store as store  # noqa: E402

DIM_REMOTE, DIM_LOCAL = 1024, 768
CHUNKS = [{"doc_id": "d1", "chunk_id": f"c{i}", "text": f"chunk {i}", "source": "s"} for i in range(3)]


def fake_remote(texts):
    return np.random.rand(len(texts), DIM_REMOTE).astype("float32")


def dead_remote(texts):
    raise urllib.error.HTTPError(store._ARCTIC_EMBED_URL, 404, "Not Found", {}, None)


store._embed_local = lambda texts: np.random.rand(len(texts), DIM_LOCAL).astype("float32")
# arm(gemini=True) stubs _embed_gemini with a random-vector lambda so the space
# tests can run without a network. Section 8 exercises the REAL function, so
# keep a handle on it before anything overwrites the module attribute.
_REAL_EMBED_GEMINI = store._embed_gemini


def arm(*, arctic, gemini=False):
    """Put the module in a known tier state."""
    store._embed_remote = fake_remote if arctic else dead_remote
    store._remote_dead = not arctic
    store._gemini_dead = not gemini
    if gemini:
        os.environ["GEMINI_API_KEYS"] = "space-identity-probe"
        store._embed_gemini = lambda texts: np.random.rand(len(texts), DIM_LOCAL).astype("float32")
    else:
        os.environ.pop("GEMINI_API_KEYS", None)


print("\n1. a healthy remote endpoint is used and its dimension is adopted")
arm(arctic=True)
vs = store.VectorStore(tenant="t_remote")
check("index starts at the 768 default", vs.index.d == DIM_LOCAL, str(vs.index.d))
check("add() accepts 1024-dim remote vectors", vs.add(CHUNKS) == 3)
check("index was actually REBUILT to 1024 (the dead-code bug)",
      vs.index.d == DIM_REMOTE and vs.dim == DIM_REMOTE, f"index.d={vs.index.d} dim={vs.dim}")
check("vectors are in the index", vs.count() == 3)
check("backend reports arctic-remote", store.embed_backend() == "arctic-remote", store.embed_backend())
check("space is pinned on the index", vs.space == "arctic-remote:snowflake-arctic-embed-l-v2.0", str(vs.space))

print("\n2. a dead remote endpoint fails over instead of 500ing")
store._embed_remote = dead_remote
store._remote_dead = False
out = store.embed_texts(["anything"])
check("embed_texts returns vectors rather than raising", out.shape == (1, DIM_LOCAL), str(out.shape))
check("remote is latched dead after one failure", store._remote_dead is True)
check("backend falls through to local", store.embed_backend() == "local", store.embed_backend())
calls = {"n": 0}


def counting_dead(texts):
    calls["n"] += 1
    return dead_remote(texts)


store._embed_remote = counting_dead
store.embed_texts(["a"]); store.embed_texts(["b"])
check("a dead endpoint is not retried on every call", calls["n"] == 0, f"retries={calls['n']}")

print("\n3. an empty store adapts to whichever backend answers")
arm(arctic=False)
vs2 = store.VectorStore(tenant="t_local")
check("add() works on the local fallback", vs2.add(CHUNKS) == 3)
check("index is 768-dim", vs2.index.d == DIM_LOCAL and vs2.count() == 3)
check("space pinned to local", vs2.space == f"local:{store._MODEL_NAME}", str(vs2.space))

print("\n4. a mid-life backend switch is refused, not mixed")
arm(arctic=True)
err = ""
try:
    vs2.add(CHUNKS)
except ValueError as e:
    err = str(e)
check("switching backends on a populated index raises", bool(err))
check("the error names both spaces and the tenant",
      "local:" in err and "arctic-remote:" in err and "t_local" in err, err[:110])
check("the existing index is untouched", vs2.index.d == DIM_LOCAL and vs2.count() == 3)

print("\n5. querying under the wrong backend is refused, not silently wrong")
qerr = ""
try:
    vs2.search("anything", k=1)
except ValueError as e:
    qerr = str(e)
check("search raises a named error, not a bare faiss assert", "meaningless" in qerr, qerr[:110])
arm(arctic=False)
hits = vs2.search("chunk 1", k=2)
check("search works again once the backend matches", len(hits) == 2, str(len(hits)))

print("\n6. regression guard on the original crash shape")
arm(arctic=True)
vs3 = store.VectorStore(tenant="t_crash")
vs3.dim = DIM_REMOTE
vs3._ensure_index()
check("_ensure_index rebuilds on a dim change", vs3.index.d == DIM_REMOTE, str(vs3.index.d))
try:
    vs3.index.add(np.random.rand(2, DIM_REMOTE).astype("float32"))
    ok = True
except AssertionError:
    ok = False
check("faiss add no longer bare-asserts", ok)

print("\n7. SAME WIDTH, DIFFERENT GEOMETRY — the check a dimension test cannot make")
arm(arctic=False)
vs4 = store.VectorStore(tenant="t_space")
vs4.add(CHUNKS)
check("indexed under the local space", vs4.space == f"local:{store._MODEL_NAME}", str(vs4.space))
arm(arctic=False, gemini=True)
check("gemini is a different space at an IDENTICAL width",
      store.embed_space().startswith("gemini:") and store._GEMINI_EMBED_DIM == DIM_LOCAL,
      store.embed_space())
serr = ""
try:
    vs4.add(CHUNKS)
except ValueError as e:
    serr = str(e)
check("adding across equal-width spaces is refused", "embedding space changed" in serr, serr[:100])
qerr2 = ""
try:
    vs4.search("chunk 1", k=1)
except ValueError as e:
    qerr2 = str(e)
check("querying across equal-width spaces is refused", "meaningless" in qerr2, qerr2[:100])
check("index untouched", vs4.count() == 3 and vs4.space == f"local:{store._MODEL_NAME}",
      f"count={vs4.count()} space={vs4.space}")

print("\n8. gemini vectors are L2-normalised before entering an IndexFlatIP")
# batchEmbedContents returns unnormalised vectors. IndexFlatIP scores raw inner
# products, so without normalisation a longer chunk wins on magnitude alone.
raw = [3.0, 4.0] + [0.0] * (DIM_LOCAL - 2)  # norm exactly 5
payload = json.dumps({"embeddings": [{"values": raw}, {"values": raw}]}).encode()


class _Resp:
    def read(self): return payload
    def __enter__(self): return self
    def __exit__(self, *a): return False


os.environ["GEMINI_API_KEYS"] = "probe"
store._embed_gemini = _REAL_EMBED_GEMINI  # undo the section-7 stub
check("section 8 is exercising the real _embed_gemini",
      store._embed_gemini.__name__ == "_embed_gemini", store._embed_gemini.__name__)
_orig = urllib.request.urlopen
urllib.request.urlopen = lambda *a, **k: _Resp()
try:
    got = store._embed_gemini(["x", "y"])
finally:
    urllib.request.urlopen = _orig
norms = np.linalg.norm(got, axis=1)
check("raw API vector had norm 5.0", abs(float(np.linalg.norm(raw)) - 5.0) < 1e-6)
check("returned vectors are unit-norm", bool(np.allclose(norms, 1.0, atol=1e-5)), str(norms))
check("direction is preserved", abs(float(got[0][0]) - 0.6) < 1e-5 and abs(float(got[0][1]) - 0.8) < 1e-5,
      f"{got[0][0]:.4f},{got[0][1]:.4f}")

print("\n9. the request is on the wire format the API actually accepts, and keys rotate")
# generativelanguage.googleapis.com authenticates an API key via the
# x-goog-api-key header. Sending it as `Authorization: Bearer` -- the shape
# every other tier in this system uses -- returns 401 with an OAuth error that
# reads like the key is invalid. Pin the header so a future refactor cannot
# quietly "harmonise" it into a bearer token.
seen: list[dict] = []


def _capture(req, timeout=None):
    seen.append({"url": req.full_url, "headers": dict(req.headers), "body": json.loads(req.data)})
    if len(seen) == 1:  # first key throttled -> must rotate, not raise
        raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {}, None)
    return _Resp()


os.environ["GEMINI_API_KEYS"] = "key_one,key_two"
urllib.request.urlopen = _capture
try:
    got9 = store._embed_gemini(["x", "y"])
finally:
    urllib.request.urlopen = _orig
hdrs = {k.lower(): v for k, v in seen[-1]["headers"].items()}
check("auth is x-goog-api-key, not an OAuth bearer",
      "x-goog-api-key" in hdrs and "authorization" not in hdrs, ",".join(sorted(hdrs)))
check("a throttled key rotates to the next one", len(seen) == 2 and got9.shape == (2, DIM_LOCAL),
      f"attempts={len(seen)} keys={[h['headers'].get('X-goog-api-key') for h in seen]}")
check("outputDimensionality is pinned so the width matches the index",
      all(r["outputDimensionality"] == DIM_LOCAL for r in seen[-1]["body"]["requests"]),
      str(seen[-1]["body"]["requests"][0]["outputDimensionality"]))
check("endpoint is batchEmbedContents, one request per text",
      seen[-1]["url"].endswith(":batchEmbedContents") and len(seen[-1]["body"]["requests"]) == 2,
      seen[-1]["url"].rsplit("/", 1)[-1])
os.environ["GEMINI_API_KEYS"] = "dead_one,dead_two"
urllib.request.urlopen = lambda req, timeout=None: (_ for _ in ()).throw(
    urllib.error.HTTPError(req.full_url, 403, "Forbidden", {}, None))
allerr = ""
try:
    store._embed_gemini(["x"])
except RuntimeError as e:
    allerr = str(e)
finally:
    urllib.request.urlopen = _orig
check("exhausting every key raises a named error, not a silent empty array",
      "all Gemini embedding keys failed" in allerr, allerr[:70])

LIVE_KEY = os.environ.get("ZETA_LIVE_GEMINI_KEY", "").strip()
if LIVE_KEY:
    print("\n10. LIVE call against generativelanguage.googleapis.com")
    os.environ["GEMINI_API_KEYS"] = LIVE_KEY
    live = store._embed_gemini(["Cayman HoldCo Ltd holds 40% of Acme OpCo Ltd",
                                "Blue Harbour Trust holds 40% of Cayman HoldCo Ltd",
                                "How many tracks are in the Rock genre?"])
    check("real API returned the pinned width", live.shape == (3, DIM_LOCAL), str(live.shape))
    check("real vectors are unit-norm", bool(np.allclose(np.linalg.norm(live, axis=1), 1.0, atol=1e-4)),
          str(np.round(np.linalg.norm(live, axis=1), 6)))
    sim_related = float(live[0] @ live[1])
    sim_unrelated = float(live[0] @ live[2])
    check("the geometry is real: two ownership sentences beat an unrelated one",
          sim_related > sim_unrelated + 0.10, f"related={sim_related:.3f} unrelated={sim_unrelated:.3f}")
else:
    print("\n10. LIVE gemini call SKIPPED (set ZETA_LIVE_GEMINI_KEY to run)")

print("\n" + ("ALL VECTOR TESTS PASS" if not FAILS else f"{len(FAILS)} FAILURES: {FAILS}"))
sys.exit(1 if FAILS else 0)
