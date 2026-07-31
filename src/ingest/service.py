"""
Ingestion service: parse unstructured documents into normalized, chunked records
using the Unstructured engine.

Real parse via unstructured.partition.auto — no fake chunks. Each output chunk
carries provenance (doc_id, chunk_id, source filename, element type, page).
"""
import os
import hashlib
import uuid
from dataclasses import dataclass, field, asdict


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    source: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def _doc_id_for(path: str, tenant: str) -> str:
    h = hashlib.sha256()
    h.update(tenant.encode())
    h.update(os.path.basename(path).encode())
    try:
        with open(path, "rb") as f:
            h.update(f.read(65536))
    except Exception:
        pass
    return h.hexdigest()[:16]


def partition_file(path: str):
    """Parse a file into Unstructured elements. Raises on failure (loud, not silent)."""
    from unstructured.partition.auto import partition
    elements = partition(filename=path)
    if not elements:
        raise ValueError(f"Unstructured returned no elements for {path}")
    return elements


def chunk_elements(elements, doc_id: str, source: str, max_chars: int = 1500, overlap: int = 200):
    """
    Group Unstructured elements into retrieval-sized chunks.
    Respects element boundaries (titles, narratives, tables). Merges small
    adjacent elements up to max_chars; splits oversized elements with overlap.
    """
    chunks = []
    buf = []
    buf_len = 0
    buf_meta = {"element_types": set(), "pages": set()}

    def flush():
        nonlocal buf, buf_len, buf_meta
        if not buf:
            return
        text = "\n".join(buf).strip()
        if text:
            chunks.append(Chunk(
                doc_id=doc_id,
                chunk_id=str(uuid.uuid4())[:12],
                text=text,
                source=source,
                metadata={
                    "element_types": sorted(buf_meta["element_types"]),
                    "pages": sorted(p for p in buf_meta["pages"] if p is not None),
                    "char_count": len(text),
                },
            ))
        buf, buf_len = [], 0
        buf_meta = {"element_types": set(), "pages": set()}

    for el in elements:
        etype = type(el).__name__
        text = str(el).strip()
        if not text:
            continue
        page = getattr(getattr(el, "metadata", None), "page_number", None)
        # Oversized single element -> split with overlap
        if len(text) > max_chars:
            flush()
            start = 0
            while start < len(text):
                seg = text[start:start + max_chars]
                chunks.append(Chunk(
                    doc_id=doc_id, chunk_id=str(uuid.uuid4())[:12],
                    text=seg, source=source,
                    metadata={"element_types": [etype], "pages": [page] if page else [], "char_count": len(seg), "split": True},
                ))
                start += max_chars - overlap
            continue
        if buf_len + len(text) > max_chars and buf:
            flush()
        buf.append(text)
        buf_len += len(text)
        buf_meta["element_types"].add(etype)
        if page is not None:
            buf_meta["pages"].add(page)
    flush()
    return chunks


def ingest_file(path: str, tenant: str = "default") -> dict:
    """
    Parse + chunk a document. Returns {doc_id, chunk_count, chunks:[...]}.
    Raises on parse failure (no silent empty results).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    source = os.path.basename(path)
    doc_id = _doc_id_for(path, tenant)
    elements = partition_file(path)
    chunks = chunk_elements(elements, doc_id, source)
    if not chunks:
        raise ValueError(f"No chunks produced for {path} ({len(elements)} elements parsed)")
    return {
        "doc_id": doc_id,
        "source": source,
        "tenant": tenant,
        "chunk_count": len(chunks),
        "chunks": [c.to_dict() for c in chunks],
    }


if __name__ == "__main__":
    import sys, json
    path = sys.argv[1]
    tenant = sys.argv[2] if len(sys.argv) > 2 else "default"
    out = ingest_file(path, tenant)
    print(json.dumps({k: v for k, v in out.items() if k != "chunks"}, indent=2))
    print("first chunk:", out["chunks"][0]["text"][:200])
