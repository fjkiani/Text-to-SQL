"""
Retrieval-augmented answering over the tenant's vector store.

For unstructured questions, retrieve the top-k chunks from the tenant's FAISS
index and answer with an LLM, citing the source chunks (provenance). Uses the
Tessera LLM client (Arctic-first, Fireworks fallback) so the backend is always
reported.
"""
from typing import Optional

from src.vector.store import VectorStore
from src.llm.client import chat


def retrieve(tenant: str, question: str, k: int = 5) -> list[dict]:
    """Return the top-k chunks for a question from the tenant's vector store."""
    vs = VectorStore(tenant=tenant)
    if vs.count() == 0:
        vs.load()  # try to pull a persisted index from R2
    return vs.search(question, k=k)


def answer_with_retrieval(tenant: str, question: str, k: int = 5) -> dict:
    """
    Answer an unstructured question using retrieved chunks as context.

    Returns {answer, backend, citations:[{doc_id, chunk_id, source, score, text}],
             retrieved_count}. Raises if the tenant has no indexed documents
    (loud — no hallucinated answer with empty context).
    """
    hits = retrieve(tenant, question, k=k)
    if not hits:
        raise ValueError(
            f"No documents indexed for tenant '{tenant}'. Upload a document first."
        )

    context = "\n\n".join(
        f"[{i+1}] (source: {h.get('source')}, score: {h.get('score', 0):.3f})\n{h.get('text')}"
        for i, h in enumerate(hits)
    )
    messages = [
        {"role": "system", "content": (
            "You are a data-intelligence assistant. Answer the user's question using ONLY "
            "the provided context chunks. Cite the chunk numbers you used (e.g. [1], [2]). "
            "If the context does not contain the answer, say so explicitly — do not invent facts."
        )},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    out = chat(messages, max_tokens=512)
    return {
        "answer": out["text"],
        "backend": out["backend"],
        "citations": [
            {"doc_id": h.get("doc_id"), "chunk_id": h.get("chunk_id"),
             "source": h.get("source"), "score": h.get("score"),
             "text": (h.get("text") or "")[:300]}
            for h in hits
        ],
        "retrieved_count": len(hits),
    }
