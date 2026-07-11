"""Semantic cache for text-to-SQL queries.

Caches (question -> SQL + results + summary) pairs. On a cache hit, returns
the stored response in <1ms without calling the LLM. Uses normalized text
matching with optional fuzzy similarity for paraphrased questions.

For production, this would be backed by Redis or a vector database (e.g.,
pgvector with embedding similarity). The in-memory implementation here
demonstrates the pattern and provides sub-100ms cache hits.
"""

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CacheEntry:
    """A single cache entry storing the full agent response."""
    question: str
    question_hash: str
    sql: str
    results: list
    columns: list
    summary: str
    latency: float
    attempts: int
    success: bool
    error: Optional[str]
    cached_at: float = field(default_factory=time.time)
    hit_count: int = 0


def normalize_question(question: str) -> str:
    """Normalize a question for cache key generation.

    Lowercases, strips whitespace, removes punctuation, collapses spaces.
    "What are the top 5 genres?" and "what are the top 5 genres ?" both
    normalize to the same key.
    """
    # Lowercase and strip
    q = question.lower().strip()
    # Remove punctuation
    q = re.sub(r'[^\w\s]', '', q)
    # Collapse whitespace
    q = re.sub(r'\s+', ' ', q)
    return q


def question_hash(question: str) -> str:
    """Generate a hash key from a normalized question."""
    normalized = normalize_question(question)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def word_set(question: str) -> set:
    """Extract the set of meaningful words from a question for similarity matching."""
    normalized = normalize_question(question)
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'of', 'in', 'on', 'at', 'to',
        'for', 'with', 'by', 'from', 'as', 'into', 'about', 'like', 'than',
        'then', 'now', 'just', 'also', 'only', 'very', 'too', 'so', 'such',
        'what', 'which', 'who', 'whom', 'whose', 'when', 'where', 'why', 'how',
        'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
        'some', 'any', 'no', 'not', 'nor', 'only', 'own', 'same', 'and', 'or',
        'but', 'if', 'because', 'while', 'show', 'list', 'give', 'me', 'tell',
        'get', 'find', 'display', 'return',
    }
    words = set(normalized.split()) - stop_words
    return words


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two word sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


class SemanticCache:
    """
    In-memory semantic cache for SQL query results.

    Features:
    - Exact match: normalized question hash lookup (O(1))
    - Fuzzy match: Jaccard similarity on word sets for paraphrased questions
    - TTL: entries expire after a configurable time-to-live
    - Hit tracking: records how often each entry is used
    - Metrics: cache hits, misses, and hit rate

    Usage:
        cache = SemanticCache(ttl_seconds=3600, similarity_threshold=0.85)
        hit = cache.get("What are the top 5 genres?")
        if hit:
            return hit  # sub-100ms response
        # ... call LLM ...
        cache.put(question, response)
    """

    def __init__(
        self,
        ttl_seconds: int = 3600,
        similarity_threshold: float = 0.85,
        max_entries: int = 10000,
    ):
        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self._cache: dict[str, CacheEntry] = {}
        self._metrics = {
            "hits_exact": 0,
            "hits_fuzzy": 0,
            "misses": 0,
            "evictions": 0,
            "total_requests": 0,
        }

    def get(self, question: str) -> Optional[CacheEntry]:
        """Look up a question in the cache. Returns CacheEntry if found, None if miss."""
        self._metrics["total_requests"] += 1
        qhash = question_hash(question)

        # 1. Exact match (normalized)
        if qhash in self._cache:
            entry = self._cache[qhash]
            if self._is_expired(entry):
                del self._cache[qhash]
                self._metrics["misses"] += 1
                return None
            entry.hit_count += 1
            self._metrics["hits_exact"] += 1
            return entry

        # 2. Fuzzy match (Jaccard similarity on word sets)
        q_words = word_set(question)
        if q_words:
            best_score = 0.0
            best_key = None
            for key, entry in self._cache.items():
                if self._is_expired(entry):
                    continue
                e_words = word_set(entry.question)
                score = jaccard_similarity(q_words, e_words)
                if score > best_score:
                    best_score = score
                    best_key = key

            if best_key and best_score >= self.similarity_threshold:
                entry = self._cache[best_key]
                entry.hit_count += 1
                self._metrics["hits_fuzzy"] += 1
                return entry

        self._metrics["misses"] += 1
        return None

    def put(self, question: str, sql: str, results: list, columns: list,
            summary: str, latency: float, attempts: int, success: bool,
            error: Optional[str] = None) -> None:
        """Store a query response in the cache."""
        # Evict expired entries if we're at capacity
        if len(self._cache) >= self.max_entries:
            self._evict_oldest()

        qhash = question_hash(question)
        self._cache[qhash] = CacheEntry(
            question=question,
            question_hash=qhash,
            sql=sql,
            results=results,
            columns=columns,
            summary=summary,
            latency=latency,
            attempts=attempts,
            success=success,
            error=error,
        )

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if a cache entry has expired."""
        return time.time() - entry.cached_at > self.ttl_seconds

    def _evict_oldest(self) -> None:
        """Evict the oldest entries to make room."""
        # First try to evict expired entries
        now = time.time()
        expired = [
            key for key, entry in self._cache.items()
            if now - entry.cached_at > self.ttl_seconds
        ]
        for key in expired:
            del self._cache[key]
            self._metrics["evictions"] += 1

        # If still at capacity, evict least-recently-used
        if len(self._cache) >= self.max_entries:
            # Sort by cached_at (oldest first)
            sorted_keys = sorted(
                self._cache.keys(),
                key=lambda k: self._cache[k].cached_at
            )
            # Evict 10% of entries
            n_evict = max(1, len(sorted_keys) // 10)
            for key in sorted_keys[:n_evict]:
                del self._cache[key]
                self._metrics["evictions"] += 1

    def invalidate(self) -> None:
        """Clear the entire cache (e.g., when the schema changes)."""
        self._cache.clear()

    def get_metrics(self) -> dict:
        """Return cache performance metrics."""
        total = self._metrics["total_requests"]
        hits = self._metrics["hits_exact"] + self._metrics["hits_fuzzy"]
        return {
            **self._metrics,
            "hit_rate": round(hits / total, 4) if total > 0 else 0.0,
            "cache_size": len(self._cache),
            "ttl_seconds": self.ttl_seconds,
            "similarity_threshold": self.similarity_threshold,
        }
