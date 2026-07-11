"""Rate limiting and usage tracking for the text-to-SQL API.

Provides per-client rate limiting using a sliding window algorithm and
tracks usage metrics (queries, tokens, cost estimates) per session and
globally. Essential for a multi-tenant platform with many users.

Usage in FastAPI:
    limiter = RateLimiter(requests_per_minute=30, requests_per_hour=500)
    tracker = UsageTracker()

    # In a route:
    client_id = request.client.host  # or session_id
    allowed, info = limiter.check(client_id)
    if not allowed:
        raise HTTPException(429, detail="Rate limit exceeded", headers={"Retry-After": str(info["retry_after"])})
    # ... process query ...
    tracker.record_query(session_id, model, latency, tokens_in, tokens_out)
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RateLimitInfo:
    """Information about the current rate limit state for a client."""
    allowed: bool
    requests_this_minute: int
    requests_this_hour: int
    limit_per_minute: int
    limit_per_hour: int
    retry_after: int  # seconds until the rate limit window resets


class RateLimiter:
    """
    Sliding window rate limiter with per-minute and per-hour limits.

    Uses in-memory deques to track request timestamps per client.
    For production with multiple server instances, this would be backed
    by Redis with INCR + EXPIRE.
    """

    def __init__(
        self,
        requests_per_minute: int = 30,
        requests_per_hour: int = 500,
    ):
        self.rpm_limit = requests_per_minute
        self.rph_limit = requests_per_hour
        self._minute_windows: dict[str, deque] = defaultdict(deque)
        self._hour_windows: dict[str, deque] = defaultdict(deque)

    def check(self, client_id: str) -> tuple[bool, RateLimitInfo]:
        """
        Check if a client is allowed to make a request.

        Call this BEFORE processing the request. If allowed, the request
        is recorded. If not allowed, returns retry_after in seconds.

        Args:
            client_id: Unique identifier (IP address, session ID, API key)

        Returns:
            Tuple of (allowed: bool, info: RateLimitInfo)
        """
        now = time.time()

        # Clean up expired entries in both windows
        minute_window = self._minute_windows[client_id]
        hour_window = self._hour_windows[client_id]

        # Remove entries older than 60 seconds from minute window
        while minute_window and minute_window[0] < now - 60:
            minute_window.popleft()

        # Remove entries older than 3600 seconds from hour window
        while hour_window and hour_window[0] < now - 3600:
            hour_window.popleft()

        rpm_count = len(minute_window)
        rph_count = len(hour_window)

        # Check limits
        if rpm_count >= self.rpm_limit:
            # Calculate retry_after: time until oldest entry in minute window expires
            retry_after = int(60 - (now - minute_window[0])) + 1
            return False, RateLimitInfo(
                allowed=False,
                requests_this_minute=rpm_count,
                requests_this_hour=rph_count,
                limit_per_minute=self.rpm_limit,
                limit_per_hour=self.rph_limit,
                retry_after=max(1, retry_after),
            )

        if rph_count >= self.rph_limit:
            retry_after = int(3600 - (now - hour_window[0])) + 1
            return False, RateLimitInfo(
                allowed=False,
                requests_this_minute=rpm_count,
                requests_this_hour=rph_count,
                limit_per_minute=self.rpm_limit,
                limit_per_hour=self.rph_limit,
                retry_after=max(1, retry_after),
            )

        # Allowed — record the request
        minute_window.append(now)
        hour_window.append(now)

        return True, RateLimitInfo(
            allowed=True,
            requests_this_minute=rpm_count + 1,
            requests_this_hour=rph_count + 1,
            limit_per_minute=self.rpm_limit,
            limit_per_hour=self.rph_limit,
            retry_after=0,
        )

    def get_status(self, client_id: str) -> dict:
        """Get current rate limit status for a client without recording a request."""
        now = time.time()
        minute_window = self._minute_windows.get(client_id, deque())
        hour_window = self._hour_windows.get(client_id, deque())

        # Count active entries
        rpm = sum(1 for t in minute_window if t > now - 60)
        rph = sum(1 for t in hour_window if t > now - 3600)

        return {
            "client_id": client_id,
            "requests_this_minute": rpm,
            "requests_this_hour": rph,
            "limit_per_minute": self.rpm_limit,
            "limit_per_hour": self.rph_limit,
            "remaining_minute": max(0, self.rpm_limit - rpm),
            "remaining_hour": max(0, self.rph_limit - rph),
        }


@dataclass
class SessionUsage:
    """Usage metrics for a single session."""
    session_id: str
    model: str
    query_count: int = 0
    total_latency: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    estimated_cost: float = 0.0
    cache_hits: int = 0
    first_query_at: float = field(default_factory=time.time)
    last_query_at: float = field(default_factory=time.time)
    questions: list = field(default_factory=list)


class UsageTracker:
    """
    Tracks usage metrics per session and globally.

    Records queries, latency, token usage, and estimated cost.
    Provides aggregation for dashboards and billing.
    """

    # Model pricing per million tokens (in, out)
    MODEL_PRICING = {
        "accounts/fireworks/models/gpt-oss-120b": (0.15, 0.60),
        "accounts/fireworks/models/deepseek-v4-flash": (0.14, 0.28),
        "accounts/fireworks/models/glm-5p2": (1.40, 4.40),
        "accounts/fireworks/models/kimi-k2p6": (0.95, 4.00),
        "accounts/fireworks/models/deepseek-v4-pro": (1.74, 3.48),
    }

    def __init__(self):
        self._sessions: dict[str, SessionUsage] = {}
        self._global = {
            "total_queries": 0,
            "total_cache_hits": 0,
            "total_latency": 0.0,
            "total_tokens_in": 0,
            "total_tokens_out": 0,
            "total_cost": 0.0,
            "started_at": time.time(),
        }

    def record_query(
        self,
        session_id: str,
        model: str,
        latency: float,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cache_hit: bool = False,
        question: str = "",
    ) -> None:
        """Record a single query's usage metrics."""
        # Estimate cost
        pricing = self.MODEL_PRICING.get(model, (0.15, 0.60))
        cost = (tokens_in / 1_000_000 * pricing[0]) + (tokens_out / 1_000_000 * pricing[1])

        # Per-session tracking
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionUsage(session_id=session_id, model=model)
        s = self._sessions[session_id]
        s.query_count += 1
        s.total_latency += latency
        s.total_tokens_in += tokens_in
        s.total_tokens_out += tokens_out
        s.estimated_cost += cost
        if cache_hit:
            s.cache_hits += 1
        s.last_query_at = time.time()
        if question:
            s.questions.append(question[:200])  # truncate for memory

        # Global tracking
        self._global["total_queries"] += 1
        self._global["total_latency"] += latency
        self._global["total_tokens_in"] += tokens_in
        self._global["total_tokens_out"] += tokens_out
        self._global["total_cost"] += cost
        if cache_hit:
            self._global["total_cache_hits"] += 1

    def get_session_usage(self, session_id: str) -> Optional[dict]:
        """Get usage metrics for a specific session."""
        s = self._sessions.get(session_id)
        if not s:
            return None
        return {
            "session_id": s.session_id,
            "model": s.model,
            "query_count": s.query_count,
            "avg_latency": round(s.total_latency / s.query_count, 3) if s.query_count else 0,
            "total_tokens_in": s.total_tokens_in,
            "total_tokens_out": s.total_tokens_out,
            "estimated_cost": round(s.estimated_cost, 6),
            "cache_hits": s.cache_hits,
            "cache_hit_rate": round(s.cache_hits / s.query_count, 4) if s.query_count else 0,
        }

    def get_global_usage(self) -> dict:
        """Get global usage metrics across all sessions."""
        total = self._global["total_queries"]
        uptime = time.time() - self._global["started_at"]
        return {
            **self._global,
            "avg_latency": round(self._global["total_latency"] / total, 3) if total else 0,
            "total_cost": round(self._global["total_cost"], 6),
            "cache_hit_rate": round(self._global["total_cache_hits"] / total, 4) if total else 0,
            "active_sessions": len(self._sessions),
            "uptime_seconds": round(uptime),
            "queries_per_minute": round(total / (uptime / 60), 2) if uptime > 0 else 0,
        }
