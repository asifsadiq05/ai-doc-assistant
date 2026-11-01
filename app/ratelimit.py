import collections
import os
import threading
import time

from fastapi import HTTPException, Request

ENABLED = os.getenv("RATE_LIMIT", "1") == "1"
_hits: dict = collections.defaultdict(collections.deque)
_lock = threading.Lock()


def limit(name: str, max_calls: int, per_seconds: int):
    """FastAPI dependency: sliding-window limit per client IP (behind a proxy, forward the real IP)."""
    def dep(request: Request):
        if not ENABLED:
            return
        key, now = (name, request.client.host if request.client else "?"), time.monotonic()
        with _lock:
            q = _hits[key]
            while q and now - q[0] > per_seconds:
                q.popleft()
            if len(q) >= max_calls:
                wait = int(per_seconds - (now - q[0])) + 1
                raise HTTPException(429, f"Too many requests. Try again in {wait} seconds.",
                                    headers={"Retry-After": str(wait)})
            q.append(now)
    return dep
