"""Semantic similarity cache for multi-agent query responses.

Unlike exact-match caches, this recognizes semantically equivalent queries
even when phrased differently — "What is 2+2?" and "Add two and two" hit the
same cache entry.  Embeddings are generated via Google's text-embedding-004
model; similarity is measured by cosine distance with a configurable threshold.

TTL policy:
  - math    : 24 h  (arithmetic answers are stable)
  - research: 1 h   (web facts may go stale)
  - general : 1 h
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from google import genai


_EMBEDDING_MODEL = "text-embedding-004"
_DEFAULT_THRESHOLD = 0.92
_DEFAULT_MATH_TTL = 86_400.0    # 24 hours
_DEFAULT_RESEARCH_TTL = 3_600.0  # 1 hour


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    query: str
    embedding: list[float]
    response: str
    agent_type: str          # "math" | "research" | "general"
    critic_decision: dict[str, Any]
    timestamp: float
    hit_count: int = 0


# ---------------------------------------------------------------------------
# Pure-Python cosine similarity (avoids numpy dependency)
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Agent-type inference from execution trace
# ---------------------------------------------------------------------------

def detect_agent_type(
    messages: list[dict[str, Any]],
    critic_decision: dict[str, Any],
) -> str:
    """Infer which specialist agent handled this response from the trace."""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        for tc in msg.get("tool_calls") or []:
            name = str(tc.get("name", "") or "").lower()
            if "math" in name:
                return "math"
            if "research" in name:
                return "research"
    action = str(critic_decision.get("required_action", "") or "")
    if "math" in action:
        return "math"
    if "research" in action:
        return "research"
    return "general"


# ---------------------------------------------------------------------------
# SemanticCache
# ---------------------------------------------------------------------------

class SemanticCache:
    """Persistent semantic cache backed by Google text embeddings.

    Usage::

        cache = SemanticCache(google_api_key=os.environ["GOOGLE_API_KEY"])

        hit = await cache.get("what is 3 * 7")
        if hit:
            return hit.response

        result = await run_supervisor_with_critic(...)
        await cache.set(
            query="what is 3 * 7",
            response=result["final_output"],
            agent_type="math",
            critic_decision=result["critic_decision"],
        )
    """

    def __init__(
        self,
        *,
        google_api_key: str,
        cache_path: str = ".cache/query_cache.json",
        similarity_threshold: float = _DEFAULT_THRESHOLD,
        math_ttl: float = _DEFAULT_MATH_TTL,
        research_ttl: float = _DEFAULT_RESEARCH_TTL,
    ) -> None:
        self._client = genai.Client(api_key=google_api_key)
        self._cache_path = Path(cache_path)
        self._threshold = similarity_threshold
        self._ttl: dict[str, float] = {
            "math": math_ttl,
            "research": research_ttl,
            "general": research_ttl,
        }
        self._entries: list[CacheEntry] = []
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._cache_path.exists():
            return
        try:
            raw = json.loads(self._cache_path.read_text())
            self._entries = [CacheEntry(**e) for e in raw.get("entries", [])]
        except Exception:
            self._entries = []

    def _save(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"entries": [asdict(e) for e in self._entries]}
        self._cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed_sync(self, text: str) -> list[float]:
        result = self._client.models.embed_content(
            model=_EMBEDDING_MODEL,
            contents=text,
        )
        return list(result.embeddings[0].values)

    async def _embed(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._embed_sync, text)

    # ------------------------------------------------------------------
    # TTL / expiry
    # ------------------------------------------------------------------

    def _is_expired(self, entry: CacheEntry) -> bool:
        ttl = self._ttl.get(entry.agent_type, self._ttl["general"])
        return (time.time() - entry.timestamp) > ttl

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get(self, query: str) -> CacheEntry | None:
        """Return the closest non-expired cache entry, or None on miss."""
        async with self._lock:
            self._entries = [e for e in self._entries if not self._is_expired(e)]

        if not self._entries:
            self._misses += 1
            return None

        query_emb = await self._embed(query)

        async with self._lock:
            best_score = -1.0
            best_entry: CacheEntry | None = None
            for entry in self._entries:
                score = _cosine(query_emb, entry.embedding)
                if score > best_score:
                    best_score = score
                    best_entry = entry

            if best_entry is not None and best_score >= self._threshold:
                best_entry.hit_count += 1
                self._hits += 1
                self._save()
                return best_entry

        self._misses += 1
        return None

    async def set(
        self,
        *,
        query: str,
        response: str,
        agent_type: str,
        critic_decision: dict[str, Any],
    ) -> None:
        """Embed query and persist the response."""
        embedding = await self._embed(query)
        entry = CacheEntry(
            query=query,
            embedding=embedding,
            response=response,
            agent_type=agent_type,
            critic_decision=critic_decision,
            timestamp=time.time(),
        )
        async with self._lock:
            self._entries.append(entry)
            self._save()

    def stats(self) -> dict[str, Any]:
        """Return cache hit/miss statistics."""
        total = self._hits + self._misses
        return {
            "entries": len(self._entries),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
        }

    def clear(self) -> None:
        """Delete all cached entries and remove the backing file."""
        self._entries = []
        if self._cache_path.exists():
            self._cache_path.unlink()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_cache_from_env() -> SemanticCache | None:
    """Return a SemanticCache if CACHE_ENABLED=true and GOOGLE_API_KEY is set.

    Returns None (cache disabled) otherwise so callers can treat the cache
    as optional without extra logic.
    """
    raw = os.environ.get("CACHE_ENABLED", "false").strip().lower()
    if raw not in {"1", "true", "yes", "on"}:
        return None

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get(
        "BRAINTRUST_GATEWAY_API_KEY"
    )
    if not api_key:
        return None

    return SemanticCache(
        google_api_key=api_key,
        cache_path=os.environ.get("CACHE_PATH", ".cache/query_cache.json"),
        similarity_threshold=float(os.environ.get("CACHE_SIMILARITY_THRESHOLD", "0.92")),
        math_ttl=float(os.environ.get("CACHE_MATH_TTL", str(_DEFAULT_MATH_TTL))),
        research_ttl=float(os.environ.get("CACHE_RESEARCH_TTL", str(_DEFAULT_RESEARCH_TTL))),
    )
