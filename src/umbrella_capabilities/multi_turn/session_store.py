"""Ephemeral in-memory session store managed by umbrella capability fan-out."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class SessionState:
    messages: list[dict[str, Any]] = field(default_factory=list)
    turn_count: int = 0
    last_seen_at: float = field(default_factory=time.time)
    session_root_span: Any | None = None


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def resolve(self, session_id: str | None, now_ts: float | None = None) -> tuple[str, SessionState]:
        active_id = session_id or str(uuid4())
        state = self._sessions.setdefault(active_id, SessionState())
        state.last_seen_at = now_ts if now_ts is not None else time.time()
        return active_id, state

    def pop(self, session_id: str) -> SessionState | None:
        return self._sessions.pop(session_id, None)

    def reset(self, session_id: str) -> bool:
        return self.pop(session_id) is not None

    def reap_expired(self, *, now_ts: float, ttl_seconds: int) -> list[tuple[str, SessionState]]:
        if ttl_seconds <= 0:
            return []

        expired: list[tuple[str, SessionState]] = []
        for session_id, state in list(self._sessions.items()):
            if now_ts - state.last_seen_at > ttl_seconds:
                removed = self._sessions.pop(session_id, None)
                if removed is not None:
                    expired.append((session_id, removed))
        return expired
