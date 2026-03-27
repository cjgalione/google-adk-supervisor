"""Ephemeral in-memory session store managed by umbrella capability fan-out."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class SessionState:
    messages: list[dict] = field(default_factory=list)


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def resolve(self, session_id: str | None) -> tuple[str, SessionState]:
        active_id = session_id or str(uuid4())
        state = self._sessions.setdefault(active_id, SessionState())
        return active_id, state

    def reset(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None
