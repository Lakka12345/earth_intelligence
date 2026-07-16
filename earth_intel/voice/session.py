"""
voice/session.py
Lightweight per-session state for the voice layer.

Each session tracks:
  - The language the user is speaking (detected from first voice input)
  - Conversation history (role + text + input channel)

Agent 1 never sees this object — it only ever receives plain English text.
This state is used by the API layer for logging and to skip re-detection on
subsequent turns from the same user.
"""

import time
from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class Turn:
    role: str          # "user" | "agent"
    text: str          # always English (translated if needed)
    via: str           # "voice" | "text"
    original: str = "" # original non-English text if translation occurred
    timestamp: float = field(default_factory=time.time)


@dataclass
class VoiceSession:
    session_id: str = field(default_factory=lambda: uuid4().hex)
    detected_language: str = "en"         # updated on first non-English voice input
    language_confirmed: bool = False      # True once we've seen at least one voice turn
    history: list[Turn] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def add_user_turn(
        self,
        text: str,
        via: str,
        original: str = "",
        language: str | None = None,
    ) -> None:
        if language and language != "en":
            self.detected_language = language
            self.language_confirmed = True
        self.history.append(Turn(role="user", text=text, via=via, original=original))

    def add_agent_turn(self, text: str) -> None:
        self.history.append(Turn(role="agent", text=text, via="text"))

    def plain_history(self) -> list[dict]:
        """Return history as plain dicts — safe to serialise to JSON."""
        return [
            {
                "role": t.role,
                "text": t.text,
                "via": t.via,
                **({"original": t.original} if t.original else {}),
            }
            for t in self.history
        ]


# ---------------------------------------------------------------------------
# Simple in-process session store.
# For multi-worker deployments replace with Redis or a DB-backed store.
# ---------------------------------------------------------------------------
_sessions: dict[str, VoiceSession] = {}


def get_or_create(session_id: str | None = None) -> VoiceSession:
    """Return an existing session or create a new one."""
    if session_id and session_id in _sessions:
        return _sessions[session_id]
    session = VoiceSession(session_id=session_id or uuid4().hex)
    _sessions[session.session_id] = session
    return session


def get(session_id: str) -> VoiceSession | None:
    return _sessions.get(session_id)
