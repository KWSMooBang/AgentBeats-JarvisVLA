"""
Session Manager

Tracks per-context agent state for concurrent A2A sessions.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class SessionManager:

    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.contexts: Dict[str, Dict[str, Any]] = {}

    def create_session(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "session_id": session_id,
                "created_at": time.time(),
                "contexts": set(),
            }
        return self.sessions[session_id]

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.sessions.get(session_id)

    def delete_session(self, session_id: str) -> None:
        if session_id in self.sessions:
            for ctx in list(self.sessions[session_id].get("contexts", set())):
                self.delete_context(ctx)
            del self.sessions[session_id]

    def create_context(self, context_id: str, session_id: str) -> Dict[str, Any]:
        if context_id not in self.contexts:
            self.contexts[context_id] = {
                "context_id": context_id,
                "session_id": session_id,
                "created_at": time.time(),
            }
            sess = self.sessions.get(session_id)
            if sess:
                sess["contexts"].add(context_id)
        return self.contexts[context_id]

    def get_context(self, context_id: str) -> Optional[Dict[str, Any]]:
        return self.contexts.get(context_id)

    def delete_context(self, context_id: str) -> None:
        ctx = self.contexts.pop(context_id, None)
        if ctx:
            sid = ctx.get("session_id")
            if sid and sid in self.sessions:
                self.sessions[sid]["contexts"].discard(context_id)

    def list_sessions(self) -> list[str]:
        return list(self.sessions.keys())
