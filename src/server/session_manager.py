# src/server/session_manager.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional
import time
import threading
import logging

logger = logging.getLogger(__name__)


@dataclass
class SessionState:
    """Single session state for a context_id."""
    # identification
    context_id: str

    # timestamps
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # task info
    task_text: Optional[str] = None
    task_started_at: Optional[float] = None

    # observation tracking
    last_step: int = -1
    last_step_received: int = -1
    num_obs: int = 0
    num_step_regressions: int = 0

    # expected action shapes
    expected_num_buttons: int = 20
    expected_camera_dims: int = 2

    def touch(self) -> None:
        """Update timestamp."""
        self.updated_at = time.time()


class SessionManager:
    """Manage multiple SessionState instances by context_id."""

    def __init__(self, *, ttl_seconds: Optional[int] = 60 * 60) -> None:
        self._sessions: Dict[str, SessionState] = {}
        self._ttl_seconds = ttl_seconds
        self._lock = threading.RLock()

    def get_or_create(self, context_id: str) -> SessionState:
        """Get or create session by context_id."""
        self._gc_if_needed()

        with self._lock:
            s = self._sessions.get(context_id)
            if s is None:
                s = SessionState(context_id=context_id)
                self._sessions[context_id] = s
            s.touch()
            return s

    def start_new_task(
        self,
        context_id: str,
        task_text: str,
        *,
        expected_num_buttons: Optional[int] = None,
        expected_camera_dims: Optional[int] = None,
    ) -> SessionState:
        """Initialize a new task session."""
        with self._lock:
            s = self.get_or_create(context_id)

            # reset task-related fields
            s.task_text = task_text
            s.task_started_at = time.time()

            s.last_step = -1
            s.last_step_received = -1
            s.num_obs = 0
            s.num_step_regressions = 0

            # allow overrides
            if expected_num_buttons is not None:
                s.expected_num_buttons = int(expected_num_buttons)
            if expected_camera_dims is not None:
                s.expected_camera_dims = int(expected_camera_dims)

            s.touch()
            logger.info("Started task: context=%s, task=%s", context_id, task_text[:100])
            return s

    def on_observation(self, context_id: str, step: int) -> SessionState:
        """Observation received for the given context_id at step."""
        self._validate_context_id(context_id)

        with self._lock:
            try:
                step_i = int(step)
            except (ValueError, TypeError):
                step_i = -1

            s = self.get_or_create(context_id)

            # Check for step regressions
            if step_i < s.last_step_received:
                s.num_step_regressions += 1
                logger.warning(
                    "Step regression: context=%s, prev=%d, current=%d, count=%d",
                    context_id, s.last_step_received, step_i, s.num_step_regressions
                )

            s.last_step_received = step_i
            if step_i > s.last_step:
                s.last_step = step_i

            s.num_obs += 1
            s.touch()

            return s

    def _validate_context_id(self, context_id: str) -> None:
        """Validate context_id exists."""
        if context_id not in self._sessions:
            logger.warning("Unknown context_id: %s, creating new session", context_id)

    def _gc_if_needed(self) -> None:
        """Garbage collect old sessions."""
        if self._ttl_seconds is None:
            return

        now = time.time()
        with self._lock:
            dead = [
                cid
                for cid, sess in self._sessions.items()
                if (now - sess.updated_at) > self._ttl_seconds
            ]
            for cid in dead:
                self._sessions.pop(cid, None)
                logger.info("GC'd session: %s", cid)
