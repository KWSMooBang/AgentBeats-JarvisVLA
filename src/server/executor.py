"""
A2A Executor

Bridges the A2A request/response cycle with the ScriptedPolicyAgent.

Message flow:
  init  → reset agent, return ack
  obs   → agent.act(), return action
"""

from __future__ import annotations

import base64
import io
import json
import logging
import time
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from src.agent.agent import ScriptedPolicyAgent, AgentState
from src.action.converter import noop_agent_action
from src.protocol.models import (
    InitPayload, ObservationPayload, ActionPayload, AckPayload,
)
from src.server.session_manager import SessionManager

logger = logging.getLogger(__name__)


def _decode_image(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(img)


class PurpleExecutor:
    """
    A2A-compatible executor for the Scripted Policy Agent.

    Designed for ``a2a`` library integration but also works standalone
    via the ``handle_message()`` method.
    """

    def __init__(
        self,
        sessions: SessionManager,
        planner_cfg: Optional[dict] = None,
        vlm_cfg: Optional[dict] = None,
        vla_cfg: Optional[dict] = None,
        device: str = "cuda",
    ):
        self.sessions = sessions
        self.agent = ScriptedPolicyAgent(
            planner_cfg=planner_cfg or {},
            vlm_cfg=vlm_cfg or {},
            vla_cfg=vla_cfg or {},
            device=device,
        )
        self.agent_states: Dict[str, AgentState] = {}
        self._touched: Dict[str, float] = {}
        logger.info("PurpleExecutor initialized")

    # ------------------------------------------------------------------
    # Unified entry-point
    # ------------------------------------------------------------------

    def handle_message(self, text: str, context_id: str = "default") -> str:
        """
        Process a JSON message string and return a JSON response string.
        Works both inside an A2A server and standalone.
        """
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return AckPayload(success=False, message="Invalid JSON").model_dump_json()

        msg_type = payload.get("type")
        if msg_type == "init":
            return self._handle_init(payload, context_id)
        elif msg_type == "obs":
            return self._handle_obs(payload, context_id)
        else:
            return AckPayload(
                success=False, message=f"Unknown type: {msg_type}"
            ).model_dump_json()

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _handle_init(self, payload: dict, context_id: str) -> str:
        try:
            init = InitPayload(**payload)
        except Exception as e:
            return AckPayload(success=False, message=str(e)).model_dump_json()

        task_text = init.text
        logger.info("INIT context=%s task=%s", context_id, task_text)

        self.agent.reset(task_text=task_text)
        state = self.agent.initial_state(task_text=task_text)
        self.agent_states[context_id] = state
        self._touched[context_id] = time.time()

        return AckPayload(success=True, message="Initialized").model_dump_json()

    # ------------------------------------------------------------------
    # Observation → Action
    # ------------------------------------------------------------------

    def _handle_obs(self, payload: dict, context_id: str) -> str:
        try:
            obs_payload = ObservationPayload(**payload)
        except Exception as e:
            return self._noop_action_json()

        state = self.agent_states.get(context_id)
        if state is None:
            logger.warning("No state for context=%s, returning noop", context_id)
            return self._noop_action_json()

        try:
            image = _decode_image(obs_payload.obs)
        except Exception as e:
            logger.error("Image decode failed: %s", e)
            return self._noop_action_json()

        action, new_state = self.agent.act(obs={"image": image}, state=state)
        self.agent_states[context_id] = new_state
        self._touched[context_id] = time.time()

        if action is None or "buttons" not in action or "camera" not in action:
            return self._noop_action_json()

        return ActionPayload(
            action_type="agent",
            buttons=action["buttons"],
            camera=action["camera"],
        ).model_dump_json()

    @staticmethod
    def _noop_action_json() -> str:
        noop = noop_agent_action()
        return ActionPayload(
            action_type="agent",
            buttons=noop["buttons"],
            camera=noop["camera"],
        ).model_dump_json()

    # ------------------------------------------------------------------
    # A2A AgentExecutor interface (optional — for a2a library)
    # ------------------------------------------------------------------

    async def execute(self, context, event_queue=None):
        """
        Async entry-point for the ``a2a`` library's DefaultRequestHandler.
        """
        from a2a.server.tasks import TaskUpdater
        from a2a.types import TextPart
        from a2a.utils import new_agent_text_message

        msg = getattr(context, "message", None)
        context_id = getattr(msg, "context_id", None) or "default"
        task_id = getattr(msg, "task_id", None) or context_id

        task_updater = TaskUpdater(
            event_queue=event_queue,
            task_id=task_id,
            context_id=context_id,
        )

        text = self._extract_text(msg)
        if not text:
            resp_text = AckPayload(success=False, message="Empty message").model_dump_json()
        else:
            resp_text = self.handle_message(text, context_id)

        response = new_agent_text_message(resp_text)
        await task_updater.complete(message=response)
        return response

    async def cancel(self, context, event_queue=None):
        ctx_id = getattr(context, "context_id", "default")
        self.agent_states.pop(ctx_id, None)
        self._touched.pop(ctx_id, None)

    @staticmethod
    def _extract_text(msg) -> Optional[str]:
        if msg is None:
            return None
        parts = getattr(msg, "parts", None)
        if not isinstance(parts, list):
            return None
        for part in parts:
            root = getattr(part, "root", None)
            if root and hasattr(root, "text"):
                return root.text
            if hasattr(part, "text"):
                return part.text
            if isinstance(part, dict) and "text" in part:
                return part["text"]
        return None
