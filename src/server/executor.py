# src/server/executor.py
from __future__ import annotations

import base64
import json
import logging
import time
import uuid
from typing import Any, Dict, Optional
from pydantic import BaseModel

import cv2
import numpy as np

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, Role, TextPart
from a2a.utils import new_agent_text_message

from src.server.session_manager import SessionManager
from src.protocol.models import InitPayload, ObservationPayload, AckPayload, ActionPayload
from src.agent import JarvisVLAAgent, AgentState
from src.action.action_space import noop_action
from src.action.converter import ActionConverter

logger = logging.getLogger(__name__)

def _noop_action_payload() -> Dict[str, Any]:
    """Return noop action payload."""
    return ActionPayload(   
        action_type="agent",
        buttons=noop_action()["buttons"],
        camera=noop_action()["camera"],
    ).model_dump()


class Executor(AgentExecutor):
    """
    JarvisVLA Purple Policy Executor (A2A-compatible).

    Contract:
      - Input: message/send with JSON in TextPart (init/obs)
      - Output: (A2A) emits exactly ONE terminal TaskStatusUpdateEvent via TaskUpdater.complete()
              and also returns Message for compatibility.
      - No streaming, no multi-event lifecycle.
      - Never returns None.
    """

    def __init__(
        self,
        sessions: SessionManager,
        *,
        base_url: str = "http://localhost:8000/v1",
        model_name: str = "CraftJarvis/JarvisVLA-Qwen2-VL-7B",
        state_ttl_seconds: Optional[int] = 60 * 60,
        debug: bool = False,
        device: Optional[str] = None,
    ) -> None:
        self.sessions = sessions
        self.base_url = base_url
        self.model_name = model_name
        self.deterministic = deterministic
        self._state_ttl_seconds = state_ttl_seconds
        self._debug = bool(debug)
        self._device = device

        # context_id -> agent/state/action
        self.agents: dict[str, Any] = {}
        self.agent_states: dict[str, AgentState] = {}
        self._agent_state_touched_at: dict[str, float] = {}
        self._last_actions: dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_text(self, msg: Optional[Message]) -> Optional[str]:
        """Extract text from A2A Message."""
        if msg is None:
            return None

        parts = getattr(msg, "parts", None)
        if not isinstance(parts, list):
            return None

        for part in parts:
            # Case 1) Part(root=TextPart(...))
            root = getattr(part, "root", None)
            if isinstance(root, TextPart) and isinstance(root.text, str):
                return root.text

            # Case 2) part itself is TextPart
            if isinstance(part, TextPart) and isinstance(part.text, str):
                return part.text

            # Case 3) dict-like
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    return text

            # Case 4) generic attribute 'text'
            text = getattr(part, "text", None)
            if isinstance(text, str):
                return text

        return None

    def _decode_obs(self, obs_base64: str) -> np.ndarray:
        """Decode base64 image to numpy array."""
        if obs_base64.startswith("data:"):
            obs_base64 = obs_base64.split("base64,", 1)[-1]

        img_bytes = base64.b64decode(obs_base64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode failed")

        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Invalid image shape: {img.shape}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _gc_agent_states(self) -> None:
        """Garbage collect old agent states."""
        if self._state_ttl_seconds is None:
            return
        now = time.time()
        dead = [
            cid
            for cid, ts in self._agent_state_touched_at.items()
            if (now - ts) > self._state_ttl_seconds
        ]
        for cid in dead:
            self._agent_state_touched_at.pop(cid, None)
            self.agent_states.pop(cid, None)
            self.agents.pop(cid, None)
            self._last_actions.pop(cid, None)
            logger.info("GC'd agent state: %s", cid)

    def _touch(self, context_id: str) -> None:
        """Update touch timestamp."""
        self._agent_state_touched_at[context_id] = time.time()

    def _get_or_create_agent(self, context_id: str) -> JarvisVLAAgent:
        """Get or create agent for context_id."""
        agent = self.agents.get(context_id)
        if agent is None:
            logger.info(
                "[Purple] Creating JarvisVLAAgent device=%s base_url=%s",
                self._device,
                self.base_url,
            )
            agent = JarvisVLAAgent(
                base_url=self.base_url,
                model_name=self.model_name,
                device=self._device,
                temperature=0.5,
                history_num=0,
                action_chunk_len=1,
                instruction_type="normal",
            )
            agent.reset()
            self.agents[context_id] = agent
        return agent

    async def _handle_init(
        self,
        payload: Dict[str, Any],
        context_id: str,
        task_id: str,
        task_updater: TaskUpdater
    ) -> Message:
        """Handle init message.
        
        Args:
            payload: Init payload dict
            context_id: Context identifier
            task_id: Task identifier
            task_updater: Task updater for emitting completion
            
        Returns:
            Ack message
        """
        init = InitPayload.model_validate(payload)
        logger.info("Init request: context=%s, task=%s", context_id, init.text[:100])

        # Start session
        self.sessions.start_new_task(
            context_id=context_id,
            task_text=init.text,
        )

        # Create agent and initial state
        agent = self._get_or_create_agent(context_id)
        state = agent.initial_state(init.text)

        self.agent_states[context_id] = state
        self._last_actions[context_id] = noop_action()
        self._touch(context_id)

        # Create ack response
        ack_payload = AckPayload(success=True, message="Initialization successful.")
        response = new_agent_text_message(ack_payload.model_dump_json())
        
        # Complete task
        await task_updater.complete(output=response)
        return response

    async def _handle_obs(
        self,
        payload: Dict[str, Any],
        context_id: str,
        task_id: str,
        task_updater: TaskUpdater
    ) -> Message:
        """Handle observation message.
        
        Args:
            payload: Observation payload dict
            context_id: Context identifier
            task_id: Task identifier
            task_updater: Task updater for emitting completion
            
        Returns:
            Action message
        """
        obs = ObservationPayload.model_validate(payload)
        
        # Decode observation
        image_rgb = self._decode_obs(obs.obs)
        obs_dict = {"image": image_rgb, "step": obs.step}

        # Update session
        self.sessions.on_observation(context_id, obs.step)

        # Get agent and state
        agent = self._get_or_create_agent(context_id)
        state = self.agent_states.get(context_id)

        if state is None:
            logger.warning("No state for context=%s, returning noop", context_id)
            noop_payload = _noop_action_payload()
            response = new_agent_text_message(json.dumps(noop_payload))
            await task_updater.complete(output=response)
            return response

        # Generate action
        action, new_state = agent.act(
            obs=obs_dict,
            state=state
        )

        # Update state
        self.agent_states[context_id] = new_state
        self._touch(context_id)
        self._last_actions[context_id] = action

        # Convert action format using ActionConverter
        purple_action = ActionConverter.jarvisvla_to_purple(action)

        # Create action response
        action_payload = ActionPayload(
            action_type="agent",
            buttons=purple_action["buttons"],
            camera=purple_action["camera"],
        )
        
        response = new_agent_text_message(action_payload.model_dump_json())
        
        # Complete task
        await task_updater.complete(output=response)
        return response



    # ------------------------------------------------------------------
    # Main entry (IMPORTANT)
    # ------------------------------------------------------------------

    async def execute(self, context: RequestContext, event_queue=None) -> Message:
        """
        Execute agent action based on request.
        
        Dispatches to _handle_init or _handle_obs based on message type.
        
        Args:
            context: Request context with message
            event_queue: Event queue for task updates
            
        Returns:
            Response message (ack for init, action for obs)
        """
        self._gc_agent_states()

        msg = getattr(context, "message", None)
        payload_text = self._extract_text(msg)

        context_id = (
            getattr(msg, "context_id", None) 
            or getattr(context, "context_id", None) 
            or "default"
        )
        task_id = (
            getattr(msg, "task_id", None) 
            or getattr(context, "task_id", None) 
            or context_id
        )
        
        # Create task updater
        task_updater = TaskUpdater(
            event_queue=event_queue,
            task_id=task_id,
            context_id=context_id,
        )

        try:
            if not payload_text:
                logger.warning("No payload text, returning noop")
                noop_payload = _noop_action_payload()
                response = new_agent_text_message(json.dumps(noop_payload))
                await task_updater.complete(output=response)
                return response

            payload = json.loads(payload_text)
            payload_type = payload.get("type")

            # Dispatch to appropriate handler
            if payload_type == "init":
                return await self._handle_init(payload, context_id, task_id, task_updater)
            elif payload_type == "obs":
                return await self._handle_obs(payload, context_id, task_id, task_updater)
            else:
                logger.warning("Unknown payload type: %s", payload_type)
                noop_payload = _noop_action_payload()
                response = new_agent_text_message(json.dumps(noop_payload))
                await task_updater.complete(output=response)
                return response

        except Exception:
            logger.exception("[EXEC] fatal error")
            noop_payload = _noop_action_payload()
            try:
                response = new_agent_text_message(json.dumps(noop_payload))
                await task_updater.complete(output=response)
                return response
            except Exception:
                return new_agent_text_message(json.dumps(noop_payload))

    async def cancel(self, context: RequestContext, event_queue=None) -> None:
        """Cancel handler (no-op for MCU)."""
        logger.warning("Cancel called (no-op)")
        return
