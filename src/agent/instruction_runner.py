"""
JarvisVLA instruction runner

Bridges a state-level instruction string to one-step agent action generation
using the JarvisVLA VLLM agent wrapper.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from src.action.converter import noop_agent_action

logger = logging.getLogger(__name__)


class JarvisVLAInstructionRunner:
    """Executes one JarvisVLA instruction step and returns an agent action."""

    def __init__(
        self,
        checkpoint_path: str,
        base_url: str,
        api_key: str = "EMPTY",
        history_num: int = 4,
        action_chunk_len: int = 1,
        bpe: int = 0,
        instruction_type: str = "normal",
        temperature: float = 0.7,
        convert_camera_21_to_11: bool = True,
    ):
        from jarvisvla.evaluate.agent_wrapper import VLLM_AGENT

        self.agent = VLLM_AGENT(
            checkpoint_path=checkpoint_path,
            base_url=base_url,
            api_key=api_key,
            history_num=history_num,
            action_chunk_len=action_chunk_len,
            bpe=bpe,
            instruction_type=instruction_type,
            temperature=temperature,
        )
        self.default_instruction_type = instruction_type
        self.convert_camera_21_to_11 = convert_camera_21_to_11

    def reset(self):
        self.agent.reset()

    def run(
        self,
        image: np.ndarray,
        instruction: str,
        instruction_type: str = "auto",
        state_def: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        resolved_type = self._resolve_instruction_type(
            instruction=instruction,
            requested_type=instruction_type,
        )
        need_crafting_table = resolved_type == "recipe"

        prev_type = self.agent.instruction_type
        try:
            self.agent.instruction_type = resolved_type
            raw_action = self.agent.forward(
                observations=[image],
                instructions=[instruction],
                verbos=False,
                need_crafting_table=need_crafting_table,
            )
        except Exception as e:
            logger.exception("JarvisVLA instruction execution failed: %s", e)
            return {
                "__action_format__": "agent",
                "action": noop_agent_action(),
            }
        finally:
            self.agent.instruction_type = prev_type

        normalized = self._normalize_agent_action(raw_action)
        return {
            "__action_format__": "agent",
            "action": normalized,
        }

    def _resolve_instruction_type(self, instruction: str, requested_type: str) -> str:
        if requested_type in {"simple", "normal", "recipe"}:
            return requested_type
        if instruction.startswith("craft_item:"):
            return "recipe"
        if instruction in self.agent.prompt_library:
            return "normal"
        return self.default_instruction_type

    def _normalize_agent_action(self, action: Any) -> dict[str, list[int]]:
        if not isinstance(action, dict):
            return noop_agent_action()

        raw_buttons = self._to_scalar(action.get("buttons", 0))
        raw_camera = self._to_scalar(action.get("camera", 60))

        buttons = np.array([int(raw_buttons)])
        camera_value = int(raw_camera)
        if self.convert_camera_21_to_11:
            camera_value = self._convert_camera_21_to_11(camera_value)
        camera = np.array([camera_value])

        return {
            "buttons": buttons,
            "camera": camera,
        }

    @staticmethod
    def _to_scalar(value: Any) -> int:
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return 0
            return int(value.reshape(-1)[0])
        if isinstance(value, (list, tuple)):
            if not value:
                return 0
            return int(value[0])
        return int(value)

    @staticmethod
    def _convert_camera_21_to_11(camera_flat: int) -> int:
        model_bins = 21
        model_binsize = 1
        model_mu = 20.0
        env_bins = 11
        env_binsize = 2
        env_mu = 10.0
        maxval = 10

        pitch_21 = camera_flat // model_bins
        yaw_21 = camera_flat % model_bins

        centered = np.array([pitch_21, yaw_21], dtype=np.float64) * model_binsize - maxval

        def mu_decode(xy: np.ndarray, mu: float, max_value: int) -> np.ndarray:
            xy_norm = xy / max_value
            return (
                np.sign(xy_norm)
                * (1.0 / mu)
                * ((1.0 + mu) ** np.abs(xy_norm) - 1.0)
                * max_value
            )

        def mu_encode(xy: np.ndarray, mu: float, max_value: int) -> np.ndarray:
            xy = np.clip(xy, -max_value, max_value).astype(np.float64)
            xy_norm = xy / max_value
            return (
                np.sign(xy_norm)
                * (np.log(1.0 + mu * np.abs(xy_norm)) / np.log(1.0 + mu))
                * max_value
            )

        continuous = mu_decode(centered, model_mu, maxval)
        encoded = mu_encode(continuous, env_mu, maxval)
        env_bins_xy = np.clip(
            np.round((encoded + maxval) / env_binsize).astype(np.int64),
            0,
            env_bins - 1,
        )

        return int(env_bins_xy[0]) * env_bins + int(env_bins_xy[1])
