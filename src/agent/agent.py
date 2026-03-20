"""
Scripted Policy Agent

Core agent class that orchestrates:
  1. LLM Planner  → FSM JSON from task_text
  2. FSM Executor → step-by-step action generation from images
  3. Action Converter → env_action → Purple Agent agent_action

Interface contract (Purple Agent compatible):
  - reset(task_text)          called once per episode
  - initial_state(task_text)  returns AgentState
  - act(obs, state)           returns (action, new_state)
  - device property           returns torch.device

The agent receives ONLY task_text + observation image.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.planner.llm_planner import LLMPlanner
from src.planner.validator import PolicySpecValidator
from src.executor.fsm_executor import FSMExecutor
from src.executor.vlm_checker import VLMStateChecker
from src.action.converter import ActionConverter, noop_agent_action

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Per-episode session state carried between act() calls."""
    memory: Optional[Any] = None
    first: bool = False
    idle_count: int = 0
    task_text: Optional[str] = None
    policy_spec: Optional[dict] = field(default=None, repr=False)
    current_fsm_state: Optional[str] = None
    total_steps: int = 0


class ScriptedPolicyAgent:
    """
    Purple-Agent-compatible scripted policy agent.

    Parameters
    ----------
    planner_cfg : dict
        Kwargs for LLMPlanner (api_key, base_url, model, …).
    vlm_cfg : dict
        Kwargs for VLMStateChecker (api_key, base_url, model, …).
    action_format : str
        "agent" → compact Purple format, "env" → expanded env format.
    device : str
        "cuda" or "cpu".
    """

    def __init__(
        self,
        planner_cfg: Optional[dict] = None,
        vlm_cfg: Optional[dict] = None,
        action_format: str = "agent",
        device: str = "cuda",
        output_dir: str = "./outputs",
    ):
        planner_cfg = planner_cfg or {}
        vlm_cfg = vlm_cfg or {}

        self._device_str = device
        self.action_format = action_format
        self._output_dir = Path(output_dir)

        self.planner = LLMPlanner(**planner_cfg)
        self.vlm_checker = VLMStateChecker(**vlm_cfg)
        self.validator = PolicySpecValidator()
        self.converter = ActionConverter() if action_format == "agent" else None

        self._executor: Optional[FSMExecutor] = None
        self._policy_spec: Optional[dict] = None
        self._episode_dir: Optional[Path] = None
        self._episode_start: Optional[datetime] = None

        logger.info("ScriptedPolicyAgent initialized (action_format=%s)", action_format)

    @property
    def device(self):
        try:
            import torch
            return torch.device(self._device_str)
        except ImportError:
            return self._device_str

    # ------------------------------------------------------------------
    # Purple Agent interface
    # ------------------------------------------------------------------

    def reset(
        self,
        task_text: Optional[str] = None,
        episode_dir: Optional[str] = None,
        **_kwargs,
    ) -> None:
        """
        Reset the agent for a new episode.

        Args:
            task_text: Natural-language task description.
            episode_dir: If given, policy_spec.json and result.json are saved
                         here (same folder as MineStudio recordings).
                         If None, a timestamped sub-folder under output_dir
                         is created automatically.
        """
        self._save_episode_result()

        self._executor = None
        self._policy_spec = None
        self._episode_dir = None
        self._episode_start = None

        if not task_text:
            logger.warning("reset() called without task_text")
            return

        logger.info("Planning for task: %s", task_text)
        self._policy_spec = self.planner.generate_policy_spec(task_text)

        errors = self.validator.validate(self._policy_spec)
        real_errors = [e for e in errors if not e.startswith("Warning")]
        if real_errors:
            logger.warning("Spec has errors, attempting replan: %s", real_errors)
            self._policy_spec = self.planner.generate_policy_spec(
                task_text, feedback=real_errors
            )

        self._executor = FSMExecutor(
            policy_spec=self._policy_spec,
            vlm_checker=self.vlm_checker,
        )
        logger.info("FSM executor ready with %d states", len(self._policy_spec.get("states", {})))

        self._episode_start = datetime.now()
        if episode_dir is not None:
            self._episode_dir = Path(episode_dir)
            self._episode_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._episode_dir = self._make_episode_dir(task_text)
        self._save_policy_spec()

    def initial_state(self, task_text: Optional[str] = None) -> AgentState:
        return AgentState(
            memory=None,
            first=True,
            idle_count=0,
            task_text=task_text,
            policy_spec=self._policy_spec,
        )

    def act(
        self,
        obs: Dict[str, Any],
        state: AgentState,
        deterministic: bool = False,
    ) -> Tuple[Optional[Dict[str, Any]], AgentState]:
        """
        Generate the next action from the current observation.

        Args:
            obs: {"image": np.ndarray[H, W, 3]}
            state: current AgentState
            deterministic: ignored (always deterministic)

        Returns:
            (action_dict, new_state)
            action_dict is in agent format (buttons/camera indices)
            or env format depending on self.action_format.
        """
        try:
            image = obs.get("image")
            if image is None:
                logger.warning("No 'image' key in obs")
                state.first = False
                return noop_agent_action(), state

            if self._executor is None:
                logger.warning("Executor not initialised — call reset() first")
                state.first = False
                return noop_agent_action(), state

            env_action = self._executor.step(image)

            if env_action is None:
                self._save_episode_result()
                state.first = False
                return noop_agent_action(), state

            if self.action_format == "agent" and self.converter is not None:
                action = self.converter.env_to_agent(env_action)
            else:
                action = env_action

            new_state = AgentState(
                memory=state.memory,
                first=False,
                idle_count=state.idle_count,
                task_text=state.task_text,
                policy_spec=state.policy_spec,
                current_fsm_state=self._executor.current_state,
                total_steps=self._executor.total_step_count,
            )
            return action, new_state

        except Exception as e:
            logger.exception("act() failed: %s", e)
            state.first = False
            return noop_agent_action(), state

    # ------------------------------------------------------------------
    # Output saving
    # ------------------------------------------------------------------

    def _make_episode_dir(self, task_text: str) -> Path:
        safe_task = re.sub(r"[^a-zA-Z0-9_\-]", "_", task_text)[:60].strip("_")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ep_dir = self._output_dir / f"{safe_task}_{ts}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        return ep_dir

    def _save_policy_spec(self) -> None:
        if self._episode_dir is None or self._policy_spec is None:
            return
        path = self._episode_dir / "policy_spec.json"
        path.write_text(json.dumps(self._policy_spec, indent=2, ensure_ascii=False))
        logger.info("Policy spec saved → %s", path)

    def _save_episode_result(self) -> None:
        if self._episode_dir is None or self._executor is None:
            return
        elapsed = None
        if self._episode_start is not None:
            elapsed = (datetime.now() - self._episode_start).total_seconds()
        result = {
            "task": self._policy_spec.get("task") if self._policy_spec else None,
            "finished": self._executor.finished,
            "result": self._executor.result,
            "total_steps": self._executor.total_step_count,
            "final_state": self._executor.current_state,
            "vlm_calls": self.vlm_checker.call_count,
            "elapsed_seconds": elapsed,
            "timestamp": datetime.now().isoformat(),
        }
        path = self._episode_dir / "result.json"
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        logger.info("Episode result saved → %s", path)
