"""
Scripted Policy Agent

Core agent class that orchestrates:
  1. LLM Planner  → FSM JSON from task_text
  2. FSM Executor → step-by-step action generation from images
    3. JarvisVLA Runner → state instruction to agent action

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
from src.planner.validator import PlanValidator
from src.executor.fsm_executor import FSMExecutor
from src.action.converter import noop_agent_action

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Per-episode session state carried between act() calls."""
    memory: Optional[Any] = None
    first: bool = False
    idle_count: int = 0
    task_text: Optional[str] = None
    plan: Optional[dict] = field(default=None, repr=False)
    current_fsm_state: Optional[str] = None
    total_steps: int = 0
    execution_mode: str = "idle"
    direct_instruction: Optional[str] = None
    direct_instruction_type: str = "normal"


class MinecraftPurpleAgent:
    """
    Purple-Agent-compatible scripted policy agent.

    Parameters
    ----------
    planner_cfg : dict
        Kwargs for LLMPlanner (api_key, base_url, model, …).
    vla_cfg : dict
        JarvisVLA runner config. State-level ``instruction`` fields
        are executed by the JarvisVLA model.
    device : str
        "cuda" or "cpu".
    """

    def __init__(
        self,
        planner_cfg: Optional[dict] = None,
        vla_cfg: Optional[dict] = None,
        device: str = "cuda",
        output_dir: str = "./outputs",
    ):
        planner_cfg = planner_cfg or {}
        vla_cfg = vla_cfg or {}

        self._device_str = device
        self._output_dir = Path(output_dir)

        self.planner = LLMPlanner(**planner_cfg)
        self.validator = PlanValidator()
        self.instruction_runner = self._build_instruction_runner(vla_cfg)

        self._executor: Optional[FSMExecutor] = None
        self._plan: Optional[dict] = None
        self._episode_dir: Optional[Path] = None
        self._episode_start: Optional[datetime] = None
        self._execution_mode: str = "idle"
        self._short_instruction: Optional[str] = None
        self._short_instruction_type: str = "normal"
        self._direct_step_count: int = 0
        self._task_text: Optional[str] = None

        logger.info("MinecraftPurpleAgent initialized")

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
            episode_dir: If given, plan.json and result.json are saved
                         here (same folder as MineStudio recordings).
                         If None, a timestamped sub-folder under output_dir
                         is created automatically.
        """
        self._save_episode_result()

        self._executor = None
        self._plan = None
        self._episode_dir = None
        self._episode_start = None
        self._execution_mode = "idle"
        self._short_instruction = None
        self._short_instruction_type = "normal"
        self._direct_step_count = 0
        self._task_text = task_text

        self.instruction_runner.reset()

        if not task_text:
            logger.warning("reset() called without task_text")
            return

        logger.info("Planning route for task: %s", task_text)
        routed = self.planner.plan_task(task_text)
        horizon = routed.get("horizon", "long")

        if horizon == "short":
            self._execution_mode = "short"
            self._short_instruction = routed.get("instruction", "")
            self._short_instruction_type = routed.get("instruction_type", "normal")
            self._plan = {
                "instruction": self._short_instruction,
                "instruction_type": self._short_instruction_type,
            }
            logger.info(
                "Short-horizon direct mode enabled: instruction=%s type=%s",
                self._short_instruction,
                self._short_instruction_type,
            )
        else:
            self._execution_mode = "long"
            self._plan = routed.get("plan", {})

            errors = self.validator.validate(self._plan)
            real_errors = [e for e in errors if not e.startswith("Warning")]
            if real_errors:
                logger.warning("Spec has errors, attempting replan: %s", real_errors)
                routed = self.planner.plan_task(
                    task_text,
                    feedback=real_errors,
                    force_horizon="long",
                )
                self._plan = routed.get("plan", self._plan)

            self._executor = FSMExecutor(
                plan=self._plan,
                instruction_runner=self._run_state_instruction,
            )

            state_count = sum(
                1 for _, value in (self._plan or {}).items() if isinstance(value, dict)
            )
            logger.info("Long-horizon FSM executor ready with %d steps", state_count)

        self._episode_start = datetime.now()
        if episode_dir is not None:
            self._episode_dir = Path(episode_dir)
            self._episode_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._episode_dir = self._make_episode_dir(task_text)
        self._save_plan()

    def initial_state(self, task_text: Optional[str] = None) -> AgentState:
        return AgentState(
            memory=None,
            first=True,
            idle_count=0,
            task_text=task_text,
            plan=self._plan,
            execution_mode=self._execution_mode,
            direct_instruction=self._short_instruction,
            direct_instruction_type=self._short_instruction_type,
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
            action_dict is always in agent format (buttons/camera indices).
        """
        try:
            image = obs.get("image")
            if image is None:
                logger.warning("No 'image' key in obs")
                state.first = False
                return noop_agent_action(), state

            if self._execution_mode == "short":
                if not self._short_instruction:
                    logger.warning("Short mode missing instruction; returning noop")
                    state.first = False
                    return noop_agent_action(), state

                action_packet = self._run_state_instruction(
                    image=image,
                    instruction=self._short_instruction,
                    instruction_type=self._short_instruction_type,
                    state_def={"description": "short_direct"},
                )
                self._direct_step_count += 1

            else:
                if self._executor is None:
                    logger.warning("Executor not initialised — call reset() first")
                    state.first = False
                    return noop_agent_action(), state

                action_packet = self._executor.step(image)

                if action_packet is None:
                    self._save_episode_result()
                    state.first = False
                    return noop_agent_action(), state

            if isinstance(action_packet, dict) and action_packet.get("__action_format__") == "agent":
                action = action_packet.get("action", noop_agent_action())
            else:
                logger.warning("Unexpected non-agent action packet; falling back to noop")
                action = noop_agent_action()

            new_state = AgentState(
                memory=state.memory,
                first=False,
                idle_count=state.idle_count,
                task_text=state.task_text,
                plan=state.plan,
                current_fsm_state=(
                    self._executor.current_state if self._execution_mode == "long" and self._executor else "short_direct"
                ),
                total_steps=(
                    self._executor.total_step_count if self._execution_mode == "long" and self._executor else self._direct_step_count
                ),
                execution_mode=self._execution_mode,
                direct_instruction=self._short_instruction,
                direct_instruction_type=self._short_instruction_type,
            )

            logger.info("Step %d: mode='%s' state='%s' action=%s",
                new_state.total_steps,
                new_state.execution_mode,
                new_state.current_fsm_state,
                action,
            )
            return action, new_state

        except Exception as e:
            logger.exception("act() failed: %s", e)
            state.first = False
            return noop_agent_action(), state

    # ------------------------------------------------------------------
    # Instruction runner bridge
    # ------------------------------------------------------------------

    def _build_instruction_runner(self, vla_cfg: dict):
        if not vla_cfg:
            raise RuntimeError("VLA configuration is required")
        if not vla_cfg.get("enabled", False):
            raise RuntimeError("VLA must be enabled")
        if not vla_cfg.get("checkpoint_path"):
            raise RuntimeError("vla_cfg.checkpoint_path is required")

        try:
            from src.agent.instruction_runner import JarvisVLAInstructionRunner

            runner = JarvisVLAInstructionRunner(
                checkpoint_path=vla_cfg["checkpoint_path"],
                base_url=vla_cfg["base_url"],
                api_key=vla_cfg.get("api_key", "EMPTY"),
                history_num=vla_cfg.get("history_num", 4),
                action_chunk_len=vla_cfg.get("action_chunk_len", 1),
                bpe=vla_cfg.get("bpe", 0),
                instruction_type=vla_cfg.get("instruction_type", "normal"),
                temperature=vla_cfg.get("temperature", 0.7),
                convert_camera_21_to_11=vla_cfg.get("convert_camera_21_to_11", True),
            )
            logger.info("JarvisVLA instruction runner enabled")
            return runner
        except Exception as e:
            logger.exception("Failed to initialize JarvisVLA instruction runner: %s", e)
            raise RuntimeError("Failed to initialize JarvisVLA instruction runner") from e

    def _run_state_instruction(
        self,
        image: np.ndarray,
        instruction: str,
        instruction_type: str,
        state_def: dict,
    ) -> Optional[dict]:
        return self.instruction_runner.run(
            image=image,
            instruction=instruction,
            instruction_type=instruction_type,
            state_def=state_def,
        )

    # ------------------------------------------------------------------
    # Output saving
    # ------------------------------------------------------------------

    def _make_episode_dir(self, task_text: str) -> Path:
        safe_task = re.sub(r"[^a-zA-Z0-9_\-]", "_", task_text)[:60].strip("_")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ep_dir = self._output_dir / f"{safe_task}_{ts}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        return ep_dir

    def _save_plan(self) -> None:
        if self._episode_dir is None or self._plan is None:
            return
        path = self._episode_dir / "plan.json"
        path.write_text(json.dumps(self._plan, indent=2, ensure_ascii=False))
        logger.info("Plan saved → %s", path)

    def _save_episode_result(self) -> None:
        if self._episode_dir is None:
            return

        elapsed = None
        if self._episode_start is not None:
            elapsed = (datetime.now() - self._episode_start).total_seconds()

        if self._execution_mode == "long" and self._executor is not None:
            result = {
                "task": self._plan.get("task") if self._plan else None,
                "execution_mode": "long",
                "finished": self._executor.finished,
                "result": self._executor.result,
                "total_steps": self._executor.total_step_count,
                "final_state": self._executor.current_state,
                "elapsed_seconds": elapsed,
                "timestamp": datetime.now().isoformat(),
            }
        elif self._execution_mode == "short":
            result = {
                "task": self._task_text,
                "execution_mode": "short",
                "finished": False,
                "result": "running_short_direct",
                "total_steps": self._direct_step_count,
                "final_state": "short_direct",
                "direct_instruction": self._short_instruction,
                "direct_instruction_type": self._short_instruction_type,
                "elapsed_seconds": elapsed,
                "timestamp": datetime.now().isoformat(),
            }
        else:
            return

        path = self._episode_dir / "result.json"
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        logger.info("Episode result saved → %s", path)
