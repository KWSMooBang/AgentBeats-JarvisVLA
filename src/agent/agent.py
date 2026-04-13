"""Minecraft purple-agent: planner → FSM executor → JarvisVLA / scripted fallback."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.action.converter import ActionConverter, noop_agent_action
from src.agent.fallback_policy import FallbackPolicyEngine
from src.agent.sequence_router import SequenceRouter
from src.executor.fsm_executor import FSMExecutor
from src.planner.instruction_registry import canonicalize_strict_instruction_key
from src.planner.planner import Planner
from src.planner.validator import PlanValidator

logger = logging.getLogger(__name__)


# Maps instruction prefix to (execution_hint, sequence_name).
# Empty sequence_name means SequenceRouter picks from catalog.
_PREFIX_SEQUENCE_MAP: dict[str, tuple[str, str]] = {
    "drop":        ("scripted", "drop_cycle"),
    "use_item":    ("hybrid",   ""),
    "kill_entity": ("vla",      ""),
    "mine_block":  ("vla",      ""),
    "craft_item":  ("hybrid",   "open_inventory_craft"),
    "pickup":      ("vla",      ""),
}


def _build_short_state_def(instruction: str, task_text: str) -> dict:
    """Build a state_def hint from a short-horizon instruction prefix."""
    prefix = instruction.split(":")[0].lower().strip() if ":" in instruction else ""
    hint, seq = _PREFIX_SEQUENCE_MAP.get(prefix, ("vla", ""))
    state_def: dict = {
        "description": f"short_direct:{instruction}",
        "execution_hint": hint,
        "task_text": task_text,
    }
    if seq:
        state_def["sequence_name"] = seq
    return state_def


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
    startup_noop_remaining: int = 0
    post_startup_assessed: bool = False


class MinecraftPurpleAgent:
    """Purple-Agent-compatible Minecraft agent."""

    def __init__(
        self,
        planner_cfg: Optional[dict] = None,
        vla_cfg: Optional[dict] = None,
        vlm_cfg: Optional[dict] = None,
        device: str = "cuda",
        output_dir: str = "./outputs",
    ):
        planner_cfg = planner_cfg or {}
        vla_cfg = vla_cfg or {}
        vlm_cfg = vlm_cfg or {}

        self._device_str = device
        self._output_dir = Path(output_dir)

        self.planner = Planner(**planner_cfg)
        self.validator = PlanValidator()
        self.vla_runner = self._build_vla_runner(vla_cfg)
        self._vqa_interval_steps = int(vlm_cfg.get("vqa_interval_steps", 600))

        self._action_converter = ActionConverter()
        self._sequence_selector = self._build_sequence_selector()
        self._fallback_policy = FallbackPolicyEngine(
            action_converter=self._action_converter,
            vla_runner=self.vla_runner,
            sequence_selector=self._sequence_selector,
        )

        self._executor: Optional[FSMExecutor] = None
        self._plan: Optional[dict] = None
        self._episode_dir: Optional[Path] = None
        self._episode_start: Optional[datetime] = None
        self._execution_mode: str = "idle"
        self._short_instruction: Optional[str] = None
        self._short_instruction_type: str = "normal"
        self._short_state_def: dict = {}
        self._direct_step_count: int = 0
        self._task_text: Optional[str] = None
        self._startup_noop_remaining: int = 0
        self._post_startup_assessed: bool = False

        logger.info("MinecraftPurpleAgent initialized")

    @property
    def device(self):
        try:
            import torch

            return torch.device(self._device_str)
        except ImportError:
            return self._device_str

    def reset(
        self,
        task_text: Optional[str] = None,
        episode_dir: Optional[str] = None,
        **_kwargs,
    ) -> None:
        self._save_episode_result()

        self._executor = None
        self._plan = None
        self._episode_dir = Path(episode_dir) if episode_dir else None
        self._episode_start = None
        self._execution_mode = "idle"
        self._short_instruction = None
        self._short_instruction_type = "normal"
        self._short_state_def = {}
        self._direct_step_count = 0
        self._task_text = task_text
        self._fallback_policy.reset_episode()

        self.vla_runner.reset()

        if task_text:
            self._startup_noop_remaining = 5
            self._post_startup_assessed = False
        else:
            logger.warning("reset() called without task_text")
            self._startup_noop_remaining = 0

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
            startup_noop_remaining=self._startup_noop_remaining,
            post_startup_assessed=self._post_startup_assessed,
        )

    def act(
        self,
        obs: Dict[str, Any],
        state: AgentState,
        deterministic: bool = False,
    ) -> Tuple[Optional[Dict[str, Any]], AgentState]:
        try:
            image = obs.get("image")
            if image is None:
                logger.warning("No 'image' key in obs")
                state.first = False
                return noop_agent_action(), state

            if self._startup_noop_remaining > 0:
                self._startup_noop_remaining -= 1
                logger.info("Startup noop step remaining=%d", self._startup_noop_remaining)
                return noop_agent_action(), AgentState(
                    memory=state.memory,
                    first=False,
                    idle_count=state.idle_count,
                    task_text=state.task_text,
                    plan=state.plan,
                    current_fsm_state=None,
                    total_steps=0,
                    execution_mode="idle",
                    direct_instruction=None,
                    direct_instruction_type="normal",
                    startup_noop_remaining=self._startup_noop_remaining,
                    post_startup_assessed=self._post_startup_assessed,
                )

            if not self._post_startup_assessed:
                try:
                    self._post_startup_assess(image)
                except Exception:
                    logger.exception("Post-startup assessment failed")
                self._post_startup_assessed = True

            if self._execution_mode == "short":
                if not self._short_instruction:
                    logger.warning("Short mode missing instruction; returning noop")
                    state.first = False
                    return noop_agent_action(), state

                action_packet = self._run_mixed_instruction(
                    image=image,
                    instruction=self._short_instruction,
                    instruction_type=self._short_instruction_type,
                    state_def=self._short_state_def or {"description": "short_direct"},
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

            action = noop_agent_action()
            if isinstance(action_packet, dict) and action_packet.get("__action_format__") == "agent":
                action = action_packet.get("action", noop_agent_action())
            else:
                logger.warning("Unexpected non-agent action packet; falling back to noop")

            new_state = AgentState(
                memory=state.memory,
                first=False,
                idle_count=state.idle_count,
                task_text=state.task_text,
                plan=state.plan,
                current_fsm_state=(
                    self._executor.current_state
                    if self._execution_mode == "long" and self._executor
                    else "short_direct"
                ),
                total_steps=(
                    self._executor.total_step_count
                    if self._execution_mode == "long" and self._executor
                    else self._direct_step_count
                ),
                execution_mode=self._execution_mode,
                direct_instruction=self._short_instruction,
                direct_instruction_type=self._short_instruction_type,
                startup_noop_remaining=self._startup_noop_remaining,
                post_startup_assessed=self._post_startup_assessed,
            )

            logger.info(
                "Step %d: mode='%s' state='%s' action=%s",
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

    def _build_vla_runner(self, vla_cfg: dict):
        if not vla_cfg:
            raise RuntimeError("VLA configuration is required")
        if not vla_cfg.get("checkpoint_path"):
            raise RuntimeError("vla_cfg.checkpoint_path is required")
        try:
            from src.agent.vla_runner import VLARunner

            runner = VLARunner(
                checkpoint_path=vla_cfg["checkpoint_path"],
                base_url=vla_cfg["base_url"],
                api_key=vla_cfg.get("api_key", "EMPTY"),
                history_num=vla_cfg.get("history_num", 4),
                action_chunk_len=vla_cfg.get("action_chunk_len", 1),
                bpe=vla_cfg.get("bpe", 0),
                instruction_type=vla_cfg.get("instruction_type", "auto"),
                temperature=vla_cfg.get("temperature", 0.7),
                convert_camera_21_to_11=vla_cfg.get("convert_camera_21_to_11", True),
            )
            logger.info("VLA runner is initialized")
            return runner
        except Exception as e:
            logger.exception("Failed to initialize VLA runner: %s", e)
            raise RuntimeError("Failed to initialize VLA runner") from e

    def _build_sequence_selector(self):
        return SequenceRouter()

    def _run_mixed_instruction(
        self,
        image: np.ndarray,
        instruction: str,
        instruction_type: str,
        state_def: dict,
    ) -> Optional[dict]:
        return self._fallback_policy.run_instruction(
            image=image,
            instruction=instruction,
            instruction_type=instruction_type,
            state_def=state_def,
        )

    def _post_startup_assess(self, image: np.ndarray) -> None:
        task_text = self._task_text or ""
        horizon = self.planner.classify_horizon(task_text, observation_image=image)

        if horizon == "short":
            directive = self.planner.generate_short_directive(
                task_text,
                observation_image=image,
            )
            instruction = directive.get("instruction") if isinstance(directive, dict) else task_text
            instruction_type = directive.get("instruction_type") if isinstance(directive, dict) else "normal"

            canonical = canonicalize_strict_instruction_key(instruction)
            self._short_instruction = canonical if canonical else instruction
            self._short_instruction_type = instruction_type or "normal"
            self._short_state_def = _build_short_state_def(
                instruction=self._short_instruction,
                task_text=task_text,
            )

            self._execution_mode = "short"
            self._plan = {
                "instruction": self._short_instruction,
                "instruction_type": self._short_instruction_type,
            }
            self._episode_start = datetime.now()
            if self._episode_dir is None:
                self._episode_dir = self._make_episode_dir(task_text)
            self._save_plan()
            logger.info(
                "Short-horizon mode: instruction=%r hint=%s",
                self._short_instruction,
                self._short_state_def.get("execution_hint"),
            )
            return

        plan = self.planner.generate_long_plan(task_text, observation_image=image)
        self._plan = plan or {}
        self._executor = FSMExecutor(
            plan=self._plan,
            instruction_runner=self._run_mixed_instruction,
            vqa_checker=self._vqa_checker,
            vqa_interval_steps=self._vqa_interval_steps,
        )

        state_count = sum(1 for _, value in (self._plan or {}).items() if isinstance(value, dict))
        logger.info("Long-horizon FSM executor ready with %d steps", state_count)

        self._execution_mode = "long"
        self._episode_start = datetime.now()
        if self._episode_dir is None:
            self._episode_dir = self._make_episode_dir(task_text)
        self._save_plan()

    def _vqa_checker(self, image: np.ndarray, state_def: dict) -> Optional[bool]:
        try:
            return self.planner.vqa_check_subgoal(
                task_text=self._task_text or "",
                state_def=state_def,
                observation_image=image,
            )
        except Exception:
            logger.exception("VQA checker failed")
            return None

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
        logger.info("Plan saved -> %s", path)

    def _build_skill_log(self) -> list[dict]:
        history = self._fallback_policy.skill_history
        if not history:
            return []
        log: list[dict] = []
        current = history[0]
        count = 1
        for skill in history[1:]:
            if skill == current:
                count += 1
            else:
                log.append({"skill": current, "count": count})
                current = skill
                count = 1
        log.append({"skill": current, "count": count})
        return log

    def _save_episode_result(self) -> None:
        if self._episode_dir is None:
            return

        elapsed = None
        if self._episode_start is not None:
            elapsed = (datetime.now() - self._episode_start).total_seconds()

        skill_history = self._fallback_policy.skill_history
        skill_log = self._build_skill_log()

        if self._execution_mode == "long" and self._executor is not None:
            result = {
                "task": self._plan.get("task") if self._plan else None,
                "execution_mode": "long",
                "finished": self._executor.finished,
                "result": self._executor.result,
                "total_steps": self._executor.total_step_count,
                "final_state": self._executor.current_state,
                "skill_history": skill_history,
                "skill_log": skill_log,
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
                "skill_history": skill_history,
                "skill_log": skill_log,
                "elapsed_seconds": elapsed,
                "timestamp": datetime.now().isoformat(),
            }
        else:
            return

        path = self._episode_dir / "result.json"
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        logger.info("Episode result saved -> %s", path)
