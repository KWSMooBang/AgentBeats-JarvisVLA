"""
FSM Executor

Runs a plan (FSM JSON) step-by-step.  Designed for the Purple Agent
protocol where the executor does NOT own the env loop — it simply produces
the next agent action each time ``step()`` is called with a new observation.

Key constraint: the agent only sees **observation images**.
All state checks use the VLM.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

from src.executor.vlm_checker import VLMStateChecker
from src.action.converter import noop_agent_action
from src.planner.spec_format import to_canonical_spec

logger = logging.getLogger(__name__)


class FSMExecutor:
    """
    Stateful FSM interpreter that yields one agent action packet per ``step()`` call.

    Lifecycle:
        1. Constructed with a validated plan.
        2. Each call to ``step(image)`` returns the next action packet (or None
           when the FSM reaches a terminal state).
    """

    def __init__(
        self,
        plan: dict,
        vlm_checker: VLMStateChecker,
        instruction_runner: Optional[
            Callable[[np.ndarray, str, str, dict], Optional[dict]]
        ] = None,
    ):
        self.spec = to_canonical_spec(plan)
        self.vlm = vlm_checker
        self.instruction_runner = instruction_runner

        if self.instruction_runner is None:
            raise ValueError("instruction_runner is required (VLA-only execution)")

        # FSM state tracking
        self.current_state: str = self.spec["initial_state"]
        self.state_step_count: int = 0
        self.total_step_count: int = 0
        self.retry_counts: dict[str, int] = {}

        global_cfg = self.spec.get("global_config", {})
        self.global_max_steps: int = global_cfg.get("max_total_steps", 600)
        self.vlm_check_interval: int = global_cfg.get("vlm_check_interval", 20)

        # Terminal flag
        self.finished: bool = False
        self.result: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, image: np.ndarray) -> Optional[dict]:
        """
        Given the latest observation image, return the next action packet.
        Returns None when the FSM has reached a terminal state.
        """
        # Use a bounded loop instead of recursion to avoid stack overflow
        # when transitions cycle back to the same state.
        for _guard in range(64):
            if self.finished:
                return None

            if self.total_step_count >= self.global_max_steps:
                self._terminate("global_timeout")
                return None

            state_def = self.spec["states"][self.current_state]
            if state_def.get("terminal"):
                self._terminate(state_def.get("result", "unknown"))
                return None

            # VLA instruction-driven execution path (mandatory)
            instruction = state_def.get("instruction")
            if not isinstance(instruction, str) or not instruction.strip():
                self._terminate("invalid_policy_no_instruction")
                return None

            if self._should_eval_instruction_transition():
                prev_state = self.current_state
                self._evaluate_transitions(image, state_def)
                if self.current_state != prev_state:
                    continue

            instruction_type = state_def.get("instruction_type", "auto")
            action = self.instruction_runner(
                image,
                instruction.strip(),
                instruction_type,
                state_def,
            )
            if action is None:
                action = {
                    "__action_format__": "agent",
                    "action": noop_agent_action(),
                }
            self._tick()
            return action

        logger.warning("step() guard limit reached — returning noop action")
        self._tick()
        return {
            "__action_format__": "agent",
            "action": noop_agent_action(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tick(self):
        self.total_step_count += 1
        self.state_step_count += 1

    def _should_eval_instruction_transition(self) -> bool:
        interval = max(1, int(self.vlm_check_interval))
        return self.state_step_count > 0 and (self.state_step_count % interval == 0)

    def _terminate(self, result: str):
        self.finished = True
        self.result = result
        logger.info(
            "FSM terminated: result=%s, total_steps=%d, state=%s",
            result, self.total_step_count, self.current_state,
        )

    def _transition_to(self, new_state: str):
        if new_state != self.current_state:
            logger.info(
                "[Step %d] Transition: %s → %s",
                self.total_step_count, self.current_state, new_state,
            )
        self.current_state = new_state
        self.state_step_count = 0

    def _evaluate_transitions(self, image: np.ndarray, state_def: dict):
        """Evaluate transition conditions and move to next state."""
        transitions = state_def.get("transitions", [])

        for trans in transitions:
            cond = trans["condition"]
            ctype = cond["type"]

            if ctype == "always":
                self._transition_to(trans["next_state"])
                return

            elif ctype == "vlm_check":
                answer = self.vlm.ask_yes_no(image, cond["query"])
                target = trans["on_true"] if answer else trans["on_false"]
                self._transition_to(target)
                return

            elif ctype == "timeout":
                if self.state_step_count >= cond["max_steps"]:
                    self._transition_to(trans["next_state"])
                    return

            elif ctype == "retry_exhausted":
                sname = self.current_state
                self.retry_counts[sname] = self.retry_counts.get(sname, 0) + 1
                max_r = state_def.get("max_retries", 3)
                if self.retry_counts[sname] >= max_r:
                    self._transition_to(trans["next_state"])
                    return

            elif ctype == "inventory_has":
                query = (
                    f"Look at the hotbar/inventory area at the bottom of the screen. "
                    f"Does the player have at least {cond.get('count', 1)} {cond['item']}?"
                )
                has = self.vlm.ask_yes_no(image, query)
                target = trans["on_true"] if has else trans.get("on_false", self.current_state)
                self._transition_to(target)
                return

            elif ctype == "scene_check":
                query = (
                    "Does this Minecraft screenshot match the following description: "
                    f"{cond['description']}"
                )
                match = self.vlm.ask_yes_no(image, query)
                target = trans["on_true"] if match else trans.get("on_false", self.current_state)
                self._transition_to(target)
                return
