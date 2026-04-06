"""
FSM Executor

Runs a plan (FSM JSON) step-by-step.  Designed for the Purple Agent
protocol where the executor does NOT own the env loop — it simply produces
the next agent action each time ``step()`` is called with a new observation.

Key design:
- Transitions are TIMEOUT-ONLY (step count based).
- No VLM state checks (too many hallucinations).
- Plans use "always" (immediate) or "timeout" (max_steps) conditions only.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

from src.action.converter import noop_agent_action
from src.planner.plan_format import to_canonical_plan

logger = logging.getLogger(__name__)


class FSMExecutor:
    """
    Stateful FSM interpreter that yields one agent action packet per ``step()`` call.

    Uses TIMEOUT-ONLY transitions (no VLM checks for robustness).
    
    Lifecycle:
        1. Constructed with a validated plan.
        2. Each call to ``step(image)`` returns the next action packet (or None
           when the FSM reaches a terminal state).
    """

    def __init__(
        self,
        plan: dict,
        instruction_runner: Optional[
            Callable[[np.ndarray, str, str, dict], Optional[dict]]
        ] = None,
        vqa_checker: Optional[Callable[[np.ndarray, dict], Optional[bool]]] = None,
        vqa_interval_steps: int = 600,
    ):
        self.plan = to_canonical_plan(plan)
        self.instruction_runner = instruction_runner

        if self.instruction_runner is None:
            raise ValueError("instruction_runner is required (VLA-only execution)")

        # FSM state tracking
        self.current_state: str = self.plan["initial_state"]
        self.state_step_count: int = 0
        self.total_step_count: int = 0

        global_cfg = self.plan.get("global_config", {})
        self.global_max_steps: int = global_cfg.get("max_total_steps", 12000)

        # VQA hooks: optional checker and interval (in steps)
        self.vqa_checker = vqa_checker
        self.vqa_interval_steps = int(vqa_interval_steps or 600)

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
        
        Transitions are evaluated every step since they are timeout-based (no VLM checks).
        """
        # Use a bounded loop instead of recursion to avoid stack overflow
        # when transitions cycle back to the same state.
        for _guard in range(64):
            if self.finished:
                return None

            if self.total_step_count >= self.global_max_steps:
                self._terminate("global_timeout")
                return None

            state_def = self.plan["states"][self.current_state]
            if state_def.get("terminal"):
                self._terminate(state_def.get("result", "unknown"))
                return None

            # VLA instruction-driven execution path (mandatory)
            instruction = state_def.get("instruction")
            if not isinstance(instruction, str) or not instruction.strip():
                self._terminate("invalid_policy_no_instruction")
                return None

            # Periodic VQA-based completion check (per-state)
            if self.vqa_checker and self.state_step_count > 0 and self.state_step_count % self.vqa_interval_steps == 0:
                try:
                    vqa_result = self.vqa_checker(image, state_def)
                    if vqa_result is True:
                        # Pick a sensible next_state from transitions if present
                        transitions = state_def.get("transitions", [])
                        next_state = None
                        for trans in transitions:
                            if isinstance(trans, dict) and trans.get("next_state"):
                                next_state = trans.get("next_state")
                                break
                        if next_state:
                            self._transition_to(next_state)
                            continue
                except Exception:
                    logger.exception("VQA checker raised an exception")

            # Evaluate timeout-based transitions every step
            # (No VLM checks otherwise)
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
        """
        Evaluate transition conditions and move to next state.
        
        Only supports timeout-based transitions:
        - "always": unconditional, immediate transition
        - "timeout": transition when step_count reaches max_steps
        """
        transitions = state_def.get("transitions", [])

        for trans in transitions:
            cond = trans["condition"]
            ctype = cond["type"]

            if ctype == "always":
                self._transition_to(trans["next_state"])
                return

            elif ctype == "timeout":
                max_steps = cond.get("max_steps")
                if max_steps is not None and self.state_step_count >= max_steps:
                    self._transition_to(trans["next_state"])
                    return
