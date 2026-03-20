"""
FSM Executor

Runs a Policy Spec (FSM JSON) step-by-step.  Designed for the Purple Agent
protocol where the executor does NOT own the env loop — it simply produces
the next env_action each time ``step()`` is called with a new observation.

Key constraint: the agent only sees **observation images**.
All state checks use the VLM.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from src.primitives.micro import MicroPrimitives
from src.primitives.perceptual import PerceptualPrimitives
from src.executor.vlm_checker import VLMStateChecker
from src.primitives.guards import TerminationGuards

logger = logging.getLogger(__name__)

PERCEPTUAL_NAMES = frozenset({
    "align_to_target", "approach_target", "mine_target_block",
    "attack_target_entity", "search_and_face", "navigate_to_target",
})


class FSMExecutor:
    """
    Stateful FSM interpreter that yields one env_action per ``step()`` call.

    Lifecycle:
        1. Constructed with a validated policy_spec.
        2. Each call to ``step(image)`` returns the next env_action (or None
           when the FSM reaches a terminal state).

    All internal queues, state counters, and perceptual generators are
    managed automatically.
    """

    def __init__(
        self,
        policy_spec: dict,
        vlm_checker: VLMStateChecker,
    ):
        self.spec = policy_spec
        self.vlm = vlm_checker
        self.guards = TerminationGuards(vlm_checker)

        self.micro = MicroPrimitives()
        self.perceptual = PerceptualPrimitives(vlm_checker)

        # FSM state tracking
        self.current_state: str = policy_spec["initial_state"]
        self.state_step_count: int = 0
        self.total_step_count: int = 0
        self.retry_counts: dict[str, int] = {}

        global_cfg = policy_spec.get("global_config", {})
        self.global_max_steps: int = global_cfg.get("max_total_steps", 600)
        self.vlm_check_interval: int = global_cfg.get("vlm_check_interval", 20)

        # Action queue: pre-computed actions waiting to be emitted
        self._action_queue: list[dict] = []
        # Active perceptual generator (if any)
        self._active_generator = None
        # Index of current primitive within the state's primitive list
        self._prim_index: int = 0
        # Result of last perceptual primitive
        self._last_prim_result: dict = {}
        # Whether we already evaluated transitions for the current state
        self._needs_transition: bool = False
        # Terminal flag
        self.finished: bool = False
        self.result: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, image: np.ndarray) -> Optional[dict]:
        """
        Given the latest observation image, return the next env_action.
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

            # 1. Drain queued actions first
            if self._action_queue:
                action = self._action_queue.pop(0)
                self._tick()
                return action

            # 2. Feed active perceptual generator
            if self._active_generator is not None:
                try:
                    batch = self._active_generator.send(image)
                    self._action_queue.extend(batch)
                    return self._drain_one()
                except StopIteration as e:
                    self._last_prim_result = e.value or {}
                    self._active_generator = None
                    self._prim_index += 1
                    continue  # re-evaluate in next iteration

            # 3. All primitives done → evaluate transitions
            primitives = state_def.get("primitives", [])
            if self._prim_index >= len(primitives):
                self._evaluate_transitions(image, state_def)
                continue  # re-evaluate with (possibly new) state

            # 4. Execute the next primitive
            prim_spec = primitives[self._prim_index]
            name = prim_spec["name"]
            params = prim_spec.get("params", {})

            if name in PERCEPTUAL_NAMES:
                prim_fn = getattr(self.perceptual, name)
                gen = prim_fn(image=image, **params)
                try:
                    batch = next(gen)
                    self._active_generator = gen
                    self._action_queue.extend(batch)
                    return self._drain_one()
                except StopIteration as e:
                    self._last_prim_result = e.value or {}
                    self._prim_index += 1
                    continue
            else:
                prim_fn = getattr(self.micro, name, None)
                if prim_fn is None:
                    logger.warning("Unknown primitive '%s', skipping", name)
                    self._prim_index += 1
                    continue
                actions = prim_fn(**params)
                self._action_queue.extend(actions)
                self._prim_index += 1
                return self._drain_one()

        logger.warning("step() guard limit reached — returning noop")
        from src.primitives.atomic import make_env_action
        self._tick()
        return make_env_action()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _drain_one(self) -> Optional[dict]:
        if not self._action_queue:
            return None
        action = self._action_queue.pop(0)
        self._tick()
        return action

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
        self._prim_index = 0
        self._action_queue.clear()
        self._active_generator = None

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

            elif ctype == "primitive_success":
                success = self._last_prim_result.get("success", False)
                target = trans["on_true"] if success else trans.get("on_false", self.current_state)
                self._transition_to(target)
                return

            elif ctype == "inventory_has":
                has = self.guards.check_inventory_has_via_vlm(
                    image, cond["item"], cond.get("count", 1)
                )
                target = trans["on_true"] if has else trans.get("on_false", self.current_state)
                self._transition_to(target)
                return

            elif ctype == "scene_check":
                match = self.guards.check_scene_matches(image, cond["description"])
                target = trans["on_true"] if match else trans.get("on_false", self.current_state)
                self._transition_to(target)
                return

        # No transition fired → stay in current state but reset primitives
        self._prim_index = 0
