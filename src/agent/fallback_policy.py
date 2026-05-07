from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from src.action.converter import ActionConverter, noop_agent_action
from src.agent.primitive_scripts import primitive_env_action, semantic_script
from src.agent.sequence_router import SequenceRouter
from src.agent.sequence_templates import (
    default_primitives as build_default_primitives,
    sequence_catalog as build_sequence_catalog,
)
from src.planner.instruction_registry import canonicalize_strict_instruction_key

logger = logging.getLogger(__name__)


class FallbackPolicyEngine:
    """Selects and executes VLA, scripted, or hybrid policies for one instruction."""

    def __init__(
        self,
        action_converter: ActionConverter,
        vla_runner: Any,
        sequence_selector: Optional[SequenceRouter] = None,
    ) -> None:
        self._action_converter = action_converter
        self._vla_runner = vla_runner
        self._sequence_selector = sequence_selector
        self.reset_episode()

    @property
    def skill_history(self) -> list[str]:
        return self._selection_history

    def reset_episode(self) -> None:
        self._selection_history: list[str] = []
        self._script_runtime_key: Optional[str] = None
        self._script_runtime_signature: Optional[str] = None
        self._script_runtime_step: int = 0
        self._primitive_runtime_index: int = 0
        self._primitive_runtime_step: int = 0
        self._selection_runtime_signature: Optional[str] = None
        self._selection_policy_spec: Optional[dict[str, Any]] = None

    def run_state_instruction(
        self,
        image: np.ndarray,
        instruction: str,
        instruction_type: str,
        state_def: dict,
    ) -> Optional[dict]:
        return self._vla_runner.run(
            image=image,
            instruction=instruction,
            instruction_type=instruction_type,
            state_def=state_def,
        )

    @staticmethod
    def normalize_execution_hint(value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None
        value = value.strip().lower()
        if value in {"vla", "scripted", "hybrid"}:
            return value
        return None

    def make_policy_spec(
        self,
        image: np.ndarray,
        instruction: str,
        state_def: Optional[dict] = None,
    ) -> dict:
        state_def = state_def or {}
        runtime_signature = self.script_signature(instruction, state_def)
        if (
            self._selection_runtime_signature == runtime_signature
            and self._selection_policy_spec is not None
        ):
            return dict(self._selection_policy_spec)

        primitives = state_def.get("primitives")
        execution_hint = self.normalize_execution_hint(state_def.get("execution_hint"))
        sequence_name = state_def.get("sequence_name")
        if isinstance(primitives, list) and primitives:
            policy_spec = {
                "execution_hint": execution_hint or "hybrid",
                "sequence_name": sequence_name,
                "primitives": primitives,
                "selector_reason": "planner_primitives",
            }
        else:
            policy_spec = self._build_selected_policy_spec(
                instruction=instruction,
                state_def=state_def,
                execution_hint=execution_hint,
                sequence_name=sequence_name,
            )

        self._selection_runtime_signature = runtime_signature
        self._selection_policy_spec = dict(policy_spec)
        history_item = policy_spec.get("sequence_name") or policy_spec["execution_hint"]
        self._selection_history.append(str(history_item))
        logger.info(
            "[PolicySpec] hint=%s seq=%s reason=%s primitives=%d instr=%r",
            policy_spec.get("execution_hint"),
            policy_spec.get("sequence_name"),
            policy_spec.get("selector_reason"),
            len(policy_spec.get("primitives") or []),
            instruction[:80],
        )
        if len(self._selection_history) > 20:
            self._selection_history.pop(0)
        return policy_spec

    def _build_selected_policy_spec(
        self,
        instruction: str,
        state_def: dict[str, Any],
        execution_hint: Optional[str],
        sequence_name: Optional[str],
    ) -> dict[str, Any]:
        task_text = state_def.get("task_text", "")
        if task_text:
            task_text_sel = self._select_sequence(task_text, {}, require_sequence=False)
            if task_text_sel.get("sequence_name"):
                sequence_name = task_text_sel["sequence_name"]
                execution_hint = (
                    self.normalize_execution_hint(task_text_sel.get("execution_hint"))
                    or execution_hint
                )
                logger.debug(
                    "[PolicySpec] task_text routing overrides planner: seq=%s reason=%s",
                    sequence_name,
                    task_text_sel.get("reason"),
                )

        require_sequence = execution_hint in {"scripted", "hybrid"} and not sequence_name
        if sequence_name and execution_hint in {"scripted", "hybrid"}:
            selected_hint = execution_hint
            selection = {
                "execution_hint": execution_hint,
                "sequence_name": sequence_name,
                "reason": "planner_sequence_name",
            }
        else:
            selection = self._select_sequence(
                instruction,
                state_def,
                require_sequence=require_sequence,
            )
            selected_hint = self.normalize_execution_hint(selection.get("execution_hint")) or "vla"
            sequence_name = selection.get("sequence_name")

        if require_sequence and not sequence_name:
            logger.warning(
                "scripted/hybrid hint lacked concrete sequence; falling back to VLA. instruction=%r",
                instruction[:120],
            )
            return {
                "execution_hint": "vla",
                "sequence_name": None,
                "primitives": self.default_primitives(
                    sequence_name=None,
                    execution_hint="vla",
                    instruction=instruction,
                ),
                "selector_reason": selection.get(
                    "reason",
                    "missing_sequence_after_selector_retry",
                ),
            }

        return {
            "execution_hint": selected_hint,
            "sequence_name": sequence_name,
            "primitives": self.default_primitives(
                sequence_name=sequence_name,
                execution_hint=selected_hint,
                instruction=instruction,
            ),
            "selector_reason": selection.get("reason", ""),
        }

    def default_primitives(
        self,
        sequence_name: Optional[str],
        execution_hint: str,
        instruction: Optional[str] = None,
    ) -> list[dict]:
        return build_default_primitives(
            sequence_name=sequence_name,
            execution_hint=execution_hint,
            instruction=instruction,
        )

    def run_instruction(
        self,
        image: np.ndarray,
        instruction: str,
        instruction_type: str,
        state_def: dict,
    ) -> Optional[dict]:
        try:
            return self._run_policy_instruction(
                image=image,
                instruction=instruction,
                instruction_type=instruction_type,
                state_def=state_def,
            )
        except Exception as e:
            logger.exception("run_instruction failed: %s", e)
            return {
                "__action_format__": "agent",
                "action": noop_agent_action(),
            }

    def _run_policy_instruction(
        self,
        image: np.ndarray,
        instruction: str,
        instruction_type: str,
        state_def: dict,
    ) -> Optional[dict]:
        runtime_signature = self.script_signature(instruction, state_def)
        policy_spec = self.make_policy_spec(image, instruction, state_def)
        execution_hint = policy_spec.get("execution_hint", "vla")
        sequence_name = policy_spec.get("sequence_name")
        primitives = policy_spec.get("primitives") or []

        if self._script_runtime_signature != runtime_signature:
            self._script_runtime_signature = runtime_signature
            self._script_runtime_key = sequence_name
            self._script_runtime_step = 0
            self._primitive_runtime_index = 0
            self._primitive_runtime_step = 0

        canonical = canonicalize_strict_instruction_key(instruction)

        if execution_hint == "vla":
            return self.run_state_instruction(
                image,
                canonical or instruction,
                instruction_type,
                state_def,
            )

        if execution_hint in {"scripted", "hybrid"} and primitives:
            packet = self._run_primitive_sequence(
                image=image,
                instruction=instruction,
                instruction_type=instruction_type,
                state_def=state_def,
                runtime_signature=runtime_signature,
                script_key=sequence_name,
                primitives=primitives,
            )
            if packet is not None:
                return packet

        return self.run_state_instruction(
            image,
            canonical or instruction,
            instruction_type,
            state_def,
        )

    def _run_primitive_sequence(
        self,
        image: np.ndarray,
        instruction: str,
        instruction_type: str,
        state_def: dict,
        runtime_signature: str,
        script_key: Optional[str],
        primitives: list[dict],
    ) -> Optional[dict]:
        if self._script_runtime_signature != runtime_signature:
            self._script_runtime_signature = runtime_signature
            self._script_runtime_key = script_key
            self._script_runtime_step = 0
            self._primitive_runtime_index = 0
            self._primitive_runtime_step = 0

        if not primitives:
            return None

        while self._primitive_runtime_index < len(primitives):
            primitive = primitives[self._primitive_runtime_index]
            steps_budget = int(primitive.get("steps", 1) or 1)
            local_step = self._primitive_runtime_step
            executor = primitive.get("executor", "script")
            if local_step == 0:
                logger.info(
                    "[Primitive] idx=%d/%d exec=%s name=%r budget=%d seq=%s",
                    self._primitive_runtime_index,
                    len(primitives),
                    executor,
                    primitive.get("primitive") or primitive.get("instruction", "")[:40],
                    steps_budget,
                    script_key,
                )

            if executor == "vla":
                action = self.run_state_instruction(
                    image=image,
                    instruction=primitive.get("instruction") or instruction,
                    instruction_type=primitive.get("instruction_type") or instruction_type,
                    state_def=state_def,
                )
            else:
                action = {
                    "__action_format__": "agent",
                    "action": self.script_primitive_action(
                        primitive_name=primitive.get("primitive") or (script_key or ""),
                        local_step=local_step,
                        script_key=script_key,
                    ),
                }

            self._primitive_runtime_step += 1
            self._script_runtime_step += 1

            if self._primitive_runtime_step >= steps_budget:
                self._primitive_runtime_index += 1
                self._primitive_runtime_step = 0

            return action

        return None

    @staticmethod
    def semantic_script(script_key: str, step: int) -> dict:
        return semantic_script(script_key, step)

    @staticmethod
    def script_signature(instruction: str, state_def: Optional[dict]) -> str:
        state_def = state_def or {}
        return "|".join([
            instruction.strip().lower(),
            str(state_def.get("description", "")),
            str(state_def.get("instruction_type", "")),
        ])

    def script_primitive_action(
        self,
        primitive_name: str,
        local_step: int,
        script_key: Optional[str] = None,
    ) -> dict:
        return self.env_to_agent_action(
            primitive_env_action(
                primitive_name=primitive_name,
                local_step=local_step,
                script_key=script_key,
            )
        )

    def env_to_agent_action(self, env_action: dict) -> dict:
        full = {
            "attack": 0, "back": 0, "drop": 0, "forward": 0,
            "hotbar.1": 0, "hotbar.2": 0, "hotbar.3": 0,
            "hotbar.4": 0, "hotbar.5": 0, "hotbar.6": 0,
            "hotbar.7": 0, "hotbar.8": 0, "hotbar.9": 0,
            "inventory": 0, "jump": 0, "left": 0, "right": 0,
            "sneak": 0, "sprint": 0, "use": 0,
            "camera": np.array([0.0, 0.0]),
        }
        for k, v in env_action.items():
            if k == "camera":
                full["camera"] = np.array(v, dtype=np.float64)
            else:
                full[k] = v
        return self._action_converter.env_to_agent(full)

    def semantic_script_action(self, script_key: Optional[str], step: int) -> dict:
        return self.env_to_agent_action(semantic_script(script_key or "", step))

    def _select_sequence(
        self,
        instruction: str,
        state_def: dict[str, Any],
        require_sequence: bool = False,
    ) -> dict[str, Any]:
        selector = self._sequence_selector or SequenceRouter()
        return selector.select_sequence(
            instruction=instruction,
            state_def=state_def,
            sequence_catalog=self.sequence_catalog(),
            require_sequence=require_sequence,
        )

    @staticmethod
    def sequence_catalog() -> dict[str, dict[str, Any]]:
        return build_sequence_catalog()
