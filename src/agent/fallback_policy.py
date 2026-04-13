from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from src.action.converter import ActionConverter, noop_agent_action
from src.planner.instruction_registry import canonicalize_strict_instruction_key
from src.agent.sequence_router import SequenceRouter

logger = logging.getLogger(__name__)


class FallbackPolicyEngine:
    """Owns non-VLA fallback policy selection and scripted execution."""

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
            # Task-text takes routing priority over planner-generated sequence_name.
            # The planner's LLM can misinterpret task intent (e.g. "lay carpet" →
            # "craft_item: carpet"), so we route by the raw task_text first.
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
                        sequence_name, task_text_sel.get("reason"),
                    )

            require_sequence = (
                execution_hint in {"scripted", "hybrid"}
                and not sequence_name
            )
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
                    "scripted/hybrid hint lacked concrete sequence; selector could not recover, falling back to VLA. instruction=%r",
                    instruction[:120],
                )
                policy_spec = {
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
            else:
                policy_spec = {
                    "execution_hint": selected_hint,
                    "sequence_name": sequence_name,
                    "primitives": self.default_primitives(
                        sequence_name=sequence_name,
                        execution_hint=selected_hint,
                        instruction=instruction,
                    ),
                    "selector_reason": selection.get("reason", ""),
                }

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

    def default_primitives(
        self,
        sequence_name: Optional[str],
        execution_hint: str,
        instruction: Optional[str] = None,
    ) -> list[dict]:
        if execution_hint == "vla":
            return [{
                "executor": "vla",
                "instruction": instruction or "",
                "instruction_type": "auto",
                "steps": 999999,
            }]

        templates: dict[str, list[dict]] = {
            "container_gui_cycle": [
                {"executor": "vla", "instruction": "open the relevant workstation or container", "instruction_type": "normal", "steps": 30},
                {"executor": "script", "primitive": "cycle_hotbar", "steps": 36},
                {"executor": "script", "primitive": "gui_click_confirm_repeat", "steps": 80},
            ],
            "approach_then_use_on_target": [
                {"executor": "vla", "instruction": "approach the target entity", "instruction_type": "normal", "steps": 40},
                {"executor": "script", "primitive": "cycle_hotbar", "steps": 24},
                {"executor": "script", "primitive": "hold_use_forward", "steps": 60},
            ],
            "approach_then_ground_use": [
                {"executor": "vla", "instruction": "move to the spot where the item should be used on the ground", "instruction_type": "normal", "steps": 24},
                {"executor": "script", "primitive": "look_down_use", "steps": 60},
            ],
            "approach_farmland_then_plant_rows": [
                {"executor": "vla", "instruction": "move to the prepared ground area", "instruction_type": "normal", "steps": 40},
                {"executor": "script", "primitive": "look_down", "steps": 16},
                {"executor": "script", "primitive": "cycle_hotbar", "steps": 24},
                {"executor": "script", "primitive": "plant_row", "steps": 80},
            ],
            "approach_then_interact": [
                {"executor": "vla", "instruction": "move to the nearby interactable object", "instruction_type": "normal", "steps": 40},
                {"executor": "script", "primitive": "look_down", "steps": 16},
                {"executor": "script", "primitive": "hold_use", "steps": 50},
            ],
            "place_while_walking": [
                {"executor": "vla", "instruction": "move through the dark area", "instruction_type": "normal", "steps": 50},
                {"executor": "script", "primitive": "cycle_hotbar", "steps": 24},
                {"executor": "script", "primitive": "look_down_use_walk", "steps": 80},
            ],
            "approach_then_open_interactable": [
                {"executor": "vla", "instruction": "move to the nearby container or interactable block", "instruction_type": "normal", "steps": 45},
                {"executor": "script", "primitive": "tap_use", "steps": 32},
            ],
            "structured_build_sequence": [
                {"executor": "vla", "instruction": "face the portal build area", "instruction_type": "normal", "steps": 30},
                {"executor": "script", "primitive": "portal_frame_place", "steps": 140},
                {"executor": "script", "primitive": "look_down_and_use", "steps": 60},
            ],
            "stack_then_trigger": [
                {"executor": "vla", "instruction": "move close to the snow block stack position", "instruction_type": "normal", "steps": 30},
                {"executor": "script", "primitive": "snow_golem_stack", "steps": 90},
            ],
            "dig_then_fill": [
                {"executor": "script", "primitive": "dig_down_attack", "steps": 80},
                {"executor": "script", "primitive": "fill_with_block", "steps": 40},
            ],
            "approach_then_place_and_use": [
                {"executor": "vla", "instruction": "move to a suitable placement surface", "instruction_type": "normal", "steps": 40},
                {"executor": "script", "primitive": "boat_place_use", "steps": 60},
            ],
            "approach_target_then_place_vehicle": [
                {"executor": "vla", "instruction": "approach the target area or entity", "instruction_type": "normal", "steps": 40},
                {"executor": "script", "primitive": "boat_place_use", "steps": 70},
            ],
            "approach_height_then_use": [
                {"executor": "vla", "instruction": "move to a suitable edge or elevated surface", "instruction_type": "normal", "steps": 50},
                {"executor": "script", "primitive": "look_down_use", "steps": 70},
            ],
            "line_place_repeat": [
                {"executor": "script", "primitive": "open_inventory", "steps": 6},
                {"executor": "vla", "instruction_type": "recipe", "steps": 50},
                {"executor": "vla", "instruction": "select the carpet in your hotbar", "instruction_type": "normal", "steps": 15},
                {"executor": "script", "primitive": "place_row_walk", "steps": 80},
            ],
            "stack_place_repeat": [
                {"executor": "script", "primitive": "stack_vertical", "steps": 100},
            ],
            "scatter_ground_placeables": [
                {"executor": "vla", "instruction": "select a small flat flower such as a dandelion, poppy, or tulip from your hotbar", "instruction_type": "normal", "steps": 20},
                {"executor": "script", "primitive": "look_down_use_walk", "steps": 120},
            ],
            "place_light_sources": [
                {"executor": "vla", "instruction": "select a torch or light-emitting block from your hotbar", "instruction_type": "normal", "steps": 20},
                {"executor": "script", "primitive": "look_down_use_walk", "steps": 120},
            ],
            "approach_then_vertical_place": [
                {"executor": "vla", "instruction": "face a nearby wall or vertical surface and move close to it", "instruction_type": "normal", "steps": 30},
                {"executor": "vla", "instruction": "select an item frame, painting, or wall decoration from your hotbar", "instruction_type": "normal", "steps": 20},
                {"executor": "script", "primitive": "place_wall_use", "steps": 90},
            ],
            "aim_then_use_repeat": [
                {"executor": "vla", "instruction": "face the nearby animal", "instruction_type": "normal", "steps": 24},
                {"executor": "script", "primitive": "rod_cast_repeat", "steps": 100},
            ],
            "approach_then_place_climbable": [
                {"executor": "vla", "instruction": "move to the vertical surface", "instruction_type": "normal", "steps": 30},
                {"executor": "script", "primitive": "place_wall_use", "steps": 60},
                {"executor": "script", "primitive": "climb_forward", "steps": 70},
            ],
            "navigate_then_dig_then_open": [
                {"executor": "vla", "instruction": "follow the treasure map and move toward the target area", "instruction_type": "normal", "steps": 80},
                {"executor": "script", "primitive": "dig_down_attack", "steps": 60},
                {"executor": "script", "primitive": "tap_use", "steps": 30},
            ],
            "consume_cycle": [
                {"executor": "script", "primitive": "cycle_hotbar_then_hold_use", "steps": 120},
            ],
            "consume_item_sequence": [
                {"executor": "script", "primitive": "cycle_hotbar_then_hold_use", "steps": 120},
            ],
            "open_inventory_craft": [
                {"executor": "script", "primitive": "open_inventory", "steps": 6},
                {"executor": "vla", "instruction_type": "recipe", "steps": 110},
            ],
            "throw_cycle": [
                {"executor": "script", "primitive": "throw_held_item_flow", "steps": 80},
            ],
            "drop_cycle": [
                {"executor": "script", "primitive": "drop_cycle", "steps": 90},
            ],
            "hold_use_cycle": [
                {"executor": "script", "primitive": "cycle_hotbar_then_hold_use", "steps": 120},
            ],
            "view_upward": [
                {"executor": "script", "primitive": "look_up", "steps": 20},
                {"executor": "script", "primitive": "maintain_up_view", "steps": 60},
            ],
            "melee_combat_loop": [
                {"executor": "vla", "instruction": "approach the nearest enemy or mob", "instruction_type": "normal", "steps": 30},
                {"executor": "script", "primitive": "melee_attack", "steps": 80},
            ],
            "ranged_combat_loop": [
                {"executor": "vla", "instruction": "face the target entity", "instruction_type": "normal", "steps": 20},
                {"executor": "script", "primitive": "ranged_attack", "steps": 80},
            ],
            "throw_projectile_loop": [
                {"executor": "vla", "instruction": "face the target or direction to throw", "instruction_type": "normal", "steps": 20},
                {"executor": "script", "primitive": "throw_weapon", "steps": 60},
            ],
            "use_on_nearby_entity": [
                {"executor": "vla", "instruction": "face and move toward the nearby entity", "instruction_type": "normal", "steps": 30},
                {"executor": "script", "primitive": "approach_and_use_on_entity", "steps": 70},
            ],
            "collect_surface_blocks": [
                {"executor": "vla", "instruction": "move above the target ground blocks", "instruction_type": "normal", "steps": 30},
                {"executor": "script", "primitive": "mine_ground", "steps": 80},
            ],
            "chop_tree_loop": [
                {"executor": "vla", "instruction": "face a nearby tree trunk", "instruction_type": "normal", "steps": 30},
                {"executor": "script", "primitive": "chop_tree", "steps": 100},
            ],
            "mine_ahead_loop": [
                {"executor": "vla", "instruction": "face the block to mine", "instruction_type": "normal", "steps": 20},
                {"executor": "script", "primitive": "mine_forward", "steps": 80},
            ],
            "rest_at_object": [
                {"executor": "vla", "instruction": "move to and face the bed or rest object", "instruction_type": "normal", "steps": 40},
                {"executor": "script", "primitive": "approach_and_use_rest_object", "steps": 60},
            ],
            "defend_with_item": [
                {"executor": "script", "primitive": "cycle_hotbar", "steps": 24},
                {"executor": "script", "primitive": "hold_defensive_item", "steps": 100},
            ],
            "shear_nearby_entity": [
                {"executor": "vla", "instruction": "face and approach the nearby animal", "instruction_type": "normal", "steps": 30},
                {"executor": "script", "primitive": "shear_target", "steps": 70},
            ],
            "clear_ground_plants": [
                {"executor": "vla", "instruction": "select a cutting tool such as shears or a sword from your hotbar", "instruction_type": "normal", "steps": 20},
                {"executor": "script", "primitive": "attack_walk_sweep", "steps": 120},
            ],
        }
        if sequence_name in templates:
            return templates[sequence_name]
        return [{
            "executor": "vla",
            "instruction": instruction or "",
            "instruction_type": "auto",
            "steps": 999999,
        }]

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
        if script_key == "look_up":
            if step < 20:
                return {"camera": [-10.0, 0.0]}
            return {"camera": [-1.0, 3.0]}
        if script_key == "drop_held_item":
            slot = min((step // 14) + 1, 9)
            if step % 14 == 0:
                return {f"hotbar.{slot}": 1}
            return {"drop": 1}
        if script_key == "throw_held_item":
            cycle = step % 15
            slot = (step // 15) % 4 + 1
            if cycle == 0:
                return {f"hotbar.{slot}": 1}
            if cycle < 3:
                return {"camera": [-5.0, 0.0]}
            if cycle < 7:
                return {"use": 1}
            return {}
        if script_key == "stack_vertical":
            if step < 2:
                return {"hotbar.1": 1}
            if step < 5:
                return {"camera": [9.0, 0.0]}
            return {"jump": 1, "use": 1, "sneak": 1}
        if script_key == "consume_held_item":
            slot = (step // 40) % 9 + 1
            cycle = step % 40
            if cycle == 0:
                return {f"hotbar.{slot}": 1}
            return {"use": 1}
        if script_key == "ranged_attack":
            slot = (step // 30) % 9 + 1
            cycle = step % 30
            if cycle == 0:
                return {f"hotbar.{slot}": 1}
            if cycle < 3:
                return {"camera": [-5.0, 0.0]}
            if cycle < 23:
                return {"use": 1}
            return {}
        if script_key == "throw_weapon":
            slot = (step // 20) % 9 + 1
            cycle = step % 20
            if cycle == 0:
                return {f"hotbar.{slot}": 1}
            if cycle < 3:
                return {"camera": [-3.0, 0.0]}
            if cycle < 8:
                return {"use": 1}
            return {}
        if script_key == "approach_and_use_on_entity":
            if step < 5:
                return {"forward": 1}
            step2 = step - 5
            slot = (step2 // 15) % 9 + 1
            cycle = step2 % 15
            if cycle == 0:
                return {f"hotbar.{slot}": 1}
            if cycle < 8:
                return {"use": 1}
            return {}
        if script_key == "look_down_and_use":
            cycle = step % 10
            if cycle < 4:
                return {"camera": [9.0, 0.0]}
            if cycle < 7:
                return {"use": 1}
            return {}
        if script_key == "place_seed_rows":
            if step < 4:
                return {"forward": 1}
            if step < 8:
                return {"camera": [9.0, 0.0]}
            step2 = step - 8
            slot = (step2 // 16) % 4 + 1
            cycle = step2 % 16
            if cycle == 0:
                return {f"hotbar.{slot}": 1}
            if cycle < 8:
                return {"use": 1}
            if cycle < 12:
                return {"forward": 1, "sneak": 1}
            return {}
        if script_key == "approach_and_use_rest_object":
            if step < 3:
                return {"forward": 1}
            if step < 7:
                return {"camera": [9.0, 0.0]}
            slot = (step // 25) % 9 + 1
            cycle = step % 25
            if cycle == 0:
                return {f"hotbar.{slot}": 1}
            return {"use": 1}
        if script_key == "hold_defensive_item":
            slot = (step // 30) % 9 + 1
            cycle = step % 30
            if cycle == 0:
                return {f"hotbar.{slot}": 1}
            return {"use": 1}
        if script_key == "container_gui_cycle":
            if step < 3:
                return {"forward": 1}
            if step < 6:
                return {"camera": [4.0, 0.0]}
            if step < 9:
                return {"use": 1}
            slot = (step // 15) % 9 + 1
            cycle = step % 15
            if cycle == 0:
                return {f"hotbar.{slot}": 1}
            if cycle < 5:
                return {"camera": [-3.0, 0.0]}
            if cycle < 7:
                return {"attack": 1}
            if cycle < 12:
                return {"camera": [3.0, 0.0]}
            if cycle < 14:
                return {"attack": 1}
            return {}
        if script_key == "line_place_repeat":
            return FallbackPolicyEngine.semantic_script("place_light_source_along_path", step)
        if script_key == "attack_walk_sweep":
            if step < 5:
                return {"camera": [-5.0, 0.0]}
            c = (step - 5) % 25
            if c < 3:   return {"attack": 1}
            if c < 5:   return {"forward": 1}
            if c < 8:   return {"attack": 1}
            if c < 10:  return {"camera": [0.0, 25.0]}
            if c < 12:  return {"forward": 1}
            if c < 15:  return {"attack": 1}
            if c < 17:  return {"camera": [0.0, -30.0]}
            if c < 19:  return {"forward": 1}
            return {"attack": 1}
        if script_key == "melee_attack":
            cycle = step % 20
            if cycle < 5:
                return {"forward": 1, "sprint": 1}
            if cycle < 12:
                return {"forward": 1, "attack": 1}
            if cycle < 15:
                return {"jump": 1, "attack": 1, "forward": 1}
            return {"camera": [0.0, 8.0]}
        if script_key == "shear_target":
            slot = (step // 18) % 9 + 1
            cycle = step % 18
            if cycle == 0:
                return {f"hotbar.{slot}": 1}
            if cycle < 6:
                return {"forward": 1}
            if cycle < 12:
                return {"use": 1}
            return {"camera": [0.0, 6.0]}
        if script_key == "chop_tree":
            slot = (step // 40) % 4 + 1
            cycle = step % 40
            if cycle == 0:
                return {f"hotbar.{slot}": 1}
            if cycle < 4:
                return {"camera": [-5.0, 0.0], "forward": 1}
            if cycle < 24:
                return {"attack": 1, "forward": 1}
            return {"camera": [6.0, 9.0]}
        if script_key == "mine_forward":
            cycle = step % 24
            if cycle == 0:
                return {"hotbar.1": 1}
            if cycle < 4:
                return {"camera": [3.0, 0.0], "forward": 1}
            if cycle < 18:
                return {"attack": 1, "forward": 1}
            return {"camera": [0.0, 10.0]}
        if script_key == "mine_ground":
            cycle = step % 20
            if cycle == 0:
                return {"hotbar.1": 1}
            if cycle < 4:
                return {"camera": [12.0, 0.0], "sneak": 1}
            if cycle < 16:
                return {"attack": 1, "sneak": 1}
            return {}
        if script_key == "place_light_source_along_path":
            slot = (step // 20) % 9 + 1
            cycle = step % 20
            if cycle == 0:
                return {f"hotbar.{slot}": 1}
            if cycle < 4:
                return {"camera": [9.0, 0.0]}
            if cycle < 8:
                return {"use": 1}
            if cycle < 14:
                return {"forward": 1, "sprint": 1}
            return {"camera": [0.0, 12.0]}
        if script_key == "approach_then_open_interactable":
            if step < 5:
                return {"forward": 1}
            if step < 8:
                return {"camera": [3.0, 0.0]}
            if step < 16:
                return {"use": 1}
            return {"camera": [0.0, 8.0]}
        if script_key == "climb_forward":
            cycle = step % 16
            if cycle < 12:
                return {"forward": 1, "jump": 1, "sprint": 1}
            return {"camera": [-2.0, 6.0]}
        if script_key == "maintain_up_view":
            cycle = step % 40
            if cycle < 20:
                return {"camera": [-1.5, 2.0]}
            return {"camera": [-1.5, -2.0]}
        if script_key == "sprint_forward":
            return {"forward": 1, "sprint": 1}
        if script_key == "scan_rotate":
            cycle = step % 30
            if cycle < 15:
                return {"camera": [0.0, 6.0]}
            return {"camera": [0.0, -6.0]}
        if script_key == "look_forward":
            return {"camera": [0.0, 0.0]}
        return {}

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
        primitive_name = (primitive_name or "").strip()
        if primitive_name in {
            "container_gui_cycle", "consume_held_item", "ranged_attack", "throw_weapon", "approach_and_use_on_entity",
            "look_down_and_use", "place_seed_rows", "approach_and_use_rest_object",
            "hold_defensive_item", "look_up", "throw_held_item",
            "stack_vertical", "drop_held_item", "melee_attack",
            "shear_target", "chop_tree", "mine_forward", "mine_ground",
            "place_light_source_along_path", "approach_then_open_interactable", "climb_forward",
            "maintain_up_view", "sprint_forward", "scan_rotate", "look_forward",
            "attack_walk_sweep",
        }:
            return self.semantic_script_action(primitive_name, local_step)
        if primitive_name == "cycle_hotbar":
            slot = (local_step // 6) % 9 + 1
            if local_step % 6 == 0:
                return self.env_to_agent_action({f"hotbar.{slot}": 1})
            return self.env_to_agent_action({})
        if primitive_name == "hold_use":
            return self.env_to_agent_action({"use": 1})
        if primitive_name == "tap_use":
            if local_step % 4 < 2:
                return self.env_to_agent_action({"use": 1})
            return self.env_to_agent_action({})
        if primitive_name == "hold_use_forward":
            return self.env_to_agent_action({"forward": 1, "use": 1})
        if primitive_name == "look_down":
            return self.env_to_agent_action({"camera": [9.0, 0.0]})
        if primitive_name == "look_down_use":
            if local_step % 10 < 4:
                return self.env_to_agent_action({"camera": [9.0, 0.0]})
            if local_step % 10 < 7:
                return self.env_to_agent_action({"use": 1})
            return self.env_to_agent_action({})
        if primitive_name == "look_down_use_walk":
            cycle = local_step % 15
            if cycle < 3:
                return self.env_to_agent_action({"camera": [5.0, 0.0]})
            if cycle < 6:
                return self.env_to_agent_action({"use": 1})
            if cycle < 10:
                return self.env_to_agent_action({"forward": 1, "camera": [0.0, 20.0]})
            return self.env_to_agent_action({"use": 1})
        if primitive_name == "plant_row":
            cycle = local_step % 16
            slot = (local_step // 16) % 4 + 1
            if cycle == 0:
                return self.env_to_agent_action({f"hotbar.{slot}": 1})
            if cycle < 8:
                return self.env_to_agent_action({"use": 1})
            if cycle < 12:
                return self.env_to_agent_action({"forward": 1, "sneak": 1})
            return self.env_to_agent_action({})
        if primitive_name == "gui_click_confirm_repeat":
            cycle = local_step % 12
            if cycle < 4:
                return self.env_to_agent_action({"attack": 1})
            return self.env_to_agent_action({})
        if primitive_name == "open_inventory":
            if local_step == 0:
                return self.env_to_agent_action({"inventory": 1})
            if local_step < 3:
                return self.env_to_agent_action({"inventory": 0})
            return self.env_to_agent_action({})
        if primitive_name == "cycle_hotbar_then_hold_use":
            cycle = local_step % 40
            slot = (local_step // 40) % 9 + 1
            if cycle == 0:
                return self.env_to_agent_action({f"hotbar.{slot}": 1})
            return self.env_to_agent_action({"use": 1})
        if primitive_name == "place_row_walk":
            cycle = local_step % 12
            if cycle < 3:
                return self.env_to_agent_action({"camera": [9.0, 0.0]})
            if cycle < 6:
                return self.env_to_agent_action({"use": 1})
            if cycle < 10:
                return self.env_to_agent_action({"forward": 1, "sneak": 1})
            return self.env_to_agent_action({"camera": [0.0, 8.0]})
        if primitive_name == "place_wall_use":
            # Approach wall, then place on face: 4 steps approach → use burst → rotate to next spot
            if local_step < 4:
                return self.env_to_agent_action({"forward": 1, "sneak": 1})
            cycle = (local_step - 4) % 16
            if cycle < 5:
                return self.env_to_agent_action({"use": 1})
            if cycle < 7:
                return self.env_to_agent_action({})
            if cycle < 11:
                return self.env_to_agent_action({"camera": [0.0, 20.0]})
            return self.env_to_agent_action({"camera": [0.0, -15.0]})
        if primitive_name == "dig_down_attack":
            cycle = local_step % 12
            if cycle < 4:
                return self.env_to_agent_action({"camera": [12.0, 0.0], "sneak": 1})
            if cycle < 10:
                return self.env_to_agent_action({"attack": 1, "sneak": 1})
            return self.env_to_agent_action({})
        if primitive_name == "fill_with_block":
            cycle = local_step % 10
            if cycle == 0:
                return self.env_to_agent_action({"hotbar.2": 1})
            if cycle < 4:
                return self.env_to_agent_action({"camera": [12.0, 0.0]})
            if cycle < 7:
                return self.env_to_agent_action({"use": 1})
            return self.env_to_agent_action({"jump": 1})
        if primitive_name == "portal_frame_place":
            cycle = local_step % 16
            if cycle < 2:
                return self.env_to_agent_action({"hotbar.1": 1})
            if cycle < 5:
                return self.env_to_agent_action({"camera": [9.0, 0.0]})
            if cycle < 8:
                return self.env_to_agent_action({"use": 1, "jump": 1})
            if cycle < 12:
                return self.env_to_agent_action({"camera": [-9.0, 8.0]})
            return self.env_to_agent_action({"use": 1})
        if primitive_name == "snow_golem_stack":
            cycle = local_step % 18
            if cycle < 2:
                return self.env_to_agent_action({"hotbar.1": 1})
            if cycle < 5:
                return self.env_to_agent_action({"camera": [9.0, 0.0]})
            if cycle < 9:
                return self.env_to_agent_action({"use": 1, "jump": 1})
            if cycle == 9:
                return self.env_to_agent_action({"hotbar.2": 1})
            if cycle < 13:
                return self.env_to_agent_action({"use": 1})
            return self.env_to_agent_action({})
        if primitive_name == "rod_cast_repeat":
            cycle = local_step % 14
            if cycle < 2:
                return self.env_to_agent_action({"hotbar.1": 1})
            if cycle < 5:
                return self.env_to_agent_action({"camera": [-3.0, 0.0]})
            if cycle < 8:
                return self.env_to_agent_action({"use": 1})
            return self.env_to_agent_action({"camera": [0.0, 8.0]})
        if primitive_name == "boat_place_use":
            cycle = local_step % 14
            if cycle < 2:
                return self.env_to_agent_action({"hotbar.1": 1})
            if cycle < 5:
                return self.env_to_agent_action({"camera": [9.0, 0.0]})
            if cycle < 8:
                return self.env_to_agent_action({"use": 1})
            if cycle < 11:
                return self.env_to_agent_action({"forward": 1})
            return self.env_to_agent_action({"use": 1})
        if primitive_name == "throw_held_item_flow":
            return self.semantic_script_action("throw_held_item", local_step)
        if primitive_name == "drop_cycle":
            return self.semantic_script_action("drop_held_item", local_step)
        if script_key:
            return self.semantic_script_action(script_key, local_step)
        return self.env_to_agent_action({})

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
        env_action = self.semantic_script(script_key or "", step)
        return self.env_to_agent_action(env_action)

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
        return {
            "container_gui_cycle": {
                "execution_hint": "scripted",
                "description": "Open a nearby interaction container/UI, then perform repeated simple insertion or confirmation actions.",
                "primitive_names": ["cycle_hotbar", "gui_click_confirm_repeat"],
            },
            "consume_cycle": {
                "execution_hint": "scripted",
                "description": "Cycle held items and consume/use them until one works.",
                "primitive_names": ["cycle_hotbar_then_hold_use"],
            },
            "throw_cycle": {
                "execution_hint": "scripted",
                "description": "Cycle held throwable items and repeatedly use them.",
                "primitive_names": ["throw_held_item_flow"],
            },
            "drop_cycle": {
                "execution_hint": "scripted",
                "description": "Cycle held items and drop them one by one.",
                "primitive_names": ["drop_cycle"],
            },
            "hold_use_cycle": {
                "execution_hint": "scripted",
                "description": "Cycle held items and hold use defensively or continuously.",
                "primitive_names": ["cycle_hotbar_then_hold_use"],
            },
            "approach_then_use_on_target": {
                "execution_hint": "hybrid",
                "description": "Approach a target entity, then run a repeated use sequence.",
                "primitive_names": ["cycle_hotbar", "hold_use_forward"],
            },
            "approach_then_ground_use": {
                "execution_hint": "hybrid",
                "description": "Approach the relevant spot, look down, and use/place on the ground.",
                "primitive_names": ["look_down_use"],
            },
            "approach_farmland_then_plant_rows": {
                "execution_hint": "hybrid",
                "description": "Move to farmland and plant/place items in a row.",
                "primitive_names": ["look_down", "cycle_hotbar", "plant_row"],
            },
            "approach_then_interact": {
                "execution_hint": "hybrid",
                "description": "Approach a nearby interactable object, align, then use repeatedly.",
                "primitive_names": ["look_down", "hold_use"],
            },
            "place_while_walking": {
                "execution_hint": "hybrid",
                "description": "Move through an area while repeatedly placing items on the ground.",
                "primitive_names": ["cycle_hotbar", "look_down_use_walk"],
            },
            "approach_then_open_interactable": {
                "execution_hint": "hybrid",
                "description": "Approach a chest or other interactable block and tap use to open it.",
                "primitive_names": ["tap_use"],
            },
            "structured_build_sequence": {
                "execution_hint": "hybrid",
                "description": "Perform a structured repeated placement routine, then activate it.",
                "primitive_names": ["portal_frame_place", "look_down_use"],
            },
            "stack_then_trigger": {
                "execution_hint": "hybrid",
                "description": "Stack blocks and then place a special top item to trigger an effect.",
                "primitive_names": ["snow_golem_stack"],
            },
            "dig_then_fill": {
                "execution_hint": "scripted",
                "description": "Dig downward and then fill/place a block into the dug space.",
                "primitive_names": ["dig_down_attack", "fill_with_block"],
            },
            "approach_then_place_and_use": {
                "execution_hint": "hybrid",
                "description": "Approach the right surface, place a vehicle/object, then use it.",
                "primitive_names": ["boat_place_use"],
            },
            "approach_target_then_place_vehicle": {
                "execution_hint": "hybrid",
                "description": "Approach a target and place a vehicle/object near it before interacting.",
                "primitive_names": ["boat_place_use"],
            },
            "approach_height_then_use": {
                "execution_hint": "hybrid",
                "description": "Move to a suitable edge or height, then use/place downward.",
                "primitive_names": ["look_down_use"],
            },
            "line_place_repeat": {
                "execution_hint": "hybrid",
                "description": "Place items or blocks in a line while moving forward.",
                "primitive_names": ["cycle_hotbar", "place_row_walk"],
            },
            "stack_place_repeat": {
                "execution_hint": "scripted",
                "description": "Build vertically by repeated jump-place actions.",
                "primitive_names": ["stack_vertical"],
            },
            "scatter_ground_placeables": {
                "execution_hint": "hybrid",
                "description": "Select a decorative natural item (flower, plant) then scatter it on the ground while moving.",
                "primitive_names": ["look_down_use_walk"],
            },
            "place_light_sources": {
                "execution_hint": "hybrid",
                "description": "Select a torch or light-emitting block then place repeatedly while moving.",
                "primitive_names": ["look_down_use_walk"],
            },
            "approach_then_vertical_place": {
                "execution_hint": "hybrid",
                "description": "Approach a wall or vertical surface and place onto it.",
                "primitive_names": ["cycle_hotbar", "place_wall_use"],
            },
            "aim_then_use_repeat": {
                "execution_hint": "hybrid",
                "description": "Aim toward a nearby target and repeatedly use a held item.",
                "primitive_names": ["rod_cast_repeat"],
            },
            "approach_then_place_climbable": {
                "execution_hint": "hybrid",
                "description": "Approach a vertical surface, place a climbable, then climb.",
                "primitive_names": ["place_wall_use", "climb_forward"],
            },
            "navigate_then_dig_then_open": {
                "execution_hint": "hybrid",
                "description": "Navigate to an area, dig downward, then interact/open.",
                "primitive_names": ["dig_down_attack", "tap_use"],
            },
            # --- view tasks ---
            "view_upward": {
                "execution_hint": "scripted",
                "description": "Look upward at the sky: tilt camera up strongly, then hold the sky view.",
                "primitive_names": ["look_up", "maintain_up_view"],
            },
            # --- combat ---
            "melee_combat_loop": {
                "execution_hint": "hybrid",
                "description": "Approach enemy and melee-attack in repeated sprint-swing cycles.",
                "primitive_names": ["melee_attack"],
            },
            "ranged_combat_loop": {
                "execution_hint": "hybrid",
                "description": "Face target and fire ranged weapon (bow, crossbow) repeatedly.",
                "primitive_names": ["ranged_attack"],
            },
            "throw_projectile_loop": {
                "execution_hint": "hybrid",
                "description": "Face target and throw a projectile (snowball, trident, ender pearl) repeatedly.",
                "primitive_names": ["throw_weapon"],
            },
            # --- entity interaction ---
            "use_on_nearby_entity": {
                "execution_hint": "hybrid",
                "description": "Approach and use held item on a nearby entity (lead, bucket, shears, name tag, etc.).",
                "primitive_names": ["approach_and_use_on_entity"],
            },
            "shear_nearby_entity": {
                "execution_hint": "hybrid",
                "description": "Approach an animal and use shears or tool on it.",
                "primitive_names": ["shear_target"],
            },
            "clear_ground_plants": {
                "execution_hint": "scripted",
                "description": "Walk forward and attack repeatedly to clear tall grass, plants, or weeds at eye level.",
                "primitive_names": ["attack_walk_sweep"],
            },
            # --- gathering / mining ---
            "collect_surface_blocks": {
                "execution_hint": "hybrid",
                "description": "Move over ground blocks and break them from above (digging surface).",
                "primitive_names": ["mine_ground"],
            },
            "chop_tree_loop": {
                "execution_hint": "hybrid",
                "description": "Face a tree and chop the trunk/leaves repeatedly.",
                "primitive_names": ["chop_tree"],
            },
            "mine_ahead_loop": {
                "execution_hint": "hybrid",
                "description": "Face a target block and mine it with the held tool.",
                "primitive_names": ["mine_forward"],
            },
            # --- consume / equip ---
            "consume_item_sequence": {
                "execution_hint": "scripted",
                "description": "Cycle hotbar and eat/drink the held consumable item (food, potion).",
                "primitive_names": ["consume_held_item"],
            },
            # --- crafting ---
            "open_inventory_craft": {
                "execution_hint": "hybrid",
                "description": "Open player inventory (E key) then use VLA recipe mode to craft the target item in the crafting GUI.",
                "primitive_names": ["open_inventory"],
            },
            # --- rest / sleep ---
            "rest_at_object": {
                "execution_hint": "hybrid",
                "description": "Navigate to a bed or hammock and use it to rest/sleep.",
                "primitive_names": ["approach_and_use_rest_object"],
            },
            # --- defense ---
            "defend_with_item": {
                "execution_hint": "scripted",
                "description": "Equip and hold a defensive item (shield, totem) continuously.",
                "primitive_names": ["hold_defensive_item"],
            },
        }
