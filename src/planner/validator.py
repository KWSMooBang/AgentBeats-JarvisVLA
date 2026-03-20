"""
Policy Spec Validator

Validates the structural and semantic correctness of FSM JSON
produced by the LLM Planner before execution.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

KNOWN_PRIMITIVES = frozenset({
    # Layer 1 — micro
    "noop", "turn_camera", "scan_left_right", "scan_360",
    "move_forward", "move_backward", "strafe_left", "strafe_right",
    "jump_forward", "sprint_forward", "jump",
    "attack_hold", "attack_forward", "use_once", "use_hold",
    "select_hotbar", "open_inventory", "close_inventory", "drop_item",
    "look_down", "look_up", "sneak_toggle",
    # Layer 2 — perceptual
    "align_to_target", "approach_target", "mine_target_block",
    "attack_target_entity", "search_and_face", "navigate_to_target",
})

KNOWN_CONDITIONS = frozenset({
    "always", "vlm_check", "inventory_has", "timeout",
    "retry_exhausted", "primitive_success", "scene_check",
})


class PolicySpecValidator:
    """Validates a Policy Spec dict and returns a list of error strings."""

    def validate(self, spec: dict[str, Any]) -> list[str]:
        errors: list[str] = []

        for field in ("task", "states", "initial_state", "global_config"):
            if field not in spec:
                errors.append(f"Missing required top-level field: '{field}'")
        if errors:
            return errors

        states: dict = spec["states"]

        if spec["initial_state"] not in states:
            errors.append(
                f"initial_state '{spec['initial_state']}' not found in states"
            )

        reachable: set[str] = set()
        self._find_reachable(spec["initial_state"], states, reachable)

        has_terminal = False

        for name, sdef in states.items():
            if sdef.get("terminal"):
                has_terminal = True
                if "result" not in sdef:
                    errors.append(f"Terminal state '{name}' missing 'result'")
                continue

            if "primitives" not in sdef:
                errors.append(f"State '{name}' missing 'primitives'")
            if "transitions" not in sdef:
                errors.append(f"State '{name}' missing 'transitions'")

            for prim in sdef.get("primitives", []):
                pname = prim.get("name")
                if pname not in KNOWN_PRIMITIVES:
                    errors.append(
                        f"Unknown primitive '{pname}' in state '{name}'"
                    )

            for trans in sdef.get("transitions", []):
                cond = trans.get("condition", {})
                ctype = cond.get("type")
                if ctype not in KNOWN_CONDITIONS:
                    errors.append(
                        f"Unknown condition type '{ctype}' in state '{name}'"
                    )
                for key in ("next_state", "on_true", "on_false"):
                    target = trans.get(key)
                    if target and target not in states:
                        errors.append(
                            f"Transition target '{target}' not in states "
                            f"(from state '{name}')"
                        )

        if not has_terminal:
            errors.append("No terminal state found in spec")

        unreachable = set(states.keys()) - reachable
        for u in unreachable:
            errors.append(f"Warning: state '{u}' unreachable from initial_state")

        return errors

    # ------------------------------------------------------------------

    def _find_reachable(self, name: str, states: dict, visited: set):
        if name in visited or name not in states:
            return
        visited.add(name)
        sdef = states[name]
        for trans in sdef.get("transitions", []):
            for key in ("next_state", "on_true", "on_false"):
                if key in trans:
                    self._find_reachable(trans[key], states, visited)
