"""
Plan Validator

Validates the structural and semantic correctness of FSM JSON
produced by the LLM Planner before execution.
"""

from __future__ import annotations

import difflib
import logging
from typing import Any

from src.planner.instruction_registry import (
    canonicalize_instruction_key,
    get_instruction_keys,
    is_strict_instruction_key,
    instructions_registry_available,
)
from src.planner.spec_format import to_canonical_spec

logger = logging.getLogger(__name__)

KNOWN_CONDITIONS = frozenset({
    "always", "vlm_check", "inventory_has", "timeout",
    "retry_exhausted", "scene_check",
})

KNOWN_INSTRUCTION_TYPES = frozenset({"auto", "simple", "normal", "recipe"})


class PlanValidator:
    """Validates a plan dict and returns a list of error strings."""

    def validate(self, spec: dict[str, Any]) -> list[str]:
        errors: list[str] = []

        spec = to_canonical_spec(spec)

        registry_ok = instructions_registry_available()
        instruction_keys = get_instruction_keys() if registry_ok else set()
        if not registry_ok:
            errors.append("instructions.json registry is unavailable; cannot validate instruction keys")

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

            is_fallback_state = name == "fallback"

            has_instruction = "instruction" in sdef
            if not has_instruction:
                errors.append(f"State '{name}' missing required 'instruction'")
            else:
                instruction = sdef.get("instruction")
                if not isinstance(instruction, str) or not instruction.strip():
                    errors.append(
                        f"State '{name}' has invalid 'instruction' (must be non-empty string)"
                    )
                elif registry_ok and not is_fallback_state:
                    canonical = canonicalize_instruction_key(instruction)
                    if canonical is None or canonical not in instruction_keys:
                        suggestions = difflib.get_close_matches(
                            instruction,
                            sorted(instruction_keys),
                            n=3,
                            cutoff=0.65,
                        )
                        hint = f" Did you mean: {', '.join(suggestions)}" if suggestions else ""
                        errors.append(
                            f"State '{name}' instruction '{instruction}' is not a valid instructions.json key.{hint}"
                        )
                    elif not is_strict_instruction_key(canonical):
                        errors.append(
                            f"State '{name}' instruction '{instruction}' must use strict prefix:item format"
                        )
                    elif instruction != canonical:
                        errors.append(
                            f"State '{name}' instruction '{instruction}' must use canonical key '{canonical}'"
                        )

            if "instruction_type" not in sdef:
                errors.append(f"State '{name}' missing required 'instruction_type'")
            else:
                instruction_type = sdef.get("instruction_type")
                if instruction_type not in KNOWN_INSTRUCTION_TYPES:
                    errors.append(
                        f"Unknown instruction_type '{instruction_type}' in state '{name}'"
                    )

            if "transitions" not in sdef:
                errors.append(f"State '{name}' missing 'transitions'")

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
