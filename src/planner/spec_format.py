"""Policy spec format conversion utilities.

Supports two shapes:
1) Canonical FSM format with top-level `states`.
2) Simplified format with step entries (e.g. `step1`, `step2`) and per-step
   `instruction` + `condition`.
"""

from __future__ import annotations

import copy
from typing import Any, Optional

RESERVED_TOP_LEVEL_KEYS = frozenset({
    "task",
    "description",
    "global_config",
    "initial_state",
    "states",
    "steps",
})

DEFAULT_GLOBAL_CONFIG = {
    "max_total_steps": 600,
    "vlm_check_interval": 30,
    "on_global_timeout": "abort",
}


def _extract_steps_dict(spec: dict[str, Any]) -> Optional[dict[str, Any]]:
    if isinstance(spec.get("steps"), dict):
        return spec["steps"]

    step_items: dict[str, Any] = {}
    for key, value in spec.items():
        if key in RESERVED_TOP_LEVEL_KEYS:
            continue
        if isinstance(value, dict):
            step_items[key] = value

    if not step_items:
        return None
    return step_items


def _normalize_transition(transition: dict[str, Any], state_name: str) -> dict[str, Any]:
    trans = copy.deepcopy(transition)
    cond = trans.get("condition", {})
    if not isinstance(cond, dict):
        cond = {"type": "always"}
        trans["condition"] = cond

    if "next" in trans and "next_state" not in trans:
        trans["next_state"] = trans.pop("next")

    if "next" in cond and "next_state" not in trans:
        trans["next_state"] = cond.pop("next")

    ctype = cond.get("type")
    if ctype in {"always", "timeout", "retry_exhausted"}:
        trans.setdefault("next_state", state_name)
    elif ctype in {"vlm_check", "inventory_has", "scene_check"}:
        trans.setdefault("on_true", state_name)
        trans.setdefault("on_false", state_name)

    return trans


def _step_to_state(step_name: str, step_def: dict[str, Any]) -> dict[str, Any]:
    state: dict[str, Any] = {}

    if step_def.get("terminal"):
        state["terminal"] = True
        state["description"] = step_def.get("description", step_name)
        state["result"] = step_def.get("result", "success" if step_name == "success" else "failure")
        return state

    state["description"] = step_def.get("description", step_name)
    state["instruction"] = step_def.get("instruction", "")
    state["instruction_type"] = step_def.get("instruction_type", "simple")

    if "max_retries" in step_def:
        state["max_retries"] = step_def["max_retries"]

    transitions = step_def.get("transitions")
    if isinstance(transitions, list) and transitions:
        state["transitions"] = [_normalize_transition(t, step_name) for t in transitions if isinstance(t, dict)]
    elif isinstance(step_def.get("condition"), dict):
        trans = {"condition": copy.deepcopy(step_def["condition"])}
        for key in ("next_state", "next", "on_true", "on_false"):
            if key in step_def:
                trans[key] = step_def[key]
        state["transitions"] = [_normalize_transition(trans, step_name)]
    else:
        state["transitions"] = [
            {
                "condition": {"type": "always"},
                "next_state": step_name,
            }
        ]

    return state


def to_canonical_spec(spec: dict[str, Any], task_text: str = "") -> dict[str, Any]:
    if not isinstance(spec, dict):
        return {}

    if isinstance(spec.get("states"), dict):
        return spec

    steps = _extract_steps_dict(spec)
    if steps is None:
        return spec

    states: dict[str, Any] = {}
    for step_name, step_def in steps.items():
        if not isinstance(step_def, dict):
            continue
        states[step_name] = _step_to_state(step_name, step_def)

    if "success" not in states:
        states["success"] = {
            "terminal": True,
            "description": "Task completed",
            "result": "success",
        }
    if "abort" not in states:
        states["abort"] = {
            "terminal": True,
            "description": "Task failed",
            "result": "failure",
        }

    initial_state = spec.get("initial_state")
    if not isinstance(initial_state, str) or initial_state not in states:
        for sname, sdef in states.items():
            if not sdef.get("terminal"):
                initial_state = sname
                break
    if not initial_state:
        initial_state = "success"

    return {
        "task": spec.get("task") or (task_text.strip() or "task"),
        "description": spec.get("description") or task_text.strip(),
        "global_config": copy.deepcopy(spec.get("global_config") or DEFAULT_GLOBAL_CONFIG),
        "states": states,
        "initial_state": initial_state,
    }


def canonical_to_simplified(spec: dict[str, Any]) -> dict[str, Any]:
    canonical = to_canonical_spec(spec)
    states = canonical.get("states", {})

    simplified: dict[str, Any] = {
        "task": canonical.get("task", "task"),
    }

    for state_name, state_def in states.items():
        if state_def.get("terminal"):
            # Keep terminal states internal to canonical runtime format.
            # The planner-facing simplified spec should contain only actionable steps.
            continue

        step: dict[str, Any] = {
            "instruction": state_def.get("instruction", ""),
        }
        if "description" in state_def:
            step["description"] = state_def["description"]
        if "instruction_type" in state_def:
            step["instruction_type"] = state_def["instruction_type"]
        if "max_retries" in state_def:
            step["max_retries"] = state_def["max_retries"]

        transitions = state_def.get("transitions", [])
        if len(transitions) <= 1:
            if transitions:
                trans = copy.deepcopy(transitions[0])
                cond = trans.get("condition", {}) if isinstance(trans.get("condition"), dict) else {"type": "always"}
                if "next_state" in trans:
                    cond["next"] = trans["next_state"]
                if "on_true" in trans:
                    cond["on_true"] = trans["on_true"]
                if "on_false" in trans:
                    cond["on_false"] = trans["on_false"]
                step["condition"] = cond
            else:
                step["condition"] = {"type": "always", "next": state_name}
        else:
            packed: list[dict[str, Any]] = []
            for trans in transitions:
                if not isinstance(trans, dict):
                    continue
                t = copy.deepcopy(trans)
                cond = t.get("condition", {}) if isinstance(t.get("condition"), dict) else {"type": "always"}
                if "next_state" in t:
                    cond["next"] = t["next_state"]
                if "on_true" in t:
                    cond["on_true"] = t["on_true"]
                if "on_false" in t:
                    cond["on_false"] = t["on_false"]
                packed.append(cond)
            step["conditions"] = packed

        simplified[state_name] = step

    return simplified
