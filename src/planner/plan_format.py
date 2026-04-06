"""Plan format conversion utilities.

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
    "states",
    "steps",
})

DEFAULT_GLOBAL_CONFIG = {
    "max_total_steps": 12000,
    "on_global_timeout": "abort",
}


def _extract_steps_dict(plan: dict[str, Any]) -> Optional[dict[str, Any]]:
    if isinstance(plan.get("steps"), dict):
        return plan["steps"]

    step_items: dict[str, Any] = {}
    for key, value in plan.items():
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
    if ctype in {"always", "timeout"}:
        trans.setdefault("next_state", state_name)

    return trans


def _auto_link_linear_steps(states: dict[str, Any]) -> None:
    """Auto-link sequential step1, step2, ... states for linear execution.
    
    When LLM generates step1, step2, ... link them linearly:
    step1 → step2 → step3 → ... → fallback
    
    This overrides any fallback-only transitions to create proper linear flow.
    """
    step_names: list[str] = []
    for name in sorted(states.keys()):
        if name.startswith("step") and name[4:].isdigit():
            step_names.append(name)
    
    if len(step_names) < 2:
        return
    
    for i, step_name in enumerate(step_names):
        state_def = states[step_name]
        if state_def.get("terminal"):
            continue
        
        transitions = state_def.get("transitions", [])
        if not transitions:
            transitions = []
            state_def["transitions"] = transitions
        
        # Determine next state: next step or fallback
        next_step = step_names[i + 1] if i + 1 < len(step_names) else "fallback"
        
        # Update or add timeout transition
        has_timeout_trans = False
        for trans in transitions:
            cond = trans.get("condition", {})
            if isinstance(cond, dict) and cond.get("type") == "timeout":
                has_timeout_trans = True
                # Override next_state to create linear flow
                trans["next_state"] = next_step
        
        # If no timeout transition, add one linking to next step.
        if not has_timeout_trans:
            transitions.append({
                "condition": {"type": "timeout", "max_steps": 1200},
                "next_state": next_step,
            })


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
        for key in ("next_state", "next"):
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


def to_canonical_plan(plan: dict[str, Any], task_text: str = "") -> dict[str, Any]:
    if not isinstance(plan, dict):
        return {}

    steps = _extract_steps_dict(plan)
    if steps is None:
        return plan

    states: dict[str, Any] = {}
    for step_name, step_def in steps.items():
        if not isinstance(step_def, dict):
            continue
        states[step_name] = _step_to_state(step_name, step_def)

    # Auto-link step1, step2, ... in linear sequence if present.
    _auto_link_linear_steps(states)

    initial_state = plan.get("initial_state")
    if not isinstance(initial_state, str) or initial_state not in states:
        for sname, sdef in states.items():
            if not sdef.get("terminal"):
                initial_state = sname
                break
    if not initial_state:
        initial_state = next(iter(states.keys()), "fallback")

    return {
        "task": task_text.strip() or plan.get("task") or "task",
        "description": plan.get("description") or task_text.strip(),
        "global_config": copy.deepcopy(plan.get("global_config") or DEFAULT_GLOBAL_CONFIG),
        "states": states,
        "initial_state": initial_state,
    }


def canonical_to_simplified_plan(plan: dict[str, Any]) -> dict[str, Any]:
    canonical = to_canonical_plan(plan)
    states = canonical.get("states", {})

    simplified: dict[str, Any] = {
        "task": canonical.get("task", "task"),
    }

    for state_name, state_def in states.items():
        if state_def.get("terminal"):
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
                packed.append(cond)
            step["conditions"] = packed

        simplified[state_name] = step

    return simplified
