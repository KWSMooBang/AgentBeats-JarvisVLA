"""Tests for short/long horizon planning controls."""

from src.planner.prompt_template import build_planner_prompt, classify_task_horizon
from src.planner.llm_planner import validate_horizon_constraints


def _spec_with_non_terminal_count(n: int) -> dict:
    states = {}
    for idx in range(n):
        s = f"s{idx}"
        nxt = f"s{idx + 1}" if idx + 1 < n else "success"
        states[s] = {
            "description": s,
            "instruction": "drop:mossy_stone_brick_slab",
            "transitions": [{"condition": {"type": "always"}, "next_state": nxt}],
        }
    states["success"] = {"terminal": True, "result": "success"}
    states["abort"] = {"terminal": True, "result": "failure"}
    return {
        "task": "t",
        "description": "d",
        "global_config": {"max_total_steps": 100, "vlm_check_interval": 10},
        "initial_state": "s0" if n > 0 else "success",
        "states": states,
    }


def test_classify_short_horizon_default():
    assert classify_task_horizon("Defeat a nearby zombie") == "short"


def test_classify_long_horizon_by_category_marker():
    text = "[task_category: ender_dragon]\nKill the ender dragon"
    assert classify_task_horizon(text) == "long"


def test_build_prompt_includes_short_mode_guidance():
    prompt = build_planner_prompt("[task_horizon: short]\nDrop one item")
    assert "Task horizon: short" in prompt
    assert "Use exactly one non-terminal step" in prompt


def test_validate_horizon_constraints_short_rejects_two_states():
    spec = _spec_with_non_terminal_count(2)
    errors = validate_horizon_constraints(spec, "short")
    assert errors
    assert "exactly one non-terminal step" in errors[0]


def test_validate_horizon_constraints_short_accepts_one_state():
    spec = _spec_with_non_terminal_count(1)
    errors = validate_horizon_constraints(spec, "short")
    assert not errors


def test_validate_horizon_constraints_long_rejects_too_few_states():
    spec = _spec_with_non_terminal_count(1)
    errors = validate_horizon_constraints(spec, "long")
    assert errors
    assert "too few non-terminal states" in errors[0]
