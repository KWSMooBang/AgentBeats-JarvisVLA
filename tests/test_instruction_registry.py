"""Tests for instruction key canonicalization used by the planner."""

from src.planner.instruction_registry import canonicalize_instruction_key
from src.planner.llm_planner import normalize_instruction_keys


def test_canonicalize_from_alias_sentence():
    key = canonicalize_instruction_key("Please craft a diamond pickaxe")
    assert key == "craft a diamond pickaxe"


def test_canonicalize_drop_key_spacing():
    key = canonicalize_instruction_key("drop: fishing_rod")
    assert key == "drop:fishing_rod"


def test_normalize_instruction_keys_rewrites_state_instruction():
    spec = {
        "task": "test",
        "global_config": {"max_total_steps": 100, "vlm_check_interval": 10},
        "initial_state": "s1",
        "states": {
            "s1": {
                "description": "state 1",
                "instruction": "Please craft a diamond pickaxe",
                "transitions": [{"condition": {"type": "always"}, "next_state": "success"}],
            },
            "success": {"terminal": True, "result": "success"},
            "abort": {"terminal": True, "result": "failure"},
        },
    }

    updated = normalize_instruction_keys(spec)
    assert updated["states"]["s1"]["instruction"] == "craft a diamond pickaxe"
