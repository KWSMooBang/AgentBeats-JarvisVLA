"""Tests for simplified plan compatibility."""

import numpy as np

from src.executor.fsm_executor import FSMExecutor
from src.planner.spec_format import canonical_to_simplified, to_canonical_spec
from src.planner.validator import PlanValidator


class _MockVLMChecker:
    def ask_yes_no(self, image, question):
        return True


def _mock_instruction_runner(image, instruction, instruction_type, state_def):
    return {
        "__action_format__": "agent",
        "action": {"buttons": [0], "camera": [60]},
    }


def test_to_canonical_from_simplified():
    simplified = {
        "task": "combat",
        "step1": {
            "instruction": "kill_entity:skeleton",
            "condition": {
                "type": "vlm_check",
                "query": "Is skeleton dead?",
                "on_true": "success",
                "on_false": "step1",
            },
        },
        "success": {"terminal": True, "result": "success"},
        "abort": {"terminal": True, "result": "failure"},
    }

    canonical = to_canonical_spec(simplified)
    assert "states" in canonical
    assert canonical["initial_state"] == "step1"
    assert canonical["states"]["step1"]["instruction"] == "kill_entity:skeleton"


def test_validator_accepts_simplified_spec():
    simplified = {
        "task": "drop",
        "step1": {
            "instruction": "drop:fishing_rod",
            "condition": {"type": "always", "next": "success"},
        },
        "success": {"terminal": True, "result": "success"},
        "abort": {"terminal": True, "result": "failure"},
    }

    errors = PlanValidator().validate(simplified)
    real_errors = [e for e in errors if not e.startswith("Warning")]
    assert not real_errors


def test_executor_runs_simplified_spec():
    simplified = {
        "task": "drop",
        "step1": {
            "instruction": "drop:fishing_rod",
            "condition": {"type": "timeout", "max_steps": 1, "next": "success"},
        },
        "success": {"terminal": True, "result": "success"},
        "abort": {"terminal": True, "result": "failure"},
    }

    exe = FSMExecutor(
        plan=simplified,
        vlm_checker=_MockVLMChecker(),
        instruction_runner=_mock_instruction_runner,
    )
    img = np.zeros((32, 32, 3), dtype=np.uint8)

    action = exe.step(img)
    assert action is not None
    assert action["__action_format__"] == "agent"


def test_canonical_to_simplified_roundtrip_has_step_key():
    canonical = {
        "task": "t",
        "description": "d",
        "global_config": {"max_total_steps": 100, "vlm_check_interval": 10},
        "initial_state": "step1",
        "states": {
            "step1": {
                "description": "s1",
                "instruction": "mine_block:barrel",
                "transitions": [{"condition": {"type": "always"}, "next_state": "success"}],
            },
            "success": {"terminal": True, "result": "success"},
            "abort": {"terminal": True, "result": "failure"},
        },
    }

    simplified = canonical_to_simplified(canonical)
    assert "step1" in simplified
    assert simplified["step1"]["instruction"] == "mine_block:barrel"
