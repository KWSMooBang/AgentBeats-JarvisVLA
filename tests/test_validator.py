"""Unit tests for PlanValidator."""

import json
import pytest
from pathlib import Path

from src.planner.validator import PlanValidator


@pytest.fixture
def validator():
    return PlanValidator()


@pytest.fixture
def valid_spec():
    return {
        "task": "test",
        "description": "test spec",
        "global_config": {"max_total_steps": 100, "vlm_check_interval": 20},
        "initial_state": "init",
        "states": {
            "init": {
                "description": "start",
                "instruction": "drop:mossy_stone_brick_slab",
                "instruction_type": "simple",
                "transitions": [
                    {"condition": {"type": "always"}, "next_state": "done"}
                ],
            },
            "done": {"terminal": True, "description": "finished", "result": "success"},
        },
    }


class TestPlanValidator:
    def test_valid_spec(self, validator, valid_spec):
        errors = validator.validate(valid_spec)
        assert not any(e for e in errors if not e.startswith("Warning"))

    def test_missing_task(self, validator, valid_spec):
        del valid_spec["task"]
        errors = validator.validate(valid_spec)
        assert any("task" in e for e in errors)

    def test_unknown_condition(self, validator, valid_spec):
        valid_spec["states"]["init"]["transitions"] = [
            {"condition": {"type": "magic"}, "next_state": "done"}
        ]
        errors = validator.validate(valid_spec)
        assert any("magic" in e for e in errors)

    def test_missing_terminal_state(self, validator, valid_spec):
        valid_spec["states"]["done"] = {
            "description": "not terminal",
            "instruction": "drop:mossy_stone_brick_slab",
            "transitions": [{"condition": {"type": "always"}, "next_state": "done"}],
        }
        errors = validator.validate(valid_spec)
        assert any("terminal" in e.lower() for e in errors)

    def test_broken_transition_target(self, validator, valid_spec):
        valid_spec["states"]["init"]["transitions"] = [
            {"condition": {"type": "always"}, "next_state": "nonexistent"}
        ]
        errors = validator.validate(valid_spec)
        assert any("nonexistent" in e for e in errors)

    def test_unreachable_state_warning(self, validator, valid_spec):
        valid_spec["states"]["orphan"] = {
            "description": "unreachable",
            "instruction": "drop:mossy_stone_brick_slab",
            "transitions": [{"condition": {"type": "always"}, "next_state": "done"}],
        }
        errors = validator.validate(valid_spec)
        assert any("orphan" in e for e in errors)

    def test_reference_specs_valid(self, validator):
        plans_dir = Path(__file__).parent.parent / "assets" / "plans"
        for p in plans_dir.glob("*.json"):
            with open(p) as f:
                spec = json.load(f)
            errors = validator.validate(spec)
            real_errors = [e for e in errors if not e.startswith("Warning")]
            assert not real_errors, f"{p.name}: {real_errors}"

    def test_instruction_only_state_valid(self, validator, valid_spec):
        valid_spec["states"]["init"] = {
            "description": "instruction-driven",
            "instruction": "drop:mossy_stone_brick_slab",
            "instruction_type": "normal",
            "transitions": [
                {"condition": {"type": "always"}, "next_state": "done"}
            ],
        }
        errors = validator.validate(valid_spec)
        real_errors = [e for e in errors if not e.startswith("Warning")]
        assert not real_errors

    def test_state_missing_instruction(self, validator, valid_spec):
        valid_spec["states"]["init"] = {
            "description": "invalid",
            "transitions": [
                {"condition": {"type": "always"}, "next_state": "done"}
            ],
        }
        errors = validator.validate(valid_spec)
        assert any("missing required 'instruction'" in e for e in errors)

    def test_instruction_must_be_instructions_key(self, validator, valid_spec):
        valid_spec["states"]["init"]["instruction"] = "Please craft a torch for me"
        errors = validator.validate(valid_spec)
        assert any("not a valid instructions.json key" in e for e in errors)
