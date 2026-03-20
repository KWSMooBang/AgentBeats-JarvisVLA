"""Unit tests for PolicySpecValidator."""

import json
import pytest
from pathlib import Path

from src.planner.validator import PolicySpecValidator


@pytest.fixture
def validator():
    return PolicySpecValidator()


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
                "primitives": [{"name": "noop", "params": {"n_steps": 1}}],
                "transitions": [
                    {"condition": {"type": "always"}, "next_state": "done"}
                ],
            },
            "done": {"terminal": True, "description": "finished", "result": "success"},
        },
    }


class TestPolicySpecValidator:
    def test_valid_spec(self, validator, valid_spec):
        errors = validator.validate(valid_spec)
        assert not any(e for e in errors if not e.startswith("Warning"))

    def test_missing_task(self, validator, valid_spec):
        del valid_spec["task"]
        errors = validator.validate(valid_spec)
        assert any("task" in e for e in errors)

    def test_unknown_primitive(self, validator, valid_spec):
        valid_spec["states"]["init"]["primitives"] = [
            {"name": "fly_to_moon", "params": {}}
        ]
        errors = validator.validate(valid_spec)
        assert any("fly_to_moon" in e for e in errors)

    def test_unknown_condition(self, validator, valid_spec):
        valid_spec["states"]["init"]["transitions"] = [
            {"condition": {"type": "magic"}, "next_state": "done"}
        ]
        errors = validator.validate(valid_spec)
        assert any("magic" in e for e in errors)

    def test_missing_terminal_state(self, validator, valid_spec):
        valid_spec["states"]["done"] = {
            "description": "not terminal",
            "primitives": [{"name": "noop", "params": {"n_steps": 1}}],
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
            "primitives": [{"name": "noop", "params": {"n_steps": 1}}],
            "transitions": [{"condition": {"type": "always"}, "next_state": "done"}],
        }
        errors = validator.validate(valid_spec)
        assert any("orphan" in e for e in errors)

    def test_reference_specs_valid(self, validator):
        specs_dir = Path(__file__).parent.parent / "assets" / "policy_specs"
        for p in specs_dir.glob("*.json"):
            with open(p) as f:
                spec = json.load(f)
            errors = validator.validate(spec)
            real_errors = [e for e in errors if not e.startswith("Warning")]
            assert not real_errors, f"{p.name}: {real_errors}"
