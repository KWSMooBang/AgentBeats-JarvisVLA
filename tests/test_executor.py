"""Unit tests for FSMExecutor with a mock VLM."""

import numpy as np
import pytest

from src.executor.fsm_executor import FSMExecutor
from src.executor.vlm_checker import VLMStateChecker


class MockVLMChecker:
    """Deterministic mock that returns pre-configured answers."""

    def __init__(self, yes_no_answers=None, locate_results=None):
        self.yes_no_answers = yes_no_answers or {}
        self.locate_results = locate_results or {}
        self._call_count = 0

    @property
    def call_count(self):
        return self._call_count

    def ask_yes_no(self, image, question):
        self._call_count += 1
        for keyword, answer in self.yes_no_answers.items():
            if keyword.lower() in question.lower():
                return answer
        return False

    def locate_target(self, image, target_type):
        self._call_count += 1
        return self.locate_results.get(target_type)

    def query(self, image, prompt):
        self._call_count += 1
        return ""

    def describe_scene(self, image):
        return "mock scene"


class TestFSMExecutor:
    def _simple_spec(self):
        return {
            "task": "test",
            "global_config": {"max_total_steps": 50, "vlm_check_interval": 10},
            "initial_state": "step1",
            "states": {
                "step1": {
                    "description": "do something",
                    "primitives": [
                        {"name": "move_forward", "params": {"n_steps": 5}}
                    ],
                    "transitions": [
                        {"condition": {"type": "always"}, "next_state": "done"}
                    ],
                },
                "done": {"terminal": True, "description": "finished", "result": "success"},
            },
        }

    def test_simple_execution(self):
        vlm = MockVLMChecker()
        exe = FSMExecutor(policy_spec=self._simple_spec(), vlm_checker=vlm)
        img = np.zeros((64, 64, 3), dtype=np.uint8)

        actions = []
        for _ in range(20):
            a = exe.step(img)
            if a is None:
                break
            actions.append(a)

        assert len(actions) == 5
        assert all(a["forward"] == 1 for a in actions)
        assert exe.finished
        assert exe.result == "success"

    def test_vlm_check_transition(self):
        spec = {
            "task": "test",
            "global_config": {"max_total_steps": 100, "vlm_check_interval": 5},
            "initial_state": "look",
            "states": {
                "look": {
                    "description": "scan",
                    "primitives": [
                        {"name": "noop", "params": {"n_steps": 3}}
                    ],
                    "transitions": [
                        {
                            "condition": {"type": "vlm_check", "query": "Is there a tree?"},
                            "on_true": "done",
                            "on_false": "look",
                        }
                    ],
                },
                "done": {"terminal": True, "description": "ok", "result": "success"},
            },
        }
        vlm = MockVLMChecker(yes_no_answers={"tree": True})
        exe = FSMExecutor(policy_spec=spec, vlm_checker=vlm)
        img = np.zeros((64, 64, 3), dtype=np.uint8)

        for _ in range(20):
            a = exe.step(img)
            if a is None:
                break

        assert exe.finished
        assert exe.result == "success"

    def test_global_timeout(self):
        spec = {
            "task": "test",
            "global_config": {"max_total_steps": 5, "vlm_check_interval": 100},
            "initial_state": "loop",
            "states": {
                "loop": {
                    "description": "infinite loop",
                    "primitives": [
                        {"name": "noop", "params": {"n_steps": 3}}
                    ],
                    "transitions": [
                        {"condition": {"type": "always"}, "next_state": "loop"}
                    ],
                },
                "abort": {"terminal": True, "description": "timeout", "result": "failure"},
            },
        }
        vlm = MockVLMChecker()
        exe = FSMExecutor(policy_spec=spec, vlm_checker=vlm)
        img = np.zeros((64, 64, 3), dtype=np.uint8)

        for _ in range(50):
            a = exe.step(img)
            if a is None:
                break

        assert exe.finished
        assert exe.result == "global_timeout"
