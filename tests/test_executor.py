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
    @staticmethod
    def _mock_instruction_runner(image, instruction, instruction_type, state_def):
        return {
            "__action_format__": "agent",
            "action": {"buttons": [1], "camera": [60]},
        }

    def _simple_spec(self):
        return {
            "task": "test",
            "global_config": {"max_total_steps": 50, "vlm_check_interval": 1},
            "initial_state": "step1",
            "states": {
                "step1": {
                    "description": "do something",
                    "instruction": "drop:mossy_stone_brick_slab",
                    "transitions": [
                        {"condition": {"type": "timeout", "max_steps": 5}, "next_state": "done"}
                    ],
                },
                "done": {"terminal": True, "description": "finished", "result": "success"},
            },
        }

    def test_simple_execution(self):
        vlm = MockVLMChecker()
        exe = FSMExecutor(
            plan=self._simple_spec(),
            vlm_checker=vlm,
            instruction_runner=self._mock_instruction_runner,
        )
        img = np.zeros((64, 64, 3), dtype=np.uint8)

        actions = []
        for _ in range(20):
            a = exe.step(img)
            if a is None:
                break
            actions.append(a)

        assert len(actions) == 5
        assert all(a["__action_format__"] == "agent" for a in actions)
        assert exe.finished
        assert exe.result == "success"

    def test_vlm_check_transition(self):
        spec = {
            "task": "test",
            "global_config": {"max_total_steps": 100, "vlm_check_interval": 1},
            "initial_state": "look",
            "states": {
                "look": {
                    "description": "scan",
                    "instruction": "drop:mossy_stone_brick_slab",
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
        exe = FSMExecutor(
            plan=spec,
            vlm_checker=vlm,
            instruction_runner=self._mock_instruction_runner,
        )
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
                    "instruction": "drop:mossy_stone_brick_slab",
                    "transitions": [
                        {"condition": {"type": "always"}, "next_state": "loop"}
                    ],
                },
                "abort": {"terminal": True, "description": "timeout", "result": "failure"},
            },
        }
        vlm = MockVLMChecker()
        exe = FSMExecutor(
            plan=spec,
            vlm_checker=vlm,
            instruction_runner=self._mock_instruction_runner,
        )
        img = np.zeros((64, 64, 3), dtype=np.uint8)

        for _ in range(50):
            a = exe.step(img)
            if a is None:
                break

        assert exe.finished
        assert exe.result == "global_timeout"

    def test_instruction_driven_state(self):
        class MockInstructionRunner:
            def __init__(self):
                self.calls = 0

            def __call__(self, image, instruction, instruction_type, state_def):
                self.calls += 1
                return {
                    "__action_format__": "agent",
                    "action": {"buttons": [1], "camera": [60]},
                }

        spec = {
            "task": "test",
            "global_config": {"max_total_steps": 20, "vlm_check_interval": 1},
            "initial_state": "drop_state",
            "states": {
                "drop_state": {
                    "description": "drop one item",
                    "instruction": "drop:mossy_stone_brick_slab",
                    "instruction_type": "normal",
                    "transitions": [
                        {"condition": {"type": "timeout", "max_steps": 3}, "next_state": "done"}
                    ],
                },
                "done": {"terminal": True, "description": "ok", "result": "success"},
            },
        }
        runner = MockInstructionRunner()
        vlm = MockVLMChecker()
        exe = FSMExecutor(plan=spec, vlm_checker=vlm, instruction_runner=runner)
        img = np.zeros((64, 64, 3), dtype=np.uint8)

        actions = []
        for _ in range(20):
            a = exe.step(img)
            if a is None:
                break
            actions.append(a)

        assert len(actions) == 3
        assert runner.calls == 3
        assert exe.finished
        assert exe.result == "success"
