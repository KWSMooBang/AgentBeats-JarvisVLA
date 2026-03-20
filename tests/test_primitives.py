"""Unit tests for Layer 0/1 primitives."""

import numpy as np
import pytest

from src.primitives.atomic import make_env_action, ENV_NULL_ACTION
from src.primitives.micro import MicroPrimitives


class TestAtomicActions:
    def test_null_action_has_zero_camera(self):
        a = make_env_action()
        np.testing.assert_array_equal(a["camera"], [0.0, 0.0])
        assert a["forward"] == 0

    def test_override_single_key(self):
        a = make_env_action(forward=1)
        assert a["forward"] == 1
        assert a["back"] == 0

    def test_override_camera(self):
        a = make_env_action(camera=np.array([10.0, -5.0]))
        np.testing.assert_array_equal(a["camera"], [10.0, -5.0])

    def test_unknown_key_raises(self):
        with pytest.raises(KeyError):
            make_env_action(fly=1)


class TestMicroPrimitives:
    def test_noop_length(self):
        actions = MicroPrimitives.noop(5)
        assert len(actions) == 5
        assert all(a["forward"] == 0 for a in actions)

    def test_move_forward(self):
        actions = MicroPrimitives.move_forward(10)
        assert len(actions) == 10
        assert all(a["forward"] == 1 for a in actions)

    def test_jump_forward(self):
        actions = MicroPrimitives.jump_forward(3)
        assert len(actions) == 3
        assert all(a["forward"] == 1 and a["jump"] == 1 for a in actions)

    def test_attack_hold(self):
        actions = MicroPrimitives.attack_hold(7)
        assert len(actions) == 7
        assert all(a["attack"] == 1 for a in actions)

    def test_use_once_two_frames(self):
        actions = MicroPrimitives.use_once()
        assert len(actions) == 2
        assert actions[0]["use"] == 1
        assert actions[1]["use"] == 0

    def test_select_hotbar(self):
        actions = MicroPrimitives.select_hotbar(3)
        assert actions[0]["hotbar.3"] == 1
        assert actions[0]["hotbar.1"] == 0

    def test_select_hotbar_out_of_range(self):
        with pytest.raises(ValueError):
            MicroPrimitives.select_hotbar(0)
        with pytest.raises(ValueError):
            MicroPrimitives.select_hotbar(10)

    def test_look_down(self):
        actions = MicroPrimitives.look_down(angle=45, speed=3.0)
        assert len(actions) == 15
        assert all(a["camera"][0] > 0 for a in actions)

    def test_scan_left_right_symmetric(self):
        actions = MicroPrimitives.scan_left_right(angle=30, speed=3.0)
        total_yaw = sum(a["camera"][1] for a in actions)
        assert abs(total_yaw) < 1e-6

    def test_turn_camera_distributes_evenly(self):
        actions = MicroPrimitives.turn_camera(dx=10.0, dy=-20.0, n_steps=5)
        assert len(actions) == 5
        total_pitch = sum(a["camera"][0] for a in actions)
        total_yaw = sum(a["camera"][1] for a in actions)
        assert abs(total_pitch - 10.0) < 1e-6
        assert abs(total_yaw - (-20.0)) < 1e-6
