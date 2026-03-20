"""
Layer 1: Micro Behaviors

Single-purpose action sequences repeated for N frames.
Each method returns a list of env_action dicts ready for env.step().
"""

import numpy as np
from src.primitives.atomic import make_env_action


class MicroPrimitives:
    """Deterministic low-level action sequences."""

    # ------------------------------------------------------------------
    # No-op
    # ------------------------------------------------------------------

    @staticmethod
    def noop(n_steps: int = 1) -> list[dict]:
        return [make_env_action() for _ in range(n_steps)]

    # ------------------------------------------------------------------
    # Camera control
    # ------------------------------------------------------------------

    @staticmethod
    def turn_camera(dx: float, dy: float, n_steps: int = 1) -> list[dict]:
        """Rotate camera. dx = pitch delta, dy = yaw delta (degrees total)."""
        step_dx = dx / max(n_steps, 1)
        step_dy = dy / max(n_steps, 1)
        return [
            make_env_action(camera=np.array([step_dx, step_dy]))
            for _ in range(n_steps)
        ]

    @staticmethod
    def look_down(angle: float = 45.0, speed: float = 3.0) -> list[dict]:
        steps = max(int(angle / speed), 1)
        return [make_env_action(camera=np.array([speed, 0.0])) for _ in range(steps)]

    @staticmethod
    def look_up(angle: float = 45.0, speed: float = 3.0) -> list[dict]:
        steps = max(int(angle / speed), 1)
        return [make_env_action(camera=np.array([-speed, 0.0])) for _ in range(steps)]

    @staticmethod
    def scan_left_right(angle: float = 60.0, speed: float = 3.0) -> list[dict]:
        """Sweep camera right → left → back to center."""
        steps_per_dir = max(int(angle / speed), 1)
        actions: list[dict] = []
        for _ in range(steps_per_dir):
            actions.append(make_env_action(camera=np.array([0.0, speed])))
        for _ in range(steps_per_dir * 2):
            actions.append(make_env_action(camera=np.array([0.0, -speed])))
        for _ in range(steps_per_dir):
            actions.append(make_env_action(camera=np.array([0.0, speed])))
        return actions

    @staticmethod
    def scan_360(speed: float = 2.0) -> list[dict]:
        """Turn 360° in one direction (yaw only). Slow sweep to scan surroundings."""
        total_angle = 360.0
        steps = max(int(total_angle / speed), 1)
        return [
            make_env_action(camera=np.array([0.0, speed]))
            for _ in range(steps)
        ]

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    @staticmethod
    def move_forward(n_steps: int = 20) -> list[dict]:
        return [make_env_action(forward=1) for _ in range(n_steps)]

    @staticmethod
    def move_backward(n_steps: int = 10) -> list[dict]:
        return [make_env_action(back=1) for _ in range(n_steps)]

    @staticmethod
    def strafe_left(n_steps: int = 10) -> list[dict]:
        return [make_env_action(left=1) for _ in range(n_steps)]

    @staticmethod
    def strafe_right(n_steps: int = 10) -> list[dict]:
        return [make_env_action(right=1) for _ in range(n_steps)]

    @staticmethod
    def jump_forward(n_steps: int = 5) -> list[dict]:
        return [make_env_action(forward=1, jump=1) for _ in range(n_steps)]

    @staticmethod
    def sprint_forward(n_steps: int = 20) -> list[dict]:
        return [make_env_action(forward=1, sprint=1) for _ in range(n_steps)]

    # ------------------------------------------------------------------
    # Combat / Interaction
    # ------------------------------------------------------------------

    @staticmethod
    def attack_hold(n_steps: int = 10) -> list[dict]:
        return [make_env_action(attack=1) for _ in range(n_steps)]

    @staticmethod
    def attack_forward(n_steps: int = 10) -> list[dict]:
        return [make_env_action(attack=1, forward=1) for _ in range(n_steps)]

    @staticmethod
    def use_once() -> list[dict]:
        return [make_env_action(use=1), make_env_action()]

    @staticmethod
    def use_hold(n_steps: int = 5) -> list[dict]:
        return [make_env_action(use=1) for _ in range(n_steps)]

    # ------------------------------------------------------------------
    # Inventory / Hotbar
    # ------------------------------------------------------------------

    @staticmethod
    def select_hotbar(slot: int) -> list[dict]:
        if not 1 <= slot <= 9:
            raise ValueError(f"Hotbar slot must be 1-9, got {slot}")
        return [make_env_action(**{f"hotbar.{slot}": 1}), make_env_action()]

    @staticmethod
    def open_inventory() -> list[dict]:
        return [make_env_action(inventory=1), make_env_action()]

    @staticmethod
    def close_inventory() -> list[dict]:
        return [make_env_action(inventory=1), make_env_action()]

    @staticmethod
    def drop_item() -> list[dict]:
        return [make_env_action(drop=1), make_env_action()]

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    @staticmethod
    def sneak_toggle(n_steps: int = 1) -> list[dict]:
        return [make_env_action(sneak=1) for _ in range(n_steps)]

    @staticmethod
    def jump(n_steps: int = 1) -> list[dict]:
        return [make_env_action(jump=1) for _ in range(n_steps)]
