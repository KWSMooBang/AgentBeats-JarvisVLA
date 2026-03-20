"""
Layer 0: Atomic Actions

Directly constructs MineStudio env_action dicts.
All higher-level primitives ultimately produce sequences of these.
"""

import copy
import numpy as np
from typing import Any

ENV_NULL_ACTION: dict[str, Any] = {
    "hotbar.1": 0, "hotbar.2": 0, "hotbar.3": 0,
    "hotbar.4": 0, "hotbar.5": 0, "hotbar.6": 0,
    "hotbar.7": 0, "hotbar.8": 0, "hotbar.9": 0,
    "forward": 0, "back": 0, "left": 0, "right": 0,
    "sprint": 0, "sneak": 0, "use": 0, "drop": 0,
    "attack": 0, "jump": 0, "inventory": 0,
    "camera": np.array([0.0, 0.0]),  # [pitch, yaw] in degrees
}


def make_env_action(**overrides) -> dict[str, Any]:
    """Create an env_action dict, overriding specific keys from the null template."""
    action = copy.deepcopy(ENV_NULL_ACTION)
    for key, value in overrides.items():
        if key not in action:
            raise KeyError(f"Unknown env_action key: {key}")
        action[key] = value
    return action
