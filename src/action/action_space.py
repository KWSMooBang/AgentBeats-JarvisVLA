# src/action/action_space.py
from __future__ import annotations
from typing import Any, Dict, Tuple

import logging
import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)

# MineRL/VPT canonical button order
BUTTONS_ORDER: Tuple[str, ...] = (
    "attack", "back", "forward", "jump", "left", "right", "sneak", "sprint", "use",
    "drop", "inventory",
    "hotbar.1", "hotbar.2", "hotbar.3", "hotbar.4",
    "hotbar.5", "hotbar.6", "hotbar.7", "hotbar.8", "hotbar.9",
)
NUM_BUTTONS = len(BUTTONS_ORDER)

def noop_action() -> Dict[str, Any]:
    """Return a no-op action."""
    return {"buttons": [0], "camera": [60]}
