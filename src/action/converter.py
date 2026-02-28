# src/action/converter.py
"""Action space conversion utilities for JarvisVLA Purple Agent.

JarvisVLA uses different action encoding than MineStudio:
- JarvisVLA buttons: 21 button combinations (bases=[21])
- JarvisVLA camera: 441 positions (bases=[21, 21] -> 21x21 grid)
- MineStudio buttons: 2,304 combinations (CameraHierarchicalMapping)
- MineStudio camera: 121 positions (11x11 grid)
"""

from __future__ import annotations
from typing import Any, Dict
import logging

from src.action.action_space import noop_action

logger = logging.getLogger(__name__)


class ActionConverter:
    """Convert between JarvisVLA and Purple Agent action formats."""
    
    JARVISVLA_CAMERA_BINS = 441  # 21x21 grid
    PURPLE_CAMERA_BINS = 121      # 11x11 grid
    
    @staticmethod
    def jarvisvla_to_purple(action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert JarvisVLA action to Purple Agent format.
        
        Args:
            action: JarvisVLA format {"buttons": int or list, "camera": int or list}
                    buttons: 0-20 (single button combination)
                    camera: 0-440 (21x21 grid position)
        
        Returns:
            Purple format: {"buttons": [int], "camera": [int]}
                           buttons: [0-2303] (button combination index)
                           camera: [0-120] (11x11 grid position)
        
        Raises:
            ValueError: If action is None or invalid format
        """
        if action is None:
            logger.warning("None action provided, using noop")
            return noop_action()
        
        if not isinstance(action, dict):
            logger.error("Invalid action type: %s, using noop", type(action))
            return noop_action()
        
        # Validate required keys
        if "buttons" not in action or "camera" not in action:
            logger.error("Action missing required keys: %s, using noop", action)
            return noop_action()
        
        # Extract and convert buttons
        buttons = action["buttons"]
        if not isinstance(buttons, (list, tuple)):
            buttons = [int(buttons)]
        else:
            buttons = [int(b) for b in buttons]
        
        # Extract and convert camera
        camera = action["camera"]
        if not isinstance(camera, (list, tuple)):
            # Scale camera: 441 bins (21x21) -> 121 bins (11x11)
            # Linear interpolation: camera_scaled = camera_value * 120 / 440
            camera_value = int(camera)
            
            # Clamp to valid range
            camera_value = max(0, min(440, camera_value))
            
            # Scale to Purple range
            camera_scaled = int(camera_value * 120 / 440)
            camera = [camera_scaled]
        else:
            camera = [int(c) for c in camera]
        
        return {
            "buttons": buttons,
            "camera": camera
        }
    
    @staticmethod
    def get_jarvisvla_camera_center() -> int:
        """Get center camera position for JarvisVLA (21x21 grid).
        
        Returns:
            220: Center position (10, 10) in 21x21 grid = 10*21 + 10
        """
        return 220
    
    @staticmethod
    def get_purple_camera_center() -> int:
        """Get center camera position for Purple (11x11 grid).
        
        Returns:
            60: Center position (5, 5) in 11x11 grid = 5*11 + 5
        """
        return 60
