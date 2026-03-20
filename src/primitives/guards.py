"""
Layer 3: Termination Guards

All state checks are VLM-based because the agent only receives
task_text and observation images — no inventory or info dict access.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.executor.vlm_checker import VLMStateChecker


class TerminationGuards:
    """VLM-only termination condition evaluators."""

    def __init__(self, vlm_checker: VLMStateChecker):
        self.vlm = vlm_checker

    def check_timeout(self, current_step: int, max_steps: int) -> bool:
        return current_step >= max_steps

    def check_vlm_condition(self, image, query: str) -> bool:
        """Ask the VLM a yes/no question about the current observation."""
        return self.vlm.ask_yes_no(image, query)

    def check_inventory_has_via_vlm(self, image, item: str, count: int = 1) -> bool:
        """Use VLM to infer inventory state from the observation image."""
        query = (
            f"Look at the hotbar/inventory area at the bottom of the screen. "
            f"Does the player have at least {count} {item}?"
        )
        return self.vlm.ask_yes_no(image, query)

    def check_target_visible(self, image, target_type: str) -> bool:
        query = f"Is there a {target_type} visible in this Minecraft screenshot?"
        return self.vlm.ask_yes_no(image, query)

    def check_target_gone(self, image, target_type: str) -> bool:
        return not self.check_target_visible(image, target_type)

    def check_scene_matches(self, image, description: str) -> bool:
        """Generic scene-matching check."""
        query = f"Does this Minecraft screenshot match the following description: {description}"
        return self.vlm.ask_yes_no(image, query)
