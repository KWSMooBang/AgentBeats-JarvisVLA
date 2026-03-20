"""
Layer 2: Perceptual Actions

VLM-based target-aware behaviours. These are generator functions that
yield action batches and receive updated observations via .send().

Since the agent has no access to info/inventory — only raw images —
all perception is delegated to the VLM.
"""

from __future__ import annotations

import logging
from typing import Generator, TYPE_CHECKING

import numpy as np

from src.primitives.micro import MicroPrimitives

if TYPE_CHECKING:
    from src.executor.vlm_checker import VLMStateChecker

logger = logging.getLogger(__name__)

# Sentinel yielded when the primitive wants a fresh observation before
# deciding the next batch.  The executor should step the yielded actions
# through the env and .send() the new image back.
ActionBatch = list[dict]


class PerceptualPrimitives:
    """VLM-driven, observation-dependent action generators."""

    def __init__(self, vlm_checker: VLMStateChecker):
        self.vlm = vlm_checker
        self.micro = MicroPrimitives()

    # ------------------------------------------------------------------
    # Target alignment
    # ------------------------------------------------------------------

    def align_to_target(
        self, image: np.ndarray, target_type: str, max_attempts: int = 10
    ) -> Generator[ActionBatch, np.ndarray, dict]:
        """
        Iteratively turn camera until the target is roughly centered.

        Yields action batches; receives updated image via .send().
        Returns {"success": bool} when finished.
        """
        for _ in range(max_attempts):
            loc = self.vlm.locate_target(image, target_type)

            if loc is None:
                actions = self.micro.scan_360(speed=2.0)
                image = yield actions
                continue

            dx, dy = loc["offset_x"], loc["offset_y"]
            if abs(dx) < 0.08 and abs(dy) < 0.08:
                return {"success": True}

            cam_dx = dy * 15.0
            cam_dy = dx * 15.0
            actions = self.micro.turn_camera(cam_dx, cam_dy, n_steps=3)
            image = yield actions

        return {"success": False}

    # ------------------------------------------------------------------
    # Approach
    # ------------------------------------------------------------------

    def approach_target(
        self, image: np.ndarray, target_type: str, max_steps: int = 80
    ) -> Generator[ActionBatch, np.ndarray, dict]:
        """Walk toward a target, periodically re-aligning."""
        steps_taken = 0

        while steps_taken < max_steps:
            loc = self.vlm.locate_target(image, target_type)

            if loc is None:
                actions = self.micro.scan_360(speed=2.0)
                image = yield actions
                steps_taken += len(actions)
                continue

            # Re-align if off-center
            if abs(loc["offset_x"]) > 0.15 or abs(loc["offset_y"]) > 0.15:
                cam_dx = loc["offset_y"] * 12.0
                cam_dy = loc["offset_x"] * 12.0
                align_actions = self.micro.turn_camera(cam_dx, cam_dy, n_steps=2)
                image = yield align_actions
                steps_taken += len(align_actions)
                continue

            # Close enough if target is large in frame
            if loc.get("size", 0) > 0.25:
                return {"success": True}

            move_actions = self.micro.move_forward(n_steps=10)
            image = yield move_actions
            steps_taken += len(move_actions)

        return {"success": False}

    # ------------------------------------------------------------------
    # Mining
    # ------------------------------------------------------------------

    def mine_target_block(
        self, image: np.ndarray, target_type: str, max_hits: int = 60
    ) -> Generator[ActionBatch, np.ndarray, dict]:
        """Align to block then hold attack."""
        # First align
        gen = self.align_to_target(image, target_type, max_attempts=6)
        try:
            batch = next(gen)
            while True:
                image = yield batch
                batch = gen.send(image)
        except StopIteration as e:
            result = e.value
            if not result["success"]:
                return {"success": False, "reason": "align_failed"}

        # Hold attack while looking at the block
        attack_actions = self.micro.attack_hold(n_steps=max_hits)
        image = yield attack_actions

        # Check if the block is gone
        still_there = self.vlm.ask_yes_no(
            image,
            f"Is there still a {target_type} block directly in front of the player?",
        )
        return {"success": not still_there}

    # ------------------------------------------------------------------
    # Entity combat
    # ------------------------------------------------------------------

    def attack_target_entity(
        self, image: np.ndarray, target_type: str, max_swings: int = 80
    ) -> Generator[ActionBatch, np.ndarray, dict]:
        """Chase and attack a mob."""
        swings = 0
        while swings < max_swings:
            loc = self.vlm.locate_target(image, target_type)

            if loc is None:
                gone = self.vlm.ask_yes_no(
                    image, f"Has the {target_type} been killed or disappeared?"
                )
                if gone:
                    return {"success": True}
                scan = self.micro.scan_360(speed=2.0)
                image = yield scan
                swings += len(scan)
                continue

            if abs(loc["offset_x"]) > 0.12 or abs(loc["offset_y"]) > 0.12:
                cam_dx = loc["offset_y"] * 10.0
                cam_dy = loc["offset_x"] * 10.0
                align = self.micro.turn_camera(cam_dx, cam_dy, n_steps=2)
                image = yield align
                swings += len(align)
                continue

            atk = self.micro.attack_forward(n_steps=5)
            image = yield atk
            swings += len(atk)

        return {"success": False}

    # ------------------------------------------------------------------
    # Search and face
    # ------------------------------------------------------------------

    def search_and_face(
        self, image: np.ndarray, target_type: str, timeout: int = 120
    ) -> Generator[ActionBatch, np.ndarray, dict]:
        """Scan surroundings, walk forward, repeat — until target found."""
        steps = 0
        while steps < timeout:
            loc = self.vlm.locate_target(image, target_type)
            if loc is not None:
                gen = self.align_to_target(image, target_type, max_attempts=5)
                try:
                    batch = next(gen)
                    while True:
                        image = yield batch
                        steps += len(batch)
                        batch = gen.send(image)
                except StopIteration as e:
                    return e.value

            scan = self.micro.scan_360(speed=2.0)
            image = yield scan
            steps += len(scan)

            walk = self.micro.sprint_forward(n_steps=20)
            image = yield walk
            steps += len(walk)

        return {"success": False}

    # ------------------------------------------------------------------
    # Navigate (longer-range)
    # ------------------------------------------------------------------

    def navigate_to_target(
        self, image: np.ndarray, target_type: str, max_steps: int = 300
    ) -> Generator[ActionBatch, np.ndarray, dict]:
        """Repeatedly search → approach until close."""
        steps = 0
        while steps < max_steps:
            gen = self.search_and_face(image, target_type, timeout=100)
            try:
                batch = next(gen)
                while True:
                    image = yield batch
                    steps += len(batch)
                    batch = gen.send(image)
            except StopIteration as e:
                if not e.value["success"]:
                    return {"success": False}

            gen2 = self.approach_target(image, target_type, max_steps=60)
            try:
                batch = next(gen2)
                while True:
                    image = yield batch
                    steps += len(batch)
                    batch = gen2.send(image)
            except StopIteration as e:
                if e.value["success"]:
                    return {"success": True}

        return {"success": False}
