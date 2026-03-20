"""
Action Converter

Converts between env_action (expanded, 20 buttons + camera degrees)
and agent_action (compact Purple Agent format: button index + camera bin).

When minestudio is available, uses its built-in ActionTransformer and
CameraHierarchicalMapping for exact conversion.  Otherwise falls back
to a pure-numpy implementation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ======================================================================
# Purple Agent noop
# ======================================================================

_CAMERA_CENTER = 60  # center of 11×11 grid


def noop_agent_action() -> Dict[str, Any]:
    """No-op in Purple Agent compact format."""
    return {"buttons": [0], "camera": [_CAMERA_CENTER]}


# ======================================================================
# Button mapping (env key order must match minestudio Buttons.ALL)
# ======================================================================

BUTTON_KEYS = [
    "attack", "back", "drop", "forward", "hotbar.1", "hotbar.2",
    "hotbar.3", "hotbar.4", "hotbar.5", "hotbar.6", "hotbar.7",
    "hotbar.8", "hotbar.9", "inventory", "jump", "left", "right",
    "sneak", "sprint", "use",
]


class ActionConverter:
    """
    Env-action ↔ Agent-action converter.

    Attempts to use minestudio's native converters if the package is
    installed.  Falls back to a self-contained numpy implementation.
    """

    def __init__(
        self,
        camera_binsize: int = 2,
        camera_maxval: int = 10,
        camera_mu: float = 10.0,
        camera_quantization_scheme: str = "mu_law",
    ):
        self.camera_binsize = camera_binsize
        self.camera_maxval = camera_maxval
        self.camera_mu = camera_mu
        self.camera_quantization_scheme = camera_quantization_scheme
        self.n_camera_bins = 2 * camera_maxval // camera_binsize + 1  # 11

        self._use_minestudio = False
        try:
            from minestudio.utils.vpt_lib.actions import ActionTransformer
            from minestudio.utils.vpt_lib.action_mapping import CameraHierarchicalMapping

            self._action_transformer = ActionTransformer(
                camera_binsize=camera_binsize,
                camera_maxval=camera_maxval,
                camera_mu=camera_mu,
                camera_quantization_scheme=camera_quantization_scheme,
            )
            self._action_mapper = CameraHierarchicalMapping(
                n_camera_bins=self.n_camera_bins
            )
            self._use_minestudio = True
            logger.info("ActionConverter: using minestudio backend")
        except ImportError:
            logger.info("ActionConverter: minestudio not found, using fallback")

    # ------------------------------------------------------------------
    # env → agent
    # ------------------------------------------------------------------

    def env_to_agent(self, env_action: Dict[str, Any]) -> Dict[str, Any]:
        if env_action is None:
            return noop_agent_action()

        if self._use_minestudio:
            return self._env_to_agent_minestudio(env_action)
        return self._env_to_agent_fallback(env_action)

    def _env_to_agent_minestudio(self, env_action: dict) -> dict:
        try:
            policy_action = self._action_transformer.env2policy(env_action)
            for k in ("buttons", "camera"):
                v = policy_action[k]
                if isinstance(v, np.ndarray) and v.ndim == 1:
                    policy_action[k] = v.reshape(1, -1)
            agent_action = self._action_mapper.from_factored(policy_action)
            return {
                "buttons": np.array([int(agent_action["buttons"].flat[0])]),
                "camera": np.array([int(agent_action["camera"].flat[0])]),
            }
        except Exception as e:
            logger.exception("minestudio conversion failed: %s", e)
            return noop_agent_action()

    def _env_to_agent_fallback(self, env_action: dict) -> dict:
        """Pure-numpy fallback without minestudio dependency."""
        # Buttons → single index via binary encoding
        bits = np.array([env_action.get(k, 0) for k in BUTTON_KEYS], dtype=np.int32)
        button_idx = int(np.packbits(np.pad(bits, (0, 24 - len(bits)))[:24], bitorder="little")
                         .view(np.uint32)[0]) if bits.any() else 0
        button_idx = min(button_idx, 2303)

        # Camera → quantise to 11×11 grid
        cam = env_action.get("camera", np.array([0.0, 0.0]))
        if isinstance(cam, list):
            cam = np.array(cam, dtype=np.float64)
        pitch, yaw = float(cam[0]), float(cam[1])

        def _quantise(val: float) -> int:
            val = np.clip(val, -self.camera_maxval, self.camera_maxval)
            bin_idx = int(round((val + self.camera_maxval) / self.camera_binsize))
            return np.clip(bin_idx, 0, self.n_camera_bins - 1)

        pitch_bin = _quantise(pitch)
        yaw_bin = _quantise(yaw)
        camera_idx = pitch_bin * self.n_camera_bins + yaw_bin

        return {"buttons": [button_idx], "camera": [int(camera_idx)]}

    # ------------------------------------------------------------------
    # agent → env  (for completeness / testing)
    # ------------------------------------------------------------------

    def agent_to_env(self, agent_action: Dict[str, Any]) -> Dict[str, Any]:
        if self._use_minestudio:
            return self._agent_to_env_minestudio(agent_action)
        return self._agent_to_env_fallback(agent_action)

    def _agent_to_env_minestudio(self, agent_action: dict) -> dict:
        try:
            compact = {
                "buttons": np.array([[agent_action["buttons"][0]]]),
                "camera": np.array([[agent_action["camera"][0]]]),
            }
            factored = self._action_mapper.to_factored(compact)
            return self._action_transformer.policy2env(factored)
        except Exception as e:
            logger.exception("minestudio reverse conversion failed: %s", e)
            from src.primitives.atomic import make_env_action
            return make_env_action()

    def _agent_to_env_fallback(self, agent_action: dict) -> dict:
        from src.primitives.atomic import make_env_action
        return make_env_action()
