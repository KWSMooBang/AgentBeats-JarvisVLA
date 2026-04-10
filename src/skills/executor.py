"""
Skill Executor

Runs a named skill from the SKILL_LIBRARY step-by-step.
Each call to step() returns one agent-format action dict,
or None when the skill sequence is exhausted.

The executor converts each env-action frame via ActionConverter so
the output is always in the compact Purple Agent format (buttons/camera indices).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from src.action.converter import ActionConverter, noop_agent_action
from src.skills.library import get_skill_sequence, get_skill_description

logger = logging.getLogger(__name__)

# Shared converter instance (stateless, safe to reuse across executors).
_DEFAULT_CONVERTER = ActionConverter()


class SkillExecutor:
    """
    Runs a single skill from SKILL_LIBRARY one step at a time.

    Usage
    -----
    executor = SkillExecutor("pillar_up")
    while not executor.done:
        action = executor.step()   # returns agent-format dict
        env.send(action)
    """

    def __init__(
        self,
        skill_name: str,
        converter: Optional[ActionConverter] = None,
    ):
        self._skill_name = skill_name
        self._converter = converter or _DEFAULT_CONVERTER

        self._sequence: List[Dict[str, Any]] = get_skill_sequence(skill_name)
        self._index: int = 0
        self._total: int = len(self._sequence)

        logger.debug(
            "SkillExecutor created: skill=%s  frames=%d",
            skill_name, self._total,
        )

    # ── public properties ────────────────────────────────────────────────────

    @property
    def skill_name(self) -> str:
        return self._skill_name

    @property
    def done(self) -> bool:
        return self._index >= self._total

    @property
    def steps_taken(self) -> int:
        return self._index

    @property
    def steps_remaining(self) -> int:
        return max(0, self._total - self._index)

    # ── public API ───────────────────────────────────────────────────────────

    def step(self) -> Optional[Dict[str, Any]]:
        """
        Advance one frame and return the agent-format action packet.

        Returns None when the skill is done (all frames consumed).
        The caller should then create a new SkillExecutor for the next skill.
        """
        if self.done:
            return None

        frame = self._sequence[self._index]
        self._index += 1

        try:
            agent_action = self._converter.env_to_agent(frame)
        except Exception:
            logger.exception(
                "SkillExecutor: env_to_agent failed for skill=%s frame=%d",
                self._skill_name, self._index,
            )
            agent_action = noop_agent_action()

        return {
            "__action_format__": "agent",
            "action": agent_action,
        }

    def peek_description(self) -> str:
        """Return the skill description (useful for logging / debugging)."""
        return get_skill_description(self._skill_name)

    def __repr__(self) -> str:
        return (
            f"SkillExecutor(skill={self._skill_name!r}, "
            f"{self._index}/{self._total})"
        )
