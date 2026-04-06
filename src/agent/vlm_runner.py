"""
OpenAI VLM runner (vision-capable helper).

Provides lightweight hooks for vision-enabled classification, short/long
planning, action generation, and VQA checks. Implementations try to use the
OpenAI Python client when available; when not available the methods return
None so callers may fall back to text-only planner logic.

This module intentionally keeps the runtime surface minimal — it provides
integration points but does not hard-depend on network availability.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

import numpy as np

from src.agent.prompt_template import (
    VLM_ACTION_SYSTEM_PROMPT,
    build_vlm_action_prompt,
)

logger = logging.getLogger(__name__)


class VLMRunner:
    """Minimal VLM runner wrapper.

    Methods return None on failure so callers can fall back to text-only
    planners/heuristics.
    """

    def __init__(
        self,
        api_key: str = "EMPTY",
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini-vision",
        temperature: float = 0.2,
    ):
        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self.model = model
            self.temperature = temperature
            self.ready = True
            logger.info("VLMRunner initialized (model=%s)", model)
        except Exception as e:
            logger.warning("VLMRunner: OpenAI client not available: %s", e)
            self.client = None
            self.model = model
            self.temperature = temperature
            self.ready = False

    # NOTE: The following methods intentionally keep signatures simple and
    # return None on error so the caller (agent) can fallback to planner logic.

    def run_action(self, image: np.ndarray, instruction: str, instruction_type: str = "auto") -> Optional[dict]:
        """Ask the VLM to generate a single agent action for the instruction.

        Must return an action packet compatible with the rest of the system
        (or None to indicate failure).
        """
        if not self.ready:
            return None
        try:
            system = VLM_ACTION_SYSTEM_PROMPT
            user = build_vlm_action_prompt(instruction, instruction_type)
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=self.temperature,
            )
            raw = (resp.choices[0].message.content or "").strip()
            # Expect the model to return JSON like {"__action_format__":"agent","action":{...}}
            parsed = self._parse_json(raw)
            if isinstance(parsed, dict) and parsed.get("__action_format__") == "agent":
                return parsed
        except Exception:
            logger.exception("VLM run_action failed")
        return None

    @staticmethod
    def _parse_json(raw: str) -> Optional[dict]:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.lstrip("`")
            raw = raw.strip()
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            # Try to extract a {...} substring
            import re

            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group())
                    return parsed if isinstance(parsed, dict) else None
                except Exception:
                    pass
        return None
