"""
LLM Planner

Generates a Policy Spec (FSM JSON) from a natural-language task description.
Calls a large language model once at the start of an episode.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from src.planner.prompt_template import (
    PLANNER_SYSTEM_PROMPT,
    REPLAN_ADDENDUM,
    build_planner_prompt,
)
from src.planner.validator import PolicySpecValidator

logger = logging.getLogger(__name__)


class LLMPlanner:
    """
    Calls an OpenAI-compatible LLM to produce a Policy Spec JSON.

    Parameters
    ----------
    api_key : str
        API key (use "EMPTY" for local servers).
    base_url : str
        Base URL of the LLM server (e.g. OpenAI, vllm, Ollama).
    model : str
        Model identifier (e.g. "gpt-4o", "gpt-4o-mini").
    temperature : float
        Sampling temperature.
    max_retries : int
        How many times to retry if validation fails.
    """

    def __init__(
        self,
        api_key: str = "EMPTY",
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
        temperature: float = 0.2,
        max_tokens: int = 4096,
        max_retries: int = 2,
    ):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.validator = PolicySpecValidator()

    def generate_policy_spec(
        self,
        task_text: str,
        feedback: Optional[list[str]] = None,
    ) -> dict:
        """
        Generate and validate a Policy Spec.

        Retries up to ``max_retries`` times if validation fails.
        """
        user_prompt = build_planner_prompt(task_text)
        if feedback:
            user_prompt += REPLAN_ADDENDUM.format(errors="\n".join(f"- {e}" for e in feedback))

        for attempt in range(1 + self.max_retries):
            raw = self._call_llm(user_prompt)
            spec = self._parse_json(raw)
            if spec is None:
                logger.warning("Attempt %d: failed to parse JSON", attempt + 1)
                user_prompt = (
                    build_planner_prompt(task_text)
                    + REPLAN_ADDENDUM.format(
                        errors="- Failed to parse JSON from your response. "
                        "Output ONLY valid JSON, no markdown fences."
                    )
                )
                continue

            errors = self.validator.validate(spec)
            real_errors = [e for e in errors if not e.startswith("Warning")]
            if not real_errors:
                if errors:
                    for w in errors:
                        logger.warning("Validation warning: %s", w)
                logger.info("Policy spec generated successfully (attempt %d)", attempt + 1)
                return spec

            logger.warning(
                "Attempt %d validation errors: %s", attempt + 1, real_errors
            )
            user_prompt = (
                build_planner_prompt(task_text)
                + REPLAN_ADDENDUM.format(errors="\n".join(f"- {e}" for e in real_errors))
            )

        logger.error("All attempts failed; returning last spec as-is")
        return spec  # type: ignore[possibly-undefined]

    # ------------------------------------------------------------------

    def _call_llm(self, user_prompt: str) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.exception("LLM call failed: %s", e)
            return ""

    @staticmethod
    def _parse_json(raw: str) -> Optional[dict]:
        raw = raw.strip()
        # Strip markdown code fences
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return None
