"""
VLM Skill Runner

Uses a vision-capable LLM to select which skill to execute next,
given the current observation image and instruction.

The runner is responsible only for *selection* — it outputs a skill name.
Actual execution of the skill's action sequence is handled by SkillExecutor.

Design:
- VLM receives: observation image + instruction + recent skill history.
- VLM outputs: {"skill": "<name>", "reason": "<why>"}.
- Invalid or unparseable outputs fall back to a heuristic default.
- Skill re-selection is triggered by the caller (agent) when the current
  SkillExecutor finishes or a stuck condition is detected.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from typing import List, Optional

import numpy as np
from PIL import Image

from src.skills.library import get_skill_names, SKILL_CATEGORIES
from src.skills.prompt_template import (
    SKILL_SELECTOR_SYSTEM_PROMPT,
    build_skill_selector_prompt,
)

logger = logging.getLogger(__name__)

# Fallback skill used when VLM is unavailable or returns garbage.
_FALLBACK_SKILL = "noop"

# Recovery skills cycled through when the agent is stuck on a repeated skill.
# noop is intentionally last — only used if every recovery attempt also loops.
_STUCK_RECOVERY = ["scan_left_right", "step_back", "look_up", "look_down", "noop"]

# Map task-keyword → highlight categories for the skill menu.
_CATEGORY_HINTS = [
    (("craft", "recipe", "furnace", "smelt", "brew", "enchant"),
     ["inventory_gui", "hotbar", "tool_use"]),
    (("build", "place", "pillar", "tower", "wall", "house", "structure"),
     ["building", "hotbar", "camera"]),
    (("combat", "kill", "hunt", "fight", "shoot", "attack"),
     ["combat", "movement", "camera"]),
    (("mine", "dig", "collect", "gather"),
     ["mining", "movement", "camera"]),
    (("use", "drink", "plant", "sleep", "ignite", "throw", "drop"),
     ["tool_use", "hotbar", "camera"]),
    (("decorate", "item_frame", "carpet", "light", "birthday", "clean", "weeds"),
     ["tool_use", "building", "camera"]),
]


def _pick_recovery_skill(history: List[str]) -> str:
    """
    Rotate through _STUCK_RECOVERY to avoid repeating the same recovery action.
    Counts how many consecutive recovery skills are already at the end of history
    and picks the next one in the rotation.
    """
    recent_recoveries = [s for s in reversed(history) if s in _STUCK_RECOVERY]
    idx = len(recent_recoveries) % len(_STUCK_RECOVERY)
    return _STUCK_RECOVERY[idx]


def _guess_highlight_categories(instruction: str) -> Optional[List[str]]:
    text = (instruction or "").lower()
    for keywords, cats in _CATEGORY_HINTS:
        if any(kw in text for kw in keywords):
            return cats
    return None


class VLMSkillRunner:
    """
    Selects the next skill for non-canonical instruction execution.

    Parameters
    ----------
    api_key, base_url, model:
        OpenAI-compatible API credentials.
    temperature:
        Sampling temperature for skill selection (low = deterministic).
    """

    def __init__(
        self,
        api_key: str = "EMPTY",
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
    ):
        self._valid_skills: set[str] = set(get_skill_names())
        self.model = model
        self.temperature = temperature
        self.ready = False

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self.ready = True
            logger.info("VLMSkillRunner initialized (model=%s)", model)
        except Exception as e:
            logger.warning("VLMSkillRunner: OpenAI client unavailable: %s", e)
            self.client = None

    # ── Public API ───────────────────────────────────────────────────────────

    def select_skill(
        self,
        image: np.ndarray,
        instruction: str,
        skill_history: Optional[List[str]] = None,
    ) -> str:
        """
        Ask the VLM to choose the next skill.

        Returns a valid skill name from SKILL_LIBRARY.
        Falls back to a heuristic default if the VLM call fails.
        """
        if not self.ready:
            return self._heuristic_default(instruction, skill_history)

        history = list(skill_history or [])
        highlight = _guess_highlight_categories(instruction)

        user_prompt = build_skill_selector_prompt(
            instruction=instruction,
            skill_history=history,
            highlight_categories=highlight,
            show_only_categories=highlight,  # show only task-relevant categories
        )

        image_url = self._encode_image(image)
        user_content = self._build_user_content(user_prompt, image_url)

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SKILL_SELECTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=self.temperature,
            )
            raw = (resp.choices[0].message.content or "").strip()
            skill = self._parse_skill(raw, history)
            logger.info(
                "VLMSkillRunner selected: %s (instruction=%r)",
                skill, instruction[:60],
            )
            return skill
        except Exception:
            logger.exception("VLMSkillRunner.select_skill failed")
            return self._heuristic_default(instruction, history)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _parse_skill(self, raw: str, history: List[str]) -> str:
        """Extract skill name from VLM JSON output; fall back on parse error."""
        raw = raw.strip()
        # Strip markdown fences if present.
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        parsed: Optional[dict] = None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*?\}", raw, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group())
                except json.JSONDecodeError:
                    pass

        if not isinstance(parsed, dict):
            logger.warning("VLMSkillRunner: could not parse JSON from: %r", raw[:120])
            return _FALLBACK_SKILL

        skill = parsed.get("skill", "")
        if not isinstance(skill, str) or skill not in self._valid_skills:
            logger.warning(
                "VLMSkillRunner: invalid skill name %r; falling back", skill
            )
            return _FALLBACK_SKILL

        # Stuck detection: if same skill selected 3+ consecutive times, rotate
        # through recovery skills instead of falling back to noop immediately.
        if len(history) >= 3 and all(s == skill for s in history[-3:]):
            recovery = _pick_recovery_skill(history)
            logger.info(
                "VLMSkillRunner: stuck on %r — rotating to recovery skill %r",
                skill, recovery,
            )
            return recovery

        return skill

    def _heuristic_default(
        self, instruction: str, history: Optional[List[str]]
    ) -> str:
        """
        Rule-based fallback skill when VLM is unavailable.
        Covers the most common non-canonical instruction patterns.
        """
        text = (instruction or "").lower()
        last = (history or [""])[-1] if history else ""

        if any(kw in text for kw in ("craft", "smelt", "brew", "enchant", "furnace")):
            if "open" not in last and "crafting" not in last:
                return "open_crafting_table"
            return "gui_click"

        if any(kw in text for kw in ("build", "place", "pillar", "tower")):
            if "select_hotbar" not in last:
                return "select_hotbar_1"
            return "place_block_forward"

        if any(kw in text for kw in ("sleep", "bed")):
            return "sleep_in_bed"

        if any(kw in text for kw in ("plant", "wheat", "seed")):
            return "plant_on_ground"

        if any(kw in text for kw in ("drink", "potion")):
            return "drink_use"

        # ── Motion task direct mappings (highest priority for known tasks) ──
        if any(kw in text for kw in ("sky", "heaven", "above")):
            return "motion_look_at_sky"
        if "look" in text and any(kw in text for kw in ("up", "above", "sky")):
            return "motion_look_at_sky"
        if "stack" in text or "acacia" in text or "fence" in text:
            return "motion_stack_fence"
        if "snowball" in text:
            return "motion_throw_snowball"
        if "drop" in text and "item" in text:
            return "motion_drop_item"

        if any(kw in text for kw in ("throw", "toss", "launch")):
            return "throw_item"

        if any(kw in text for kw in ("bow", "shoot", "arrow")):
            return "bow_charge_and_release"

        if any(kw in text for kw in ("combat", "kill", "hunt", "fight", "attack")):
            return "attack_sweep"

        if any(kw in text for kw in ("mine", "dig", "break")):
            return "mine_forward"

        return "noop"

    @staticmethod
    def _encode_image(image: np.ndarray) -> Optional[str]:
        """Encode observation image as a base64 data URL for the API."""
        try:
            if image is None or not isinstance(image, np.ndarray):
                return None
            if image.ndim != 3 or image.shape[2] != 3:
                return None
            pil = Image.fromarray(image.astype(np.uint8), mode="RGB")
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
        except Exception:
            logger.exception("VLMSkillRunner: image encoding failed")
            return None

    @staticmethod
    def _build_user_content(user_prompt: str, image_url: Optional[str]) -> object:
        """Attach image to the user message if available."""
        if image_url is None:
            return user_prompt
        return [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
