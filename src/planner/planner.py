"""
LLM Planner

Routing policy:
1) Classify horizon via LLM
2) short  -> return direct instruction + instruction_type
3) long   -> generate staged plan (FSM-like simplified format)

Fallback behavior (long mode):
- A dedicated fallback step is always ensured.
- fallback.instruction is always task_text.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from typing import Optional

import numpy as np
from PIL import Image

from src.planner.prompt_template import (
    HORIZON_SYSTEM_PROMPT,
    SHORT_DIRECTIVE_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    VQA_SUBGOAL_SYSTEM_PROMPT,
    REPLAN_ADDENDUM,
    build_horizon_prompt,
    build_short_directive_prompt,
    build_planner_prompt,
    build_vqa_subgoal_prompt,
    fallback_classify_task_horizon,
)
from src.planner.instruction_registry import (
    canonicalize_instruction_key,
    canonicalize_strict_instruction_key,
    get_strict_instruction_keys,
)
from src.planner.plan_format import canonical_to_simplified_plan, to_canonical_plan
from src.planner.validator import PlanValidator

logger = logging.getLogger(__name__)

DEFAULT_FALLBACK_MAX_STEPS = 1200


def _extract_task_tokens(task_text: str) -> list[str]:
    text = (task_text or "").lower().replace("-", " ").replace("_", " ")
    words = re.findall(r"[a-z0-9]+", text)
    if not words:
        return []

    tokens: list[str] = []
    stop_words = {"a", "an", "the", "to", "from", "with", "and", "or", "of"}

    for word in words:
        if word not in stop_words:
            tokens.append(word)

    # Add joined n-grams to capture items like diamond_pickaxe.
    for n in (2, 3):
        if len(words) < n:
            continue
        for i in range(0, len(words) - n + 1):
            chunk = words[i : i + n]
            if all(w in stop_words for w in chunk):
                continue
            tokens.append("_".join(chunk))

    # Keep order while removing duplicates.
    deduped: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token and token not in seen:
            seen.add(token)
            deduped.append(token)
    return deduped


def _contains_any(text: str, words: tuple[str, ...]) -> bool:
    lower = (text or "").lower()
    return any(w in lower for w in words)


def _singularize_token(token: str) -> str:
    token = (token or "").strip().lower()
    if not token:
        return token

    # Common irregular plural in Minecraft entities.
    if token == "endermen":
        return "enderman"

    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("es") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def _token_variants(token: str) -> list[str]:
    """Generate robust variants for malformed short-directive targets."""
    base = (token or "").strip().lower().replace(" ", "_")
    if not base:
        return []

    variants: list[str] = [base]

    # Extract likely object suffixes from labels like combat_skeletons, task_enderman.
    parts = [p for p in base.split("_") if p]
    if len(parts) >= 1:
        variants.append(parts[-1])
    if len(parts) >= 2:
        variants.append("_".join(parts[-2:]))

    # Add singular forms for each variant.
    variants.extend(_singularize_token(v) for v in list(variants))

    deduped: list[str] = []
    seen: set[str] = set()
    for v in variants:
        if v and v not in seen:
            seen.add(v)
            deduped.append(v)
    return deduped


def _guess_strict_instruction_from_task(task_text: str, strict_keys: set[str]) -> Optional[str]:
    tokens = _extract_task_tokens(task_text)
    if not tokens:
        return None

    prefers_combat = _contains_any(task_text, ("combat", "kill", "defeat", "hunt"))
    prefers_craft = _contains_any(task_text, ("craft", "make", "recipe"))
    prefers_mine = _contains_any(task_text, ("mine", "collect", "gather"))

    prefix_order = ["kill_entity", "craft_item", "mine_block", "pickup", "use_item", "drop"]
    if prefers_craft:
        prefix_order = ["craft_item", "kill_entity", "mine_block", "pickup", "use_item", "drop"]
    elif prefers_mine:
        prefix_order = ["mine_block", "kill_entity", "craft_item", "pickup", "use_item", "drop"]
    elif prefers_combat:
        prefix_order = ["kill_entity", "mine_block", "craft_item", "pickup", "use_item", "drop"]

    for token in tokens:
        for variant in _token_variants(token):
            if len(variant) < 3:
                continue
            for prefix in prefix_order:
                candidate = f"{prefix}:{variant}"
                if candidate in strict_keys:
                    return candidate

    # Try last token variants first as object candidates.
    for variant in _token_variants(tokens[-1]):
        for prefix in prefix_order:
            candidate = f"{prefix}:{variant}"
            if candidate in strict_keys:
                return candidate

    return None


def _repair_instruction_candidate(
    raw_instruction: str,
    task_text: str,
    strict_keys: set[str],
) -> Optional[str]:
    if not isinstance(raw_instruction, str) or not raw_instruction.strip():
        return None

    strict = canonicalize_strict_instruction_key(raw_instruction)
    if strict is not None:
        return strict

    lowered = raw_instruction.strip().lower()

    if ":" in lowered:
        _, suffix = lowered.split(":", 1)
        suffix = suffix.strip().replace(" ", "_")
        if suffix:
            suffix_variants = _token_variants(suffix)
            for normalized_suffix in suffix_variants:
                if _contains_any(task_text, ("combat", "kill", "defeat", "hunt")):
                    candidate = f"kill_entity:{normalized_suffix}"
                    if candidate in strict_keys:
                        return candidate
                if _contains_any(task_text, ("craft", "make", "recipe")):
                    candidate = f"craft_item:{normalized_suffix}"
                    if candidate in strict_keys:
                        return candidate
                if _contains_any(task_text, ("mine", "collect", "gather")):
                    candidate = f"mine_block:{normalized_suffix}"
                    if candidate in strict_keys:
                        return candidate

    return _guess_strict_instruction_from_task(task_text, strict_keys)


def _build_instruction_examples_addendum(
    task_text: str,
    strict_keys: set[str],
    max_examples: int = 12,
) -> str:
    """Return a compact, task-relevant strict key sample list for prompts.

    This intentionally provides only a few examples (not exhaustive) to avoid
    prompt bloat while still anchoring the model to valid key patterns.
    """
    if not strict_keys:
        return ""

    tokens = _extract_task_tokens(task_text)
    variants: list[str] = []
    for token in tokens:
        variants.extend(_token_variants(token))
    variants = list(dict.fromkeys(v for v in variants if v))

    prefers_combat = _contains_any(task_text, ("combat", "kill", "defeat", "hunt"))
    prefers_craft = _contains_any(task_text, ("craft", "make", "recipe"))
    prefers_mine = _contains_any(task_text, ("mine", "collect", "gather"))

    prefix_order = ["kill_entity", "craft_item", "mine_block", "pickup", "use_item", "drop"]
    if prefers_craft:
        prefix_order = ["craft_item", "mine_block", "kill_entity", "pickup", "use_item", "drop"]
    elif prefers_mine:
        prefix_order = ["mine_block", "craft_item", "kill_entity", "pickup", "use_item", "drop"]
    elif prefers_combat:
        prefix_order = ["kill_entity", "use_item", "pickup", "mine_block", "craft_item", "drop"]

    chosen: list[str] = []

    # Prefer token-related examples first.
    if variants:
        for prefix in prefix_order:
            for key in sorted(strict_keys):
                if not key.startswith(f"{prefix}:"):
                    continue
                item = key.split(":", 1)[1]
                if any(v in item or item in v for v in variants):
                    chosen.append(key)
                    if len(chosen) >= max_examples:
                        break
            if len(chosen) >= max_examples:
                break

    # Backfill with a small generic sample by preferred prefixes.
    if len(chosen) < max_examples:
        for prefix in prefix_order:
            for key in sorted(strict_keys):
                if key.startswith(f"{prefix}:") and key not in chosen:
                    chosen.append(key)
                    if len(chosen) >= max_examples:
                        break
            if len(chosen) >= max_examples:
                break

    if not chosen:
        return ""

    lines = "\n".join(f"- {k}" for k in chosen[:max_examples])
    return (
        "\n\nReference examples from instructions.json (NOT exhaustive):\n"
        f"{lines}\n"
        "You may output other valid strict keys not listed above if they better fit the task."
    )


def _iter_transition_targets(trans: dict) -> list[str]:
    targets: list[str] = []
    for key in ("next_state", "on_true", "on_false"):
        target = trans.get(key)
        if isinstance(target, str):
            targets.append(target)
    return targets


def _pick_fallback_source_state(states: dict) -> Optional[str]:
    step1 = states.get("step1")
    if isinstance(step1, dict) and not step1.get("terminal"):
        return "step1"

    for name, sdef in states.items():
        if name == "fallback":
            continue
        if isinstance(sdef, dict) and not sdef.get("terminal"):
            return name
    return None


def ensure_timeout_fallback(plan: dict, task_text: str) -> dict:
    """
    Ensure fallback exists and each non-fallback state has a timeout transition.
    
    Safety measure: Since LLM now generates timeout-only conditions,
    this function adds fallback state and appends a timeout → fallback transition
    only when a state has no timeout transition at all.
    """
    states = plan.get("states")
    if not isinstance(states, dict):
        return plan

    source_state_name = _pick_fallback_source_state(states)
    if not source_state_name:
        return plan

    fallback_state = states.get("fallback")
    if not isinstance(fallback_state, dict) or fallback_state.get("terminal"):
        states["fallback"] = {
            "description": "fallback",
            "instruction": task_text,
            "instruction_type": "normal",
            "transitions": [
                {
                    "condition": {"type": "always"},
                    "next_state": "fallback",
                }
            ],
        }
        logger.info("Injected fallback state with self-loop transition")
    else:
        fallback_state["description"] = fallback_state.get("description", "fallback")

        fallback_state["instruction"] = task_text
        fallback_state["instruction_type"] = "normal"
        transitions = fallback_state.get("transitions")
        if not isinstance(transitions, list) or not transitions:
            fallback_state["transitions"] = [
                {
                    "condition": {"type": "always"},
                    "next_state": "fallback",
                }
            ]
        else:
            for trans in transitions:
                if not isinstance(trans, dict):
                    continue
                cond = trans.get("condition", {})
                if isinstance(cond, dict) and cond.get("type") == "always":
                    trans["next_state"] = "fallback"

    for state_name, state_def in states.items():
        if not isinstance(state_def, dict) or state_def.get("terminal"):
            continue
        if state_name == "fallback":
            continue

        transitions = state_def.get("transitions")
        if not isinstance(transitions, list):
            transitions = []

        has_timeout = False
        for trans in transitions:
            if not isinstance(trans, dict):
                continue
            cond = trans.get("condition", {})
            if isinstance(cond, dict) and cond.get("type") == "timeout":
                has_timeout = True

        if not has_timeout:
            transitions.append(
                {
                    "condition": {
                        "type": "timeout",
                        "max_steps": DEFAULT_FALLBACK_MAX_STEPS,
                    },
                    "next_state": "fallback",
                }
            )

        state_def["transitions"] = transitions

    return plan


def normalize_instruction_keys(plan: dict) -> dict:
    """Canonicalize instructions to exact instructions.json keys for non-fallback states."""
    states = plan.get("states")
    if not isinstance(states, dict):
        return plan

    for state_name, state_def in states.items():
        if not isinstance(state_def, dict) or state_def.get("terminal"):
            continue
        if state_name == "fallback":
            continue

        instruction = state_def.get("instruction")
        if not isinstance(instruction, str):
            continue

        canonical = canonicalize_instruction_key(instruction)
        if canonical and canonical != instruction:
            logger.info(
                "Canonicalized instruction in state '%s': %r -> %r",
                state_name,
                instruction,
                canonical,
            )
            state_def["instruction"] = canonical

    return plan


def validate_long_horizon_constraints(plan: dict) -> list[str]:
    states = plan.get("states")
    if not isinstance(states, dict):
        return []

    fallback = states.get("fallback")
    if not isinstance(fallback, dict) or fallback.get("terminal"):
        return ["Plan must include a non-terminal fallback step."]

    primary_non_terminal_count = sum(
        1
        for name, sdef in states.items()
        if name != "fallback" and isinstance(sdef, dict) and not sdef.get("terminal")
    )

    if primary_non_terminal_count < 2:
        return [
            "Long-horizon task generated too few non-terminal states "
            f"({primary_non_terminal_count})."
        ]

    return []


class Planner:
    """
    OpenAI-compatible planner.

    Public API:
    - plan_task(task_text): route short/long and return executable plan payload.
    """

    def __init__(
        self,
        api_key: str = "EMPTY",
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
        temperature: float = 0.2,
        max_completion_tokens: int = 4096,
        max_retries: int = 2,
    ):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.max_retries = max_retries
        self.validator = PlanValidator()

    # ------------------------------------------------------------------
    # New entry-point
    # ------------------------------------------------------------------

    def plan_task(
        self,
        task_text: str,
        feedback: Optional[list[str]] = None,
        force_horizon: Optional[str] = None,
        observation_image: Optional[np.ndarray] = None,
    ) -> dict:
        horizon = force_horizon or self.classify_horizon(task_text, observation_image=observation_image)

        if horizon == "short":
            short_directive = self.generate_short_directive(
                task_text,
                feedback=feedback,
                observation_image=observation_image,
            )
            return {
                "horizon": "short",
                "instruction": short_directive["instruction"],
                "instruction_type": short_directive["instruction_type"],
            }

        long_plan = self.generate_long_plan(
            task_text,
            feedback=feedback,
            observation_image=observation_image,
        )
        return {
            "horizon": "long",
            "plan": long_plan,
        }

    def classify_horizon(self, task_text: str, observation_image: Optional[np.ndarray] = None) -> str:
        user_prompt = build_horizon_prompt(task_text)
        raw = self._call_llm(
            HORIZON_SYSTEM_PROMPT,
            user_prompt,
            observation_image=observation_image,
        )
        parsed = self._parse_json(raw)
        if isinstance(parsed, dict):
            horizon = parsed.get("horizon")
            if horizon in {"short", "long"}:
                return horizon

        fallback = fallback_classify_task_horizon(task_text)
        logger.warning("Horizon classification fallback heuristic used -> %s", fallback)
        return fallback

    def generate_short_directive(
        self,
        task_text: str,
        feedback: Optional[list[str]] = None,
        observation_image: Optional[np.ndarray] = None,
    ) -> dict:
        strict_keys = get_strict_instruction_keys()
        user_prompt = build_short_directive_prompt(task_text)
        user_prompt += _build_instruction_examples_addendum(task_text, strict_keys)
        if feedback:
            user_prompt += REPLAN_ADDENDUM.format(errors="\n".join(f"- {e}" for e in feedback))

        for attempt in range(1 + self.max_retries):
            raw = self._call_llm(
                SHORT_DIRECTIVE_SYSTEM_PROMPT,
                user_prompt,
                observation_image=observation_image,
            )
            parsed = self._parse_json(raw)
            if not isinstance(parsed, dict):
                user_prompt = (
                    build_short_directive_prompt(task_text)
                    + REPLAN_ADDENDUM.format(
                        errors="- Failed to parse JSON. Output ONLY valid JSON with instruction and instruction_type."
                    )
                )
                continue

            instruction = parsed.get("instruction")
            instruction_type = parsed.get("instruction_type")

            errors: list[str] = []
            if not isinstance(instruction, str) or not instruction.strip():
                errors.append("instruction must be a non-empty string")
            else:
                repaired = _repair_instruction_candidate(
                    raw_instruction=instruction,
                    task_text=task_text,
                    strict_keys=strict_keys,
                )
                if repaired is not None:
                    instruction = repaired
                else:
                    # If strict conversion is ambiguous/impossible, keep raw task text
                    # so downstream VLM execution can handle free-form instruction.
                    instruction = task_text
                    instruction_type = "normal"

            if instruction_type not in {"auto", "simple", "normal", "recipe"}:
                errors.append("instruction_type must be one of: auto, simple, normal, recipe")

            if not errors:
                return {
                    "instruction": instruction,
                    "instruction_type": instruction_type,
                }

            user_prompt = (
                build_short_directive_prompt(task_text)
                + REPLAN_ADDENDUM.format(errors="\n".join(f"- {e}" for e in errors))
            )

            logger.warning("Short directive attempt %d errors: %s", attempt + 1, errors)

        logger.error(
            "Short directive generation failed after retries; using task_text as fallback"
        )
        return {
            "instruction": task_text,
            "instruction_type": "normal",
        }

    def generate_long_plan(
        self,
        task_text: str,
        feedback: Optional[list[str]] = None,
        observation_image: Optional[np.ndarray] = None,
    ) -> dict:
        strict_keys = get_strict_instruction_keys()
        user_prompt = build_planner_prompt(task_text)
        user_prompt += _build_instruction_examples_addendum(task_text, strict_keys)
        if feedback:
            user_prompt += REPLAN_ADDENDUM.format(errors="\n".join(f"- {e}" for e in feedback))

        for attempt in range(1 + self.max_retries):
            raw = self._call_llm(
                PLANNER_SYSTEM_PROMPT,
                user_prompt,
                observation_image=observation_image,
            )
            parsed = self._parse_json(raw)
            if parsed is None:
                logger.warning("Attempt %d: failed to parse JSON", attempt + 1)
                user_prompt = (
                    build_planner_prompt(task_text)
                    + REPLAN_ADDENDUM.format(
                        errors="- Failed to parse JSON from your response. Output ONLY valid JSON, no markdown fences."
                    )
                )
                continue

            canonical_plan = to_canonical_plan(parsed, task_text=task_text)
            canonical_plan["task"] = task_text
            canonical_plan = normalize_instruction_keys(canonical_plan)
            canonical_plan = ensure_timeout_fallback(canonical_plan, task_text=task_text)

            errors = self.validator.validate(canonical_plan)
            real_errors = [e for e in errors if not e.startswith("Warning")]
            real_errors.extend(validate_long_horizon_constraints(canonical_plan))

            if not real_errors:
                if errors:
                    for warning in errors:
                        logger.warning("Validation warning: %s", warning)
                logger.info("Long-horizon plan generated successfully (attempt %d)", attempt + 1)
                return canonical_to_simplified_plan(canonical_plan)

            logger.warning("Long plan attempt %d errors: %s", attempt + 1, real_errors)
            user_prompt = (
                build_planner_prompt(task_text)
                + REPLAN_ADDENDUM.format(errors="\n".join(f"- {e}" for e in real_errors))
            )

        logger.error("All long-plan attempts failed; returning last plan as-is")
        if "canonical_plan" in locals():
            return canonical_to_simplified_plan(canonical_plan)
        return parsed if "parsed" in locals() and parsed is not None else {}

    def vqa_check_subgoal(
        self,
        task_text: str,
        state_def: dict,
        observation_image: Optional[np.ndarray] = None,
    ) -> Optional[bool]:
        """Planner-owned VQA check for subgoal completion.

        Returns True / False, or None when the result is not parseable.
        """
        if observation_image is None:
            return None

        user_prompt = build_vqa_subgoal_prompt(task_text, state_def)
        raw = self._call_llm(
            VQA_SUBGOAL_SYSTEM_PROMPT,
            user_prompt,
            observation_image=observation_image,
        )

        parsed = self._parse_json(raw)
        if isinstance(parsed, dict) and isinstance(parsed.get("completed"), bool):
            return bool(parsed["completed"])

        lowered = (raw or "").strip().lower()
        if lowered.startswith("yes"):
            return True
        if lowered.startswith("no"):
            return False
        return None

    # ------------------------------------------------------------------

    def _default_short_instruction(self, task_text: str) -> str:
        guess = canonicalize_strict_instruction_key(task_text)
        if isinstance(guess, str):
            return guess

        strict_keys = get_strict_instruction_keys()
        guessed = _guess_strict_instruction_from_task(task_text, strict_keys)
        if isinstance(guessed, str):
            return guessed

        return task_text.strip()

    def _encode_image_data_url(self, image: np.ndarray) -> Optional[str]:
        try:
            if image is None:
                return None
            if not isinstance(image, np.ndarray):
                return None
            if image.ndim != 3 or image.shape[2] != 3:
                return None

            pil_img = Image.fromarray(image.astype(np.uint8), mode="RGB")
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=90)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
        except Exception:
            logger.exception("Failed to encode observation image for LLM")
            return None

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        observation_image: Optional[np.ndarray] = None,
    ) -> str:
        try:
            user_content: object
            if observation_image is None:
                user_content = user_prompt
            else:
                image_url = self._encode_image_data_url(observation_image)
                if image_url is None:
                    user_content = user_prompt
                else:
                    user_content = [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ]

            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=self.temperature,
                max_completion_tokens=self.max_completion_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.exception("LLM call failed: %s", e)
            return ""

    @staticmethod
    def _parse_json(raw: str) -> Optional[dict]:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\\s*", "", raw)
            raw = re.sub(r"\\s*```$", "", raw)
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                    return parsed if isinstance(parsed, dict) else None
                except json.JSONDecodeError:
                    pass
        return None
