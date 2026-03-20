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

import json
import logging
import re
from typing import Optional

from src.planner.prompt_template import (
    HORIZON_SYSTEM_PROMPT,
    SHORT_DIRECTIVE_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    REPLAN_ADDENDUM,
    build_horizon_prompt,
    build_short_directive_prompt,
    build_planner_prompt,
    classify_task_horizon,
)
from src.planner.instruction_registry import (
    canonicalize_instruction_key,
    canonicalize_strict_instruction_key,
    get_strict_instruction_keys,
)
from src.planner.spec_format import canonical_to_simplified, to_canonical_spec
from src.planner.validator import PlanValidator

logger = logging.getLogger(__name__)

DEFAULT_FALLBACK_MAX_STEPS = 180


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
        if len(token) < 3:
            continue
        for prefix in prefix_order:
            candidate = f"{prefix}:{token}"
            if candidate in strict_keys:
                return candidate

    # Try last token first as object candidate (e.g., combat_enderman -> enderman).
    last = tokens[-1]
    for prefix in prefix_order:
        candidate = f"{prefix}:{last}"
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
            if _contains_any(task_text, ("combat", "kill", "defeat", "hunt")):
                candidate = f"kill_entity:{suffix}"
                if candidate in strict_keys:
                    return candidate
            if _contains_any(task_text, ("craft", "make", "recipe")):
                candidate = f"craft_item:{suffix}"
                if candidate in strict_keys:
                    return candidate
            if _contains_any(task_text, ("mine", "collect", "gather")):
                candidate = f"mine_block:{suffix}"
                if candidate in strict_keys:
                    return candidate

    return _guess_strict_instruction_from_task(task_text, strict_keys)


def _iter_transition_targets(trans: dict) -> list[str]:
    targets: list[str] = []
    for key in ("next_state", "on_true", "on_false"):
        target = trans.get(key)
        if isinstance(target, str):
            targets.append(target)
    return targets


def _choose_retry_fallback_state(
    state_name: str,
    transitions: list[dict],
    states: dict,
) -> Optional[str]:
    # Prefer explicit non-abort forward targets from non-retry transitions.
    for trans in transitions:
        cond = trans.get("condition", {})
        if cond.get("type") == "retry_exhausted":
            continue
        for target in _iter_transition_targets(trans):
            if target in states and target not in {"abort", state_name}:
                return target

    # Fallback: pick another non-terminal state in declared order.
    for candidate, cdef in states.items():
        if candidate in {state_name, "abort"}:
            continue
        if isinstance(cdef, dict) and not cdef.get("terminal"):
            return candidate

    # Last resort: success terminal if present.
    if "success" in states and state_name != "success":
        return "success"
    return None


def normalize_retry_exhausted_transitions(spec: dict) -> dict:
    """Rewrite retry_exhausted -> abort to a non-abort fallback when possible."""
    states = spec.get("states")
    if not isinstance(states, dict):
        return spec

    for state_name, state_def in states.items():
        if not isinstance(state_def, dict) or state_def.get("terminal"):
            continue

        transitions = state_def.get("transitions")
        if not isinstance(transitions, list):
            continue

        fallback = _choose_retry_fallback_state(state_name, transitions, states)
        if not fallback:
            continue

        for trans in transitions:
            cond = trans.get("condition", {})
            if cond.get("type") != "retry_exhausted":
                continue
            if trans.get("next_state") == "abort":
                trans["next_state"] = fallback
                logger.info(
                    "Rewrote retry_exhausted target in state '%s': abort -> %s",
                    state_name,
                    fallback,
                )

    return spec


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


def ensure_timeout_fallback(spec: dict, task_text: str) -> dict:
    """Ensure fallback exists and stalled states route to fallback by timeout."""
    states = spec.get("states")
    if not isinstance(states, dict):
        return spec

    source_state_name = _pick_fallback_source_state(states)
    if not source_state_name:
        return spec

    fallback_state = states.get("fallback")
    if not isinstance(fallback_state, dict) or fallback_state.get("terminal"):
        states["fallback"] = {
            "description": "fallback",
            "instruction": task_text,
            "instruction_type": "normal",
            "transitions": [
                {
                    "condition": {"type": "always"},
                    "next_state": source_state_name,
                }
            ],
        }
        logger.info("Injected fallback state with task_text -> %s", source_state_name)
    else:
        fallback_state["description"] = fallback_state.get("description", "fallback")
        fallback_state["instruction"] = task_text
        fallback_state["instruction_type"] = "normal"
        transitions = fallback_state.get("transitions")
        if not isinstance(transitions, list) or not transitions:
            fallback_state["transitions"] = [
                {
                    "condition": {"type": "always"},
                    "next_state": source_state_name,
                }
            ]

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
                trans["next_state"] = "fallback"

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

    return spec


def normalize_instruction_keys(spec: dict) -> dict:
    """Canonicalize instructions to exact instructions.json keys for non-fallback states."""
    states = spec.get("states")
    if not isinstance(states, dict):
        return spec

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

    return spec


def validate_long_horizon_constraints(spec: dict) -> list[str]:
    states = spec.get("states")
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


class LLMPlanner:
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
        max_tokens: int = 4096,
        max_retries: int = 2,
    ):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
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
    ) -> dict:
        horizon = force_horizon or self.classify_horizon(task_text)

        if horizon == "short":
            short_directive = self.generate_short_directive(task_text, feedback=feedback)
            return {
                "horizon": "short",
                "instruction": short_directive["instruction"],
                "instruction_type": short_directive["instruction_type"],
            }

        long_plan = self.generate_long_plan(task_text, feedback=feedback)
        return {
            "horizon": "long",
            "plan": long_plan,
        }

    def classify_horizon(self, task_text: str) -> str:
        user_prompt = build_horizon_prompt(task_text)
        raw = self._call_llm(HORIZON_SYSTEM_PROMPT, user_prompt)
        parsed = self._parse_json(raw)
        if isinstance(parsed, dict):
            horizon = parsed.get("horizon")
            if horizon in {"short", "long"}:
                return horizon

        fallback = classify_task_horizon(task_text)
        logger.warning("Horizon classification fallback heuristic used -> %s", fallback)
        return fallback

    def generate_short_directive(
        self,
        task_text: str,
        feedback: Optional[list[str]] = None,
    ) -> dict:
        strict_keys = get_strict_instruction_keys()
        user_prompt = build_short_directive_prompt(task_text)
        if feedback:
            user_prompt += REPLAN_ADDENDUM.format(errors="\n".join(f"- {e}" for e in feedback))

        for attempt in range(1 + self.max_retries):
            raw = self._call_llm(SHORT_DIRECTIVE_SYSTEM_PROMPT, user_prompt)
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
                if repaired is None:
                    errors.append(
                        f"instruction '{instruction}' is not a valid strict instructions.json key"
                    )
                else:
                    instruction = repaired

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

        fallback_instruction = self._default_short_instruction(task_text)
        logger.error(
            "Short directive generation failed after retries; fallback instruction=%s",
            fallback_instruction,
        )
        return {
            "instruction": fallback_instruction,
            "instruction_type": "normal",
        }

    def generate_long_plan(
        self,
        task_text: str,
        feedback: Optional[list[str]] = None,
    ) -> dict:
        user_prompt = build_planner_prompt(task_text)
        if feedback:
            user_prompt += REPLAN_ADDENDUM.format(errors="\n".join(f"- {e}" for e in feedback))

        for attempt in range(1 + self.max_retries):
            raw = self._call_llm(PLANNER_SYSTEM_PROMPT, user_prompt)
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

            canonical_spec = to_canonical_spec(parsed, task_text=task_text)
            canonical_spec = normalize_instruction_keys(canonical_spec)
            canonical_spec = normalize_retry_exhausted_transitions(canonical_spec)
            canonical_spec = ensure_timeout_fallback(canonical_spec, task_text=task_text)

            errors = self.validator.validate(canonical_spec)
            real_errors = [e for e in errors if not e.startswith("Warning")]
            real_errors.extend(validate_long_horizon_constraints(canonical_spec))

            if not real_errors:
                if errors:
                    for warning in errors:
                        logger.warning("Validation warning: %s", warning)
                logger.info("Long-horizon plan generated successfully (attempt %d)", attempt + 1)
                return canonical_to_simplified(canonical_spec)

            logger.warning("Long plan attempt %d errors: %s", attempt + 1, real_errors)
            user_prompt = (
                build_planner_prompt(task_text)
                + REPLAN_ADDENDUM.format(errors="\n".join(f"- {e}" for e in real_errors))
            )

        logger.error("All long-plan attempts failed; returning last plan as-is")
        if "canonical_spec" in locals():
            return canonical_to_simplified(canonical_spec)
        return parsed if "parsed" in locals() and parsed is not None else {}

    # ------------------------------------------------------------------

    def _default_short_instruction(self, task_text: str) -> str:
        guess = canonicalize_strict_instruction_key(task_text)
        if isinstance(guess, str):
            return guess

        strict_keys = get_strict_instruction_keys()
        guessed = _guess_strict_instruction_from_task(task_text, strict_keys)
        if isinstance(guessed, str):
            return guessed

        if "kill_entity:zombie" in strict_keys:
            return "kill_entity:zombie"

        keys = sorted(strict_keys)
        if keys:
            return keys[0]
        return "kill_entity:zombie"

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
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
