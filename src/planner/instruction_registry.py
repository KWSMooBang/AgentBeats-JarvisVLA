"""Instruction key registry utilities for planner/validator.

Loads canonical instruction keys from jarvisvla/evaluate/assets/instructions.json
and supports resolving free-text aliases to canonical keys.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _instructions_path() -> Path:
    return Path(__file__).resolve().parents[2] / "jarvisvla" / "evaluate" / "assets" / "instructions.json"


def _normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*:\s*", ":", text)
    return text


def _expand_candidate_forms(text: str) -> set[str]:
    raw = _normalize_text(text)
    lower = raw.lower()
    forms = {raw, lower}

    if lower.startswith("drop:"):
        item = lower.split(":", 1)[1].replace(" ", "_")
        forms.add(f"drop:{item}")

    if lower.startswith("craft_item "):
        item = lower[len("craft_item "):].replace(" ", "_")
        forms.add(f"craft_item {item}")

    if lower.startswith("kill_entity:"):
        item = lower.split(":", 1)[1].replace(" ", "_")
        forms.add(f"kill_entity:{item}")

    if lower.startswith("mine_block:"):
        item = lower.split(":", 1)[1].replace(" ", "_")
        forms.add(f"mine_block:{item}")
    
    if lower.startswith("use_item:"):
        item = lower.split(":", 1)[1].replace(" ", "_")
        forms.add(f"use_item:{item}")
        
    if lower.startswith("pickup:"):
        item = lower.split(":", 1)[1].replace(" ", "_")
        forms.add(f"pickup:{item}")

    return forms


@lru_cache(maxsize=1)
def _load_registry() -> tuple[set[str], dict[str, str], bool]:
    path = _instructions_path()
    if not path.exists():
        logger.error("instructions.json not found at %s", path)
        return set(), {}, False

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.exception("Failed to load instructions.json: %s", e)
        return set(), {}, False

    if not isinstance(data, dict):
        logger.error("instructions.json root must be an object")
        return set(), {}, False

    canonical_keys: set[str] = set()
    alias_to_key: dict[str, str] = {}

    for key, meta in data.items():
        if not isinstance(key, str):
            continue

        canonical = key.strip()
        if not canonical:
            continue
        canonical_keys.add(canonical)

        for form in _expand_candidate_forms(canonical):
            alias_to_key.setdefault(form.lower(), canonical)

        if isinstance(meta, dict):
            instructs = meta.get("instruct", [])
            if isinstance(instructs, list):
                for s in instructs:
                    if isinstance(s, str) and s.strip():
                        for form in _expand_candidate_forms(s):
                            alias_to_key.setdefault(form.lower(), canonical)

    return canonical_keys, alias_to_key, True


def instructions_registry_available() -> bool:
    _, _, ok = _load_registry()
    return ok


def get_instruction_keys() -> set[str]:
    keys, _, _ = _load_registry()
    return keys


def is_strict_instruction_key(key: str) -> bool:
    if not isinstance(key, str):
        return False
    if key.count(":") != 1:
        return False
    prefix, item = key.split(":", 1)
    if not prefix or not item:
        return False
    if " " in prefix or " " in item:
        return False
    return True


def get_strict_instruction_keys() -> set[str]:
    return {key for key in get_instruction_keys() if is_strict_instruction_key(key)}


def canonicalize_instruction_key(instruction: str) -> Optional[str]:
    if not isinstance(instruction, str):
        return None

    keys, alias_to_key, ok = _load_registry()
    if not ok:
        return None

    for form in _expand_candidate_forms(instruction):
        if form in keys:
            return form
        mapped = alias_to_key.get(form.lower())
        if mapped:
            return mapped
    return None


def canonicalize_strict_instruction_key(instruction: str) -> Optional[str]:
    canonical = canonicalize_instruction_key(instruction)
    if canonical is None:
        return None
    if not is_strict_instruction_key(canonical):
        return None
    return canonical
