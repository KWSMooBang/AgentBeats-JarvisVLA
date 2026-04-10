from __future__ import annotations

import base64
import io
import json
import logging
import re
from typing import Any, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


SEQUENCE_SELECTOR_SYSTEM_PROMPT = """You are a policy selector for a Minecraft agent.

Choose only from the provided sequence catalog. Do not invent new primitives.
Your job is not to produce actions. Your job is to choose a high-level execution mode and one catalog sequence.

Rules:
- Prefer "vla" for open-ended navigation, combat, mining, and general manipulation.
- Prefer "scripted" for deterministic repeated control loops such as hotbar cycling, repeated use/drop/throw, or stable short fixed routines.
- Prefer "hybrid" when VLA should handle approach/alignment but a deterministic scripted routine should finish the interaction.
- Use the current image to judge whether the task looks like GUI interaction, placement, use-on-target, or open-ended world interaction.
- Return valid JSON only.
"""


class VLMSequenceSelector:
    """Selects a policy/sequence once per state, not per action step."""

    def __init__(
        self,
        api_key: str = "EMPTY",
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.ready = False
        self._disabled_after_error = False
        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self.ready = True
            logger.info("VLMSequenceSelector initialized (model=%s)", model)
        except Exception as exc:
            logger.warning("VLMSequenceSelector unavailable: %s", exc)
            self.client = None

    def select_sequence(
        self,
        image: np.ndarray,
        instruction: str,
        state_def: Optional[dict],
        sequence_catalog: dict[str, dict[str, Any]],
        primitive_catalog: dict[str, str],
        require_sequence: bool = False,
    ) -> dict[str, Any]:
        if not self.ready:
            return self.heuristic_select(
                instruction,
                state_def,
                sequence_catalog,
                require_sequence=require_sequence,
            )

        try:
            user_text = self._build_user_prompt(
                instruction=instruction,
                state_def=state_def or {},
                sequence_catalog=sequence_catalog,
                primitive_catalog=primitive_catalog,
            )
            image_url = self._encode_image(image)
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SEQUENCE_SELECTOR_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    },
                ],
                temperature=self.temperature,
            )
            raw = (resp.choices[0].message.content or "").strip()
            parsed = self._parse_json(raw)
            if isinstance(parsed, dict):
                validated = self._validate_selection(parsed, sequence_catalog)
                if validated is not None:
                    return validated
        except Exception as exc:
            logger.warning("VLMSequenceSelector request failed (will retry next call): %s", exc)
        return self.heuristic_select(
            instruction,
            state_def,
            sequence_catalog,
            require_sequence=require_sequence,
        )

    def heuristic_select(
        self,
        instruction: str,
        state_def: Optional[dict],
        sequence_catalog: dict[str, dict[str, Any]],
        require_sequence: bool = False,
    ) -> dict[str, Any]:
        hinted = (state_def or {}).get("execution_hint")
        if hinted == "vla":
            return {
                "execution_hint": "vla",
                "sequence_name": None,
                "reason": "planner_hint",
            }
        if hinted in {"scripted", "hybrid"} and not require_sequence:
            return {
                "execution_hint": hinted,
                "sequence_name": None,
                "reason": "planner_hint",
            }
        if hinted in {"scripted", "hybrid"} and require_sequence:
            return {
                "execution_hint": hinted,
                "sequence_name": None,
                "reason": "selector_unavailable_missing_concrete_sequence",
            }

        # Semantic keyword heuristic — matches instruction text to game mechanics.
        # Not benchmark-specific: these are general game action mappings.
        # Also check task_text if present in state_def (richer context).
        task_text = (state_def or {}).get("task_text", "")
        combined = f"{instruction} {task_text}".strip()
        seq = self._semantic_keyword_match(combined, sequence_catalog)
        if seq:
            meta = sequence_catalog[seq]
            return {
                "execution_hint": meta["execution_hint"],
                "sequence_name": seq,
                "reason": "heuristic_keyword_match",
            }

        return {
            "execution_hint": "vla",
            "sequence_name": None,
            "reason": "selector_unavailable_default_vla",
        }

    @staticmethod
    def _semantic_keyword_match(
        instruction: str,
        sequence_catalog: dict[str, dict[str, Any]],
    ) -> Optional[str]:
        """Map instruction text to a catalog sequence via game-mechanic keywords.

        Order matters: more specific patterns first.
        Only returns a name if the sequence exists in the catalog.
        """
        text = (instruction or "").lower()

        def has(catalog_name: str) -> bool:
            return catalog_name in sequence_catalog

        # --- view / posture ---
        if any(k in text for k in ("look at the sky", "look up", "look skyward", "face the sky", "look toward the sky")):
            if has("view_upward"):
                return "view_upward"

        # --- drop ---
        if any(k in text for k in ("drop", "discard", "throw away", "toss away")):
            if has("drop_cycle"):
                return "drop_cycle"

        # --- throw / projectile ---
        if any(k in text for k in ("throw", "toss", "launch", "snowball", "ender pearl", "trident", "egg")):
            if has("throw_projectile_loop"):
                return "throw_projectile_loop"
            if has("throw_cycle"):
                return "throw_cycle"

        # --- consume / drink / eat / potion ---
        if any(k in text for k in ("drink", "eat", "consume", "potion", "food", "apple", "bread")):
            if has("consume_item_sequence"):
                return "consume_item_sequence"
            if has("consume_cycle"):
                return "consume_cycle"

        # --- brew ---
        if any(k in text for k in ("brew", "brewing stand")):
            if has("container_gui_cycle"):
                return "container_gui_cycle"

        # --- sleep / rest / bed ---
        if any(k in text for k in ("sleep", "bed", "rest", "hammock")):
            if has("rest_at_object"):
                return "rest_at_object"

        # --- smelt / furnace / enchant ---
        if any(k in text for k in ("smelt", "furnace", "enchant", "anvil", "grindstone", "loom", "stonecutter")):
            if has("container_gui_cycle"):
                return "container_gui_cycle"

        # --- plant / farm ---
        if any(k in text for k in ("plant", "farm", "sow", "seed", "wheat", "carrot", "potato", "beetroot")):
            if has("approach_farmland_then_plant_rows"):
                return "approach_farmland_then_plant_rows"

        # --- stack / pillar / fence / build vertically ---
        if any(k in text for k in ("stack", "pillar", "tower", "build up", "place on top")):
            if has("stack_place_repeat"):
                return "stack_place_repeat"

        # --- carve / use on ground ---
        if any(k in text for k in ("carve", "pumpkin", "use on ground", "place on ground")):
            if has("approach_then_ground_use"):
                return "approach_then_ground_use"

        # --- fire / flint / steel ---
        if any(k in text for k in ("fire", "flint", "steel", "ignite", "light")):
            if has("approach_then_ground_use"):
                return "approach_then_ground_use"

        # --- shield / defend ---
        if any(k in text for k in ("shield", "defend", "block", "totem")):
            if has("defend_with_item"):
                return "defend_with_item"

        # --- lead / rope / attach ---
        if any(k in text for k in ("lead", "leash", "rope", "attach")):
            if has("use_on_nearby_entity"):
                return "use_on_nearby_entity"

        # --- shear ---
        if any(k in text for k in ("shear", "shears", "wool")):
            if has("shear_nearby_entity"):
                return "shear_nearby_entity"

        # --- bow / crossbow / shoot / arrow ---
        if any(k in text for k in ("bow", "crossbow", "shoot", "arrow", "shoot")):
            if has("ranged_combat_loop"):
                return "ranged_combat_loop"

        # --- melee / sword / attack / kill ---
        if any(k in text for k in ("attack", "kill", "defeat", "slay", "sword", "axe", "hit")):
            if has("melee_combat_loop"):
                return "melee_combat_loop"

        # --- chop / tree / log / wood ---
        if any(k in text for k in ("chop", "cut", "tree", "log", "wood")):
            if has("chop_tree_loop"):
                return "chop_tree_loop"

        # --- clear / weed / grass / sweep ---
        if any(k in text for k in ("clean", "clear", "weed", "tall grass", "remove grass", "sweep")):
            if has("clear_ground_plants"):
                return "clear_ground_plants"

        # --- mine / dig ---
        if any(k in text for k in ("mine", "dig", "break block", "excavate")):
            if has("mine_ahead_loop"):
                return "mine_ahead_loop"

        # --- fish / rod / cast ---
        if any(k in text for k in ("fish", "fishing", "rod", "cast")):
            if has("aim_then_use_repeat"):
                return "aim_then_use_repeat"

        # --- boat / vehicle / mount ---
        if any(k in text for k in ("boat", "ride", "minecart", "saddle", "mount")):
            if has("approach_then_place_and_use"):
                return "approach_then_place_and_use"

        # --- chest / open / container ---
        if any(k in text for k in ("chest", "open", "container", "barrel", "hopper")):
            if has("approach_then_open_interactable"):
                return "approach_then_open_interactable"

        return None

    def _build_user_prompt(
        self,
        instruction: str,
        state_def: dict[str, Any],
        sequence_catalog: dict[str, dict[str, Any]],
        primitive_catalog: dict[str, str],
    ) -> str:
        lines = [
            f"Instruction: {instruction}",
            f"State description: {state_def.get('description', '')}",
            "",
            "Available sequences:",
        ]
        for name, meta in sequence_catalog.items():
            primitives = ", ".join(meta.get("primitive_names", []))
            lines.append(
                f'- {name}: hint={meta.get("execution_hint")} | {meta.get("description")} | primitives=[{primitives}]'
            )
        lines.append("")
        lines.append("Primitive catalog:")
        for name, desc in primitive_catalog.items():
            lines.append(f"- {name}: {desc}")
        lines.append("")
        lines.append(
            'Return JSON with keys: execution_hint, sequence_name, reason. sequence_name must be null for pure vla.'
        )
        return "\n".join(lines)

    def _validate_selection(
        self,
        parsed: dict[str, Any],
        sequence_catalog: dict[str, dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        execution_hint = parsed.get("execution_hint")
        sequence_name = parsed.get("sequence_name")
        reason = parsed.get("reason", "")
        if execution_hint not in {"vla", "scripted", "hybrid"}:
            return None
        if sequence_name is not None:
            if not isinstance(sequence_name, str) or sequence_name not in sequence_catalog:
                return None
            hinted = sequence_catalog[sequence_name]["execution_hint"]
            if execution_hint == "vla":
                execution_hint = hinted
        elif execution_hint != "vla":
            return None
        return {
            "execution_hint": execution_hint,
            "sequence_name": sequence_name,
            "reason": reason if isinstance(reason, str) else "",
        }

    @staticmethod
    def _parse_json(raw: str) -> Optional[dict[str, Any]]:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                return None
            try:
                parsed = json.loads(match.group())
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                return None

    @staticmethod
    def _encode_image(image: np.ndarray) -> str:
        pil = Image.fromarray(image.astype("uint8"))
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"
