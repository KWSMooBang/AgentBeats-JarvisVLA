from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SequenceRouter:
    """Routes instructions to scripted sequences via keyword heuristic and hint-based fallback."""

    def select_sequence(
        self,
        instruction: str,
        state_def: Optional[dict],
        sequence_catalog: dict[str, dict[str, Any]],
        require_sequence: bool = False,
    ) -> dict[str, Any]:
        task_text = (state_def or {}).get("task_text", "")
        combined = f"{instruction} {task_text}".strip()
        seq = self._keyword_match(combined, sequence_catalog)
        if seq:
            meta = sequence_catalog[seq]
            logger.info(
                "[Router] keyword matched %r for instruction=%r",
                seq, instruction[:60],
            )
            return {
                "execution_hint": meta["execution_hint"],
                "sequence_name": seq,
                "reason": "keyword_priority",
            }

        return self._hint_fallback(
            instruction,
            state_def,
            sequence_catalog,
            require_sequence=require_sequence,
        )

    def _hint_fallback(
        self,
        instruction: str,
        state_def: Optional[dict],
        sequence_catalog: dict[str, dict[str, Any]],
        require_sequence: bool = False,
    ) -> dict[str, Any]:
        hinted = (state_def or {}).get("execution_hint")
        task_text = (state_def or {}).get("task_text", "")
        combined = f"{instruction} {task_text}".strip()

        if hinted == "vla":
            seq = self._keyword_match(combined, sequence_catalog)
            if seq:
                meta = sequence_catalog[seq]
                return {
                    "execution_hint": meta["execution_hint"],
                    "sequence_name": seq,
                    "reason": "keyword_match",
                }
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
            seq = self._keyword_match(combined, sequence_catalog)
            if seq:
                meta = sequence_catalog[seq]
                return {
                    "execution_hint": meta["execution_hint"],
                    "sequence_name": seq,
                    "reason": "keyword_match",
                }
            return {
                "execution_hint": hinted,
                "sequence_name": None,
                "reason": "no_sequence_found",
            }

        seq = self._keyword_match(combined, sequence_catalog)
        if seq:
            meta = sequence_catalog[seq]
            return {
                "execution_hint": meta["execution_hint"],
                "sequence_name": seq,
                "reason": "keyword_match",
            }

        return {
            "execution_hint": "vla",
            "sequence_name": None,
            "reason": "default_vla",
        }

    @staticmethod
    def _keyword_match(
        instruction: str,
        sequence_catalog: dict[str, dict[str, Any]],
    ) -> Optional[str]:
        raw = (instruction or "").lower()
        # Normalize underscores → spaces so task_text keys like "lay_carpet",
        # "light_up_the_surroundings", "place_a_item_frame" match their space-based keywords.
        text = raw.replace("_", " ")

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
            if has("throw_cycle"):
                return "throw_cycle"
            if has("throw_projectile_loop"):
                return "throw_projectile_loop"

        # --- lay / pave / floor (must precede craft check: "lay carpet" must not fall into craft route) ---
        if any(k in text for k in ("lay ", "pave ", "floor ")):
            if has("line_place_repeat"):
                return "line_place_repeat"

        # --- craft / recipe / inventory crafting ---
        # Use raw (pre-normalization) to preserve "craft_item" from planner output.
        if (text.startswith("craft") or
                "craft_item" in raw or
                "recipe" in text or
                ("open" in text and "craft" in text)):
            if has("open_inventory_craft"):
                return "open_inventory_craft"

        # --- consume / drink / potion ---
        # "eat" excluded: substring-matches "defeat" (combat instruction).
        if any(k in text for k in ("drink", "consume", "potion", "food")):
            if has("consume_cycle"):
                return "consume_cycle"
            if has("consume_item_sequence"):
                return "consume_item_sequence"

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
        if any(k in text for k in ("plant", "farm", "sow", "seed")):
            if has("approach_farmland_then_plant_rows"):
                return "approach_farmland_then_plant_rows"

        # --- stack / pillar / fence / build vertically ---
        if any(k in text for k in ("stack", "pillar", "tower", "build up", "place on top")):
            if has("stack_place_repeat"):
                return "stack_place_repeat"

        # --- carve / use on ground ---
        if any(k in text for k in ("carve", "use on ground", "place on ground")):
            if has("approach_then_ground_use"):
                return "approach_then_ground_use"

        # --- light up / spread light sources ---
        # "light" alone excluded: collides with "ignite" / "flint and steel".
        if any(k in text for k in ("light up", "light the surr", "place light", "illuminate")):
            if has("place_light_sources"):
                return "place_light_sources"

        # --- decorate / place on wall (item frame, painting, banner) ---
        if any(k in text for k in ("item frame", "painting", "banner", "picture frame")):
            if has("approach_then_vertical_place"):
                return "approach_then_vertical_place"
        if "wall" in text and any(k in text for k in ("decorate", "place", "hang", "attach", "put")):
            if has("approach_then_vertical_place"):
                return "approach_then_vertical_place"

        # --- decorate / place items on ground ---
        if "decorate" in text and "wall" not in text:
            if has("scatter_ground_placeables"):
                return "scatter_ground_placeables"

        # --- fire / flint / steel / ignite ---
        if any(k in text for k in ("fire", "flint", "steel", "ignite")):
            if has("approach_then_ground_use"):
                return "approach_then_ground_use"

        # --- shield / defend ---
        if any(k in text for k in ("shield", "defend", "totem")):
            if has("defend_with_item"):
                return "defend_with_item"

        # --- lead / rope / attach ---
        if any(k in text for k in ("lead", "leash", "rope", "attach")):
            if has("use_on_nearby_entity"):
                return "use_on_nearby_entity"

        # --- clear / weed / grass / sweep ---
        if any(k in text for k in ("clean", "clear", "weed", "tall grass", "remove grass", "sweep")):
            if has("clear_ground_plants"):
                return "clear_ground_plants"

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
