"""Scripted sequence catalog and primitive templates.

This module is data-only on purpose.  The policy engine decides *which*
sequence to run; this file describes what each sequence means and which
primitive steps compose it.
"""

from __future__ import annotations

import copy
from typing import Any, Optional


SEQUENCE_PRIMITIVES: dict[str, list[dict[str, Any]]] = {
    "container_gui_cycle": [
        {"executor": "vla", "instruction": "open the relevant workstation or container", "instruction_type": "normal", "steps": 30},
        {"executor": "script", "primitive": "cycle_hotbar", "steps": 36},
        {"executor": "script", "primitive": "gui_click_confirm_repeat", "steps": 80},
    ],
    "approach_then_use_on_target": [
        {"executor": "vla", "instruction": "approach the target entity", "instruction_type": "normal", "steps": 40},
        {"executor": "script", "primitive": "cycle_hotbar", "steps": 24},
        {"executor": "script", "primitive": "hold_use_forward", "steps": 60},
    ],
    "approach_then_ground_use": [
        {"executor": "vla", "instruction": "move to the spot where the item should be used on the ground", "instruction_type": "normal", "steps": 24},
        {"executor": "script", "primitive": "look_down_use", "steps": 60},
    ],
    "approach_farmland_then_plant_rows": [
        {"executor": "vla", "instruction": "move to the prepared ground area", "instruction_type": "normal", "steps": 40},
        {"executor": "script", "primitive": "look_down", "steps": 16},
        {"executor": "script", "primitive": "cycle_hotbar", "steps": 24},
        {"executor": "script", "primitive": "plant_row", "steps": 80},
    ],
    "approach_then_interact": [
        {"executor": "vla", "instruction": "move to the nearby interactable object", "instruction_type": "normal", "steps": 40},
        {"executor": "script", "primitive": "look_down", "steps": 16},
        {"executor": "script", "primitive": "hold_use", "steps": 50},
    ],
    "place_while_walking": [
        {"executor": "vla", "instruction": "move through the dark area", "instruction_type": "normal", "steps": 50},
        {"executor": "script", "primitive": "cycle_hotbar", "steps": 24},
        {"executor": "script", "primitive": "look_down_use_walk", "steps": 80},
    ],
    "approach_then_open_interactable": [
        {"executor": "vla", "instruction": "move to the nearby container or interactable block", "instruction_type": "normal", "steps": 45},
        {"executor": "script", "primitive": "tap_use", "steps": 32},
    ],
    "structured_build_sequence": [
        {"executor": "vla", "instruction": "face the portal build area", "instruction_type": "normal", "steps": 30},
        {"executor": "script", "primitive": "portal_frame_place", "steps": 140},
        {"executor": "script", "primitive": "look_down_and_use", "steps": 60},
    ],
    "stack_then_trigger": [
        {"executor": "vla", "instruction": "move close to the snow block stack position", "instruction_type": "normal", "steps": 30},
        {"executor": "script", "primitive": "snow_golem_stack", "steps": 90},
    ],
    "dig_then_fill": [
        {"executor": "script", "primitive": "dig_down_attack", "steps": 80},
        {"executor": "script", "primitive": "fill_with_block", "steps": 40},
    ],
    "approach_then_place_and_use": [
        {"executor": "vla", "instruction": "move to a suitable placement surface", "instruction_type": "normal", "steps": 40},
        {"executor": "script", "primitive": "boat_place_use", "steps": 60},
    ],
    "approach_target_then_place_vehicle": [
        {"executor": "vla", "instruction": "approach the target area or entity", "instruction_type": "normal", "steps": 40},
        {"executor": "script", "primitive": "boat_place_use", "steps": 70},
    ],
    "approach_height_then_use": [
        {"executor": "vla", "instruction": "move to a suitable edge or elevated surface", "instruction_type": "normal", "steps": 50},
        {"executor": "script", "primitive": "look_down_use", "steps": 70},
    ],
    "line_place_repeat": [
        {"executor": "script", "primitive": "open_inventory", "steps": 6},
        {"executor": "vla", "instruction_type": "recipe", "steps": 50},
        {"executor": "vla", "instruction": "select the carpet in your hotbar", "instruction_type": "normal", "steps": 15},
        {"executor": "script", "primitive": "place_row_walk", "steps": 80},
    ],
    "stack_place_repeat": [
        {"executor": "script", "primitive": "stack_vertical", "steps": 100},
    ],
    "scatter_ground_placeables": [
        {"executor": "vla", "instruction": "select a small flat flower such as a dandelion, poppy, or tulip from your hotbar", "instruction_type": "normal", "steps": 20},
        {"executor": "script", "primitive": "look_down_use_walk", "steps": 120},
    ],
    "place_light_sources": [
        {"executor": "vla", "instruction": "select a torch or light-emitting block from your hotbar", "instruction_type": "normal", "steps": 20},
        {"executor": "script", "primitive": "look_down_use_walk", "steps": 120},
    ],
    "approach_then_vertical_place": [
        {"executor": "vla", "instruction": "face a nearby wall or vertical surface and move close to it", "instruction_type": "normal", "steps": 30},
        {"executor": "vla", "instruction": "select an item frame, painting, or wall decoration from your hotbar", "instruction_type": "normal", "steps": 20},
        {"executor": "script", "primitive": "place_wall_use", "steps": 90},
    ],
    "aim_then_use_repeat": [
        {"executor": "vla", "instruction": "face the nearby animal", "instruction_type": "normal", "steps": 24},
        {"executor": "script", "primitive": "rod_cast_repeat", "steps": 100},
    ],
    "approach_then_place_climbable": [
        {"executor": "vla", "instruction": "move to the vertical surface", "instruction_type": "normal", "steps": 30},
        {"executor": "script", "primitive": "place_wall_use", "steps": 60},
        {"executor": "script", "primitive": "climb_forward", "steps": 70},
    ],
    "navigate_then_dig_then_open": [
        {"executor": "vla", "instruction": "follow the treasure map and move toward the target area", "instruction_type": "normal", "steps": 80},
        {"executor": "script", "primitive": "dig_down_attack", "steps": 60},
        {"executor": "script", "primitive": "tap_use", "steps": 30},
    ],
    "consume_cycle": [
        {"executor": "script", "primitive": "cycle_hotbar_then_hold_use", "steps": 120},
    ],
    "consume_item_sequence": [
        {"executor": "script", "primitive": "cycle_hotbar_then_hold_use", "steps": 120},
    ],
    "open_inventory_craft": [
        {"executor": "script", "primitive": "open_inventory", "steps": 6},
        {"executor": "vla", "instruction_type": "recipe", "steps": 110},
    ],
    "throw_cycle": [
        {"executor": "script", "primitive": "throw_held_item_flow", "steps": 80},
    ],
    "drop_cycle": [
        {"executor": "script", "primitive": "drop_cycle", "steps": 90},
    ],
    "hold_use_cycle": [
        {"executor": "script", "primitive": "cycle_hotbar_then_hold_use", "steps": 120},
    ],
    "view_upward": [
        {"executor": "script", "primitive": "look_up", "steps": 20},
        {"executor": "script", "primitive": "maintain_up_view", "steps": 60},
    ],
    "melee_combat_loop": [
        {"executor": "vla", "instruction": "approach the nearest enemy or mob", "instruction_type": "normal", "steps": 30},
        {"executor": "script", "primitive": "melee_attack", "steps": 80},
    ],
    "ranged_combat_loop": [
        {"executor": "vla", "instruction": "face the target entity", "instruction_type": "normal", "steps": 20},
        {"executor": "script", "primitive": "ranged_attack", "steps": 80},
    ],
    "throw_projectile_loop": [
        {"executor": "vla", "instruction": "face the target or direction to throw", "instruction_type": "normal", "steps": 20},
        {"executor": "script", "primitive": "throw_weapon", "steps": 60},
    ],
    "use_on_nearby_entity": [
        {"executor": "vla", "instruction": "face and move toward the nearby entity", "instruction_type": "normal", "steps": 30},
        {"executor": "script", "primitive": "approach_and_use_on_entity", "steps": 70},
    ],
    "collect_surface_blocks": [
        {"executor": "vla", "instruction": "move above the target ground blocks", "instruction_type": "normal", "steps": 30},
        {"executor": "script", "primitive": "mine_ground", "steps": 80},
    ],
    "chop_tree_loop": [
        {"executor": "vla", "instruction": "face a nearby tree trunk", "instruction_type": "normal", "steps": 30},
        {"executor": "script", "primitive": "chop_tree", "steps": 100},
    ],
    "mine_ahead_loop": [
        {"executor": "vla", "instruction": "face the block to mine", "instruction_type": "normal", "steps": 20},
        {"executor": "script", "primitive": "mine_forward", "steps": 80},
    ],
    "rest_at_object": [
        {"executor": "vla", "instruction": "move to and face the bed or rest object", "instruction_type": "normal", "steps": 40},
        {"executor": "script", "primitive": "approach_and_use_rest_object", "steps": 60},
    ],
    "defend_with_item": [
        {"executor": "script", "primitive": "cycle_hotbar", "steps": 24},
        {"executor": "script", "primitive": "hold_defensive_item", "steps": 100},
    ],
    "shear_nearby_entity": [
        {"executor": "vla", "instruction": "face and approach the nearby animal", "instruction_type": "normal", "steps": 30},
        {"executor": "script", "primitive": "shear_target", "steps": 70},
    ],
    "clear_ground_plants": [
        {"executor": "vla", "instruction": "select a cutting tool such as shears or a sword from your hotbar", "instruction_type": "normal", "steps": 20},
        {"executor": "script", "primitive": "attack_walk_sweep", "steps": 120},
    ],
}


SEQUENCE_CATALOG: dict[str, dict[str, Any]] = {
    "container_gui_cycle": {
        "execution_hint": "scripted",
        "description": "Open a nearby interaction container/UI, then perform repeated simple insertion or confirmation actions.",
        "primitive_names": ["cycle_hotbar", "gui_click_confirm_repeat"],
    },
    "consume_cycle": {
        "execution_hint": "scripted",
        "description": "Cycle held items and consume/use them until one works.",
        "primitive_names": ["cycle_hotbar_then_hold_use"],
    },
    "throw_cycle": {
        "execution_hint": "scripted",
        "description": "Cycle held throwable items and repeatedly use them.",
        "primitive_names": ["throw_held_item_flow"],
    },
    "drop_cycle": {
        "execution_hint": "scripted",
        "description": "Cycle held items and drop them one by one.",
        "primitive_names": ["drop_cycle"],
    },
    "hold_use_cycle": {
        "execution_hint": "scripted",
        "description": "Cycle held items and hold use defensively or continuously.",
        "primitive_names": ["cycle_hotbar_then_hold_use"],
    },
    "approach_then_use_on_target": {
        "execution_hint": "hybrid",
        "description": "Approach a target entity, then run a repeated use sequence.",
        "primitive_names": ["cycle_hotbar", "hold_use_forward"],
    },
    "approach_then_ground_use": {
        "execution_hint": "hybrid",
        "description": "Approach the relevant spot, look down, and use/place on the ground.",
        "primitive_names": ["look_down_use"],
    },
    "approach_farmland_then_plant_rows": {
        "execution_hint": "hybrid",
        "description": "Move to farmland and plant/place items in a row.",
        "primitive_names": ["look_down", "cycle_hotbar", "plant_row"],
    },
    "approach_then_interact": {
        "execution_hint": "hybrid",
        "description": "Approach a nearby interactable object, align, then use repeatedly.",
        "primitive_names": ["look_down", "hold_use"],
    },
    "place_while_walking": {
        "execution_hint": "hybrid",
        "description": "Move through an area while repeatedly placing items on the ground.",
        "primitive_names": ["cycle_hotbar", "look_down_use_walk"],
    },
    "approach_then_open_interactable": {
        "execution_hint": "hybrid",
        "description": "Approach a chest or other interactable block and tap use to open it.",
        "primitive_names": ["tap_use"],
    },
    "structured_build_sequence": {
        "execution_hint": "hybrid",
        "description": "Perform a structured repeated placement routine, then activate it.",
        "primitive_names": ["portal_frame_place", "look_down_use"],
    },
    "stack_then_trigger": {
        "execution_hint": "hybrid",
        "description": "Stack blocks and then place a special top item to trigger an effect.",
        "primitive_names": ["snow_golem_stack"],
    },
    "dig_then_fill": {
        "execution_hint": "scripted",
        "description": "Dig downward and then fill/place a block into the dug space.",
        "primitive_names": ["dig_down_attack", "fill_with_block"],
    },
    "approach_then_place_and_use": {
        "execution_hint": "hybrid",
        "description": "Approach the right surface, place a vehicle/object, then use it.",
        "primitive_names": ["boat_place_use"],
    },
    "approach_target_then_place_vehicle": {
        "execution_hint": "hybrid",
        "description": "Approach a target and place a vehicle/object near it before interacting.",
        "primitive_names": ["boat_place_use"],
    },
    "approach_height_then_use": {
        "execution_hint": "hybrid",
        "description": "Move to a suitable edge or height, then use/place downward.",
        "primitive_names": ["look_down_use"],
    },
    "line_place_repeat": {
        "execution_hint": "hybrid",
        "description": "Place items or blocks in a line while moving forward.",
        "primitive_names": ["cycle_hotbar", "place_row_walk"],
    },
    "stack_place_repeat": {
        "execution_hint": "scripted",
        "description": "Build vertically by repeated jump-place actions.",
        "primitive_names": ["stack_vertical"],
    },
    "scatter_ground_placeables": {
        "execution_hint": "hybrid",
        "description": "Select a decorative natural item (flower, plant) then scatter it on the ground while moving.",
        "primitive_names": ["look_down_use_walk"],
    },
    "place_light_sources": {
        "execution_hint": "hybrid",
        "description": "Select a torch or light-emitting block then place repeatedly while moving.",
        "primitive_names": ["look_down_use_walk"],
    },
    "approach_then_vertical_place": {
        "execution_hint": "hybrid",
        "description": "Approach a wall or vertical surface and place onto it.",
        "primitive_names": ["cycle_hotbar", "place_wall_use"],
    },
    "aim_then_use_repeat": {
        "execution_hint": "hybrid",
        "description": "Aim toward a nearby target and repeatedly use a held item.",
        "primitive_names": ["rod_cast_repeat"],
    },
    "approach_then_place_climbable": {
        "execution_hint": "hybrid",
        "description": "Approach a vertical surface, place a climbable, then climb.",
        "primitive_names": ["place_wall_use", "climb_forward"],
    },
    "navigate_then_dig_then_open": {
        "execution_hint": "hybrid",
        "description": "Navigate to an area, dig downward, then interact/open.",
        "primitive_names": ["dig_down_attack", "tap_use"],
    },
    "view_upward": {
        "execution_hint": "scripted",
        "description": "Look upward at the sky: tilt camera up strongly, then hold the sky view.",
        "primitive_names": ["look_up", "maintain_up_view"],
    },
    "melee_combat_loop": {
        "execution_hint": "hybrid",
        "description": "Approach enemy and melee-attack in repeated sprint-swing cycles.",
        "primitive_names": ["melee_attack"],
    },
    "ranged_combat_loop": {
        "execution_hint": "hybrid",
        "description": "Face target and fire ranged weapon (bow, crossbow) repeatedly.",
        "primitive_names": ["ranged_attack"],
    },
    "throw_projectile_loop": {
        "execution_hint": "hybrid",
        "description": "Face target and throw a projectile (snowball, trident, ender pearl) repeatedly.",
        "primitive_names": ["throw_weapon"],
    },
    "use_on_nearby_entity": {
        "execution_hint": "hybrid",
        "description": "Approach and use held item on a nearby entity (lead, bucket, shears, name tag, etc.).",
        "primitive_names": ["approach_and_use_on_entity"],
    },
    "shear_nearby_entity": {
        "execution_hint": "hybrid",
        "description": "Approach an animal and use shears or tool on it.",
        "primitive_names": ["shear_target"],
    },
    "clear_ground_plants": {
        "execution_hint": "scripted",
        "description": "Walk forward and attack repeatedly to clear tall grass, plants, or weeds at eye level.",
        "primitive_names": ["attack_walk_sweep"],
    },
    "collect_surface_blocks": {
        "execution_hint": "hybrid",
        "description": "Move over ground blocks and break them from above (digging surface).",
        "primitive_names": ["mine_ground"],
    },
    "chop_tree_loop": {
        "execution_hint": "hybrid",
        "description": "Face a tree and chop the trunk/leaves repeatedly.",
        "primitive_names": ["chop_tree"],
    },
    "mine_ahead_loop": {
        "execution_hint": "hybrid",
        "description": "Face a target block and mine it with the held tool.",
        "primitive_names": ["mine_forward"],
    },
    "consume_item_sequence": {
        "execution_hint": "scripted",
        "description": "Cycle hotbar and eat/drink the held consumable item (food, potion).",
        "primitive_names": ["consume_held_item"],
    },
    "open_inventory_craft": {
        "execution_hint": "hybrid",
        "description": "Open player inventory (E key) then use VLA recipe mode to craft the target item in the crafting GUI.",
        "primitive_names": ["open_inventory"],
    },
    "rest_at_object": {
        "execution_hint": "hybrid",
        "description": "Navigate to a bed or hammock and use it to rest/sleep.",
        "primitive_names": ["approach_and_use_rest_object"],
    },
    "defend_with_item": {
        "execution_hint": "scripted",
        "description": "Equip and hold a defensive item (shield, totem) continuously.",
        "primitive_names": ["hold_defensive_item"],
    },
}


def default_primitives(
    sequence_name: Optional[str],
    execution_hint: str,
    instruction: Optional[str] = None,
) -> list[dict[str, Any]]:
    if execution_hint == "vla":
        return [{
            "executor": "vla",
            "instruction": instruction or "",
            "instruction_type": "auto",
            "steps": 999999,
        }]

    if sequence_name in SEQUENCE_PRIMITIVES:
        return copy.deepcopy(SEQUENCE_PRIMITIVES[sequence_name])

    return [{
        "executor": "vla",
        "instruction": instruction or "",
        "instruction_type": "auto",
        "steps": 999999,
    }]


def sequence_catalog() -> dict[str, dict[str, Any]]:
    return copy.deepcopy(SEQUENCE_CATALOG)
