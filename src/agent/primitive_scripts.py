"""Low-level scripted primitive action patterns.

Functions here return expanded environment actions.  The policy engine owns
the conversion into Purple Agent compact action indices.
"""

from __future__ import annotations

from typing import Optional


SEMANTIC_SCRIPT_NAMES = {
    "container_gui_cycle",
    "consume_held_item",
    "ranged_attack",
    "throw_weapon",
    "approach_and_use_on_entity",
    "look_down_and_use",
    "place_seed_rows",
    "approach_and_use_rest_object",
    "hold_defensive_item",
    "look_up",
    "throw_held_item",
    "stack_vertical",
    "drop_held_item",
    "melee_attack",
    "shear_target",
    "chop_tree",
    "mine_forward",
    "mine_ground",
    "place_light_source_along_path",
    "approach_then_open_interactable",
    "climb_forward",
    "maintain_up_view",
    "sprint_forward",
    "scan_rotate",
    "look_forward",
    "attack_walk_sweep",
}


def semantic_script(script_key: str, step: int) -> dict:
    if script_key == "look_up":
        if step < 20:
            return {"camera": [-10.0, 0.0]}
        return {"camera": [-1.0, 3.0]}
    if script_key == "drop_held_item":
        slot = min((step // 14) + 1, 9)
        if step % 14 == 0:
            return {f"hotbar.{slot}": 1}
        return {"drop": 1}
    if script_key == "throw_held_item":
        cycle = step % 15
        slot = (step // 15) % 4 + 1
        if cycle == 0:
            return {f"hotbar.{slot}": 1}
        if cycle < 3:
            return {"camera": [-5.0, 0.0]}
        if cycle < 7:
            return {"use": 1}
        return {}
    if script_key == "stack_vertical":
        if step < 2:
            return {"hotbar.1": 1}
        if step < 5:
            return {"camera": [9.0, 0.0]}
        return {"jump": 1, "use": 1, "sneak": 1}
    if script_key == "consume_held_item":
        slot = (step // 40) % 9 + 1
        cycle = step % 40
        if cycle == 0:
            return {f"hotbar.{slot}": 1}
        return {"use": 1}
    if script_key == "ranged_attack":
        slot = (step // 30) % 9 + 1
        cycle = step % 30
        if cycle == 0:
            return {f"hotbar.{slot}": 1}
        if cycle < 3:
            return {"camera": [-5.0, 0.0]}
        if cycle < 23:
            return {"use": 1}
        return {}
    if script_key == "throw_weapon":
        slot = (step // 20) % 9 + 1
        cycle = step % 20
        if cycle == 0:
            return {f"hotbar.{slot}": 1}
        if cycle < 3:
            return {"camera": [-3.0, 0.0]}
        if cycle < 8:
            return {"use": 1}
        return {}
    if script_key == "approach_and_use_on_entity":
        if step < 5:
            return {"forward": 1}
        step2 = step - 5
        slot = (step2 // 15) % 9 + 1
        cycle = step2 % 15
        if cycle == 0:
            return {f"hotbar.{slot}": 1}
        if cycle < 8:
            return {"use": 1}
        return {}
    if script_key == "look_down_and_use":
        cycle = step % 10
        if cycle < 4:
            return {"camera": [9.0, 0.0]}
        if cycle < 7:
            return {"use": 1}
        return {}
    if script_key == "place_seed_rows":
        if step < 4:
            return {"forward": 1}
        if step < 8:
            return {"camera": [9.0, 0.0]}
        step2 = step - 8
        slot = (step2 // 16) % 4 + 1
        cycle = step2 % 16
        if cycle == 0:
            return {f"hotbar.{slot}": 1}
        if cycle < 8:
            return {"use": 1}
        if cycle < 12:
            return {"forward": 1, "sneak": 1}
        return {}
    if script_key == "approach_and_use_rest_object":
        if step < 3:
            return {"forward": 1}
        if step < 7:
            return {"camera": [9.0, 0.0]}
        slot = (step // 25) % 9 + 1
        cycle = step % 25
        if cycle == 0:
            return {f"hotbar.{slot}": 1}
        return {"use": 1}
    if script_key == "hold_defensive_item":
        slot = (step // 30) % 9 + 1
        cycle = step % 30
        if cycle == 0:
            return {f"hotbar.{slot}": 1}
        return {"use": 1}
    if script_key == "container_gui_cycle":
        if step < 3:
            return {"forward": 1}
        if step < 6:
            return {"camera": [4.0, 0.0]}
        if step < 9:
            return {"use": 1}
        slot = (step // 15) % 9 + 1
        cycle = step % 15
        if cycle == 0:
            return {f"hotbar.{slot}": 1}
        if cycle < 5:
            return {"camera": [-3.0, 0.0]}
        if cycle < 7:
            return {"attack": 1}
        if cycle < 12:
            return {"camera": [3.0, 0.0]}
        if cycle < 14:
            return {"attack": 1}
        return {}
    if script_key == "line_place_repeat":
        return semantic_script("place_light_source_along_path", step)
    if script_key == "attack_walk_sweep":
        if step < 5:
            return {"camera": [-5.0, 0.0]}
        cycle = (step - 5) % 25
        if cycle < 3:
            return {"attack": 1}
        if cycle < 5:
            return {"forward": 1}
        if cycle < 8:
            return {"attack": 1}
        if cycle < 10:
            return {"camera": [0.0, 25.0]}
        if cycle < 12:
            return {"forward": 1}
        if cycle < 15:
            return {"attack": 1}
        if cycle < 17:
            return {"camera": [0.0, -30.0]}
        if cycle < 19:
            return {"forward": 1}
        return {"attack": 1}
    if script_key == "melee_attack":
        cycle = step % 20
        if cycle < 5:
            return {"forward": 1, "sprint": 1}
        if cycle < 12:
            return {"forward": 1, "attack": 1}
        if cycle < 15:
            return {"jump": 1, "attack": 1, "forward": 1}
        return {"camera": [0.0, 8.0]}
    if script_key == "shear_target":
        slot = (step // 18) % 9 + 1
        cycle = step % 18
        if cycle == 0:
            return {f"hotbar.{slot}": 1}
        if cycle < 6:
            return {"forward": 1}
        if cycle < 12:
            return {"use": 1}
        return {"camera": [0.0, 6.0]}
    if script_key == "chop_tree":
        slot = (step // 40) % 4 + 1
        cycle = step % 40
        if cycle == 0:
            return {f"hotbar.{slot}": 1}
        if cycle < 4:
            return {"camera": [-5.0, 0.0], "forward": 1}
        if cycle < 24:
            return {"attack": 1, "forward": 1}
        return {"camera": [6.0, 9.0]}
    if script_key == "mine_forward":
        cycle = step % 24
        if cycle == 0:
            return {"hotbar.1": 1}
        if cycle < 4:
            return {"camera": [3.0, 0.0], "forward": 1}
        if cycle < 18:
            return {"attack": 1, "forward": 1}
        return {"camera": [0.0, 10.0]}
    if script_key == "mine_ground":
        cycle = step % 20
        if cycle == 0:
            return {"hotbar.1": 1}
        if cycle < 4:
            return {"camera": [12.0, 0.0], "sneak": 1}
        if cycle < 16:
            return {"attack": 1, "sneak": 1}
        return {}
    if script_key == "place_light_source_along_path":
        slot = (step // 20) % 9 + 1
        cycle = step % 20
        if cycle == 0:
            return {f"hotbar.{slot}": 1}
        if cycle < 4:
            return {"camera": [9.0, 0.0]}
        if cycle < 8:
            return {"use": 1}
        if cycle < 14:
            return {"forward": 1, "sprint": 1}
        return {"camera": [0.0, 12.0]}
    if script_key == "approach_then_open_interactable":
        if step < 5:
            return {"forward": 1}
        if step < 8:
            return {"camera": [3.0, 0.0]}
        if step < 16:
            return {"use": 1}
        return {"camera": [0.0, 8.0]}
    if script_key == "climb_forward":
        cycle = step % 16
        if cycle < 12:
            return {"forward": 1, "jump": 1, "sprint": 1}
        return {"camera": [-2.0, 6.0]}
    if script_key == "maintain_up_view":
        cycle = step % 40
        if cycle < 20:
            return {"camera": [-1.5, 2.0]}
        return {"camera": [-1.5, -2.0]}
    if script_key == "sprint_forward":
        return {"forward": 1, "sprint": 1}
    if script_key == "scan_rotate":
        cycle = step % 30
        if cycle < 15:
            return {"camera": [0.0, 6.0]}
        return {"camera": [0.0, -6.0]}
    if script_key == "look_forward":
        return {"camera": [0.0, 0.0]}
    return {}


def primitive_env_action(
    primitive_name: str,
    local_step: int,
    script_key: Optional[str] = None,
) -> dict:
    primitive_name = (primitive_name or "").strip()

    if primitive_name in SEMANTIC_SCRIPT_NAMES:
        return semantic_script(primitive_name, local_step)
    if primitive_name == "cycle_hotbar":
        slot = (local_step // 6) % 9 + 1
        if local_step % 6 == 0:
            return {f"hotbar.{slot}": 1}
        return {}
    if primitive_name == "hold_use":
        return {"use": 1}
    if primitive_name == "tap_use":
        if local_step % 4 < 2:
            return {"use": 1}
        return {}
    if primitive_name == "hold_use_forward":
        return {"forward": 1, "use": 1}
    if primitive_name == "look_down":
        return {"camera": [9.0, 0.0]}
    if primitive_name == "look_down_use":
        cycle = local_step % 10
        if cycle < 4:
            return {"camera": [9.0, 0.0]}
        if cycle < 7:
            return {"use": 1}
        return {}
    if primitive_name == "look_down_use_walk":
        cycle = local_step % 15
        if cycle < 3:
            return {"camera": [5.0, 0.0]}
        if cycle < 6:
            return {"use": 1}
        if cycle < 10:
            return {"forward": 1, "camera": [0.0, 20.0]}
        return {"use": 1}
    if primitive_name == "plant_row":
        cycle = local_step % 16
        slot = (local_step // 16) % 4 + 1
        if cycle == 0:
            return {f"hotbar.{slot}": 1}
        if cycle < 8:
            return {"use": 1}
        if cycle < 12:
            return {"forward": 1, "sneak": 1}
        return {}
    if primitive_name == "gui_click_confirm_repeat":
        cycle = local_step % 12
        if cycle < 4:
            return {"attack": 1}
        return {}
    if primitive_name == "open_inventory":
        if local_step == 0:
            return {"inventory": 1}
        if local_step < 3:
            return {"inventory": 0}
        return {}
    if primitive_name == "cycle_hotbar_then_hold_use":
        cycle = local_step % 40
        slot = (local_step // 40) % 9 + 1
        if cycle == 0:
            return {f"hotbar.{slot}": 1}
        return {"use": 1}
    if primitive_name == "place_row_walk":
        cycle = local_step % 12
        if cycle < 3:
            return {"camera": [9.0, 0.0]}
        if cycle < 6:
            return {"use": 1}
        if cycle < 10:
            return {"forward": 1, "sneak": 1}
        return {"camera": [0.0, 8.0]}
    if primitive_name == "place_wall_use":
        if local_step < 4:
            return {"forward": 1, "sneak": 1}
        cycle = (local_step - 4) % 16
        if cycle < 5:
            return {"use": 1}
        if cycle < 7:
            return {}
        if cycle < 11:
            return {"camera": [0.0, 20.0]}
        return {"camera": [0.0, -15.0]}
    if primitive_name == "dig_down_attack":
        cycle = local_step % 12
        if cycle < 4:
            return {"camera": [12.0, 0.0], "sneak": 1}
        if cycle < 10:
            return {"attack": 1, "sneak": 1}
        return {}
    if primitive_name == "fill_with_block":
        cycle = local_step % 10
        if cycle == 0:
            return {"hotbar.2": 1}
        if cycle < 4:
            return {"camera": [12.0, 0.0]}
        if cycle < 7:
            return {"use": 1}
        return {"jump": 1}
    if primitive_name == "portal_frame_place":
        cycle = local_step % 16
        if cycle < 2:
            return {"hotbar.1": 1}
        if cycle < 5:
            return {"camera": [9.0, 0.0]}
        if cycle < 8:
            return {"use": 1, "jump": 1}
        if cycle < 12:
            return {"camera": [-9.0, 8.0]}
        return {"use": 1}
    if primitive_name == "snow_golem_stack":
        cycle = local_step % 18
        if cycle < 2:
            return {"hotbar.1": 1}
        if cycle < 5:
            return {"camera": [9.0, 0.0]}
        if cycle < 9:
            return {"use": 1, "jump": 1}
        if cycle == 9:
            return {"hotbar.2": 1}
        if cycle < 13:
            return {"use": 1}
        return {}
    if primitive_name == "rod_cast_repeat":
        cycle = local_step % 14
        if cycle < 2:
            return {"hotbar.1": 1}
        if cycle < 5:
            return {"camera": [-3.0, 0.0]}
        if cycle < 8:
            return {"use": 1}
        return {"camera": [0.0, 8.0]}
    if primitive_name == "boat_place_use":
        cycle = local_step % 14
        if cycle < 2:
            return {"hotbar.1": 1}
        if cycle < 5:
            return {"camera": [9.0, 0.0]}
        if cycle < 8:
            return {"use": 1}
        if cycle < 11:
            return {"forward": 1}
        return {"use": 1}
    if primitive_name == "throw_held_item_flow":
        return semantic_script("throw_held_item", local_step)
    if primitive_name == "drop_cycle":
        return semantic_script("drop_held_item", local_step)
    if script_key:
        return semantic_script(script_key, local_step)
    return {}
