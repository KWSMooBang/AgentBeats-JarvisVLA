"""
Skill Library

Reusable skills defined as sequences of env-action dicts.
Each frame is one step and is compatible with ActionConverter.env_to_agent().

Camera convention (Minecraft VPT action space):
  camera = [pitch_delta, yaw_delta] in degrees, range -10 to +10 per step.
  Positive pitch  → look DOWN
  Negative pitch  → look UP
  Positive yaw    → turn RIGHT
  Negative yaw    → turn LEFT

Button keys (partial dicts are fine — missing keys default to 0):
  attack, back, drop, forward, hotbar.1-9, inventory,
  jump, left, right, sneak, sprint, use
"""

from __future__ import annotations

from typing import Any, Dict, List

# ── type aliases ────────────────────────────────────────────────────────────
Frame = Dict[str, Any]
Sequence = List[Frame]

# ── reusable frame builders ──────────────────────────────────────────────────

def _noop(n: int = 1) -> Sequence:
    return [{"camera": [0.0, 0.0]}] * n

def _look_down(steps: int = 3, deg: float = 7.0) -> Sequence:
    return [{"camera": [deg, 0.0]}] * steps

def _look_up(steps: int = 3, deg: float = 7.0) -> Sequence:
    return [{"camera": [-deg, 0.0]}] * steps

def _turn_right(steps: int = 3, deg: float = 7.0) -> Sequence:
    return [{"camera": [0.0, deg]}] * steps

def _turn_left(steps: int = 3, deg: float = 7.0) -> Sequence:
    return [{"camera": [0.0, -deg]}] * steps

def _forward(steps: int = 10, sprint: bool = False) -> Sequence:
    f: Frame = {"forward": 1, "camera": [0.0, 0.0]}
    if sprint:
        f["sprint"] = 1
    return [f] * steps

def _attack(steps: int = 12) -> Sequence:
    return [{"attack": 1, "camera": [0.0, 0.0]}] * steps

def _use(steps: int = 8) -> Sequence:
    return [{"use": 1, "camera": [0.0, 0.0]}] * steps

def _hotbar(slot: int, hold_steps: int = 1) -> Sequence:
    key = f"hotbar.{slot}"
    return [{key: 1}] + [{key: 0}] * max(hold_steps - 1, 1)


# ── skill definitions ────────────────────────────────────────────────────────
# Each entry:
#   "description": str  – shown to VLM for skill selection
#   "sequence": Sequence – list of env-action frames (one per step)

SKILL_LIBRARY: Dict[str, Dict[str, Any]] = {

    # ────────────────────────────────────────────────────────────────────────
    # Basic movement
    # ────────────────────────────────────────────────────────────────────────
    "noop": {
        "description": "Do nothing; short pause.",
        "sequence": _noop(6),
    },
    "move_forward": {
        "description": "Walk forward for ~10 steps.",
        "sequence": _forward(10),
    },
    "move_forward_sprint": {
        "description": "Sprint forward for ~10 steps to close distance quickly.",
        "sequence": _forward(10, sprint=True),
    },
    "jump_forward": {
        "description": "Jump while moving forward, useful for climbing over a block.",
        "sequence": [{"forward": 1, "jump": 1, "camera": [0.0, 0.0]}] * 6,
    },
    "step_back": {
        "description": "Step backward a few steps.",
        "sequence": [{"back": 1, "camera": [0.0, 0.0]}] * 6,
    },

    # ────────────────────────────────────────────────────────────────────────
    # Camera / view control
    # ────────────────────────────────────────────────────────────────────────
    "look_down": {
        "description": "Tilt view downward to see the ground or a low block.",
        "sequence": _look_down(4, 7.0),
    },
    "look_up": {
        "description": "Tilt view upward toward the sky or a tall structure.",
        "sequence": _look_up(4, 7.0),
    },
    "look_left": {
        "description": "Pan view to the left.",
        "sequence": _turn_left(4, 7.0),
    },
    "look_right": {
        "description": "Pan view to the right.",
        "sequence": _turn_right(4, 7.0),
    },
    "scan_left_right": {
        "description": "Sweep camera left then right to search for a target or mob.",
        "sequence": (
            _turn_left(6, 6.0)
            + _turn_right(12, 6.0)
            + _turn_left(6, 6.0)
        ),
    },
    "scan_up_down": {
        "description": "Sweep camera up then down to inspect a tall area.",
        "sequence": (
            _look_up(4, 6.0)
            + _look_down(8, 6.0)
            + _look_up(4, 6.0)
        ),
    },

    # ────────────────────────────────────────────────────────────────────────
    # Combat
    # ────────────────────────────────────────────────────────────────────────
    "attack_forward": {
        "description": "Attack repeatedly while facing forward, for melee combat.",
        "sequence": [{"attack": 1, "forward": 1, "camera": [0.0, 0.0]}] * 12,
    },
    "attack_sweep": {
        "description": (
            "Sweep camera left-right while attacking; use when mob position is "
            "uncertain or mob is moving."
        ),
        "sequence": (
            [{"attack": 1, "camera": [0.0, -5.0]}] * 4
            + [{"attack": 1, "camera": [0.0, 5.0]}] * 8
            + [{"attack": 1, "camera": [0.0, -5.0]}] * 4
        ),
    },
    "approach_mob": {
        "description": "Sprint toward a visible mob ahead.",
        "sequence": _forward(10, sprint=True),
    },
    "bow_charge_and_release": {
        "description": "Charge bow by holding right-click then release to shoot.",
        "sequence": (
            _use(20)
            + [{"use": 0, "camera": [0.0, 0.0]}] * 3
        ),
    },

    # ────────────────────────────────────────────────────────────────────────
    # Mining / breaking
    # ────────────────────────────────────────────────────────────────────────
    "mine_forward": {
        "description": "Hold attack on the block directly ahead.",
        "sequence": _attack(15),
    },
    "mine_below": {
        "description": "Look straight down then mine the block below.",
        "sequence": (
            _look_down(3, 9.0)
            + _attack(15)
        ),
    },
    "dig_down": {
        "description": "Look fully down and dig a shaft downward.",
        "sequence": (
            _look_down(2, 9.0)
            + _attack(20)
        ),
    },

    # ────────────────────────────────────────────────────────────────────────
    # Placing / building
    # ────────────────────────────────────────────────────────────────────────
    "place_block_below": {
        "description": "Look straight down and place a block at your feet.",
        "sequence": (
            _look_down(2, 9.0)
            + _use(5)
        ),
    },
    "place_block_forward": {
        "description": "Look slightly downward-forward and place a block in front.",
        "sequence": (
            _look_down(2, 3.0)
            + _use(5)
        ),
    },
    "place_block_wall": {
        "description": (
            "Face a wall surface and place a block on it; "
            "use for decorating walls or building vertically."
        ),
        "sequence": (
            _noop(1)
            + _use(5)
            + _look_up(2, 6.0)
            + _use(5)
            + _look_down(2, 6.0)
        ),
    },
    "pillar_up": {
        "description": "Jump while sneaking and placing block below to build a pillar upward.",
        "sequence": (
            _look_down(2, 8.0)
            + [{"jump": 1, "use": 1, "sneak": 1, "camera": [0.0, 0.0]}] * 18
        ),
    },
    "place_row_forward": {
        "description": "Walk forward while sneaking and placing blocks below to lay a floor row.",
        "sequence": (
            _look_down(2, 8.0)
            + [{"forward": 1, "use": 1, "sneak": 1, "camera": [0.0, 0.0]}] * 12
        ),
    },
    "sneak_place": {
        "description": "Sneak and place at an edge; use for bridging over a gap.",
        "sequence": [{"sneak": 1, "use": 1, "camera": [0.0, 0.0]}] * 8,
    },

    # ────────────────────────────────────────────────────────────────────────
    # Inventory & GUI interaction
    # ────────────────────────────────────────────────────────────────────────
    "open_inventory": {
        "description": "Press E to open the player inventory screen.",
        "sequence": (
            [{"inventory": 1}]
            + [{"inventory": 0}] * 2
            + _noop(3)
        ),
    },
    "close_gui": {
        "description": "Close any open GUI (inventory, crafting table, furnace) by pressing E.",
        "sequence": (
            [{"inventory": 1}]
            + [{"inventory": 0}] * 2
            + _noop(3)
        ),
    },
    "open_crafting_table": {
        "description": (
            "Face a crafting table and right-click to open the 3x3 crafting GUI."
        ),
        "sequence": (
            _look_down(2, 3.0)
            + [{"use": 1, "camera": [0.0, 0.0]}] * 3
            + [{"use": 0, "camera": [0.0, 0.0]}] * 2
            + _noop(2)
        ),
    },
    "gui_cursor_left": {
        "description": "Move GUI cursor to the left (while inventory or crafting screen is open).",
        "sequence": _turn_left(5, 4.0),
    },
    "gui_cursor_right": {
        "description": "Move GUI cursor to the right.",
        "sequence": _turn_right(5, 4.0),
    },
    "gui_cursor_up": {
        "description": "Move GUI cursor upward.",
        "sequence": _look_up(5, 4.0),
    },
    "gui_cursor_down": {
        "description": "Move GUI cursor downward.",
        "sequence": _look_down(5, 4.0),
    },
    "gui_click": {
        "description": "Left-click the slot under the GUI cursor to pick up or place an item.",
        "sequence": (
            [{"attack": 1}] * 2
            + [{"attack": 0}] * 2
        ),
    },
    "gui_right_click": {
        "description": "Right-click the slot under cursor to split stack or place one item.",
        "sequence": (
            [{"use": 1}] * 2
            + [{"use": 0}] * 2
        ),
    },

    # ────────────────────────────────────────────────────────────────────────
    # Tool use / item interactions
    # ────────────────────────────────────────────────────────────────────────
    "use_item_held": {
        "description": "Right-click with the currently held item (activate / use).",
        "sequence": (
            _use(5)
            + [{"use": 0, "camera": [0.0, 0.0]}] * 2
        ),
    },
    "use_item_on_ground": {
        "description": "Look at the ground and right-click to place or use the held item there.",
        "sequence": (
            _look_down(2, 6.0)
            + _use(5)
            + [{"use": 0, "camera": [0.0, 0.0]}]
        ),
    },
    "drink_use": {
        "description": "Hold right-click to drink a potion or eat food (takes ~30 steps).",
        "sequence": _use(30),
    },
    "throw_item": {
        "description": "Quick right-click to throw the held item (snowball, trident, etc.).",
        "sequence": (
            [{"use": 1, "camera": [0.0, 0.0]}] * 2
            + [{"use": 0, "camera": [0.0, 0.0]}] * 3
        ),
    },
    "drop_held_item": {
        "description": "Drop the currently held item on the ground.",
        "sequence": (
            [{"drop": 1}] * 2
            + [{"drop": 0}] * 2
        ),
    },
    "ignite_ground": {
        "description": "Look at the ground and right-click with flint-and-steel to ignite it.",
        "sequence": (
            _look_down(2, 6.0)
            + [{"use": 1, "camera": [0.0, 0.0]}] * 3
            + [{"use": 0, "camera": [0.0, 0.0]}]
        ),
    },
    "sleep_in_bed": {
        "description": "Face a bed and right-click to sleep; looks slightly downward to aim at bed.",
        "sequence": (
            _look_down(2, 5.0)
            + _use(5)
            + [{"use": 0, "camera": [0.0, 0.0]}] * 3
        ),
    },
    "plant_on_ground": {
        "description": "Look at tilled soil and right-click to plant seeds or place crop items.",
        "sequence": (
            _look_down(2, 7.0)
            + _use(8)
            + [{"use": 0, "camera": [0.0, 0.0]}]
        ),
    },
    "place_on_wall": {
        "description": "Look at a vertical wall surface and right-click to hang an item frame or painting.",
        "sequence": (
            _look_up(1, 3.0)
            + _use(5)
            + [{"use": 0, "camera": [0.0, 0.0]}] * 2
        ),
    },
    "use_furnace": {
        "description": "Face a furnace and right-click to open its GUI.",
        "sequence": (
            _look_down(2, 3.0)
            + [{"use": 1, "camera": [0.0, 0.0]}] * 3
            + [{"use": 0, "camera": [0.0, 0.0]}] * 2
            + _noop(2)
        ),
    },

    # ────────────────────────────────────────────────────────────────────────
    # Hotbar selection
    # ────────────────────────────────────────────────────────────────────────
    "select_hotbar_1": {
        "description": "Switch active hotbar slot to slot 1.",
        "sequence": _hotbar(1),
    },
    "select_hotbar_2": {
        "description": "Switch active hotbar slot to slot 2.",
        "sequence": _hotbar(2),
    },
    "select_hotbar_3": {
        "description": "Switch active hotbar slot to slot 3.",
        "sequence": _hotbar(3),
    },
    "select_hotbar_4": {
        "description": "Switch active hotbar slot to slot 4.",
        "sequence": _hotbar(4),
    },
    "select_hotbar_5": {
        "description": "Switch active hotbar slot to slot 5.",
        "sequence": _hotbar(5),
    },
    "select_hotbar_6": {
        "description": "Switch active hotbar slot to slot 6.",
        "sequence": _hotbar(6),
    },
    "select_hotbar_7": {
        "description": "Switch active hotbar slot to slot 7.",
        "sequence": _hotbar(7),
    },
    "select_hotbar_8": {
        "description": "Switch active hotbar slot to slot 8.",
        "sequence": _hotbar(8),
    },
    "select_hotbar_9": {
        "description": "Switch active hotbar slot to slot 9.",
        "sequence": _hotbar(9),
    },

    # ────────────────────────────────────────────────────────────────────────
    # Motion task direct scripts (bypass VLM skill selection)
    # ────────────────────────────────────────────────────────────────────────
    "motion_look_at_sky": {
        "description": "Aggressively tilt view upward to face the sky, then hold and pan.",
        "sequence": (
            [{"camera": [-10.0, 0.0]}] * 15   # slam up — guaranteed sky view
            + [{"camera": [0.0, 0.0]}] * 5    # hold
            + [{"camera": [-1.0, 3.0]}] * 10  # slow pan while looking up (video score)
            + [{"camera": [0.0, 0.0]}] * 5
        ),
    },
    "motion_drop_item": {
        "description": "Cycle through hotbar slots 1-3 and drop items from each.",
        "sequence": (
            [{"hotbar.1": 1}] + [{"hotbar.1": 0}]   # select slot 1
            + [{"drop": 1}] * 12 + [{"drop": 0}] * 2
            + [{"hotbar.2": 1}] + [{"hotbar.2": 0}]  # select slot 2
            + [{"drop": 1}] * 12 + [{"drop": 0}] * 2
            + [{"hotbar.3": 1}] + [{"hotbar.3": 0}]  # select slot 3
            + [{"drop": 1}] * 12 + [{"drop": 0}] * 2
            + [{"hotbar.4": 1}] + [{"hotbar.4": 0}]  # select slot 4
            + [{"drop": 1}] * 12 + [{"drop": 0}] * 2
        ),
    },
    "motion_throw_snowball": {
        "description": "Select throwable items from hotbar and throw them with aim.",
        "sequence": (
            [{"hotbar.1": 1}] + [{"hotbar.1": 0}]   # select slot 1
            + [{"camera": [-5.0, 0.0]}] * 2          # aim slightly up
            + [{"use": 1}] * 3 + [{"use": 0}] * 3
            + [{"hotbar.2": 1}] + [{"hotbar.2": 0}]  # select slot 2
            + [{"camera": [-5.0, 0.0]}] * 2
            + [{"use": 1}] * 3 + [{"use": 0}] * 3
            + [{"hotbar.3": 1}] + [{"hotbar.3": 0}]  # select slot 3
            + [{"camera": [-5.0, 0.0]}] * 2
            + [{"use": 1}] * 3 + [{"use": 0}] * 3
            + [{"hotbar.4": 1}] + [{"hotbar.4": 0}]  # select slot 4
            + [{"camera": [-5.0, 0.0]}] * 2
            + [{"use": 1}] * 3 + [{"use": 0}] * 3
        ),
    },
    "motion_stack_fence": {
        "description": "Select fence from hotbar and stack blocks vertically by jumping and placing.",
        "sequence": (
            [{"hotbar.1": 1}] + [{"hotbar.1": 0}]   # select slot 1 (likely fence)
            + _look_down(2, 9.0)
            + [{"jump": 1, "use": 1, "sneak": 1, "camera": [0.0, 0.0]}] * 20
            + [{"hotbar.2": 1}] + [{"hotbar.2": 0}]  # try slot 2
            + _look_down(1, 5.0)
            + [{"jump": 1, "use": 1, "sneak": 1, "camera": [0.0, 0.0]}] * 20
        ),
    },
}

# ── Categories for prompt building ──────────────────────────────────────────
SKILL_CATEGORIES: Dict[str, List[str]] = {
    "movement": [
        "noop", "move_forward", "move_forward_sprint", "jump_forward", "step_back",
    ],
    "camera": [
        "look_down", "look_up", "look_left", "look_right",
        "scan_left_right", "scan_up_down",
    ],
    "combat": [
        "attack_forward", "attack_sweep", "approach_mob", "bow_charge_and_release",
    ],
    "mining": [
        "mine_forward", "mine_below", "dig_down",
    ],
    "building": [
        "place_block_below", "place_block_forward", "place_block_wall",
        "pillar_up", "place_row_forward", "sneak_place",
    ],
    "inventory_gui": [
        "open_inventory", "close_gui", "open_crafting_table", "use_furnace",
        "gui_cursor_left", "gui_cursor_right", "gui_cursor_up", "gui_cursor_down",
        "gui_click", "gui_right_click",
    ],
    "tool_use": [
        "use_item_held", "use_item_on_ground", "drink_use", "throw_item",
        "drop_held_item", "ignite_ground", "sleep_in_bed",
        "plant_on_ground", "place_on_wall",
    ],
    "hotbar": [
        "select_hotbar_1", "select_hotbar_2", "select_hotbar_3",
        "select_hotbar_4", "select_hotbar_5", "select_hotbar_6",
        "select_hotbar_7", "select_hotbar_8", "select_hotbar_9",
    ],
    "motion": [
        "motion_look_at_sky", "motion_drop_item",
        "motion_throw_snowball", "motion_stack_fence",
    ],
}

# ── Public helpers ───────────────────────────────────────────────────────────

def get_skill_names() -> List[str]:
    """Return sorted list of all skill names."""
    return sorted(SKILL_LIBRARY.keys())


def get_skill_sequence(skill_name: str) -> Sequence:
    """Return the action sequence for a skill, or a noop sequence if not found."""
    entry = SKILL_LIBRARY.get(skill_name)
    if entry is None:
        return _noop(5)
    return list(entry["sequence"])


def get_skill_description(skill_name: str) -> str:
    """Return the description string for a skill."""
    entry = SKILL_LIBRARY.get(skill_name)
    if entry is None:
        return "Unknown skill."
    return entry["description"]


def build_skill_menu(
    highlight_categories: List[str] | None = None,
    show_only_categories: List[str] | None = None,
) -> str:
    """
    Build a compact menu of skills grouped by category for use in prompts.

    Parameters
    ----------
    highlight_categories:
        List those categories first (shown with a * prefix).
    show_only_categories:
        If given, only include skills from these categories.
        This reduces VLM selection entropy for task-specific contexts.
        Always adds the 'movement' and 'camera' base categories unless
        they are already included.
    """
    # Determine which categories to show.
    if show_only_categories is not None:
        # Always include movement + camera as navigation baseline.
        base = ["movement", "camera"]
        visible = list(dict.fromkeys(show_only_categories + base))
        visible = [c for c in visible if c in SKILL_CATEGORIES]
    else:
        visible = list(SKILL_CATEGORIES.keys())

    # Order: highlighted first, then the rest.
    if highlight_categories:
        ordered = [c for c in highlight_categories if c in visible] + [
            c for c in visible if c not in highlight_categories
        ]
    else:
        ordered = visible

    lines: List[str] = []
    for cat in ordered:
        prefix = "* " if highlight_categories and cat in highlight_categories else "  "
        lines.append(f"{prefix}[{cat.upper()}]")
        for name in SKILL_CATEGORIES[cat]:
            desc = get_skill_description(name)
            lines.append(f"    {name}: {desc}")

    return "\n".join(lines)
