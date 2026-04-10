"""
Prompt templates for VLM-based skill selection.

The VLMSkillRunner uses these prompts to ask an LLM/VLM:
  "Given the current screen and instruction, which skill should run next?"

Design goals:
- Output is a single skill name string (easy to parse, no ambiguity).
- Skills are described concisely so the model can reason about them.
- Stuck detection hint embedded in the prompt when history shows repetition.
"""

from __future__ import annotations

from typing import List, Optional

from src.skills.library import build_skill_menu


# ── System prompt ────────────────────────────────────────────────────────────

SKILL_SELECTOR_SYSTEM_PROMPT = """\
You are a Minecraft agent controller.
You see the current game screen and an instruction.
Your job: choose the SINGLE best skill to run next from the SKILL MENU.

Output ONLY valid JSON:
{"skill": "<skill_name>", "reason": "<one short sentence>"}

Rules:
1. skill MUST be an exact name from the SKILL MENU — no other values allowed.
2. reason is for logging only; keep it under 15 words.
3. Choose a skill that makes concrete progress toward the instruction.
4. If a GUI screen is open (inventory, crafting, furnace):
   - Use gui_cursor_* skills to navigate the cursor.
   - Use gui_click or gui_right_click to interact with slots.
   - Use close_gui only when you are done with the GUI.
5. If no progress seems possible, use a recovery skill (scan_left_right, step_back, look_up).
6. Do NOT invent skill names. Use ONLY names from the SKILL MENU.
7. Output ONLY the JSON — no markdown, no explanation outside the JSON.
"""


# ── User prompt builder ──────────────────────────────────────────────────────

def build_skill_selector_prompt(
    instruction: str,
    skill_history: List[str],
    highlight_categories: Optional[List[str]] = None,
    show_only_categories: Optional[List[str]] = None,
) -> str:
    """
    Build the user prompt for skill selection.

    Parameters
    ----------
    instruction:
        The current task/subgoal instruction the agent should fulfil.
    skill_history:
        List of recently selected skill names (most recent last).
    highlight_categories:
        Category names shown first (marked with *).
    show_only_categories:
        If given, only skills from these categories appear in the menu.
        Reduces VLM selection entropy for task-specific contexts.
    """
    menu = build_skill_menu(
        highlight_categories=highlight_categories,
        show_only_categories=show_only_categories,
    )

    history_str = (
        " → ".join(skill_history[-6:]) if skill_history else "none"
    )

    stuck_hint = _build_stuck_hint(skill_history)

    return (
        "# Current Instruction\n"
        f"{instruction}\n\n"
        "# Recent Skill History (oldest → newest)\n"
        f"{history_str}\n"
        f"{stuck_hint}"
        "# Skill Menu\n"
        f"{menu}\n\n"
        "# Task\n"
        "Choose the single best skill to run next.\n"
        "Output ONLY JSON: {\"skill\": \"<name>\", \"reason\": \"<why>\"}"
    )


def _build_stuck_hint(history: List[str]) -> str:
    """Return a warning string if the agent appears to be stuck."""
    if len(history) < 3:
        return ""
    last = history[-1]
    if all(s == last for s in history[-3:]):
        return (
            "# ⚠ Stuck Detection\n"
            f"The skill '{last}' was selected 3+ times in a row without progress.\n"
            "Pick a DIFFERENT skill to break the loop.\n\n"
        )
    return ""


# ── Addendum for re-selection after failure ──────────────────────────────────

RESELECT_ADDENDUM = """\

## Note
Previous skill selection failed or produced no visible progress.
Try a different approach for the same instruction.
"""
