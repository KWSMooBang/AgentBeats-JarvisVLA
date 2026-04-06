"""Prompt templates for VLM-based runtime action calls.

Planning/horizon/VQA prompts are managed by planner templates.
"""

from __future__ import annotations


VLM_ACTION_SYSTEM_PROMPT = """\
You are an expert autonomous Minecraft action policy.
You receive a current observation image and an instruction.
Return ONLY one valid JSON object in this exact schema:

{
    "__action_format__": "agent",
    "action": {
        "buttons": [<int>],
        "camera": [<int>]
    }
}

Constraints:
1) __action_format__ must be exactly "agent".
2) action.buttons must be a one-element integer list.
3) action.camera must be a one-element integer list.
4) camera index range is 0..120 (11x11 grid); 60 is center view.
5) Choose purposeful actions for the instruction, not random/noisy outputs.
6) Output JSON only. No markdown. No explanation.

Action guidance:
- For collect/mine/break tasks: hold attack-like behavior consistently across steps.
- For place/use tasks: align camera toward target and apply use-like behavior.
- For navigation/find tasks: move while sweeping camera.
- For combat tasks: approach and keep offensive behavior with tracking camera.
"""


TASK_STRATEGIES = {
        "craft": "Prioritize opening inventory/crafting interaction and stable camera.",
        "build": "Prioritize placement-oriented behavior with controlled camera alignment.",
        "collect": "Prioritize repeated breaking behavior with stable aim.",
        "mine": "Prioritize sustained breaking behavior and stable view on block.",
        "combat": "Prioritize target tracking and repeated offensive behavior.",
        "kill": "Prioritize target tracking and repeated offensive behavior.",
        "find": "Prioritize movement plus camera scan sweeps.",
        "explore": "Prioritize forward progress and periodic camera scanning.",
        "drop": "Prioritize immediate drop behavior for currently selected item.",
        "look": "Prioritize camera movement over button-heavy actions.",
        "use": "Prioritize use-oriented behavior with correct camera alignment.",
}


def get_task_strategy(task_text: str) -> str:
        text = (task_text or "").lower()
        hits = [hint for key, hint in TASK_STRATEGIES.items() if key in text]
        if not hits:
                return "Use instruction-focused, deterministic action selection."
        return " ".join(hits)


def build_vlm_action_prompt(instruction: str, instruction_type: str) -> str:
    strategy = get_task_strategy(instruction)
    return (
        "# Runtime Task\n"
        f"Instruction: {instruction}\n"
        f"Instruction_type: {instruction_type}\n\n"
        "# Policy Hint\n"
        f"{strategy}\n\n"
        "# Output Requirement\n"
        "Return ONLY one JSON object with __action_format__='agent' and"
        " one-element integer lists for action.buttons/action.camera."
    )


