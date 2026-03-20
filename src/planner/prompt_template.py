"""
Prompt templates for the LLM Planner.

The planner supports three LLM calls:
1) Horizon classification (short vs long)
2) Short-horizon direct instruction generation
3) Long-horizon policy planning
"""

from __future__ import annotations


LONG_HORIZON_CATEGORIES = frozenset({"ender_dragon", "mine_diamond_from_scratch"})


def classify_task_horizon(task_text: str) -> str:
    """Heuristic fallback when horizon-classification LLM output is invalid."""
    text = (task_text or "").lower()

    if "[task_horizon: long]" in text:
        return "long"
    if "[task_horizon: short]" in text:
        return "short"

    for cat in LONG_HORIZON_CATEGORIES:
        if f"[task_category: {cat}]" in text:
            return "long"

    if "ender dragon" in text or "kill_ender_dragon" in text:
        return "long"
    if "mine_diamond_from_scratch" in text:
        return "long"
    if "diamond" in text and "from scratch" in text:
        return "long"

    return "short"


HORIZON_SYSTEM_PROMPT = """\
You are a strict task horizon classifier for Minecraft tasks.
Return ONLY JSON: {"horizon": "short"} or {"horizon": "long"}.

Use "long" only when the task clearly needs multi-stage long-horizon planning
(e.g., kill_ender_dragon, mine_diamond_from_scratch).
Otherwise return "short".
"""


def build_horizon_prompt(task_text: str) -> str:
    return (
        "Classify this task horizon.\n"
        "Return JSON with key horizon only.\n\n"
        f"Task: {task_text}"
    )


SHORT_DIRECTIVE_SYSTEM_PROMPT = """\
You produce a direct JarvisVLA execution directive for short-horizon tasks.
Return ONLY JSON:
{
  "instruction": "<EXACT instructions.json key in prefix:item format>",
  "instruction_type": "<auto|simple|normal|recipe>"
}

Rules:
1. instruction MUST be an EXACT instructions.json root key.
2. instruction MUST use prefix:item syntax (exactly one colon).
3. instruction_type MUST be one of: auto, simple, normal, recipe.
4. Output ONLY the JSON.
"""


def build_short_directive_prompt(task_text: str) -> str:
    return (
        "Generate direct execution directive for this short task.\n"
        "Output JSON only.\n\n"
        f"Task: {task_text}"
    )


PLANNER_SYSTEM_PROMPT = """\
You are a Minecraft task planner. Given a task description you MUST output
an actionable long-horizon Plan JSON with staged step decomposition.

IMPORTANT: The agent can only see observation images and has NO access
to inventory data, health, or any game info. All state checks must be done
via VLM visual queries (vlm_check) or step counting (timeout).

## Transition Conditions
| type            | extra fields                 | description                           |
|-----------------|------------------------------|---------------------------------------|
| always          | -                            | Unconditional transition              |
| vlm_check       | query: str                   | VLM yes/no visual question            |
| inventory_has   | item: str, count: int        | VLM-inferred inventory check (visual) |
| timeout         | max_steps: int               | Steps spent in this step              |
| retry_exhausted | -                            | Exceeded step's max_retries           |
| scene_check     | description: str             | VLM scene-matching check              |

## Output Format (Simplified)

```json
{
  "task": "<task_name>",
  "step1": {
    "instruction": "<EXACT instructions.json key in prefix:item format>",
    "instruction_type": "<auto|simple|normal|recipe>",
    "condition": {
      "type": "<condition_type>",
      "next": "step2"
    }
  },
  "step2": {
    "instruction": "<EXACT instructions.json key in prefix:item format>",
    "instruction_type": "<auto|simple|normal|recipe>",
    "condition": {
      "type": "timeout",
      "max_steps": 180,
      "next": "fallback"
    }
  },
  "fallback": {
    "instruction": "<RAW task_text string>",
    "instruction_type": "normal",
    "condition": {"type": "always", "next": "step1"}
  }
}
```

## Rules
1. Use top-level step entries (e.g. step1, step2), not nested states.
2. Do NOT output top-level success or abort entries.
3. Every non-terminal step MUST include instruction, instruction_type, and condition.
4. instruction MUST be an EXACT instructions.json root key in prefix:item format.
5. instruction_type MUST be one of: auto, simple, normal, recipe.
6. Include a dedicated non-terminal fallback step.
7. fallback.instruction MUST be the raw input task_text string.
8. For steps that can stall, route timeout recovery to fallback.
9. Use vlm_check sparingly - only at step boundaries.
10. Execution is VLA-only at runtime. Use instruction + condition logic.
11. Output ONLY the JSON, no explanation.
"""


def build_planner_prompt(task_text: str) -> str:
    return (
        "Task horizon is already LONG. Build a staged long-horizon plan.\n"
        "Output Plan JSON only.\n"
        "Ensure fallback.instruction is exactly this task_text string.\n\n"
        f"Task: {task_text}"
    )


REPLAN_ADDENDUM = """\

## Validation Errors from Previous Attempt
The following errors were found. Fix them and regenerate:
{errors}
"""
