"""
Prompt templates for the LLM Planner.

The planner supports three LLM calls:
1) Horizon classification (short vs long)
2) Short-horizon direct instruction generation
3) Long-horizon policy planning
"""

from __future__ import annotations


def classify_task_horizon(task_text: str) -> str:
    """
    Heuristic fallback when horizon-classification LLM output is invalid.
    """
    text = (task_text or "").lower()

    # Multi-step indicators: tasks requiring preparation or full progression
    long_indicators = [
        "from the starting",
        "from the start",
        "starting from scratch",
        "with empty inventory",
        "starting with empty",
        "from scratch",
    ]
    
    for indicator in long_indicators:
        if indicator in text:
            return "long"
        
    return "short"


HORIZON_SYSTEM_PROMPT = """\
You are a strict task horizon classifier for Minecraft tasks.
Return ONLY JSON: {"horizon": "short"} or {"horizon": "long"}.

Use "long" only when the task clearly needs multi-stage long-horizon planning
that start from scratch or initial empty inventory.
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

IMPORTANT: The agent uses ONLY visual instruction execution (VLA) and step
counting for state transitions. All transitions are TIMEOUT-ONLY.

## Timing Reference
- All actions run at ~20 steps per second (50ms per step)
- Execution time = max_steps / 20
- Examples: 600 steps = 30s, 1200 steps = 60s (1min), 1800 steps = 90s, 2400 steps = 120s (2min)

## Transition Conditions (TIMEOUT-ONLY)
| type            | extra fields       | description                              |
|-----------------|--------------------|------------------------------------------|
| always          | -                  | Unconditional transition                 |
| timeout         | max_steps: int     | After max_steps in this step → next      |

CRITICAL: 
- ALL transitions MUST use "type": "timeout" with appropriate max_steps.
- NO VLM queries or state checks allowed.
- Set max_steps GENEROUSLY per subgoal type (see guidelines below).
- Each step is a single VLA instruction repeated for up to max_steps.

## Max Steps Guidelines (GENEROUS Per-Subgoal Time Allocation)
Each subgoal gets 40-120 seconds. Set appropriate max_steps based on instruction type:

| Instruction Type | Typical Time | Recommended max_steps |
|---|---|---|
| move_to_* (navigation) | 40-60 seconds | 800-1200 |
| mine_block:* | 50-80 seconds | 1000-1600 |
| kill_entity:* (combat) | 60-90 seconds | 1200-1800 |
| craft_item:* | 40-60 seconds | 800-1200 |
| pickup:*, use_item:* | 30-40 seconds | 600-800 |
| drop:* | 20-30 seconds | 400-600 |
| fallback (recovery) | 60-120 seconds | 1200-2400 |

IMPORTANT NOTES:
- Each subgoal typically takes 1-2 minutes when fallback is included
- Complex subgoals: 1200-1800 steps (60-90 seconds)
- Moderate subgoals: 800-1200 steps (40-60 seconds)
- Simple subgoals: 600-800 steps (30-40 seconds)
- Fallback recovery gets plenty of time for retries
- Total episode time budget: up to 2 minutes

## Output Format (Simplified, TIMEOUT-ONLY)

```json
{
  "task": "<task_name>",
  "step1": {
    "instruction": "<EXACT instructions.json key in prefix:item format>",
    "instruction_type": "<auto|simple|normal|recipe>",
    "condition": {
      "type": "timeout",
      "max_steps": 1000,
      "next": "step2"
    }
  },
  "step2": {
    "instruction": "<EXACT instructions.json key in prefix:item format>",
    "instruction_type": "<auto|simple|normal|recipe>",
    "condition": {
      "type": "timeout",
      "max_steps": 1200,
      "next": "fallback"
    }
  },
  "fallback": {
    "instruction": "<RAW task_text string>",
    "instruction_type": "normal",
    "condition": {"type": "always", "next": "fallback"}
  }
}
```

## Rules
1. Use top-level step entries (e.g. step1, step2), not nested states.
2. Do NOT output top-level success or abort entries.
3. Every non-terminal step MUST have: instruction, instruction_type, condition.
4. instruction MUST be an EXACT instructions.json root key (prefix:item format).
5. instruction_type MUST be one of: auto, simple, normal, recipe.
6. Include a dedicated non-terminal fallback step.
7. fallback.instruction MUST be the raw input task_text string.
8. fallback.condition MUST be {"type": "always", "next": "fallback"}.
9. Every non-fallback step condition MUST be "type": "timeout" with max_steps.
10. Use the max_steps guidelines above; prefer 600-1800 steps per subgoal for 1-2 minute total time.
11. Execution is VLA-only. Step transitions happen purely on step counts.
12. Output ONLY the JSON, no explanation.
"""


def build_planner_prompt(task_text: str) -> str:
    return (
        "Task horizon is already LONG. Build a staged long-horizon plan.\n"
        "Output Plan JSON only.\n"
        "Ensure fallback.instruction is exactly this task_text string.\n"
        "Use GENEROUS max_steps per subgoal (600-1800 steps typical, up to 2400 for fallback).\n"
        "Total episode should take 1-2 minutes to complete.\n\n"
        f"Task: {task_text}"
    )


REPLAN_ADDENDUM = """\

## Validation Errors from Previous Attempt
The following errors were found. Fix them and regenerate:
{errors}
"""
