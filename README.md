# JarvisVLA-based Purple Agent README

This document explains the architecture and usage of the JarvisVLA-based Purple Agent implemented under src.

## 1. Overview

This Purple Agent follows the pipeline below.

1. The Planner receives task_text and classifies it as short or long horizon.
2. In short mode, a single instruction is executed directly. In long mode, an FSM plan is generated and executed state by state.
3. Each instruction is dispatched through the FallbackPolicyEngine, which routes it to either a scripted sequence or JarvisVLA.
4. Scripted sequences are hybrid: VLA handles navigation and item selection, deterministic primitives handle the core action (place, sweep, etc.).
5. The final action is returned in the Purple Agent compact format (buttons, camera).

Core constraints:

- Scripted sequences take routing priority over VLA when a keyword match exists.
- Transitions in the FSM are timeout-only (no visual-state-check transitions).
- A GPU with at least 24GB VRAM is required for JarvisVLA inference.

## 2. Source Structure (src)

- src/server/app.py: A2A server entrypoint, CLI parsing, AgentCard registration
- src/server/executor.py: handles A2A messages (init, obs) and returns actions
- src/agent/agent.py: orchestrates Planner + FSMExecutor + FallbackPolicyEngine
- src/agent/fallback_policy.py: routes instructions to scripted sequences or VLA; defines sequence templates and scripted primitives
- src/agent/sequence_router.py: keyword-based routing from instruction text to sequence name
- src/agent/vla_runner.py: JarvisVLA one-step instruction runner wrapper
- src/executor/fsm_executor.py: timeout-only FSM executor
- src/planner/planner.py: horizon classification, short directive generation, long plan generation
- src/planner/prompt_template.py: LLM prompt templates for planner
- src/planner/validator.py: plan/instruction validation
- src/planner/instruction_registry.py: instruction key normalization/validation
- src/protocol/models.py: A2A payload schema (Pydantic)
- src/action/converter.py: noop and action-format utilities

## 3. Execution Architecture Details

### 3.1 A2A Layer

- On init:
  - reset agent with task_text
  - return ack
- On obs:
  - decode base64 image
  - call agent act()
  - return action

### 3.2 Agent Layer

reset(task_text):

- 5 startup noop frames to let the environment settle
- _post_startup_assess() classifies horizon and initializes execution
  - short: generates single instruction + instruction_type, enters direct execution mode
  - long: generates FSM plan, creates FSMExecutor
- creates episode directory and saves plan.json

act(obs, state):

- short: runs the same instruction through FallbackPolicyEngine every step
- long: FSMExecutor advances through FSM states, each step runs through FallbackPolicyEngine
- on termination, returns noop and saves result.json

### 3.3 Planner

- Horizon classification via LLM (HORIZON_SYSTEM_PROMPT)
  - SHORT: single instruction family (kill, mine, craft, pickup, use, drop)
  - LONG: multi-step sequential tasks (mine → smelt → craft chains)
- Short horizon: returns {instruction, instruction_type}; instruction is a canonical prefix:item key
- Long horizon: generates step-based FSM plan; ensures fallback state and timeout transitions
- Transitions are limited to always or timeout conditions

### 3.4 FallbackPolicyEngine

Routing order in make_policy_spec():

1. task_text keyword match (overrides planner intent — prevents misrouting like "lay carpet" → craft route)
2. instruction keyword match via SequenceRouter
3. planner execution_hint fallback (vla / scripted / hybrid)

Sequence templates map a sequence name to an ordered list of {executor, primitive/instruction, steps}. VLA steps handle navigation and item selection; script steps execute deterministic motor actions. Key sequences:

- open_inventory_craft: open inventory → VLA crafts item in recipe mode
- line_place_repeat: open inventory → VLA crafts carpet from wool → VLA selects carpet → place in line
- scatter_ground_placeables: VLA selects flower → scatter on ground while walking
- place_light_sources: VLA selects torch → place while walking
- approach_then_vertical_place: VLA faces wall → VLA selects item → place on wall face
- clear_ground_plants: VLA selects tool → sweep attack across ground
- approach_farmland_then_plant_rows: VLA moves to farmland → cycle hotbar → plant rows

### 3.5 SequenceRouter

Keyword-based routing with underscore normalization:

- task_texts from the benchmark arrive with underscores (e.g. "lay_carpet", "light_up_the_surroundings")
- underscores are normalized to spaces before keyword matching so space-based keywords match correctly

### 3.6 JarvisVLA Runner

- uses jarvisvla.evaluate.agent_wrapper.VLLM_AGENT
- executes one instruction per step using image history + instruction text
- instruction_type: normal (natural language) or recipe (crafting GUI mode)
- supports 21-bin to 11-bin camera conversion (enabled by default)
- VLA handles negation instructions poorly; use positive descriptions for item selection

## 4. Server Run Guide

### 4.1 Minimal Run

```bash
cd /workspace/woosung/AgentBeats-JarvisVLA

python -m src.server.app \
  --host 0.0.0.0 \
  --port 9019 \
  --planner-model gpt-4o \
  --vla-checkpoint-path /ABS/PATH/TO/JarvisVLA-Checkpoint \
  --vla-url http://localhost:8000/v1
```

--vla-checkpoint-path is required.

### 4.2 Main CLI Arguments

Planner:

- --planner-api-key
- --planner-url (default: https://api.openai.com/v1)
- --planner-model (default: gpt-4o)
- --planner-temperature (default: 0.2)

VLA:

- --vla-checkpoint-path (required)
- --vla-url (default: http://localhost:8000/v1)
- --vla-api-key
- --vla-history-num (default: 4)
- --vla-action-chunk-len (default: 1)
- --vla-bpe (default: 0)
- --vla-instruction-type (default: normal)
- --vla-temperature (default: 0.7)
- --vla-no-camera-convert (disables 21->11 camera conversion)

## 5. A2A Message Protocol

### 5.1 Init Request

```json
{
  "type": "init",
  "text": "lay_carpet"
}
```

Response:

```json
{
  "type": "ack",
  "success": true,
  "message": "Initialized"
}
```

### 5.2 Observation Request

```json
{
  "type": "obs",
  "step": 1,
  "obs": "<base64-encoded-rgb-image>"
}
```

Response:

```json
{
  "type": "action",
  "action_type": "agent",
  "buttons": [0],
  "camera": [60]
}
```

Even in failure cases (before init, decode failure, etc.), the server safely returns a noop action.

## 6. Output Artifacts

The following files are saved per episode.

- plan.json: planner output plan
- result.json: execution summary including skill_log (sequence of sequences used)

Main fields in result.json:

- execution_mode: short or long
- finished: FSM termination flag (long mode)
- result: termination reason or current status
- total_steps: accumulated steps
- final_state: last FSM state

## 7. Troubleshooting

- --vla-checkpoint-path is required on startup.

- action keeps returning noop:
  - obs was sent before init, or
  - base64 image decoding failed, or
  - JarvisVLA execution raised an exception.

- task routes to wrong sequence:
  - Check SequenceRouter._keyword_match keyword order.
  - task_text routing overrides planner intent; verify task_text keywords match intended sequence.

## 8. Operational Recommendations

- Keep planner temperature low (0.1 to 0.2) to reduce plan variance.
- Do not use negation in VLA item selection instructions ("NOT a flower pot"). Use positive descriptions only.
- Reserve long-horizon mode for tasks with genuine multi-step dependencies.

## 9. References

- JarvisVLA (original repository): https://github.com/CraftJarvis/JarvisVLA
- MineStudio (framework repository): https://github.com/CraftJarvis/MineStudio
- MCU (original repository): https://github.com/CraftJarvis/MCU
- MCU-AgentBeats (MCU benchmark task repository): https://github.com/KWSMooBang/MCU-AgentBeats
- Minecraft-Benchmark-Agentbeats-Leaderboard: https://github.com/KWSMooBang/Minecraft-Benchmark-Agentbeats-Leaderboard
- Minecraft-agentbeats-leaderboard (RDI Foundation): https://github.com/RDI-Foundation/Minecraft-agentbeats-leaderboard
