# JarvisVLA-based Purple Agent README

This document explains the architecture and usage of the JarvisVLA-based Purple Agent implemented under src.

## 1. Overview

This Purple Agent follows the pipeline below.

1. The Planner receives task_text and routes it to short or long horizon mode.
2. In short mode, a single instruction is executed directly. In long mode, an FSM plan is generated and executed state by state.
3. Each state instruction is executed by the JarvisVLA Runner to produce a one-step action.
4. The final action is returned in the Purple Agent compact format (buttons, camera).

Core constraints:

- VLA-only execution path (no primitive policy path)
- instruction must be a strict instructions.json key (prefix:item)
- transitions are timeout/always based (no visual-state-check transitions)

## 2. Source Structure (src)

- src/server/app.py: A2A server entrypoint, CLI parsing, AgentCard registration
- src/server/executor.py: handles A2A messages (init, obs) and returns actions
- src/agent/agent.py: orchestrates Planner + FSMExecutor + JarvisVLA Runner
- src/agent/instruction_runner.py: JarvisVLA one-step instruction runner wrapper
- src/executor/fsm_executor.py: timeout-only FSM executor
- src/planner/llm_planner.py: horizon classification, short directive generation, long plan generation
- src/planner/validator.py: plan/instruction validation
- src/planner/instruction_registry.py: instruction key normalization/validation against instructions.json
- src/protocol/models.py: A2A payload schema (Pydantic)
- src/action/converter.py: noop and action-format utilities

## 3. Execution Architecture Details

### 3.1 A2A Layer

- On init:
  - reset agent
  - initialize per-context state
  - return ack
- On obs:
  - decode base64 image
  - call agent act()
  - return action

### 3.2 Agent Layer

- reset(task_text):
  - routes with Planner (short or long)
  - short:
    - stores single instruction and instruction_type, then enters direct execution mode
  - long:
    - generates/validates/re-generates plan if needed
    - creates FSMExecutor
  - creates episode directory and saves plan.json

- act(obs, state):
  - short: executes the same instruction with JarvisVLA every step
  - long: executes current FSM-state instruction and evaluates timeout transitions
  - on termination, returns noop and saves result.json

### 3.3 Planner/Plan Rules

- short horizon:
  - JSON format: {instruction, instruction_type}
  - instruction must be a strict key (prefix:item)
- long horizon:
  - generates step-based plan and converts it to canonical FSM
  - ensures fallback state exists
  - ensures non-fallback states have timeout transitions
  - transition conditions are limited to always or timeout

### 3.4 JarvisVLA Runner

- uses jarvisvla.evaluate.agent_wrapper.VLLM_AGENT
- executes one state instruction to produce one-step action
- supports optional 21-bin to 11-bin camera conversion (enabled by default)

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
  "text": "kill_entity:enderman"
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
- result.json: execution summary

Main fields in result.json:

- execution_mode: short or long
- finished: FSM termination flag (long mode)
- result: termination reason or current status
- total_steps: accumulated steps
- final_state: last state

## 7. Standalone Benchmark Run

To run directly in the MineStudio loop:

```bash
cd /workspace/woosung/AgentBeats-JarvisVLA

python examples/run_standalone.py \
  --category combat \
  --tasks-dir ./tasks \
  --output-dir ./outputs \
  --planner-model gpt-4o \
  --vla-checkpoint-path /ABS/PATH/TO/JarvisVLA-Checkpoint \
  --vla-url http://localhost:9020/v1
```

If you prefer shell-script execution, see scripts/run_benchmark.sh.

## 8. Troubleshooting

- On startup, --vla-checkpoint-path is required:
  - A required argument is missing. Set a valid checkpoint directory.

- instruction ... is not a valid instructions.json key:
  - The instruction is not a strict prefix:item key.
  - Tighten planner prompts/task text or verify the instruction registry.

- action keeps returning noop:
  - obs was sent before init, or
  - base64 image decoding failed, or
  - JarvisVLA execution raised an exception.

## 9. Operational Recommendations

- For short-horizon tasks, prefer single-instruction direct execution.
- Reserve deep long-horizon decomposition for tasks such as ender_dragon and mine_diamond_from_scratch.
- Keep planner temperature low (for example, 0.1 to 0.2) to reduce plan variance.

## 10. References

- JarvisVLA (original repository): https://github.com/CraftJarvis/JarvisVLA
- MineStudio (framework repository): https://github.com/CraftJarvis/MineStudio
- MCU (original repository): https://github.com/CraftJarvis/MCU
- MCU-AgentBeats (MCU benchmark task repository): https://github.com/KWSMooBang/MCU-AgentBeats
- Minecraft-Benchmark-Agentbeats-Leaderboard: https://github.com/KWSMooBang/Minecraft-Benchmark-Agentbeats-Leaderboard
- Minecraft-agentbeats-leaderboard (RDI Foundation): https://github.com/RDI-Foundation/Minecraft-agentbeats-leaderboard
