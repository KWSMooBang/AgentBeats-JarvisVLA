# JarvisVLA-based Purple Agent

This repository implements a Purple Agent for Minecraft benchmark tasks by combining a planner, a timeout-driven FSM executor, JarvisVLA inference, and an optional script policy layer.

## 1. What the agent actually does

The agent is not a single monolithic policy. It runs a control loop with four distinct stages.

1. The planner reads the task text and classifies it as short-horizon or long-horizon.
2. Short-horizon tasks become one direct instruction. Long-horizon tasks become a multi-state FSM plan.
3. Each state instruction is executed by JarvisVLA, which consumes the current image and returns one compact action.
4. If a scripted sequence is a better fit, the sequence router can select it instead of pure VLA execution.

The most important design choice is that state transitions are not driven by visual-state hallucination checks. They are driven by step counts, timeouts, and explicit always transitions.

Core constraints:

- The main execution path is VLA-first.
- Every instruction must resolve to a strict registry key in prefix:item form.
- Long-horizon plans must contain a fallback state.
- Transitions are limited to always and timeout.
- JarvisVLA inference requires a GPU with enough memory for the selected checkpoint.

## 2. Internal control loop

The runtime flow is easiest to understand as a state machine around a single environment loop.

```text
task_text
  ↓
Planner.classify_horizon()
  ↓
short  → store one instruction and keep executing it every step
long   → build FSM plan, validate it, and hand it to FSMExecutor
  ↓
act(obs)
  ↓
if short: run the direct instruction through JarvisVLA
if long:  ask FSMExecutor for the current state's instruction
  ↓
VLARunner.run()
  ↓
normalize action to {buttons, camera}
  ↓
return action
```

Short mode is deliberately simple. The same instruction is repeated every step, which lets JarvisVLA adapt to the current image without replanning.

Long mode is explicit and stateful. Each FSM node owns one instruction, one instruction type, and one timeout budget. When the budget expires, the executor moves to the next state.

## 3. Source structure

- [src/agent/agent.py](src/agent/agent.py): top-level orchestration for planner, executor, and fallback policy
- [src/agent/vla_runner.py](src/agent/vla_runner.py): JarvisVLA wrapper and action normalization
- [src/agent/fallback_policy.py](src/agent/fallback_policy.py): route selection between VLA and scripted sequences
- [src/agent/sequence_router.py](src/agent/sequence_router.py): keyword and hint based sequence selection
- [src/executor/fsm_executor.py](src/executor/fsm_executor.py): timeout-only FSM interpreter
- [src/planner/planner.py](src/planner/planner.py): horizon classification and plan generation
- [src/planner/validator.py](src/planner/validator.py): plan validation and strict instruction checks
- [src/planner/instruction_registry.py](src/planner/instruction_registry.py): registry normalization for strict instruction keys
- [src/action/converter.py](src/action/converter.py): noop action and compact action format helpers
- [src/server/app.py](src/server/app.py): A2A server entrypoint and CLI wiring

## 4. Planner behavior

### 4.1 Horizon classification

`classify_horizon()` decides whether the task is short or long. The decision is based on task complexity, not on a fixed keyword list. The result changes the entire execution strategy.

- Short horizon: direct instruction execution.
- Long horizon: multi-step plan generation and FSM execution.

### 4.2 Short directive generation

For short tasks, the planner returns a compact directive with two fields.

```json
{
  "instruction": "kill_entity:zombie",
  "instruction_type": "normal"
}
```

The instruction must be a strict registry key. The planner canonicalizes task text, filters candidates, and prefers valid prefix:item keys from the instruction registry.

### 4.3 Long plan generation

For long tasks, the planner builds a simplified FSM plan.

Each state usually contains:

- `instruction`: the action to attempt in that state
- `instruction_type`: normal, recipe, or simple
- `transitions`: only always or timeout rules

The fallback state matters. It is the recovery path when the main decomposition is not enough. In practice it often contains the original task instruction so the agent can retry the full task after a failed decomposition.

### 4.4 Validation

Before the plan is used, the validator checks that:

- every instruction resolves to a valid registry key
- the state graph is structurally valid
- fallback exists
- transition types stay within the supported set

If validation fails, the planner regenerates or repairs the plan instead of sending a broken FSM into execution.

## 5. FSM execution

`FSMExecutor` is the long-horizon runtime. It does not own the environment loop. It only receives the latest image and returns the next action packet.

The executor tracks three things:

- `current_state`: which state is active now
- `state_step_count`: how long the current state has been running
- `total_step_count`: how long the episode has been running overall

The step function follows this sequence.

1. Check global timeout.
2. Load the current state definition.
3. Stop immediately if the state is terminal.
4. Evaluate timeout and always transitions.
5. If the state does not transition, run the state's instruction through JarvisVLA.
6. Increase counters and return the action packet.

The practical effect is that long-horizon behavior becomes predictable. A state stays active until its timeout expires or an explicit always transition moves execution forward.

## 6. JarvisVLA execution

`VLARunner` bridges the internal instruction format to the actual JarvisVLA model.

The runner resolves the instruction type first.

- `normal`: general text command
- `recipe`: crafting command that requires the crafting-table path
- `simple`: minimal command for simple control actions

After that, it calls the underlying JarvisVLA agent with the current image and instruction. The raw model output is normalized into the Purple Agent compact format.

One more detail matters here: the model output uses a different camera discretization than the environment expects. The runner optionally converts the 21-bin camera space into the 11-bin space used by the compact action format.

## 7. Script policy and routing

The agent is not hard-wired to VLA for every operation. Some prefixes can be routed through scripted sequences when that is more reliable.

The sequence router works in two stages.

1. It first tries to match the instruction and task text against known scripted sequences.
2. If that fails, it falls back to the planner hint and the execution hint for the state.

This is where the agent becomes hybrid.

- `vla`: always run JarvisVLA
- `scripted`: force a scripted sequence
- `hybrid`: prefer the best available route

The result is that repetitive tasks such as inventory manipulation can be handled more deterministically, while open-ended tasks stay with VLA.

## 8. Runtime state and episode outputs

Each environment episode keeps a mutable state object between steps. That state stores the current execution mode, current FSM state, the direct short instruction if one exists, and startup bookkeeping.

The agent also writes episode artifacts.

- `plan.json`: the generated plan for long-horizon execution
- `result.json`: the execution summary, including mode, total steps, and termination result

## 9. Running the server

### 9.1 Minimal run

```bash
cd /workspace/woosung/AgentBeats-JarvisVLA

python -m src.server.app \
  --host 0.0.0.0 \
  --port 9019 \
  --planner-model gpt-4o \
  --vla-checkpoint-path /ABS/PATH/TO/JarvisVLA-Checkpoint \
  --vla-url http://localhost:8000/v1
```

The VLA checkpoint path is mandatory.

### 9.2 Important CLI arguments

Planner:

- `--planner-api-key`
- `--planner-url` defaulting to `https://api.openai.com/v1`
- `--planner-model` defaulting to `gpt-4o`
- `--planner-temperature` defaulting to `0.2`

VLA:

- `--vla-checkpoint-path` required
- `--vla-url` defaulting to `http://localhost:8000/v1`
- `--vla-api-key`
- `--vla-history-num` defaulting to `4`
- `--vla-action-chunk-len` defaulting to `1`
- `--vla-bpe` defaulting to `0`
- `--vla-instruction-type` defaulting to `normal`
- `--vla-temperature` defaulting to `0.7`
- `--vla-no-camera-convert` disables 21-bin to 11-bin conversion

## 10. Deep dive

For a more detailed walk-through of the control flow, FSM behavior, and script-policy routing, see [AGENT_LOGIC.md](AGENT_LOGIC.md).

## 11. Troubleshooting

- If startup fails, check that the VLA checkpoint path exists and is readable.
- If the planner emits an invalid instruction, verify the instruction registry and the strict prefix:item format.
- If the action keeps becoming noop, check init ordering, image decoding, and JarvisVLA runtime errors.

## 12. References

- JarvisVLA original repository: https://github.com/CraftJarvis/JarvisVLA
- MineStudio framework: https://github.com/CraftJarvis/MineStudio
- MCU repository: https://github.com/CraftJarvis/MCU
- MCU-AgentBeats benchmark tasks: https://github.com/KWSmBang/MCU-AgentBeats
- Minecraft-agentbeats-leaderboard: https://github.com/RDI-Foundation/Minecraft-agentbeats-leaderboard
