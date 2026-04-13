# Purple Agent Internal Logic

This document explains how the JarvisVLA-based Purple Agent actually behaves at runtime. The goal is not only to list components, but to show how the planner, FSM executor, JarvisVLA runner, and script policy connect to each other during execution.

---

## 1. End-to-End Control Flow

```text
Task text
  ↓
Planner.classify_horizon()
  ↓
short  → generate one direct instruction and keep executing it each step
long   → generate a multi-state FSM plan and hand it to FSMExecutor
  ↓
act(obs)
  ↓
if short: execute the direct instruction through JarvisVLA
if long:  execute the current FSM state's instruction
  ↓
VLARunner.run()
  ↓
normalize action to {buttons, camera}
  ↓
return action
```

The agent is not a single policy. It is a control loop with three layers:

- The planner decides whether the task is short-horizon or long-horizon.
- The FSM executor manages state progression for long-horizon tasks.
- The JarvisVLA runner converts a state instruction into one action step.

A script policy layer can intercept the route when a scripted sequence is more reliable than pure VLA execution.

### Core constraints

- The main execution path is VLA-first.
- Every instruction must resolve to a strict registry key in prefix:item form.
- Long-horizon plans must include a fallback state.
- State transitions are limited to always and timeout.
- Visual-state hallucination checks are intentionally avoided in the main loop.

---

## 2. Planner Behavior

### 2.1 Horizon Classification

**Method**: `Planner.classify_horizon(task_text: str) -> str`

The planner first decides whether a task is short or long.

- Short horizon: a task that can be attempted with one direct instruction.
- Long horizon: a task that should be decomposed into multiple states.

The classification is based on task complexity, not on a fixed keyword list. This decision changes the entire runtime strategy.

Example:

```text
Task: craft diamond pickaxe
  ↓
Planner classifies it as long
  ↓
A multi-step FSM plan is generated
```

### 2.2 Short Directive Generation

**Method**: `Planner.generate_short_directive(task_text: str, image: np.ndarray) -> dict`

For short tasks, the planner returns a compact directive:

```json
{
  "instruction": "kill_entity:zombie",
  "instruction_type": "normal"
}
```

The instruction must be a strict registry key. The planner canonicalizes the task text, filters candidates, and prefers valid prefix:item keys from the instruction registry.

### 2.3 Long Plan Generation

**Method**: `Planner.generate_long_plan(task_text: str, image: np.ndarray) -> dict`

For long tasks, the planner builds a simplified finite state machine.

A typical state contains:

- `instruction`: the action to attempt in that state
- `instruction_type`: normal, recipe, or simple
- `transitions`: only always or timeout rules

Example:

```json
{
  "task": "craft_diamond_pickaxe",
  "initial_state": "step1",
  "global_config": {
    "max_total_steps": 1200,
    "max_replanning": 3
  },
  "states": {
    "step1": {
      "description": "Find stone block",
      "instruction": "find_block:stone",
      "instruction_type": "normal",
      "transitions": [
        {
          "condition": { "type": "timeout", "max_steps": 100 },
          "next_state": "step2"
        }
      ]
    },
    "step2": {
      "description": "Mine stone block",
      "instruction": "mine_block:stone",
      "instruction_type": "normal",
      "transitions": [
        {
          "condition": { "type": "timeout", "max_steps": 50 },
          "next_state": "step3"
        }
      ]
    },
    "step3": {
      "description": "Craft stone pickaxe",
      "instruction": "craft_item:stone_pickaxe",
      "instruction_type": "recipe",
      "transitions": [
        { "condition": { "type": "always" }, "next_state": "fallback" }
      ]
    },
    "fallback": {
      "description": "Last resort: attempt the full task",
      "instruction": "craft_diamond_pickaxe",
      "instruction_type": "normal",
      "terminal": false,
      "transitions": [
        { "condition": { "type": "always" }, "next_state": "step1" }
      ]
    }
  }
}
```

### 2.4 Validation

**Method**: `PlanValidator.validate(plan: dict) -> Tuple[bool, Optional[str]]`

Before a plan is used, the validator checks that:

- every instruction resolves to a valid registry key
- the state graph is structurally valid
- the fallback state exists
- transition types stay within the supported set

If validation fails, the planner regenerates or repairs the plan instead of letting a broken FSM enter execution.

---

## 3. FSM Executor

`FSMExecutor` is the long-horizon runtime. It does not own the environment loop. It only receives the latest image and returns the next action packet.

The executor tracks three main counters:

- `current_state`: which state is active now
- `state_step_count`: how long the current state has been running
- `total_step_count`: how long the episode has been running overall

### 3.1 Step Function

**Method**: `FSMExecutor.step(image: np.ndarray) -> Optional[dict]`

The step function follows this sequence:

1. Check whether the FSM is already finished.
2. Check the global timeout.
3. Load the current state definition.
4. Stop immediately if the state is terminal.
5. Evaluate timeout and always transitions.
6. If the state does not transition, execute the state's instruction through JarvisVLA.
7. Increase counters and return the action packet.

### 3.2 Transition Types

#### Timeout transition

```json
{
  "condition": { "type": "timeout", "max_steps": 150 },
  "next_state": "step2"
}
```

This means the current state stays active until its step budget is exhausted.

#### Always transition

```json
{
  "condition": { "type": "always" },
  "next_state": "fallback"
}
```

This means the executor moves to the next state immediately.

### 3.3 State Tracking

```python
def _tick(self):
    self.state_step_count += 1
    self.total_step_count += 1

def _transition_to(self, new_state: str):
    self.current_state = new_state
    self.state_step_count = 0

def _terminate(self, result: str):
    self.finished = True
    self.result = result
```

The practical effect is that long-horizon behavior becomes predictable. A state stays active until its timeout expires or an explicit always transition moves execution forward.

---

## 4. JarvisVLA Runner

`VLARunner` bridges the internal instruction format to the actual JarvisVLA model.

### 4.1 Instruction Type Resolution

The runner resolves the instruction type before calling the model.

- `normal`: general text command
- `recipe`: crafting command that requires the crafting-table path
- `simple`: minimal command for simple control actions

The default behavior is to respect the requested type, then fall back to a rule-based inference when needed.

### 4.2 Model Call

The runner calls JarvisVLA with the current image and instruction.

```python
raw_action = self.agent.forward(
    observations=[image],
    instructions=[instruction],
    verbos=False,
    need_crafting_table=(resolved_type == "recipe"),
)
```

The model output is then normalized into the Purple Agent compact action format.

### 4.3 Action Normalization

The raw output is converted into:

```python
{
  "buttons": np.array([3]),
  "camera": np.array([60])
}
```

One additional detail matters here: JarvisVLA and the environment do not use the same camera discretization. The runner can convert the 21-bin camera space into the 11-bin space used by the compact action format.

### 4.4 Camera Conversion

JarvisVLA uses a 21×21 grid, while the Purple Agent environment expects an 11×11 grid. The conversion path is:

```text
21-bin index
  ↓
mu-law decode to continuous angle space
  ↓
re-encode into 11-bin space
  ↓
11-bin index
```

This keeps the model output compatible with the action space expected by the environment.

---

## 5. Script Policy and Sequence Routing

The agent is not hard-wired to VLA for every operation. Some prefixes can be routed through scripted sequences when that is more reliable.

### 5.1 Execution Hints

The planner can attach an execution hint to help route the instruction:

```python
_PREFIX_SEQUENCE_MAP = {
    "drop":        ("scripted", "drop_cycle"),
    "use_item":    ("hybrid",   ""),
    "kill_entity": ("vla",      ""),
    "mine_block":  ("vla",      ""),
    "craft_item":  ("hybrid",   "open_inventory_craft"),
    "pickup":      ("vla",      ""),
}
```

The hint can mean:

- `vla`: always run JarvisVLA
- `scripted`: force a scripted sequence
- `hybrid`: prefer the best available route

### 5.2 SequenceRouter

`SequenceRouter` selects either a scripted sequence or VLA execution.

It works in two stages:

1. It first tries to match the instruction and task text against known scripted sequences.
2. If that fails, it falls back to the planner hint and the execution hint for the state.

This is what makes the agent hybrid. Repetitive tasks such as inventory manipulation can be handled more deterministically, while open-ended tasks stay with VLA.

### 5.3 FallbackPolicyEngine

`FallbackPolicyEngine` owns the policy selection logic.

Its selection order is:

1. Planner primitives, when the planner explicitly provides them.
2. Task text routing, which can override a weak planner sequence name.
3. Instruction routing, which matches the actual command.
4. VLA fallback when no reliable scripted sequence is found.

This design keeps scripted execution available without making it mandatory for every task.

---

## 6. Short Mode vs Long Mode

### 6.1 Short Mode

In short mode, the agent stores one instruction and executes that same instruction every step.

```text
reset(task_text="kill_entity:zombie")
  ↓
Planner.classify_horizon() → short
  ↓
Planner.generate_short_directive() → one instruction
  ↓
act(obs)
  ↓
run the same instruction through JarvisVLA every step
```

This mode is intentionally simple. It lets JarvisVLA adapt to the current image without the overhead of planning.

### 6.2 Long Mode

In long mode, the agent creates an FSM plan, validates it, and runs it through `FSMExecutor`.

```text
reset(task_text="craft_diamond_pickaxe")
  ↓
Planner.classify_horizon() → long
  ↓
Planner.generate_long_plan()
  ↓
PlanValidator.validate()
  ↓
FSMExecutor(plan)
  ↓
act(obs)
  ↓
execute current state's instruction
  ↓
state transition happens on timeout or always rules
```

This mode is better for multi-step tasks because the agent knows which subgoal it is pursuing at each stage.

---

## 7. Runtime State and Episode Outputs

Each environment episode keeps a mutable state object between steps. That state stores the current execution mode, current FSM state, the direct short instruction if one exists, and startup bookkeeping.

The agent also writes episode artifacts.

- `plan.json`: the generated plan for long-horizon execution
- `result.json`: the execution summary, including mode, total steps, and termination result

These files make it easier to inspect how the agent behaved during an episode.

---

## 8. Why This Design Works

### 8.1 VLA-centered execution

The agent uses vision and language to choose the next action instead of relying only on scripted routines.

### 8.2 Timeout-based transitions

The executor avoids visual-state hallucination checks and uses explicit time budgets instead. This makes long-horizon behavior more stable and easier to reason about.

### 8.3 Optional scripted routes

Some tasks are more reliable with a scripted sequence. The router keeps that option available without forcing every instruction through scripts.

### 8.4 Stateless environment compatibility

The environment loop is kept simple: the agent returns one action at a time, and all persistent state is carried in the agent state object.

---

## 9. Example Flows

### 9.1 Short Mode Example

```text
Input: kill one zombie

1. Horizon classification returns short.
2. The planner emits kill_entity:zombie.
3. The agent enters short mode.
4. Each step calls JarvisVLA with the same instruction.
5. The action changes only as the image changes.
```

### 9.2 Long Mode Example

```text
Input: craft diamond pickaxe

1. Horizon classification returns long.
2. The planner creates a multi-step FSM.
3. The fallback state is added.
4. The executor stays in step1 until the timeout expires.
5. The executor transitions to step2 and continues.
6. Execution ends on completion or on global timeout.
```

---

## 10. Troubleshooting

### Problem: instruction is not a valid registry key

The planner generated an instruction that does not exist in the registry.

Fix:

- Check the instruction registry.
- Tighten the planner prompt.
- Ensure the task text produces a strict prefix:item instruction.

### Problem: the action keeps becoming noop

Possible causes:

- the image failed to decode
- the model failed to load
- JarvisVLA raised an exception
- the environment called the agent before initialization

### Problem: the FSM gets stuck in fallback

Possible causes:

- the main plan is too weak
- the fallback instruction is too vague
- the global timeout is too high

---

## References

- JarvisVLA: https://github.com/CraftJarvis/JarvisVLA
- MineStudio: https://github.com/CraftJarvis/MineStudio
- MCU-AgentBeats: https://github.com/KWSmBang/MCU-AgentBeats
- Minecraft-agentbeats-leaderboard: https://github.com/RDI-Foundation/Minecraft-agentbeats-leaderboard
