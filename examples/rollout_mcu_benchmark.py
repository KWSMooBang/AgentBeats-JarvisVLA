"""
MCU Benchmark Rollout for JarvisVLA

Run JarvisVLA on MCU benchmark task configurations (YAML).
Requires a running vLLM server serving the JarvisVLA model.
"""

import argparse
import json
import re
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jarvisvla.evaluate.agent_wrapper import VLLM_AGENT

MCU_TASK_CATEGORIES = [
    "building",
    "combat",
    "crafting",
    "decoration",
    "ender_dragon",
    "explore",
    "find",
    "mine_diamond_from_scratch",
    "mining_and_collecting",
    "motion",
    "tool_use",
    "trapping",
]


###############################################################################
# MCU task_text → instructions.json key resolution
###############################################################################

# Verb patterns → instructions.json prefix (ordered by priority)
_VERB_TO_PREFIX = [
    (re.compile(r'\b(defeat|kill|hunt|combat|slay|shoot|fend off)\b', re.I), 'kill_entity'),
    (re.compile(r'\b(mine|collect|dig|shear|chop|gather|cut|break|pick up)\b', re.I), 'mine_block'),
    (re.compile(r'\b(craft|brew|smelt|cook)\b', re.I), 'craft_item'),
    (re.compile(r'\b(use|equip|place|plant|sleep|drink|enchant|carve|ignite|throw|hook|trap|ride|eat)\b', re.I), 'use_item'),
    (re.compile(r'\b(drop|discard|remove)\b', re.I), 'drop'),
]

# Words in task_text that don't directly match Minecraft IDs
_OBJECT_ALIASES = {
    "wood": "oak_log", "woods": "oak_log",
    "wool": "white_wool", "wools": "white_wool",
    "bed": "white_bed", "beds": "white_bed",
    "grass": "grass_block",
    "stone": "stone", "stones": "stone",
}

# Regex to split text before instrumental phrases
_INSTRUMENT_RE = re.compile(
    r'\b(?:using|with|from|for smelting|for various|for crafting'
    r'|to enchant|to create|to discover|to skip|to block|to obtain'
    r'|to get|to ignite|to secure|to explore)\b',
    re.I,
)

# Verbs where the INSTRUMENT (after "using/with") is the actual target,
# not the direct object.  For these, we search the instrument clause first.
_INSTRUMENT_FIRST_VERBS = re.compile(
    r'\b(enchant|hook|trap|smelt|brew|ignite)\b', re.I,
)


def _depluralize(word: str) -> str:
    if word.endswith('ves'):
        return word[:-3] + 'f'
    if word.endswith('ies'):
        return word[:-3] + 'y'
    if word.endswith('es') and not word.endswith('ses'):
        return word[:-2]
    if word.endswith('s') and not word.endswith('ss'):
        return word[:-1]
    return word


def _extract_candidates(text: str) -> List[str]:
    """Generate candidate Minecraft object names from a text fragment."""
    words = re.findall(r'[a-z]+', text.lower())
    candidates = []
    for length in range(3, 0, -1):
        for i in range(len(words) - length + 1):
            raw = '_'.join(words[i:i + length])
            dep = '_'.join(_depluralize(w) for w in words[i:i + length])
            candidates.append(raw)
            if dep != raw:
                candidates.append(dep)
    for w in words:
        for key in (w, _depluralize(w)):
            if key in _OBJECT_ALIASES:
                candidates.append(_OBJECT_ALIASES[key])
    return candidates


def build_instruction_index(prompt_library: dict) -> Dict[str, Dict[str, str]]:
    """Build {prefix: {object_name: full_key}} from instructions.json."""
    index: Dict[str, Dict[str, str]] = {}
    prefixes = ('kill_entity', 'mine_block', 'craft_item', 'use_item', 'drop')
    for key in prompt_library:
        for p in prefixes:
            tag = p + ':'
            if key.startswith(tag):
                index.setdefault(p, {})[key[len(tag):]] = key
                break
    return index


def resolve_task_instruction(
    task_text: str,
    prompt_library: dict,
    index: Dict[str, Dict[str, str]],
) -> Tuple[Optional[str], str]:
    """
    Map a free-form MCU task_text to the closest instructions.json key and
    determine the appropriate instruction_type for the agent.

    Returns:
        (matched_key, instruction_type)
        - matched (craft_item:*)  → (key, "recipe")
        - matched (other prefix)  → (key, "normal")
        - no match                → (None, "simple")
    """
    text_lower = task_text.lower().rstrip('.')

    # 1. Detect action prefixes from verbs
    matched_prefixes = []
    for pattern, prefix in _VERB_TO_PREFIX:
        if pattern.search(text_lower):
            matched_prefixes.append(prefix)
    if not matched_prefixes:
        return None, "simple"

    # 2. Split at instrumental phrases
    parts = _INSTRUMENT_RE.split(text_lower, maxsplit=1)
    main_clause = parts[0].strip()
    instrument_clause = parts[1].strip() if len(parts) > 1 else ""

    instrument_first = bool(_INSTRUMENT_FIRST_VERBS.search(text_lower))

    if instrument_first and instrument_clause:
        search_order = [instrument_clause, main_clause]
    else:
        search_order = [main_clause, instrument_clause] if instrument_clause else [main_clause]

    # 3. Search candidates in priority order
    for clause in search_order:
        candidates = _extract_candidates(clause)
        for prefix in matched_prefixes:
            names = index.get(prefix, {})
            for c in candidates:
                if c in names:
                    key = names[c]
                    if prompt_library.get(key, {}).get('instruct'):
                        itype = "recipe" if prefix == "craft_item" else "normal"
                        return key, itype

    # 4. Broader fallback: search full text (catches aliases etc.)
    full_candidates = _extract_candidates(text_lower)
    for prefix in matched_prefixes:
        names = index.get(prefix, {})
        for c in full_candidates:
            if c in names:
                key = names[c]
                if prompt_library.get(key, {}).get('instruct'):
                    itype = "recipe" if prefix == "craft_item" else "normal"
                    return key, itype

    return None, "simple"


def list_mcu_tasks(tasks_dir: str, category: Optional[str] = None) -> List[str]:
    tasks_root = Path(tasks_dir)
    if not tasks_root.exists():
        raise FileNotFoundError(f"tasks_dir does not exist: {tasks_root}")

    if category:
        category_dir = tasks_root / category
        if not category_dir.exists():
            return []
        return sorted(str(p) for p in category_dir.glob("*.yaml"))

    all_tasks = []
    for category_dir in tasks_root.iterdir():
        if category_dir.is_dir():
            all_tasks.extend(str(p) for p in category_dir.glob("*.yaml"))
    return sorted(all_tasks)


# ---------------------------------------------------------------------------
# Camera-space conversion: model (21×21) → env (11×11)
#
#   Model (JarvisVLA):  binsize=1, mu=20, maxval=10  →  n_bins=21
#   Env   (MineStudio):  binsize=2, mu=10, maxval=10  →  n_bins=11
#
# Flow: model flat idx → (pitch,yaw) 21-bins → mu-law decode (mu=20, bs=1)
#       → continuous → mu-law encode (mu=10, bs=2) → (pitch,yaw) 11-bins → env flat idx
# ---------------------------------------------------------------------------

_MODEL_BINS = 21
_MODEL_BINSIZE = 1
_MODEL_MU = 20.0
_ENV_BINS = 11
_ENV_BINSIZE = 2
_ENV_MU = 10.0
_MAXVAL = 10


def _mu_decode(xy: np.ndarray, mu: float, maxval: int) -> np.ndarray:
    """Mu-law undiscretize: bin index → continuous value."""
    xy = xy.astype(np.float64)
    xy_norm = xy / maxval
    return np.sign(xy_norm) * (1.0 / mu) * ((1.0 + mu) ** np.abs(xy_norm) - 1.0) * maxval


def _mu_encode(xy: np.ndarray, mu: float, maxval: int) -> np.ndarray:
    """Mu-law discretize: continuous value → bin index (before rounding)."""
    xy = np.clip(xy, -maxval, maxval).astype(np.float64)
    xy_norm = xy / maxval
    return np.sign(xy_norm) * (np.log(1.0 + mu * np.abs(xy_norm)) / np.log(1.0 + mu)) * maxval


def convert_camera_action(camera_flat: int) -> int:
    """Convert a flat camera index from the model's 21×21 space to the env's 11×11 space."""
    pitch_21 = camera_flat // _MODEL_BINS
    yaw_21 = camera_flat % _MODEL_BINS

    centered = np.array([pitch_21, yaw_21], dtype=np.float64) * _MODEL_BINSIZE - _MAXVAL
    continuous = _mu_decode(centered, _MODEL_MU, _MAXVAL)

    encoded = _mu_encode(continuous, _ENV_MU, _MAXVAL)
    env_bins = np.clip(np.round((encoded + _MAXVAL) / _ENV_BINSIZE).astype(np.int64), 0, _ENV_BINS - 1)

    return int(env_bins[0]) * _ENV_BINS + int(env_bins[1])


def env_init_mcu(
    yaml_path: str,
    rollout_path: str,
    obs_size: Tuple[int, int] = (640, 360),
    render_size: Tuple[int, int] = (640, 360),
    max_steps: int = 600,
    fps: int = 20,
    record_video: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks import (
        CommandsCallback,
        JudgeResetCallback,
        RecordCallback,
        RewardsCallback,
    )

    with open(yaml_path, "r", encoding="utf-8") as f:
        task_config = yaml.safe_load(f)

    commands = task_config.get("custom_init_commands", [])
    reward_cfg = task_config.get("reward_cfg", [])

    rollout_dir = Path(rollout_path)
    rollout_dir.mkdir(parents=True, exist_ok=True)

    callbacks = []
    if commands:
        callbacks.append(CommandsCallback(commands))
    callbacks.append(JudgeResetCallback(max_steps))
    if reward_cfg:
        callbacks.append(RewardsCallback(reward_cfg))
    if record_video:
        callbacks.append(
            RecordCallback(record_path=str(rollout_dir), fps=fps, frame_type="pov")
        )

    env = MinecraftSim(
        action_type="env",
        obs_size=obs_size,
        render_size=render_size,
        preferred_spawn_biome=None,
        callbacks=callbacks,
    )

    config_path = rollout_dir / "task_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {"task_name": Path(yaml_path).stem, "yaml_path": yaml_path, **task_config},
            f,
            indent=2,
            ensure_ascii=False,
        )

    raw_action_path = rollout_dir / "raw_action.jsonl"
    with open(raw_action_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"step": 0, "raw_action": ""}, ensure_ascii=False) + "\n")

    return env, task_config


def run_mcu_task(
    yaml_path: str,
    agent: VLLM_AGENT,
    rollout_path: str,
    instruction_index: Optional[Dict[str, Dict[str, str]]] = None,
    obs_size: Tuple[int, int] = (640, 360),
    max_steps: int = 600,
    verbose: bool = True,
) -> Dict[str, Any]:
    env, task_config = env_init_mcu(
        yaml_path=yaml_path,
        rollout_path=rollout_path,
        obs_size=obs_size,
        max_steps=max_steps,
        record_video=True,
    )

    task_name = Path(yaml_path).stem
    task_text = task_config.get("text", task_name)

    # Resolve task_text → instruction library key + instruction_type
    matched_key = None
    resolved_instruction_type = "simple"
    agent_instruction = task_text
    if instruction_index is not None:
        matched_key, resolved_instruction_type = resolve_task_instruction(
            task_text, agent.prompt_library, instruction_index,
        )
        if matched_key:
            agent_instruction = matched_key
        # else: keep original task_text with "simple" type

    # Dynamically set agent instruction_type for this task
    original_instruction_type = agent.instruction_type
    agent.instruction_type = resolved_instruction_type
    need_crafting_table = resolved_instruction_type == "recipe"

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Task: {task_name}")
        print(f"Original text    : {task_text}")
        print(f"Mapped key       : {matched_key or '(no match)'}")
        print(f"Instruction type : {resolved_instruction_type}")
        print(f"Agent input      : {agent_instruction}")
        print(f"{'=' * 70}\n")

    agent.reset()

    obs, info = env.reset()
    env.action_type = "agent"

    done = False
    step = 0
    total_reward = 0.0
    raw_action_file_path = Path(rollout_path) / "raw_action.jsonl"

    while not done and step < max_steps:
        action = agent.forward(
            observations=[obs["image"]],
            instructions=[agent_instruction],
            verbos=verbose,
            need_crafting_table=need_crafting_table,
        )

        action["camera"] = np.array(convert_camera_action(int(action["camera"])))

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += float(reward)
        step += 1

        with open(raw_action_file_path, "a", encoding="utf-8") as f:
            action_record = {}
            for key, value in action.items():
                action_record[key] = value.tolist() if hasattr(value, "tolist") else int(value)
            f.write(
                json.dumps(
                    {
                        "step": step,
                        "reward": float(reward),
                        "raw_action": action_record,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        if verbose and step % 50 == 0:
            print(
                f"Step {step:04d} | reward={reward:.3f} | total={total_reward:.3f}"
            )

        if reward > 0:
            break

    env.close()

    # Restore original instruction_type
    agent.instruction_type = original_instruction_type

    success = total_reward > 0
    results = {
        "task_name": task_name,
        "task_text": task_text,
        "matched_key": matched_key,
        "instruction_type": resolved_instruction_type,
        "agent_instruction": agent_instruction,
        "yaml_path": yaml_path,
        "total_reward": float(total_reward),
        "steps_taken": step,
        "success": success,
    }

    result_path = Path(rollout_path) / "episode_results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    status_path = Path(rollout_path) / ("success.json" if success else "loss.json")
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if verbose:
        status = "SUCCESS" if success else "FAIL"
        print(f"[{status}] {task_name}: reward={total_reward:.3f}, steps={step}")

    return results


def run_mcu_benchmark(
    agent: VLLM_AGENT,
    tasks_dir: str,
    output_dir: str,
    task_categories: Optional[List[str]] = None,
    obs_size: Tuple[int, int] = (640, 360),
    max_steps: int = 600,
    verbose: bool = True,
) -> Dict[str, Any]:
    tasks_root = Path(tasks_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # Build instruction index from agent's prompt library
    instruction_index = build_instruction_index(agent.prompt_library)
    if verbose:
        total_keys = sum(len(v) for v in instruction_index.values())
        print(f"Instruction index built: {total_keys} entries across {list(instruction_index.keys())}")

    if task_categories is None:
        task_categories = sorted(p.name for p in tasks_root.iterdir() if p.is_dir())

    date_folder = time.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir_with_date = output_root / date_folder
    output_dir_with_date.mkdir(parents=True, exist_ok=True)

    benchmark_results: Dict[str, Any] = {
        "categories": {},
        "total_tasks": 0,
        "successful_tasks": 0,
        "total_reward": 0.0,
    }

    for category in task_categories:
        yaml_files = list_mcu_tasks(str(tasks_root), category)
        if not yaml_files:
            if verbose:
                print(f"No tasks found in category={category}")
            continue

        if verbose:
            print(f"\n{'#' * 70}")
            print(f"Category: {category} ({len(yaml_files)} tasks)")
            print(f"{'#' * 70}")

        category_results = []
        for yaml_path in yaml_files:
            task_name = Path(yaml_path).stem
            task_output_dir = output_dir_with_date / category / task_name
            task_output_dir.mkdir(parents=True, exist_ok=True)

            try:
                result = run_mcu_task(
                    yaml_path=yaml_path,
                    agent=agent,
                    rollout_path=str(task_output_dir),
                    instruction_index=instruction_index,
                    obs_size=obs_size,
                    max_steps=max_steps,
                    verbose=verbose,
                )
            except Exception as exc:
                import traceback
                traceback.print_exc()
                print(f"Error running task {task_name}: {exc}")
                result = {
                    "task_name": task_name,
                    "yaml_path": yaml_path,
                    "success": False,
                    "total_reward": 0.0,
                    "steps_taken": 0,
                    "error": str(exc),
                }

            category_results.append(result)
            benchmark_results["total_tasks"] += 1
            benchmark_results["successful_tasks"] += int(
                bool(result.get("success", False))
            )
            benchmark_results["total_reward"] += float(
                result.get("total_reward", 0.0)
            )

        successful = sum(1 for r in category_results if r.get("success", False))
        n_tasks = len(category_results)
        success_rate = successful / n_tasks if n_tasks else 0.0
        avg_reward = (
            sum(float(r.get("total_reward", 0.0)) for r in category_results) / n_tasks
            if n_tasks
            else 0.0
        )

        benchmark_results["categories"][category] = {
            "num_tasks": n_tasks,
            "successful_tasks": successful,
            "avg_reward": avg_reward,
            "success_rate": success_rate,
            "tasks": category_results,
        }

        if verbose:
            print(f"\nCategory '{category}' Summary:")
            print(f"  Tasks: {n_tasks}")
            print(f"  Success Rate: {success_rate:.2%}")
            print(f"  Average Reward: {avg_reward:.3f}")

    total = benchmark_results["total_tasks"]
    if total > 0:
        benchmark_results["overall_success_rate"] = (
            benchmark_results["successful_tasks"] / total
        )
        benchmark_results["avg_reward"] = benchmark_results["total_reward"] / total
    else:
        benchmark_results["overall_success_rate"] = 0.0
        benchmark_results["avg_reward"] = 0.0

    result_path = output_dir_with_date / "benchmark_results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"\n{'=' * 70}")
        print("BENCHMARK SUMMARY")
        print(f"{'=' * 70}")
        print(f"Total Tasks: {benchmark_results['total_tasks']}")
        print(f"Successful Tasks: {benchmark_results['successful_tasks']}")
        print(f"Overall Success Rate: {benchmark_results['overall_success_rate']:.2%}")
        print(f"Average Reward: {benchmark_results['avg_reward']:.3f}")
        print(f"Results saved to: {result_path}")
        print(f"{'=' * 70}")

    return benchmark_results


def resolve_tasks_dir(user_tasks_dir: str) -> str:
    candidate = Path(user_tasks_dir)
    if candidate.exists():
        return str(candidate)

    fallbacks = [
        Path("/workspace/woosung/MCU-AgentBeats/MCU_benchmark/task_configs/tasks"),
        Path("/workspace/woosung/AgentBeats-OpenHA/openagents/assets/mcu_tasks"),
    ]
    for fallback in fallbacks:
        if fallback.exists():
            print(f"[WARN] tasks_dir not found: {candidate}. Falling back to {fallback}")
            return str(fallback)

    raise FileNotFoundError(
        f"tasks_dir does not exist: {candidate}. "
        "Provide --tasks-dir with the MCU task_configs/tasks path."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run JarvisVLA on MCU Benchmark tasks"
    )

    parser.add_argument(
        "--category",
        type=str,
        default="combat",
        choices=MCU_TASK_CATEGORIES,
        help="Task category to run",
    )
    parser.add_argument(
        "--tasks-dir",
        type=str,
        default="/workspace/woosung/MCU-AgentBeats/MCU_benchmark/task_configs/tasks",
        help="Directory containing MCU task YAML files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Directory to save benchmark results",
    )
    parser.add_argument("--max-steps", type=int, default=600)

    # vLLM / model configuration
    parser.add_argument(
        "--model-path",
        type=str,
        default="/workspace/models/JarvisVLA-Qwen2-VL-7B",
        help="Local checkpoint path (used for tokenizer and backbone detection)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:9020/v1",
        help="vLLM server base URL",
    )
    parser.add_argument("--api-key", type=str, default="EMPTY")

    # Agent inference parameters
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument(
        "--history-num",
        type=int,
        default=2,
        help="Number of history frames to include in context",
    )
    parser.add_argument(
        "--action-chunk-len",
        type=int,
        default=1,
        help="Number of actions to predict per inference",
    )
    parser.add_argument(
        "--instruction-type",
        type=str,
        default="normal",
        choices=["normal", "simple", "recipe"],
        help="Instruction prompt type",
    )

    # Observation
    parser.add_argument("--obs-width", type=int, default=640)
    parser.add_argument("--obs-height", type=int, default=360)

    parser.add_argument("--verbose", action="store_true", default=True)

    args = parser.parse_args()

    tasks_dir = resolve_tasks_dir(args.tasks_dir)

    print("\n" + "=" * 70)
    print("ROLLOUT: MCU Benchmark (JarvisVLA)")
    print("=" * 70)
    print(f"Category       : {args.category}")
    print(f"Tasks Dir      : {tasks_dir}")
    print(f"Output Dir     : {args.output_dir}")
    print(f"Model Path     : {args.model_path}")
    print(f"vLLM Base URL  : {args.base_url}")
    print(f"Max Steps      : {args.max_steps}")
    print(f"Temperature    : {args.temperature}")
    print(f"History Num    : {args.history_num}")
    print(f"Action Chunk   : {args.action_chunk_len}")
    print(f"Instruction    : {args.instruction_type}")
    print(f"Obs Size       : ({args.obs_width}, {args.obs_height})")
    print("=" * 70 + "\n")

    print("Initializing JarvisVLA agent (VLLM_AGENT)...")
    agent = VLLM_AGENT(
        checkpoint_path=args.model_path,
        base_url=args.base_url,
        api_key=args.api_key,
        history_num=args.history_num,
        action_chunk_len=args.action_chunk_len,
        instruction_type=args.instruction_type,
        temperature=args.temperature,
    )
    print("Agent initialized.\n")

    results = run_mcu_benchmark(
        agent=agent,
        tasks_dir=tasks_dir,
        output_dir=args.output_dir,
        task_categories=[args.category],
        obs_size=(args.obs_width, args.obs_height),
        max_steps=args.max_steps,
        verbose=args.verbose,
    )

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Total Tasks        : {results['total_tasks']}")
    print(f"Successful Tasks   : {results['successful_tasks']}")
    print(f"Overall Success Rate: {results['overall_success_rate']:.2%}")
    print(f"Average Reward     : {results['avg_reward']:.3f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
