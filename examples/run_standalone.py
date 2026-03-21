"""
MCU Benchmark Runner for Scripted Policy Agent

Loads MCU task YAML configurations, lets the user pick a category,
then runs every task in that category through the MinecraftPurpleAgent
inside a MineStudio environment loop (no A2A server required).

Usage:
    python examples/run_standalone.py --category combat
    python examples/run_standalone.py --category mining_and_collecting --max-steps 800
    python examples/run_standalone.py --list-categories
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MCU_CATEGORIES = [
    "building", "combat", "crafting", "decoration", "ender_dragon",
    "explore", "find", "mine_diamond_from_scratch", "mining_and_collecting",
    "motion", "tool_use", "trapping",
]

LONG_HORIZON_CATEGORIES = {"ender_dragon", "mine_diamond_from_scratch"}


# ======================================================================
# MCU task discovery
# ======================================================================

def discover_categories(tasks_dir: Path) -> List[str]:
    return sorted(d.name for d in tasks_dir.iterdir() if d.is_dir())


def list_tasks_in_category(tasks_dir: Path, category: str) -> List[Path]:
    cat_dir = tasks_dir / category
    if not cat_dir.exists():
        return []
    return sorted(cat_dir.glob("*.yaml"))


def load_task_config(yaml_path: Path) -> Dict[str, Any]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ======================================================================
# Environment initialisation (MineStudio)
# ======================================================================

def init_env(
    task_config: Dict[str, Any],
    rollout_path: str,
    obs_size: Tuple[int, int] = (640, 360),
    render_size: Tuple[int, int] = (640, 360),
    max_steps: int = 600,
    fps: int = 20,
):
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks import (
        CommandsCallback,
        RecordCallback,
        RewardsCallback,
        JudgeResetCallback,
    )

    commands = task_config.get("custom_init_commands", [])
    reward_cfg = task_config.get("reward_cfg", [])
    os.makedirs(rollout_path, exist_ok=True)

    callbacks = []
    if commands:
        callbacks.append(CommandsCallback(commands))
    callbacks.append(JudgeResetCallback(max_steps))
    if reward_cfg:
        callbacks.append(RewardsCallback(reward_cfg))
    callbacks.append(RecordCallback(record_path=rollout_path, fps=fps, frame_type="pov"))

    env = MinecraftSim(
        obs_size=obs_size,
        render_size=render_size,
        callbacks=callbacks,
    )
    return env


# ======================================================================
# Single-task runner
# ======================================================================

def run_single_task(
    agent,
    yaml_path: Path,
    output_dir: Path,
    obs_size: Tuple[int, int],
    max_steps: int,
    verbose: bool,
) -> Dict[str, Any]:
    task_name = yaml_path.stem
    task_category = yaml_path.parent.name
    task_config = load_task_config(yaml_path)
    task_text = task_config.get("text", task_name).strip()
    task_horizon = "long" if task_category in LONG_HORIZON_CATEGORIES else "short"
    planner_task_text = task_text
    rollout_path = str(output_dir / task_name)

    print(f"\n{'='*70}")
    print(f"  Task   : {task_name}")
    print(f"  Text   : {task_text}")
    print(f"  Output : {rollout_path}")
    print(f"{'='*70}")

    env = init_env(
        task_config,
        rollout_path=rollout_path,
        obs_size=obs_size,
        max_steps=max_steps,
    )

    agent.reset(task_text=planner_task_text, episode_dir=rollout_path)
    state = agent.initial_state(task_text=task_text)

    os.makedirs(rollout_path, exist_ok=True)
    with open(os.path.join(rollout_path, "task_config.json"), "w") as f:
        json.dump({"task_name": task_name, "yaml_path": str(yaml_path), **task_config}, f, indent=2, ensure_ascii=False)

    obs, info = env.reset()
    env.action_type = "agent"
    total_reward = 0.0
    step = 0
    start_time = time.time()

    env_error = False
    env_error_message = ""
    for step in range(1, max_steps + 1):
        image = obs.get("image", obs.get("pov"))
        action, state = agent.act(obs={"image": image}, state=state)

        if action is None:
            logger.info("[%s] FSM finished at step %d", task_name, step)
            break

        try:
            obs, reward, terminated, truncated, info = env.step(action)
        except Exception as e:
            logger.error("[%s] env.step() crashed at step %d: %s", task_name, step, e)
            env_error = True
            env_error_message = str(e)
            break

        if isinstance(info, dict) and "error" in info:
            logger.error("[%s] env reported error at step %d: %s", task_name, step, info["error"])
            env_error = True
            env_error_message = str(info["error"])
            break

        total_reward += reward

        if verbose and step % 50 == 0:
            logger.info(
                "[%s] step=%d  fsm_state=%s  reward=%.2f  total=%.2f",
                task_name, step, state.current_fsm_state, reward, total_reward,
            )

        if terminated or truncated:
            break

    elapsed = time.time() - start_time
    try:
        env.close()
    except Exception:
        pass

    success = total_reward > 0 and not env_error
    result = {
        "task_name": task_name,
        "task_category": task_category,
        "task_horizon": task_horizon,
        "task_text": task_text,
        "planner_task_text": planner_task_text,
        "total_reward": float(total_reward),
        "steps_taken": step,
        "success": success,
        "env_error": env_error,
        "env_error_message": env_error_message,
        "fsm_result": agent._executor.result if agent._executor else None,
        "fsm_finished": agent._executor.finished if agent._executor else None,
        "elapsed_seconds": round(elapsed, 2),
    }

    with open(os.path.join(rollout_path, "episode_results.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    status = "SUCCESS" if success else "FAIL"
    print(f"  [{status}] reward={total_reward:.2f}  steps={step}  time={elapsed:.1f}s")
    return result


# ======================================================================
# Benchmark runner (all tasks in selected categories)
# ======================================================================

def run_benchmark(
    agent,
    tasks_dir: Path,
    categories: List[str],
    output_root: Path,
    obs_size: Tuple[int, int],
    max_steps: int,
    verbose: bool,
) -> Dict[str, Any]:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = output_root / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    benchmark = {
        "categories": {},
        "total_tasks": 0,
        "successful_tasks": 0,
        "total_reward": 0.0,
    }

    for category in categories:
        print(f"\n{'#'*70}")
        print(f"#  Category: {category}")
        print(f"{'#'*70}")

        yaml_files = list_tasks_in_category(tasks_dir, category)
        if not yaml_files:
            print(f"  (no tasks found)")
            continue

        cat_output = run_dir / category
        cat_results = []

        for yaml_path in yaml_files:
            try:
                result = run_single_task(
                    agent=agent,
                    yaml_path=yaml_path,
                    output_dir=cat_output,
                    obs_size=obs_size,
                    max_steps=max_steps,
                    verbose=verbose,
                )
                cat_results.append(result)
                benchmark["total_tasks"] += 1
                benchmark["total_reward"] += result["total_reward"]
                if result["success"]:
                    benchmark["successful_tasks"] += 1
            except Exception as e:
                logger.error("Task %s failed: %s", yaml_path.stem, e)
                traceback.print_exc()

        if cat_results:
            n = len(cat_results)
            n_success = sum(r["success"] for r in cat_results)
            avg_reward = sum(r["total_reward"] for r in cat_results) / n
            benchmark["categories"][category] = {
                "num_tasks": n,
                "successful_tasks": n_success,
                "success_rate": n_success / n,
                "avg_reward": avg_reward,
                "tasks": cat_results,
            }
            print(f"\n  Category summary: {n_success}/{n} success, avg_reward={avg_reward:.2f}")

    total = benchmark["total_tasks"]
    benchmark["overall_success_rate"] = (benchmark["successful_tasks"] / total) if total else 0.0
    benchmark["avg_reward"] = (benchmark["total_reward"] / total) if total else 0.0

    summary_path = run_dir / "benchmark_results.json"
    with open(summary_path, "w") as f:
        json.dump(benchmark, f, indent=2, ensure_ascii=False)

    print(f"\n{'#'*70}")
    print(f"#  BENCHMARK SUMMARY")
    print(f"{'#'*70}")
    print(f"  Total tasks      : {total}")
    print(f"  Successful       : {benchmark['successful_tasks']}")
    print(f"  Success rate     : {benchmark['overall_success_rate']:.2%}")
    print(f"  Average reward   : {benchmark['avg_reward']:.2f}")
    print(f"  Results saved to : {summary_path}")
    print(f"{'#'*70}\n")

    return benchmark


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run MCU benchmark tasks with the Minecraft Purple Agent",
    )

    parser.add_argument(
        "--category", type=str, default=None, choices=MCU_CATEGORIES,
        help="Task category to run (e.g. combat, crafting, mining_and_collecting)",
    )
    parser.add_argument(
        "--list-categories", action="store_true",
        help="List available categories and their task counts, then exit",
    )
    parser.add_argument("--tasks-dir", type=str, default="assets/mcu_tasks")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--obs-size", type=int, nargs=2, default=[640, 360])
    parser.add_argument("--verbose", action="store_true", default=True)

    # LLM Planner
    parser.add_argument("--planner-api-key", type=str, default="EMPTY")
    parser.add_argument("--planner-url", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--planner-model", type=str, default="gpt-4o")
    parser.add_argument("--planner-temperature", type=float, default=0.2)

    # JarvisVLA instruction runner (required)
    parser.add_argument("--vla-checkpoint-path", type=str, required=True)
    parser.add_argument("--vla-url", type=str, default="http://localhost:9020/v1")
    parser.add_argument("--vla-api-key", type=str, default="EMPTY")
    parser.add_argument("--vla-history-num", type=int, default=4)
    parser.add_argument("--vla-action-chunk-len", type=int, default=1)
    parser.add_argument("--vla-bpe", type=int, default=0)
    parser.add_argument("--vla-instruction-type", type=str, default="normal")
    parser.add_argument("--vla-temperature", type=float, default=0.7)
    parser.add_argument("--vla-no-camera-convert", action="store_true")

    # Manual plan (skip LLM for a single task)
    parser.add_argument("--plan", type=str, default=None,
                        help="Path to a pre-written plan JSON (single-task mode only)")
    parser.add_argument("--task", type=str, default=None,
                        help="Single task text (bypasses MCU category selection)")

    args = parser.parse_args()
    tasks_dir = Path(args.tasks_dir)

    # --list-categories
    if args.list_categories:
        print("\nAvailable MCU categories:\n")
        for cat in discover_categories(tasks_dir):
            n = len(list_tasks_in_category(tasks_dir, cat))
            print(f"  {cat:<30s}  ({n} tasks)")
        print()
        return

    from src.agent.agent import MinecraftPurpleAgent

    planner_cfg = {
        "api_key": args.planner_api_key,
        "base_url": args.planner_url,
        "model": args.planner_model,
        "temperature": args.planner_temperature,
    }
    vla_cfg = {
        "enabled": True,
        "checkpoint_path": args.vla_checkpoint_path,
        "base_url": args.vla_url,
        "api_key": args.vla_api_key,
        "history_num": args.vla_history_num,
        "action_chunk_len": args.vla_action_chunk_len,
        "bpe": args.vla_bpe,
        "instruction_type": args.vla_instruction_type,
        "temperature": args.vla_temperature,
        "convert_camera_21_to_11": not args.vla_no_camera_convert,
    }

    agent = MinecraftPurpleAgent(
        planner_cfg=planner_cfg,
        vla_cfg=vla_cfg,
        output_dir=args.output_dir,
    )

    # --- Single task mode (--task) ---
    if args.task:
        if args.plan:
            with open(args.plan) as f:
                plan = json.load(f)
            logger.info("Using pre-written plan: %s", args.plan)
            from src.executor.fsm_executor import FSMExecutor
            agent._plan = plan
            agent._executor = FSMExecutor(
                plan=plan,
                instruction_runner=agent._run_state_instruction,
            )
        else:
            agent.reset(task_text=args.task)

        try:
            from minestudio.simulator import MinecraftSim
            env = MinecraftSim(
                obs_size=tuple(args.obs_size)
            )
            obs, info = env.reset()
            env.action_type = "agent"
            state = agent.initial_state(task_text=args.task)
            total_reward = 0.0

            for step in range(1, args.max_steps + 1):
                image = obs.get("image", obs.get("pov"))
                action, state = agent.act(obs={"image": image}, state=state)
                logger.info("action: %s", action)
                if action is None:
                    break
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break

            logger.info("Done: steps=%d  reward=%.2f", step, total_reward)
            env.close()
        except ImportError:
            logger.warning("minestudio not installed — dry-run mode (plan saved)")
        return

    # --- Category benchmark mode ---
    if args.category is None:
        print("\nNo --category specified. Available categories:\n")
        for cat in discover_categories(tasks_dir):
            n = len(list_tasks_in_category(tasks_dir, cat))
            print(f"  {cat:<30s}  ({n} tasks)")
        print("\nUse --category <name> to run a benchmark.")
        return

    run_benchmark(
        agent=agent,
        tasks_dir=tasks_dir,
        categories=[args.category],
        output_root=Path(args.output_dir),
        obs_size=tuple(args.obs_size),
        max_steps=args.max_steps,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
