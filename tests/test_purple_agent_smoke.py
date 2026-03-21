from __future__ import annotations

from pathlib import Path

import numpy as np

from src.agent.agent import MinecraftPurpleAgent
from src.action.converter import noop_agent_action


class _DummyInstructionRunner:
    def reset(self) -> None:
        return None

    def run(self, image, instruction: str, instruction_type: str, state_def: dict):
        return {
            "__action_format__": "agent",
            "action": noop_agent_action(),
        }


def _make_agent(monkeypatch, tmp_path: Path) -> MinecraftPurpleAgent:
    monkeypatch.setattr(
        MinecraftPurpleAgent,
        "_build_instruction_runner",
        lambda self, _vla_cfg: _DummyInstructionRunner(),
    )
    return MinecraftPurpleAgent(
        planner_cfg={},
        vla_cfg={"enabled": True, "checkpoint_path": "dummy"},
        device="cpu",
        output_dir=str(tmp_path),
    )


def test_purple_agent_short_mode_smoke(monkeypatch, tmp_path):
    agent = _make_agent(monkeypatch, tmp_path)

    monkeypatch.setattr(
        agent.planner,
        "plan_task",
        lambda task_text, feedback=None, force_horizon=None: {
            "horizon": "short",
            "instruction": "kill_entity:zombie",
            "instruction_type": "normal",
        },
    )

    episode_dir = tmp_path / "short_episode"
    agent.reset(task_text="kill one zombie", episode_dir=str(episode_dir))
    state = agent.initial_state(task_text="kill one zombie")

    obs = {"image": np.zeros((8, 8, 3), dtype=np.uint8)}
    action, new_state = agent.act(obs=obs, state=state)

    assert isinstance(action, dict)
    assert new_state.execution_mode == "short"
    assert new_state.current_fsm_state == "short_direct"
    assert new_state.total_steps == 1
    assert (episode_dir / "plan.json").exists()


def test_purple_agent_long_mode_smoke(monkeypatch, tmp_path):
    agent = _make_agent(monkeypatch, tmp_path)

    long_plan = {
        "task": "mine one cobblestone",
        "step1": {
            "instruction": "mine_block:stone",
            "instruction_type": "auto",
            "condition": {
                "type": "timeout",
                "max_steps": 2,
                "next": "fallback",
            },
        },
        "fallback": {
            "instruction": "mine one cobblestone",
            "instruction_type": "normal",
            "condition": {"type": "always", "next": "step1"},
        },
    }

    monkeypatch.setattr(
        agent.planner,
        "plan_task",
        lambda task_text, feedback=None, force_horizon=None: {
            "horizon": "long",
            "plan": long_plan,
        },
    )

    episode_dir = tmp_path / "long_episode"
    agent.reset(task_text="mine one cobblestone", episode_dir=str(episode_dir))
    state = agent.initial_state(task_text="mine one cobblestone")

    obs = {"image": np.zeros((8, 8, 3), dtype=np.uint8)}
    action, new_state = agent.act(obs=obs, state=state)

    assert isinstance(action, dict)
    assert new_state.execution_mode == "long"
    assert new_state.current_fsm_state == "step1"
    assert new_state.total_steps == 1
    assert (episode_dir / "plan.json").exists()
