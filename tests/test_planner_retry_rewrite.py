"""Tests for planner retry_exhausted postprocessing."""

from src.planner.llm_planner import normalize_retry_exhausted_transitions


def test_retry_exhausted_abort_rewritten_to_progress_state():
    spec = {
        "task": "defeat_enderman",
        "global_config": {"max_total_steps": 600, "vlm_check_interval": 30},
        "initial_state": "check_gear",
        "states": {
            "check_gear": {
                "description": "verify readiness",
                "instruction": "none",
                "max_retries": 2,
                "transitions": [
                    {
                        "condition": {"type": "vlm_check", "query": "ready?"},
                        "on_true": "locate_enderman",
                        "on_false": "equip_gear",
                    },
                    {
                        "condition": {"type": "retry_exhausted"},
                        "next_state": "abort",
                    },
                ],
            },
            "equip_gear": {
                "description": "equip",
                "instruction": "equip:diamond_sword",
                "transitions": [
                    {"condition": {"type": "always"}, "next_state": "check_gear"}
                ],
            },
            "locate_enderman": {
                "description": "locate",
                "instruction": "search for enderman",
                "transitions": [
                    {"condition": {"type": "always"}, "next_state": "success"}
                ],
            },
            "success": {"terminal": True, "result": "success"},
            "abort": {"terminal": True, "result": "failure"},
        },
    }

    updated = normalize_retry_exhausted_transitions(spec)
    retry_trans = updated["states"]["check_gear"]["transitions"][1]
    assert retry_trans["next_state"] == "locate_enderman"


def test_retry_exhausted_non_abort_target_kept():
    spec = {
        "task": "test",
        "global_config": {"max_total_steps": 100, "vlm_check_interval": 10},
        "initial_state": "s1",
        "states": {
            "s1": {
                "description": "state 1",
                "instruction": "search",
                "transitions": [
                    {
                        "condition": {"type": "retry_exhausted"},
                        "next_state": "s2",
                    }
                ],
            },
            "s2": {
                "description": "state 2",
                "instruction": "attack",
                "transitions": [{"condition": {"type": "always"}, "next_state": "success"}],
            },
            "success": {"terminal": True, "result": "success"},
            "abort": {"terminal": True, "result": "failure"},
        },
    }

    updated = normalize_retry_exhausted_transitions(spec)
    retry_trans = updated["states"]["s1"]["transitions"][0]
    assert retry_trans["next_state"] == "s2"
