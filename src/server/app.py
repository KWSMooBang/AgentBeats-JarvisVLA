"""Purple Agent A2A Server.

A2A protocol server for the JarvisVLA-based Purple Agent.
"""

from __future__ import annotations

import argparse
import logging

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from src.server.executor import PurpleExecutor
from src.server.session_manager import SessionManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _build_card_url(host: str, port: int, card_url: str | None) -> str:
    if card_url:
        return card_url
    advertised_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    return f"http://{advertised_host}:{port}/"


def main() -> None:
    parser = argparse.ArgumentParser(description="Scripted Policy Purple Agent Server")

    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9019)
    parser.add_argument("--card-url", type=str, default=None)

    # LLM Planner config
    parser.add_argument("--planner-api-key", type=str, default="EMPTY")
    parser.add_argument("--planner-url", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--planner-model", type=str, default="gpt-5.4-mini")
    parser.add_argument("--planner-temperature", type=float, default=0.2)

    # JarvisVLA instruction runner
    parser.add_argument("--vla-checkpoint-path", type=str, default="")
    parser.add_argument("--vla-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--vla-api-key", type=str, default="EMPTY")
    parser.add_argument("--vla-history-num", type=int, default=4)
    parser.add_argument("--vla-action-chunk-len", type=int, default=1)
    parser.add_argument("--vla-bpe", type=int, default=0)
    parser.add_argument("--vla-instruction-type", type=str, default="auto")
    parser.add_argument("--vla-temperature", type=float, default=0.7)
    parser.add_argument("--vla-no-camera-convert", action="store_true")

    # OpenAI VLM runner
    parser.add_argument("--vlm-api-key", type=str, default="EMPTY")
    parser.add_argument("--vlm-url", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--vlm-model", type=str, default="gpt-5.4-mini")
    parser.add_argument("--vlm-temperature", type=float, default=0.2)
    parser.add_argument("--vqa-interval-steps", type=int, default=600)

    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    if not args.vla_checkpoint_path:
        parser.error("--vla-checkpoint-path is required (VLA is mandatory)")

    planner_cfg = {
        "api_key": args.planner_api_key,
        "base_url": args.planner_url,
        "model": args.planner_model,
        "temperature": args.planner_temperature,
    }
    vla_cfg = {
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

    vlm_cfg = {
        "api_key": args.vlm_api_key,
        "base_url": args.vlm_url,
        "model": args.vlm_model,
        "temperature": args.vlm_temperature,
        "vqa_interval_steps": args.vqa_interval_steps,
    }

    sessions = SessionManager()
    executor = PurpleExecutor(
        sessions=sessions,
        planner_cfg=planner_cfg,
        vla_cfg=vla_cfg,
        vlm_cfg=vlm_cfg,
        device=args.device,
    )

    skill = AgentSkill(
        id="Planning-JarvisVLA",
        name="Planning JarvisVLA",
        description="Generate plan and execute with JarvisVLA instruction runner",
        tags=["minecraft", "jarvisvla", "planning"],
        examples=[],
    )

    agent_card = AgentCard(
        name="Minecraft Scripted Policy Agent",
        description="Purple Agent for Minecraft tasks via JarvisVLA and planning architecture.",
        url=_build_card_url(args.host, args.port, args.card_url),
        version="0.1.0",
        default_input_modes=["text", "application/json"],
        default_output_modes=["text", "application/json"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    logger.info("Starting server on %s:%d", args.host, args.port)
    uvicorn.run(app.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
