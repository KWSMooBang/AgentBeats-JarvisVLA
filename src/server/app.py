"""
Purple Agent A2A Server

HTTP server implementing the A2A (Agent-to-Agent) protocol.
Launches the ScriptedPolicyAgent as a Purple Agent service.

Usage:
    python -m src.server.app --port 9019 \\
        --planner-model gpt-4o --planner-url https://api.openai.com/v1 \\
        --vlm-model gpt-4o-mini --vlm-url http://localhost:11000/v1
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Optional

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

from src.server.executor import PurpleExecutor
from src.server.session_manager import SessionManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _build_agent_card(host: str, port: int, url_override: Optional[str] = None) -> dict:
    return {
        "name": "Minecraft Scripted Policy Agent",
        "description": (
            "LLM Planner + FSM Executor + VLM State Checker. "
            "Purple Agent for Minecraft tasks via scripted policy architecture."
        ),
        "url": url_override or f"http://{host}:{port}/",
        "version": "0.1.0",
        "defaultInputModes": ["text", "application/json"],
        "defaultOutputModes": ["text", "application/json"],
        "capabilities": {"streaming": False},
        "skills": [
            {
                "id": "scripted-policy",
                "name": "Scripted Policy",
                "description": "Generate and execute FSM-based Minecraft task plans",
                "tags": ["minecraft", "scripted-policy", "fsm", "llm-planner"],
                "examples": [],
            }
        ],
    }


def create_app(executor: PurpleExecutor, host: str, port: int, card_url: Optional[str] = None) -> Starlette:
    agent_card = _build_agent_card(host, port, card_url)

    async def health(request: Request):
        return PlainTextResponse("OK")

    async def agent_card_endpoint(request: Request):
        return JSONResponse(agent_card)

    async def execute_endpoint(request: Request):
        """
        Simplified A2A execute endpoint.
        Expects JSON body with a "message" field containing text.
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)

        # Extract text from various message formats
        text = None
        msg = body.get("message", body)
        if isinstance(msg, str):
            text = msg
        elif isinstance(msg, dict):
            parts = msg.get("parts", [])
            for part in parts:
                if isinstance(part, dict) and "text" in part:
                    text = part["text"]
                    break
            if text is None:
                text = msg.get("text")
        if text is None:
            text = json.dumps(body)

        context_id = body.get("context_id", "default")
        response_text = executor.handle_message(text, context_id)
        return JSONResponse({
            "message": {
                "role": "agent",
                "parts": [{"text": response_text}],
            }
        })

    routes = [
        Route("/health", health, methods=["GET"]),
        Route("/.well-known/agent-card.json", agent_card_endpoint, methods=["GET"]),
        Route("/execute", execute_endpoint, methods=["POST"]),
    ]

    return Starlette(routes=routes)


def main():
    parser = argparse.ArgumentParser(description="Scripted Policy Purple Agent Server")

    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9019)
    parser.add_argument("--card-url", type=str, default=None)

    # LLM Planner config
    parser.add_argument("--planner-api-key", type=str, default="EMPTY")
    parser.add_argument("--planner-url", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--planner-model", type=str, default="gpt-4o")
    parser.add_argument("--planner-temperature", type=float, default=0.2)

    # VLM State Checker config
    parser.add_argument("--vlm-api-key", type=str, default="EMPTY")
    parser.add_argument("--vlm-url", type=str, default="http://localhost:11000/v1")
    parser.add_argument("--vlm-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--vlm-temperature", type=float, default=0.1)

    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    planner_cfg = {
        "api_key": args.planner_api_key,
        "base_url": args.planner_url,
        "model": args.planner_model,
        "temperature": args.planner_temperature,
    }
    vlm_cfg = {
        "api_key": args.vlm_api_key,
        "base_url": args.vlm_url,
        "model": args.vlm_model,
        "temperature": args.vlm_temperature,
    }

    sessions = SessionManager()
    executor = PurpleExecutor(
        sessions=sessions,
        planner_cfg=planner_cfg,
        vlm_cfg=vlm_cfg,
        device=args.device,
    )

    app = create_app(executor, args.host, args.port, args.card_url)

    logger.info("Starting server on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
