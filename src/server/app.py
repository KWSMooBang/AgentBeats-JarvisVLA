# src/server/app.py
from __future__ import annotations

import argparse
import logging
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from src.server.executor import Executor
from src.server.session_manager import SessionManager

from starlette.responses import PlainTextResponse

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="Run JarvisVLA Purple Agent")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9019)
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    parser.add_argument("--base-url", type=str, default="http://localhost:9020/v1",
                        help="vLLM server URL")
    parser.add_argument("--model-name", type=str, default="CraftJarvis/JarvisVLA-Qwen2-VL-7B",
                        help="Model name")
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--device", default=None)
    parser.add_argument("--state-ttl", type=int, default=3600)
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Model temperature")
    parser.add_argument("--history-num", type=int, default=0,
                        help="Number of history steps to keep")
    parser.add_argument("--action-chunk-len", type=int, default=1,
                        help="Action chunk length")
    parser.add_argument("--instruction-type", type=str, default="normal",
                        choices=["normal", "simple", "recipe"],
                        help="Instruction type")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Agent Card 
    # ------------------------------------------------------------------
    skill = AgentSkill(
        id="jarvisvla-purple-policy",
        name="JarvisVLA Purple Policy",
        description="Purple policy server for JarvisVLA on MCU AgentBeats",
        tags=["jarvisvla", "minecraft", "vla", "purple"],
        examples=[],
    )

    agent_card = AgentCard(
        name="JarvisVLA Purple Agent",
        description="Purple agent for JarvisVLA compatible with MCU Green evaluator",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="0.1.0",
        default_input_modes=["text", "application/json"],
        default_output_modes=["text", "application/json"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
    )

    # ------------------------------------------------------------------
    # Session + Executor
    # ------------------------------------------------------------------
    sessions = SessionManager(ttl_seconds=args.state_ttl)

    executor = Executor(
        sessions=sessions,
        base_url=args.base_url,
        model_name=args.model_name,
        device=args.device,
        deterministic=args.deterministic,
        state_ttl_seconds=args.state_ttl,
        debug=True,
    )

    # ------------------------------------------------------------------
    # Request handler
    # ------------------------------------------------------------------
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    # ------------------------------------------------------------------
    # A2A application
    # ------------------------------------------------------------------
    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    asgi_app = app.build()

    @asgi_app.route("/health")
    async def health(request):
        return PlainTextResponse("OK")

    logging.info("Starting JarvisVLA Purple Agent on %s:%d", args.host, args.port)
    logging.info("vLLM server: %s", args.base_url)
    logging.info("Model: %s", args.model_name)

    uvicorn.run(
        asgi_app,
        host=args.host,
        port=args.port,
        log_level="debug",
    )


if __name__ == "__main__":
    main()
