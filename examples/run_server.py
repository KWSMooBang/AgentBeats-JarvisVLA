"""
Launch the Purple Agent A2A server.

Equivalent to:
    python -m src.server.app [options]

Usage:
    python examples/run_server.py --port 9019 \\
        --planner-model gpt-4o \\
        --vlm-model gpt-4o-mini --vlm-url http://localhost:11000/v1
"""

import sys
from src.server.app import main

if __name__ == "__main__":
    main()
