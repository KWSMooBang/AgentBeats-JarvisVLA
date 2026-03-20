"""
VLM State Checker

All perception is funneled through this module.  The agent has NO access
to env info — only raw observation images — so every state check
(inventory, target visibility, scene description) goes through the VLM.

Supports two backends:
  1. OpenAI-compatible API  (GPT-4o, Qwen2-VL via vllm, etc.)
  2. Anthropic API          (Claude)
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _encode_image_b64(image: np.ndarray, fmt: str = "JPEG") -> str:
    pil = Image.fromarray(image)
    buf = io.BytesIO()
    pil.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


class VLMStateChecker:
    """
    Thin wrapper around an OpenAI-compatible VLM endpoint.

    Every query sends a single image + text prompt and returns
    a short text response.
    """

    def __init__(
        self,
        api_key: str = "EMPTY",
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
        temperature: float = 0.1,
        max_tokens: int = 64,
    ):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    # ------------------------------------------------------------------
    # Core query
    # ------------------------------------------------------------------

    def query(self, image: np.ndarray, prompt: str) -> str:
        """Send image + prompt to VLM; return raw text response."""
        b64 = _encode_image_b64(image)
        self._call_count += 1
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64}",
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.exception("VLM query failed: %s", e)
            return ""

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def ask_yes_no(self, image: np.ndarray, question: str) -> bool:
        """Ask a yes/no question about the current observation."""
        prompt = (
            f"Look at this Minecraft screenshot and answer: {question}\n"
            "Reply with ONLY 'Yes' or 'No'."
        )
        answer = self.query(image, prompt)
        print(f"VLM prompt: {prompt}\nVLM answer: {answer}")
        return answer.lower().startswith("yes")

    def describe_scene(self, image: np.ndarray) -> str:
        prompt = "Briefly describe what you see in this Minecraft screenshot in 1-2 sentences."
        return self.query(image, prompt)

    def locate_target(
        self, image: np.ndarray, target_type: str
    ) -> Optional[dict]:
        """
        Ask VLM where a target is in the frame.

        Returns None if not found, otherwise:
            {"offset_x": float, "offset_y": float, "size": float}

        offset_x/y are in [-1, 1] relative to image center.
        size is approximate fraction of image area.
        """
        prompt = (
            f"Look at this Minecraft screenshot. "
            f"Is there a '{target_type}' visible? "
            f"If yes, respond with JSON: "
            f'{{"found": true, "x": <0-1 horizontal position>, '
            f'"y": <0-1 vertical position>, "size": <fraction of image>}}. '
            f"If not visible, respond: "
            f'{{"found": false}}'
        )
        raw = self.query(image, prompt)
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                return None
            data = json.loads(match.group())
            if not data.get("found"):
                return None
            x = float(data.get("x", 0.5))
            y = float(data.get("y", 0.5))
            size = float(data.get("size", 0.05))
            return {
                "offset_x": (x - 0.5) * 2.0,
                "offset_y": (y - 0.5) * 2.0,
                "size": size,
            }
        except (json.JSONDecodeError, ValueError, TypeError):
            return None
