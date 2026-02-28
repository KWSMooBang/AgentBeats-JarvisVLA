# src/agent/agent.py
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass

import logging
import numpy as np 
import torch

logger = logging.getLogger(__name__)

@dataclass
class AgentState:
    """Agent state for session management."""
    memory: Optional[Any] = None
    first: bool = False
    idle_count: int = 0
    task_text: Optional[str] = None


class JarvisVLAAgent:
    """
    JarvisVLA Purple Agent wrapper for MCU evaluation.
    
    Wraps the existing VLLM_AGENT from jarvisvla.evaluate.agent_wrapper
    to provide Purple Agent interface.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str = "/workspace/models/JarvisVLA-Qwen2-VL-7B",
        api_key: str = "EMPTY",
        device: Optional[str] = None,
        temperature: float = 0.5,
        history_num: int = 0,
        action_chunk_len: int = 1,
        instruction_type: str = "normal",
    ):
        self._device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Import and initialize the original VLLM_AGENT
        try:
            from jarvisvla.evaluate.agent_wrapper import VLLM_AGENT
            
            self.agent = VLLM_AGENT(
                checkpoint_path=model_name,
                base_url=base_url,
                api_key=api_key,
                history_num=history_num,
                action_chunk_len=action_chunk_len,
                instruction_type=instruction_type,
                temperature=temperature,
            )
            logger.info(
                "JarvisVLAAgent initialized: base_url=%s, model=%s, "
                "history_num=%d, instruction_type=%s",
                base_url, model_name, history_num, instruction_type
            )
        except Exception as e:
            logger.exception("Failed to initialize VLLM_AGENT: %s", e)
            raise

    @property
    def device(self) -> torch.device:
        return self._device

    def reset(self) -> None:
        """Reset agent state."""
        self.agent.reset()
        logger.debug("JarvisVLAAgent reset")

    def initial_state(self, task_text: Optional[str] = None) -> AgentState:
        """Create initial agent state."""
        return AgentState(memory=None, first=True, idle_count=0, task_text=task_text)

    def act(
        self,
        obs: Dict[str, Any],
        state: AgentState
    ) -> Tuple[Any, AgentState]:
        """
        Generate action from observation.
        
        Args:
            obs: Dict with 'image' key (numpy array)
            state: Current agent state
            
        Returns:
            (action_dict, new_state) or (None, state) if error occurs
        """        
        try:
            # Get image from observation
            image = obs.get("image")
            if image is None:
                logger.warning("No image in observation, returning None")
                state.first = False
                return None, state
            
            # Get task text
            task_text = state.task_text or "Complete the task in Minecraft."
            
            # Call the original VLLM_AGENT.forward()
            # forward(self, observations, instructions, verbos=False, need_crafting_table=False)
            action = self.agent.forward(
                observations=[image],
                instructions=[task_text],
                verbos=False,
                need_crafting_table=False,
            )
            
            if action is None:
                logger.warning("Agent returned None action")
                state.first = False
                return None, state
            
            # Validate action format
            if not isinstance(action, dict):
                logger.error("Invalid action type: %s", type(action))
                state.first = False
                return None, state
            
            if "buttons" not in action or "camera" not in action:
                logger.error("Action missing required keys: %s", action)
                state.first = False
                return None, state
            
            # Update state
            new_state = AgentState(
                memory=None,
                first=False,
                idle_count=state.idle_count,
                task_text=state.task_text,
            )
            
            logger.debug("Generated action: %s", action)
            return action, new_state
            
        except Exception as e:
            logger.exception("Agent.act failed: %s", e)
            state.first = False
            return None, state



