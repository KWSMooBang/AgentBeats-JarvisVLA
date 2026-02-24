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
        model_name: str = "CraftJarvis/JarvisVLA-Qwen2-VL-7B",
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
        state: AgentState,
        deterministic: bool = True,
    ) -> Tuple[Any, AgentState]:
        """
        Generate action from observation.
        
        Args:
            obs: Dict with 'image' key (numpy array)
            state: Current agent state
            deterministic: Whether to use deterministic sampling
            
        Returns:
            (action_dict, new_state)
        """        
        try:
            # Get image from observation
            image = obs.get("image")
            if image is None:
                logger.warning("No image in observation, returning None")
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

            
            # Update state memory with current output
            new_state = AgentState(
                memory={"last_output": output},
                first=False,
                idle_count=state.idle_count,
                task_text=state.task_text,
            )
            
            return actions[0], new_state
            
        except Exception as e:
            logger.exception("Failed to generate action: %s", e)
            return None, state
        except Exception as e:
            logger.exception("Agent.act failed: %s", e)
            state.first = False
            return None, state

    def _build_messages(
        self,
        image: np.ndarray,
        task_text: str,
        state: AgentState,
    ) -> list:
        """Build message list for vLLM."""
        messages = []
        
        if self.processor_wrapper is None:
            # Fallback: simple text message
            messages.append({
                "role": "user",
                "content": f"{task_text}\nGenerate action for this observation."
            })
            return messages
        
        # Use processor wrapper to create proper message
        try:
            # Convert image to PIL or base64 format expected by processor
            image_input = self.processor_wrapper.create_image_input(image)
            
            # Create message with image
            prompt = task_text if state.first else "\nobservation: "
            message = self.processor_wrapper.create_message_vllm(
                role="user",
                input_type="image",
                prompt=[prompt],
                image=[image_input]
            )
            messages.append(message)
            
        except Exception as e:
            logger.warning("Failed to create image message: %s, using text only", e)
            messages.append({
                "role": "user",
                "content": f"{task_text}\nGenerate action."
            })
        
        return messages

    def _decode_actions(self, output: str) -> list:
        """
        Decode LLM output to action list.
        
        Returns:
            List of action dicts with 'buttons' and 'camera' keys
        """
        if self.action_tokenizer is None:
            logger.warning("No action_tokenizer, returning empty list")
            return []
        
        try:
            # Tokenize output
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct",
                trust_remote_code=True
            )
            token_ids = tokenizer(output)["input_ids"]
            
            # Decode using action tokenizer
            actions = self.action_tokenizer.decode(token_ids)
            
            logger.debug("Decoded %d actions from output", len(actions))
            return actions
            
        except Exception as e:
            logger.exception("Failed to decode actions: %s", e)
            return []



