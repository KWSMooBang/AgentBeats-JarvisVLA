"""
A2A Protocol Message Models

Defines the JSON payloads exchanged between Green Agent (environment)
and this Purple Agent (policy).
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class InitPayload(BaseModel):
    """Green → Purple: episode initialisation."""
    type: str = Field(default="init")
    text: str = Field(..., description="Task description")


class ObservationPayload(BaseModel):
    """Green → Purple: per-step observation."""
    type: str = Field(default="obs")
    step: int = Field(..., description="Current step number")
    obs: str = Field(..., description="Base64-encoded observation image")


class ActionPayload(BaseModel):
    """Purple → Green: action response."""
    type: str = Field(default="action")
    action_type: str = Field(default="agent")
    buttons: List[int] = Field(..., description="Button action indices")
    camera: List[int] = Field(..., description="Camera action indices")


class AckPayload(BaseModel):
    """Purple → Green: initialisation acknowledgement."""
    type: str = Field(default="ack")
    success: bool = Field(...)
    message: Optional[str] = Field(default="")
