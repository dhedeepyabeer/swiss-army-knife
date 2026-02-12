from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    messages: List[ChatMessage]
    provided_parameters: Optional[Dict[str, Any]] = None
    force_tool: Optional[str] = None


class ToolDecision(BaseModel):
    tool_name: Optional[str] = None
    confidence: float = 0.0
    require_approval: bool = False
    missing_parameters: List[str] = Field(default_factory=list)
    collected_parameters: Dict[str, Any] = Field(default_factory=dict)
    action: str = "none"  # none|need_parameters|need_approval|executed|no_tool
    evaluation: Optional[Dict[str, Any]] = None 

class ChatResponse(BaseModel):
    id: str
    object: str
    session_id: str
    choices: List[Dict[str, Any]]
    tool_decision: ToolDecision
    tools: List[Dict[str, Any]]


