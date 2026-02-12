#app/api.py
from __future__ import annotations

import json
import uuid
from fastapi import FastAPI

from app.agent import process_message
from app.models import ChatRequest, ChatResponse, ToolDecision
from app.store import SESSION_STORE
from app.tools import openai_tools_schema
from evaluation.evaluation_llm import evaluate_chat_completion


app = FastAPI(title="swiss-army-knife")


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(payload: ChatRequest) -> ChatResponse:
    # Retrieve or create conversation state
    state = SESSION_STORE.get(payload.session_id)

    if state is None:
        # If your SESSION_STORE auto-creates state, you can remove this block
        raise ValueError(f"No session found for id: {payload.session_id}")

    last_message = payload.messages[-1].content if payload.messages else ""

    # --------------------------------------------------
    # 1️⃣ Process user message (agent logic)
    # --------------------------------------------------
    result = process_message(
        state,
        last_message,
        provided_parameters=payload.provided_parameters,
        force_tool=payload.force_tool,
    )

    # --------------------------------------------------
    # 2️⃣ LLM-as-judge evaluation (only runs if tool executed)
    # --------------------------------------------------
    evaluation = evaluate_chat_completion(state, result)

    # --------------------------------------------------
    # 3️⃣ Build ToolDecision (now includes evaluation)
    # --------------------------------------------------
    tool_decision = ToolDecision(
        tool_name=result.get("tool_name"),
        confidence=result.get("confidence", 0.0),
        require_approval=result.get("action") == "need_approval",
        missing_parameters=result.get("missing_parameters", []),
        collected_parameters=result.get("collected_parameters", {}),
        action=result.get("action", "none"),
        evaluation=evaluation,  # ✅ evaluation included
    )

    # --------------------------------------------------
    # 4️⃣ Build assistant message
    # --------------------------------------------------
    content = result.get("assistant_message", "")
    message: dict = {
        "role": "assistant",
        "content": content,
    }

    # If tool was executed, attach OpenAI-style tool call
    if result.get("action") == "executed":
        message["tool_calls"] = [
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": result.get("tool_name"),
                    "arguments": json.dumps(
                        result.get("tool_parameters", {}),
                        ensure_ascii=False,
                    ),
                },
            }
        ]

    # --------------------------------------------------
    # 5️⃣ Return OpenAI-compatible response
    # --------------------------------------------------
    response = ChatResponse(
        id=f"chatcmpl_{uuid.uuid4().hex}",
        object="chat.completion",
        session_id=state.session_id,
        choices=[
            {
                "index": 0,
                "message": message,
                "finish_reason": "stop",
            }
        ],
        tool_decision=tool_decision,
        tools=openai_tools_schema(),
    )

    return response
