from __future__ import annotations
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage

from app.config import get_confidence_threshold
from app.confidence import get_confidence_model
from app.llm import build_langchain_tools, build_llm_messages, get_llm, parse_tool_call
from app.logging_utils import log_event
from app.tools import TOOLS, get_tool
from app.store import ConversationState, PendingTool
from evaluation.evaluation_llm import evaluate_chat_completion



APPROVAL_YES = {"yes", "y", "approve", "approved", "go ahead", "ok", "okay", "do it"}
APPROVAL_NO = {"no", "n", "decline", "deny", "stop", "cancel"}


def _normalize(text: str) -> str:
    return text.lower().strip()


def _extract_kv(text: str) -> Dict[str, str]:
    pairs = re.findall(r"(\w+)\s*[:=]\s*([^,\n]+)", text)
    extracted = {k.strip(): v.strip() for k, v in pairs}
    extracted.update(_extract_domain_params(text))
    return extracted


def _extract_domain_params(text: str) -> Dict[str, str]:
    lowered = text.lower()
    params: Dict[str, str] = {}

    id_match = re.search(r"(?:apt|appt|appointment)\s*id\s*(?:is\s*)?([a-z0-9_-]+)", lowered)
    if id_match:
        params["appointment_id"] = id_match.group(1)

    if "same time tomorrow" in lowered:
        params["new_start_time"] = "tomorrow same time"
    else:
        time_match = re.search(r"tomorrow(?:\s+at)?\s+(\d{1,2}(:\d{2})?\s*(am|pm)?)", lowered)
        if time_match:
            params["new_start_time"] = f"tomorrow {time_match.group(1).strip()}"

    return params


def _select_tool(message: str) -> Tuple[Optional[str], float]:
    model = get_confidence_model()
    result = model.score(message, TOOLS)
    return result.tool_name, result.confidence


def _select_tool_with_scores(message: str) -> Tuple[Optional[str], float, Dict[str, float]]:
    model = get_confidence_model()
    result = model.score(message, TOOLS)
    return result.tool_name, result.confidence, result.scores


def _blend_confidence(selector_confidence: float, llm_used: bool, llm_args: Dict[str, Any]) -> float:
    if not llm_used:
        return selector_confidence
    llm_confidence = 0.85 if llm_args else 0.75
    return max(selector_confidence, llm_confidence)


def _use_llm() -> bool:
    return os.getenv("SAK_USE_LLM", "true").lower() in {"1", "true", "yes", "on"}


def _missing_params(required: List[str], provided: Dict[str, Any]) -> List[str]:
    return [param for param in required if param not in provided or provided[param] in {"", None}]


def _confidence_requires_approval(confidence: float) -> bool:
    return confidence < get_confidence_threshold()


def _approval_decision(message: str) -> Optional[bool]:
    text = _normalize(message)
    if any(token == text or token in text for token in APPROVAL_YES):
        return True
    if any(token == text or token in text for token in APPROVAL_NO):
        return False
    return None


def process_message(
    state: ConversationState,
    message: str,
    provided_parameters: Optional[Dict[str, Any]] = None,
    force_tool: Optional[str] = None,
) -> Dict[str, Any]:
    provided_parameters = provided_parameters or {}
    state.messages.append({"role": "user", "content": message})
    log_event("user_message", {"session_id": state.session_id, "message": message})

    if state.awaiting_approval and state.pending_tool:
        approval = _approval_decision(message)
        if approval is True:
            log_event(
                "approval_received",
                {"session_id": state.session_id, "tool": state.pending_tool.name, "approved": True},
            )
            return _execute_pending(state)
        if approval is False:
            state.awaiting_approval = False
            state.pending_tool = None
            log_event(
                "approval_received",
                {"session_id": state.session_id, "tool": None, "approved": False},
            )
            return _with_assistant(state, {
                "action": "no_tool",
                "assistant_message": "Understood. I won't run that tool. What would you like to do next?",
            })
        return _with_assistant(state, {
            "action": "need_approval",
            "assistant_message": "Please confirm: should I proceed with the tool call? (yes/no)",
        })

    if state.pending_tool:
        tool = get_tool(state.pending_tool.name)
        if tool:
            extracted = _extract_kv(message)
            log_event(
                "extracted_parameters",
                {"session_id": state.session_id, "source": "pending_tool", "extracted": extracted},
            )
            merged = {**state.pending_tool.parameters, **extracted, **provided_parameters}
            missing = _missing_params(tool.required, merged)
            state.pending_tool.parameters = merged
            state.pending_tool.missing = missing
            log_event(
                "collect_parameters",
                {
                    "session_id": state.session_id,
                    "tool": tool.name,
                    "missing": missing,
                    "collected": merged,
                },
            )
            if missing:
                return {
                    "action": "need_parameters",
                    "assistant_message": _format_missing_prompt(tool.name, missing),
                    "tool_name": tool.name,
                    "missing_parameters": missing,
                    "collected_parameters": merged,
                }
            return _decide_or_execute(state, tool, merged, confidence=state.pending_tool.confidence)

    if _use_llm() and not force_tool:
        return _process_with_llm(state, message, provided_parameters)

    confidence = 1.0
    scores: Dict[str, float] = {}
    if force_tool:
        tool_name = force_tool
    else:
        tool_name, confidence, scores = _select_tool_with_scores(message)
    log_event(
        "tool_selection",
        {
            "session_id": state.session_id,
            "tool": tool_name,
            "confidence": confidence,
            "scores": scores,
            "force_tool": force_tool,
        },
    )

    if not tool_name:
        return _with_assistant(state, {
            "action": "no_tool",
            "assistant_message": "I couldn't determine the right tool. Can you rephrase or be more specific?",
        })

    tool = get_tool(tool_name)
    if not tool:
        return _with_assistant(state, {
            "action": "no_tool",
            "assistant_message": "That tool isn't available. Please try a different request.",
        })

    extracted = _extract_kv(message)
    log_event(
        "extracted_parameters",
        {"session_id": state.session_id, "source": "initial", "extracted": extracted},
    )
    merged = {**extracted, **provided_parameters}
    missing = _missing_params(tool.required, merged)
    state.pending_tool = PendingTool(name=tool.name, parameters=merged, missing=missing, confidence=confidence)

    if missing:
        log_event(
            "tool_missing_parameters",
            {
                "session_id": state.session_id,
                "tool": tool.name,
                "missing": missing,
                "collected": merged,
            },
        )
        return _with_assistant(state, {
            "action": "need_parameters",
            "assistant_message": _format_missing_prompt(tool.name, missing),
            "tool_name": tool.name,
            "missing_parameters": missing,
            "collected_parameters": merged,
            "confidence": confidence,
        })

    return _decide_or_execute(state, tool, merged, confidence=confidence)


def _process_with_llm(
    state: ConversationState,
    message: str,
    provided_parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    provided_parameters = provided_parameters or {}

    try:
        llm = get_llm()
    except RuntimeError:
        tool_name, confidence, scores = _select_tool_with_scores(message)
        log_event(
            "tool_selection",
            {
                "session_id": state.session_id,
                "tool": tool_name,
                "confidence": confidence,
                "scores": scores,
                "source": "fallback",
            },
        )
        return _process_with_selector(state, tool_name, confidence, provided_parameters, {})

    tools = build_langchain_tools(TOOLS)
    llm_with_tools = llm.bind_tools(tools)
    messages = build_llm_messages(state.messages)
    ai_message = llm_with_tools.invoke(messages)
    log_event(
        "llm_response",
        {
            "session_id": state.session_id,
            "content": ai_message.content,
            "tool_calls": getattr(ai_message, "tool_calls", None),
        },
    )

    tool_call = parse_tool_call(ai_message)
    if not tool_call:
        assistant_message = ai_message.content or "How can I help you today?"
        return _with_assistant(state, {
            "action": "none",
            "assistant_message": assistant_message,
        })

    tool_name, args = tool_call
    _, selector_confidence, scores = _select_tool_with_scores(message)
    confidence = _blend_confidence(selector_confidence, llm_used=True, llm_args=args)
    log_event(
        "tool_selection",
        {
            "session_id": state.session_id,
            "tool": tool_name,
            "confidence": confidence,
            "scores": scores,
            "source": "llm",
            "llm_args": args,
            "selector_confidence": selector_confidence,
        },
    )
    return _process_with_selector(state, tool_name, confidence, provided_parameters, args)


def _process_with_selector(
    state: ConversationState,
    tool_name: Optional[str],
    confidence: float,
    provided_parameters: Dict[str, Any],
    llm_args: Dict[str, Any],
) -> Dict[str, Any]:
    if not tool_name:
        return _with_assistant(state, {
            "action": "no_tool",
            "assistant_message": "I couldn't determine the right tool. Can you rephrase or be more specific?",
        })

    tool = get_tool(tool_name)
    if not tool:
        return _with_assistant(state, {
            "action": "no_tool",
            "assistant_message": "That tool isn't available. Please try a different request.",
        })

    extracted = _extract_kv(state.messages[-1]["content"])
    log_event(
        "extracted_parameters",
        {"session_id": state.session_id, "source": "llm_selector", "extracted": extracted},
    )
    merged = {**llm_args, **extracted, **provided_parameters}
    missing = _missing_params(tool.required, merged)
    state.pending_tool = PendingTool(name=tool.name, parameters=merged, missing=missing, confidence=confidence)
    log_event(
        "tool_parameters",
        {
            "session_id": state.session_id,
            "tool": tool.name,
            "missing": missing,
            "collected": merged,
        },
    )

    if missing:
        return _with_assistant(state, {
            "action": "need_parameters",
            "assistant_message": _format_missing_prompt(tool.name, missing),
            "tool_name": tool.name,
            "missing_parameters": missing,
            "collected_parameters": merged,
            "confidence": confidence,
        })

    return _decide_or_execute(state, tool, merged, confidence=confidence)


def _decide_or_execute(
    state: ConversationState,
    tool: Any,
    parameters: Dict[str, Any],
    confidence: float,
) -> Dict[str, Any]:
    if _confidence_requires_approval(confidence):
        state.awaiting_approval = True
        state.pending_tool = PendingTool(name=tool.name, parameters=parameters, missing=[], confidence=confidence)
        log_event(
            "approval_required",
            {
                "session_id": state.session_id,
                "tool": tool.name,
                "confidence": confidence,
                "threshold": get_confidence_threshold(),
            },
        )
        return _with_assistant(state, {
            "action": "need_approval",
            "assistant_message": (
                f"I think we should call `{tool.name}`, but my confidence is {confidence:.2f}. "
                "Do you want me to proceed? (yes/no)"
            ),
            "tool_name": tool.name,
            "collected_parameters": parameters,
            "confidence": confidence,
        })

    state.pending_tool = PendingTool(name=tool.name, parameters=parameters, missing=[], confidence=confidence)
    return _execute_pending(state, confidence=confidence)


def _execute_pending(state: ConversationState, confidence: float | None = None) -> Dict[str, Any]:
    if not state.pending_tool:
        return _with_assistant(state, {
            "action": "no_tool",
            "assistant_message": "No tool is pending execution.",
        })
    if confidence is None:
        confidence = state.pending_tool.confidence
    tool = get_tool(state.pending_tool.name)
    if not tool:
        return _with_assistant(state, {
            "action": "no_tool",
            "assistant_message": "That tool isn't available.",
        })

    parameters = dict(state.pending_tool.parameters)
    result = tool.handler(parameters)
    state.awaiting_approval = False
    state.pending_tool = None
    log_event(
        "tool_executed",
        {
            "session_id": state.session_id,
            "tool": tool.name,
            "parameters": parameters,
            "result": result,
        },
    )

    assistant_message = f"Tool `{tool.name}` executed successfully."
    if _use_llm():
        try:
            llm = get_llm()
            messages = build_llm_messages(state.messages)
            tool_payload = json.dumps(result, ensure_ascii=False)
            messages.append(
                HumanMessage(
                    content=(
                        f"Tool `{tool.name}` returned: {tool_payload}. "
                        "Respond to the user with a concise update and next steps if needed."
                    )
                )
            )
            ai_message = llm.invoke(messages)
            if ai_message.content:
                assistant_message = ai_message.content
        except RuntimeError:
            pass

    return _with_assistant(state, {
        "action": "executed",
        "assistant_message": assistant_message,
        "tool_name": tool.name,
        "tool_parameters": parameters,
        "tool_result": result,
        "confidence": confidence,
    })


def _format_missing_prompt(tool_name: str, missing: List[str]) -> str:
    joined = ", ".join(missing)
    return f"To run `{tool_name}`, I still need: {joined}. Please provide them as `param: value`."


def _with_assistant(state: ConversationState, payload: Dict[str, Any]) -> Dict[str, Any]:
    message = payload.get("assistant_message")
    if message:
        state.messages.append({"role": "assistant", "content": message})
    return payload
