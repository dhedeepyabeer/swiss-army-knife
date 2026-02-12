# agent_with_eval.py
from __future__ import annotations
import os
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
load_dotenv()

# --- LLM & Tool imports ---
from langchain_core.messages import HumanMessage
from app.llm import get_llm, build_llm_messages, parse_tool_call
from app.tools import TOOLS, get_tool
from app.store import ConversationState, PendingTool

# --- Langfuse integration ---
from langfuse import Langfuse

# --- Mock Tools for testing ---
@dataclass
class MockTool:
    name: str
    keywords: List[str]
    required: List[str]
    handler: Any

def dummy_handler(params):
    return {"status": "ok", "params": params}

TOOLS = [
    MockTool(name="create_appointment", keywords=["create", "appointment"], required=["new_start_time"], handler=dummy_handler),
    MockTool(name="change_appointment", keywords=["change", "appointment"], required=["new_start_time"], handler=dummy_handler),
    MockTool(name="cancel_appointment", keywords=["cancel", "appointment"], required=["appointment_id"], handler=dummy_handler),
    MockTool(name="reschedule_appointment", keywords=["reschedule", "appointment"], required=["new_start_time"], handler=dummy_handler),
]

# Override get_tool for testing
def get_tool(name: str) -> MockTool:
    for t in TOOLS:
        if t.name == name:
            return t
    return None


# --- Constants ---
APPROVAL_YES = {"yes", "y", "approve", "approved", "go ahead", "ok", "okay", "do it"}
APPROVAL_NO = {"no", "n", "decline", "deny", "stop", "cancel"}

# --- Initialize Langfuse ---
# LF = Langfuse(
#     secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
#     public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
#     base_url=os.getenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com"),
# )
# --- Dummy Langfuse logger for testing ---
class DummyLF:
    def log(self, *args, **kwargs):
        # Just print for debugging
        print(f"[DummyLF log] args={args}, kwargs={kwargs}")

LF = DummyLF()  # replace Langfuse initialization

# --- Helper Functions ---
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

def _use_llm() -> bool:
    return os.getenv("SAK_USE_LLM", "true").lower() in {"1", "true", "yes", "on"}

def _approval_decision(message: str) -> Optional[bool]:
    text = _normalize(message)
    if any(token == text or token in text for token in APPROVAL_YES):
        return True
    if any(token == text or token in text for token in APPROVAL_NO):
        return False
    return None

def _missing_params(required: List[str], provided: Dict[str, Any]) -> List[str]:
    return [p for p in required if p not in provided or provided[p] in {"", None}]

# # --- Langfuse Logging (without Trace) ---
# def log_step(state: ConversationState, step: str, tool_name=None, parameters=None, result=None, confidence=None):
#     try:
#         LF.log(
#             name=step,
#             session_id=state.session_id,
#             metadata={
#                 "project_name": os.getenv("LANGFUSE_PROJECT_NAME", "evaluation_project"),
#                 "tool_name": tool_name,
#                 "parameters": parameters,
#                 "result": result,
#                 "confidence": confidence
#             }
#         )
#     except Exception as e:
#         print(f"[Langfuse log failed] step='{step}' tool_name={tool_name} parameters={parameters} result={result} confidence={confidence} {e}")

def log_step(state, step, tool_name=None, parameters=None, result=None, confidence=None):
    try:
        LF.log(
            step=step,
            tool_name=tool_name,
            parameters=parameters,
            result=result,
            confidence=confidence
        )
    except Exception as e:
        print(f"[Langfuse log failed] step='{step}' tool_name={tool_name} parameters={parameters} result={result} confidence={confidence} {e}")

# --- Evaluation System ---
@dataclass
class EvalMetrics:
    tool_selection_accuracy: float = 1.0
    parameter_correctness: float = 1.0
    task_completion: float = 0.0
    error_handling: float = 0.0
    consistency: float = 1.0

    def update(self, step_metrics: Dict[str, float]):
        for k, v in step_metrics.items():
            setattr(self, k, min(getattr(self, k), v))

# --- Core Agent ---
def process_message(state: ConversationState, message: str, provided_parameters: Optional[Dict[str, Any]] = None):
    provided_parameters = provided_parameters or {}
    state.messages.append({"role": "user", "content": message})
    log_step(state, "user_message", parameters={"message": message})

    # --- Approval Step ---
    if state.awaiting_approval and state.pending_tool:
        approval = _approval_decision(message)
        if approval is True:
            log_step(state, "approval_received", tool_name=state.pending_tool.name, parameters=state.pending_tool.parameters)
            return _execute_tool(state)
        if approval is False:
            state.awaiting_approval = False
            log_step(state, "approval_denied", tool_name=state.pending_tool.name)
            state.pending_tool = None
            return _with_assistant(state, {"action": "no_tool", "assistant_message": "Understood, tool cancelled."})
        return _with_assistant(state, {"action": "need_approval", "assistant_message": "Please approve (yes/no)."})

    # --- Tool Execution Flow ---
    if state.pending_tool:
        tool = get_tool(state.pending_tool.name)
        if tool:
            extracted = _extract_kv(message)
            merged = {**state.pending_tool.parameters, **extracted, **provided_parameters}
            missing = _missing_params(tool.required, merged)
            state.pending_tool.parameters = merged
            state.pending_tool.missing = missing
            log_step(state, "parameter_collection", tool_name=tool.name, parameters=merged)
            if missing:
                return _with_assistant(state, {"action": "need_parameters", "assistant_message": f"Missing: {missing}"})
            return _execute_tool(state)

    # --- LLM Tool Selection ---
    if _use_llm():
        return _process_with_llm(state, message, provided_parameters)

    # --- Fallback: Keyword Selector ---
    tool_name, confidence = _select_tool_with_keyword(message)
    if not tool_name:
        return _with_assistant(state, {"action": "no_tool", "assistant_message": "Cannot determine tool."})

    tool = get_tool(tool_name)
    extracted = _extract_kv(message)
    merged = {**extracted, **provided_parameters}
    state.pending_tool = PendingTool(
        name=tool.name,
        parameters=merged,
        missing=_missing_params(tool.required, merged),
        confidence=confidence
    )
    log_step(state, "tool_selected", tool_name=tool.name, parameters=merged, confidence=confidence)
    return _execute_tool(state)

# --- Helper: Keyword-based tool selection ---
def _select_tool_with_keyword(message: str) -> Tuple[Optional[str], float]:
    for tool in TOOLS:
        if any(kw in message.lower() for kw in tool.keywords):
            print([(t.name, t.keywords) for t in TOOLS])
            print(_select_tool_with_keyword("Create appointment with Dr. Smith tomorrow 3pm"))
            return tool.name, 1.0
    return None, 0.0

# --- LLM processing (optional) ---
def _process_with_llm(state, message, provided_parameters):
    try:
        llm = get_llm()
        messages = build_llm_messages(state.messages)
        ai_message = llm.invoke(messages)
        tool_call = parse_tool_call(ai_message)
        if not tool_call:
            return _with_assistant(state, {"action": "none", "assistant_message": ai_message.content})
        tool_name, args = tool_call
        tool = get_tool(tool_name)
        merged = {**args, **provided_parameters}
        state.pending_tool = PendingTool(
            name=tool.name,
            parameters=merged,
            missing=_missing_params(tool.required, merged)
        )
        log_step(state, "llm_tool_selected", tool_name=tool_name, parameters=merged)
        return _execute_tool(state)
    except RuntimeError:
        return _with_assistant(state, {"action": "error", "assistant_message": "LLM failed, try again."})

# --- Execute Tool ---
# def _execute_tool(state: ConversationState):
#     tool = get_tool(state.pending_tool.name)
#     params = state.pending_tool.parameters
#     result = tool.handler(params)
#     log_step(state, "tool_executed", tool_name=tool.name, parameters=params, result=result)
#     state.pending_tool = None
#     return _with_assistant(state, {"action": "executed", "assistant_message": f"{tool.name} executed successfully."})

def _execute_tool(state: ConversationState):
    tool = get_tool(state.pending_tool.name)
    params = state.pending_tool.parameters
    result = tool.handler(params)
    log_step(state, "tool_executed", tool_name=tool.name, parameters=params, result=result)
    state.pending_tool = None
    return {
        "action": "executed",
        "assistant_message": f"{tool.name} executed successfully.",
        "tool_name": tool.name,
        "tool_parameters": params,
        "tool_result": result
    }

# --- Update assistant state ---
def _with_assistant(state, payload):
    if msg := payload.get("assistant_message"):
        state.messages.append({"role": "assistant", "content": msg})
    return payload

# --- Simple Evaluation System for CLI Testing ---
def evaluate_conversation(conversation: List[str]) -> EvalMetrics:
    state = ConversationState(session_id=str(uuid.uuid4()))
    metrics = EvalMetrics()
    for msg in conversation:
        response = process_message(state, msg)
        metrics.tool_selection_accuracy *= 1.0 if response.get("tool_name") else 0.0
        metrics.parameter_correctness *= 1.0 if "tool_parameters" in response else 0.0
        metrics.task_completion *= 1.0 if response.get("action") == "executed" else 0.0
    return metrics

# --- Example CLI ---
if __name__ == "__main__":
    conversations = [
        ["Create appointment with Dr. Smith tomorrow 3pm", "Change it to 4pm"],
        ["Cancel appointment id abc123"],
        ["Reschedule my appointment to same time tomorrow"],
    ]
    for i, conv in enumerate(conversations, 1):
        metrics = evaluate_conversation(conv)
        print(f"Conversation {i} metrics: {metrics}")
