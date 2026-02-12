# evaluation_llm.py
from __future__ import annotations

import json
from typing import Dict, Any
from app.llm import get_llm
from app.tools import TOOLS
from app.store import ConversationState
import langfuse

# ------------------------------------------------------
# LLM-as-judge prompt template (updated for virtual care)
# ------------------------------------------------------
JUDGE_PROMPT = """
You are an expert evaluator for a virtual care assistant. The assistant can call specific tools
to perform tasks like booking, rescheduling, canceling appointments, checking availability,
verifying insurance, adding dependents, retrieving lab results, symptom triage, prescription refills, 
billing estimates, and handoffs to human agents.

Below is a catalog of the tools you can use:
{tools_catalog}

Your task is to evaluate a single step where the assistant chose a tool.

User message: {user_message}
Agent tool called: {tool_name}
Parameters passed: {parameters}
Tool execution result: {tool_result}

Evaluation criteria:
1) **Tool choice**: Was the agent’s selected tool appropriate for fulfilling the user request? 
   - Check if the tool name matches the intent in the user message.
   - Use the tool keywords and description to verify appropriateness.
2) **Parameter correctness**: Were all required parameters present and valid according to the tool definition? 
   - Check that all required fields are provided.
   - Check if values match expected formats (e.g., date, patient_id, provider_id, service_id).
3) **Task completion**: Based on the tool execution result, was the task successfully completed?
   - A "success" or "confirmed" status usually indicates completion.
   - If errors or missing outputs appear, mark as incomplete.
4) **Optional comments**: Provide brief reasoning for each evaluation point, pointing out missing parameters, wrong tool choice, or failed execution.

Answer in **EXACT JSON format** with these keys:
{{
  "tool_selection_correct": bool,
  "parameters_correct": bool,
  "task_completed": bool,
  "comments": str
}}

Notes:
- Always refer to the tool definitions provided in `tools_catalog` when judging appropriateness.
- Check that all required parameters are included and relevant.
- If a parameter is optional, missing it does NOT count as incorrect.
- If multiple tools could fit, use the one that best matches the user's intent.
"""

# ------------------------------------------------------
# Convert tools to string catalog for LLM prompt
# ------------------------------------------------------
def tools_catalog_str(tools: list) -> str:
    lines = []
    for t in tools:
        lines.append(
            f"- {t.name}: {t.description} (Required: {t.required})"
        )
    return "\n".join(lines)

# ------------------------------------------------------
# Main LLM-as-judge evaluation
# ------------------------------------------------------
def evaluate_tool_call(
    user_message: str,
    tool_name: str,
    parameters: dict,
    tool_result: dict,
    state_snapshot: dict,
) -> Dict[str, Any]:
    """
    Evaluate a single tool call using LLM-as-judge.
    """
    llm = get_llm()
    prompt = JUDGE_PROMPT.format(
        tools_catalog=tools_catalog_str(TOOLS),
        user_message=user_message,
        tool_name=tool_name,
        parameters=json.dumps(parameters, ensure_ascii=False),
        tool_result=json.dumps(tool_result, ensure_ascii=False),
    )

    # Build LangChain-style messages
    messages = [
        {"role": "system", "content": "You are an evaluator for AI tool calls in a healthcare assistant."},
        {"role": "user", "content": prompt},
    ]

    # Call LLM
    ai_response = llm.chat(messages)
    try:
        eval_result = json.loads(ai_response.content)
    except json.JSONDecodeError:
        eval_result = {
            "tool_selection_correct": False,
            "parameters_correct": False,
            "task_completed": False,
            "comments": "LLM judge did not return valid JSON.",
        }

    return eval_result

# ------------------------------------------------------
# Helper to build state snapshot
# ------------------------------------------------------
def snapshot_state(state: ConversationState) -> dict:
    return {
        "pending_tool": state.pending_tool.__dict__ if state.pending_tool else None,
        "awaiting_approval": state.awaiting_approval,
        "recent_messages": state.messages[-5:],
    }

# ------------------------------------------------------
# Hook: evaluate chat completions
# ------------------------------------------------------
def evaluate_chat_completion(state: ConversationState, chat_result: dict) -> dict:
    """
    Given a chat completion (from process_message),
    evaluate any executed tool call and log to Langfuse.
    """
    if chat_result.get("action") != "executed" or not chat_result.get("tool_name"):
        return {}  # No tool executed, nothing to evaluate

    user_message = state.messages[-2]["content"] if len(state.messages) >= 2 else ""
    tool_name = chat_result["tool_name"]
    parameters = chat_result.get("tool_parameters", {})
    tool_result = chat_result.get("tool_result", {})
    state_snapshot = snapshot_state(state)

    eval_result = evaluate_tool_call(
        user_message=user_message,
        tool_name=tool_name,
        parameters=parameters,
        tool_result=tool_result,
        state_snapshot=state_snapshot,
    )

    # Log to Langfuse
    try:
        langfuse.track(
            event_name="tool_call_evaluation",
            properties={
                "session_id": state.session_id,
                "tool_name": tool_name,
                "user_message": user_message,
                "evaluation": eval_result,
            },
        )
    except Exception as e:
        print(f"[Langfuse] failed to log evaluation: {e}")

    return eval_result
