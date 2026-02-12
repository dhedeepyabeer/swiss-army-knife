# evaluation_llm.py
from __future__ import annotations
import os
import json
from typing import Any, Dict, List
import uuid
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langfuse import get_client

# Local store import from swiss-army-knife
from app.store import ConversationState

# -----------------------------
# Initialize Langfuse client (keys read from environment variables)
LF = get_client()

# -----------------------------
# LLM-as-judge configuration
JUDGE_LLM = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("EVAL_JUDGE_MODEL", "gpt-4o-mini"),
    temperature=0.0,
)

# -----------------------------
# Prompt template for the judge
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

# -----------------------------
def llm_as_judge(
    user_message: str,
    tool_name: str,
    parameters: dict,
    tool_result: dict
) -> dict[str, Any]:
    prompt = JUDGE_PROMPT.format(
        user_message=user_message,
        tool_name=tool_name,
        parameters=json.dumps(parameters),
        tool_result=json.dumps(tool_result),
    )

    try:
        ai_msg = JUDGE_LLM.invoke([HumanMessage(content=prompt)])
        return json.loads(ai_msg.content)
    except Exception as e:
        return {
            "tool_selection_correct": False,
            "parameters_correct": False,
            "task_completed": False,
            "comments": f"LLM judge failed or returned invalid JSON: {e}"
        }

# -----------------------------
def log_evaluation_to_langfuse(
    session_id: str,
    step_name: str,
    eval_output: dict[str, Any]
):
    try:
        LF.track_event(
            name=f"eval_{step_name}",
            session_id=session_id,
            metadata=eval_output
        )
    except Exception as e:
        print(f"[Langfuse logging failed] step='{step_name}' {e}")

# -----------------------------
def evaluate_conversation_llm(
    conversation: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    results = []

    for turn in conversation:
        user_msg = turn["user"]
        tool_name = turn["tool_name"]
        params = turn.get("tool_parameters", {})
        result = turn.get("tool_result", {})

        step_eval = llm_as_judge(
            user_message=user_msg,
            tool_name=tool_name,
            parameters=params,
            tool_result=result,
        )

        log_evaluation_to_langfuse(
            session_id=turn["session_id"],
            step_name=tool_name,
            eval_output=step_eval
        )

        results.append({
            "session_id": turn["session_id"],
            "user_message": user_msg,
            "tool": tool_name,
            "evaluation": step_eval,
        })

    return results

# -----------------------------
# CLI test
if __name__ == "__main__":
    conversation = [
        {
            "session_id": str(uuid.uuid4()),
            "user": "Create appointment with Dr. Smith tomorrow 3pm",
            "tool_name": "create_appointment",
            "tool_parameters": {"doctor": "Dr. Smith", "time": "tomorrow 3pm"},
            "tool_result": {"status": "success", "appointment_id": "apt123"}
        },
        {
            "session_id": str(uuid.uuid4()),
            "user": "Change it to 4pm",
            "tool_name": "update_appointment",
            "tool_parameters": {"appointment_id": "apt123", "new_time": "4pm"},
            "tool_result": {"status": "success"}
        }
    ]

    results = evaluate_conversation_llm(conversation)
    for r in results:
        print(json.dumps(r, indent=2))
