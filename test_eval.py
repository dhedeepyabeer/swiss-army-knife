# test_eval.py

# from evaluation.runner import run_dynamic_evaluation
# from evaluation.dataset import DYNAMIC_TEST_CONVERSATIONS

# def main():
#     for idx, conv_messages in enumerate(DYNAMIC_TEST_CONVERSATIONS):
#         result = run_dynamic_evaluation(conv_messages)
#         print(f"Conversation {idx+1} metrics:")
#         print("Step metrics:", result["step_metrics"])
#         print("Aggregate metrics:", result["aggregate_metrics"])
#         print("-"*60)

# if __name__ == "__main__":
#     main()

# from evaluation.evaluation_llm import evaluate_conversation_llm
# # Example dataset
# dataset = [
#     {
#         "session_id": "xyz123",
#         "user": "Create an appointment for tomorrow at 3pm",
#         "tool_name": "appointment_book",
#         "tool_parameters": {"start_time": "2026-02-13T15:00:00"},
#         "tool_result": {"appointment_id": "apt_abc", "status": "confirmed"}
#     },
#     # ... more steps
# ]

# results = evaluate_conversation_llm(dataset)

# for r in results:
#     print(r)

import uuid
from evaluation.evaluation_llm  import evaluate_conversation_llm

conversation = [
    {

        "session_id": str(uuid.uuid4()),
        "user": "Create appointment with Dr. Smith tomorrow 3pm",
        "tool_name": "create_appointment",
        "tool_parameters": {"doctor": "Dr. Smith", "time": "tomorrow 3pm"},
        "tool_result": {"status": "success", "appointment_id": "abc123"}
    },
    {
        "session_id": str(uuid.uuid4()),
        "user": "Change it to 4pm",
        "tool_name": "update_appointment",
        "tool_parameters": {"appointment_id": "abc123", "new_time": "4pm"},
        "tool_result": {"status": "success"}
    },
]

results = evaluate_conversation_llm(conversation)

for r in results:
    print(r)

