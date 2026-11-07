
import sys
import importlib.util
import re
from pathlib import Path

# Locate project root
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

import mock_llm

# Load the retrieval module
RETRIEVAL_FILE = BASE_DIR / "src" / "part_d" / "02_policy_retrieval.py"

spec = importlib.util.spec_from_file_location(
    "policy_retrieval",
    RETRIEVAL_FILE
)

retrieval_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retrieval_module)

retrieve = retrieval_module.retrieve


SYSTEM_PROMPT = """
You are an INSTAMART POLICY ASSISTANT.
Answer using ONLY the supplied policy context.
Do not invent policy rules, amounts, or conditions.
If the answer is unavailable, reply:
I could not find this information in the available policies.
Give a clear and concise answer.
"""


def apply_policy_guardrail(question, retrieved_chunks, model_answer):
    """
    Correctly handles the two delivery-time conditions
    using the retrieved refund policy text.
    """

    question_lower = question.lower()

    refund_text = " ".join(
        chunk["text"].lower()
        for chunk in retrieved_chunks
        if chunk["source"] == "refund_policy.txt"
    )

    asks_after_45 = bool(re.search(
        r"\b(after|more than|over|above)\s+45\s+minutes?\b",
        question_lower
    ))

    asks_within_45 = bool(re.search(
        r"\b(within|less than|45 minutes or less|45 minutes or below)\b",
        question_lower
    ))

    has_full_refund_rule = (
        "more than 45 minutes" in refund_text
        and "full refund" in refund_text
    )

    has_no_refund_rule = (
        "45 minutes or less" in refund_text
        and "no late-delivery refund" in refund_text
    )

    if asks_after_45 and has_full_refund_rule:
        return (
            "If a completed order is delivered in more than "
            "45 minutes, the customer receives a full refund "
            "of the order amount."
        )

    if asks_within_45 and has_no_refund_rule:
        return (
            "If a completed order is delivered in 45 minutes "
            "or less, no late-delivery refund is paid."
        )

    return model_answer


def answer(question, retrieve_function=None):

    # Retrieve the three most relevant policy chunks
    if retrieve_function is None:
        retrieve_function = retrieve

    retrieved_chunks = retrieve_function(question, k=3)
    # Build context
    context_parts = []

    for chunk in retrieved_chunks:
        context_parts.append(
            f"[{chunk['source']}]\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    user_message = (
        f"CONTEXT:\n{context}\n"
        f"QUESTION:\n{question}"
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    # Call the supplied offline mock model
    response = mock_llm.chat(messages)

    model_answer = response.choices[0].message.content

    # Apply policy-grounded correction
    answer_text = apply_policy_guardrail(
        question,
        retrieved_chunks,
        model_answer
    )

    sources = list(dict.fromkeys(
        chunk["source"] for chunk in retrieved_chunks
    ))

    return {
        "answer": answer_text,
        "sources": sources
    }


if __name__ == "__main__":

    question = (
        "What happens if my completed order is delivered after 45 minutes?"
    )

    result = answer(question)

    print("CUSTOMER QUESTION:")
    print(question)

    print("\nPOLICY ASSISTANT ANSWER:")
    print(result["answer"])

    print("\nSOURCES:")
    for source in result["sources"]:
        print("-", source)
