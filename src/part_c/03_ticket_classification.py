
import json
import re
import time
import random
import sys
import pandas as pd
from pathlib import Path

# Locate project root
BASE_DIR = Path(__file__).resolve().parents[2]

# Add project root to Python import path
sys.path.insert(0, str(BASE_DIR))

import mock_llm

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------



INPUT_FILE = BASE_DIR / "data" / "clean" / "tickets_masked.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

tickets = pd.read_csv(INPUT_FILE, keep_default_na=False)

CATEGORIES = {"billing", "delivery", "technical", "fraud"}

# --------------------------------------------------
# 2. ZERO-SHOT PROMPT
# --------------------------------------------------

ZERO_SHOT_SYSTEM = """
CLASSIFY customer support tickets.

Choose exactly one category from:
billing, delivery, technical, fraud.

Return JSON only in this format:
{"category":"billing","reason":"Short explanation"}

Do not include Markdown or additional text.
"""

# --------------------------------------------------
# 3. FEW-SHOT PROMPT
# --------------------------------------------------

FEW_SHOT_SYSTEM = """
CLASSIFY customer support tickets.

Choose exactly one category from:
billing, delivery, technical, fraud.

EXAMPLES:

Ticket: I was charged twice for my purchase.
Response: {"category":"billing","reason":"Duplicate charge"}

Ticket: Someone used my account without permission.
Response: {"category":"fraud","reason":"Unauthorized account activity"}

Ticket: My order has not arrived yet.
Response: {"category":"delivery","reason":"Order not received"}

Return JSON only in this format:
{"category":"billing","reason":"Short explanation"}

Do not include Markdown or additional text.
"""

# --------------------------------------------------
# 4. CLASSIFICATION FUNCTION
# --------------------------------------------------


def classify_ticket(ticket_text, prompt, budget_state=None, token_budget=None):

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"TICKET:\n{ticket_text}"}
    ]

    response = None
    retry_count = 0

    # Retry temporary errors, maximum 3 attempts
    for attempt in range(3):

        # Check budget before EVERY model call
        if (
            token_budget is not None
            and budget_state["total_tokens"] >= token_budget
        ):
            return {
                "category": "",
                "status": "budget_exhausted",
                "reason": "Token budget exhausted",
                "retry_count": retry_count,
                "prompt_tokens": 0,
                "completion_tokens": 0
            }

        try:
            response = mock_llm.chat(messages)
            break

        except mock_llm.TransientLLMError:
            retry_count += 1

            if attempt == 2:
                return {
                    "category": "",
                    "status": "retry_exhausted",
                    "reason": "Temporary errors exceeded retry limit",
                    "retry_count": retry_count,
                    "prompt_tokens": 0,
                    "completion_tokens": 0
                }

            delay = 0.05 * (2 ** attempt) * random.uniform(0.9, 1.15)
            time.sleep(delay)

    # Extract response and token usage
    content = response.choices[0].message.content or ""

    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens

    # Update running token usage immediately
    if budget_state is not None:
        budget_state["total_tokens"] += (
            prompt_tokens + completion_tokens
        )

    # First attempt: validate JSON
    try:
        result = json.loads(content)
        category = str(result.get("category", "")).lower().strip()

        if category in CATEGORIES:
            return {
                "category": category,
                "status": "classified_json",
                "reason": result.get("reason", ""),
                "retry_count": retry_count,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens
            }

        return {
            "category": "",
            "status": "flagged_invalid_output",
            "reason": "Category outside allowed list",
            "retry_count": retry_count,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        }

    except (json.JSONDecodeError, AttributeError, TypeError):
        pass

    # Second attempt: keyword fallback
    words = re.findall(r"[a-z]+", content.lower())
    valid_words = [word for word in words if word in CATEGORIES]

    if len(valid_words) == 1:
        return {
            "category": valid_words[0],
            "status": "classified_keyword",
            "reason": "Recovered using keyword fallback",
            "retry_count": retry_count,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        }

    return {
        "category": "",
        "status": "flagged_invalid_output",
        "reason": "Could not recover a valid category",
        "retry_count": retry_count,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens
    }


# --------------------------------------------------
# 5. RUN A COMPLETE CLASSIFICATION EXPERIMENT
# --------------------------------------------------


def run_classification(prompt, experiment_name, token_budget=None):

    # Reset mock model before each complete experiment
    mock_llm.reset()

    results = []

    # Track cumulative token usage
    budget_state = {"total_tokens": 0}

    for _, row in tickets.iterrows():

        ticket_text = str(row["masked_text"]).strip()

        # Empty tickets are skipped
        if not ticket_text:
            result = {
                "category": "",
                "status": "skipped_empty",
                "reason": "Empty ticket",
                "retry_count": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0
            }

        # Long tickets are sent for human review
        elif len(ticket_text) > 1000:
            result = {
                "category": "",
                "status": "flagged_long",
                "reason": "Ticket exceeds length limit",
                "retry_count": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0
            }

        else:
            result = classify_ticket(
                ticket_text,
                prompt,
                budget_state,
                token_budget
            )

        results.append({
            "ticket_id": row["ticket_id"],
            "true_category": row["true_category"],
            "predicted_category": result["category"],
            "status": result["status"],
            "reason": result["reason"],
            "retry_count": result["retry_count"],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": (
                result["prompt_tokens"] + result["completion_tokens"]
            )
        })

    df = pd.DataFrame(results)

    # Calculate metrics
    classified = df[
        df["status"].isin(["classified_json", "classified_keyword"])
    ]

    correct = (
        classified["predicted_category"] ==
        classified["true_category"]
    ).sum()

    accuracy = (
        correct / len(classified) * 100
        if len(classified) > 0 else 0
    )

    print(f"\n{experiment_name.upper()} RESULTS")
    print("-" * 40)

    print("\nStatus counts:")
    print(df["status"].value_counts())

    print("\nCorrect classifications:", correct)
    print("Classified tickets:", len(classified))
    print(f"Accuracy: {accuracy:.2f}%")

    print("Total retries:", df["retry_count"].sum())
    print("Total prompt tokens:", df["prompt_tokens"].sum())
    print("Total completion tokens:", df["completion_tokens"].sum())
    print("Total tokens:", df["total_tokens"].sum())

    if token_budget is not None:
        print("Token budget:", token_budget)
        print("Running token usage:", budget_state["total_tokens"])

    return df


# --------------------------------------------------
# 6. RUN ZERO-SHOT AND FEW-SHOT EXPERIMENTS
# --------------------------------------------------

print("STARTING ZERO-SHOT CLASSIFICATION")

zero_shot = run_classification(
    ZERO_SHOT_SYSTEM,
    "zero_shot"
)

print("\nSTARTING FEW-SHOT CLASSIFICATION")

few_shot = run_classification(
    FEW_SHOT_SYSTEM,
    "few_shot",
    token_budget=100000
)
# --------------------------------------------------
# 7. SAVE COMPARISON RESULTS
# --------------------------------------------------

comparison = tickets[["ticket_id", "true_category"]].copy()

comparison["zero_shot_category"] = zero_shot["predicted_category"]
comparison["zero_shot_status"] = zero_shot["status"]

comparison["few_shot_category"] = few_shot["predicted_category"]
comparison["few_shot_status"] = few_shot["status"]

comparison["zero_shot_tokens"] = zero_shot["total_tokens"]
comparison["few_shot_tokens"] = few_shot["total_tokens"]

output_file = OUTPUT_DIR / "classification_comparison.csv"

comparison.to_csv(output_file, index=False)

print("\nCLASSIFICATION EXPERIMENTS COMPLETED")
print("Saved:", output_file)
