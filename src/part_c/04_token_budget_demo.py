
import sys
import json
import re
import time
import random
import pandas as pd
from pathlib import Path

# Add project root to Python import path
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

import mock_llm

# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

INPUT_FILE = BASE_DIR / "data" / "clean" / "tickets_masked.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "triage_budget_demo.csv"

TOKEN_BUDGET = 6000

CATEGORIES = {"billing", "delivery", "technical", "fraud"}

ZERO_SHOT_SYSTEM = """
CLASSIFY customer support tickets.

Choose exactly one category from:
billing, delivery, technical, fraud.

Return JSON only in this format:
{"category":"billing","reason":"Short explanation"}

Do not include Markdown or additional text.
"""

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

tickets = pd.read_csv(INPUT_FILE, keep_default_na=False)

# Reset mock model before the full experiment
mock_llm.reset()

results = []
tokens_used = 0
total_retries = 0
budget_exhausted_count = 0

# --------------------------------------------------
# 3. PROCESS TICKETS
# --------------------------------------------------

for _, row in tickets.iterrows():

    ticket_text = str(row["masked_text"]).strip()

    result = {
        "ticket_id": row["ticket_id"],
        "true_category": row["true_category"],
        "predicted_category": "",
        "status": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "retry_count": 0
    }

    # Skip empty tickets
    if not ticket_text:
        result["status"] = "skipped_empty"

    # Flag long tickets
    elif len(ticket_text) > 1000:
        result["status"] = "flagged_long"

    # Stop new model calls after budget is reached
    elif tokens_used >= TOKEN_BUDGET:
        result["status"] = "budget_exhausted"
        budget_exhausted_count += 1

    else:

        messages = [
            {"role": "system", "content": ZERO_SHOT_SYSTEM},
            {"role": "user", "content": f"TICKET:\n{ticket_text}"}
        ]

        response = None
        retry_count = 0

        # Retry temporary errors, maximum 3 attempts
        for attempt in range(3):
            try:
                response = mock_llm.chat(messages)
                break

            except mock_llm.TransientLLMError:

                retry_count += 1

                if attempt == 2:
                    break

                delay = (
                    0.05
                    * (2 ** attempt)
                    * random.uniform(0.9, 1.15)
                )
                time.sleep(delay)

        result["retry_count"] = retry_count
        total_retries += retry_count

        if response is None:
            result["status"] = "retry_exhausted"

        else:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            call_tokens = prompt_tokens + completion_tokens

            tokens_used += call_tokens

            result["prompt_tokens"] = prompt_tokens
            result["completion_tokens"] = completion_tokens
            result["total_tokens"] = call_tokens

            content = response.choices[0].message.content or ""

            # Validate JSON response
            try:
                parsed = json.loads(content)
                category = str(
                    parsed.get("category", "")
                ).lower().strip()

                if category in CATEGORIES:
                    result["predicted_category"] = category
                    result["status"] = "classified_json"
                else:
                    result["status"] = "flagged_invalid_output"

            except (json.JSONDecodeError, AttributeError, TypeError):

                # Keyword fallback
                words = re.findall(r"[a-z]+", content.lower())
                valid_words = [
                    word for word in words if word in CATEGORIES
                ]

                if len(valid_words) == 1:
                    result["predicted_category"] = valid_words[0]
                    result["status"] = "classified_keyword"
                else:
                    result["status"] = "flagged_invalid_output"

    results.append(result)

# --------------------------------------------------
# 4. SAVE RESULTS
# --------------------------------------------------

df = pd.DataFrame(results)

df.to_csv(OUTPUT_FILE, index=False)

print("TOKEN BUDGET DEMONSTRATION COMPLETED")
print("-" * 45)

print("\nToken budget:", TOKEN_BUDGET)
print("Actual tokens used:", tokens_used)

print("\nStatus counts:")
print(df["status"].value_counts())

print("\nTotal retries:", total_retries)
print("Budget-exhausted tickets:", budget_exhausted_count)

print("\nSaved:", OUTPUT_FILE)
