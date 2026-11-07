
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

# Final few-shot token budget
TOKEN_BUDGET = 100000

# Load input files
tickets = pd.read_csv(
    BASE_DIR / "data/clean/tickets_masked.csv",
    keep_default_na=False
)

comparison = pd.read_csv(
    BASE_DIR / "outputs/classification_comparison.csv"
)

segments = pd.read_csv(
    BASE_DIR / "outputs/customer_segments.csv"
)

# Keep only the required classification results
classification = comparison[
    ["ticket_id", "few_shot_category",
     "few_shot_status", "few_shot_tokens"]
].copy()

# Rename columns to match the final output
classification = classification.rename(columns={
    "few_shot_category": "category",
    "few_shot_status": "status",
    "few_shot_tokens": "tokens_used"
})

# Apply the 100000-token budget to the completed few-shot results.
# The total recorded usage is below this budget.
classification["tokens_used"] = (
    pd.to_numeric(classification["tokens_used"], errors="coerce")
    .fillna(0)
    .astype(int)
)

total_tokens = int(classification["tokens_used"].sum())

if total_tokens > TOKEN_BUDGET:
    raise ValueError(
        f"Few-shot token usage {total_tokens} exceeds "
        f"the budget {TOKEN_BUDGET}"
    )

# Merge ticket information and customer segments
result = tickets.merge(
    classification,
    on="ticket_id",
    how="left"
)

result = result.merge(
    segments[["customer_id", "segment_name", "is_premium"]],
    on="customer_id",
    how="left"
)

# Assign route and priority
def assign_route(row):
    status = row["status"]
    category = row["category"]

    # Unclassified tickets require human review
    if status not in ["classified_json", "classified_keyword"]:
        return pd.Series(["human_review", "NORMAL"])

    # Fraud has highest routing priority
    if category == "fraud":
        return pd.Series(["security_team", "HIGH"])

    # Premium customers get priority support
    if str(row["is_premium"]).strip().lower() in ["true", "1"]:
        return pd.Series(["premium_desk", "HIGH"])

    return pd.Series(["support_queue", "NORMAL"])


result[["route", "priority"]] = result.apply(
    assign_route,
    axis=1
)

# Blank unavailable classification and segment values
result["category"] = result["category"].fillna("")
result["segment_name"] = result["segment_name"].fillna("")

# Create the exact output columns required by the assignment
final_columns = [
    "ticket_id",
    "customer_id",
    "status",
    "category",
    "segment_name",
    "priority",
    "route",
    "tokens_used"
]

final_result = result[final_columns].copy()

# Validate output
assert len(final_result) == 164, "Expected exactly 164 tickets"
assert final_result.columns.tolist() == final_columns
assert final_result["ticket_id"].is_unique

# Save final output
output_path = BASE_DIR / "outputs/triage_results.csv"
final_result.to_csv(output_path, index=False)

print("FINAL TRIAGE COMPLETED")
print("Total tickets:", len(final_result))
print("Token budget:", TOKEN_BUDGET)
print("Total few-shot tokens used:", total_tokens)

print("\nStatus counts:")
print(final_result["status"].value_counts())

print("\nRoute counts:")
print(final_result["route"].value_counts())

print("\nPriority counts:")
print(final_result["priority"].value_counts())

print("\nFinal columns:")
print(final_result.columns.tolist())

print("\nFirst 5 rows:")
print(final_result.head().to_string(index=False))

print("\nSaved:", output_path)
