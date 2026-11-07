
import re
import pandas as pd
from pathlib import Path

# Locate project folder
BASE_DIR = Path(__file__).resolve().parents[2]

# File paths
RAW_FILE = BASE_DIR / "data" / "raw" / "tickets.csv"
OUTPUT_DIR = BASE_DIR / "data" / "clean"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load tickets
tickets = pd.read_csv(RAW_FILE, keep_default_na=False)

# Count replacements
counts = {
    "email": 0,
    "phone": 0,
    "order_id": 0
}


def mask_pii(text):
    """Mask email addresses, phone numbers and order IDs."""

    text = str(text)

    # 1. Mask email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    text, email_count = re.subn(
        email_pattern, "[EMAIL]", text
    )
    counts["email"] += email_count

    # 2. Mask phone numbers
    phone_pattern = r'(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)'
    text, phone_count = re.subn(
        phone_pattern, "[PHONE]", text
    )
    counts["phone"] += phone_count

    # 3. Mask order IDs
    order_pattern = r'\bORD-\d+\b'
    text, order_count = re.subn(
        order_pattern, "[ORDER_ID]", text, flags=re.IGNORECASE
    )
    counts["order_id"] += order_count

    return text


# Apply masking to ticket text
tickets["masked_text"] = tickets["text"].apply(mask_pii)

# Save cleaned ticket data
output_file = OUTPUT_DIR / "tickets_masked.csv"
tickets.to_csv(output_file, index=False)

# Print results
print("PII MASKING COMPLETED")
print("-" * 40)

print("Email replacements:", counts["email"])
print("Phone replacements:", counts["phone"])
print("Order ID replacements:", counts["order_id"])

total = sum(counts.values())
print("Total replacements:", total)

print("\nSaved:", output_file)
