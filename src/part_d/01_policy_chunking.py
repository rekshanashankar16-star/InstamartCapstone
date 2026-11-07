
from pathlib import Path
import pandas as pd

# Project folder
BASE_DIR = Path(__file__).resolve().parents[2]

# Policy document folder
POLICY_DIR = BASE_DIR / "data" / "policies"

# Output folder
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Chunk settings
CHUNK_SIZE = 40
OVERLAP = 10
STEP_SIZE = CHUNK_SIZE - OVERLAP

# Five policy documents
POLICY_FILES = [
    "refund_policy.txt",
    "delivery_policy.txt",
    "escalation_policy.txt",
    "payment_policy.txt",
    "privacy_policy.txt"
]

all_chunks = []

# Read and split each policy document
for filename in POLICY_FILES:

    file_path = POLICY_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Policy file not found: {file_path}")

    text = file_path.read_text(encoding="utf-8")

    words = text.split()

    print(f"\nProcessing: {filename}")
    print("Total words:", len(words))

    chunk_number = 1

    for start in range(0, len(words), STEP_SIZE):

        end = start + CHUNK_SIZE

        chunk_words = words[start:end]

        if not chunk_words:
            break

        chunk_text = " ".join(chunk_words)

        all_chunks.append({
            "chunk_id": f"{filename.replace('.txt', '')}_{chunk_number}",
            "source": filename,
            "chunk_number": chunk_number,
            "word_count": len(chunk_words),
            "text": chunk_text
        })

        chunk_number += 1

        if end >= len(words):
            break

# Convert chunks into a DataFrame
chunks_df = pd.DataFrame(all_chunks)

# Expected chunk counts from the assignment
expected_counts = {
    "refund_policy.txt": 4,
    "delivery_policy.txt": 3,
    "escalation_policy.txt": 3,
    "payment_policy.txt": 3,
    "privacy_policy.txt": 2
}

actual_counts = chunks_df["source"].value_counts().to_dict()

print("\nCHUNK COUNT VALIDATION")

for filename, expected in expected_counts.items():

    actual = actual_counts.get(filename, 0)

    print(f"{filename}: {actual} / {expected}")

    if actual != expected:
        raise ValueError(
            f"Expected {expected} chunks for {filename}, "
            f"but generated {actual}"
        )

# Save chunks
output_file = OUTPUT_DIR / "policy_chunks.csv"

chunks_df.to_csv(output_file, index=False)

print("\nPOLICY CHUNKING COMPLETED")
print("Total documents:", len(POLICY_FILES))
print("Total chunks:", len(chunks_df))
print("Chunk size:", CHUNK_SIZE)
print("Overlap:", OVERLAP)
print("Saved:", output_file)
