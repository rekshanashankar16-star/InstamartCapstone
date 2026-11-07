
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
TICKETS_FILE = BASE_DIR / "data" / "raw" / "tickets.csv"

# Preserve empty strings while reading
tickets = pd.read_csv(TICKETS_FILE, keep_default_na=False)

print("SUPPORT TICKET AUDIT")
print("-" * 40)

print("\nTotal tickets:", len(tickets))
print("\nColumn names:")
print(tickets.columns.tolist())

print("\nFirst 5 tickets:")
print(tickets.head())

# Count empty tickets
empty_tickets = (
    tickets["text"].astype(str).str.strip().eq("")
).sum()

# Count long tickets (more than 1000 characters)
long_tickets = (
    tickets["text"].astype(str).str.len() > 1000
).sum()

print("\nEmpty tickets:", empty_tickets)
print("Long tickets:", long_tickets)

# Category distribution
print("\nTicket category distribution:")
print(tickets["true_category"].value_counts())

# Missing values
print("\nMissing values:")
print(tickets.isna().sum())

print("\nTICKET AUDIT COMPLETED")
