
from pathlib import Path
import pandas as pd

# Find the project root folder
ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = ROOT / "data" / "raw"

# Raw CSV files to inspect
files = [
    "customers.csv",
    "products.csv",
    "orders.csv",
    "tickets.csv",
    "policy_eval.csv",
    "agent_scenarios.csv"
]

for filename in files:
    file_path = RAW_DATA / filename

    # Preserve blank strings in tickets
    if filename == "tickets.csv":
        df = pd.read_csv(file_path, keep_default_na=False)
    else:
        df = pd.read_csv(file_path)

    print("\n" + "=" * 60)
    print(f"DATA AUDIT: {filename}")
    print("=" * 60)

    # Dataset dimensions
    print("\nShape (rows, columns):")
    print(df.shape)

    # Column data types
    print("\nData types:")
    print(df.dtypes)

    # Missing values
    print("\nMissing values:")
    missing = pd.DataFrame({
        "Missing_Count": df.isna().sum(),
        "Missing_Percentage": (df.isna().mean() * 100).round(2)
    })
    print(missing)

    # Exact duplicate rows
    print("\nExact duplicate rows:")
    print(df.duplicated().sum())

print("\nDATA AUDIT COMPLETED!")
