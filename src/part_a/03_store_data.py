
import sqlite3
import pandas as pd
from pathlib import Path

# Find the project folder
BASE_DIR = Path(__file__).resolve().parents[2]

# Define file locations
CLEAN_DATA = BASE_DIR / "data" / "clean"
DB_PATH = BASE_DIR / "data" / "instamart.db"

# Read cleaned CSV files
customers = pd.read_csv(CLEAN_DATA / "customers.csv")
products = pd.read_csv(CLEAN_DATA / "products.csv")
orders = pd.read_csv(CLEAN_DATA / "orders.csv")

# Store order timestamps in the required text format
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"]
).dt.strftime("%Y-%m-%d %H:%M:%S")

# Create SQLite database and store tables
with sqlite3.connect(DB_PATH) as conn:
    customers.to_sql("customers", conn, if_exists="replace", index=False)
    products.to_sql("products", conn, if_exists="replace", index=False)
    orders.to_sql("orders", conn, if_exists="replace", index=False)

    print("SQLite tables created successfully!")

    for table in ["customers", "products", "orders"]:
        count = pd.read_sql_query(
            f"SELECT COUNT(*) AS total FROM {table}", conn
        )["total"][0]

        print(f"{table}: {count} rows")

print(f"\nDatabase saved at: {DB_PATH}")
