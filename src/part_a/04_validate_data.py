
import sqlite3
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "data" / "instamart.db"

checks_passed = 0

def check(condition, message):
    global checks_passed
    assert condition, f"FAILED: {message}"
    checks_passed += 1
    print(f"PASSED: {message}")


with sqlite3.connect(DB_PATH) as conn:

    # Read all three database tables
    customers = pd.read_sql_query("SELECT * FROM customers", conn)
    products = pd.read_sql_query("SELECT * FROM products", conn)
    orders = pd.read_sql_query("SELECT * FROM orders", conn)

    # 1–3: Check row counts
    check(len(customers) == 1200, "Customer count is 1200")
    check(len(products) == 40, "Product count is 40")
    check(len(orders) == 12309, "Order count is 12309")

    # 4–6: Check customer ages
    check(customers["age"].isna().sum() == 0, "No missing customer ages")
    check(customers["age"].min() == 18, "Minimum customer age is 18")
    check(customers["age"].max() == 60, "Maximum customer age is 60")

    # 7: Check customer gender counts
    gender_counts = customers["gender"].value_counts().to_dict()

    check(
        gender_counts == {
            "female": 610,
            "male": 560,
            "unknown": 30
        },
        "Customer gender counts are correct"
    )

    # 8–9: Check order status counts
    status_counts = orders["status"].value_counts().to_dict()

    check(
        status_counts.get("completed", 0) == 11110,
        "Completed order count is 11110"
    )

    check(
        status_counts.get("cancelled", 0) == 1199,
        "Cancelled order count is 1199"
    )

    # 10: Check timestamp format
    timestamp_format = orders["order_ts"].astype(str).str.fullmatch(
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
    ).all()

    check(timestamp_format, "Order timestamps use YYYY-MM-DD HH:MM:SS")

print("\nALL CHECKPOINTS PASSED")
print(f"Total checkpoints: {checks_passed}")
