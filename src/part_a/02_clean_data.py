
from pathlib import Path
import pandas as pd

# Project folders
ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = ROOT / "data" / "raw"
CLEAN_DATA = ROOT / "data" / "clean"

CLEAN_DATA.mkdir(parents=True, exist_ok=True)

# Load raw customer data
customers = pd.read_csv(RAW_DATA / "customers.csv")

# 1. Standardize city names
customers["city"] = customers["city"].str.strip().str.title()

# 2. Standardize gender
gender_mapping = {
    "F": "female",
    "Female": "female",
    "M": "male",
    "Male": "male"
}

customers["gender"] = (
    customers["gender"]
    .str.strip()
    .str.title()
    .map(gender_mapping)
    .fillna("unknown")
)

# 3. Convert age to numeric
customers["age"] = pd.to_numeric(
    customers["age"], errors="coerce"
)

# 4. Invalid ages become missing
customers.loc[
    ~customers["age"].between(10, 100),
    "age"
] = float("nan")

# 5. Fill missing ages using city median
customers["age"] = customers["age"].fillna(
    customers.groupby("city")["age"].transform("median")
)

# 6. Convert signup date
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="coerce"
)

# 7. Save cleaned customer data
customers.to_csv(
    CLEAN_DATA / "customers.csv",
    index=False
)

# Display validation results
print("CUSTOMER CLEANING COMPLETED")
print("\nCity counts:")
print(customers["city"].value_counts().sort_index())

print("\nGender counts:")
print(customers["gender"].value_counts())

print("\nMissing ages:", customers["age"].isna().sum())
print("Minimum age:", customers["age"].min())
print("Maximum age:", customers["age"].max())
print("\nSaved: data/clean/customers.csv")



# --------------------------------------------------
# PART 2: CLEAN ORDERS
# --------------------------------------------------

# Load raw orders and products
orders = pd.read_csv(RAW_DATA / "orders.csv")
products = pd.read_csv(RAW_DATA / "products.csv")

products.to_csv(CLEAN_DATA / "products.csv", index=False)
print("Saved: data/clean/products.csv")

print("\nStarting orders cleaning...")

# A. Convert order_ts to datetime
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# B. Remove exact duplicate rows
before_duplicates = len(orders)

orders = orders.drop_duplicates()

removed_duplicates = before_duplicates - len(orders)

print("Exact duplicate rows removed:", removed_duplicates)

# C. Keep only the latest record for each order_id
before_latest = len(orders)

orders = orders.sort_values("order_ts")

orders = orders.drop_duplicates(
    subset="order_id",
    keep="last"
).copy()

removed_old_records = before_latest - len(orders)

print("Older order records removed:", removed_old_records)

# D. Convert order_amount to numeric
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"],
    errors="coerce"
)

# Convert quantity and discount to numeric
orders["quantity"] = pd.to_numeric(
    orders["quantity"],
    errors="coerce"
)

orders["discount_pct"] = pd.to_numeric(
    orders["discount_pct"],
    errors="coerce"
)

# Attach product price and category
orders = orders.merge(
    products[["product_id", "price_inr", "category"]],
    on="product_id",
    how="left",
    validate="many_to_one"
)

# Calculate missing order amounts using product price
missing_amounts = orders["order_amount"].isna().sum()

calculated_amounts = (
    orders["price_inr"]
    * orders["quantity"]
    * (1 - orders["discount_pct"] / 100)
).round(2)

orders["order_amount"] = orders["order_amount"].fillna(
    calculated_amounts
)

print("Missing amounts repaired:", missing_amounts)

# E. Clean delivery_minutes
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"],
    errors="coerce"
)

# Replace negative delivery times with missing values
negative_delivery = (
    orders["delivery_minutes"] < 0
).sum()

orders.loc[
    orders["delivery_minutes"] < 0,
    "delivery_minutes"
] = float("nan")

print("Negative delivery times corrected:", negative_delivery)

# Fill missing delivery times only for completed orders
completed_mask = orders["status"] == "completed"

category_median = (
    orders.loc[completed_mask]
    .groupby("category")["delivery_minutes"]
    .transform("median")
)

orders.loc[completed_mask, "delivery_minutes"] = (
    orders.loc[completed_mask, "delivery_minutes"]
    .fillna(category_median)
)

# Cancelled orders must retain missing delivery times

# Remove temporary product columns
orders = orders.drop(
    columns=["price_inr", "category"]
)

# Save cleaned orders
orders.to_csv(
    CLEAN_DATA / "orders.csv",
    index=False
)

# --------------------------------------------------
# ORDERS CLEANING VALIDATION
# --------------------------------------------------

print("\nORDERS CLEANING COMPLETED")

print("\nOrder status counts:")
print(orders["status"].value_counts())

print("\nTotal cleaned orders:", len(orders))

print(
    "Missing order amounts:",
    orders["order_amount"].isna().sum()
)

print(
    "Missing delivery times in completed orders:",
    orders.loc[
        orders["status"] == "completed",
        "delivery_minutes"
    ].isna().sum()
)

print(
    "Missing delivery times in cancelled orders:",
    orders.loc[
        orders["status"] == "cancelled",
        "delivery_minutes"
    ].isna().sum()
)

print(
    "Completed order revenue:",
    round(
        orders.loc[
            orders["status"] == "completed",
            "order_amount"
        ].sum(),
        2
    )
)
