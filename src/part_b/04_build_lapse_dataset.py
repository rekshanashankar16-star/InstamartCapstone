import pandas as pd

# Load cleaned orders and customer data
orders = pd.read_csv("data/clean/orders.csv")
customers = pd.read_csv("data/clean/customers.csv")

# Convert order timestamp into datetime format
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# Define the lapse cutoff date
cutoff_date = pd.Timestamp("2025-09-01")

print("Orders shape:", orders.shape)
print("Customers shape:", customers.shape)
print("Cutoff date:", cutoff_date)

# Select completed orders before the cutoff date
historical_orders = orders[
    (orders["status"] == "completed") &
    (orders["order_ts"] < cutoff_date)
]

# Identify eligible customers
cohort_customers = historical_orders["customer_id"].unique()

print("\nHistorical completed orders:", len(historical_orders))
print("Eligible customers:", len(cohort_customers))

# Select completed orders during September 2025
september_orders = orders[
    (orders["status"] == "completed") &
    (orders["order_ts"] >= cutoff_date) &
    (orders["order_ts"] < pd.Timestamp("2025-10-01"))
]

# Identify customers who ordered during September
september_customers = september_orders["customer_id"].unique()

# Create the lapse dataset
lapse_data = pd.DataFrame({
    "customer_id": cohort_customers
})

# 1 = Lapsed, 0 = Not lapsed
lapse_data["lapsed"] = (
    ~lapse_data["customer_id"].isin(september_customers)
).astype(int)

print("\nLapse distribution:")
print(lapse_data["lapsed"].value_counts().sort_index())

print("\nTotal cohort:", len(lapse_data))
print("Lapsed customers:", lapse_data["lapsed"].sum())
print(
    "Lapse percentage:",
    round(lapse_data["lapsed"].mean() * 100, 2),
    "%"
)

print("\nOrders columns:")
print(orders.columns.tolist())

print("\nCustomers columns:")
print(customers.columns.tolist())

# Calculate recency using historical completed orders
rfm_features = historical_orders.groupby("customer_id").agg(
    last_order_date=("order_ts", "max")
).reset_index()

rfm_features["recency"] = (
    cutoff_date - rfm_features["last_order_date"]
).dt.days

# Display first five customers
print("\nCustomer Recency:")
print(rfm_features[["customer_id", "recency"]].head())

# Calculate frequency and monetary value
purchase_features = historical_orders.groupby("customer_id").agg(
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum")
).reset_index()

print("\nCustomer Purchase Features:")
print(purchase_features.head())

# Calculate average discount percentage
discount_features = historical_orders.groupby("customer_id").agg(
    avg_discount_pct=("discount_pct", "mean")
).reset_index()

print("\nAverage Discount Features:")
print(discount_features.head())

# Calculate cancellation rate using all historical orders
all_historical_orders = orders[
    orders["order_ts"] < cutoff_date
]

cancellation_features = all_historical_orders.groupby("customer_id").agg(
    total_orders=("order_id", "nunique"),
    cancelled_orders=("status", lambda x: (x == "cancelled").sum())
).reset_index()

cancellation_features["cancellation_rate"] = (
    cancellation_features["cancelled_orders"] /
    cancellation_features["total_orders"]
)

print("\nCancellation Rate Features:")
print(cancellation_features.head())

# Calculate average delivery time
delivery_features = historical_orders.groupby("customer_id").agg(
    avg_delivery_minutes=("delivery_minutes", "mean")
).reset_index()

print("\nAverage Delivery Time Features:")
print(delivery_features.head())

# Load cleaned product data
products = pd.read_csv("data/clean/products.csv")

print("\nProducts columns:")
print(products.columns.tolist())

# Merge historical completed orders with product categories
orders_with_category = historical_orders.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

# Count orders in each category for every customer
category_counts = (
    orders_with_category
    .groupby(["customer_id", "category"])["order_id"]
    .nunique()
    .reset_index(name="category_order_count")
)

# Find the most frequently purchased category
favorite_category = (
    category_counts
    .sort_values(
        ["customer_id", "category_order_count", "category"],
        ascending=[True, False, True]
    )
    .drop_duplicates("customer_id")
    [["customer_id", "category"]]
    .rename(columns={"category": "favorite_category"})
)

print("\nFavourite Product Categories:")
print(favorite_category.head())

# Select customer demographic information
customer_features = customers[
    ["customer_id", "city", "signup_date", "age", "gender"]
].copy()

# Convert signup date into datetime format
customer_features["signup_date"] = pd.to_datetime(
    customer_features["signup_date"]
)

# Calculate customer tenure in days
customer_features["tenure_days"] = (
    cutoff_date - customer_features["signup_date"]
).dt.days

print("\nCustomer Demographic Features:")
print(customer_features.head())

# Combine all customer features
lapse_dataset = lapse_data.merge(
    rfm_features[["customer_id", "recency"]],
    on="customer_id",
    how="left"
)

lapse_dataset = lapse_dataset.merge(
    purchase_features,
    on="customer_id",
    how="left"
)

lapse_dataset = lapse_dataset.merge(
    discount_features,
    on="customer_id",
    how="left"
)

lapse_dataset = lapse_dataset.merge(
    cancellation_features[["customer_id", "cancellation_rate"]],
    on="customer_id",
    how="left"
)

lapse_dataset = lapse_dataset.merge(
    delivery_features,
    on="customer_id",
    how="left"
)

lapse_dataset = lapse_dataset.merge(
    favorite_category,
    on="customer_id",
    how="left"
)

lapse_dataset = lapse_dataset.merge(
    customer_features.drop(columns=["signup_date"]),
    on="customer_id",
    how="left"
)

print("\nFinal Lapse Dataset:")
print(lapse_dataset.head())

print("\nDataset shape:", lapse_dataset.shape)
print("\nMissing values:")
print(lapse_dataset.isnull().sum())

# Validate the final dataset
print("\nTotal customers:", len(lapse_dataset))

print("\nDuplicate customer IDs:")
print(lapse_dataset["customer_id"].duplicated().sum())

print("\nMissing values in each column:")
print(lapse_dataset.isnull().sum())

print("\nLapse distribution:")
print(lapse_dataset["lapsed"].value_counts())

# Save the final lapse prediction dataset
lapse_dataset.to_csv(
    "outputs/lapse_dataset.csv",
    index=False
)

print("\nLapse dataset saved successfully!")