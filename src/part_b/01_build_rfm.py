import pandas as pd

# Load cleaned orders data
orders = pd.read_csv("data/clean/orders.csv")

# Keep only completed orders
completed_orders = orders[orders["status"] == "completed"]

# Set snapshot date
snapshot_date = pd.Timestamp("2025-10-01")

# Group completed orders by customer
rfm = completed_orders.groupby("customer_id").agg(
    recency=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum")
)

# Calculate recency in days
rfm["recency"] = (
    snapshot_date - pd.to_datetime(rfm["recency"])
).dt.days

# Reset customer_id from index into a column
rfm = rfm.reset_index()

print(rfm.head())
print("\nRFM shape:", rfm.shape)

# Save RFM table
rfm.to_csv("outputs/customer_rfm.csv", index=False)

# Display summary statistics
print("\nRFM Summary:")
print(rfm[["recency", "frequency", "monetary"]].describe())

# Check total customers
print("\nTotal customers:", len(rfm))

print("\nRFM table saved successfully!")