
import pandas as pd
import sqlite3
from pathlib import Path

# Locate project folders
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "clean"
DB_PATH = BASE_DIR / "data" / "instamart.db"

# Load cleaned CSV files
customers = pd.read_csv(DATA_DIR / "customers.csv")
products = pd.read_csv(DATA_DIR / "products.csv")
orders = pd.read_csv(DATA_DIR / "orders.csv")

print("CLEANED DATA LOADED SUCCESSFULLY")

print("\nCustomers:", customers.shape)
print("Products:", products.shape)
print("Orders:", orders.shape)



# Part A: Pandas revenue cross-check

# Keep only completed orders
completed_orders = orders[
    orders["status"] == "completed"
].copy()

# Join orders with customers
orders_customers = completed_orders.merge(
    customers,
    on="customer_id",
    how="inner"
)

# Join with products
full_data = orders_customers.merge(
    products,
    on="product_id",
    how="inner"
)

# Calculate revenue by city and category
revenue_pivot = pd.pivot_table(
    full_data,
    values="order_amount",
    index="city",
    columns="category",
    aggfunc="sum",
    fill_value=0,
    margins=True,
    margins_name="Grand Total"
)

print("\nREVENUE BY CITY AND PRODUCT CATEGORY")
print(revenue_pivot.round(2))

# Overall completed revenue
print(
    "\nPandas Total Revenue:",
    round(completed_orders["order_amount"].sum(), 2)
)


# Part A: GroupBy analysis by city

city_analysis = completed_orders.merge(
    customers[["customer_id", "city"]],
    on="customer_id",
    how="inner"
)

city_summary = city_analysis.groupby("city").agg(
    total_revenue=("order_amount", "sum"),
    average_order_amount=("order_amount", "mean"),
    completed_order_count=("order_amount", "count")
)

print("\nCITY-WISE GROUPBY ANALYSIS")
print(city_summary.round(2))



# Part A: Statistical analysis of completed orders

amounts = completed_orders["order_amount"]

Q1 = amounts.quantile(0.25)
Q3 = amounts.quantile(0.75)
IQR = Q3 - Q1

upper_fence = Q3 + (1.5 * IQR)
lower_fence = Q1 - (1.5 * IQR)

outliers = amounts[
    (amounts < lower_fence) | (amounts > upper_fence)
]

print("\nCOMPLETED ORDER AMOUNT STATISTICS")
print("Mean:", round(amounts.mean(), 2))
print("Median:", round(amounts.median(), 2))
print("Skewness:", round(amounts.skew(), 2))
print("Q1:", round(Q1, 2))
print("Q3:", round(Q3, 2))
print("IQR:", round(IQR, 2))
print("Lower fence:", round(lower_fence, 2))
print("Upper fence:", round(upper_fence, 2))
print("Outliers below lower fence:", (amounts < lower_fence).sum())
print("Outliers above upper fence:", (amounts > upper_fence).sum())
print("Number of outliers:", len(outliers))



# Part A: Correlation analysis

correlation_data = completed_orders[
    [
        "order_amount",
        "discount_pct",
        "quantity",
        "delivery_minutes"
    ]
]

correlation_matrix = correlation_data.corr()

print("\nCORRELATION MATRIX")
print(correlation_matrix.round(3))




# Find strongest correlation excluding self-correlations
correlation_values = correlation_matrix.abs().copy()

for column in correlation_values.columns:
    correlation_values.loc[column, column] = 0

strongest_pair = correlation_values.stack().idxmax()

print("\nStrongest correlation pair:", strongest_pair)
print(
    "Correlation value:",
    round(
        correlation_matrix.loc[strongest_pair[0], strongest_pair[1]],
        3
    )
)