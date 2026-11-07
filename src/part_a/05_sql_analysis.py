
import sqlite3
from pathlib import Path

# Locate the database
BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "data" / "instamart.db"

# Connect to SQLite database
conn = sqlite3.connect(DB_PATH)

# SQL Query 1: Total revenue from completed orders
query1 = """
SELECT SUM(order_amount) AS total_revenue
FROM orders
WHERE status = 'completed';
"""

result = conn.execute(query1).fetchone()

print("Query 1: Total Revenue")
print("Total Revenue:", result[0])


# SQL Query 2: Total number of orders
query2 = """
SELECT COUNT(*) AS total_orders
FROM orders;
"""

result = conn.execute(query2).fetchone()

print("\nQuery 2: Total Number of Orders")
print("Total Orders:", result[0])


conn.close()



import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "data" / "instamart.db"

conn = sqlite3.connect(DB_PATH)

for table in ["customers", "products", "orders"]:
    print(f"\n{table.upper()} TABLE")
    result = conn.execute(f"PRAGMA table_info({table})").fetchall()

    for column in result:
        print(column[1])

conn.close()



# Execute all 7 official SQL queries using pandas

import pandas as pd

SQL_PATH = BASE_DIR / "sql" / "queries.sql"

conn = sqlite3.connect(DB_PATH)

sql_text = SQL_PATH.read_text(encoding="utf-8")
queries = [q.strip() for q in sql_text.split(";") if q.strip()]
print("Number of SQL queries found:", len(queries))

print("\n" + "=" * 50)
print("OFFICIAL SQL ANALYSIS - ALL 7 QUERIES")
print("=" * 50)

for i, query in enumerate(queries, start=1):
    df = pd.read_sql_query(query, conn)

    print(f"\nQUERY {i}")
    print("Rows returned:", len(df))
    print(df.head(5).to_string(index=False))

    if i == 2:
        print("Total revenue:", round(df["total_revenue"].sum(), 2))

    if i == 6:
        differences = (
            df["rank_position"] != df["row_number_position"]
        ).sum()
        print("RANK vs ROW_NUMBER differences:", differences)

    if i == 7:
        print(
            "Final cumulative revenue:",
            df.iloc[-1]["cumulative_revenue"]
        )

conn.close()
