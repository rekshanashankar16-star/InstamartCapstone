
"""
05_refund_tools.py

Task 5: Instamart Refund Agent Tools

Five tools:
1. get_order
2. search_policy
3. calculate_refund
4. issue_refund
5. escalate_to_human

Uses:
- Existing SQLite database
- Existing Policy RAG implementation
- Separate CSV files for refund and escalation records

Does not modify mock_llm.py or completed Part D files.
"""

import sys
import csv
import json
import sqlite3
import importlib.util

from pathlib import Path
from datetime import datetime


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(BASE_DIR))

DB_PATH = BASE_DIR / "data" / "instamart.db"

OUTPUT_DIR = BASE_DIR / "outputs"

REFUND_LEDGER = OUTPUT_DIR / "refund_ledger.csv"

ESCALATION_LEDGER = OUTPUT_DIR / "refund_escalations.csv"

MAX_AUTO_REFUND_INR = 1000.0


# ============================================================
# 2. LOAD EXISTING POLICY RAG
# ============================================================

POLICY_FILE = BASE_DIR / "src" / "part_d" / "03_policy_rag.py"

spec = importlib.util.spec_from_file_location(
    "policy_rag",
    POLICY_FILE
)

policy_module = importlib.util.module_from_spec(spec)

spec.loader.exec_module(policy_module)

policy_answer = policy_module.answer


# ============================================================
# 3. HELPER: CREATE CSV FILE IF REQUIRED
# ============================================================

def initialize_csv(file_path, fieldnames):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not file_path.exists():

        with open(
            file_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()


# ============================================================
# HELPER: CHECK WHETHER ORDER EXISTS IN A CSV LEDGER
# ============================================================

def ledger_contains_order(file_path, order_id):

    """Check whether an order ID already exists in a CSV ledger."""

    if not file_path.exists():
        return False

    with open(
        file_path,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            if row.get("order_id") == order_id:
                return True

    return False

# ============================================================
# TOOL 1: GET ORDER
# ============================================================

def get_order(order_id: str):

    """
    Retrieve order details from SQLite.
    """

    connection = sqlite3.connect(DB_PATH)

    connection.row_factory = sqlite3.Row

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                order_id,
                customer_id,
                product_id,
                quantity,
                order_ts,
                status,
                discount_pct,
                order_amount,
                delivery_minutes

            FROM orders

            WHERE order_id = ?
            """,
            (order_id,)
        )

        row = cursor.fetchone()

        if row is None:

            return {
                "error": "NOT_FOUND",
                "order_id": order_id
            }

        order = dict(row)

        return order

    finally:

        connection.close()


# ============================================================
# TOOL 2: SEARCH POLICY
# ============================================================

def search_policy(query: str):

    """
    Search existing refund policies through Policy RAG.
    """

    result = policy_answer(query)

    return {
        "query": query,
        "answer": result["answer"],
        "sources": result["sources"]
    }


# ============================================================
# TOOL 3: CALCULATE REFUND
# ============================================================

def calculate_refund(
    order_amount: float,
    delivery_minutes: float
):

    """
    Calculate refund according to delivery-time policy.

    More than 45 minutes:
        Full refund.

    45 minutes or less:
        No late-delivery refund.
    """

    order_amount = float(order_amount)

    delivery_minutes = float(delivery_minutes)

    if order_amount < 0 or delivery_minutes < 0:

        return {
            "refund_inr": 0.0,
            "status": "INVALID_INPUT",
            "reason": "Negative values are not accepted."
        }

    if delivery_minutes > 45:

        refund = round(order_amount, 2)

        if refund > 1000:

            return {
                "refund_inr": refund,
                "status": "HUMAN_REVIEW_REQUIRED",
                "reason": (
                    "Refund exceeds Rs.1000. "
                    "Human approval is required."
                )
            }

        return {
            "refund_inr": refund,
            "status": "ELIGIBLE",
            "reason": (
                "Delivery exceeded 45 minutes. "
                "Full refund is eligible."
            )
        }

    return {
        "refund_inr": 0.0,
        "status": "NOT_ELIGIBLE",
        "reason": (
            "Delivery was within 45 minutes. "
            "No late-delivery refund is applicable."
        )
    }


# ============================================================
# TOOL 4: ISSUE REFUND
# ============================================================

def issue_refund(order_id: str, refund_inr: float):

    """Issue and record an eligible refund."""

    refund_inr = round(float(refund_inr), 2)

    if refund_inr <= 0:
        return {
            "status": "REJECTED",
            "reason": "Refund amount must be positive."
        }

    # Prevent duplicate refunds
    if ledger_contains_order(REFUND_LEDGER, order_id):
        return {
            "status": "ALREADY_REFUNDED",
            "order_id": order_id
        }

    # Prevent automatic refund during human review
    if ledger_contains_order(ESCALATION_LEDGER, order_id):
        return {
            "status": "ALREADY_ESCALATED",
            "order_id": order_id
        }

    order = get_order(order_id)

    if "error" in order:
        return {
            "status": "REJECTED",
            "reason": "Order not found."
        }

    if str(order["status"]).lower() != "completed":
        return {
            "status": "REJECTED",
            "reason": "Only completed orders qualify."
        }

    if order["delivery_minutes"] is None:
        return {
            "status": "REJECTED",
            "reason": "Delivery time is unavailable."
        }

    if float(order["delivery_minutes"]) <= 45:
        return {
            "status": "REJECTED",
            "reason": "Delivery did not exceed 45 minutes."
        }

    expected_refund = round(float(order["order_amount"]), 2)

    if refund_inr != expected_refund:
        return {
            "status": "REJECTED",
            "reason": "Refund does not match the order amount."
        }

    if refund_inr > MAX_AUTO_REFUND_INR:
        return {
            "status": "HUMAN_REVIEW_REQUIRED",
            "reason": "Refund exceeds automatic refund limit."
        }

    fields = [
        "order_id",
        "refund_inr",
        "status",
        "timestamp"
    ]

    initialize_csv(REFUND_LEDGER, fields)

    record = {
        "order_id": order_id,
        "refund_inr": refund_inr,
        "status": "REFUND_ISSUED",
        "timestamp": datetime.now().isoformat()
    }

    with open(
        REFUND_LEDGER,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writerow(record)

    return {
        "status": "REFUND_ISSUED",
        "order_id": order_id,
        "refund_inr": refund_inr
    }
# ============================================================
# TOOL 5: ESCALATE TO HUMAN
# ============================================================

def escalate_to_human(order_id: str, refund_inr: float):

    """Record a high-value refund request for human approval."""

    refund_inr = round(float(refund_inr), 2)

    if refund_inr <= MAX_AUTO_REFUND_INR:
        return {
            "status": "REJECTED",
            "reason": "Escalation is reserved for refunds above the automatic limit."
        }

    order = get_order(order_id)

    if "error" in order:
        return {
            "status": "REJECTED",
            "reason": "Order not found."
        }

    if str(order["status"]).lower() != "completed":
        return {
            "status": "REJECTED",
            "reason": "Only completed orders qualify."
        }

    if order["delivery_minutes"] is None:
        return {
            "status": "REJECTED",
            "reason": "Delivery time is unavailable."
        }

    if float(order["delivery_minutes"]) <= 45:
        return {
            "status": "REJECTED",
            "reason": "Delivery did not exceed 45 minutes."
        }

    expected_refund = round(float(order["order_amount"]), 2)

    if refund_inr != expected_refund:
        return {
            "status": "REJECTED",
            "reason": "Refund does not match the order amount."
        }

    # Prevent escalation after refund
    if ledger_contains_order(REFUND_LEDGER, order_id):
        return {
            "status": "ALREADY_REFUNDED",
            "order_id": order_id
        }

    # Prevent duplicate escalation
    if ledger_contains_order(ESCALATION_LEDGER, order_id):
        return {
            "status": "ALREADY_ESCALATED",
            "order_id": order_id
        }

    fields = [
        "order_id",
        "refund_inr",
        "status",
        "timestamp"
    ]

    initialize_csv(ESCALATION_LEDGER, fields)

    record = {
        "order_id": order_id,
        "refund_inr": refund_inr,
        "status": "PENDING_HUMAN_APPROVAL",
        "timestamp": datetime.now().isoformat()
    }

    with open(
        ESCALATION_LEDGER,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writerow(record)

    return {
        "status": "ESCALATED",
        "order_id": order_id,
        "refund_inr": refund_inr,
        "message": "Sent for human approval."
    }

# ============================================================
# 4. TOOL SPECIFICATIONS FOR THE LLM
# ============================================================

TOOLS_SPEC = [

    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Retrieve order details using order ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string"
                    }
                },
                "required": ["order_id"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "Search refund policy using the policy RAG system.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string"
                    }
                },
                "required": ["query"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "calculate_refund",
            "description": "Calculate refund based on order amount and delivery time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_amount": {
                        "type": "number"
                    },
                    "delivery_minutes": {
                        "type": "number"
                    }
                },
                "required": [
                    "order_amount",
                    "delivery_minutes"
                ]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Issue and record an eligible refund.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string"
                    },
                    "refund_inr": {
                        "type": "number"
                    }
                },
                "required": [
                    "order_id",
                    "refund_inr"
                ]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Escalate high-value refunds for human approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string"
                    },
                    "refund_inr": {
                        "type": "number"
                    }
                },
                "required": [
                    "order_id",
                    "refund_inr"
                ]
            }
        }
    }

]


# ============================================================
# 5. TOOL MAP
# ============================================================

TOOL_MAP = {

    "get_order": get_order,

    "search_policy": search_policy,

    "calculate_refund": calculate_refund,

    "issue_refund": issue_refund,

    "escalate_to_human": escalate_to_human

}


# ============================================================
# 6. DIRECT EXECUTION TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("INSTAMART REFUND TOOLS - TASK 5")
    print("=" * 60)

    print("\nAVAILABLE TOOLS:")

    for tool_name in TOOL_MAP:

        print("-", tool_name)

    print("\nDATABASE:", DB_PATH)

    print("REFUND LEDGER:", REFUND_LEDGER)

    print("ESCALATION LEDGER:", ESCALATION_LEDGER)

    print("\nTOOL SPECIFICATION COUNT:", len(TOOLS_SPEC))

    print("\nTASK 5 MODULE LOADED SUCCESSFULLY.")
