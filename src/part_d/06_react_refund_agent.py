import json
import time
import sys
import importlib.util
from pathlib import Path

# Project root directory
ROOT = Path(__file__).resolve().parents[2]

# Import mock LLM
sys.path.insert(0, str(ROOT))
import mock_llm

# Agent configuration
MAX_ITERATIONS = 5
MAX_TOKENS = 6000
MAX_TIME_SECONDS = 15

MAX_AUTO_REFUND_INR = 1000.0

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Retrieve order details using order ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"}
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "Search refund policy documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_refund",
            "description": "Calculate eligible refund amount.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_amount": {"type": "number"},
                    "delivery_minutes": {"type": "number"}
                },
                "required": ["order_amount", "delivery_minutes"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Issue an approved refund.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"}
                },
                "required": ["order_id", "refund_inr"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Escalate a refund for human approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"}
                },
                "required": ["order_id", "refund_inr"]
            }
        }
    }
]

# Load the existing refund tools module
tool_path = ROOT / "src" / "part_d" / "05_refund_tools.py"

spec = importlib.util.spec_from_file_location(
    "refund_tools_module", tool_path
)

refund_tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refund_tools)


def execute_tool(name, arguments):
    """Execute the requested refund tool."""

    if name == "get_order":
        return refund_tools.get_order(**arguments)

    elif name == "search_policy":
        return refund_tools.search_policy(**arguments)

    elif name == "calculate_refund":
        return refund_tools.calculate_refund(**arguments)

    elif name == "issue_refund":
        return refund_tools.issue_refund(**arguments)

    elif name == "escalate_to_human":
        return refund_tools.escalate_to_human(**arguments)

    return {"error": f"Unknown tool: {name}"}


def run_agent(
    user_message,
    max_iterations=None,
    token_limit=None,
    time_limit_seconds=None
):
    """Run the refund agent with three safety limits."""

    # Use defaults while allowing individual test configurations
    if max_iterations is None:
        max_iterations = MAX_ITERATIONS

    if token_limit is None:
        token_limit = MAX_TOKENS

    if time_limit_seconds is None:
        time_limit_seconds = MAX_TIME_SECONDS

    messages = [
        {
            "role": "system",
            "content": "You are an AGENT that handles refund requests."
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    tokens_used = 0
    steps = []
    start_time = time.monotonic()

    def fallback(status, summary, iterations):
        final_answer = f"Human handoff required: {summary}"

        return {
            "status": status,
            "final_answer": final_answer,
            "answer": final_answer,
            "steps": steps.copy(),
            "iterations": iterations,
            "tokens": tokens_used,
            "tokens_used": tokens_used
        }

    for iteration in range(max_iterations):

        # (c) Check time limit before every model call
        elapsed = time.monotonic() - start_time

        if elapsed >= time_limit_seconds:
            return fallback(
                "FALLBACK_TIMEOUT",
                "The agent exceeded its execution time limit.",
                iteration
            )

        # Make model call
        response = mock_llm.chat(messages, tools=TOOLS)

        # Count tokens immediately after the model call
        tokens_used += response.usage.total_tokens

        # (b) Stop before executing any requested tool
        if tokens_used > token_limit:
            return fallback(
                "FALLBACK_TOKEN_LIMIT",
                "The agent exceeded its token budget.",
                iteration + 1
            )

        message = response.choices[0].message

        # Check whether the model returned a final answer
        if not message.tool_calls:
            answer = message.content or ""
            status = answer.split(":")[0].strip()

            return {
                "status": status,
                "final_answer": answer,
                "answer": answer,
                "steps": steps.copy(),
                "iterations": iteration + 1,
                "tokens": tokens_used,
                "tokens_used": tokens_used
            }

        # Execute requested tools
        for tool_call in message.tool_calls:

            tool_name = tool_call.name

            arguments = (
                json.loads(tool_call.arguments)
                if isinstance(tool_call.arguments, str)
                else tool_call.arguments
            )

            result = execute_tool(tool_name, arguments)

            # Record completed steps
            steps.append({
                "tool": tool_name,
                "arguments": arguments,
                "result": result
            })

            # Store tool result as an observation
            messages.append({
                "role": "tool",
                "name": tool_name,
                "content": json.dumps(result, default=str)
            })

    # (a) Maximum model calls reached without a final answer
    return fallback(
        "FALLBACK_MAX_ITERATIONS",
        "The agent reached its maximum model-call limit.",
        max_iterations
    )


def load_scenarios():
    """Load refund test scenarios from CSV."""

    import csv

    scenario_path = ROOT / "data" / "raw" / "agent_scenarios.csv"

    with open(scenario_path, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def evaluate_agent():
    """Evaluate the agent using all seven scenarios."""

    scenarios = load_scenarios()
    results = []

    for scenario in scenarios:
        mock_llm.reset()

        result = run_agent(scenario["message"])

        expected_status = scenario["expected_outcome"]
        actual_status = result["status"]

        passed = expected_status == actual_status

        evaluation = {
            "scenario_id": scenario["scenario_id"],
            "expected": expected_status,
            "actual": actual_status,
            "passed": passed,
            "answer": result["answer"],
            "iterations": result["iterations"],
            "tokens": result["tokens"]
        }

        results.append(evaluation)

        print(
            f'{scenario["scenario_id"]}: '
            f'Expected={expected_status}, '
            f'Actual={actual_status}, '
            f'Passed={passed}'
        )

    passed_count = sum(r["passed"] for r in results)

    print("\n========== EVALUATION SUMMARY ==========")
    print(f"Total scenarios: {len(results)}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {len(results) - passed_count}")

    save_results_jsonl(results)

    return results


def save_results_jsonl(results):
    """Save evaluation results in JSONL format."""

    log_path = ROOT / "data" / "processed" / "agent_evaluation.jsonl"

    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"\nEvaluation log saved: {log_path}")


if __name__ == "__main__":
    print("Starting Instamart Refund Agent Evaluation...\n")

    results = evaluate_agent()