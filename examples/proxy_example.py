"""
Example: Using the RLM Proxy with OpenAI-compatible clients.

This demonstrates how to use the RLM proxy as a drop-in replacement
for OpenAI API calls.
"""

import json

from openai import OpenAI

# Point client to the RLM proxy instead of OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",  # RLM proxy endpoint
    api_key="dummy-key",  # Not used by proxy, but required by OpenAI client
)

# Example 1: Simple task without context
print("Example 1: Simple task")
response = client.chat.completions.create(
    model="gpt-5-nano",
    messages=[{"role": "user", "content": "Print the first 10 Fibonacci numbers"}],
)
print(f"Response: {response.choices[0].message.content}")
print(f"Tokens used: {response.usage.total_tokens}\n")

# Example 2: Task with large context (RLM shines here)
print("Example 2: Task with large context")
large_json = json.dumps(
    {
        "employees": [
            {
                "id": f"EMP-{i:04d}",
                "name": f"Employee {i}",
                "department": "Engineering" if i % 2 == 0 else "Sales",
            }
            for i in range(1000)
        ]
    }
)

response = client.chat.completions.create(
    model="gpt-5-nano",
    messages=[
        {
            "role": "user",
            "content": f"Find employee EMP-0042. Here is the data:\n\n{large_json}",
        }
    ],
)
print(f"Response: {response.choices[0].message.content}")
print(f"Tokens used: {response.usage.total_tokens}\n")

# Example 3: Large context automatically detected
# The proxy automatically treats messages with >50K chars as context
print("Example 3: Large context (auto-detected)")
large_data = "Employee data:\n" + "\n".join([f"EMP-{i:04d}: Employee {i}" for i in range(2000)])

response = client.chat.completions.create(
    model="gpt-5-nano",
    messages=[
        {
            "role": "user",
            "content": f"Find employee EMP-0042\n\n{large_data}",  # Large content treated as context
        }
    ],
)
print(f"Response: {response.choices[0].message.content}")
print(f"Tokens used: {response.usage.total_tokens}")
print("\nNote: The proxy automatically extracts large content (>50K chars) as context,")
print("      routing it through RLM for efficient processing.")

# Example 4: Large JSON context with 2 user messages
# First message sets the task, second message contains large JSON (>50K chars)
print("\n" + "=" * 70)
print("Example 4: Large JSON context with 2 user messages")
print("=" * 70)

# Create a large JSON dataset (>50K characters)
# Generate employees with more detailed information to exceed 50K chars
large_employee_data = {
    "company": "TechCorp Inc.",
    "employees": [
        {
            "id": f"EMP-{i:04d}",
            "name": f"Employee {i}",
            "email": f"employee.{i}@techcorp.com",
            "department": ("Engineering" if i % 3 == 0 else "Sales" if i % 3 == 1 else "Marketing"),
            "salary": 50000 + (i * 100),
            "start_date": f"2020-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
            "skills": ["Python", "JavaScript", "SQL", "Docker", "Kubernetes"][: (i % 5) + 1],
            "projects": [
                {
                    "name": f"Project {j}",
                    "status": "active" if j % 2 == 0 else "completed",
                    "budget": 10000 * (j + 1),
                }
                for j in range((i % 3) + 1)
            ],
            "performance_reviews": [
                {
                    "year": 2020 + (k % 4),
                    "rating": 3 + (k % 2),
                    "comments": f"Excellent performance in Q{k + 1}",
                }
                for k in range((i % 2) + 1)
            ],
        }
        for i in range(2000)  # Generate 2000 employees with detailed data
    ],
    "metadata": {
        "total_employees": 2000,
        "departments": ["Engineering", "Sales", "Marketing"],
        "last_updated": "2025-02-02",
    },
}

large_json_str = json.dumps(large_employee_data, indent=2)
print(f"Generated JSON size: {len(large_json_str):,} characters")
print(f"Exceeds 50K threshold: {len(large_json_str) > 50000}")

# First message: Task instruction
# Second message: Large JSON data (will be automatically treated as context)
response = client.chat.completions.create(
    model="gpt-5-nano",
    messages=[
        {
            "role": "user",
            "content": "Find all employees in the Engineering department who have a salary above 75000 and list their names and email addresses.",
        },
        {
            "role": "user",
            "content": large_json_str,  # This will be automatically treated as context (>50K chars)
        },
    ],
)

print(f"\nResponse: {response.choices[0].message.content}")
print(f"Tokens used: {response.usage.total_tokens}")
print("\nNote: With 2 user messages, the first sets the task and the second")
print("      (if >50K chars) is automatically extracted as context data.")
print("      RLM processes this efficiently without sending all data to the LLM.")
