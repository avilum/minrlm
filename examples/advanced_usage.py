#!/usr/bin/env python3
"""
Advanced minrlm usage - all parameters and features.

Run: uv run python examples/advanced_usage.py
"""

from minrlm import RLM, RLMResult

# =============================================================================
# Configuration
# =============================================================================

# All RLM constructor parameters:
rlm = RLM(
    # Required
    model="gpt-5-nano",  # Any OpenAI-compatible model
    # Optional - API configuration
    api_key=None,  # Default: uses OPENAI_API_KEY env var
    base_url=None,  # Custom endpoint (e.g., "http://localhost:8000/v1")
    # Optional - Execution control
    max_iterations=20,  # Max code execution loops before giving up
    max_output_tokens=2000,  # Limit LLM response length (None = unlimited)
    # Optional - Logging
    log_dir=None,  # Directory for JSONL execution logs
    # Optional - Performance
    async_batch=True,  # Enable parallel sub_llm_batch() calls
    # Optional - Callbacks
    on_step=None,  # Callback for real-time execution events
)

# =============================================================================
# Basic Usage
# =============================================================================

print("\n" + "=" * 60)
print("1. Basic completion")
print("=" * 60)

result: RLMResult = rlm.completion("What is the 15th Fibonacci number?")

# RLMResult fields:
print(f"  response:      {result.response}")
print(f"  iterations:    {result.iterations}")
print(f"  total_tokens:  {result.total_tokens}")
print(f"  input_tokens:  {result.input_tokens}")
print(f"  output_tokens: {result.output_tokens}")
print(f"  history:       {len(result.history)} messages")

# =============================================================================
# With Context
# =============================================================================

print("\n" + "=" * 60)
print("2. Completion with context (input_0)")
print("=" * 60)

context = """
Sales Report Q4 2025:
- Product A: $1.2M revenue, 15% growth
- Product B: $800K revenue, -5% growth
- Product C: $2.1M revenue, 32% growth
Total: $4.1M
"""

result = rlm.completion(
    task="Which product had the highest growth rate?",
    context=context,  # Available as `input_0` in the REPL
)
print(f"  Answer: {result.response}")

# =============================================================================
# Real-time Callbacks
# =============================================================================

print("\n" + "=" * 60)
print("3. Real-time execution callbacks (on_step)")
print("=" * 60)


def my_callback(event: str, data: dict):
    """
    Events emitted by RLM:
    - "thinking"   : LLM is generating (iteration number)
    - "llm_response": LLM responded (response text, has_code)
    - "executing"  : About to run code (code string)
    - "executed"   : Code finished (stdout, output, error)
    """
    if event == "executing":
        print(f"    [exec] {data['code'][:50].replace(chr(10), ' ')}...")
    elif event == "executed" and data.get("output"):
        print(f"    [done] output = {data['output']}")
    else:
        print(f"    [event] {event}")
        print(f"    [data] {data}")


rlm_verbose = RLM(model="gpt-5-nano", on_step=my_callback)
result = rlm_verbose.completion("Calculate 2^20")
print(f"  Answer: {result.response}")

# =============================================================================
# Logging to File
# =============================================================================

print("\n" + "=" * 60)
print("4. Logging execution traces")
print("=" * 60)

rlm_logged = RLM(model="gpt-5-nano", log_dir="./logs")
result = rlm_logged.completion("What is 100 factorial mod 97?")
print(f"  Answer: {result.response}")
print("  Log saved to: ./logs/")

# =============================================================================
# Custom API Endpoint
# =============================================================================

print("\n" + "=" * 60)
print("5. Custom API endpoint (local/self-hosted)")
print("=" * 60)

print(
    """
  # For local models (e.g., ollama, vllm, text-generation-webui):

  rlm = RLM(
      model="llama-3-70b",
      base_url="http://localhost:8000/v1",
      api_key="not-needed",  # or your local key
  )
"""
)

# =============================================================================
# Sub-LLM Usage (in generated code)
# =============================================================================

print("=" * 60)
print("6. sub_llm() for recursive decomposition")
print("=" * 60)

print(
    """
  # The LLM can call sub_llm() in its generated code:

  # Single call:
  answer = sub_llm("Summarize this paragraph", context=paragraph)

  # Batch calls (parallel):
  answers = sub_llm_batch([
      ("Summarize section 1", section1),
      ("Summarize section 2", section2),
      ("Summarize section 3", section3),
  ])
"""
)

# =============================================================================
# Docker Sandboxing (requires Docker)
# =============================================================================

print("=" * 60)
print("7. Docker sandboxing for secure code execution")
print("=" * 60)

print(
    """
  # Run code in a Docker container with strict security:

  from minrlm import RLM, check_docker_available

  if check_docker_available():
      rlm = RLM(
          model="gpt-5-nano",
          use_docker=True,          # Enable Docker mode
          docker_image="python:3.14-slim",  # Custom image
          docker_memory="256m",     # Memory limit
          docker_timeout=60,        # Execution timeout
      )
      result = rlm.completion("Calculate 2^1000")

  # Security features:
  # - No network access (seccomp blocks all socket syscalls)
  # - Memory limited (default 256MB)
  # - CPU limited (default 1 core)
  # - Process limit (100 processes max)
  # - Read-only filesystem (except /tmp)
  # - Execution timeout

  # Note: sub_llm() IS supported in Docker mode via a retry protocol.
  # The container signals sub_llm requests to the host, which calls
  # the real LLM and re-runs the container with cached results.
"""
)

# Check if Docker is available
from minrlm import check_docker_available

if check_docker_available():
    print("  ✓ Docker is available on this system")
else:
    print("  ⚠ Docker is not available")

# =============================================================================
# Environment Variables
# =============================================================================

print("\n" + "=" * 60)
print("8. Environment variables")
print("=" * 60)

print(
    """
  OPENAI_API_KEY     - API key (required)
  OPENAI_BASE_URL    - Custom base URL

  # For examples:
  MINRLM_MODEL       - Model to use (default: gpt-5-nano)
  MINRLM_VERBOSE     - Set to "1" for verbose output
"""
)

print("\n✓ Advanced usage demo complete.\n")
