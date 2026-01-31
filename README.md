# Recursive Language Model (RLM) - Minimal Implementation

A minimal implementation of Recursive Language Models based on [arXiv:2512.24601](https://arxiv.org/abs/2512.24601).

## Concept

RLMs are an **inference-time paradigm** (no fine-tuning required!) where the LLM:
1. Outputs Python code that executes in a persistent REPL
2. Can make recursive `sub_llm()` calls to decompose complex problems
3. Uses `set_output()` to return the final answer

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Control Plane (RLM)                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │   OpenAI    │◄──►│   RLM       │◄──►│  Python REPL    │  │
│  │   Client    │    │   Engine    │    │  (subprocess)   │  │
│  └─────────────┘    └─────────────┘    └─────────────────┘  │
│                            │                    │           │
│                            │   sub_llm()        │           │
│                            └────────────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
uv sync
```

## Quick Start

```python
from absl import logging
logging.set_verbosity(logging.INFO)

from rlm import RLM

rlm = RLM(model="gpt-4o")
result = rlm.completion("Calculate the first 10 Fibonacci numbers.")
logging.info("Answer: %s", result.response)
```

## Usage

### Basic completion

```python
from absl import logging
logging.set_verbosity(logging.INFO)  # or DEBUG for more detail

from rlm import RLM

rlm = RLM(
    model="gpt-4o",           # Any OpenAI-compatible model
    use_subprocess=False,     # Use separate process for REPL
    max_iterations=20,        # Max code execution loops
)

result = rlm.completion(
    task="Your task here",
    context="Optional input data",
)

logging.info("Answer: %s", result.response)       # Final answer
logging.info("Iterations: %d", result.iterations) # Number of code executions
logging.info("Tokens: %d", result.total_tokens)   # Total tokens used
```

### With a custom API endpoint

```python
rlm = RLM(
    model="your-model",
    base_url="http://localhost:8000/v1",
    api_key="your-api-key",
)
```

### Subprocess isolation

For better isolation (recommended for untrusted inputs):

```python
rlm = RLM(use_subprocess=True)
```

## How It Works

1. User provides a task
2. LLM receives a system prompt instructing it to output Python code
3. Code is executed in a persistent REPL environment
4. REPL provides special functions:
   - `sub_llm(prompt)`: Make a recursive LLM call
   - `set_output(value)`: Set the final answer
5. Execution continues until `set_output()` is called or max iterations reached

## Examples

See `example.py` for detailed examples, or `example_minimal.py` for a minimal demo.

```bash
uv run python example_minimal.py
```

## Security Notes

⚠️ **This implementation executes arbitrary Python code!**

For production use:
- Use `use_subprocess=True` for process isolation
- Consider Docker/container sandboxing
- Implement resource limits (CPU, memory, time)
- Restrict available imports/modules
- Use cloud sandboxes (Modal, Prime Intellect) - see the [reference implementation](https://github.com/alexzhang13/rlm)

## References

- Paper: [Recursive Language Models (arXiv:2512.24601)](https://arxiv.org/abs/2512.24601)
- Reference Implementation: [github.com/alexzhang13/rlm](https://github.com/alexzhang13/rlm)

