# minrlm

Minimal implementation of [Recursive Language Models](https://arxiv.org/abs/2512.24601) in ~400 lines.

## What

RLM lets an LLM write and execute Python code in a loop until it solves the task. No fine-tuning, just inference.

```python
from minrlm import RLM

rlm = RLM(model="gpt-4o")
result = rlm.completion(
    task="Find the secret code hidden in this text",
    context=haystack_50k_chars
)
print(result.response)  # "SECRET-X7K2M9"
```

The LLM sees a persistent REPL with:
- `input_0` — the context data
- `sub_llm(task, context)` — recursive calls
- `set_output(answer)` — return final answer

## Why

On 50K character contexts, minrlm uses **2-3x fewer tokens** than vanilla prompting:

| Method | SNIAH | Multi-Needle | Tokens |
|--------|-------|--------------|--------|
| Vanilla LLM | 100% | 100% | 6,500 |
| minrlm | 100% | 100% | 2,800 |

The LLM doesn't re-read the full context each turn—it writes code to extract what it needs.

## Install

```bash
pip install minrlm

# with UV
uv run --with minrlm python -c "from minrlm import RLM; print(RLM)"
```

Or from source:
```bash
git clone https://github.com/avilum/minrlm
cd minrlm && uv sync
```

## Usage

```python
from minrlm import RLM

# Basic
rlm = RLM(model="gpt-4o")
result = rlm.completion("What is 2+2?")

# With context
result = rlm.completion(
    task="Find all email addresses",
    context=document
)

# Custom endpoint
rlm = RLM(
    model="llama-3",
    base_url="http://localhost:8000/v1",
    api_key="..."
)
```

## How it works

1. LLM receives task + system prompt describing available functions
2. LLM outputs Python code in ```python blocks
3. Code executes in persistent namespace
4. If `set_output()` called → return answer
5. Otherwise → feed stdout back to LLM, repeat

The context is stored as `input_0` in the REPL, not in the conversation history. This avoids re-tokenizing large contexts on each turn.

## Security

⚠️ Executes arbitrary Python. For untrusted inputs, run in a sandbox (Docker, gVisor, etc).

## References

- Zhang et al. [Recursive Language Models](https://arxiv.org/abs/2512.24601), 2025
- Official implementation: [github.com/alexzhang13/rlm](https://github.com/alexzhang13/rlm)

## License

MIT
