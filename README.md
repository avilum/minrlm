# minrlm

A minimal, token-efficient implementation of [Recursive Language Models](https://arxiv.org/abs/2512.24601) in ~500 lines of Python.

## The Problem

When you ask an LLM to find something in a 262K character document:

```python
# Vanilla LLM - sends entire context to the model
response = openai.chat.completions.create(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": f"Find the secret code in: {document_262k}"}]
)
# → 93,438 tokens, $0.003, 14s
```

**93,000 tokens** just to find a single value buried in JSON.

## The Solution

RLM lets the LLM write code to search the context instead of reading it all:

```python
# minRLM - LLM writes code to search
from minrlm import RLM

rlm = RLM(model="gpt-5-mini")
result = rlm.completion(
    task="Find the secret code",
    context=document_262k  # stored as variable, not in prompt
)
# → 1,048 tokens, $0.00003, 13s
```

**89x fewer tokens.** Same answer. Same time. **100x cheaper.**

## How It Works

Instead of stuffing context into the prompt, minRLM:

1. Stores the context as `input_0` in a Python REPL
2. Asks the LLM to write code that searches/processes it
3. Executes the code and returns `stdout` to the LLM
4. Repeats until `set_output(answer)` is called

```
┌─────────────────────────────────────────────────────────────┐
│  LLM sees:                                                  │
│                                                             │
│  input_0 = "<262K chars of JSON>"                           │
│  search(text, pattern) → find patterns                      │
│  set_output(answer) → return final answer                   │
│                                                             │
│  Task: Find the employee with id EMP-0042-02317             │
├─────────────────────────────────────────────────────────────┤
│  LLM writes:                                                │
│                                                             │
│  ```python                                                  │
│  import json                                                │
│  data = json.loads(input_0)                                 │
│  for emp in data:                                           │
│      if emp["id"] == "EMP-0042-02317":                      │
│          set_output(emp["code"])                            │
│  ```                                                        │
└─────────────────────────────────────────────────────────────┘
```

The context never enters the conversation. Token usage stays constant regardless of context size.

## Benchmarks

Real results on gpt-5-mini across 46 evaluation tasks:

| Task | Context | Vanilla | minRLM | Savings |
|------|---------|---------|--------|---------|
| Needle in haystack | 50K | 6,365 tokens | 1,098 tokens | **6x** |
| Find in JSON | 262K | 93,438 tokens | 1,048 tokens | **89x** |
| Aggregate JSON | 131K | 26,906 tokens | 669 tokens | **40x** |
| Multi-needle | 256K | 33,312 tokens | 1,293 tokens | **26x** |

### vs Official RLM

Compared to the [official implementation](https://github.com/alexzhang13/rlm):

| Method | Avg Tokens | Avg Cost | Token Efficiency |
|--------|------------|----------|------------------|
| Vanilla LLM | 14,315 | $0.024 | - |
| Official RLM | 5,496 | $0.018 | 2.6x |
| **minRLM** | **893** | **$0.008** | **16x** |

See [`eval/`](eval/) for the full evaluation suite and reproducible benchmarks.

## Install

```bash
pip install minrlm

# Or from source
git clone https://github.com/avilum/minrlm
cd minrlm && uv sync
```

## Quick Start

```python
from minrlm import RLM

rlm = RLM(model="gpt-5-mini")

# Simple task (no context)
result = rlm.completion("What is 100 factorial mod 97?")
print(result.response)      # "61"
print(result.total_tokens)  # ~500
print(result.iterations)    # 1

# With large context
result = rlm.completion(
    task="Find all employees in Engineering department",
    context=massive_json_file  # 500K chars
)
print(result.response)      # "['Alice', 'Bob', 'Carol', ...]"
print(result.total_tokens)  # ~800 (not 150,000!)
```

### Available REPL Functions

The LLM has access to:

| Function | Description |
|----------|-------------|
| `input_0` | The context data (stored as variable) |
| `search(text, pattern)` | Find pattern matches with surrounding context |
| `peek(data)` | Preview first/last portions of large data |
| `sub_llm(task, context)` | Make recursive LLM calls |
| `set_output(answer)` | Return the final answer |

### Custom Endpoint

```python
rlm = RLM(
    model="llama-3.1-70b",
    base_url="http://localhost:8000/v1",
    api_key="..."
)
```

## Evaluation Suite

Run the full benchmark suite:

```bash
# Quick test
uv run python eval/run.py --model gpt-5-mini --tasks scaling --runs 1

# Full evaluation (8K to 256K contexts)  
uv run python eval/run.py --model gpt-5-mini --extended

# Specific tasks
uv run python eval/run.py --model gpt-5-mini --tasks json_extraction,json_aggregation
```

Generates plots and detailed results in `eval/results/`. See [`eval/README.md`](eval/README.md) for task descriptions and methodology.

## Interactive Visualizer

Compare methods side-by-side with live execution tracing:

```bash
uv run python examples/visualizer.py
```

Opens a Gradio UI showing token usage, iterations, and model trajectories in real-time.

## When to Use

| Context Size | Recommendation |
|--------------|----------------|
| < 10K chars | Vanilla LLM (lower latency) |
| 10K - 50K chars | Either (similar cost) |
| 50K - 200K chars | minRLM (**5-20x cheaper**) |
| > 200K chars | minRLM (**20-100x cheaper**) |

**Use minRLM when:**
- Processing large documents, logs, or JSON
- Cost matters more than latency
- Task involves searching/filtering/aggregating

**Use vanilla LLM when:**
- Context is small
- Latency is critical
- Task requires understanding the full context

## Security

⚠️ minRLM executes LLM-generated Python code. For untrusted inputs, run in a sandbox (Docker, gVisor, etc).

## References

This is a minimal reimplementation of Recursive Language Models. Please cite the original work:

```bibtex
@misc{zhang2025recursivelanguagemodels,
      title={Recursive Language Models}, 
      author={Alex L. Zhang and Tim Kraska and Omar Khattab},
      year={2025},
      eprint={2512.24601},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2512.24601}, 
}
```

- Paper: [arxiv.org/abs/2512.24601](https://arxiv.org/abs/2512.24601)
- Official implementation: [github.com/alexzhang13/rlm](https://github.com/alexzhang13/rlm)

## License

MIT
