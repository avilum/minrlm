# minrlm

Minimal implementation of [Recursive Language Models](https://arxiv.org/abs/2512.24601) in ~500 lines.

## What

RLM lets an LLM write and execute Python code in a loop until it solves the task. No fine-tuning, just inference.

```python
from minrlm import RLM

rlm = RLM(model="gpt-5-mini")
result = rlm.completion(
    task="Find the secret code hidden in this text",
    context=haystack_1m_chars
)
print(result.response)  # "SECRET-X7K2M9"
print(result.iterations)  # 1
print(result.total_tokens)  # 1,433 (vs 124,251 for vanilla!)
```

The LLM sees a persistent REPL with:
- `input_0` — the context data
- `search(text, pattern)` — find patterns in large text
- `peek(data)` — preview data structure
- `sub_llm(task, context)` — recursive calls
- `set_output(answer)` — return final answer

## Why

At large contexts, minrlm delivers **massive token and cost savings**:

| Context Size | Vanilla LLM | minRLM | Token Efficiency | Cost Savings |
|--------------|-------------|--------|------------------|--------------|
| 8K chars | 1,182 tokens | 1,498 tokens | 0.8x | - |
| 262K chars | 32,607 tokens | 1,342 tokens | **24x** | 10x cheaper |
| 1M chars | 124,251 tokens | 1,433 tokens | **87x** | 13x cheaper |

### vs Official RLM

minRLM is simpler and more token-efficient than the [official implementation](https://github.com/alexzhang13/rlm):

| Method | Accuracy | Avg Tokens | Token Efficiency |
|--------|----------|------------|------------------|
| Vanilla LLM | 100% | 10,839 | - |
| Official RLM | 94% | 5,275 | 2.1x |
| **minRLM** | **100%** | **1,646** | **6.6x** |

*Benchmarks on gpt-5-mini across 8K-262K contexts, 3 runs each.*

## Install

```bash
pip install minrlm

# Or with uv
uv add minrlm
```

From source:
```bash
git clone https://github.com/avilum/minrlm
cd minrlm && uv sync
```

## Usage

### Basic

```python
from minrlm import RLM

rlm = RLM(model="gpt-5-mini")
result = rlm.completion("What is 2+2?")
print(result.response)  # "4"
```

### With Context

```python
result = rlm.completion(
    task="Find all email addresses",
    context=large_document
)
```

### Custom Endpoint

```python
rlm = RLM(
    model="llama-3",
    base_url="http://localhost:8000/v1",
    api_key="..."
)
```

### Streaming Progress

```python
def on_step(event, data):
    print(f"{event}: iteration {data.get('iteration')}")

rlm = RLM(model="gpt-5-mini", on_step=on_step)
```

## Evaluation Suite

Run benchmarks comparing vanilla LLM, minRLM, and official RLM:

```bash
# Quick test
uv run python eval/run.py --model gpt-5-mini --tasks scaling --runs 1

# Full evaluation (8K to 256K contexts)
uv run python eval/run.py --model gpt-5-mini --tasks scaling --extended --runs 3

# JSON extraction/aggregation tasks
uv run python eval/run.py --model gpt-5-mini --tasks json_extraction,json_aggregation

# All tasks
uv run python eval/run.py --model gpt-5-mini --tasks sniah,multi_needle,pairs,scaling,qa_retrieval,json_extraction,json_aggregation
```

### Available Tasks

| Task | Description |
|------|-------------|
| `sniah` | Single needle-in-a-haystack retrieval |
| `multi_needle` | Find multiple needles scattered in text |
| `pairs` | Match definitions to concepts (hardest) |
| `scaling` | S-NIAH across context sizes (8K-256K) |
| `qa_retrieval` | Answer questions from scattered facts |
| `json_extraction` | Find specific values in large JSON |
| `json_aggregation` | Compute aggregates (count/sum) from JSON |
| `long_context` | Retrieval at 128K-256K contexts |
| `multi_needle_long` | 10 needles in 128K+ context |

Generates plots and cost analysis in `eval/results/`.

## Interactive Visualizer

```bash
uv run python examples/visualizer.py
```

Opens a Gradio UI to compare methods side-by-side with live progress.

## How It Works

1. LLM receives task + system prompt describing available functions
2. LLM outputs Python code in \`\`\`python blocks
3. Code executes in persistent namespace
4. If `set_output()` called → return answer
5. Otherwise → feed stdout back to LLM, repeat

The context is stored as `input_0` in the REPL, not in the conversation history. This avoids re-tokenizing large contexts on each turn.

## When to Use

| Scenario | Recommendation |
|----------|----------------|
| Small context (<10K chars) | Use vanilla LLM (faster) |
| Large context (>50K chars) | Use minRLM (cheaper) |
| Very large context (>200K chars) | Use minRLM (much cheaper) |
| Latency-critical | Use vanilla LLM |
| Cost-critical at scale | Use minRLM |

## Security

⚠️ Executes arbitrary Python. For untrusted inputs, run in a sandbox (Docker, gVisor, etc).

## References

- Zhang et al. [Recursive Language Models](https://arxiv.org/abs/2512.24601), 2025
- Official implementation: [github.com/alexzhang13/rlm](https://github.com/alexzhang13/rlm)

## License

MIT
