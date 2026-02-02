# minrlm

A minimal implementation of [Recursive Language Models](https://arxiv.org/abs/2512.24601) — let LLMs write code instead of reading your data.

## Why?

When you ask an LLM to find something in a large document, you pay for every token in that document. A 256K character JSON file costs ~93,000 tokens just to find one value.

```python
# You're paying for the model to "read" 93K tokens
response = client.chat.completions.create(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": f"Find employee EMP-0042 in: {huge_json}"}]
)
```

What if the model wrote code to search instead?

```python
from minrlm import RLM

rlm = RLM(model="gpt-5-mini")
result = rlm.completion(
    task="Find employee EMP-0042",
    context=huge_json  # stored as variable, never sent to the model
)
```

The model writes `[e for e in json.loads(input_0) if e["id"] == "EMP-0042"]` and you pay for ~1,000 tokens instead of 93,000. Same answer, ~90x cheaper.

## The Evolution: Why RLMs Are Inevitable

The path from GPT-3.5 to today tells a story of escalating costs and complexity:

**1. The "Think Step by Step" Era (GPT-3.5)**
Prompt engineering discovered that asking models to reason explicitly ("think step by step") dramatically improved accuracy. This worked, but it was manual and fragile.

**2. Reasoning Models (GPT-5, o1, o3)**
Models learned to reason internally. No more prompt engineering — the model just thinks. But here's the catch: **you pay for every token of that reasoning**. Queries became significantly more expensive as models generated thousands of hidden reasoning tokens before answering.

**3. Agents: Tools + Reasoning**
We connected reasoning models to tools (APIs, calculators, code execution). The model could now choose what to execute. Powerful, but context management exploded. Every tool call, every intermediate result, every error message — it all went back into the prompt. Conversations grew from 1K tokens to 100K tokens in a few turns.

**4. The Context Crisis**
Context windows grew (8K → 32K → 128K → 1M tokens), but so did costs. We built workarounds: summarize chat history, compress context, truncate old messages. These helped, but they're hacks around a fundamental problem: **we're asking the model to hold everything in its attention mechanism**.

**5. RLMs: The Inevitable Solution**
RLMs are agents with one tool: a Python REPL. The model writes code to interact with data, but **the data never enters the LLM's context**. 

Think about it: an agent with a code execution tool still sends your 200K document to the model so it can "decide what to do." An RLM doesn't — it just says "you have `input_0` with 200K chars, write code to search it." The model never sees the data. It only sees metadata ("200K chars") and writes code.

This isn't just cheaper — it's architecturally different. The data lives in the REPL, not in the conversation. Token usage stays flat regardless of context size. No summarization hacks. No context window limits. No paying for the model to "read" your data.

But there's another benefit: **the data transformations are now visible and reproducible**. Instead of mysterious attention "permutations" happening inside the model's memory, you get Python code that does the work. You can inspect it, debug it, reuse it, and even pre-process your context based on what the model learned. This code becomes a blueprint for how to handle your data efficiently.

**RLMs aren't just another tool in the agent toolkit. They're a fundamental shift: from "send data to the model" to "send code from the model."**

## How it works

1. Your context is stored as `input_0` in a Python REPL
2. The model writes code to search/process it
3. Code runs, output goes back to the model
4. Repeat until `FINAL(answer)` is called

```
┌─────────────────────────────────────────────────────────────┐
│  LLM sees:                                                  │
│                                                             │
│  input_0 = "string with 262144 chars"                       │
│  Task: Find employee EMP-0042                               │
├─────────────────────────────────────────────────────────────┤
│  LLM writes:                                                │
│                                                             │
│  data = json.loads(input_0)                                 │
│  for emp in data:                                           │
│      if emp["id"] == "EMP-0042":                            │
│          FINAL(emp["name"])                                 │
└─────────────────────────────────────────────────────────────┘
```

The data never enters the conversation. Token usage stays flat regardless of context size.

**Note on prompts**: The example above shows a simplified view. The actual system prompt is comprehensive (~150 lines) with detailed rules, examples for structured vs unstructured data, guidance on using `search()` vs regex, and best practices. The prompt can be optimized for your use case — see [`minrlm/prompts.py`](minrlm/prompts.py) for the full implementation. The prompt has been refined through benchmarking to improve accuracy on complex tasks.

## How is this different from an agent?

**RLMs are agents whose only tool is a Python REPL.**

Traditional agents have a menu of tools (search, calculator, APIs, code execution) and the model chooses which to use. But here's the problem: **the context still goes in the prompt** so the agent can reason about it. A ReAct agent with code execution still sends your 200K document to the model so it can "decide what to do."

RLM is simpler: the model just writes code. No tool selection step. No context in the prompt. The data lives in the REPL as `input_0`, and the model only sees metadata ("200K chars"). It writes code to search it, but **the model never sees the actual data**.

That's the architectural difference. It's not about having code execution as a capability — it's about **keeping the data out of the conversation entirely**. The model generates code, not answers. The REPL executes the code, not the model.

This is why token usage stays flat regardless of context size. You're not paying the model to "read" your data — you're paying it to write code that searches your data.

This pattern helps with several common headaches:

- **Large context management** — Data lives in the REPL, not the conversation. You can process 500K documents without hitting context limits or paying for 500K tokens per turn.

- **Prompt engineering fragility** — Different phrasings produce the same code. "Find engineers" and "who works in engineering?" both become `[e for e in data if e["dept"] == "Engineering"]`. The code layer normalizes your intent.

- **Multi-agent context bloat** — When agents pass context between each other, conversations grow fast. With RLM, the data stays in the REPL. `sub_llm()` calls can process chunks without copying everything into messages.

## Prompts become code

Different prompts often produce the same Python code.

Ask "find all engineers" or "who works in engineering?" or "list engineering team members" — the model writes roughly the same thing:

```python
[emp for emp in data if emp["dept"] == "Engineering"]
```

This is a form of **prompt normalization**. Instead of worrying whether you phrased your prompt perfectly, the model translates your intent into deterministic code. You can verify the code does what you meant, fix it if not, and trust the execution.

Prompt engineering becomes less fragile when there's a code layer between your words and the answer.

## Why code beats attention (on many tasks)

Traditional LLMs process your data through attention — expensive O(n²) operations where every token looks at every other token. Even though there are many forms of optimization for the Attention mechanism, for large documents, that's billions of floating-point operations to find one value.

But here's the deeper problem: **the data transformations happen inside the attention mechanism's memory**. These "permutations" (filtering, searching, aggregating) are opaque — you can't see what the model did, you can't reproduce it, and you can't understand why it found (or missed) something. It's a black box.

RLMs move these data transformations out of the attention mechanism and into CPU-bound Python code. The model writes `[emp for emp in data if emp["dept"] == "Engineering"]` — you can see exactly what it's doing. This code is:
- **Reproducible** — Run it again, get the same result
- **Debuggable** — Step through it, add print statements, inspect variables
- **Reusable** — Use the generated code to pre-process your context before sending it to other models
- **Faster** — O(n) string operations instead of O(n²) attention operations

Code execution is O(n) string operations. And you can read exactly what it did:

| Approach | How it finds data | Cost | Debuggable? | Reproducible? |
|----------|-------------------|------|-------------|---------------|
| Vanilla LLM | Attention over all tokens | O(n²) | No | No |
| minrlm | `json.loads()`, regex, loops | O(n) | Yes | Yes |

When the model writes `[emp for emp in data if emp["dept"] == "Engineering"]`, you can trust the output in a way you can't trust "I looked through the document and found..." More importantly, **you can use that code later** to pre-process your data, build pipelines, or understand what transformations the model actually applied.

## Benchmarks

**Note**: The original RLM paper authors report that RLMs *improve* accuracy on long-context tasks (e.g., CodeQA: 24% → 62%, OOLONG: RLM outperforms GPT-5 by 2x+). The results below are from **this minrlm implementation** tested on gpt-5-nano across 162 evaluations:

**Overall Performance:**
- **Accuracy**: 87.0% (vs 92.6% vanilla, 79.6% official)
- **Token Efficiency**: **4.20x fewer tokens** than vanilla (2,247 vs 9,441 average)
- **Token Efficiency**: **6.1x fewer tokens** than official RLM (2,247 vs 13,602 average)
- **Cost**: **4.1x cheaper** than vanilla ($0.001750 vs $0.007099)
- **Speed**: Faster on average (5.0s vs 15.3s vanilla, 53.2s official)
- **Average iterations**: 1.1 per task

**Where RLMs Excel:**
- **Large contexts (128K+)**: RLMs often outperform vanilla (JSON_AGGREGATION_131K: 100% vs 0%, OOLONG_128K: 100% vs 0%)
- **Extreme contexts (6M-11M)**: minRLM achieves 100% accuracy where vanilla fails (token limit exceeded)
- **JSON tasks**: 100% accuracy, massive savings (e.g., 131K JSON extraction: 1,993 vs 46,890 tokens = **23.5x**)
- **Token usage stays flat**: ~2K tokens regardless of context size (vs vanilla's linear growth)
- **Scaling tasks**: Consistent performance across all sizes (8K to 131K+), while vanilla token usage grows linearly
- **Multi-hop reasoning**: BrowseComp+ at 11M contexts — minRLM succeeds where vanilla cannot even attempt the task

**Where RLMs Trade Accuracy for Efficiency:**
- **Small contexts (<64K)**: Vanilla often has higher accuracy (better for simple tasks where token cost is negligible)
- **Some code understanding (CODEQA)**: Mixed results, depends on context size and task complexity

| Task | Context | Vanilla | minrlm | Savings | Notes |
|------|---------|---------|--------|---------|-------|
| JSON Extraction | 131K | 46,890 tokens | 1,993 tokens | **23.5x** | 100% accuracy both |
| JSON Aggregation | 131K | 38,746 tokens (0%) | 1,975 tokens (100%) | **19.6x** | RLM wins on accuracy |
| OOLONG | 128K | 37,873 tokens (0%) | 1,917 tokens (100%) | **19.7x** | RLM wins on accuracy |
| Multi-needle | 128K | 17,059 tokens | 1,848 tokens | **9.2x** | 100% accuracy both |
| Long context | 128K | 16,576 tokens | 1,827 tokens | **9.1x** | 100% accuracy both |
| BrowseComp+ | 11M | ❌ Fails | ✅ 100% (~2K tokens) | **∞** | Vanilla hits token limit |

**The takeaway (minrlm results)**: This implementation trades a small accuracy drop (5.6%) for massive token savings (4.20x) and cost reduction (4.1x). The approach shines on structured data and large contexts where token costs dominate. At extreme scales (6M-11M), RLMs are the only viable option. 

**Why the accuracy difference?** The original paper shows RLMs *improve* accuracy on long-context tasks (e.g., CodeQA: 24% → 62%). Our results differ likely due to:
- **Context sizes**: Paper tests up to 1M-10M tokens; our tests include up to 11M for BrowseComp+
- **Model choice**: Paper uses GPT-5-mini; we tested with GPT-5-nano (weaker reasoning)
- **Task focus**: Paper emphasizes complex aggregation tasks where RLMs excel; our mix includes simpler tasks

**Key insight**: At large contexts (128K+), RLMs often match or exceed vanilla accuracy while using 10-90x fewer tokens. At extreme scales (6M-11M), RLMs are the only viable option — vanilla fails due to token limits.

See [`eval/`](eval/) for the full benchmark suite and results.

## Install

```bash
pip install minrlm

# Or from source
git clone https://github.com/avilum/minrlm
cd minrlm && uv sync
```

## Quick start

```python
from minrlm import RLM

# Use gpt-4o-mini for simple tasks (fast, cheap)
rlm = RLM(model="gpt-4o-mini")

# Or gpt-5-nano for complex multi-step tasks (reasoning enabled)
# rlm = RLM(model="gpt-5-nano")

# No context — model just writes code
result = rlm.completion("Print the first 100 powers of two")
print(result.response)
# 1, 2, 4, 8, 16, ... 633825300114114700748351602688

# With context — data stored as input_0
result = rlm.completion(
    task="Find all employees in Engineering",
    context=massive_json  # 500K chars
)
print(result.total_tokens)  # Typically <2K tokens regardless of context size
```

### What the model generates

```python
# Iteration 1: writes this
for i in range(100):
    print(1 << i)
# → stdout captured, but no FINAL()

# Iteration 2: realizes it needs FINAL()
FINAL("\n".join(str(1 << i) for i in range(100)))
```

### Available functions

| Function | What it does |
|----------|--------------|
| `input_0` | Your context data |
| `search(text, pattern)` | Find matches with surrounding context |
| `peek(data)` | Preview structure of large data |
| `sub_llm(task, context)` | Recursive LLM call on a chunk |
| `FINAL(answer)` | Return the final answer |

**System prompt**: The system prompt guides the model on when to use each function, how to parse structured vs unstructured data, and best practices for code generation. The prompt is comprehensive (~150 lines) and has been optimized through benchmarking. You can customize it in [`minrlm/prompts.py`](minrlm/prompts.py) for your specific use case.

### Custom endpoints

```python
rlm = RLM(
    model="llama-3.1-70b",
    base_url="http://localhost:8000/v1",
    api_key="..."
)
```

## When to use this

| Context size | Recommendation |
|--------------|----------------|
| < 10K chars | Vanilla LLM (faster) |
| 10K - 50K chars | Either works |
| 50K+ chars | minrlm (**typically 20-90x cheaper**) |

**Good for:** Large documents, logs, JSON, anything you'd search/filter/aggregate.

**Skip if:** Context is small, latency matters more than cost, or the task needs holistic understanding of the whole document.

## Model selection

| Model | Best for | Tokens | Speed |
|-------|----------|--------|-------|
| **gpt-5-nano** | Complex tasks (multi-step logic) | More | Slower |
| **gpt-4o-mini** | Simple tasks (direct lookups) | Less | Faster |

Reasoning models (gpt-5, o1, o3) have internal chain-of-thought that helps with complex code generation. For complex multi-step parsing tasks, reasoning models typically achieve higher accuracy than non-reasoning models.

By default, minrlm uses `reasoning_effort="low"` for reasoning models to control token cost. You can tune this:

```python
# Fast and cheap for simple tasks
rlm = RLM(model="gpt-4o-mini")

# Accurate for complex tasks (reasoning with low token overhead)
rlm = RLM(model="gpt-5-nano", reasoning_effort="low")

# Maximum reasoning (expensive but most capable)
rlm = RLM(model="gpt-5-nano", reasoning_effort="high")
```

## Running evals

```bash
# Quick test
uv run python eval/run.py --model gpt-5-mini --tasks scaling --runs 1

# Full suite (8K to 256K contexts)
uv run python eval/run.py --model gpt-5-mini --extended
```

Results go to `eval/results/`. Partial results are saved if you ctrl+C.

## Interactive UI

```bash
uv sync --extra visualizer
uv run python examples/visualizer.py
```

Opens a Gradio interface showing token usage and model traces.

## Security

⚠️ This runs LLM-generated Python code. For untrusted inputs, use Docker:

```python
from minrlm import RLM, check_docker_available

# Check if Docker is available
if check_docker_available():
    rlm = RLM(
        model="gpt-5-mini",
        use_docker=True,
        docker_memory="256m",
        docker_timeout=60
    )
```

Docker mode sandboxes execution: no network, memory limits, read-only filesystem. Note: `sub_llm()` isn't available in Docker mode.

## Credits

Based on the RLM paper by Zhang, Kraska, and Khattab:

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

Paper: [arxiv.org/abs/2512.24601](https://arxiv.org/abs/2512.24601)  
Official implementation: [github.com/alexzhang13/rlm](https://github.com/alexzhang13/rlm)

## License

MIT

## Personal note 
> I am a security researcher. This is far from secure - but this is fucking cool!
> I recommend using it with docker backends (default if Docker is installed) because it uses custom seccomp policies that block network and most processing syscalls. Keep the LLM's REPL seperated and confined.
> Bonus: You can use gVisor as docker runtime.