# minRLM

**minRLM** is a token-efficient implementation of [Recursive Language Models](https://arxiv.org/abs/2512.24601). The data never enters the prompt. The cost stays flat regardless of context size. Every step is Python code you can read, rerun, and debug.

**3.6× fewer tokens** than the official RLM. **+30pp accuracy** over vanilla on GPT-5.2, winning **11 of 12 tasks**. On AIME 2025: **96% vs 0%** vanilla. On Sudoku Extreme: **80% vs 0%** vanilla - the REPL writes a constraint solver; vanilla can't solve Sudoku by token prediction alone.

**[Read the full blog post](https://avilum.github.io/minrlm/recursive-language-model.html)** — 13 tasks, 4 models, 6,600+ evaluations, all the details.

## How it works

```
┌──────────────────────────────────────────────────────────┐
│  Standard LLM                                            │
│                                                          │
│  [System prompt]                                         │
│  [500,000 tokens of raw context]   ← you pay for all of it
│  [Question]                                              │
│  → Answer (maybe right, maybe not)                      │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  Recursive LLM (minRLM)                                  │
│                                                          │
│  input_0 = "<500k chars stored in REPL>"  ← never in prompt
│  Task: "Count errors in last hour"                       │
├──────────────────────────────────────────────────────────┤
│  LLM writes:                                             │
│                                                          │
│  errors = re.findall(r'\[ERROR\].*', input_0)            │
│  cutoff = datetime.now() - timedelta(hours=1)            │
│  FINAL(len([e for e in errors if parse_time(e) > cutoff]))
│                                                          │
│  → Code runs, answer returned. ~4k tokens total.        │
└──────────────────────────────────────────────────────────┘
```

The model writes Python to query the data; attention runs only on the results. A 7M-character document becomes as cheap as a 7K one. Not ReAct — one REPL, 1–2 iterations, no growing context.

**What makes minRLM different from the [reference implementation](https://github.com/alexzhang13/rlm):**

- **Entropy profiling** — compression-based entropy map of the input via `zlib`. A needle in a 7MB haystack shows up as an entropy spike; the model skips straight to it.
- **Context preview** — head/mid/tail sample gives the model the data's structure without the full input.
- **Task routing** — auto-detects structured data, MCQ, code retrieval, math, search & extract. Each type has a specialized code pattern.
- **Two-pass search** — if the first pass returns "unknown", a second pass runs with new keywords from first-pass evidence.
- **Reasoning-first** — `# REASONING:` comment before code. `from minrlm import RLM` gives you this by default.
- **Sub-LLM delegation** — outer model gathers evidence via `search()`, passes it to `sub_llm(task, evidence)`. The sub-LLM reasons over a small, relevant context.
- **Flat token cost** — context never enters the conversation. Only the entropy map and preview do. 1–2 iterations, done.
- **DockerREPL** — every execution in a fresh container with seccomp. No network, no filesystem, stdlib only.

When a query fails, you see exactly which search missed, which filter was wrong, which assumption broke. Every step is Python — deterministic, testable, reproducible. That's something you can't do with a vanilla LLM call.

---

## Results

**Runners:** [minRLM](https://github.com/avilum/minrlm) · Vanilla (plain GPT-5-mini, no REPL) · [Official RLM](https://github.com/alexzhang13/rlm) (HEAD, March 2026)

| Runner | Accuracy | Avg Tokens | Avg Latency | Total Cost (50/task) |
|--------|----------|------------|-------------|----------------------|
| **minRLM** | **72.7%** | **8,151** | 25.8s | **$2.86** |
| Vanilla | 69.5% | 20,967 | 24.2s | $4.74 |
| Official RLM | 69.7% | 29,327 | 60.9s | $7.92 |

<sub>GPT-5-mini, 12 tasks, 1,800 evaluations (50 per task × 3 runners). Full per-task breakdown in [`eval/README.md`](eval/README.md). Sudoku Extreme evaluated separately on GPT-5.4-mini.</sub>

### Scaling trend

| Model | minRLM | Vanilla | Δ | Tasks won by minRLM |
|-------|--------|---------|---|---------------------|
| GPT-5-nano (small) | 53.7% | 63.2% | −9.5 | 4 of 12 |
| GPT-5-mini (mid) | 72.7% | 69.5% | +3.2 | 7 of 12 |
| GPT-5.4-mini (mid, newer) | 69.5% | 47.2% | +22.3 | 8 of 12 |
| GPT-5.2 (frontier) | **78.2%** | 48.2% | **+30.0** | **11 of 12** |

> **The REPL isn't a crutch for weak models — it's a lever that better models pull harder.**

- **GPT-5-nano:** Vanilla wins on accuracy, but minRLM still beats official by +10.4pp at 3.6× lower cost. RLM helps most on structured decomposition (BrowseComp, GDP Val, CodeQA).
- **GPT-5-mini:** minRLM wins overall — +3.2pp accuracy, 2.6× fewer tokens than vanilla, 3.6× fewer than official. $2.86 vs $7.92.
- **GPT-5.4-mini:** Largest minRLM advantage on a mini-class model - +22.3pp over vanilla, 8 of 12 tasks. Vanilla and official both regressed vs GPT-5-mini; the REPL-based approach held steady (72.7% -> 69.5%). AIME: 80% vs 0%. Sudoku Extreme: 80% vs 0% - vanilla can't solve Sudoku by token prediction; the REPL writes a backtracking solver.
- **GPT-5.2:** +30pp accuracy, 11 of 12 tasks. AIME 96% vs 0% - vanilla outputs 4 tokens (just a guess); the REPL forces the model to actually compute the answer via code. RepoQA remains the one consistent weak spot.

| | | |
|---|---|---|
| ![Summary](docs/summary_dashboard.png) | ![Accuracy](docs/accuracy_per_task.png) | ![Tokens](docs/token_savings.png) |
| ![Cost](docs/accuracy_vs_cost.png) | ![Latency](docs/accuracy_vs_latency.png) | ![Per Task](docs/cost_per_task.png) |

---

## When to use (and when not)

**Use minRLM when:**

- **Large context** (documents, logs, CSV, JSON) — search, aggregate, or extract without paying for the whole thing in the prompt. Cost stays roughly flat as context grows.
- **Token and cost efficiency at scale** — a 10MB document costs about the same as a 10KB one to process.
- **Code over data** (filter, count, search, parse) — the model writes Python in a stdlib-only sandbox.
- **Debuggability matters** — every step is readable Python, not hidden attention patterns.

**Skip it when:**

- **Short context (<8K tokens)** — if everything fits in one prompt, a direct call is simpler and about as good.
- **Pure reasoning with no data** (math, MCQ) on **small models** — the REPL can add overhead. On larger models it often helps (AIME: 96% vs 0%). When in doubt, try both.
- **Code retrieval (RepoQA)** — the one task family where vanilla wins across all model sizes.
- **Third-party packages needed** — the sandbox is stdlib-only (no `numpy`, `pandas`, `requests`).

---

## Why this matters

[Context window rot](https://arxiv.org/abs/2509.21361) is real — model accuracy degrades as input grows, even when the answer is right there. Bigger windows aren't the fix. **Less input, better targeted** is.

The same pattern is showing up in production: Anthropic's [web search tool](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/web-search-tool) writes code to filter results, [MCP](https://modelcontextprotocol.io/) standardizes code execution access, [smolagents](https://huggingface.co/docs/smolagents/en/index) goes further. They all converge on the same idea: **let the model use code to work with data instead of attending to all of it**.

---

## Quick start

```bash
pip install minrlm   # or: uv add minrlm
export OPENAI_API_KEY="sk-..."
```

### CLI (one-liners with <a href="https://docs.astral.sh/uv/getting-started/installation/">uv</a> python manager)

```bash
# Task + file as context (data never enters the prompt)
uvx minrlm "How many ERROR lines in the last hour?" ./server.log

# Just a task — no context
uvx minrlm "What is the sum of the first 100 primes?"

# Pipe context from stdin
cat huge_dataset.csv | uvx minrlm "Which product had the highest return rate?"

# Show generated code (-s) and token stats (-v)
uvx minrlm -sv "Return the sum of all primes up to 1,000,000."
# -> Sieve of Eratosthenes in 6,215 tokens, 1 iteration
# -> Answer: 37550402023

uvx minrlm -sv "Return all primes up to 1,000,000, reversed. Return a list of numbers."
# -> 999983, 999979, 999961, 999959, 999953, ...
# -> Tokens: 6,258 | Output: 616,964 chars (~154K tokens) | 25x savings
```

### Python

For fast experimentation, I recommend using <a href="https://docs.astral.sh/uv/getting-started/installation/">uv</a> python manager.

```
uv run --with minrlm python
```

```python
from minrlm import RLM

client = RLM(model="gpt-5-mini")

# Large context - data never enters the prompt
answer = client.completion(
    task="Which product had the highest return rate in Q3?",
    context=open("q3_returns.csv").read()  # could be 50MB
)

# No context - the REPL computes via code
result = client.completion(
    "Return all prime numbers up to 1,000,000, reversed. Return a list of numbers."
)
# Output: 999983, 999979, 999961, 999959, 999953, ...
# Tokens used: 6,258 | Output chars: 616,964 (~154K tokens) | Savings: 25x
```

### REPL tools

| Function | What it does |
|----------|--------------|
| `input_0` | Your context data (string) |
| `search(text, pattern)` | Substring search with context windows |
| `sub_llm(task, context)` | Recursive LLM call on a sub-chunk |
| `FINAL(answer)` | Return answer and stop |

### Any OpenAI-compatible endpoint

minRLM works with any provider that exposes an OpenAI-compatible API — just pass `base_url`, or inject your own `OpenAI` client:

```python
# Local / self-hosted
rlm = RLM(model="llama-3.1-70b", base_url="http://localhost:8000/v1")

# Hugging Face Inference API
from openai import OpenAI

hf_client = OpenAI(base_url="https://router.huggingface.co/v1", api_key="hf_...")
rlm = RLM(model="openai/gpt-oss-120b", client=hf_client)
result = rlm.completion("How many g's in 'huggingface'?")
```

Works with: OpenAI, Hugging Face, Anthropic (via proxy), vLLM, Ollama, LiteLLM, or anything OpenAI-compatible. See [`examples/huggingface_inference_endpoints.py`](examples/huggingface_inference_endpoints.py) for a full example.

### Visualizer

```bash
git clone https://github.com/avilum/minrlm && cd minrlm
uv sync --extra visualizer
uv run python examples/visualizer.py   # http://localhost:7860
```

### OpenCode

Use [OpenCode](https://opencode.ai) with minRLM by pointing it at the proxy:

**1. Start the proxy** (in one terminal):

```bash
uv run --with ".[proxy]" examples/proxy.py
# RLM Proxy initialized | model=gpt-5-mini | docker=False
# Uvicorn running on http://0.0.0.0:8000
```

**2. Config** (`opencode/opencode.json`): set `provider.minrlm.api` to `http://localhost:8000/v1` and add a model entry. See [opencode/opencode.json](opencode/opencode.json) in this repo.

**3. Run OpenCode** (in another terminal):

```bash
OPENCODE_CONFIG=opencode.json opencode run "Explain what is the first prime number after 1 million"
# > build · gpt-5-mini-rlm
# 1000003
# The first prime number after 1000000 is 1000003.
```

**[Full tutorial](docs/opencode-minrlm-tutorial.md)** — config details, example output, and troubleshooting.

---

## What's in this repo

| Component | Location | Description |
|-----------|----------|-------------|
| **Client** | [`minrlm/`](minrlm/) | `RLM` class — the LLM ↔ REPL loop |
| **DockerREPL** | [`minrlm/docker_repl.py`](minrlm/docker_repl.py) | Sandboxed execution via Docker + seccomp |
| **Evals** | [`eval/`](eval/) | 13-task benchmark framework, 4 model sizes |
| **Examples** | [`examples/`](examples/) | Quickstart, proxy server, Gradio UI |

### DockerREPL

LLM-generated code runs in isolated Docker containers. Docker is auto-detected. No network, read-only filesystem, memory-capped, seccomp-filtered.

```python
client = RLM(model="gpt-5-mini", use_docker=True, docker_memory="256m")
```

### Evals

```bash
git clone https://github.com/avilum/minrlm && cd minrlm
uv sync --extra eval

# Smoke test
uv run python eval/quickstart.py

# Full benchmark (reproduces the table above)
uv run python eval/run.py \
    --tasks all \
    --runners minrlm-reasoning,vanilla,official \
    --runs 50 --parallel 12 --task-parallel 12 \
    --output-dir logs/my_eval
```

Full results, per-task breakdowns, reproduction steps: [`eval/README.md`](eval/README.md)

### Examples

```bash
uv run python examples/minimal.py               # vanilla vs RLM side-by-side
uv run python examples/advanced_usage.py        # search, sub_llm, callbacks
uv run python examples/visualizer.py            # Gradio UI (uv sync --extra visualizer)
uv run uvicorn examples.proxy:app --port 8000   # OpenAI-compatible proxy (uv sync --extra proxy)
```

---

## Future work

- **More models** — Claude Opus 4.6, Gemini 2.5, open-weight models. Does the scaling trend hold across providers?
- **Agentic pipelines** — using the RLM pattern as a retrieval step inside multi-step agent workflows.
- **More tasks** - stress-testing edge cases and domains where the approach might break.

Contributions welcome. Open an issue or PR.

---

## Credits

Built by [Avi Lumelsky](https://github.com/avilum). Independent implementation — not a fork.
The RLM concept comes from [Zhang, Kraska, and Khattab (2025)](https://arxiv.org/abs/2512.24601). Official implementation: [github.com/alexzhang13/rlm](https://github.com/alexzhang13/rlm).

```
@misc{zhang2026recursivelanguagemodels,
      title={Recursive Language Models},
      author={Alex L. Zhang and Tim Kraska and Omar Khattab},
      year={2026},
      eprint={2512.24601},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2512.24601},
}
```

## Star History

<a href="https://www.star-history.com/?repos=avilum%2Fminrlm&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/image?repos=avilum/minrlm&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/image?repos=avilum/minrlm&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/image?repos=avilum/minrlm&type=date&legend=top-left" />
 </picture>
</a>

## License

MIT
