<p align="center">
  <h1 align="center">minRLM</h1>
  <p align="center">
    <b>Stop forcing LLMs to answer in one pass. Give them a runtime.</b>
  </p>
  <p align="center">
    <a href="https://pypi.org/project/minrlm/"><img src="https://img.shields.io/pypi/v/minrlm?color=blue" alt="PyPI"></a>
    <a href="https://github.com/avilum/minrlm/stargazers"><img src="https://img.shields.io/github/stars/avilum/minrlm?style=social" alt="Stars"></a>
    <a href="https://github.com/avilum/minrlm/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License"></a>
    <a href="https://avilum.github.io/minrlm/recursive-language-model.html"><img src="https://img.shields.io/badge/blog-post-orange" alt="Blog Post"></a>
  </p>
</p>

<p align="center">
  <img src="docs/minrlm-demo.gif" alt="minRLM demo - LLM writes code, REPL executes, answer returned" width="700">
</p>

Took a base model. Wrapped it in a tiny recursive loop: **generate code - execute - refine - repeat**.

Didn't change the model. Didn't add training. Didn't add data.

Just stopped forcing it to answer in one pass.

The performance jump is not subtle:

| | Vanilla (one-shot) | minRLM (recursive) |
|---|---|---|
| **AIME 2025** | 0% | **96%** |
| **Sudoku Extreme** | 0% | **80%** |
| **Overall (GPT-5.2)** | 48.2% | **78.2%** (+30pp) |
| **Tokens used** | 20,967 | **8,151** (3.6x less) |
| **Cost** | $7.92 | **$2.86** (2.8x cheaper) |

<sub>6,600+ evaluations across 4 models and 13 tasks. <a href="https://avilum.github.io/minrlm/recursive-language-model.html">Full blog post</a> | <a href="eval/README.md">Detailed results</a></sub>

---

## Try it in 10 seconds

```bash
pip install minrlm
export OPENAI_API_KEY="sk-..."

# Analyze a file - data never enters the prompt
uvx minrlm "How many ERROR lines in the last hour?" ./server.log

# Pure computation - the REPL writes the algorithm
uvx minrlm "Return all primes up to 1,000,000, reversed."
# -> 78,498 primes in 6,258 tokens. Output: 616K chars. 25x savings.

# Pipe anything
cat huge_dataset.csv | uvx minrlm "Which product had the highest return rate?"
```

```python
from minrlm import RLM

rlm = RLM(model="gpt-5-mini")

# 50MB CSV? Same cost as 5KB. Data never enters the prompt.
answer = rlm.completion(
    task="Which product had the highest return rate in Q3?",
    context=open("q3_returns.csv").read()
)
```

---

## How it works

```
Standard LLM:
  [System prompt] + [500K tokens of raw context] + [Question]
  = Expensive. Slow. Accuracy degrades with length.

minRLM:
  input_0 = "<500K chars in REPL memory>"     # never in prompt
  LLM writes: errors = [l for l in input_0.splitlines() if "ERROR" in l]
              FINAL(len(errors))
  = Code runs. Answer returned. ~4K tokens total.
```

The model writes Python to query the data. Attention runs only on the results. A 7M-character document costs the same as a 7K one.

**Not ReAct.** One REPL, 1-2 iterations, no growing context. Every step is Python you can read, rerun, and debug.

### What makes it work

- **Entropy profiling** - zlib compression heatmap of the input. A needle in 7MB shows up as an entropy spike; the model skips straight to it
- **Task routing** - auto-detects structured data, MCQ, code retrieval, math, search & extract. Each gets a specialized code pattern
- **Two-pass search** - if the first pass returns "unknown", a second pass runs with keywords from first-pass evidence
- **Sub-LLM delegation** - outer model gathers evidence via `search()`, passes it to `sub_llm(task, evidence)` for focused reasoning
- **Flat token cost** - context never enters the conversation. Only the entropy map and a head/mid/tail preview do
- **DockerREPL** - every execution in a sandboxed container with seccomp. No network, no filesystem, stdlib only

---

## The scaling story

The REPL isn't a crutch for weak models - it's a lever that better models pull harder.

| Model | minRLM | Vanilla | Gap | Tasks won |
|-------|--------|---------|-----|-----------|
| GPT-5-nano (small) | 53.7% | 63.2% | -9.5 | 4/12 |
| GPT-5-mini (mid) | 72.7% | 69.5% | +3.2 | 7/12 |
| GPT-5.4-mini (mid, newer) | 69.5% | 47.2% | +22.3 | 8/12 |
| GPT-5.2 (frontier) | **78.2%** | 48.2% | **+30.0** | **11/12** |

Small model? Recursion adds overhead. Frontier model? Recursion dominates.

The gap isn't model size. It's the execution model.

| | | |
|---|---|---|
| ![Summary](docs/summary_dashboard.png) | ![Accuracy](docs/accuracy_per_task.png) | ![Tokens](docs/token_savings.png) |
| ![Cost](docs/accuracy_vs_cost.png) | ![Latency](docs/accuracy_vs_latency.png) | ![Per Task](docs/cost_per_task.png) |

---

## When to use it (and when not to)

**Use it when:**
- Large context (docs, logs, CSV, JSON) - cost stays flat as data grows
- You want debuggable reasoning - every step is readable Python, not hidden attention
- Token efficiency matters - 3.6x fewer tokens than comparable approaches

**Skip it when:**
- Short context (<8K tokens) - a direct call is simpler
- Code retrieval (RepoQA) - the one task where vanilla wins everywhere
- You need third-party packages - the sandbox is stdlib-only

---

## REPL tools

| Function | What it does |
|----------|--------------|
| `input_0` | Your context data (string, never in the prompt) |
| `search(text, pattern)` | Substring search with context windows |
| `sub_llm(task, context)` | Recursive LLM call on a sub-chunk |
| `FINAL(answer)` | Return answer and stop |

---

## Works with any OpenAI-compatible endpoint

```python
# Local / self-hosted
rlm = RLM(model="llama-3.1-70b", base_url="http://localhost:8000/v1")

# Hugging Face
from openai import OpenAI
hf = OpenAI(base_url="https://router.huggingface.co/v1", api_key="hf_...")
rlm = RLM(model="openai/gpt-oss-120b", client=hf)
```

Works with: OpenAI, Hugging Face, Anthropic (via proxy), vLLM, Ollama, LiteLLM, or anything OpenAI-compatible.

---

## More ways to run

<details>
<summary><b>Visualizer (Gradio UI)</b></summary>

```bash
git clone https://github.com/avilum/minrlm && cd minrlm
uv sync --extra visualizer
uv run python examples/visualizer.py   # http://localhost:7860
```
</details>

<details>
<summary><b>OpenCode integration</b></summary>

**1. Start the proxy:**
```bash
uv run --with ".[proxy]" examples/proxy.py
# RLM Proxy initialized | model=gpt-5-mini | docker=False
# Uvicorn running on http://0.0.0.0:8000
```

**2. Config** (`opencode/opencode.json`): set `provider.minrlm.api` to `http://localhost:8000/v1`. See [opencode/opencode.json](opencode/opencode.json).

**3. Run:**
```bash
OPENCODE_CONFIG=opencode.json opencode run "First prime after 1 million"
# > 1000003
```

**[Full tutorial](docs/opencode-minrlm-tutorial.md)**
</details>

<details>
<summary><b>Docker sandbox</b></summary>

LLM-generated code runs in isolated Docker containers. No network, read-only filesystem, memory-capped, seccomp-filtered.

```python
rlm = RLM(model="gpt-5-mini", use_docker=True, docker_memory="256m")
```
</details>

<details>
<summary><b>Run the benchmarks yourself</b></summary>

```bash
git clone https://github.com/avilum/minrlm && cd minrlm
uv sync --extra eval

# Smoke test
uv run python eval/quickstart.py

# Full benchmark (reproduces the tables above)
uv run python eval/run.py \
    --tasks all \
    --runners minrlm-reasoning,vanilla,official \
    --runs 50 --parallel 12 --task-parallel 12 \
    --output-dir logs/my_eval
```

Full results: [`eval/README.md`](eval/README.md)
</details>

<details>
<summary><b>Examples</b></summary>

```bash
uv run python examples/minimal.py            # vanilla vs RLM side-by-side
uv run python examples/advanced_usage.py     # search, sub_llm, callbacks
uv run python examples/visualizer.py         # Gradio UI
uv run uvicorn examples.proxy:app --port 8000  # OpenAI-compatible proxy
```
</details>

---

## Why this matters

[Context window rot](https://arxiv.org/abs/2509.21361) is real - model accuracy degrades as input grows, even when the answer is right there. Bigger windows aren't the fix. Less input, better targeted, is.

The same pattern is showing up everywhere: Anthropic's [web search tool](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/web-search-tool) writes code to filter results, [MCP](https://modelcontextprotocol.io/) standardizes code execution access, [smolagents](https://huggingface.co/docs/smolagents/en/index) goes further. They all converge on the same idea: let the model use code to work with data instead of attending to all of it.

Feels less like "prompting" and more like giving the model a runtime.

---

## Future work

- **More models** - Claude Opus 4.6, Gemini 2.5, open-weight models. Does the scaling trend hold across providers?
- **Agentic pipelines** - using the RLM pattern as a retrieval step inside multi-step agent workflows
- **More tasks** - stress-testing edge cases and domains where the approach might break

Contributions welcome. Open an issue or PR.

---

## Credits

Built by [Avi Lumelsky](https://github.com/avilum). Independent implementation - not a fork.

The RLM concept comes from [Zhang, Kraska, and Khattab (2025)](https://arxiv.org/abs/2512.24601). Official implementation: [github.com/alexzhang13/rlm](https://github.com/alexzhang13/rlm).

<details>
<summary>Citation</summary>

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
</details>

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
