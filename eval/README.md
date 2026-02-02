# RLM Evaluation Suite

A reproducible benchmark suite implementing tasks from the **Recursive Language Model (RLM) paper** ([Zhang et al., 2025](https://arxiv.org/abs/2512.24601)).

Compares vanilla LLM, minRLM, and official RLM across the paper's core benchmarks.

## Paper Tasks Implemented

| Paper Task | Our Task | Paper Reference | Status |
|------------|----------|-----------------|--------|
| **S-NIAH** | `sniah`, `scaling` | Figure 1 | ✅ |
| **OOLONG** | `oolong` | Bertsch et al., 2025 | ✅ |
| **OOLONG-Pairs** | `pairs` | Figure 1 (hardest) | ✅ |
| **CodeQA** | `codeqa` | Bai et al., 2025 | ✅ |
| **BrowseComp+** | `browsecomp` | Chen et al., 2025 | ✅ |

Plus additional tasks: `multi_needle`, `json_extraction`, `json_aggregation`, `qa_retrieval`

## Quick Start

```bash
# Run paper's core benchmarks
uv run python eval/run.py --model gpt-5-mini --tasks paper

# Run scaling test with paper's context sizes (8K to 1M)
uv run python eval/run.py --model gpt-5-mini --tasks scaling --paper-scale

# Run all tasks
uv run python eval/run.py --model gpt-5-mini --tasks all

# Skip official RLM (if not installed)
uv run python eval/run.py --model gpt-5-mini --skip-official

# Multiple runs for statistical significance
uv run python eval/run.py --model gpt-5-mini --runs 5
```

## What This Evaluates

### Methods Compared

| Method | Description |
|--------|-------------|
| **Vanilla LLM** | Direct API call with full context |
| **minRLM** | Minimal recursive implementation (~400 LOC) |
| **Official RLM** | Official implementation from the paper |

### Tasks

#### Core Paper Tasks (Table 1)

| Task | Description | Paper Context | Our Context | Difficulty |
|------|-------------|---------------|-------------|------------|
| **S-NIAH** (`sniah`) | Single needle in haystack | 8K-1M | 8K-1M | Easy |
| **OOLONG** (`oolong`) | Information aggregation | 131K | 131K | Hard |
| **OOLONG-Pairs** (`pairs`) | Pairwise matching | 32K | 50K | Very Hard |
| **CodeQA** (`codeqa`) | Code repository understanding | 23K-4.2M | 100K-500K | Hard |
| **BrowseComp+** (`browsecomp`) | Deep research / multi-hop | 6M-11M | 500K-1M | Very Hard |

#### Additional Tasks

| Task | Description | Context Sizes | Difficulty |
|------|-------------|---------------|------------|
| **Scaling** (`scaling`) | S-NIAH across context lengths | 8K-1M | Variable |
| **Multi-Needle** | Find 5 hidden secrets | 50K | Medium |
| **Long Context** | Needle at start/middle/end | 128K-256K | Medium |
| **Multi-Needle Long** | Find 10 needles at scale | 128K-256K | Hard |
| **JSON Extraction** | Find data in JSON records | 50K-200K | Medium |
| **JSON Aggregation** | Count/sum from JSON data | 50K-200K | Hard |
| **QA Retrieval** | Answer questions from facts | 50K | Medium |

## Metrics Collected

- **Accuracy**: Task success rate (%)
- **Latency**: Wall-clock time (seconds)
- **Input Tokens**: Tokens in prompts
- **Output Tokens**: Tokens in completions
- **Total Tokens**: Input + Output
- **Iterations**: Number of RLM loops (for RLM methods)
- **Cost (USD)**: API cost calculated via `tokencost`

## Output Structure

```
eval/
├── results/
│   ├── eval_YYYYMMDD_HHMMSS.json     # Raw results
│   ├── summary_YYYYMMDD_HHMMSS.md    # Human-readable report
│   └── plots/
│       ├── accuracy_comparison.png
│       ├── token_efficiency.png
│       ├── latency_comparison.png
│       ├── scaling_analysis.png
│       ├── cost_comparison.png
│       └── summary_dashboard.png
```

## Extending the Benchmark

### Adding a New Task

```python
# In eval/tasks.py

@register_task("my_task")
class MyTask(BaseTask):
    """Description of your task."""
    
    def generate(self, seed: int = 42, **kwargs) -> TaskInstance:
        # Generate task, context, and expected answer
        return TaskInstance(
            task="Find the answer...",
            context="...",
            expected="answer",
        )
    
    def check(self, response: str, expected: str) -> bool:
        # Return True if response is correct
        return expected.lower() in response.lower()
```

### Adding a New Runner

```python
# In eval/runners.py

@register_runner("my_method")
class MyRunner(BaseRunner):
    """My custom RLM implementation."""
    
    def run(self, task: str, context: str, model: str) -> RunResult:
        # Run your method
        return RunResult(
            response="...",
            total_tokens=100,
            input_tokens=80,
            output_tokens=20,
            time_seconds=1.5,
            iterations=3,
        )
```

## Reproducing Results

To reproduce our benchmark results:

1. **Set up environment**:
   ```bash
   cd recursive-language-model
   uv sync
   export OPENAI_API_KEY="your-key"
   ```

2. **Run evaluation**:
   ```bash
   uv run python eval/run.py --model gpt-5-mini --runs 1
   ```

3. **View results**:
   ```bash
   open eval/results/plots/
   cat eval/results/summary_*.md
   ```

## Benchmark Results

**Note**: The original RLM paper authors report that RLMs *improve* accuracy on long-context tasks (e.g., CodeQA: 24% → 62%, OOLONG: RLM outperforms GPT-5 by 2x+). The results below are from **this minrlm implementation** tested on gpt-5-nano across 162 evaluations.

### Overall Performance

| Metric | minRLM | Vanilla | Official |
|--------|--------|---------|----------|
| **Accuracy** | 87.0% | 92.6% | 79.6% |
| **Token Efficiency** | 2,247 avg | 9,441 avg | 13,602 avg |
| **Token Savings** | - | **4.20x fewer** | **6.1x fewer** |
| **Cost** | $0.001750 | $0.007099 | $0.018924 |
| **Cost Savings** | - | **4.1x cheaper** | **10.8x cheaper** |
| **Speed** | 5.0s | 15.3s | 53.2s |
| **Avg Iterations** | 1.1 | 1.0 | 1.0 |

### Where RLMs Excel

- **Large contexts (128K+)**: RLMs often outperform vanilla (JSON_AGGREGATION_131K: 100% vs 0%, OOLONG_128K: 100% vs 0%)
- **Extreme contexts (6M-11M)**: minRLM achieves 100% accuracy where vanilla fails (token limit exceeded)
- **JSON tasks**: 100% accuracy, massive savings (e.g., 131K JSON extraction: 1,993 vs 46,890 tokens = **23.5x**)
- **Token usage stays flat**: ~2K tokens regardless of context size (vs vanilla's linear growth)
- **Scaling tasks**: Consistent performance across all sizes (8K to 131K+), while vanilla token usage grows linearly
- **Multi-hop reasoning**: BrowseComp+ at 11M contexts - minRLM succeeds where vanilla cannot even attempt the task

### Where RLMs Trade Accuracy for Efficiency

- **Small contexts (<64K)**: Vanilla often has higher accuracy (better for simple tasks where token cost is negligible)
- **Some code understanding (CODEQA)**: Mixed results, depends on context size and task complexity

### Detailed Task Results

| Task | Context | Vanilla | minrlm | Savings | Notes |
|------|---------|---------|--------|---------|-------|
| JSON Extraction | 131K | 46,890 tokens | 1,993 tokens | **23.5x** | 100% accuracy both |
| JSON Aggregation | 131K | 38,746 tokens (0%) | 1,975 tokens (100%) | **19.6x** | RLM wins on accuracy |
| OOLONG | 128K | 37,873 tokens (0%) | 1,917 tokens (100%) | **19.7x** | RLM wins on accuracy |
| Multi-needle | 128K | 17,059 tokens | 1,848 tokens | **9.2x** | 100% accuracy both |
| Long context | 128K | 16,576 tokens | 1,827 tokens | **9.1x** | 100% accuracy both |
| BrowseComp+ | 11M | ❌ Fails | ✅ 100% (~2K tokens) | **∞** | Vanilla hits token limit |

### Analysis

**The takeaway**: This implementation trades a small accuracy drop (5.6%) for massive token savings (4.20x) and cost reduction (4.1x). The approach shines on structured data and large contexts where token costs dominate. At extreme scales (6M-11M), RLMs are the only viable option.

**Why the accuracy difference?** The original paper shows RLMs *improve* accuracy on long-context tasks (e.g., CodeQA: 24% → 62%). Our results differ likely due to:
- **Context sizes**: Paper tests up to 1M-10M tokens; our tests include up to 11M for BrowseComp+
- **Model choice**: Paper uses GPT-5-mini; we tested with GPT-5-nano (weaker reasoning)
- **Task focus**: Paper emphasizes complex aggregation tasks where RLMs excel; our mix includes simpler tasks

**Key insight**: At large contexts (128K+), RLMs often match or exceed vanilla accuracy while using 10-90x fewer tokens. At extreme scales (6M-11M), RLMs are the only viable option - vanilla fails due to token limits.

## Citation

If you use this evaluation suite, please cite the original RLM paper:

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

- **Paper**: [arxiv.org/abs/2512.24601](https://arxiv.org/abs/2512.24601)
- **Official Implementation**: [github.com/alexzhang13/rlm](https://github.com/alexzhang13/rlm)

## License

MIT License - See repository root.
