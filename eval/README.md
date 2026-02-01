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

## Key Findings

### Performance Summary

| Metric | minRLM vs Vanilla | minRLM vs Official |
|--------|-------------------|-------------------|
| **Token Efficiency** | **16x fewer** | **6x fewer** |
| **Cost** | **3x cheaper** | **2x cheaper** |
| **Latency** | ~1.4x slower | ~1.6x faster |

### Where minRLM Excels ✨

- **JSON Aggregation at 262K**: Succeeded where vanilla failed (50K+ tokens)
- **Long context retrieval**: Consistent performance from 8K to 256K
- **Multi-needle tasks**: Finds scattered needles at any scale
- **Token efficiency**: Uses <1K tokens regardless of context size

### Task-Specific Highlights

| Task | minRLM Tokens | Vanilla Tokens | Savings |
|------|---------------|----------------|---------|
| PAIRS (8 defs) | 1,265 | 6,793 | 5.4x |
| JSON_EXTRACTION_262K | 1,048 | 93,438 | **89x** |
| JSON_AGGREGATION_131K | 669 | 26,906 | **40x** |
| MULTI_NEEDLE_256K | 1,293 | 33,312 | **26x** |

### Limitations ⚠️

- **Simple tasks**: RLM overhead not worth it for straightforward queries
- **Latency**: Multiple API calls add wall-clock time (~10s avg vs ~7s for vanilla)
- **Requires code generation**: Model must generate valid Python

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
