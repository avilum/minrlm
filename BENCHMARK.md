# RLM-Bench

A small, reproducible benchmark for Recursive Language Models (RLMs) and any other "LLM + runtime" system. 12 tasks from the [RLM paper](https://arxiv.org/abs/2512.24601) (+ one constraint-satisfaction puzzle), behind a single CLI and a one-method plugin interface.

The full spec, leaderboard, per-model results, datasets and submission instructions live in **[eval/BENCHMARK.md](eval/BENCHMARK.md)**.

Quick links:

- [Headline leaderboard](eval/BENCHMARK.md#headline-leaderboard)
- [Run it](eval/BENCHMARK.md#run-it)
- [Add your own RLM](eval/BENCHMARK.md#add-your-own-rlm)
- [Submitting results](eval/BENCHMARK.md#submitting-results)
- [Datasets](eval/BENCHMARK.md#datasets)

```bash
git clone https://github.com/avilum/minrlm && cd minrlm
uv sync --extra eval
export OPENAI_API_KEY="sk-..."

# After install, the CLI is also exposed as:
rlm-bench --tasks all --runners minrlm-reasoning,vanilla --runs 50

# Or directly via the repo:
uv run python eval/run.py \
    --tasks all \
    --runners minrlm-reasoning,vanilla,official \
    --runs 50 --parallel 12 --task-parallel 12 \
    --output-dir logs/my_eval
```
