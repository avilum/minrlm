"""
RLM-Bench - a small, reproducible benchmark for Recursive Language Models.

12 tasks from the RLM paper (Zhang, Kraska & Khattab 2025) plus one
constraint-satisfaction puzzle, behind a one-method plugin interface.

Full spec, leaderboard, and submission instructions: eval/BENCHMARK.md

Quick start:
    rlm-bench --tasks all --runners minrlm-reasoning,vanilla --runs 50
    uv run python eval/quickstart.py
    uv run python eval/run.py --model gpt-5-mini --tasks official_sniah --runs 3
"""

__version__ = "0.1.5"
