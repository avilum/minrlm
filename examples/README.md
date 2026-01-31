# Examples

## Quick Start

```bash
# Minimal example - compares vanilla LLM vs RLM
uv run python examples/minimal.py

# Advanced example
uv run python examples/advanced_usage.py

# With verbose output (shows code execution)
MINRLM_VERBOSE=1 uv run python examples/minimal.py

# Use a different model
MINRLM_MODEL=gpt-5-nano uv run python examples/minimal.py
```

## Files

| File | Description |
|------|-------------|
| `minimal.py` | Quick comparison: vanilla LLM vs RLM |
| `advanced_usage.py` | All parameters, callbacks, logging |
| `visualizer.py` | Gradio web UI (requires `minrlm[visualizer]`) |

## Visualizer

```bash
uv pip install minrlm[visualizer]
uv run python examples/visualizer.py
# Open http://localhost:7860
```
