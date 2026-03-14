# Examples

## Quick Start

```bash
# Minimal example - compares vanilla LLM vs RLM
uv run python examples/minimal.py

# Advanced example - search, sub_llm, callbacks
uv run python examples/advanced_usage.py

# With verbose output (shows generated code and execution)
MINRLM_VERBOSE=1 uv run python examples/minimal.py

# Use a different model
MINRLM_MODEL=gpt-5-mini uv run python examples/minimal.py
```

## Files

| File | Description |
|------|-------------|
| `minimal.py` | Quick comparison: vanilla LLM vs RLM on the same task |
| `advanced_usage.py` | All parameters, callbacks, logging, multi-context usage |
| `huggingface_inference_endpoints.py` | Using minRLM with HuggingFace Inference API via injected OpenAI client |
| `proxy.py` | OpenAI-compatible proxy server — large contexts auto-route through RLM |
| `proxy_example.py` | Client-side usage of the proxy server |
| `visualizer.py` | Gradio web UI for side-by-side runner comparison (requires `minrlm[visualizer]`) |

## Custom Providers (HuggingFace, Ollama, etc.)

Any OpenAI-compatible endpoint works. Pass a pre-initialized `OpenAI` client:

```python
from openai import OpenAI
from minrlm import RLM

# HuggingFace Inference API
client = OpenAI(base_url="https://router.huggingface.co/v1", api_key="hf_...")
rlm = RLM(model="openai/gpt-oss-120b", client=client)

# Ollama
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
rlm = RLM(model="llama3.1:70b", client=client)

# Any OpenAI-compatible proxy
client = OpenAI(base_url="https://my-proxy.com/v1", api_key="sk-...")
rlm = RLM(model="gpt-5-mini", client=client)
```

See [`huggingface_inference_endpoints.py`](huggingface_inference_endpoints.py) for a full working example.

## Proxy Server

Drop-in replacement for the OpenAI API. Large contexts (>50K chars) are automatically routed through RLM; short contexts pass through directly.

```bash
uv sync --extra proxy
uv run uvicorn examples.proxy:app --host 0.0.0.0 --port 8000
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": "Analyze this CSV..."}],
)
```

## Visualizer

Interactive Gradio UI for comparing runners on evaluation tasks or custom prompts.

```bash
uv sync --extra visualizer
uv run python examples/visualizer.py
# Open http://localhost:7860
```
