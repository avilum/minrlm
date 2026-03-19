# Tutorial: Using OpenCode with minRLM

This tutorial shows how to run [OpenCode](https://opencode.ai) with minRLM so that builds and prompts use the RLM proxy: the model writes Python code to compute answers (e.g. search, aggregate, or reason over data) instead of sending large context into the prompt. Token usage stays flat regardless of context size.

## What you need

- [uv](https://docs.astral.sh/uv/getting-started/installation/) (or pip) and Python 3.11+
- [OpenCode](https://opencode.ai) installed (e.g. `opencode` CLI)
- An OpenAI-compatible API key (e.g. `OPENAI_API_KEY`)

## Step 1: Start the minRLM proxy

From the minrlm repo root, install the proxy extra and run the FastAPI server:

```bash
uv run --with ".[proxy]" examples/proxy.py
```

Or with uvicorn explicitly:

```bash
uv sync --extra proxy
uv run uvicorn examples.proxy:app --host 0.0.0.0 --port 8000
```

You should see:

```
Built minrlm @ file:///...
Installed 28 packages in 98ms
INFO:     Started server process [...]
INFO:     Waiting for application startup.
RLM Proxy initialized | model=gpt-5-mini | docker=False
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

The proxy is OpenAI-compatible. Large contexts are routed through minRLM; short prompts pass through to the underlying model.

## Step 2: OpenCode config

Create a config file (e.g. `opencode/opencode.json`) so OpenCode uses the minRLM proxy and exposes a model like `gpt-5-mini-rlm`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "minrlm": {
      "name": "minRLM",
      "api": "http://localhost:8000/v1",
      "npm": "@ai-sdk/openai-compatible",
      "env": ["OPENAI_API_KEY"],
      "models": {
        "gpt-5-mini-rlm": {
          "id": "gpt-5-mini-rlm",
          "name": "GPT-5 Mini (RLM)",
          "attachment": false,
          "reasoning": false,
          "temperature": true,
          "tool_call": true,
          "release_date": "2025-01-01",
          "limit": { "context": 128000, "output": 16384 },
          "cost": { "input": 0.4, "output": 1.6 },
          "options": {}
        }
      }
    }
  }
}
```

- `api`: proxy URL (no trailing slash; OpenCode will call `/v1/chat/completions` etc.).
- `env`: list of env var names; OpenCode will send `OPENAI_API_KEY` to the proxy (the proxy forwards it to the real API).

## Step 3: Run OpenCode with the RLM model

From a directory that contains your `opencode.json` (or set `OPENCODE_CONFIG` to its path):

```bash
OPENCODE_CONFIG=opencode.json opencode run "Explain what is the first prime number after 1 million"
```

Or from the `opencode` folder:

```bash
cd opencode
opencode run "Explain what is the first prime number after 1 million"
```

OpenCode will use the configured provider and model (`gpt-5-mini-rlm`), which points at your local proxy. The proxy runs the request through minRLM.

## Example output

**Terminal (OpenCode):**

```
Performing one time database migration, may take a few minutes...
Database migration complete.

> build · gpt-5-mini-rlm

1000003
The first prime number after 1000000 is 1000003. This was found by checking successive integers greater than 1000000 and testing primality via trial division up to the square root; 1000003 has no divisors other than 1 and itself.
```

**Terminal (proxy):**

```
INFO:     127.0.0.1:50584 - "POST /v1/chat/completions HTTP/1.1" 200 OK
INFO:     127.0.0.1:50586 - "POST /v1/chat/completions HTTP/1.1" 200 OK
```

The model used the RLM loop: it wrote and ran Python (e.g. trial division or a small sieve) to compute the first prime after 1,000,000 and then explained the result. No need to stream a huge context into the prompt.

## Summary

| Step | Command / config |
|------|-------------------|
| 1. Start proxy | `uv run --with "minrlm[proxy]" examples/proxy.py` |
| 2. Config | `opencode.json` with `provider.minrlm.api = "http://localhost:8000/v1"` and a model (e.g. `gpt-5-mini-rlm`) |
| 3. Run | `OPENCODE_CONFIG=opencode.json opencode run "Your prompt"` |

For more on minRLM (Python API, CLI, evals), see the [main README](../README.md) and [examples](../examples/README.md).
