"""
FastAPI proxy for OpenAI-compatible requests with optional RLM enhancement.

Supports streaming (SSE), tool/function calling, and multi-turn conversations.
When large context data is detected, routes through RLM for token efficiency.
Otherwise, acts as a streaming pass-through to the underlying LLM.

Compatible with opencode, Vercel AI SDK, and any OpenAI-compatible client.

Usage:
    cd examples
    uvicorn proxy:app --host 0.0.0.0 --port 8000

    # With verbose logging
    MINRLM_VERBOSE=1 uvicorn proxy:app --host 0.0.0.0 --port 8000

    # Configure in opencode.json:
    # {
    #   "provider": {
    #     "minrlm": {
    #       "name": "minRLM",
    #       "api": "http://localhost:8000/v1",
    #       "npm": "@ai-sdk/openai-compatible",
    #       "env": ["OPENAI_API_KEY"],
    #       "models": {
    #         "gpt-5-mini-rlm": {
    #           "id": "gpt-5-mini-rlm",
    #           "name": "GPT-5 Mini (RLM)",
    #           "attachment": false,
    #           "reasoning": false,
    #           "temperature": true,
    #           "tool_call": true,
    #           "release_date": "2025-01-01",
    #           "limit": { "context": 128000, "output": 16384 },
    #           "cost": { "input": 0.4, "output": 1.6 },
    #           "options": {}
    #         }
    #       }
    #     }
    #   }
    # }
"""

import json
import os
import sys
import time
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAI

from minrlm import RLM

# Check for verbose mode
VERBOSE = os.environ.get("MINRLM_VERBOSE", "").lower() in ("1", "true", "yes")

# Threshold for treating message content as context (chars)
# Messages larger than this trigger RLM mode instead of pass-through
LARGE_CONTEXT_THRESHOLD = 50000

app = FastAPI(
    title="RLM Proxy",
    description="OpenAI-compatible proxy with RLM enhancement for large context",
    version="0.2.0",
)

# Global state
openai_client: OpenAI | None = None
rlm_instance: RLM | None = None
default_model: str = "gpt-5-mini"


def log(msg: str) -> None:
    if VERBOSE:
        print(f"[RLM Proxy] {msg}", file=sys.stderr)


@app.on_event("startup")
async def startup_event():
    """Initialize OpenAI client and RLM instance on startup."""
    global openai_client, rlm_instance, default_model

    model = os.getenv("RLM_MODEL", "gpt-5-mini")
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    use_docker = os.getenv("RLM_USE_DOCKER", "false").lower() == "true"
    default_model = model

    # Direct OpenAI client for pass-through mode
    openai_client = OpenAI(api_key=api_key, base_url=base_url)

    # RLM instance for enhanced mode
    rlm_instance = RLM(
        model=model,
        api_key=api_key,
        base_url=base_url,
        use_docker=use_docker,
    )

    print(f"RLM Proxy initialized | model={model} | docker={use_docker}")


@app.get("/")
async def root():
    return {"status": "ok", "service": "RLM Proxy", "version": "0.2.0"}


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    model_name = default_model
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": 1677610602,
                "owned_by": "rlm-proxy",
            },
            {
                "id": f"{model_name}-rlm",
                "object": "model",
                "created": 1677610602,
                "owned_by": "rlm-proxy",
            },
        ],
    }


# --- Pass-through streaming mode ---


def _stream_passthrough(body: dict[str, Any]):
    """
    Stream from the underlying LLM and yield SSE chunks.
    Passes through tool calls, content, and usage as-is.
    """
    model = body.get("model", default_model)
    # Strip -rlm suffix if present (RLM mode is handled separately)
    if model.endswith("-rlm"):
        model = model[:-4]

    # Build kwargs for OpenAI client
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": body["messages"],
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    # Pass through optional parameters
    for param in ["temperature", "top_p", "max_tokens", "max_completion_tokens",
                   "frequency_penalty", "presence_penalty", "stop", "seed",
                   "response_format", "logprobs", "top_logprobs", "n"]:
        if param in body and body[param] is not None:
            kwargs[param] = body[param]

    # Pass through tools and tool_choice
    if "tools" in body and body["tools"]:
        kwargs["tools"] = body["tools"]
    if "tool_choice" in body:
        kwargs["tool_choice"] = body["tool_choice"]

    log(f"Streaming pass-through | model={model} | tools={len(kwargs.get('tools', []))}")

    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    try:
        stream = openai_client.chat.completions.create(**kwargs)

        for chunk in stream:
            # Convert chunk to SSE format
            chunk_data: dict[str, Any] = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": body.get("model", model),
            }

            choices = []
            for choice in chunk.choices:
                choice_data: dict[str, Any] = {
                    "index": choice.index,
                    "delta": {},
                    "finish_reason": choice.finish_reason,
                }

                delta = choice.delta
                if delta.role is not None:
                    choice_data["delta"]["role"] = delta.role
                if delta.content is not None:
                    choice_data["delta"]["content"] = delta.content
                if delta.tool_calls:
                    tool_calls_data = []
                    for tc in delta.tool_calls:
                        tc_data: dict[str, Any] = {"index": tc.index}
                        if tc.id is not None:
                            tc_data["id"] = tc.id
                        if tc.type is not None:
                            tc_data["type"] = tc.type
                        if tc.function is not None:
                            fn: dict[str, Any] = {}
                            if tc.function.name is not None:
                                fn["name"] = tc.function.name
                            if tc.function.arguments is not None:
                                fn["arguments"] = tc.function.arguments
                            tc_data["function"] = fn
                        tool_calls_data.append(tc_data)
                    choice_data["delta"]["tool_calls"] = tool_calls_data

                choices.append(choice_data)

            chunk_data["choices"] = choices

            # Include usage if present (last chunk)
            if chunk.usage is not None:
                chunk_data["usage"] = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens,
                }

            yield f"data: {json.dumps(chunk_data)}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as e:
        # Send error as SSE event
        error_data = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": body.get("model", model),
            "choices": [{
                "index": 0,
                "delta": {"content": f"\n\n[Error: {str(e)}]"},
                "finish_reason": "stop",
            }],
        }
        yield f"data: {json.dumps(error_data)}\n\n"
        yield "data: [DONE]\n\n"


def _non_stream_passthrough(body: dict[str, Any]) -> dict[str, Any]:
    """Non-streaming pass-through to underlying LLM."""
    model = body.get("model", default_model)
    if model.endswith("-rlm"):
        model = model[:-4]

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": body["messages"],
    }

    for param in ["temperature", "top_p", "max_tokens", "max_completion_tokens",
                   "frequency_penalty", "presence_penalty", "stop", "seed",
                   "response_format", "logprobs", "top_logprobs", "n"]:
        if param in body and body[param] is not None:
            kwargs[param] = body[param]

    if "tools" in body and body["tools"]:
        kwargs["tools"] = body["tools"]
    if "tool_choice" in body:
        kwargs["tool_choice"] = body["tool_choice"]

    log(f"Non-stream pass-through | model={model}")

    resp = openai_client.chat.completions.create(**kwargs)

    choices = []
    for choice in resp.choices:
        choice_data: dict[str, Any] = {
            "index": choice.index,
            "message": {
                "role": choice.message.role,
                "content": choice.message.content,
            },
            "finish_reason": choice.finish_reason,
        }
        if choice.message.tool_calls:
            choice_data["message"]["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in choice.message.tool_calls
            ]
        choices.append(choice_data)

    result = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", model),
        "choices": choices,
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
            "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
            "total_tokens": resp.usage.total_tokens if resp.usage else 0,
        },
    }
    return result


# --- RLM-enhanced mode ---


def _extract_task_and_context(messages: list[dict[str, Any]]) -> tuple[str, str]:
    """
    Extract task and context from messages for RLM processing.
    Large messages (>50K chars) become context, rest becomes task.
    """
    system_parts: list[str] = []
    user_parts: list[str] = []
    context_parts: list[str] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            # Handle multi-part content (text + images)
            content = " ".join(
                part.get("text", "") for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )

        if role == "system":
            system_parts.append(content)
        elif role == "user":
            if len(content) > LARGE_CONTEXT_THRESHOLD:
                context_parts.append(content)
            else:
                user_parts.append(content)

    task = "\n\n".join(user_parts) if user_parts else ""
    context = "\n\n".join(context_parts) if context_parts else ""

    if system_parts and not context:
        task = "\n\n".join(system_parts) + "\n\n" + task if task else "\n\n".join(system_parts)

    return task, context


def _rlm_completion(body: dict[str, Any]) -> dict[str, Any]:
    """Run RLM completion and return OpenAI-compatible response."""
    task, context = _extract_task_and_context(body["messages"])

    if not task:
        user_messages = [
            m.get("content", "") for m in body["messages"] if m.get("role") == "user"
        ]
        task = user_messages[-1] if user_messages else "Analyze the data."

    last_stdout = ""

    def on_step(event: str, data: dict[str, Any]) -> None:
        nonlocal last_stdout
        if event == "executed" and data.get("stdout"):
            last_stdout = data["stdout"]
        if VERBOSE:
            if event == "thinking":
                log(f"RLM iter {data.get('iteration', '?')}: thinking...")
            elif event == "executed":
                if data.get("error"):
                    log(f"RLM iter {data.get('iteration', '?')}: error - {data['error'][:100]}")
                elif data.get("output"):
                    log(f"RLM iter {data.get('iteration', '?')}: FINAL() - {str(data['output'])[:100]}")

    rlm_inst = RLM(
        model=body.get("model", default_model).removesuffix("-rlm"),
        api_key=openai_client.api_key if openai_client else None,
        base_url=str(openai_client.base_url) if openai_client and openai_client.base_url else None,
        temperature=body.get("temperature"),
        max_output_tokens=body.get("max_tokens") or body.get("max_completion_tokens"),
        on_step=on_step,
    )

    log(f"RLM mode | task={task[:80]}... | context={len(context)} chars")
    result = rlm_inst.completion(task=task, context=context)

    response_content = result.response
    placeholder_patterns = ["answer", "result", "output", "done", "complete"]
    if response_content.lower().strip() in placeholder_patterns and last_stdout.strip():
        response_content = last_stdout.strip()
    elif not response_content.strip() and last_stdout.strip():
        response_content = last_stdout.strip()

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", default_model),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": result.input_tokens,
            "completion_tokens": result.output_tokens,
            "total_tokens": result.total_tokens,
        },
    }


def _stream_rlm(body: dict[str, Any]):
    """Stream RLM result as SSE (runs RLM, then streams the result)."""
    result = _rlm_completion(body)
    response_id = result["id"]
    created = result["created"]
    model = result["model"]
    content = result["choices"][0]["message"]["content"]

    # First chunk: role
    yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"

    # Stream content in chunks for a natural feel
    chunk_size = 50
    for i in range(0, len(content), chunk_size):
        chunk = content[i:i + chunk_size]
        yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'content': chunk}, 'finish_reason': None}]})}\n\n"

    # Final chunk with finish_reason and usage
    yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}], 'usage': result['usage']})}\n\n"
    yield "data: [DONE]\n\n"


# --- Routing ---


def _should_use_rlm(body: dict[str, Any]) -> bool:
    """
    Decide whether to use RLM mode or pass-through.

    RLM mode activates when:
    1. Model name ends with "-rlm" (explicit opt-in)
    2. Messages contain large context data (>50K chars)
    3. No tools are present (RLM can't call external tools)
    """
    model = body.get("model", "")
    if model.endswith("-rlm"):
        return True

    # If tools are present, always pass through (opencode's agentic loop)
    if body.get("tools"):
        return False

    # Check for large context in messages
    for msg in body.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, str) and len(content) > LARGE_CONTEXT_THRESHOLD:
            return True

    return False


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI-compatible chat completions endpoint.

    Routes to either:
    - Pass-through streaming (with tools) for agentic use
    - RLM-enhanced mode for large context processing
    """
    body = await request.json()
    is_stream = body.get("stream", False)
    use_rlm = _should_use_rlm(body)

    log(f"Request | stream={is_stream} | rlm={use_rlm} | model={body.get('model', '?')} | tools={len(body.get('tools', []))}")

    if use_rlm:
        if is_stream:
            return StreamingResponse(
                _stream_rlm(body),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return JSONResponse(content=_rlm_completion(body))
    else:
        if is_stream:
            return StreamingResponse(
                _stream_passthrough(body),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return JSONResponse(content=_non_stream_passthrough(body))


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
