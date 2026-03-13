"""
FastAPI proxy for OpenAI-compatible RLM requests.
THIS IS NOT A PRODUCTION-READY PROXY. IT IS A SIMPLE TOOL TO HELP YOU GET STARTED WITH RLM.

This proxy accepts OpenAI-compatible chat completion requests and routes them
through the RLM implementation, allowing existing OpenAI-based pipelines to use
RLM without code changes.

Usage:
    cd examples
    uvicorn proxy:app --host 0.0.0.0 --port 8000

    # With verbose logging
    MINRLM_VERBOSE=1 uvicorn proxy:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from minrlm import RLM

# Check for verbose mode
VERBOSE = os.environ.get("MINRLM_VERBOSE", "").lower() in ("1", "true", "yes")

# Threshold for treating message content as context (chars)
# Messages larger than this are treated as context data rather than task instructions
LARGE_CONTEXT_THRESHOLD = 50000

app = FastAPI(
    title="RLM Proxy",
    description="OpenAI-compatible proxy for Recursive Language Models",
    version="0.1.0",
)

# Global RLM instance and config (initialized on startup)
rlm: RLM | None = None
rlm_config: dict[str, Any] = {}


class ChatMessage(BaseModel):
    """OpenAI chat message format."""

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI chat completion request format."""

    model: str = Field(default="gpt-5-nano", description="Model name (passed to RLM)")
    messages: list[ChatMessage] = Field(..., description="Conversation messages")
    temperature: float | None = Field(
        default=None, description="Temperature (passed to RLM)"
    )
    max_tokens: int | None = Field(
        default=None, description="Max output tokens (passed to RLM)"
    )
    # Additional RLM-specific options
    context: str | None = Field(
        default=None, description="Optional context data (as input_0)"
    )
    use_docker: bool = Field(default=False, description="Use Docker sandboxing")
    max_iterations: int | None = Field(default=None, description="Max RLM iterations")


def extract_task_and_context(messages: list[ChatMessage]) -> tuple[str, str]:
    """
    Extract task and context from OpenAI message format.

    Strategy:
    - System messages are ignored (RLM has its own system prompt)
    - User messages are combined into the task
    - If a message contains very large content (>50K chars), treat it as context
    - Otherwise, combine all user messages as the task
    """
    system_parts: list[str] = []
    user_parts: list[str] = []
    context_parts: list[str] = []

    for msg in messages:
        if msg.role == "system":
            system_parts.append(msg.content)
        elif msg.role == "user":
            # If content is very large, treat as context
            if len(msg.content) > LARGE_CONTEXT_THRESHOLD:
                context_parts.append(msg.content)
            else:
                user_parts.append(msg.content)
        # Assistant messages are ignored (RLM handles its own conversation)

    # Combine user messages as task
    task = "\n\n".join(user_parts) if user_parts else ""

    # Combine context parts
    context = "\n\n".join(context_parts) if context_parts else ""

    # If system messages exist and no explicit context, prepend to task
    if system_parts and not context:
        task = (
            "\n\n".join(system_parts) + "\n\n" + task
            if task
            else "\n\n".join(system_parts)
        )

    return task, context


@app.on_event("startup")
async def startup_event():
    """Initialize RLM instance on startup."""
    global rlm, rlm_config

    # Get configuration from environment variables
    model = os.getenv("RLM_MODEL", "gpt-5-nano")
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    use_docker = os.getenv("RLM_USE_DOCKER", "false").lower() == "true"

    # Store config for creating new instances
    rlm_config = {
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        "use_docker": use_docker,
    }

    rlm = RLM(**rlm_config)

    print(f"✓ RLM Proxy initialized with model: {model}")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "RLM Proxy", "version": "0.1.0"}


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible endpoint)."""
    model_name = rlm.model if rlm else "gpt-5-nano"
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": 1677610602,
                "owned_by": "rlm-proxy",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> JSONResponse:
    """
    OpenAI-compatible chat completions endpoint.

    Accepts standard OpenAI chat completion requests and routes them through RLM.
    """
    if rlm is None:
        raise HTTPException(status_code=503, detail="RLM not initialized")

    try:
        # Extract task and context from messages
        task, context_from_messages = extract_task_and_context(request.messages)

        # Use explicit context if provided, otherwise use extracted context
        context = (
            request.context if request.context is not None else context_from_messages
        )

        # If no task extracted, use the last user message or raise error
        if not task:
            # Try to get last user message
            user_messages = [
                msg.content for msg in request.messages if msg.role == "user"
            ]
            if user_messages:
                task = user_messages[-1]
            else:
                raise HTTPException(
                    status_code=400, detail="No user message found in request"
                )

        # Create a temporary RLM instance with request-specific settings
        # (or use global instance if no overrides needed)
        rlm_instance = rlm

        # Track last stdout for fallback when response is a placeholder
        last_stdout: str = ""
        last_output: str | None = None

        # Create verbose logging callback if enabled
        def on_step(event: str, data: dict[str, Any]) -> None:
            """Verbose logging callback for RLM execution steps."""
            nonlocal last_stdout, last_output

            # Always track stdout and output (even when not verbose) for fallback
            if event == "executed":
                if data.get("stdout"):
                    last_stdout = data["stdout"]
                if data.get("output") is not None:
                    last_output = data["output"]

            # Only print verbose logs if enabled
            if not VERBOSE:
                return

            if event == "thinking":
                print(
                    f"[RLM] 🔄 Iteration {data.get('iteration', '?')}: Thinking...",
                    file=sys.stderr,
                )
            elif event == "llm_response":
                iteration = data.get("iteration", "?")
                has_code = data.get("has_code", False)
                response_preview = data.get("response", "")[:200]
                code_indicator = "✓ Has code" if has_code else "✗ No code"
                print(
                    f"[RLM] 📝 Iteration {iteration}: LLM Response ({code_indicator})",
                    file=sys.stderr,
                )
                if VERBOSE and response_preview:
                    print(f"      Preview: {response_preview}...", file=sys.stderr)
            elif event == "executing":
                iteration = data.get("iteration", "?")
                code = data.get("code", "")
                print(
                    f"[RLM] ⚙️  Iteration {iteration}: Executing code", file=sys.stderr
                )
                if code:
                    # Show first few lines of code
                    code_lines_list = code.split("\n")
                    code_lines = code_lines_list[:5]
                    for line in code_lines:
                        print(f"      {line}", file=sys.stderr)
                    if len(code_lines_list) > 5:
                        remaining = len(code_lines_list) - 5
                        print(f"      ... ({remaining} more lines)", file=sys.stderr)
            elif event == "executed":
                iteration = data.get("iteration", "?")
                if data.get("error"):
                    print(
                        f"[RLM] ❌ Iteration {iteration}: Error - {data['error']}",
                        file=sys.stderr,
                    )
                elif data.get("output"):
                    output = data["output"]
                    output_preview = (
                        output[:200] + "..." if len(output) > 200 else output
                    )
                    print(
                        f"[RLM] ✅ Iteration {iteration}: FINAL() called",
                        file=sys.stderr,
                    )
                    print(f"      Output: {output_preview}", file=sys.stderr)
                elif data.get("stdout"):
                    stdout = data["stdout"]
                    stdout_preview = (
                        stdout[:200] + "..." if len(stdout) > 200 else stdout
                    )
                    print(
                        f"[RLM] 📤 Iteration {iteration}: Code executed (stdout captured)",
                        file=sys.stderr,
                    )
                    print(f"      Stdout: {stdout_preview}", file=sys.stderr)
            elif event == "timeout":
                print(
                    f"[RLM] ⏱️  Timeout after {data.get('elapsed', '?')}s",
                    file=sys.stderr,
                )

        # Override settings if provided in request
        # Always create instance with on_step callback to track stdout for fallback
        if (
            request.temperature is not None
            or request.max_tokens is not None
            or request.max_iterations is not None
        ):
            rlm_instance = RLM(
                model=request.model or rlm.model,
                api_key=rlm_config.get("api_key"),
                base_url=rlm_config.get("base_url"),
                temperature=(
                    request.temperature
                    if request.temperature is not None
                    else rlm.temperature
                ),
                max_output_tokens=(
                    request.max_tokens
                    if request.max_tokens is not None
                    else rlm.max_output_tokens
                ),
                max_iterations=(
                    request.max_iterations
                    if request.max_iterations is not None
                    else rlm.max_iterations
                ),
                use_docker=request.use_docker or rlm.use_docker,
                on_step=on_step,  # Always use callback to track stdout
            )
        else:
            # Create new instance with callback (always needed for stdout tracking)
            rlm_instance = RLM(
                model=rlm.model,
                api_key=rlm_config.get("api_key"),
                base_url=rlm_config.get("base_url"),
                temperature=rlm.temperature,
                max_output_tokens=rlm.max_output_tokens,
                max_iterations=rlm.max_iterations,
                use_docker=rlm.use_docker,
                on_step=on_step,
            )

        if VERBOSE:
            print("[RLM] 🚀 Starting completion", file=sys.stderr)
            print(
                f"      Task: {task[:100]}{'...' if len(task) > 100 else ''}",
                file=sys.stderr,
            )
            if context:
                print(f"      Context: {len(context)} chars", file=sys.stderr)

        # Run RLM completion
        result = rlm_instance.completion(task=task, context=context)

        # Check if response is a placeholder and use stdout as fallback
        response_content = result.response
        placeholder_patterns = ["answer", "result", "output", "done", "complete"]

        # If response is a placeholder and we have stdout, use stdout instead
        if (
            response_content.lower().strip() in placeholder_patterns
            and last_stdout
            and len(last_stdout.strip()) > 0
        ):
            if VERBOSE:
                print(
                    f"[RLM] ⚠️  Response is placeholder '{response_content}', using stdout as fallback",
                    file=sys.stderr,
                )
            response_content = last_stdout.strip()
        # If response is empty or just whitespace and we have stdout, use stdout
        elif (
            not response_content.strip()
            and last_stdout
            and len(last_stdout.strip()) > 0
        ):
            if VERBOSE:
                print(
                    "[RLM] ⚠️  Response is empty, using stdout as fallback",
                    file=sys.stderr,
                )
            response_content = last_stdout.strip()

        if VERBOSE:
            print("[RLM] ✨ Completion finished", file=sys.stderr)
            print(
                f"      Response: {response_content[:200]}{'...' if len(response_content) > 200 else ''}",
                file=sys.stderr,
            )
            print(f"      Iterations: {result.iterations}", file=sys.stderr)
            print(
                f"      Tokens: {result.total_tokens} (in: {result.input_tokens}, out: {result.output_tokens})",
                file=sys.stderr,
            )

        # Format response in OpenAI-compatible format
        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        response_data: dict[str, Any] = {
            "id": response_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": result.input_tokens,
                "completion_tokens": result.output_tokens,
                "total_tokens": result.total_tokens,
            },
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"RLM completion failed: {str(e)}"
        ) from e


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
