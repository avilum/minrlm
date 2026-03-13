"""
Reasoning-enhanced runner for testing the impact of reasoning before code generation.
"""

import re
import sys
import time
from pathlib import Path

from .runners import BaseRunner, RunResult, register_runner


@register_runner("minrlm-reasoning")
class OursReasoningRunner(BaseRunner):
    """
    minRLM with Reasoning Step

    Adds a reasoning phase before code generation to catch:
    - Regex pattern bugs
    - Strategy errors (e.g., calling sub_llm on already-labeled data)
    - Task misunderstandings
    - Parsing bugs
    """

    description = "minRLM with reasoning before code generation"

    def __init__(
        self, model: str, max_iterations: int = 10, log_dir: str | None = None, use_docker: bool | None = None, **kwargs
    ):
        super().__init__(model, **kwargs)
        self.max_iterations = max_iterations
        self.log_dir = log_dir

        # Import our RLM
        module_path = str(Path(__file__).parent.parent)
        if module_path not in sys.path:
            sys.path.insert(0, module_path)

        from minrlm.core_reasoning import RLMReasoning
        from minrlm.docker_repl import check_docker_available

        self.RLM = RLMReasoning

        # Auto-detect Docker availability if not explicitly set
        # sub_llm() now works with Docker via request/response protocol
        if use_docker is None:
            self.use_docker = check_docker_available()
        else:
            self.use_docker = use_docker

    def run(self, task: str, context: str) -> RunResult:
        start = time.time()

        try:
            rlm = self.RLM(
                model=self.model,
                max_iterations=self.max_iterations,
                max_time_seconds=300,
                log_dir=self.log_dir,
                async_batch=True,
                use_docker=self.use_docker,
                max_output_tokens=8000,  # Increased for complex code generation with reasoning prompts
            )
            result = rlm.completion(task=task, context=context)
            elapsed = time.time() - start

            # Extract generated code from history (last assistant message with code)
            generated_code = None
            for msg in reversed(result.history):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if "```python" in content:
                        generated_code = content
                        break

            return RunResult(
                response=result.response,
                total_tokens=result.total_tokens,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                time_seconds=elapsed,
                iterations=result.iterations,
                metadata={"reasoning": getattr(result, "reasoning", "")},
                generated_code=generated_code,
                log_file_path=result.log_file_path,
            )
        except Exception as e:
            return RunResult(
                response="",
                error=str(e),
                time_seconds=time.time() - start,
            )
