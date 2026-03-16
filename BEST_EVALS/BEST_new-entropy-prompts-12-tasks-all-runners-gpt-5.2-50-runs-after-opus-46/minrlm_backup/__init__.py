"""
minrlm - Minimal Recursive Language Model
Based on https://arxiv.org/abs/2512.24601
Implemented by Avi Lumelsky
"""

from .core import RLM as RLMBase
from .core import PythonREPL, RLMResult
from .core_reasoning import RLMReasoning, RLMReasoningResult
from .docker_repl import DockerREPL, check_docker_available

# RLMReasoning is the default - it's what the benchmarks use.
# Use RLMBase if you want the bare-bones version.
RLM = RLMReasoning

__all__ = [
    "RLM",
    "RLMBase",
    "RLMReasoning",
    "RLMResult",
    "RLMReasoningResult",
    "PythonREPL",
    "DockerREPL",
    "check_docker_available",
]
__version__ = "0.1.0"
