"""
minrlm - Minimal Recursive Language Model
Based on https://arxiv.org/abs/2512.24601
Implemented by Avi Lumelsky
"""

from .core import RLM, PythonREPL, RLMResult
from .docker_repl import DockerREPL, check_docker_available

__all__ = ["RLM", "RLMResult", "PythonREPL", "DockerREPL", "check_docker_available"]
__version__ = "0.1.0"
