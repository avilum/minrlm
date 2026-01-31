"""
minrlm - Minimal Recursive Language Model
Based on https://arxiv.org/abs/2512.24601
Implemented by Avi Lumelsky
"""

from .core import RLM, PythonREPL, RLMResult

__all__ = ["RLM", "RLMResult", "PythonREPL"]
__version__ = "0.1.0"
