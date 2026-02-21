"""
DockerREPL - Secure Python REPL running inside Docker container.

Provides sandboxed code execution with:
- No network access (via seccomp)
- Limited syscalls
- Memory/CPU limits
- Timeout protection

Platform Compatibility:
----------------------
- **Linux**: Full support. Seccomp profiles are applied directly to the container.
- **macOS**: Full support via Docker Desktop. Containers run inside a Linux VM,
  so seccomp profiles are applied to the Linux kernel inside that VM.
- **Windows**: Supported via Docker Desktop (Linux containers mode).

Note: Seccomp is a Linux kernel feature. On macOS/Windows, Docker Desktop runs
a lightweight Linux VM, and the seccomp profile is applied inside that VM.
This provides equivalent security isolation.

Requirements:
- Docker must be installed and running
- User must have permission to run Docker commands

Usage:
    from minrlm import RLM
    rlm = RLM(model="gpt-5-mini", use_docker=True)

If Docker is not available, RLM will fall back to the local PythonREPL
with a warning message.
"""

import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# Set up logging
log = logging.getLogger(__name__)

# Enable debug logging if MINRLM_VERBOSE is set
import os

if os.environ.get("MINRLM_VERBOSE"):
    logging.basicConfig(level=logging.DEBUG, format="[DockerREPL] %(message)s", stream=sys.stderr)

# Strict seccomp profile - blocks networking and dangerous syscalls
# NOTE: We cannot block execve as it's needed to start the Python interpreter.
# Instead, we rely on Docker's other isolation: --network=none, --read-only, memory limits.
SECCOMP_PROFILE = {
    "defaultAction": "SCMP_ACT_ALLOW",
    "syscalls": [
        # Block all network-related syscalls
        {"names": ["socket", "socketpair", "setsockopt", "getsockopt"], "action": "SCMP_ACT_ERRNO", "errnoRet": 1},
        {"names": ["connect", "accept", "accept4", "bind", "listen"], "action": "SCMP_ACT_ERRNO", "errnoRet": 1},
        {"names": ["sendto", "recvfrom", "sendmsg", "recvmsg"], "action": "SCMP_ACT_ERRNO", "errnoRet": 1},
        {"names": ["sendmmsg", "recvmmsg"], "action": "SCMP_ACT_ERRNO", "errnoRet": 1},
        # NOTE: Cannot block execve - needed to start Python interpreter
        # Process isolation is handled by Docker's --pids-limit and container boundaries
        # Block kernel module loading
        {"names": ["init_module", "finit_module", "delete_module"], "action": "SCMP_ACT_ERRNO", "errnoRet": 1},
        # Block mount operations
        {"names": ["mount", "umount", "umount2", "pivot_root"], "action": "SCMP_ACT_ERRNO", "errnoRet": 1},
        # Block keyring access
        {"names": ["add_key", "request_key", "keyctl"], "action": "SCMP_ACT_ERRNO", "errnoRet": 1},
        # Block ptrace (debugging/tracing)
        {"names": ["ptrace"], "action": "SCMP_ACT_ERRNO", "errnoRet": 1},
        # Block reboot/kexec
        {"names": ["reboot", "kexec_load", "kexec_file_load"], "action": "SCMP_ACT_ERRNO", "errnoRet": 1},
    ],
}

# Python wrapper script that runs inside the container
DOCKER_WRAPPER_SCRIPT = '''
import sys
import json
import re

# Read input data
input_data = json.loads(sys.stdin.read())
code = input_data["code"]
namespace = input_data.get("namespace", {})
input_0 = input_data.get("input_0")

# Set up namespace with input_0 if provided
exec_namespace = {"__builtins__": __builtins__}
if input_0 is not None:
    exec_namespace["input_0"] = input_0

# Add variables from namespace
for k, v in namespace.items():
    exec_namespace[k] = v

# Track output
_output = None
def FINAL(value):
    global _output
    _output = str(value)

def FINAL_var(var_name):
    global _output
    if var_name not in exec_namespace:
        raise NameError(f"Variable '{var_name}' not found")
    _output = str(exec_namespace[var_name])

def search(text, pattern, context=100):
    """Search for pattern in text and return matches with context."""
    matches = []
    text_lower = text.lower()
    pattern_lower = pattern.lower()
    start = 0

    while True:
        pos = text_lower.find(pattern_lower, start)
        if pos == -1:
            break
        ctx_start = max(0, pos - context)
        ctx_end = min(len(text), pos + len(pattern) + context)
        match_text = text[ctx_start:ctx_end]
        matches.append(match_text)
        start = pos + 1

    return matches

def peek(data, max_len=500, max_items=5, depth=0):
    """Efficient preview of data."""
    if isinstance(data, str):
        if len(data) <= max_len:
            preview = repr(data)
        else:
            preview = f"str[{len(data):,}]"
        return preview
    elif isinstance(data, (list, tuple)):
        bracket = "[]" if isinstance(data, list) else "()"
        if len(data) == 0:
            return bracket
        for item in data[:max_items]:
            peek(item, max_len, max_items, depth + 1)
        return f"{bracket} ({len(data)} items)"
    elif isinstance(data, dict):
        if len(data) == 0:
            return "{}"
        for k, v in list(data.items())[:max_items]:
            peek(v, max_len, max_items, depth + 2)
        return f"{{}} ({len(data)} keys)"
    else:
        preview = repr(data)
        if len(preview) > max_len:
            preview = preview[:max_len] + "..."
        return preview

def sub_llm(task, context=""):
    """sub_llm is not supported in Docker mode - raises error."""
    raise RuntimeError("sub_llm() is not available in Docker mode. Use non-Docker RLM for recursive calls.")

def sub_llm_batch(tasks):
    """sub_llm_batch is not supported in Docker mode - raises error."""
    raise RuntimeError("sub_llm_batch() is not available in Docker mode. Use non-Docker RLM for recursive calls.")

# Add helper functions to namespace
exec_namespace["FINAL"] = FINAL
exec_namespace["FINAL_var"] = FINAL_var
exec_namespace["search"] = search
exec_namespace["peek"] = peek
exec_namespace["sub_llm"] = sub_llm
exec_namespace["sub_llm_batch"] = sub_llm_batch

# Capture stdout
from io import StringIO
import sys as _sys
_stdout_capture = StringIO()
_old_stdout = _sys.stdout
_sys.stdout = _stdout_capture

error = None
try:
    exec(code, exec_namespace)
except Exception as e:
    error = f"{type(e).__name__}: {e}"
finally:
    _sys.stdout = _old_stdout
    stdout = _stdout_capture.getvalue()

# Build state dict (visible variables)
state = {}
HIDDEN = {"__builtins__", "FINAL", "FINAL_var", "search", "peek", "sub_llm", "sub_llm_batch"}
for name, value in exec_namespace.items():
    if name in HIDDEN or name.startswith("_"):
        continue
    t = type(value).__name__
    if isinstance(value, str):
        preview = value[:80] + "..." if len(value) > 80 else value
        state[name] = f"str({len(value)} chars) = {repr(preview)}"
    elif isinstance(value, (list, tuple)):
        state[name] = f"{t}({len(value)} items)"
    elif isinstance(value, dict):
        state[name] = f"dict({len(value)} keys)"
    elif isinstance(value, (int, float, bool)):
        state[name] = f"{t} = {value}"
    else:
        state[name] = t

# Output result as JSON
result = {
    "stdout": stdout,
    "output": _output,
    "error": error,
    "state": state
}
print("__DOCKER_RESULT__:" + json.dumps(result))
'''


class DockerREPL:
    """
    Secure Python REPL running inside a Docker container.

    Provides sandboxed execution with:
    - No network access (seccomp blocks socket syscalls)
    - Memory limits (default 256MB)
    - CPU limits (default 1 core)
    - Process limits (100 max)
    - Read-only filesystem (except /tmp)
    - Execution timeout
    - Strict seccomp profile blocking dangerous syscalls

    Platform Support:
    - Linux: Native seccomp support
    - macOS: Via Docker Desktop (seccomp applied in Linux VM)
    - Windows: Via Docker Desktop (Linux containers mode)

    Limitations:
    - sub_llm() is NOT supported (no callback to host)
    - State is not persisted between executions
    - Complex objects cannot be transferred back from container
    """

    HIDDEN_KEYS = {"__builtins__", "sub_llm", "sub_llm_batch", "FINAL", "FINAL_var", "peek", "search"}

    def __init__(
        self,
        image: str = "python:3.11-slim",
        memory_limit: str = "256m",
        cpu_limit: float = 1.0,
        timeout: int = 60,
        network_disabled: bool = True,
    ):
        """
        Initialize DockerREPL.

        Args:
            image: Docker image to use (default: python:3.11-slim)
            memory_limit: Memory limit (e.g., "256m", "1g")
            cpu_limit: CPU limit (1.0 = 1 CPU core)
            timeout: Execution timeout in seconds
            network_disabled: Disable all networking (default: True)
        """
        self.image = image
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.timeout = timeout
        self.network_disabled = network_disabled

        self._output: str | None = None
        self._namespace: dict[str, Any] = {}
        self._input_0: str | None = None

        # Verify Docker is available
        self._docker_available = self._check_docker()

    def _check_docker(self) -> bool:
        """Check if Docker is available and working."""
        log.debug("Checking if Docker is available...")
        if not shutil.which("docker"):
            log.debug("Docker binary not found in PATH")
            return False
        try:
            log.debug("Running 'docker info' to verify Docker daemon...")
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0:
                log.debug("Docker is available and running")
            else:
                log.debug(f"Docker info failed: {result.stderr.decode()[:200]}")
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            log.debug("Docker info timed out - daemon may be unresponsive")
            return False
        except OSError as e:
            log.debug(f"Docker check failed with OSError: {e}")
            return False

    def is_available(self) -> bool:
        """Check if Docker REPL is available."""
        return self._docker_available

    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable in the namespace."""
        if name == "input_0":
            self._input_0 = value
        else:
            self._namespace[name] = value

    def reset(self) -> None:
        """Reset REPL state."""
        self._output = None
        self._namespace = {}
        self._input_0 = None

    def save_output(self) -> str | None:
        """Save current output state."""
        return self._output

    def restore_output(self, saved: str | None) -> None:
        """Restore output state."""
        self._output = saved

    def execute(self, code: str) -> dict[str, Any]:
        """
        Execute code in Docker container.

        Returns:
            Dict with {stdout, output, error, state}
        """
        log.debug(f"Executing code in Docker (len={len(code)} chars)")

        if not self._docker_available:
            log.debug("Docker not available, returning error")
            return {
                "stdout": "",
                "output": None,
                "error": "Docker is not available. Install Docker or use non-Docker REPL.",
                "state": {},
            }

        self._output = None

        # Prepare input data
        input_data = {
            "code": code,
            "namespace": {
                k: v for k, v in self._namespace.items() if isinstance(v, str | int | float | bool | list | dict)
            },
            "input_0": self._input_0,
        }
        log.debug(f"Input data prepared, input_0 len={len(self._input_0) if self._input_0 else 0}")

        # Create temp directory for seccomp profile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write seccomp profile
            seccomp_path = Path(tmpdir) / "seccomp.json"
            with open(seccomp_path, "w") as f:
                json.dump(SECCOMP_PROFILE, f)
            log.debug(f"Wrote seccomp profile to {seccomp_path}")

            # Write wrapper script
            wrapper_path = Path(tmpdir) / "wrapper.py"
            with open(wrapper_path, "w") as f:
                f.write(DOCKER_WRAPPER_SCRIPT)
            log.debug(f"Wrote wrapper script to {wrapper_path}")

            # Build Docker command
            cmd = [
                "docker",
                "run",
                "--rm",  # Remove container after execution
                "-i",  # Interactive (for stdin)
                f"--memory={self.memory_limit}",
                f"--cpus={self.cpu_limit}",
                "--pids-limit=100",  # Limit processes
                "--read-only",  # Read-only filesystem
                "--tmpfs=/tmp:rw,noexec,nosuid,size=64m",  # Writable /tmp
                f"--security-opt=seccomp={seccomp_path}",
            ]

            # Disable networking
            if self.network_disabled:
                cmd.append("--network=none")

            # Mount wrapper script
            cmd.extend(
                [
                    "-v",
                    f"{wrapper_path}:/app/wrapper.py:ro",
                    self.image,
                    "python",
                    "/app/wrapper.py",
                ]
            )

            log.debug(f"Docker command: {' '.join(cmd)}")
            log.debug(f"Running with timeout={self.timeout}s...")

            try:
                result = subprocess.run(
                    cmd,
                    input=json.dumps(input_data),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )

                log.debug(f"Docker exited with code {result.returncode}")
                log.debug(f"stdout len={len(result.stdout)}, stderr len={len(result.stderr)}")

                # Parse output
                stdout = result.stdout
                stderr = result.stderr

                if stderr:
                    log.debug(f"stderr: {stderr[:500]}")

                # Look for our result marker
                if "__DOCKER_RESULT__:" in stdout:
                    log.debug("Found result marker in output")
                    parts = stdout.split("__DOCKER_RESULT__:", 1)
                    visible_stdout = parts[0]
                    try:
                        exec_result = json.loads(parts[1].strip())
                        self._output = exec_result.get("output")
                        log.debug(
                            f"Parsed result: output={exec_result.get('output')}, error={exec_result.get('error')}"
                        )

                        # Update namespace from container state
                        # Note: We can't actually transfer complex objects back,
                        # but we can track what variables exist

                        return {
                            "stdout": visible_stdout + exec_result.get("stdout", ""),
                            "output": exec_result.get("output"),
                            "error": exec_result.get("error"),
                            "state": exec_result.get("state", {}),
                        }
                    except json.JSONDecodeError as e:
                        log.debug(f"Failed to parse JSON result: {e}")
                        return {
                            "stdout": stdout,
                            "output": None,
                            "error": f"Failed to parse container output: {stderr}",
                            "state": {},
                        }
                else:
                    # No result marker - might be an error
                    log.debug(f"No result marker found. stdout preview: {stdout[:200]}")
                    error = stderr if stderr else "No output from container"
                    if result.returncode != 0:
                        error = f"Container exited with code {result.returncode}: {stderr}"
                    log.debug(f"Returning error: {error[:200]}")
                    return {
                        "stdout": stdout,
                        "output": None,
                        "error": error,
                        "state": {},
                    }

            except subprocess.TimeoutExpired:
                log.debug(f"Docker execution timed out after {self.timeout}s")
                return {
                    "stdout": "",
                    "output": None,
                    "error": f"Execution timed out after {self.timeout}s",
                    "state": {},
                }
            except Exception as e:
                log.debug(f"Docker execution failed with exception: {e}")
                return {
                    "stdout": "",
                    "output": None,
                    "error": f"Docker execution failed: {e}",
                    "state": {},
                }

    def get_state(self) -> dict[str, str]:
        """Get current namespace state."""
        # In Docker mode, we don't persist state between executions
        # This returns what we know from the last execution
        return {}

    def _describe_value(self, value: Any, max_len: int = 80) -> str:
        """Describe a value with type and preview."""
        t = type(value).__name__
        if isinstance(value, str):
            preview = value[:max_len] + "..." if len(value) > max_len else value
            return f"str({len(value)} chars) = {repr(preview)}"
        elif isinstance(value, list | tuple):
            return f"{t}({len(value)} items)"
        elif isinstance(value, dict):
            return f"dict({len(value)} keys)"
        elif isinstance(value, int | float | bool):
            return f"{t} = {value}"
        else:
            return t


def check_docker_available() -> bool:
    """Check if Docker is available on this system."""
    repl = DockerREPL()
    return repl.is_available()
