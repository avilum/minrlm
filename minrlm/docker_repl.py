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

import atexit
import itertools
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

# Set up logging
log = logging.getLogger(__name__)

# Enable debug logging if MINRLM_VERBOSE is set
if os.environ.get("MINRLM_VERBOSE"):
    logging.basicConfig(
        level=logging.DEBUG, format="[DockerREPL] %(message)s", stream=sys.stderr
    )

# ---------------------------------------------------------------------------
# Process-wide container registry — ensures all containers spawned by this
# process are killed when the process exits, even on SIGTERM / SIGKILL-induced
# parent death, crash, or KeyboardInterrupt.
# ---------------------------------------------------------------------------

_REGISTRY: set[str] = set()  # container names (minrlm_<pid>_<n>)
_REGISTRY_LOCK = threading.Lock()
_CONTAINER_COUNTER = itertools.count(1)


def _register_container(name: str) -> None:
    with _REGISTRY_LOCK:
        _REGISTRY.add(name)


def _unregister_container(name: str) -> None:
    with _REGISTRY_LOCK:
        _REGISTRY.discard(name)


def _kill_container(name: str) -> None:
    """Best-effort docker kill + rm for a single container."""
    try:
        subprocess.run(["docker", "kill", name], capture_output=True, timeout=10)
    except Exception:
        pass
    try:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=10)
    except Exception:
        pass


def _cleanup_all_containers() -> None:
    """Kill every container this process ever started.  Called via atexit."""
    with _REGISTRY_LOCK:
        names = list(_REGISTRY)
    if names:
        log.debug(f"[DockerREPL] atexit: killing {len(names)} container(s): {names}")
    for name in names:
        _kill_container(name)
    with _REGISTRY_LOCK:
        _REGISTRY.clear()


def _signal_handler(signum: int, frame) -> None:
    """Kill containers then re-raise the signal so the process exits normally."""
    _cleanup_all_containers()
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


# Register atexit hook (handles normal exit, sys.exit, unhandled exceptions)
atexit.register(_cleanup_all_containers)

# Register signal handlers for graceful and forced termination
for _sig in (signal.SIGTERM, signal.SIGINT):
    try:
        signal.signal(_sig, _signal_handler)
    except (OSError, ValueError):
        pass  # can't set signal handlers in non-main threads; atexit still covers it

# Strict seccomp profile - blocks networking and dangerous syscalls
# NOTE: We cannot block execve as it's needed to start the Python interpreter.
# Instead, we rely on Docker's other isolation: --network=none, --read-only, memory limits.
SECCOMP_PROFILE = {
    "defaultAction": "SCMP_ACT_ALLOW",
    "syscalls": [
        # Block all network-related syscalls
        {
            "names": ["socket", "socketpair", "setsockopt", "getsockopt"],
            "action": "SCMP_ACT_ERRNO",
            "errnoRet": 1,
        },
        {
            "names": ["connect", "accept", "accept4", "bind", "listen"],
            "action": "SCMP_ACT_ERRNO",
            "errnoRet": 1,
        },
        {
            "names": ["sendto", "recvfrom", "sendmsg", "recvmsg"],
            "action": "SCMP_ACT_ERRNO",
            "errnoRet": 1,
        },
        {"names": ["sendmmsg", "recvmmsg"], "action": "SCMP_ACT_ERRNO", "errnoRet": 1},
        # NOTE: Cannot block execve - needed to start Python interpreter
        # Process isolation is handled by Docker's --pids-limit and container boundaries
        # Block kernel module loading
        {
            "names": ["init_module", "finit_module", "delete_module"],
            "action": "SCMP_ACT_ERRNO",
            "errnoRet": 1,
        },
        # Block mount operations
        {
            "names": ["mount", "umount", "umount2", "pivot_root"],
            "action": "SCMP_ACT_ERRNO",
            "errnoRet": 1,
        },
        # Block keyring access
        {
            "names": ["add_key", "request_key", "keyctl"],
            "action": "SCMP_ACT_ERRNO",
            "errnoRet": 1,
        },
        # Block ptrace (debugging/tracing)
        {"names": ["ptrace"], "action": "SCMP_ACT_ERRNO", "errnoRet": 1},
        # Block reboot/kexec
        {
            "names": ["reboot", "kexec_load", "kexec_file_load"],
            "action": "SCMP_ACT_ERRNO",
            "errnoRet": 1,
        },
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

def search(text, pattern, context=500):
    """Search for pattern in text and return matches with context.

    Returns:
        List of tuples: (match, before_context, after_context)
    """
    matches = []
    text_lower = text.lower()
    pattern_lower = pattern.lower()
    start = 0

    while True:
        pos = text_lower.find(pattern_lower, start)
        if pos == -1:
            break

        # Extract the actual matched text (preserve original case)
        match = text[pos:pos + len(pattern)]

        # Extract before and after context
        before_start = max(0, pos - context)
        before = text[before_start:pos]

        after_end = min(len(text), pos + len(pattern) + context)
        after = text[pos + len(pattern):after_end]

        matches.append((match, before, after))
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

# sub_llm support via request/response protocol
# Code can call sub_llm() - it will be handled by the host
_sub_llm_cache = input_data.get("sub_llm_cache", {})

def sub_llm(task, context=""):
    """Call sub_llm via host. Results are cached and injected by host."""
    cache_key = f"{task}||{context}"
    if cache_key in _sub_llm_cache:
        return _sub_llm_cache[cache_key]
    # Signal to host that we need a sub_llm call
    # This will cause execution to pause and request will be sent to host
    raise RuntimeError(f"__SUB_LLM_REQUEST__:{cache_key}")

def sub_llm_batch(tasks):
    """Call sub_llm_batch via host. Results are cached and injected by host."""
    results = []
    for task, context in tasks:
        results.append(sub_llm(task, context))
    return results

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

    HIDDEN_KEYS = {
        "__builtins__",
        "sub_llm",
        "sub_llm_batch",
        "FINAL",
        "FINAL_var",
        "peek",
        "search",
    }

    def __init__(
        self,
        image: str = "python:3.14-slim",
        memory_limit: str = "256m",
        cpu_limit: float = 1.0,
        timeout: int = 60,
        network_disabled: bool = True,
        sub_llm_callback=None,
        sub_llm_batch_callback=None,
    ):
        """
        Initialize DockerREPL.

        Args:
            image: Docker image to use (default: python:3.14-slim)
            memory_limit: Memory limit (e.g., "256m", "1g")
            cpu_limit: CPU limit (1.0 = 1 CPU core)
            timeout: Execution timeout in seconds
            sub_llm_callback: Optional callback for sub_llm() support
            sub_llm_batch_callback: Optional callback for sub_llm_batch() support
            network_disabled: Disable all networking (default: True)
        """
        self.image = image
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.timeout = timeout
        self.network_disabled = network_disabled
        self._sub_llm_callback = sub_llm_callback
        self._sub_llm_batch_callback = sub_llm_batch_callback

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

    def set_variable(self, name: str, value: Any, allow_override: bool = False) -> None:
        """Set a variable in the namespace.

        Args:
            name: Variable name
            value: Variable value
            allow_override: Accepted for API compatibility with PythonREPL
                           (Docker provides isolation via containers, not protected namespaces)
        """
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
        Execute code in Docker container with sub_llm support.

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

        # Cache for sub_llm results (persists across retries)
        sub_llm_cache: dict[str, str] = {}
        max_sub_llm_calls = 10  # Prevent infinite loops
        attempt = 0

        while attempt < max_sub_llm_calls:
            attempt += 1
            log.debug(
                f"Execute attempt {attempt} (sub_llm_cache has {len(sub_llm_cache)} entries)"
            )

            # Prepare input data
            input_data = {
                "code": code,
                "namespace": {
                    k: v
                    for k, v in self._namespace.items()
                    if isinstance(v, str | int | float | bool | list | dict)
                },
                "input_0": self._input_0,
                "sub_llm_cache": sub_llm_cache,  # Inject cached sub_llm results
            }
            log.debug(
                f"Input data prepared, input_0 len={len(self._input_0) if self._input_0 else 0}"
            )

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

                # Unique container name — tracked for cleanup on process exit
                container_name = f"minrlm_{os.getpid()}_{next(_CONTAINER_COUNTER)}"
                _register_container(container_name)

                # Build Docker command
                cmd = [
                    "docker",
                    "run",
                    "--rm",  # Remove container after execution
                    "-i",  # Interactive (for stdin)
                    f"--name={container_name}",  # Named for reliable kill-on-exit
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
                log.debug(
                    f"Running with timeout={self.timeout}s, container={container_name}..."
                )

                try:
                    result = subprocess.run(
                        cmd,
                        input=json.dumps(input_data),
                        capture_output=True,
                        text=True,
                        timeout=self.timeout,
                    )

                    log.debug(f"Docker exited with code {result.returncode}")
                    log.debug(
                        f"stdout len={len(result.stdout)}, stderr len={len(result.stderr)}"
                    )

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
                            error_msg = exec_result.get("error")
                            log.debug(
                                f"Parsed result: output={exec_result.get('output')}, error={error_msg}"
                            )

                            # Check if this is a sub_llm request
                            if error_msg and "__SUB_LLM_REQUEST__:" in str(error_msg):
                                cache_key = str(error_msg).split(
                                    "__SUB_LLM_REQUEST__:", 1
                                )[1]
                                log.debug(
                                    f"Detected sub_llm request: {cache_key[:100]}..."
                                )

                                # Parse task and context from cache key
                                if "||" in cache_key:
                                    task, context = cache_key.split("||", 1)
                                else:
                                    task, context = cache_key, ""

                                # Call host's sub_llm callback
                                if self._sub_llm_callback:
                                    log.debug(
                                        f"Calling host sub_llm: task={task[:50]}..."
                                    )
                                    try:
                                        llm_result = self._sub_llm_callback(
                                            task, context
                                        )
                                        sub_llm_cache[cache_key] = str(llm_result)
                                        log.debug(
                                            f"Got sub_llm result: {str(llm_result)[:100]}..."
                                        )
                                        continue  # Retry with cached result
                                    except Exception as e:
                                        log.debug(f"sub_llm callback failed: {e}")
                                        return {
                                            "stdout": visible_stdout,
                                            "output": None,
                                            "error": f"sub_llm failed - {e}",
                                            "state": exec_result.get("state", {}),
                                        }
                                else:
                                    log.debug("No sub_llm callback available")
                                    return {
                                        "stdout": visible_stdout,
                                        "output": None,
                                        "error": "sub_llm() not available - no callback provided",
                                        "state": exec_result.get("state", {}),
                                    }

                            # Normal result (no sub_llm request)
                            return {
                                "stdout": visible_stdout
                                + exec_result.get("stdout", ""),
                                "output": exec_result.get("output"),
                                "error": error_msg,
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
                        log.debug(
                            f"No result marker found. stdout preview: {stdout[:200]}"
                        )
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
                    log.debug(
                        f"Docker timed out after {self.timeout}s — killing {container_name}"
                    )
                    _kill_container(container_name)
                    return {
                        "stdout": "",
                        "output": None,
                        "error": f"Execution timed out after {self.timeout}s",
                        "state": {},
                    }
                except Exception as e:
                    log.debug(f"Docker execution failed with exception: {e}")
                    _kill_container(container_name)
                    return {
                        "stdout": "",
                        "output": None,
                        "error": f"Docker execution failed: {e}",
                        "state": {},
                    }
                finally:
                    # Container exited cleanly (--rm handles removal); drop from registry
                    _unregister_container(container_name)

        # If we exhausted retries (too many sub_llm calls)
        return {
            "stdout": "",
            "output": None,
            "error": f"Too many sub_llm calls (max {max_sub_llm_calls})",
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
