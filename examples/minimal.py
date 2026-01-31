#!/usr/bin/env python3
"""Minimal RLM example comparing vanilla LLM vs RLM."""

import os

from openai import OpenAI

from minrlm import RLM

verbose = os.environ.get("MINRLM_VERBOSE", "0") == "1"
model = os.environ.get("MINRLM_MODEL", "gpt-5-nano")
task = "What is 2^9419?"


def on_step(event: str, data: dict):
    if event == "executing":
        code = data["code"].strip().replace("\n", "\n         ")
        print(f"       ├─ Code: {code[:150]}")
    elif event == "executed":
        if data.get("error"):
            print(f"       └─ ❌ Error: {data['error']}")
        elif data.get("output"):
            print(f"       └─ ✓ Result: {data['output']}")
        elif data.get("stdout"):
            print(f"       └─ stdout: {data['stdout'][:80].strip()}")


print(f"\n  Task: {task}\n  Model: {model}\n")

# ─────────────────────────────────────────────────────
# Vanilla LLM
# ─────────────────────────────────────────────────────
print("  ┌─ Vanilla LLM (direct API)")
client = OpenAI()
resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": task}])
answer = resp.choices[0].message.content.strip()
tokens = resp.usage.total_tokens if resp.usage else 0
for line in answer.split("\n"):
    print(f"  │  {line}")
print(f"  └─ {tokens} tokens\n")

# ─────────────────────────────────────────────────────
# RLM
# ─────────────────────────────────────────────────────
print("  ┌─ minRLM (code execution)")
rlm = RLM(model=model, on_step=on_step if verbose else None)
result = rlm.completion(task)
print(f"  │  Answer: {result.response}")
print(f"  └─ {result.total_tokens} tokens ({result.iterations} iteration{'s' if result.iterations != 1 else ''})\n")

# ─────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────
if tokens > 0 and result.total_tokens > 0:
    if tokens > result.total_tokens:
        ratio = tokens / result.total_tokens
        print(f"  ⚡ RLM used {ratio:.1f}x fewer tokens\n")
    else:
        ratio = result.total_tokens / tokens
        print(f"  → Vanilla used {ratio:.1f}x fewer tokens (simple task)\n")
