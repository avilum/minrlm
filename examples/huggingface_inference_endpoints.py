"""
Example: Using minRLM with Hugging Face Inference API.

Hugging Face exposes an OpenAI-compatible endpoint at https://router.huggingface.co/v1.
Use a standard OpenAI client pointed at that URL — this gives full compatibility with
minRLM's token tracking, async batching, and sub_llm calls.

Requirements:
    pip install openai
    export HF_TOKEN="hf_..."   # Your Hugging Face API token
"""

import os

from openai import OpenAI

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN:
    raise ValueError("Set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN environment variable")

# Standard OpenAI client pointed at HuggingFace's OpenAI-compatible endpoint
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

MODEL = "openai/gpt-oss-120b"
PROMPT = "How many 'g's in 'huggingface'? return ONLY the number and nothing else."

# 1. Direct call (sanity check)
completion = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": PROMPT}],
)
print("Direct:", completion.choices[0].message.content)

# 2. minRLM with injected client
from minrlm import RLM

rlm = RLM(model=MODEL, client=client, debug=False)
result = rlm.completion(PROMPT)
print("minRLM:", result.response)
print(f"Tokens: {result.total_tokens} | Iterations: {result.iterations}")
