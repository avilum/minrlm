---
name: use-minrlm
description: Delegate large-data tasks to minRLM via `uvx minrlm` CLI. Use when working with large files (logs, CSV, JSON, code repos, documents) that are too big to fit in context, or when asked to search/aggregate/extract from large data. Also use when the user explicitly asks to use RLM or minRLM. Requires `uvx` or `uv` to be available.
---

# use-minrlm

When a task involves large data (files, logs, datasets, documents) that would consume significant context, delegate to `uvx minrlm` instead of reading the file into context.

## When to use

- File is **large** (>50KB, or thousands of lines)
- Task is search, count, filter, aggregate, extract, or summarize over data
- File would eat most of your context window if read directly
- User explicitly asks to use RLM/minRLM

## When NOT to use

- File is small (<8K tokens) and fits comfortably in context
- Task requires editing the file (minRLM is read-only analysis)
- Task needs third-party packages (minRLM sandbox is stdlib-only)

## How to use

### Check availability first

```bash
which uvx || which uv
```

If neither is available, fall back to your normal approach.

### Basic patterns

```bash
# Task + file
uvx minrlm "How many ERROR lines in the last hour?" ./server.log

# Task + large file, show code and stats
uvx minrlm -sv "Which user had the most errors?" ./app.log

# Pipe data from stdin
cat huge_dataset.csv | uvx minrlm "Which product had the highest return rate?"

# Multiple files via cat
cat src/*.py | uvx minrlm "Find all functions that call the database"

# Just computation (no file)
uvx minrlm "What is the sum of all primes up to 1,000,000?"
```

### Flags

| Flag | What it does |
|------|--------------|
| `-s` | Show generated Python code (steps) |
| `-v` | Verbose: show token count, cost, timing |
| `-sv` | Both |

### Workflow

1. **Try minRLM first** for the large-data task
2. **Read the output** - it returns the answer directly
3. **If it fails or returns wrong result**, fall back to reading the file into context (or a portion of it) and handling it yourself
4. **Use `-sv` flags** if you need to debug what code was generated

### Examples of good delegation

```bash
# Log analysis (100MB log)
uvx minrlm "Count requests per endpoint in the last 24 hours" ./nginx.log

# CSV analysis (50K rows)
uvx minrlm "What is the average salary by department?" ./employees.csv

# JSON extraction (large API response)
uvx minrlm "Extract all user emails where status is active" ./users.json

# Code search (large repo dump)
cat src/**/*.py | uvx minrlm "Find all REST API endpoint definitions"

# Document search
uvx minrlm "What are the key terms of the indemnification clause?" ./contract.txt
```

## How it works

minRLM stores the data in a Python REPL variable (`input_0`). The model writes Python code to query it - search, regex, parse, count, aggregate. The data never enters the LLM prompt. Token cost is flat regardless of file size.

A 10MB file costs roughly the same as a 10KB file to process.

## Fallback

If `uvx` is not available or minRLM fails:
1. Read only the relevant portion of the file (head, tail, or grep for relevant lines)
2. Process in chunks if needed
3. Use standard tools (grep, awk, jq) for simple queries
