#!/bin/bash
# Lint and format the codebase
# Usage: ./lint.sh [--fix]

set -e

DIRS="minrlm/ examples/ eval/"
MYPY_DIRS="minrlm/"  # Only type-check the library (examples/eval have optional deps)

FIX=""
UNSAFE=""
if [[ "$1" == "--fix" ]]; then
    FIX="--fix"
    UNSAFE="--unsafe-fixes"
    echo "Running in fix mode (with unsafe fixes)..."
else
    echo "Running in check mode (use --fix to auto-fix)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Ruff (linting)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
uvx ruff check $DIRS $FIX $UNSAFE

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Ruff (formatting)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [[ -n "$FIX" ]]; then
    uvx ruff format $DIRS
else
    uvx ruff format --check $DIRS
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  MyPy (type checking)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
uv run mypy $MYPY_DIRS

echo ""
echo "All checks passed!"
