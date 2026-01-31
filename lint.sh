#!/bin/bash
# Lint and format the codebase
# Usage: ./lint.sh [--fix]

set -e

FIX=""
UNSAFE=""
if [[ "$1" == "--fix" ]]; then
    FIX="--fix"
    UNSAFE="--unsafe-fixes"
    echo "🔧 Running in fix mode (with unsafe fixes)..."
else
    echo "🔍 Running in check mode (use --fix to auto-fix)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Ruff (linting)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
uv run ruff check minrlm/ examples/ eval/ $FIX $UNSAFE

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Ruff (formatting)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [[ -n "$FIX" ]]; then
    uv run ruff format minrlm/ examples/ eval/
else
    uv run ruff format --check minrlm/ examples/ eval/
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  MyPy (type checking)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
uv run mypy minrlm/

echo ""
echo "✅ All checks passed!"


