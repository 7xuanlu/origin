#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

echo "Generating Rust coverage report..."
(cd app && cargo llvm-cov --html)
echo "  -> app/target/llvm-cov/html/index.html"

echo ""
echo "Generating frontend coverage report..."
pnpm vitest run --coverage
echo "  -> coverage/index.html"

# Open reports on macOS
if command -v open &> /dev/null; then
    open app/target/llvm-cov/html/index.html
    open coverage/index.html
fi
