#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
git config core.hooksPath .githooks
chmod +x .githooks/*
echo "Git hooks configured. Pre-commit and pre-push hooks active."
echo "  Pre-commit: cargo check + vitest (fast, <30s)"
echo "  Pre-push: full test suite + 90% coverage gate"
echo ""

# Check for optional cargo-llvm-cov
if ! command -v cargo-llvm-cov &> /dev/null && ! cargo llvm-cov --version &> /dev/null 2>&1; then
    echo "NOTE: cargo-llvm-cov not installed. Pre-push will run tests without coverage gate."
    echo "  Install for full coverage enforcement: cargo install cargo-llvm-cov"
fi
