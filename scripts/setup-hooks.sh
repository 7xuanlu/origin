#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
git config core.hooksPath .githooks
chmod +x .githooks/*
echo "Git hooks configured. Pre-commit and pre-push hooks active."
echo "  Pre-commit: formatting check + clippy on directly changed crates (fast)"
echo "  Pre-push:   clippy + tests for the affected reverse-dependency closure"
