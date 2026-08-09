#!/usr/bin/env bash
set -euo pipefail

bash -n .githooks/pre-commit
bash -n .githooks/pre-push

grep -Fq 'cargo clippy $TOUCHED_CRATES -- -D warnings' .githooks/pre-commit
grep -Fq 'cargo metadata --format-version 1 --locked --no-deps' .githooks/pre-commit
if grep -Fq 'cargo check --workspace' .githooks/pre-commit; then
  echo 'pre-commit must not compile the complete workspace for ownerless inputs' >&2
  exit 1
fi

grep -Fq 'scripts/ci_test_plan.py local' .githooks/pre-push
if grep -Eq 'cargo (check|clippy|test) --workspace' .githooks/pre-push; then
  echo 'pre-push must delegate changed-owner routing to the fail-closed planner' >&2
  exit 1
fi

echo 'git hook routing contracts: PASS'
