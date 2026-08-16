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

grep -Fq 'scripts/m5-reader-sweep.py --update-inventory' .githooks/pre-commit
grep -Fq 'git add "$INVENTORY"' .githooks/pre-commit
grep -Fq 'scripts/m5-reader-sweep.py --check' .githooks/pre-push
grep -Fq 'lint::serving::tests::review_tests::route_catalog_freezes_exact_global_and_scoped_keys' .githooks/pre-push
grep -Fq '1 passed' .githooks/pre-push

fast_gate_line=$(grep -n 'scripts/m5-reader-sweep.py --check' .githooks/pre-push | head -n 1 | cut -d: -f1)
planner_line=$(grep -n 'scripts/ci_test_plan.py local' .githooks/pre-push | head -n 1 | cut -d: -f1)
if [ "$fast_gate_line" -ge "$planner_line" ]; then
  echo 'pre-push must run the fast drift gates before the ci_test_plan.py planner' >&2
  exit 1
fi

grep -Fq 'scripts/ci_test_plan.py local' .githooks/pre-push
if grep -Eq 'cargo (check|clippy|test) --workspace' .githooks/pre-push; then
  echo 'pre-push must delegate changed-owner routing to the fail-closed planner' >&2
  exit 1
fi

echo 'git hook routing contracts: PASS'
