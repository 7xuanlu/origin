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

# WENLAN_PUSH_FULL must guard both compiling steps: the route-catalog cargo
# test and the ci_test_plan.py planner. Everything before the guard line must
# stay non-compiling.
guard_line=$(grep -n '"\${WENLAN_PUSH_FULL:-}"' .githooks/pre-push | head -n 1 | cut -d: -f1)
[ -n "$guard_line" ] || { echo 'pre-push must gate the heavy section behind WENLAN_PUSH_FULL' >&2; exit 1; }
cargo_line=$(grep -n 'cargo ' .githooks/pre-push | head -n 1 | cut -d: -f1)
if [ -n "$cargo_line" ] && [ "$guard_line" -ge "$cargo_line" ]; then
  echo 'pre-push must gate every cargo invocation behind WENLAN_PUSH_FULL' >&2
  exit 1
fi
if [ "$guard_line" -ge "$planner_line" ]; then
  echo 'pre-push must gate the ci_test_plan.py planner behind WENLAN_PUSH_FULL' >&2
  exit 1
fi

# Rebase-safe base: the merge-base fix must be present and the old bare
# assignment to the remote's stale tip must not have regressed back in.
grep -Fq 'merge-base' .githooks/pre-push
if grep -Fq 'base="$remote_sha"' .githooks/pre-push; then
  echo 'pre-push must not fall back to a bare base="$remote_sha" assignment' >&2
  exit 1
fi

echo 'git hook routing contracts: PASS'
