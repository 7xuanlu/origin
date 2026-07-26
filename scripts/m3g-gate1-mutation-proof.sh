#!/usr/bin/env bash
# M3g Gate 1 mutation proof (docs/plans/2026-07-25-m3g-gate-criteria.md §1).
#
# Proves the zero-false-grounding gate is NON-VACUOUS: the committed hermetic
# test `edge_grounding::tests::gate1_hermetic_zero_promoted_and_wiring` is green
# only because each protective gate actually rejects its class. We break one gate
# at a time and require the test to go RED; if a mutation leaves it green, the
# gate is not load-bearing and the proof fails loud.
#
#   span gate broken   → class A (fabricated span, hermetic_score 0.9) promotes
#   entailment forced  → classes B/C/D (present-non-entailing / negation /
#                        injection) promote
#   origin gate broken → class N (non-external true, score 0.9) promotes
#
# Driven entirely by this script (no LLM-in-the-loop): each mutation is an exact
# literal string replacement, the file is restored from a byte backup between
# mutations, and RED logs are kept under the gitignored gate-logs dir.
#
# Cargo MUST run outside any sandbox and with sccache disabled — invoke this
# script from an unsandboxed shell; it exports RUSTC_WRAPPER= itself.
#
# Usage:  bash scripts/m3g-gate1-mutation-proof.sh
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
export RUSTC_WRAPPER=

FILE="crates/wenlan-core/src/edge_grounding.rs"
TEST="edge_grounding::tests::gate1_hermetic_zero_promoted_and_wiring"
LOGDIR="docs/superpowers/gate-logs/m3g/mutation-logs"
mkdir -p "$LOGDIR"

BACKUP="$(mktemp)"
cp "$FILE" "$BACKUP"
restore() { cp "$BACKUP" "$FILE"; }
trap restore EXIT

run_test() { # 0 == test passed (green), non-zero == failed (red)
  cargo test -p wenlan-core --lib "$TEST" -- --exact --nocapture
}

echo "== baseline (unmutated) — expect PASS =="
if run_test >"$LOGDIR/baseline.log" 2>&1; then
  echo "  baseline PASS"
else
  echo "  BASELINE FAILED — the gate test is not green before mutation; aborting." >&2
  exit 1
fi

mutate_expect_red() { # name  old_literal  new_literal
  local name="$1" old="$2" new="$3"
  restore
  python3 - "$FILE" "$old" "$new" <<'PY'
import sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(path).read()
n = src.count(old)
assert n == 1, f"mutation anchor not unique ({n} matches): {old!r}"
open(path, "w").write(src.replace(old, new, 1))
PY
  echo "== mutation: $name — expect RED =="
  if run_test >"$LOGDIR/$name.log" 2>&1; then
    echo "  MUTATION '$name' DID NOT REDDEN — Gate 1 is vacuous for this class!" >&2
    restore
    exit 1
  fi
  echo "  mutation '$name' correctly RED (log: $LOGDIR/$name.log)"
  restore
}

# 1. Break the deterministic span gate → class A (fabricated span) promotes.
mutate_expect_red span_gate \
  'if !content.contains(quote) {' \
  'if false && !content.contains(quote) {'

# 2. Force the entailment verdict to auto-pass → classes B/C/D promote.
mutate_expect_red entailment \
  'let score = parse_entailment(&raw).unwrap_or(0.0);' \
  'let score = parse_entailment(&raw).unwrap_or(0.0).max(1.0);'

# 3. Break the external-origin gate → class N (non-external true) promotes.
mutate_expect_red origin_gate \
  'if cand.mem_source_agent.as_deref() != Some(EXTERNAL_SOURCE_AGENT) {' \
  'if false && cand.mem_source_agent.as_deref() != Some(EXTERNAL_SOURCE_AGENT) {'

restore
echo
echo "ALL MUTATIONS REDDENED — Gate 1 is non-vacuous (span, entailment, and origin"
echo "gates each independently reject their false-grounding class)."
