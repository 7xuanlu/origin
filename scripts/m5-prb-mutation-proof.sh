#!/usr/bin/env bash
# M5 PR-B stage 2 mutation proof
# (docs/plans/2026-07-27-m5-reader-manifest.md §7, §9; inventory teeth 8 + 13).
#
# Proves the exposure-contract tests are NON-VACUOUS. Every weakening §9 names
# for the negotiation, the intent marker, the refusal, and the audit row is
# applied one at a time, and the matching test must go RED. A mutation that
# leaves the suite green means that tooth is decoration, and this script fails
# loud rather than letting a process promise stand in for a failing build.
#
#   contract accepts any version   → trust contracts negotiate upward
#   marker check dropped           → a declared client's poll sees provisional
#   refusal downgraded to ignore   → /api/context accepts a marker
#   contract requirement dropped   → an undeclared client's marker grants
#   marker treated as blanket      → a marker naming no page grants anyway
#   grant covers neighbours        → page A's grant opens page B
#   cutover gate removed           → PR-B is a live cutover, not an inert one
#   guard ignores the marker       → the wire gate is not wired
#   audit write removed            → the only compensating control is a sentence
#   page IDs not taken from path   → the audit records a walk it cannot see
#
# Driven entirely by this script (no LLM-in-the-loop): each mutation is an exact
# literal replacement asserted unique, the file is restored from a byte backup
# between mutations, and RED logs land in the gitignored gate-logs dir.
#
# Cargo MUST run outside any sandbox; the script exports RUSTC_WRAPPER= itself.
#
# Usage:  bash scripts/m5-prb-mutation-proof.sh
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
export RUSTC_WRAPPER=

CORE="crates/wenlan-core/src/truth_contract.rs"
GUARD="crates/wenlan-server/src/truth_guard.rs"
LOGDIR="docs/superpowers/gate-logs/m5/prb-mutation-logs"
mkdir -p "$LOGDIR"

# Explicit template: a bare `mktemp` resolves under /var/folders, which the
# macOS agent sandbox denies.
CORE_BACKUP="$(mktemp "${TMPDIR:-/tmp}/m5prb-core.XXXXXX")"; cp "$CORE" "$CORE_BACKUP"
GUARD_BACKUP="$(mktemp "${TMPDIR:-/tmp}/m5prb-guard.XXXXXX")"; cp "$GUARD" "$GUARD_BACKUP"
restore() { cp "$CORE_BACKUP" "$CORE"; cp "$GUARD_BACKUP" "$GUARD"; }
trap restore EXIT

run_suite() { # 0 == green
  cargo test -p wenlan-core --lib truth_contract \
    && cargo test -p wenlan-core --lib truth_exposure \
    && cargo test -p wenlan-server --lib truth_guard
}

echo "== baseline (unmutated) — expect PASS =="
if run_suite >"$LOGDIR/baseline.log" 2>&1; then
  echo "  baseline PASS"
else
  echo "  BASELINE FAILED — the suite is not green before mutation; aborting." >&2
  exit 1
fi

mutate_expect_red() { # name  file  old_literal  new_literal
  local name="$1" file="$2" old="$3" new="$4"
  restore
  python3 - "$file" "$old" "$new" <<'PY'
import sys
path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(path).read()
n = src.count(old)
assert n == 1, f"mutation anchor not unique ({n} matches): {old!r}"
open(path, "w").write(src.replace(old, new, 1))
PY
  echo "== mutation: $name — expect RED =="
  if run_suite >"$LOGDIR/$name.log" 2>&1; then
    echo "  MUTATION '$name' DID NOT REDDEN — that tooth is vacuous!" >&2
    restore
    exit 1
  fi
  echo "  mutation '$name' correctly RED (log: $LOGDIR/$name.log)"
  restore
}

# --- negotiation (§6) ------------------------------------------------------

# 1. Treat any parseable version as supported — trust contracts negotiating up.
mutate_expect_red contract_accepts_any_version "$CORE" \
  '        Some(1) => ClientContract::V1,' \
  '        Some(_) => ClientContract::V1,'

# --- the per-call marker (§4, §7) ------------------------------------------

# 2. Stop distinguishing a human click from a 10-second poll.
mutate_expect_red marker_check_dropped "$CORE" \
  '    if !marker {
        return TruthDecision::Grant(TruthGrant::Automatic);
    }' \
  '    if false && !marker {
        return TruthDecision::Grant(TruthGrant::Automatic);
    }'

# 3. Ignore a marker at a `none` route instead of refusing it.
mutate_expect_red refusal_downgraded_to_ignore "$CORE" \
  '        return TruthDecision::Refuse;' \
  '        return TruthDecision::Grant(TruthGrant::Automatic);'

# 4. Let an undeclared client's marker grant anyway.
mutate_expect_red contract_requirement_dropped "$CORE" \
  '    if contract == ClientContract::Legacy {
        return TruthDecision::Grant(TruthGrant::Automatic);
    }' \
  '    if false && contract == ClientContract::Legacy {
        return TruthDecision::Grant(TruthGrant::Automatic);
    }'

# 5. Make the marker a blanket header — grant even when it names no page.
mutate_expect_red marker_as_blanket_header "$CORE" \
  '        if named_pages.is_empty() {
            // A marker that names no page grants nothing. This is what stops the' \
  '        if false && named_pages.is_empty() {
            // A marker that names no page grants nothing. This is what stops the'

# 6. Let a grant for page A cover page B riding along in A'"'"'s payload.
mutate_expect_red grant_covers_embedded_pages "$CORE" \
  '            if ids.iter().any(|id| id == page_id) {' \
  '            if true || ids.iter().any(|id| id == page_id) {'

# --- inertness (§8) --------------------------------------------------------

# 7. Remove the cutover gate — PR-B becomes a live cutover.
mutate_expect_red cutover_gate_removed "$CORE" \
  '    if generation == 0 {
        return Visibility::Full;
    }' \
  '    if false && generation == 0 {
        return Visibility::Full;
    }'

# --- the wire (inventory teeth 8 + 13) -------------------------------------

# 8. The guard stops seeing the marker at all.
mutate_expect_red guard_ignores_the_marker "$GUARD" \
  '    let marker = truth_contract::marker_present(header(&req, INTENT_HEADER).as_deref());' \
  '    let marker = false && truth_contract::marker_present(header(&req, INTENT_HEADER).as_deref());'

# 9. Serve the request instead of refusing it, at the wire this time.
mutate_expect_red wire_refusal_downgraded "$GUARD" \
  '            refusal(&method_str, &path)' \
  '            run_with(req, next, TruthView::automatic()).await'

# 10. Drop the audit write. The conceded composition attack goes invisible.
mutate_expect_red audit_write_removed "$GUARD" \
  '    let db = { guard_state.state.read().await.db.clone() };' \
  '    let db: Option<std::sync::Arc<wenlan_core::db::MemoryDB>> = None;'

# 11. Stop taking the named page IDs from the path.
mutate_expect_red page_ids_not_read_from_the_path "$GUARD" \
  'const PAGE_ID_PARAMS: &[&str] = &["id", "page_id"];' \
  'const PAGE_ID_PARAMS: &[&str] = &[];'

restore
echo
echo "ALL MUTATIONS RED — the M5 PR-B stage 2 teeth are load-bearing."
