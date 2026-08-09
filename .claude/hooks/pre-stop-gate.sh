#!/usr/bin/env bash
# Stop hook: block agent declaring done if changes contain high-signal unfinished work.
# - Scope drift (diff stat shown for info, not blocking)
# - High-signal unfinished markers (assert!(true), todo!(), unimplemented!(), FIXME)
# Exit 2 + stderr → Claude must address.

set -eo pipefail

REPO="${CLAUDE_PROJECT_DIR:-$(pwd)}"
cd "$REPO"

[ -f "$REPO/Cargo.toml" ] || exit 0

# Collect changed source files (staged + unstaged + untracked, .rs only)
FILES_LIST=$({
  git diff --name-only --diff-filter=ACMR -- '*.rs' 2>/dev/null
  git diff --name-only --cached --diff-filter=ACMR -- '*.rs' 2>/dev/null
  git ls-files --others --exclude-standard -- '*.rs' 2>/dev/null
} | sort -u)

PROBLEMS=()
FILES=()
if [ -n "$FILES_LIST" ]; then
  while IFS= read -r f; do
    [ -n "$f" ] && FILES+=("$f")
  done <<< "$FILES_LIST"
fi

if [ "${#FILES[@]}" -gt 0 ]; then
  # High-signal "fake / unfinished" tells only. Look at NEW lines (added in diff).
  # Deliberately NOT flagged (too many legit uses → false-positive noise):
  #   #[ignore] (87 legit GPU/eval tests), let _ = (189 legit discards),
  #   unreachable!() (exhaustive match arms), #[allow(dead_code)] (test scaffolding).
  PATTERNS=(
    'assert!\s*\(\s*true\s*\)'
    'assert_eq!\s*\(\s*true\s*,\s*true\s*\)'
    'todo!\s*\('
    'unimplemented!\s*\('
    '^\s*//.*FIXME'
  )

  # Added lines from both unstaged and staged changes. Untracked files are counted
  # whole-file below.
  ADDED_LINES="$({
    git diff --unified=0 -- "${FILES[@]}" 2>/dev/null
    git diff --cached --unified=0 -- "${FILES[@]}" 2>/dev/null
  } | grep -E '^\+[^+]' || true)"
  # For untracked .rs files, include all lines
  for f in "${FILES[@]}"; do
    if git ls-files --error-unmatch "$f" >/dev/null 2>&1; then continue; fi
    [ -f "$f" ] && ADDED_LINES+=$'\n'"$(sed 's/^/+ /' "$f")"
  done

  for pat in "${PATTERNS[@]}"; do
    if HIT=$(printf '%s\n' "$ADDED_LINES" | grep -nE "$pat" || true); [ -n "$HIT" ]; then
      PROBLEMS+=("⚠ pattern '$pat' in changes:")
      while IFS= read -r line; do PROBLEMS+=("    $line"); done <<< "$HIT"
    fi
  done
fi

# NOTE: no cargo invocation here by design. A workspace test-compile on every Stop
# starved on the target/ build lock, timed out, and false-blocked. Verification scope
# belongs to the task, explicit repo commands, and CI; this hook is only an instant
# pattern scan.

if [ "${#PROBLEMS[@]}" -gt 0 ]; then
  {
    echo "🛑 pre-stop-gate caught issues — agent must address before stopping:"
    echo
    git diff --stat 2>/dev/null | tail -20
    echo
    for line in "${PROBLEMS[@]}"; do echo "$line"; done
    echo
    echo "Fix all flagged items, then retry."
  } >&2
  exit 2
fi

# Clean stop: append one-line session summary to progress.txt for Wenlan reader.
PROGRESS="$REPO/.claude/progress.txt"
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
HEAD_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo '?')"
CHANGED="$(git diff --shortstat 2>/dev/null | sed 's/^ *//')"
STAGED="$(git diff --cached --shortstat 2>/dev/null | sed 's/^ *//')"
{
  echo "[$TS] branch=$BRANCH head=$HEAD_SHA"
  [ -n "$CHANGED" ] && echo "  unstaged: $CHANGED"
  [ -n "$STAGED" ]  && echo "  staged:   $STAGED"
  echo "  files: ${#FILES[@]} .rs touched, pattern scan clear"
} >> "$PROGRESS"

exit 0
