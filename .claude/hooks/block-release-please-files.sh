#!/usr/bin/env bash
# Block edits to files owned by release-please.
# - CHANGELOG.md (generated)
# - Cargo.toml version lines (marked `# x-release-please-version`)
# Manual edits break the next release PR.

set -euo pipefail

# Hook receives tool input as JSON on stdin.
input="$(cat)"

# Fail CLOSED: a jq failure (missing binary, bad JSON) previously killed the
# script under `set -euo pipefail` with a non-2 exit — which Claude Code
# treats as allow, silently disabling this guard.
file_path="$(echo "$input" | jq -r '.tool_input.file_path // empty')" || {
  echo "BLOCKED: release-please guard could not parse tool input (jq missing?) — failing closed." >&2
  exit 2
}

if [[ -z "$file_path" ]]; then
  exit 0
fi

# Block CHANGELOG.md anywhere
if [[ "$file_path" == */CHANGELOG.md || "$file_path" == CHANGELOG.md ]]; then
  echo "BLOCKED: $file_path is managed by release-please. Update by merging a PR with a conventional commit." >&2
  exit 2
fi

# Block Cargo.toml ONLY if it has release-please version marker
if [[ "$file_path" == */Cargo.toml || "$file_path" == Cargo.toml ]]; then
  # Resolve relative paths against the repo root so a bare `Cargo.toml` or
  # `app/Cargo.toml` can't bypass the marker check when the hook's cwd is
  # not the repo root.
  REPO_DIR="${CLAUDE_PROJECT_DIR:-}"
  if [[ -z "$REPO_DIR" ]]; then
    REPO_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
  fi
  candidates=("$file_path")
  if [[ "$file_path" != /* ]]; then
    candidates+=("$REPO_DIR/$file_path")
  fi
  for cand in "${candidates[@]}"; do
    if [[ -f "$cand" ]] && grep -q 'x-release-please-version' "$cand" 2>/dev/null; then
      echo "BLOCKED: $file_path has release-please-managed version line. Bump via release-please PR, not direct edit." >&2
      exit 2
    fi
  done
fi

exit 0
