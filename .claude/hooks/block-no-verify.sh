#!/usr/bin/env bash
# PreToolUse hook on Bash: block git commands using --no-verify or --no-gpg-sign.
# This hook is the policy: hooks must run and signing must not be bypassed.
set -euo pipefail

INPUT="$(cat)"

# Resolve the interpreter from PATH. Git Bash on Windows has no
# /usr/bin/python3, so the hardcoded absolute path made this hook fail closed on
# *every* Bash call there — no agent could run a shell command in this repo on
# Windows. Same python3-then-python probe .githooks/pre-push already uses.
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
else
  echo "🛑 block-no-verify hook: no Python interpreter on PATH — failing closed." >&2
  exit 2
fi

# Fail CLOSED: if extraction fails (malformed JSON, broken interpreter) we
# cannot prove the command is safe — block instead of silently allowing. The old
# `|| true` swallowed the error and let `git commit --no-verify` through.
CMD="$(printf '%s' "$INPUT" | "$PYTHON_BIN" -c 'import sys,json; d=json.load(sys.stdin); print(d.get("tool_input",{}).get("command",""))' 2>/dev/null)" || {
  echo "🛑 block-no-verify hook: could not parse tool input — failing closed." >&2
  exit 2
}

# Only gate actual git invocations — otherwise an innocent grep/heredoc that merely
# mentions "--no-verify" gets blocked.
case "$CMD" in
  *git\ *)
    # Normalize whitespace so `-c commit.gpgsign = false` (spaced =) can't bypass
    # the literal `-c commit.gpgsign=false` match.
    NORM_CMD="$(printf '%s' "$CMD" | tr -d '[:space:]')"
    case "$CMD" in
      *"--no-verify"*|*"--no-gpg-sign"*)
        {
          echo "🛑 Blocked: git command uses --no-verify / --no-gpg-sign / signing-bypass."
          echo "    Command: $CMD"
          echo "    Per project policy hooks must run. Fix the underlying issue."
        } >&2
        exit 2
        ;;
    esac
    case "$NORM_CMD" in
      *"-ccommit.gpgsign=false"*)
        {
          echo "🛑 Blocked: git command uses --no-verify / --no-gpg-sign / signing-bypass."
          echo "    Command: $CMD"
          echo "    Per project policy hooks must run. Fix the underlying issue."
        } >&2
        exit 2
        ;;
    esac
    ;;
esac

exit 0
