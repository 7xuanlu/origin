#!/usr/bin/env bash
# Hook canary — verify-the-verifier for the repo's Claude Code gates.
# Feeds synthetic tool input or repository fixtures through each protection hook and asserts
# BOTH directions: the block path fires (exit 2) and the allow path stays
# quiet (exit 0). A gate nobody has ever seen fire is indistinguishable from
# a dead one — this is the durable evidence (audit 2026-07-02, PR #324).
# Run: bash .claude/hooks/test-hooks.sh   (the weekly harness retro runs it too)
set -u
cd "$(dirname "$0")"
fails=0
t() { # desc script stdin want-exit
  local got
  printf '%s' "$3" | bash "$2" >/dev/null 2>&1
  got=$?
  if [ "$got" -eq "$4" ]; then
    echo "PASS  $1"
  else
    echo "FAIL  $1 (want exit $4, got $got)"
    fails=$((fails + 1))
  fi
}

t "no-verify: block --no-verify"           block-no-verify.sh '{"tool_input":{"command":"git commit --no-verify -m x"}}' 2
t "no-verify: allow plain git"             block-no-verify.sh '{"tool_input":{"command":"git status"}}' 0
t "no-verify: allow innocent mention"      block-no-verify.sh '{"tool_input":{"command":"grep -- --no-verify README.md"}}' 0
t "no-verify: malformed JSON fails closed" block-no-verify.sh 'not json' 2
t "release-please: block CHANGELOG.md"     block-release-please-files.sh '{"tool_input":{"file_path":"/x/CHANGELOG.md"}}' 2
t "release-please: allow normal file"      block-release-please-files.sh '{"tool_input":{"file_path":"/x/src/main.rs"}}' 0
t "release-please: allow no file_path"     block-release-please-files.sh '{"tool_input":{}}' 0
t "release-please: malformed JSON fails closed" block-release-please-files.sh '{{{' 2

stop_t() { # desc fixture want-exit
  local tmp got filename
  tmp=$(mktemp -d "${TMPDIR:-/tmp}/wenlan-stop-hook.XXXXXX")
  mkdir -p "$tmp/.claude"
  printf '[workspace]\n' > "$tmp/Cargo.toml"
  git -C "$tmp" init -q
  filename=""
  case "$2" in
    rust-todo)
      filename=canary.rs
      printf 'fn main() { todo!(); }\n' > "$tmp/$filename"
      ;;
    ts-skip)
      filename=canary.ts
      printf 'it.skip("pending", () => {});\n' > "$tmp/$filename"
      ;;
    clean)
      ;;
  esac
  if [ -n "$filename" ]; then
    git -C "$tmp" add -- "$filename"
  fi
  CLAUDE_PROJECT_DIR="$tmp" bash "$PWD/pre-stop-gate.sh" >/dev/null 2>&1
  got=$?
  rm -rf "$tmp"
  if [ "$got" -eq "$3" ]; then
    echo "PASS  $1"
  else
    echo "FAIL  $1 (want exit $3, got $got)"
    fails=$((fails + 1))
  fi
}

stop_t "pre-stop: block staged todo!" rust-todo 2
stop_t "pre-stop: block staged TS skip" ts-skip 2
stop_t "pre-stop: allow clean tree" clean 0

# ---- wrapper-level cases: exercise the REAL settings.json command strings, not
# just the inner scripts above. These cover the cwd-resolution fallback chain
# (CLAUDE_PROJECT_DIR env -> git toplevel of the event .cwd -> fail closed inside
# a repo / allow outside any repo) that the inner-script tests never touch.
ROOT="$(cd ../.. && pwd)"
SETTINGS="$ROOT/.claude/settings.json"
W_BASH="$(jq -r '.hooks.PreToolUse[] | select(.matcher=="Bash") | .hooks[0].command' "$SETTINGS")"
W_WRITE="$(jq -r '.hooks.PreToolUse[] | select(.matcher=="Edit|Write|MultiEdit") | .hooks[0].command' "$SETTINGS")"
W_STOP="$(jq -r '.hooks.Stop[0].hooks[0].command' "$SETTINGS")"

wt() { # desc wrapper-cmd stdin-json CLAUDE_PROJECT_DIR-or-UNSET want-exit [want-stderr-substr]
  local desc="$1" cmd="$2" json="$3" pd="$4" want="$5" substr="${6:-}"
  local got err
  if [ "$pd" = "UNSET" ]; then
    err="$(printf '%s' "$json" | env -u CLAUDE_PROJECT_DIR bash -c "$cmd" 2>&1 >/dev/null)"
  else
    err="$(printf '%s' "$json" | env CLAUDE_PROJECT_DIR="$pd" bash -c "$cmd" 2>&1 >/dev/null)"
  fi
  got=$?
  if [ "$got" -ne "$want" ]; then
    echo "FAIL  $desc (want exit $want, got $got)"
    fails=$((fails + 1))
  elif [ -n "$substr" ] && [[ "$err" != *"$substr"* ]]; then
    echo "FAIL  $desc (exit $got ok, but stderr missing '$substr': $err)"
    fails=$((fails + 1))
  else
    echo "PASS  $desc"
  fi
}

D="$(mktemp -d "${TMPDIR:-/tmp}/wenlan-wrap.XXXXXX")"
J="$(jq -n --arg cwd "$D" '{cwd:$cwd, tool_input:{command:"git commit --no-verify -m x"}}')"
wt "wrapper bash: CLAUDE_PROJECT_DIR resolves, inner blocks --no-verify" "$W_BASH" "$J" "$ROOT" 2
rm -rf "$D"

J="$(jq -n --arg cwd "$ROOT" '{cwd:$cwd, tool_input:{command:"echo hi"}}')"
wt "wrapper bash: env unset, resolves via git toplevel of cwd" "$W_BASH" "$J" "UNSET" 0

# Outside any repo: nothing to guard, so allow even a command the inner script
# would otherwise block (--no-verify). This is the intended new behavior.
D="$(mktemp -d "${TMPDIR:-/tmp}/wenlan-wrap.XXXXXX")"
J="$(jq -n --arg cwd "$D" '{cwd:$cwd, tool_input:{command:"git commit --no-verify -m x"}}')"
wt "wrapper bash: cwd outside any git repo -> allow" "$W_BASH" "$J" "/nonexistent/path" 0 "nothing to guard"
rm -rf "$D"

# Inside a repo that's just missing the script: still fail closed.
D="$(mktemp -d "${TMPDIR:-/tmp}/wenlan-wrap.XXXXXX")"
git init -q "$D"
J="$(jq -n --arg cwd "$D" '{cwd:$cwd, tool_input:{command:"git commit --no-verify -m x"}}')"
wt "wrapper bash: cwd inside a repo missing the script -> fail closed" "$W_BASH" "$J" "/nonexistent/path" 2 "inside a repo"
rm -rf "$D"

D="$(mktemp -d "${TMPDIR:-/tmp}/wenlan-wrap.XXXXXX")"
J="$(jq -n --arg cwd "$D" '{cwd:$cwd, tool_input:{file_path:"CHANGELOG.md"}}')"
wt "wrapper write: cwd outside any git repo -> allow" "$W_WRITE" "$J" "/nonexistent" 0
rm -rf "$D"

J="$(jq -n --arg cwd "$ROOT" --arg fp "$ROOT/CHANGELOG.md" '{cwd:$cwd, tool_input:{file_path:$fp}}')"
wt "wrapper write: CLAUDE_PROJECT_DIR resolves, inner blocks CHANGELOG.md" "$W_WRITE" "$J" "$ROOT" 2

D="$(mktemp -d "${TMPDIR:-/tmp}/wenlan-wrap.XXXXXX")"
J="$(jq -n --arg cwd "$D" '{cwd:$cwd}')"
wt "wrapper stop: cwd outside any git repo -> allow" "$W_STOP" "$J" "/nonexistent" 0
rm -rf "$D"

echo "----"
if [ "$fails" -eq 0 ]; then
  echo "hook canary: ALL PASS"
else
  echo "hook canary: $fails FAILURE(S)"
  exit 1
fi
