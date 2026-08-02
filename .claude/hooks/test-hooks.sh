#!/usr/bin/env bash
# Hook canary — verify-the-verifier for the repo's Claude Code gates.
# Feeds synthetic tool-input JSON through each protection hook and asserts
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
echo "----"
if [ "$fails" -eq 0 ]; then
  echo "hook canary: ALL PASS"
else
  echo "hook canary: $fails FAILURE(S)"
  exit 1
fi
