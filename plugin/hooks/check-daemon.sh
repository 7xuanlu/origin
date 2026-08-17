#!/usr/bin/env bash
# SessionStart hook: probe the local Wenlan daemon and surface two issues:
#   1. Daemon not running → point user at /wenlan:setup (it auto-installs).
#   2. Daemon version mismatches the plugin manifest → point user at
#      /wenlan:setup (it upgrades/restarts and verifies the runtime).
# Hook never blocks (always exit 0) and never prints command soup.
set -u

URL="${WENLAN_HEALTH_URL:-http://127.0.0.1:7878/api/health}"
W="${WENLAN_CLI:-$(command -v wenlan || echo "$HOME/.wenlan/bin/wenlan")}"
PLUGIN_JSON="${CLAUDE_PLUGIN_ROOT:-}/.claude-plugin/plugin.json"

# Print a one-line outbox report when queued or failed writes exist. Silent
# on any error (old CLI without the subcommand, no python3, bad JSON).
report_outbox() {
  [ -x "$W" ] || return 0
  command -v python3 >/dev/null 2>&1 || return 0
  status=$("$W" --format json outbox status 2>/dev/null) || return 0
  printf '%s' "$status" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    q, f = d.get("queued", 0), d.get("failed", 0)
    if q > 0 or f > 0:
        print(f"[wenlan] outbox: {q} queued handoff write(s), {f} failed — run \`wenlan outbox status\`.")
except Exception:
    pass
' 2>/dev/null
}

RESP=""
for i in 1 2 3; do
  RESP=$(curl -fsS -m 3 "$URL" 2>/dev/null) && break
  sleep 1
done

if [ -z "$RESP" ]; then
  cat <<MSG
[wenlan] local runtime not running. Handoff writes will queue in the outbox; run /wenlan:setup or \`wenlan background on\`.
MSG
  report_outbox
  exit 0
fi

report_outbox

# Compare daemon version vs plugin manifest version. Silent unless mismatch.
[ -r "$PLUGIN_JSON" ] || exit 0
command -v python3 >/dev/null 2>&1 || exit 0  # fail closed without python3

extract_version() {
  python3 -c 'import json,sys; print(json.load(sys.stdin).get("version",""))' 2>/dev/null
}

DAEMON_VER=$(printf '%s' "$RESP" | extract_version)
EXPECTED_VER=$(extract_version <"$PLUGIN_JSON")

# A dev daemon reports a `+g<sha>` build-metadata suffix (local source build, via
# build.rs). Its release-granular version is stale by construction, so a drift
# arrow would be pure noise — stay quiet.
case "$DAEMON_VER" in
  *+g*) exit 0 ;;
esac

# Compare only major.minor: the daemon and plugin ride one release train, so a
# patch drift (e.g. 0.13.1 vs 0.13.2) is compatible and must NOT nag every
# session. Only a minor/major gap is a real, actionable drift worth surfacing.
mm() { printf '%s' "$1" | cut -d. -f1,2; }
DAEMON_MM=$(mm "$DAEMON_VER")
EXPECTED_MM=$(mm "$EXPECTED_VER")

if [ -n "$DAEMON_MM" ] && [ -n "$EXPECTED_MM" ] && [ "$DAEMON_MM" != "$EXPECTED_MM" ]; then
  cat <<MSG
[wenlan] daemon v${DAEMON_VER}, plugin expects v${EXPECTED_VER}.
  Run /wenlan:setup to repair. It will upgrade/restart the runtime and verify MCP.
MSG
fi

exit 0
