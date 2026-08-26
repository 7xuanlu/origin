#!/usr/bin/env bash
# Bash tests for check-daemon.sh. A closed port drives the not-running branch;
# a loopback HTTP stub answering /api/health drives the version-drift branch.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOOK="$SCRIPT_DIR/check-daemon.sh"
tmpbase="$(mktemp -d)"
trap 'rm -rf "$tmpbase"' EXIT

pass=0
fail=0

check() {
    name="$1"; expect="$2"; shift 2
    case "$1" in
        contains)
            case "$2" in *"$expect"*) ok=1 ;; *) ok=0 ;; esac ;;
        missing)
            case "$2" in *"$expect"*) ok=0 ;; *) ok=1 ;; esac ;;
    esac
    if [ "$ok" -eq 1 ]; then
        printf 'PASS  %s\n' "$name"; pass=$((pass + 1))
    else
        printf 'FAIL  %s\n  needle: %s\n  got:    %s\n' "$name" "$expect" "$2" >&2
        fail=$((fail + 1))
    fi
}

stub="$tmpbase/wenlan"
cat > "$stub" <<'EOF'
#!/usr/bin/env bash
if [ "$1" = "--format" ] && [ "$2" = "json" ] && [ "$3" = "outbox" ] && [ "$4" = "status" ]; then
    printf '%s\n' "$OUTBOX_JSON"
    exit 0
fi
exit 1
EOF
chmod +x "$stub"

HEALTH='http://127.0.0.1:1/api/health'

out=$(OUTBOX_JSON='{"queued":2,"failed":1}' WENLAN_HEALTH_URL="$HEALTH" WENLAN_CLI="$stub" bash "$HOOK" 2>&1)
rc=$?
check 'not-running line present' 'local runtime not running' contains "$out"
check 'outbox counts reported' '2 queued handoff write(s), 1 failed' contains "$out"
[ "$rc" -eq 0 ] || { echo "FAIL exit 0 with queued+failed (got $rc)" >&2; fail=$((fail + 1)); }

out=$(OUTBOX_JSON='{"queued":0,"failed":0}' WENLAN_HEALTH_URL="$HEALTH" WENLAN_CLI="$stub" bash "$HOOK" 2>&1)
rc=$?
check 'no outbox line when empty' 'outbox:' missing "$out"
[ "$rc" -eq 0 ] || { echo "FAIL exit 0 with empty outbox (got $rc)" >&2; fail=$((fail + 1)); }

out=$(WENLAN_HEALTH_URL="$HEALTH" WENLAN_CLI='/nonexistent' bash "$HOOK" 2>&1)
rc=$?
check 'no outbox line when CLI missing' 'outbox:' missing "$out"
[ "$rc" -eq 0 ] || { echo "FAIL exit 0 with missing CLI (got $rc)" >&2; fail=$((fail + 1)); }

# Version drift: the stub answers with a chosen daemon version.
www="$tmpbase/www"
mkdir -p "$www/api"
port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
(cd "$www" && exec python3 -m http.server "$port" --bind 127.0.0.1 >/dev/null 2>&1) &
server=$!
disown "$server"
trap 'kill "$server" 2>/dev/null; rm -rf "$tmpbase"' EXIT
for _ in $(seq 1 50); do
    curl -fsS -m 1 "http://127.0.0.1:$port/" >/dev/null 2>&1 && break
    sleep 0.1
done
plugin_root="$tmpbase/plugin"
mkdir -p "$plugin_root/.claude-plugin"

drift_run() { # $1 daemon version, $2 plugin version
    printf '{"status":"ok","version":"%s"}' "$1" > "$www/api/health"
    printf '{"version":"%s"}' "$2" > "$plugin_root/.claude-plugin/plugin.json"
    CLAUDE_PLUGIN_ROOT="$plugin_root" WENLAN_HEALTH_URL="http://127.0.0.1:$port/api/health" \
        WENLAN_CLI='/nonexistent' bash "$HOOK" 2>&1
}

out=$(drift_run '0.17.0+gf240c141' '0.18.0')
check 'published daemon with build metadata still warns on minor drift' \
    'daemon v0.17.0+gf240c141, plugin expects v0.18.0' contains "$out"
out=$(drift_run '0.17.0+gf240c141' '0.17.0')
check 'build metadata alone is not drift' 'plugin expects' missing "$out"
out=$(drift_run '0.17.2' '0.17.0')
check 'patch drift stays quiet' 'plugin expects' missing "$out"
out=$(drift_run '0.16.0+gd1c92ea2' '0.17.0')
check 'source build a minor behind warns' 'plugin expects v0.17.0' contains "$out"

printf '\n%d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
