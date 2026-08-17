#!/usr/bin/env bash
# Bash tests for check-daemon.sh outbox reporting. No network reachable: a
# closed port always drives the not-running branch.

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

printf '\n%d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
