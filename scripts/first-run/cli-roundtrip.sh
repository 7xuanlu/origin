#!/usr/bin/env bash
# First-run gauntlet: the `wenlan` CLI round-trip against a running daemon —
# status, capture a sentinel, memories lists it, search finds it (polled up
# to 60s). Lifted from scripts/smoke-cli.sh; every step is recorded through
# lib.sh and the script never exits early, so one broken step still lets the
# later ones report. Always exits 0 — the channel script calls `evaluate`.
#
# Env:
#   WENLAN_BIN    path to the wenlan binary (required)
#   WENLAN_HOST   daemon URL; leave unset to exercise the CLI's own default
#   GAUNTLET_OUT / GAUNTLET_CHANNEL   see lib.sh
set -uo pipefail
# shellcheck source=scripts/first-run/lib.sh
. "$(dirname "$0")/lib.sh"

# Never let a connect failure start the developer's registered daemon; the
# gauntlet boots its own. Overridable for a channel that tests autostart.
export WENLAN_NO_AUTOSTART="${WENLAN_NO_AUTOSTART:-1}"
# An empty WENLAN_HOST would be passed to the CLI as a real (invalid) value.
if [ -z "${WENLAN_HOST:-}" ]; then
    unset WENLAN_HOST
else
    export WENLAN_HOST
fi

if [ -z "${WENLAN_BIN:-}" ]; then
    check cli-status -- bash -c 'echo "WENLAN_BIN (path to the wenlan binary) is required"; exit 2'
    exit 0
fi

SENTINEL="kumquat-lighthouse-8231"
# Commands are passed to `check` as argv (not shell functions) because lib.sh
# wraps them in `timeout` where it exists, and timeout cannot exec a function.
CLI=("$WENLAN_BIN" --format json)

echo "==> wenlan status"
# `status --format json` prints {"status":"unreachable"} with exit 0 when the
# daemon is down, so require the health payload's version field, not just rc=0.
check_output cli-status '"version"' -- "${CLI[@]}" status

echo "==> wenlan capture (sentinel)"
check cli-capture -- "${CLI[@]}" capture \
    "The ${SENTINEL} sentinel sentence lives in the CLI smoke." --type fact

echo "==> wenlan memories contains the sentinel"
check_output cli-memories "$SENTINEL" -- "${CLI[@]}" memories --limit 20

echo "==> wenlan search finds the sentinel"
# Poll first (embedding/indexing is async), then record one final search so
# the check log holds the output that actually matched — or the last miss.
hit=""
for i in $(seq 1 30); do
    SEARCH_OUT="$("${CLI[@]}" search "kumquat lighthouse sentinel sentence" --limit 5 2>&1)" || break
    case "$SEARCH_OUT" in
        *"$SENTINEL"*)
            echo "    hit after ${i} poll(s)"
            hit=1
            break
            ;;
    esac
    sleep 2
done
[ -n "$hit" ] || echo "    sentinel not retrievable via wenlan search within 60s"
check_output cli-search "$SENTINEL" -- "${CLI[@]}" search "kumquat lighthouse sentinel sentence" --limit 5

exit 0
