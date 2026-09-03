#!/usr/bin/env bash
# Record whether the shared production port is free -- as a MEASUREMENT.
#
# Every channel ran its own copy of this, and both copies turned a probe that
# could not run into "free":
#
#   POSIX:   listeners="$(lsof -nP -iTCP:7878 -sTCP:LISTEN 2>/dev/null || true)"
#   Windows: $listeners = Get-NetTCPConnection -LocalPort 7878 -State Listen `
#                           -ErrorAction SilentlyContinue
#
# A denied /proc, a busybox lsof, a WinRM hiccup: empty result, "free", and the
# channel then installs over a daemon it never started and passes every health
# and CLI check against it. The free row was also INFO, so it counted for
# nothing in the ledger and no channel declared it -- the precheck could be
# deleted outright and every run would stay green.
#
# One script for every channel now, POSIX and Windows, over the tri-state probe
# in scripts/lib/host-process.sh. "Could not measure" is recorded as FAIL,
# because a port that cannot be measured is not a port known to be free.
#
# THE ROW MUST SAY WHICH RUN TOOK THE MEASUREMENT. This row is written BEFORE
# the channel script starts, so it lands above that script's window mark and
# lib.ps1's Record-CarriedRow finds it in the CARRIED region and restates its
# verdict as one of this run's rows. GAUNTLET_OUT is reused -- the workflow
# header below tells a human to keep one $PWD/gauntlet-out and run channel after
# channel into it, and a re-run of a leg writes into the same artifact directory
# again -- so the carried region is the union of every earlier run into that
# directory. Position alone therefore says "before the mark", never "for this
# run": ONE PASS row left behind by an earlier run is exactly one hit, and was
# carried into every later run of the same channel. No precheck ran that time,
# the port may since have gone busy, and the channel recorded a PASS anyway.
#
# So the detail column carries `run=$GAUNTLET_RUN_TOKEN`, and Record-CarriedRow
# requires it to equal the token the channel process reads from the same
# variable. An UNSET token is not a licence to skip the binding: a row nothing
# downstream can attribute to a run is unusable evidence, so it FAILS here
# rather than passing as one more anonymous PASS for the next run to inherit.
# The port is not probed in that state, because there is nothing a later reader
# could do with the answer.
#
# Usage: bash scripts/first-run/port-precheck.sh [port]
#   GAUNTLET_OUT and GAUNTLET_CHANNEL must be set, as they are for the channel.
#   GAUNTLET_RUN_TOKEN must be set to a value unique to this run, and the
#   channel step that follows must read the SAME value -- in the workflow it is
#   a job-level env var so both steps see one variable. No whitespace and no
#   tab in it: it travels in an unquoted TSV column and is read back as one
#   whitespace-delimited word, so a token with a space in it is compared as its
#   first word and the carried row is refused as a different run's.
set -uo pipefail

port="${1:-7878}"
: "${GAUNTLET_OUT:?GAUNTLET_OUT must be set}"
: "${GAUNTLET_CHANNEL:?GAUNTLET_CHANNEL must be set}"
# Not `:?`. A missing token is recorded as a FAIL ROW, not as an abort: an abort
# leaves no row at all, and a channel that declares `port-7878-precheck-carried`
# would then report the absence with the wrong diagnosis.
run_token="${GAUNTLET_RUN_TOKEN:-}"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/host-process.sh
. "$here/../lib/host-process.sh"

ledger="$GAUNTLET_OUT/findings.tsv"
if ! mkdir -p "$GAUNTLET_OUT" 2>/dev/null; then
  echo "[LEDGER] cannot create $GAUNTLET_OUT; the port row cannot be recorded" >&2
  exit 3
fi

if [ -z "$run_token" ]; then
  # Bookkeeping, not the probe: the port may well be free, and this row still
  # could not be told from one an earlier run left in this reused directory.
  st=FAIL
  detail="no run token to bind this precheck to: GAUNTLET_RUN_TOKEN is unset, so this row cannot be told from one an earlier run left in this reused GAUNTLET_OUT"
else
  probe_listener_port "$port"
  case "$LISTENER_PROBE_STATE" in
    none)
      st=PASS
      detail="measured free; run=$run_token"
      ;;
    found)
      # A busy port means later health and CLI checks could pass against a daemon
      # this channel never installed.
      st=FAIL
      detail="BUSY: pid $LISTENER_PROBE_PID is listening on $port; run=$run_token"
      ;;
    *)
      st=FAIL
      detail="could not measure whether $port is free; recorded as unusable, not as free; run=$run_token"
      ;;
  esac
fi

if ! printf '%s\tport-%s-precheck\t%s\t0\t%s\n' \
     "$GAUNTLET_CHANNEL" "$port" "$st" "$detail" >> "$ledger" 2>/dev/null; then
  # The row IS the verdict, so a row that could not be written leaves no
  # verdict at all -- and this script measured "$st" before losing it. On the
  # Windows channels Evaluate would notice the declared row missing; on the
  # POSIX channels scripts/first-run/lib.sh has no declaration mechanism, so
  # nothing downstream can tell this apart from a run where the port was free.
  # A non-zero exit is the only signal left, and it is the honest one.
  echo "[LEDGER] cannot append to $ledger" >&2
  echo "[LEDGER] the port-$port-precheck row ($st: $detail) was measured and LOST;" >&2
  echo "[LEDGER] exiting 3 rather than letting an unrecorded FAIL read as a pass" >&2
  exit 3
fi
echo "[$st] port-$port-precheck: $detail"

# The status of this STEP is the status of the measurement.
#
# This was an unconditional `exit 0`, on the reasoning that the row is the
# verdict and Evaluate is what turns a FAIL row into a failed channel, so
# exiting non-zero would only skip the channel's own teardown. That reasoning
# does not survive contact with where this actually runs. A PRECHECK happens
# before the channel owns anything -- before the install, before the daemon --
# so there is no teardown yet to protect by swallowing the failure. And on the
# POSIX channels there is no Evaluate at all: scripts/first-run/lib.sh has no
# declaration mechanism, so a FAIL row written here is judged by nothing. The
# workflow's own "Port 7878 precheck" step was therefore green on a busy port
# and green on a port nobody could measure -- the two states this script exists
# to tell apart, collapsed back into a pass at the step boundary.
#
# The row stays, because it is the evidence summary.py renders and the FAIL-row
# grep in first-run-gauntlet.yml reads. The exit status now carries the same
# verdict, so a failed measurement cannot reach the next step wearing a green
# check.
[ "$st" = PASS ]
