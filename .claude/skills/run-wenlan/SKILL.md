---
name: run-wenlan
description: Build, launch, and stop the wenlan daemon (wenlan-server) for local dev and verification. Use when asked to run or restart the daemon, or before driving any surface (HTTP, CLI, MCP) against a live instance.
---

# Run wenlan-server (dev)

Build: `cargo build -p wenlan-server -p wenlan` (add `-p wenlan-mcp` for the MCP bridge).

## Ports — one allocation, no guessing

Three ports exist and they are not interchangeable. Stopping "the isolated
daemon" on the wrong one leaves a live daemon behind and reports success.

| Port | Owner | Notes |
|---|---|---|
| 7878 | the user's real daemon | never kill casually, never verify against |
| 17878 | manual isolated instance (this skill) | the recipes below |
| 17881 | `scripts/smoke-cli.sh` | `PORT=` overrides |
| 17882 | `scripts/smoke-mcp.sh` | `PORT=` overrides; runs alongside the CLI smoke |

The smokes own their own ports so both can run at once. Do not repoint them at
17878 — a manual daemon and a smoke would then fight over one listener, and the
smoke's ownership assertion would fail on a daemon it did not start.

## Isolated instance (default for verification)

Never verify against the shared prod daemon on :7878 — dev and prod share the
platform data dir by default. Isolating the port and `WENLAN_DATA_DIR` is not
enough: the default pages folder is `.wenlan/pages` under the OS user-home, NOT
under `WENLAN_DATA_DIR`, so a capture will write into the user's real notes
unless `knowledge_path` is set explicitly. Write the scratch `config.json`
BEFORE the daemon starts, then read the live value back from
`/api/knowledge/path` — the daemon reloads config per request, so a value
written afterwards proves nothing about what was already ingested.

`WENLAN_NO_AUTOSTART=1` belongs on every harness command: without it a failed
connect starts the user's registered background service.

```bash
DATA_DIR="$(mktemp -d "${TMPDIR:-/tmp}/run.XXXXXX")"
mkdir -p "$DATA_DIR/pages"
printf '{"knowledge_path":"%s"}' "$DATA_DIR/pages" >"$DATA_DIR/config.json.tmp"
mv "$DATA_DIR/config.json.tmp" "$DATA_DIR/config.json"   # atomic, before spawn
WENLAN_NO_AUTOSTART=1 WENLAN_PORT=17878 WENLAN_DATA_DIR="$DATA_DIR" \
  ./target/debug/wenlan-server &
# ready:  curl -sf --max-time 2 http://127.0.0.1:17878/api/health  (poll up to ~120s)
# check:  curl -sf http://127.0.0.1:17878/api/knowledge/path   → must be $DATA_DIR/pages
# stop:   lsof -ti :17878 | xargs kill -9
```

Sandbox gotcha: always give mktemp a template (`"${TMPDIR:-/tmp}/x.XXXXXX"`);
bare `mktemp -d` lands in a denied dir on macOS.

## Windows (Git Bash) — the same instance, four differences

`TMPDIR` is unset, `lsof` is absent, `$!` is not the pid the OS knows the daemon
by, and an MSYS path is not a path the native daemon can open. Each one fails
quietly in a different way, so none of them is optional.

```bash
# 1. mktemp has no TMPDIR to fall back on: give it an explicit template.
DATA_DIR="$(mktemp -d "${TMPDIR:-/tmp}/run.XXXXXX")"
mkdir -p "$DATA_DIR/pages"

# 2. TWO variables, not one rewritten one. The shell keeps the MSYS spelling for
#    its own `rm -rf`; the daemon is handed the native spelling. `cygpath -m`
#    converts at the daemon boundary ONLY: /tmp/run.EiVYlp →
#    C:/Users/<you>/AppData/Local/Temp/run.EiVYlp
NATIVE_DATA_DIR="$(cygpath -m "$DATA_DIR")"
NATIVE_PAGES_DIR="$(cygpath -m "$DATA_DIR/pages")"
printf '{"knowledge_path":"%s"}' "$NATIVE_PAGES_DIR" >"$DATA_DIR/config.json.tmp"
mv "$DATA_DIR/config.json.tmp" "$DATA_DIR/config.json"

WENLAN_NO_AUTOSTART=1 WENLAN_PORT=17878 WENLAN_DATA_DIR="$NATIVE_DATA_DIR" \
  /c/wl-target/debug/wenlan-server.exe &
JOB_PID=$!   # an MSYS job pid — NOT the pid netstat and tasklist report
```

**Listener check — source the library; do not hand-roll `netstat`.** There is no
`lsof`, and the obvious `netstat -ano | awk ... $4=="LISTENING"` one-liner is
NOT a substitute. It has two states where the question has three, and the state
it cannot express is the dangerous one:

- `netstat` missing, killed, or exiting non-zero → no matching row → reads as
  **free**.
- a non-English Windows → the State column says `ABHOEREN`, not `LISTENING` →
  no matching row → reads as **free**, with the port genuinely busy.
- an unexpected or truncated table → no matching row → reads as **free**.

Starting a second daemon on a port nobody measured is how two daemons end up
sharing one data directory. `scripts/lib/host-process.sh` already answers this
in three states, keys on the structural shape of a listening row rather than on
the localised word, and validates that what it parsed really is the table. Use
it, on Windows and POSIX alike:

```bash
. scripts/lib/host-process.sh

probe_listener_port 17878
case "$LISTENER_PROBE_STATE" in
  found)      echo "busy: pid $LISTENER_PROBE_PID"; exit 1 ;;
  none)       echo "measured free" ;;
  unmeasured) echo "COULD NOT MEASURE port 17878 — refusing to start"; exit 2 ;;
esac
```

Underneath is `listener_pid_for_port` (0 = found, 1 = measured free, 2 = could
not measure); `probe_listener_port` is the wrapper that turns those into
`LISTENER_PROBE_STATE`, because `out="$(f)"; rc=$?` aborts under `set -e` and
`if out="$(f)"; then …; fi; rc=$?` reads the compound's status, which is 0.
Branch on all three states every time. There is no fourth branch to add and no
default that is safe to leave off.

**Resolve the real pid before killing anything.** `$!` is the MSYS job pid; the
Windows pid is the `WINPID` column of `ps -W`, and the image is the `COMMAND`
column read from the header's own offset — `STIME` is one token (`10:23:45`) or
two (`Aug 27`), so a field-index parse shifts on every row for a process that
outlived midnight. That parse also has to reject a table it cannot read instead
of reporting the process absent. All of it lives in `ps_w_row_for`, the single
validated `ps -W` parse in the repository:

```bash
. scripts/lib/host-process.sh

rc=0
row="$(ps_w_row_for PID "$JOB_PID")" || rc=$?
case "$rc" in
  0) WIN_PID="${row%% *}"; IMAGE="${row#* }" ;;
  1) echo "no such job in the process table"; exit 1 ;;
  *) echo "COULD NOT READ the process table — refusing to kill anything"; exit 2 ;;
esac
```

`windows_pid_for_job "$JOB_PID" "$PROGRAM"` is the better entry point when the
daemon was just spawned: it polls until the row names the program actually
launched, because bash's `$!` first maps to the WINPID of the `env` that is
about to be replaced. Do not write a second `ps -W` parse — there have been
three copies of it in this repository and each time the hardening landed on one
of them while the others went on counting words.

**Kill by identity, never by pid alone.** Windows recycles pids fast, and the
pid in a stale state file may now be something else entirely. Compare the image
path first; `Handle` is read to pin the pid so the process cannot exit and be
replaced between the lookup and the kill:

```bash
MSYS2_ARG_CONV_EXCL='*' powershell.exe -NoProfile -NonInteractive -Command '
  $want = "C:/wl-target/debug/wenlan-server.exe" -replace "/", "\"
  try { $p = [System.Diagnostics.Process]::GetProcessById('"$WIN_PID"') } catch { exit 3 }
  try { $null = $p.Handle; $got = $p.MainModule.FileName } catch { exit 6 }
  if ($got -ine $want) { exit 4 }   # someone else owns this pid — do NOT kill
  $p.Kill(); exit 0'
```

Exit 3 = already gone, 4 = a different image (refuse), 6 = could not read the
image (refuse). Only 0 means "we killed the process we meant to kill".

**Then assert the teardown.** Confirm the process is gone AND the port released
before printing anything that looks like a pass; `scripts/smoke-cli.sh` and
`scripts/smoke-mcp.sh` do exactly this and hold their PASS line until cleanup
succeeds.

## Prod daemon (the user's real instance on :7878)

Managed service: `wenlan background on` / `wenlan background off` / `wenlan restart`
(launchd on macOS, systemd-user on Linux, schtasks on Windows). Other agents and the
desktop app share it — never kill it casually.

## Lifecycle checklist

- **Which binary owns :7878** — launchd, the main checkout, a stale worktree, or a
  previous session can all own it. `lsof -i :7878` for the PID, then
  `lsof -p <PID> | grep "txt.*wenlan-server"` for the binary path and size. Kill and
  restart from the current working tree when in doubt.
- **Stale binary after merge/pull** — `cargo build -p wenlan-server` may report
  "Finished" without recompiling after a fast-forward (source timestamps unchanged).
  Force it: `touch crates/wenlan-server/src/router.rs && cargo build -p wenlan-server`,
  then confirm the timestamp: `ls -la target/debug/wenlan-server`.
- **kill vs kill -9** — plain `kill <PID>` may not terminate the daemon cleanly. Use
  `kill -9 <PID>` and verify with `lsof -ti :7878`; if the port is still busy, another
  process took over.
- **Worktree target dirs are per-worktree** — a binary built inside `.worktrees/<name>`
  lands in that worktree's `target/`, not the main repo's. Verify a running binary's
  source with `lsof -p <PID> | grep wenlan-server`.
- **Upgrading requires a restart** — installing a new binary never replaces a running
  daemon: the new process detects the healthy incumbent on :7878 and exits
  (`crates/wenlan-server/src/main.rs`). `wenlan background on` stops the running service
  before reinstalling; `wenlan restart` (stop then start) reloads it explicitly. The MCP
  version handshake surfaces a stale daemon (`VersionStatus::DaemonOutdated`) and points
  users at `wenlan restart`.
- **Reranker startup** — enabling the cross-encoder (`WENLAN_RERANKER_ENABLED=1`) blocks
  startup on a one-time ~1.1GB model download and, on failure, serves with no rerank;
  `/api/status` reports `reranker` as `disabled` / `active` / `failed` so the degraded
  state is visible.
