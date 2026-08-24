---
name: run-wenlan
description: Build, launch, and stop the wenlan daemon (wenlan-server) for local dev and verification. Use when asked to run or restart the daemon, or before driving any surface (HTTP, CLI, MCP) against a live instance.
---

# Run wenlan-server (dev)

Build: `cargo build -p wenlan-server -p wenlan` (add `-p wenlan-mcp` for the MCP bridge).

## Isolated instance (default for verification)

Never verify against the shared prod daemon on :7878 — dev and prod share the
platform data dir by default.

```bash
WENLAN_PORT=17878 WENLAN_DATA_DIR="$(mktemp -d "${TMPDIR:-/tmp}/run.XXXXXX")" \
  ./target/debug/wenlan-server &
# ready:  curl -sf --max-time 2 http://127.0.0.1:17878/api/health  (poll up to ~120s)
# stop:   lsof -ti :17878 | xargs kill -9
```

Sandbox gotcha: always give mktemp a template (`"${TMPDIR:-/tmp}/x.XXXXXX"`);
bare `mktemp -d` lands in a denied dir on macOS.

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
  before reinstalling; `wenlan restart` (graceful stop, wait for exit, start, health
  check) reloads it explicitly. The MCP
  version handshake surfaces a stale daemon (`VersionStatus::DaemonOutdated`) and points
  users at `wenlan restart`.
- **Reranker startup** — enabling the cross-encoder (`WENLAN_RERANKER_ENABLED=1`) blocks
  startup on a one-time ~1.1GB model download and, on failure, serves with no rerank;
  `/api/status` reports `reranker` as `disabled` / `active` / `failed` so the degraded
  state is visible.
