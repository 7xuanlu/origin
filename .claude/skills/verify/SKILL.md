---
name: verify
description: Drive/evidence recipe for verifying wenlan changes at their real surfaces (daemon HTTP, CLI, MCP stdio). The handle file the built-in verify protocol expects; launch primitives live in the run-wenlan skill, deeper machinery (mutation audit, behavior trace, weekly sweep) in the prove skill.
---

# Verifying wenlan — drive the real surfaces

Launch: use the `run-wenlan` skill — build, isolated boot, stop, the port
allocation (manual instance :17878, `smoke-cli.sh` :17881, `smoke-mcp.sh`
:17882 — they are not interchangeable), and the daemon lifecycle checklist all
live there. Never verify against the shared prod daemon on :7878.

Drive by surface — these scripts ARE the drive recipes (read them for the flow,
run them for a full round-trip):

- Daemon HTTP: curl the changed route on the isolated port; recipe in
  `.claude/skills/prove/references/daemon.md`.
- CLI: `bash scripts/smoke-cli.sh` (capture → memories → search, black-box).
- MCP: `bash scripts/smoke-mcp.sh` (stdio JSON-RPC initialize → capture → recall).

Gotchas (drive-time):

- Ingest is async (batcher + embedding): poll search up to ~60s before calling
  a miss a failure.
- Record evidence: prefix any check with `bash scripts/attest.sh`. It appends one
  JSON line to `.claude/attest.jsonl` — the ledger the weekly sweep audits — and
  passes the command's exit status through unchanged. If it cannot write the
  ledger it exits non-zero even when the command passed: an unrecorded run reads
  to the sweep as "the smoke never ran", so it must never be reported as a pass.
  (`~/.claude/bin/attest.sh` is a personal macOS helper; it is optional, and it
  does not exist on Windows or in a fresh checkout. Use the repo script.)
- `WENLAN_NO_AUTOSTART=1` on every command that talks to a daemon. Without it a
  connect failure starts the user's registered background service, and the check
  then passes against the wrong instance.

## Windows (Git Bash)

Both smokes run here as-is; the platform differences are handled inside them via
`scripts/lib/host-process.sh`. What changes for hand-driven checks:

- **No `lsof`.** Port checks go through `netstat -ano` (see `run-wenlan`). A
  probe that could not run is not a free port — treat a failed probe as fatal,
  never as "nothing is listening".
- **No `TMPDIR`.** Always give `mktemp -d` an explicit template.
- **MSYS paths are not daemon paths.** Convert with `cygpath -m` at the daemon
  boundary only, into a second variable; the shell keeps the MSYS spelling for
  its own cleanup.
- **`$!` is not the OS pid**, and killing by pid alone can kill a recycled pid's
  new owner. Resolve the WINPID and compare the image path before killing.
- **`python3` may not exist** — `python` is the usual spelling. `scripts/smoke-mcp.sh`
  resolves it; do the same in ad-hoc checks rather than assuming.
- `bash scripts/attest.sh` works here, so Windows `/verify` satisfies its own
  evidence contract. `.claude/skills/prove/SKILL.md` names the same repo wrapper.

Deeper verification — mutation audit (`suite`), behavior trace (`behaviors`),
weekly verify-the-verifier (`sweep`): invoke the `prove` skill.
