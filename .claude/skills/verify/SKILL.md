---
name: verify
description: Drive/evidence recipe for verifying wenlan changes at their real surfaces (daemon HTTP, CLI, MCP stdio). The handle file the built-in verify protocol expects; launch primitives live in the run-wenlan skill, deeper machinery (mutation audit, behavior trace, weekly sweep) in the prove skill.
---

# Verifying wenlan — drive the real surfaces

Launch: use the `run-wenlan` skill — build, isolated boot on :17878, stop, and the
daemon lifecycle checklist all live there. Never verify against the shared prod
daemon on :7878.

Drive by surface — these scripts ARE the drive recipes (read them for the flow,
run them for a full round-trip):

- Daemon HTTP: curl the changed route on the isolated port; recipe in
  `.claude/skills/prove/references/daemon.md`.
- CLI: `bash scripts/smoke-cli.sh` (capture → memories → search, black-box).
- MCP: `bash scripts/smoke-mcp.sh` (stdio JSON-RPC initialize → capture → recall).

Gotchas (drive-time):

- Ingest is async (batcher + embedding): poll search up to ~60s before calling
  a miss a failure.
- Record evidence: prefix any check with `~/.claude/bin/attest.sh`.

Deeper verification — mutation audit (`suite`), behavior trace (`behaviors`),
weekly verify-the-verifier (`sweep`): invoke the `prove` skill.
