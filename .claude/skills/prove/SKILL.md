---
name: prove
description: Per-surface verification loops for wenlan — daemon, cli, mcp, plugin, suite strength (mutation), behavior trace, weekly sweep. Routes to scripts; every check records evidence via attest. Deeper than the built-in verify skill — use prove for mutation audits, behavior tracing, and the weekly sweep; built-in /verify handles generic runtime observation.
---

# /prove [surface] — wenlan verification loops

Run every check through the evidence wrapper (records to `.claude/attest.jsonl`,
which the weekly sweep audits): `bash scripts/attest.sh <command>`. That wrapper
ships in the repo and runs on macOS, Linux and Windows (Git Bash); the older
`~/.claude/bin/attest.sh` is a personal copy that does not exist on every
machine, and a missing wrapper means the run is unrecorded, which the sweep
reads as "the smoke never ran". Sandboxed local runs need `TMPDIR=/tmp/claude`
prefixed (macOS mktemp gotcha).

| Surface | Command | Proves |
|---|---|---|
| `daemon` | read `references/daemon.md` | HTTP behavior on an isolated daemon |
| `cli` | `attest.sh bash scripts/smoke-cli.sh` | shipped CLI black-box round-trip |
| `mcp` | `attest.sh bash scripts/smoke-mcp.sh` | stdio JSON-RPC round-trip |
| `plugin` | `/wenlan:setup` verify path | plugin → MCP → daemon wiring |
| `suite` | breadth: `cargo mutants -f <file>` (auto-generated mutants, lib-test oracle); depth: read `references/mutprove.md` | tests can actually go red (mutation audit) |
| `behaviors` | `attest.sh python3 scripts/check-behavior-trace.py <plan.md> <tests...>` | tests trace to intent, both directions |
| `sweep` | read `references/sweep.md` | weekly verify-the-verifier |

No argument → pick the surfaces the current diff touches (server/core → daemon;
cli crate → cli; mcp crate → mcp; new/changed tests → behaviors + suite).
App/UI surface (`app/`, `src/`, or anything rendered): rendered evidence
required — drive the running app and look at the screen; unit/build green is
not proof of visible correctness.

Live-smoke legs that need the real GPU model (L7) stay manual:
`scripts/live-smoke-*.sh` — run before merging features whose e2e stubs the LLM.
