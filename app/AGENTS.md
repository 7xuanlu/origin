# AGENTS.md - app/

## OVERVIEW

Tauri 2 Rust crate and desktop runtime boundary. This directory owns app
startup, Tauri plugins, sidecar declarations, macOS lifecycle helpers, daemon
HTTP calls, and eval fixtures used by the app crate.

## WHERE TO LOOK

| Task | Location | Notes |
| --- | --- | --- |
| Add/modify Tauri command | `src/search.rs`, `src/lib.rs` | command body plus `generate_handler!` registration |
| Change daemon route usage | `src/api.rs` | keep typed request/response wrappers here |
| Startup, tray, window behavior | `src/lib.rs` | high fan-out; verify app lifecycle |
| Run-at-login or plist repair | `src/lifecycle.rs` | macOS persistence and legacy Origin paths |
| Remote MCP/tunnel behavior | `src/remote_access.rs` | cloudflared, `wenlan-mcp`, relay registration |
| Sources integration | `src/sources/` | source traits, sync, uploads, wire types |
| Eval fixture edits | `eval/fixtures/` | data-only TOML scenarios |

## CONVENTIONS

- **Debug builds refuse to start outside `scripts/dev-runtime.sh`.** See that
  script for the variable set; release builds read none of these.
- Keep daemon access behind `WenlanClient` in `src/api.rs`; do not scatter raw
  URLs or response-shape parsing through command handlers. Anything that talks
  to "the daemon" must go through `client.base_url()`, never a literal
  `127.0.0.1:7878` — an isolated dev app selects a different port.
- Register new Tauri commands in `src/lib.rs` after adding the command function.
- Prefer module-local Rust unit tests near the behavior under `#[cfg(test)]`.
  Use `app/tests/*.rs` only for cross-module or daemon-backed integration.
- `app/tests/sources_integration.rs` is ignored because it needs a live daemon.
- `tauri.conf.json` declares `wenlan`, `wenlan-server`, `wenlan-mcp`, and
  `cloudflared` as `externalBin`; packaging can fail before app code runs.
- `eval/fixtures/gen/` is generated-data territory.

## ANTI-PATTERNS

- Do not interpret CI's touched sidecar placeholders as real daemon validation.
- Do not make `remote_access.rs` failures silent without a recovery path or log.
- Do not change launchd/plist behavior without checking stale Origin and Wenlan
  migration paths in `lifecycle.rs`.
- Do not add a new daemon API shape in Rust without updating the frontend wrapper
  if the UI consumes it.

## COMMANDS

```bash
cd app && cargo test
cargo fmt --check --all
cargo clippy --workspace --exclude wenlan-app --all-targets -- -D warnings
pnpm test:all
```
