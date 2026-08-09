# AGENTS.md - app/

## OVERVIEW

Tauri 2 Rust crate and desktop runtime boundary. This directory owns app
startup, Tauri plugins, sidecar declarations, macOS lifecycle helpers, daemon
HTTP calls, and eval fixtures used by the app crate.

## STRUCTURE

```text
app/
|-- Cargo.toml           # crate version, deps, features, lint cfg
|-- tauri.conf.json      # bundle/updater/window/externalBin contract
|-- src/
|   |-- lib.rs           # runtime hub and invoke_handler registration
|   |-- search.rs        # large Tauri command surface
|   |-- api.rs           # typed daemon HTTP client
|   |-- lifecycle.rs     # launchctl, plist, service management
|   |-- remote_access.rs # cloudflared and wenlan-mcp sidecar orchestration
|   |-- router/          # intent, bundle, keyword routing
|   |-- sources/         # filesystem, Obsidian, upload/source sync
|   `-- sensor/          # capture sensing
`-- eval/fixtures/       # TOML retrieval/eval datasets, not runtime code
```

## WHERE TO LOOK

| Task | Location | Notes |
| --- | --- | --- |
| Add/modify Tauri command | `src/search.rs`, `src/lib.rs` | command body plus `generate_handler!` registration |
| Change daemon route usage | `src/api.rs` | keep typed request/response wrappers here |
| Startup, tray, window behavior | `src/lib.rs` | high fan-out; verify app lifecycle |
| Run-at-login or plist repair | `src/lifecycle.rs` | macOS persistence and legacy Origin paths |
| Remote MCP/tunnel behavior | `src/remote_access.rs` | cloudflared, `wenlan-mcp`, relay registration |
| Sources integration | `src/sources/` | source traits, sync, uploads, wire types |
| Router behavior | `src/router/` | intent classification and bundle assembly |
| Eval fixture edits | `eval/fixtures/` | data-only TOML scenarios |

## CONVENTIONS

- Keep daemon access behind `WenlanClient` in `src/api.rs`; do not scatter raw
  URLs or response-shape parsing through command handlers.
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
cargo clippy --workspace --all-targets -- -D warnings
pnpm test:all
```
