# Wenlan App Webpage Ingest Wrapper Checkpoint

- **Date:** 2026-06-29 UTC / 2026-06-28 PDT
- **Worktree:** `/Users/lucian/Repos/wenlan-app/.worktrees/wenlan-app-api-parity-audit`
- **Branch:** `codex/wenlan-app-webpage-ingest-wrapper`
- **Backend source:** `/Users/lucian/Repos/wenlan`

## Scope

Close the direct app-route gap for `/api/ingest/webpage` by adding a typed app seam only:

- Rust `WenlanClient::ingest_webpage`
- Tauri command `ingest_webpage`
- TypeScript `ingestWebpage`
- generated API route-diff refresh

This checkpoint intentionally does **not** add a visible URL-ingest UI. The route-gap classification identified the UI as product design work because duplicate behavior, URL/content capture, and user copy still need decisions.

## Tool Boundary

| Tool | Use |
|---|---|
| CodeGraph | `sync .`, `query WenlanClient --json`, and `affected app/src/api.rs app/src/search.rs src/lib/tauri.ts --json` to locate the client surface and broad dependent test set. |
| ast-grep | Local `sg` binary used for Rust async function and TS invoke-wrapper pattern checks. No remote `npx -p @ast-grep/cli` fetch was used. |
| LSP | `rust-analyzer` binary is present, but this Codex session has no stable interactive LSP tool. Rust semantic validation came from `cargo test`. |
| grep/source reads | Used for backend handler and `wenlan-types` request/response contract verification. |

## Contract Evidence

Backend and shared types confirm:

- `wenlan_types::requests::IngestWebpageRequest`
- `wenlan_types::responses::IngestResponse`
- daemon route `POST /api/ingest/webpage`
- source stored as `webpage`, `source_id`/`document_id` as URL, and metadata domain derived by daemon

## TDD Evidence

RED:

- `cargo test --manifest-path app/Cargo.toml api::tests::ingest_webpage_uses_daemon_webpage_ingest_endpoint`
  - failed with `no method named ingest_webpage`
- `pnpm vitest run src/lib/tauri.webpage-ingest.test.ts`
  - failed with `TypeError: ingestWebpage is not a function`
- `cargo test --manifest-path app/Cargo.toml search::ingest_command_tests::webpage_ingest_command_response_type_is_checked`
  - failed with `cannot find function ingest_webpage`

GREEN:

- Rust client test posts the shared request to `POST /api/ingest/webpage` and parses `IngestResponse`.
- Rust command contract test type-checks the Tauri command as returning shared `IngestResponse`.
- Vitest wrapper test verifies `ingestWebpage(req)` invokes `ingest_webpage` with `{ req }`.

## Route-Diff Result

`pnpm refactor:api-routes --json` after implementation:

```json
{"backendRoutes":123,"appSourceRoutes":111,"missingInApp":12,"appOnly":0}
```

The previous direct gap count was 13; `/api/ingest/webpage` is now app-covered by a typed seam.
