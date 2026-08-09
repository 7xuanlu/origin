# Wenlan App Space Move Wrapper Checkpoint

- **Date:** 2026-06-29 UTC / 2026-06-28 PDT
- **Worktree:** `/Users/lucian/Repos/wenlan-app/.worktrees/wenlan-app-api-parity-audit`
- **Branch:** `codex/wenlan-app-space-move-wrapper`
- **Backend source:** `/Users/lucian/Repos/wenlan`

## Scope

Close the direct app-route gap for `/api/spaces/{from}/move-to/{to}` by adding a typed app seam only:

- Rust `MoveSpaceResponse { affected }`
- Rust `WenlanClient::move_space`
- Tauri command `move_space`
- TypeScript `moveSpace`
- generated API route-diff and structural inventory refresh

This checkpoint intentionally does **not** add a visible bulk-move UI. The route can affect many memories, so the UI still needs affected-count confirmation and copy before users can trigger it from the sidebar.

## Tool Boundary

| Tool | Use |
|---|---|
| CodeGraph | `sync .`, `query "move space" --json`, `query "delete_space" --json`, and `affected app/src/api.rs app/src/search.rs src/lib/tauri.ts src/lib/tauri.test.ts --json`. |
| ast-grep | Local `sg` used for space-related Rust async method and TS invoke-wrapper pattern checks. |
| LSP | `rust-analyzer` binary is present, but this Codex session has no stable interactive LSP tool. Rust semantic validation came from `cargo test`. |
| grep/source reads | Used for backend handler evidence and existing app space command/test patterns. |

## Contract Evidence

Backend source confirms:

- `POST /api/spaces/{from}/move-to/{to}`
- calls `MemoryDB::reassign_memories_space(&from, &to)`
- returns `{ "affected": usize }`
- auto-creates the destination space if it does not exist

## TDD Evidence

RED:

- `cargo test --manifest-path app/Cargo.toml api::tests::move_space_uses_daemon_space_move_endpoint`
  - failed with `no method named move_space`
- `cargo test --manifest-path app/Cargo.toml search::space_command_type_tests::space_response_envelopes_deserialize_daemon_payloads`
  - failed with missing `MoveSpaceResponse` and `move_space`
- `pnpm vitest run src/lib/tauri.test.ts -t moveSpace`
  - failed with `TypeError: moveSpace is not a function`

GREEN:

- Rust client test posts to `POST /api/spaces/Inbox/move-to/Archive` and parses `{ affected }`.
- Rust client regression test percent-encodes path-sensitive space names such as `Work/Clients?old=true` and `Archive#2026`.
- Rust command contract test type-checks the Tauri command as returning `MoveSpaceResponse`.
- Vitest wrapper test verifies `moveSpace(from, to)` invokes `move_space` with `{ from, to }`.

## Review Finding

Fresh review found that the initial wrapper interpolated raw space names into path segments. That would break names containing `/`, `?`, or `#`. The wrapper now percent-encodes both `from` and `to` as path segments before building the daemon route.

## Route-Diff Result

`pnpm refactor:api-routes --json` after implementation:

```json
{"backendRoutes":123,"appSourceRoutes":112,"missingInApp":11,"appOnly":0}
```

The previous direct gap count was 12; `/api/spaces/{from}/move-to/{to}` is now app-covered by a typed seam.
