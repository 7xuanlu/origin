# Wenlan App API Parity Revalidation

- **Date:** 2026-06-29 UTC / 2026-06-28 PDT
- **App repo:** `/Users/lucian/Repos/wenlan-app`
- **Worktree:** `/Users/lucian/Repos/wenlan-app/.worktrees/wenlan-app-api-parity-audit`
- **Branch:** `codex/wenlan-app-api-parity-audit`
- **Backend source:** `/Users/lucian/Repos/wenlan`
- **Base:** `origin/main` after PR #39 (`22e41cd`)
- **Daemon probe:** `127.0.0.1:7878`, version `0.9.1`

## Tooling Boundary

| Tool | Use in this pass | Boundary |
|---|---|---|
| CodeGraph | `sync .` on app branch; current graph already up to date | advisory for dependency context only |
| ast-grep | `sg outline app/src/api.rs` | structural method/command inventory |
| LSP/compiler | deferred until implementation | proves types/builds, not product parity |
| `rg`/read | backend router, app route strings, UI call sites, core delete semantics | fallback and source-of-truth line evidence |
| live daemon probes | `/api/setup/status`, `/api/config`, `/api/refinery/queue`, `/api/memory/pending-revisions` | confirms current daemon response shape, not full route coverage |

## Current Evidence

Structural inventory remains current after PR #39:

```text
frontend invoke calls: 153
registered Tauri commands: 176
origin_types references in Rust app code: 0
runtime identity references: 148
stale taxonomy references: 182
source files under app/src and src: 164
```

Route comparison:

```text
backend_routes 123
app_source_api_string_paths 109
```

Backend routes with no direct app source path string after normalization:

```text
/api/config/skip-apps
/api/context
/api/debug/pipeline
/api/distill
/api/distill/{page_id}
/api/ingest/webpage
/api/memory/entities/{entity_id}/observations
/api/memory/link-entity
/api/memory/relations
/api/ping
/api/shutdown
/api/spaces/{from}/move-to/{to}
/api/steep
/ws/updates
```

Initial classification:

| Route | Classification | Next action |
|---|---|---|
| `/api/config/skip-apps` | superseded in the app by sparse `/api/config` updates | no app work unless daemon removes the legacy route |
| `/api/context`, `/api/steep`, `/api/distill*` | agent/daemon workflow routes, no current desktop control surface | defer; add only with an explicit manual distill/product workflow |
| `/api/ingest/webpage` | missing desktop URL ingest surface | P2 candidate after parity blockers |
| `/api/memory/relations`, `/api/memory/link-entity`, `/api/memory/entities/{entity_id}/observations` | graph authoring routes not exposed as desktop edit flows | P2 graph-editor candidate; read-only graph views already use entity detail/recent relation routes |
| `/api/ping`, `/api/debug/pipeline` | health/diagnostics | optional Settings diagnostics; health/status already covered |
| `/api/shutdown` | false positive from comparison script; lifecycle uses a full URL string | no action |
| `/api/spaces/{from}/move-to/{to}` | real space-management gap adjacent to delete-space behavior | design before implementation |
| `/ws/updates` | app currently uses command/query invalidation, not daemon WebSocket | defer unless live daemon events become a product requirement |

Live probe evidence:

```text
GET /api/setup/status
{"setup_completed":true,"mode":"basic-memory","anthropic_key_configured":false,"local_model_selected":null,"local_model_loaded":null,"local_model_cached":false}

GET /api/config
{"skip_apps":[],"skip_title_patterns":[],"private_browsing_detection":false,"setup_completed":true,"clipboard_enabled":false,"screen_capture_enabled":false,"remote_access_enabled":false}

GET /api/refinery/queue?limit=3
{"proposals":[{"id":"contradiction_mem_bc5603e911bf_mem_a647c8eb4214","action":"detect_contradiction",...}]}

GET /api/memory/pending-revisions?limit=3
[{"target_source_id":"mem_437b01d62188","revision_source_id":"mem_8c34a28c42f4",...}]
```

## Concrete Gap: Space Delete Memory Action

This was the one route-diff result that mapped to a user-visible behavior mismatch.

Pre-fix app UI sent a memory action:

```text
src/components/memory/SpaceDetail.tsx:96-98
deleteMutation -> deleteSpace(spaceName, memoryAction)

src/components/memory/SpaceDetail.tsx:309-320
UI asks "Keep N memories?" and calls "unassign" for Keep, "delete" for Delete
```

Pre-fix app Tauri command accepted but ignored the action:

```text
app/src/search.rs:2394-2403
delete_space(..., _memory_action: Option<String>) -> delete_space_response(&client, &name)
```

Current daemon HTTP route hardcodes keep:

```text
wenlan/crates/wenlan-server/src/memory_routes.rs:1826-1837
handle_delete_space(...) -> db.delete_space(&name, "keep")
```

Core already supports the intended semantics:

```text
wenlan/crates/wenlan-core/src/db.rs:6307-6311
"keep", "unassign", "delete", "move:target"
```

Implication before the fix: against daemon `0.9.1`, the desktop UI could present "Keep" / "Delete" choices while the daemon preserved the old space tag either way. That was worse than a missing feature because it could silently mislead the user about destructive or cleanup behavior.

## Recommended Boundary

Adversarial review verdict: app-only honesty now, backend-first API extension later. The app may expose only daemon HTTP semantics; core-only capabilities are not product/API capabilities.

Do not implement an app-only compatibility shim that pretends `unassign` or `delete` worked against daemon `0.9.1`.

Preferred sequence:

1. **Backend first:** add a tested daemon API contract for delete-space memory action.
   - Prefer `DELETE /api/spaces/{name}?memory_action=unassign|delete|keep|move:target` over a DELETE body.
   - Preserve default `keep` for old callers.
   - Add server route tests that prove `unassign` and `delete` reach `MemoryDB::delete_space` with the requested action.
2. **App second:** wire `delete_space` to pass the chosen action only when the daemon supports the contract.
   - If version/capability cannot prove support, show one honest action matching current daemon behavior.
   - Keep frontend tests for argument shape, and add Rust command tests around action forwarding.
3. **UI copy third:** make the delete prompt say exactly what will happen.
   - Current daemon behavior: deleting a space removes the space row but keeps existing memories/entities carrying the old tag.
   - New daemon behavior: "Keep" should mean unassign or preserve only if copy says so; "Delete" must actually delete matching memories/entities.

Implemented app-only fallback because backend API extension is a separate daemon contract change:

- Collapse the current two-choice delete prompt into a single honest delete action.
- Remove the `memoryAction` TypeScript parameter until a daemon capability exists.
- Remove the ignored Rust Tauri command argument.
- Record the richer action set as a blocked backend/API gap in the parity matrix.

Implemented files:

```text
src/lib/tauri.ts
src/lib/tauri.test.ts
src/components/memory/SpaceDetail.tsx
src/components/memory/SpaceDetail.header.test.tsx
src/components/memory/SpaceList.tsx
src/components/memory/__tests__/SpaceList.test.tsx
app/src/search.rs
```

Red/green evidence:

```text
RED: pnpm vitest run src/lib/tauri.test.ts src/components/memory/SpaceDetail.header.test.tsx src/components/memory/__tests__/SpaceList.test.tsx
3 failed:
- deleteSpace still sent memoryAction: null
- SpaceDetail still showed unsupported Keep/Delete choices
- SpaceList still called deleteSpace("work", "unassign")

GREEN: same command
3 files passed, 117 tests passed

Final checkpoint evidence:
- `pnpm test` -> 49 files passed; 419 passed, 1 skipped
- `pnpm build` -> passed
- `cargo test --manifest-path app/Cargo.toml --lib` -> 204 passed
- `git diff --check` -> passed
```

## Verification Needed For Implementation

Backend:

```bash
cargo test -p wenlan-server space_delete
cargo test -p wenlan-core reassign_memories_space delete_space
```

App:

```bash
pnpm vitest run src/lib/tauri.test.ts src/components/memory/SpaceDetail.header.test.tsx src/components/memory/__tests__/SpaceList.test.tsx
cargo test --manifest-path app/Cargo.toml --lib space
pnpm test
pnpm build
```

Runtime:

```bash
curl -s http://127.0.0.1:7878/api/health
pnpm tauri dev
```

Do not mark this route parity item complete until the UI action, Tauri command, daemon route, and core behavior all agree under test.
