# Wenlan App Route-Diff Tooling Checkpoint

- **Date:** 2026-06-29 UTC / 2026-06-28 PDT
- **Worktree:** `/Users/lucian/Repos/wenlan-app/.worktrees/wenlan-app-api-parity-audit`
- **Branch:** `codex/wenlan-app-api-parity-next`
- **Backend source:** `/Users/lucian/Repos/wenlan`

## Scope

This checkpoint makes the app-vs-daemon route comparison reproducible instead of relying on a hand-built route list in the refactor notes.

New command:

```text
pnpm refactor:api-routes
```

Generated artifacts:

```text
docs/superpowers/refactor/wenlan-app-inventory/api-route-diff.md
docs/superpowers/refactor/wenlan-app-inventory/api-route-diff.json
```

## Structural Tool Boundary

| Tool | Boundary |
|---|---|
| CodeGraph | `sync .` and `query WenlanClient --json` bounded the app client surface before adding the route-diff tool. |
| ast-grep | Existing `scripts/refactor/inventory.sh` still owns AST outlines and frontend invoke inventory. |
| LSP/compiler | Not available as a callable tool in this Codex session; `node --check` and Vitest prove the route-diff script shape. |
| `rg`/file reads | Used to inspect backend router shape, app route literal patterns, and verify suspected false positives. |

## Current Route Diff

```text
backend route paths: 123
app source route paths: 110
backend routes with no direct app source path: 13
app source paths with no backend router path: 0
```

Current backend routes with no direct app source path:

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
/api/spaces/{from}/move-to/{to}
/api/steep
/ws/updates
```

The corrected tool no longer reports `/api/pages/export` as missing; that route is present in `app/src/search.rs`. It also recognizes full URL literals such as the lifecycle call to `/api/shutdown`, and ignores Tauri package imports like `@tauri-apps/api/core`.

## Verification

```text
pnpm vitest run scripts/refactor/api-route-diff.test.ts
1 passed

node --check scripts/refactor/api-route-diff.mjs
exit 0

pnpm refactor:api-routes --json
{"backendRoutes":123,"appSourceRoutes":110,"missingInApp":13,"appOnly":0}
```

## Next Use

Use this route-diff artifact at the start of every app/API parity checkpoint. It is a coverage signal, not a product requirement by itself: missing direct app paths still need classification before implementation because some daemon routes are agent-only, diagnostic-only, or intentionally hidden from the desktop app.
