# Wenlan App Route Gap Classification

- **Date:** 2026-06-29 UTC / 2026-06-29 PDT
- **Worktree:** `/Users/lucian/Repos/wenlan-app/.worktrees/wenlan-app-route-classification`
- **Branch:** `codex/wenlan-app-route-classification`
- **Backend source:** `/Users/lucian/Repos/wenlan`
- **Base:** `origin/main` at `7cb6a29` after PR #49

## Scope

Codify the remaining app-vs-daemon route gaps so future parity work does not
treat every direct route mismatch as a missing desktop feature. The generated
route diff now reads `api-route-classifications.json` and distinguishes raw
missing paths from unclassified parity gaps.

## Tool Boundary

| Tool | Result | Boundary |
|---|---|---|
| CodeGraph | `init .` and `sync .` indexed 177 files, 2,650 nodes, and 6,568 edges. | Advisory structural context. |
| ast-grep | `sg outline scripts/refactor/api-route-diff.mjs` and targeted `sg run` verified the script entry points. | Structural matching only. |
| LSP/compiler | `rust-analyzer 1.95.0` is installed, but this checkpoint is Node/tooling only. | Use tests/builds for verification. |
| Route diff | `pnpm refactor:api-routes --json` returned 123 backend routes, 114 app routes, 9 missing, 9 classified, 0 unclassified, 0 app-only. | Coverage and classification signal, not product proof by itself. |

## Current Route Gap State

```text
backend route paths: 123
app source route paths: 114
backend routes with no direct app source path: 9
classified backend route gaps: 9
unclassified backend route gaps: 0
app source paths with no backend router path: 0
```

## Classification

| Route | Category | Status | Next action |
|---|---|---|---|
| `/api/config/skip-apps` | `superseded_route` | intentional | No app work unless the daemon removes the legacy route. |
| `/api/context` | `agent_route` | intentional | Keep hidden unless a diagnostics context panel is designed. |
| `/api/debug/pipeline` | `operator_diagnostics` | design_ready | Implement the read-only Settings diagnostics panel described in `docs/superpowers/refactor/2026-06-29-wenlan-app-settings-diagnostics-design.md`. |
| `/api/memory/entities/{entity_id}/observations` | `alternate_route` | intentional | No direct wrapper unless the generic observation route is deprecated. |
| `/api/memory/link-entity` | `graph_authoring` | design_required | Design graph-authoring or entity-suggestion acceptance before exposing. |
| `/api/memory/relations` | `graph_authoring` | design_required | Design a relation editor/review flow with provenance and undo states. |
| `/api/ping` | `redundant_health` | intentional | No app work. |
| `/api/steep` | `operator_maintenance` | deferred | Keep deferred; do not expose until a separate maintenance design covers cost, capability, progress, and failure states. |
| `/ws/updates` | `architecture_gap` | deferred | Revisit only if daemon WebSocket events become an explicit app architecture goal. |

## Design Verdict

After PR #49, the raw direct route gap count is no longer an implementation
backlog. The migration backlog is now:

1. **Intentional/no app work:** superseded, redundant, and alternate routes.
2. **Designed surfaces only:** diagnostics, operator maintenance, graph
   authoring, and daemon WebSocket events.
3. **Regression guard:** any new backend route that appears without a
   classification will show up as an unclassified route gap in
   `api-route-diff.md` and `api-route-diff.json`.

Recommended next product checkpoint: implement the read-only Settings diagnostics
surface for `/api/debug/pipeline` described in
`docs/superpowers/refactor/2026-06-29-wenlan-app-settings-diagnostics-design.md`.
Do not add a raw `/api/steep` button or graph-authoring write wrappers without
product semantics, provenance, confirmation, and recovery.
