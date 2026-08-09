# Wenlan App Remaining Route Gap Audit

- **Date:** 2026-06-29 UTC / 2026-06-29 PDT
- **Worktree:** `/Users/lucian/Repos/wenlan-app/.worktrees/wenlan-app-remaining-route-audit`
- **Branch:** `codex/wenlan-app-remaining-route-audit`
- **Backend source:** `/Users/lucian/Repos/wenlan`
- **Base:** `origin/main` at `cee6d4f` after PR #47

## Scope

Classify the remaining direct backend route gaps after the merged distill-review
checkpoint. This is a design boundary before the next implementation slice: the
goal is to avoid turning risky daemon write routes into desktop commands without
the corresponding product semantics.

## Tool Boundary

| Tool | Result | Boundary |
|---|---|---|
| CodeGraph | `sync .` completed; `query "WenlanClient distill_review PageDetail IdentityDetail" --json` found the current client, page, and identity surfaces. | Advisory dependency context. |
| ast-grep | `sg outline app/src/api.rs` and `sg outline app/src/search.rs` captured the current Rust client and Tauri command surfaces. | Structural inventory only. |
| LSP | `rust-analyzer 1.95.0` is installed. | No interactive LSP tool is exposed in this Codex session; use Cargo diagnostics when implementation starts. |
| Route diff | `pnpm refactor:api-routes --json` returned `{"backendRoutes":123,"appSourceRoutes":113,"missingInApp":10,"appOnly":0}`. | Direct path-string coverage, not product parity by itself. |
| Source reads | Backend handlers and shared wire types were read in `/Users/lucian/Repos/wenlan`. | Source of truth for route semantics. |

## Remaining Direct Route Gaps

| Route | Backend semantics | Current app coverage | Classification | Next action |
|---|---|---|---|---|
| `/api/config/skip-apps` | Legacy GET/PUT list route that reads/writes `cfg.skip_apps`. | App reads and writes skip apps through daemon `/api/config` sparse patches. | Superseded compatibility route. | No app work. Keep route-diff exception unless backend removes the legacy route. |
| `/api/ping` | Plain `pong` health route. | App uses typed `/api/health` and `/api/status`. | Redundant health route. | No app work. |
| `/api/context` | Trust-gated context bundle for MCP/agent callers, keyed by `x-agent-name`, with access logging. | Desktop has copy-as-context and memory/page views, but no agent-context injection caller. | Agent/MCP route, not a desktop parity blocker. | Do not add a desktop button. Consider only for future diagnostics. |
| `/api/debug/pipeline` | Returns raw DB pipeline status JSON. | Status and reranker diagnostics already use `/api/status`. | Operator diagnostics. | Implement the read-only Settings diagnostics panel described in `docs/superpowers/refactor/2026-06-29-wenlan-app-settings-diagnostics-design.md`. |
| `/api/steep` | Manually runs refinery backstop phases and may invoke LLM/refinery/distillation work. | Activity feed displays steep events; no manual maintenance trigger. | Operator maintenance route. | Keep deferred; do not expose until a separate maintenance design covers cost, capability, progress, and failure states. |
| `/ws/updates` | WebSocket subscribe/ingest channel for index progress and ingest completion. | App uses Tauri events, query invalidation, and status polling. | Architecture replacement, not a route-wrapper task. | Defer until replacing app event flow with daemon events is an explicit product goal. |
| `/api/distill/{page_id}` and `/api/distill` with `force` | Both destructive re-distill paths can clear `user_edited`; the page route does this before checking daemon LLM availability, and the global route does the same when a page target resolves with `force=true`. | PR #47 exposes safe global `/api/distill` review only, with an empty body and no `target` or `force`. PageDetail shows revisions and stale/source data but has no rebuild action. | Risky write route. | Backend/API contract should change before UI: add a non-destructive dry-run or confirmation contract, and avoid clearing `user_edited` before LLM availability is proven. |
| `/api/memory/entities/{entity_id}/observations` | Adds an observation directly under the path entity id. | IdentityDetail already adds observations through `/api/memory/observations` with `entity_id` in the typed request. | Alternate route for existing graph write. | No direct wrapper needed unless the generic route is deprecated. |
| `/api/memory/relations` | Creates an explicit relation between two entities via `CreateRelationRequest`. | Graph views read relations; app does not author relations directly. | Graph-authoring write route. | Design a relation editor/review flow before exposing. Prefer refinement-queue accept flows where available. |
| `/api/memory/link-entity` | Mutates a memory row to point at an existing entity and returns `{"linked": true}`. | Entity suggestions/refinery flows exist, but no explicit link-memory-to-entity editor. | Graph-authoring write route. | Design around provenance and undo before exposing. A blind wrapper could corrupt KG links. |

## Design Verdict

The remaining route diff is not a list of ten implementation tasks. After PR #47,
the safe user-facing app parity gaps are mostly closed or already classified. The
remaining direct gaps fall into three groups:

1. **No app work:** `/api/config/skip-apps`, `/api/ping`, and the entity-path
   observation route are covered by stronger existing app contracts.
2. **Operator/agent architecture:** `/api/context`, `/api/steep`, and
   `/ws/updates` still need a designed diagnostics/event surface before app
   work. `/api/debug/pipeline` now has a read-only Settings diagnostics design.
3. **Risky write surfaces:** `/api/distill/{page_id}`, `/api/distill` with
   `force`, `/api/memory/relations`, and `/api/memory/link-entity` should not
   be exposed as raw wrappers. They mutate user-authored pages or KG structure
   and need confirmation, provenance, undo/error states, and a backend contract
   that cannot silently damage state.

## Recommended Next Checkpoint

Implement the read-only Settings diagnostics surface for `/api/debug/pipeline`
described in
`docs/superpowers/refactor/2026-06-29-wenlan-app-settings-diagnostics-design.md`.
Keep `/api/steep` out of P1 UI until a separate maintenance design covers cost,
capability, progress, and failure states.
