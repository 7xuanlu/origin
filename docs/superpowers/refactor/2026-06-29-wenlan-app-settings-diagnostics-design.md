# Wenlan App Settings Diagnostics Design

- **Date:** 2026-06-29 UTC / 2026-06-29 PDT
- **Worktree:** `/Users/lucian/Repos/wenlan-app/.worktrees/wenlan-app-settings-diagnostics`
- **Branch:** `codex/wenlan-app-settings-diagnostics`
- **Backend source:** `/Users/lucian/Repos/wenlan`
- **Base:** `origin/main` at `cc1df97` after PR #51

## Scope

Design the next app-only parity slice for the remaining operator routes:

- `GET /api/debug/pipeline`
- `POST /api/steep`

This is a boundary document before implementation. The goal is to make useful
daemon diagnostics available in Settings without turning a maintenance route
into an unsafe desktop command.

## Tool Boundary

| Tool | Result | Boundary |
|---|---|---|
| CodeGraph | Worktree-local `codegraph init -i .` indexed 177 files, 2,654 nodes, and 6,573 edges. `codegraph explore SettingsPage SettingsSidebar get_index_status getCaptureStats RemoteAccessPanel` mapped the Settings blast radius. | Advisory dependency and call-flow context. It does not prove product safety. |
| ast-grep | `sg outline` captured `SettingsPage.tsx`, `SettingsSidebar.tsx`, `app/src/api.rs`, and `app/src/search.rs`. | Structural inventory for where a future wrapper and Settings section would land. |
| LSP/compiler | Fresh baseline: `cargo test --manifest-path app/Cargo.toml` passed 213 tests; `pnpm test` passed 436 tests with 1 skipped. | Type/import correctness comes from compiler and tests, not CodeGraph. |
| Route diff | `pnpm refactor:api-routes --json` returned 123 backend routes, 114 app routes, 9 missing, 9 classified, 0 unclassified, 0 app-only. | Coverage signal only; route names do not imply UI work. |
| Source reads | Backend handlers and core payload builders were read directly from `/Users/lucian/Repos/wenlan`. | Source of truth for route semantics. |

## Backend Semantics

### `GET /api/debug/pipeline`

Backend handler: `crates/wenlan-server/src/routes.rs` `handle_pipeline_status`.

The route returns raw JSON from `MemoryDB::pipeline_status()`:

- `enrichment`: map of enrichment step status to count, plus optional `raw`.
- `entity_linking`: `{ linked, unlinked }` memory counts.
- `refinement_queue`: list of `{ action, status, count }`.
- `recaps`: recap memory count.
- `types`: memory type count map.
- `quality`: quality bucket count map.

This route is read-only and cheap enough for a manual Settings diagnostics
screen. It is raw JSON, so the app should use a local tolerant DTO until the
daemon publishes a shared `wenlan-types` response.

### `POST /api/steep`

Backend handler: `crates/wenlan-server/src/routes.rs` `handle_steep`.

The route runs `wenlan_core::refinery::run_periodic_steep_with_api(...)` with
`TriggerKind::Backstop` and returns a typed `SteepResponse`:

- `memories_decayed`
- `recaps_generated`
- `distilled`
- `pending_remaining`
- `phases: Vec<PhaseResult>`

Even though the response is typed, this route is not read-only. It may run
refinery, LLM, recap, and distillation phases. A raw button would invite users
to trigger background work without phase cost, duration, cancellation, or
failure semantics.

## Options Considered

| Option | Description | Pros | Cons | Verdict |
|---|---|---|---|---|
| A. Read-only diagnostics only | Add a Settings Diagnostics section backed by `GET /api/debug/pipeline`; no maintenance trigger. | Useful parity progress, low risk, testable, works with existing Settings structure. | Leaves `/api/steep` classified as deferred. | Recommended default. |
| B. Diagnostics plus manual steep button | Add diagnostics and a "Run maintenance" action backed by `POST /api/steep`. | Closes two direct route gaps visibly. | Exposes a mutating/costly workflow without a product contract; can confuse background scheduler behavior. | Reject for P1. |
| C. No app work | Keep both routes classified and move to another surface. | Zero risk. | Leaves a useful read-only operator route unexercised and gives no better migration signal. | Too passive after graph authoring was deferred. |

## Recommended Design

Add a Settings group named **Diagnostics** as a P1 read-only operator surface.
It should be quiet and utilitarian, matching the existing Settings layout:

- Sidebar group id: `diagnostics`.
- Section title: `Diagnostics`.
- Section hint: `Daemon pipeline health`.
- Primary action: `Refresh`.
- No auto-running maintenance action.
- Optional polling only while visible, and no faster than existing Settings
  status polling. Manual refresh is enough for the first implementation.

The section should render a single pipeline snapshot:

| Panel | Source field | Display |
|---|---|---|
| Enrichment | `enrichment` | compact key/count rows, sorted by count descending. |
| Entity linking | `entity_linking.linked`, `entity_linking.unlinked` | linked vs unlinked counts and linked percentage when denominator is non-zero. |
| Refinery queue | `refinement_queue[]` | grouped rows by action and status; empty state reads as no pending refinery work. |
| Recaps | `recaps` | single count. |
| Memory types | `types` | key/count rows, sorted by count descending. |
| Quality | `quality` | key/count rows, sorted by count descending. |

Error handling:

- Daemon offline: reuse the app's existing daemon-offline pattern where
  possible; do not blank the whole Settings page.
- Route missing or old daemon: show "Diagnostics require a newer daemon" inside
  the Diagnostics section only.
- Shape drift: fail the wrapper test if required top-level keys disappear, but
  tolerate additive keys and unknown map keys.
- DB not initialized: show the daemon error as a diagnostics error, not as an
  app crash.

Data contract:

- Add an app-local Rust DTO for `PipelineStatusResponse`.
- Keep map fields as `BTreeMap<String, u64>` or equivalent deterministic
  structure for testable ordering.
- Keep `refinement_queue` as a typed vector with `action`, `status`, and
  `count`.
- Public TypeScript wrapper should expose typed data, not `unknown` or
  `Record<string, unknown>`.
- Do not add a raw `/api/steep` wrapper in the same implementation slice.

## Deferred Maintenance Design

`POST /api/steep` needs a separate maintenance design before UI exposure.
Minimum requirements before a manual action:

- explicit cost/capability copy for LLM-backed phases;
- disabled state when daemon intelligence is unavailable;
- confirmation copy that says background maintenance may take time;
- in-progress state and no duplicate trigger;
- phase result rendering using `PhaseResult`;
- error rendering per phase;
- a decision on whether this competes with or complements the scheduler.

Until those exist, `/api/steep` remains classified as `operator_maintenance`
and deferred.

## Implementation Boundary

The next implementation plan should be one narrow task:

1. Add a typed read-only pipeline diagnostics wrapper.
2. Add a `Diagnostics` Settings section that calls the wrapper.
3. Add frontend tests for the section's rendered fields, route-missing error,
   and no maintenance button.
4. Add Rust tests proving the wrapper calls `/api/debug/pipeline` and
   deserializes the current payload shape.
5. Update route-gap docs to say `/api/debug/pipeline` is designed and surfaced
   through read-only Settings diagnostics.

Do not combine this with graph authoring, WebSocket migration, or manual steep
maintenance.

## Self-Review

- Red-flag scan: no unfinished markers or deferred fields without a concrete
  boundary.
- Scope check: one app-only read path plus Settings UI; `/api/steep` remains a
  separate future design.
- Ambiguity check: "diagnostics" means read-only pipeline status, not manual
  maintenance.
- Type consistency: backend raw JSON fields are named exactly as read from
  `MemoryDB::pipeline_status()`.
