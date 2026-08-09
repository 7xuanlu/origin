# Wenlan App Graph-Authoring Design Boundary

- **Date:** 2026-06-29 UTC / 2026-06-29 PDT
- **Worktree:** `/Users/lucian/Repos/wenlan-app/.worktrees/wenlan-app-graph-authoring-design`
- **Branch:** `codex/wenlan-app-graph-authoring-design`
- **Backend source:** `/Users/lucian/Repos/wenlan/.worktrees/wenlan-redistill-contract`
- **Base:** `origin/main` at `f37a4fb` after PR #50
- **Status:** Proposed design boundary for boule/user review. Not an implementation plan yet.

## Why This Exists

PR #50 reduced the remaining direct backend/app route gap to nine classified
routes and zero unclassified gaps. The only remaining `design_required` desktop
routes are:

- `POST /api/memory/link-entity`
- `POST /api/memory/relations`

Both mutate knowledge-graph state. They should not become raw desktop wrappers
until the daemon exposes a contract with provenance, validation, and recovery
semantics.

## Tool Boundary

| Tool | Current Result | Boundary |
|---|---|---|
| CodeGraph | `init .` indexed 177 files, 2,654 nodes, and 6,577 edges. `status .` reports the index is up to date. `query`/`explore` require escalation because SQLite opens the `.codegraph` DB in WAL mode. | Use for blast radius and symbol/source context. Do not treat it as behavioral proof. |
| ast-grep | `sg outline app/src/api.rs` and `sg outline app/src/search.rs` identified the client and command surfaces for refinery, entity detail, observations, and recent relations. | Use for structural routing and symbol inventory. |
| LSP / compiler | `rust-analyzer` is available; no interactive LSP tool is exposed in this Codex session. | Use Cargo tests when implementation begins. |
| grep fallback | `rg` confirmed current route/classification/source references. | Use when CodeGraph query is unavailable or too broad. |

## Current Evidence

### App State

- `IdentityDetail` already reads entity observations and relations, edits
  observations, adds observations through `/api/memory/observations`, and shows
  linked memories by search.
- `ConstellationMap`, `IdentityDetail`, and recent relation surfaces are
  read-only relation consumers.
- `EntitySuggestions` only offers Dismiss because current daemon
  `suggest_entity` proposals do not have a safe accept path.
- Home's refinery lane accepts only `entity_merge`, `relation_conflict`, and
  `detect_contradiction`; it deliberately does not accept `suggest_entity`.

### Backend State

- `handle_create_relation` calls `wenlan_core::post_write::create_relation` and
  returns a typed `CreateRelationResponse`.
- `handle_link_entity` directly updates `memory.entity_id` and returns untyped
  JSON `{ "linked": true }`.
- `CreateRelationRequest` and `LinkEntityRequest` exist in `wenlan-types`, but
  both are marked `#[doc(hidden)]`; the server still has a local shadow for
  `LinkEntityRequest`.
- `apply_refinement("suggest_entity")` returns validation error 422 with
  `"action 'suggest_entity' has no accept path (reserved for future producer)"`.

## Design Options

### Option A - Refinement-First Graph Authoring (Recommended)

Treat graph authoring as review of daemon proposals, not a free-form graph
editor. The backend should add a typed accept flow for entity suggestions that
can create an entity and link the proposal's source memories in one validated
transaction. The app can then surface that flow in the existing review lane,
with clear copy, confirmation, and cache invalidation.

This matches the current Home refinery pattern and avoids teaching users to
hand-edit graph internals. It also keeps provenance attached to the proposal
that caused the mutation.

### Option B - Manual Identity Detail Editor

Add direct UI controls in `IdentityDetail` for "link memory to this entity" and
"create relation to another entity". This gives expert users power quickly, but
it pushes validation and recovery burden into the app. It also risks corrupting
graph provenance because relation/link intent is not anchored to a daemon
proposal or source review.

This should wait until after Option A exists, if it is ever needed.

### Option C - Keep Graph Authoring Deferred

Leave all graph mutation in daemon/refinery internals and keep the app read-only
for relations. This is safest and may be enough for P1 desktop parity, but it
does not close the user-visible gap around accepting high-confidence entity
suggestions.

This is acceptable if the migration scope is "do not expose graph mutation in
desktop v0.9", but it does not move graph authoring forward.

## Review Outcome

The fallback merit review found no blocking flaw in the graph boundary, but it
did identify one scope correction: graph authoring is not proven required for
P1 desktop migration parity. The current app already keeps `suggest_entity`
accept hidden, and the daemon deliberately rejects generic `suggest_entity`
accepts with 422. Therefore the migration default is to keep graph authoring
deferred unless accepting entity suggestions becomes an explicit product
requirement.

Option A remains the correct future shape if graph authoring is in scope. It is
not the default next implementation slice for P1 parity.

## Recommended Design

If graph authoring becomes explicit product scope, use Option A:

1. **Backend first:** add or promote a typed daemon contract for accepting
   `suggest_entity` proposals. The contract must not reuse the generic
   `/api/refinery/queue/{id}/accept` endpoint unless the daemon can apply the
   proposal without missing required input.
2. **One transaction:** validate proposal state, entity name/type, and every
   source memory before any write; then create the entity, link the proposal
   source memories, record the refinement as resolved, and emit an
   activity/provenance event together. Do not compose this out of current
   primitives unless they are wrapped in a new atomic core operation.
3. **Typed wire surface:** move the public request/response shape into
   `wenlan-types`; avoid app-local `serde_json::Value` and avoid untyped
   `{linked: true}` responses for desktop-facing mutation.
4. **App review UI:** expose the action through Home/Worth-a-glance or a
   dedicated review lane, not as a hidden button in `IdentityDetail`. The action
   should show proposal name, affected memory count, confidence, and a
   reversible failure state.
5. **No manual relation editor yet:** keep `/api/memory/relations` classified as
   design-required until relation proposals carry enough source evidence and
   undo semantics for a desktop flow.

## Non-Goals

- Do not add a raw `linkEntity(sourceId, entityId)` app wrapper in this slice.
- Do not add a raw `createRelation(from, to, type)` app wrapper in this slice.
- Do not expose `/api/memory/link-entity` directly from `IdentityDetail`.
- Do not turn `suggest_entity` Accept back on while daemon accept returns 422.
- Do not introduce a broad graph editor UI during the migration refactor.

## Proposed Backend Contract

The daemon should expose one of these, in order of preference:

1. `POST /api/refinery/queue/{id}/accept-entity-suggestion`
2. `POST /api/refinery/entity-suggestions/{id}/accept`
3. Extend `POST /api/refinery/queue/{id}/accept` only if the proposal payload is
   self-contained or the request body supplies the missing fields.

Minimum request fields:

- `entity_name`
- optional `entity_type`, defaulting only if the daemon has a documented default
- optional `space`
- optional `source_agent`

Minimum response fields:

- `id`
- `entity_id`
- `linked_source_ids`
- `warnings`
- `action_applied`

Failure modes must be typed:

- proposal not found
- proposal already resolved
- missing entity name
- no source memories to link
- partial mutation refused; no entity/link side effects committed

## App Surface

Initial app UI should reuse existing review affordances:

- Show `suggest_entity` proposals in Worth-a-glance or a compact review panel.
- Show title as "Entity suggestion".
- Show proposed entity name, affected memory count, and confidence.
- Enable Accept only when the daemon advertises or returns a typed accept
  capability.
- On Accept, invalidate:
  - `["refineryQueue"]`
  - `["entity-suggestions"]`
  - `["entities"]`
  - `["constellation-entities"]`
  - affected memory/detail queries if source IDs are returned
- On typed validation failure, keep the card visible and show a non-destructive
  error state.

`IdentityDetail` can remain the read surface for the result after accept. It
does not need to own the mutation flow.

## Test Gates

Backend:

- accepting a valid entity suggestion creates one entity, links all source
  memories, resolves the proposal, and returns typed response fields.
- validation failures leave entity rows, memory links, and proposal status
  unchanged.
- accepting the same proposal twice returns a typed already-resolved error.
- `suggest_entity` generic accept behavior is either updated with tests or kept
  explicitly rejecting with tests.

App:

- `suggest_entity` proposal does not show Accept when daemon accept is
  unavailable.
- once the typed accept wrapper exists, Accept calls the new endpoint and
  invalidates the graph/refinery query keys.
- validation failure keeps the proposal visible and does not fake success.
- route-diff remains `9 classified / 0 unclassified` until the backend exposes a
  new typed route; then the classification file must be updated with the new
  intended app surface.

## Boule Review Prompt

Use this prompt for adversarial review before implementation:

```text
Evaluate the proposed Wenlan App graph-authoring design on the merits.

Context:
- origin-app is being migrated to wenlan-app.
- PR #50 classified remaining direct route gaps.
- The only design_required app gaps are POST /api/memory/link-entity and POST /api/memory/relations.
- Current daemon suggest_entity proposals list in the app but generic accept returns 422.
- handle_link_entity directly updates memory.entity_id and returns untyped {linked:true}.
- The proposal recommends backend-first, refinement-first graph authoring instead of raw app wrappers.

Please review for:
- correctness of backend/app boundary
- risk of graph corruption or provenance loss
- missing validation or undo semantics
- whether this should be P1 migration scope or explicitly deferred
- whether an alternate diagnostics/read-only path is better

Do not force a devil's-advocate stance. Evaluate honestly and cite concrete flaws or tradeoffs.
```

## Open Decision

Default recommendation after review: keep both graph routes classified as
`design_required/deferred` for P1 migration and move the next implementation
slice to Settings diagnostics or another read-only parity surface. Proceed with
Option A only if graph authoring, specifically accepting entity suggestions,
becomes explicit product scope.
