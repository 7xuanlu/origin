# Wenlan App Parity and Compatibility Matrix

- **Date:** 2026-06-30 UTC / 2026-06-29 PDT
- **App checkout:** `/Users/lucian/Repos/wenlan-app/.worktrees/wenlan-app-migration-revalidation`
- **App branch:** `codex/wenlan-app-post-merge-closure`
- **Wenlan backend source:** `/Users/lucian/Repos/wenlan`
- **Purpose:** post-merge parity and compatibility record for the `origin-app` -> `wenlan-app` migration checkpoint.
- **Current status:** PR #54 merged the P1 closure audit; PR #55 marked the checkpoint merged; PR #56 converted stale pre-run gates into post-closure guardrails. The merged checkpoint follows typed-client, sidecar, MCP bridge, Dock/app-activation, avatar path, neutral theme fallback, Home pending-revision/refinery review work, daemon-backed setup status work, post-merge API parity wrappers, `/api/capture-stats`, daemon-backed global tag inventory, daemon-backed status/reranker diagnostics, entity-suggestion action compatibility with the refinery queue, typed space/page/search/setup response envelopes, typed MCP setup entry surface, `wenlan-types` 0.9.2, `/api/search` supplemental page consumption/navigation, shared page-list/search response typing, daemon-config-backed settings toggles, daemon data-root alignment for sidecar plus launchd, Intelligence setup copy moved from Concept to Page language, repeatable Tauri sidecar prep before validation, route-diff revalidation in `docs/superpowers/refactor/2026-06-29-wenlan-app-api-parity-revalidation.md`, startup Dock-icon/reveal fallback work in `docs/superpowers/refactor/2026-06-29-wenlan-app-runtime-assets.md`, local bundled `.app` validation in `docs/superpowers/refactor/2026-06-29-wenlan-app-bundled-runtime.md`, reproducible API route-diff tooling in `docs/superpowers/refactor/2026-06-29-wenlan-app-route-diff-tooling.md`, route-gap classification in `docs/superpowers/refactor/2026-06-29-wenlan-app-route-gap-classification.md`, graph-authoring design boundary in `docs/superpowers/refactor/2026-06-29-wenlan-app-graph-authoring-design.md`, Settings diagnostics implementation in PR #53, typed webpage ingest wrapper work in `docs/superpowers/refactor/2026-06-29-wenlan-app-webpage-ingest-wrapper.md`, typed space move wrapper work in `docs/superpowers/refactor/2026-06-29-wenlan-app-space-move-wrapper.md`, PR #47 global `/api/distill` review-panel work, and the P1 closure audit in `docs/superpowers/refactor/2026-06-30-wenlan-app-p1-closure.md`.

## Evidence Snapshot

| Evidence | Current value | Source |
|---|---:|---|
| Frontend `invoke(...)` calls | 158 | `docs/superpowers/refactor/wenlan-app-inventory/frontend-invokes.txt` |
| Registered Tauri commands | 181 | `app/src/lib.rs` + `search-rs-outline.txt` |
| Rust `origin_types` references | 0 | `app/src` residual scan |
| Runtime identity references | 152 | `Origin`/`origin-server`/`origin-mcp`/`com.origin`/relay residual scan |
| Stale taxonomy references | 184 | `concept`/`goal`/`domain` residual scan |
| Source files under `app/src` and `src` | 169 | `docs/superpowers/refactor/wenlan-app-inventory/summary.md` |
| API route diff | 123 backend paths / 115 app paths / 8 missing app direct paths / 8 classified / 0 unclassified / 0 app-only paths | `docs/superpowers/refactor/wenlan-app-inventory/api-route-diff.md` |
| Wenlan typed request/response declarations in `requests.rs` + `responses.rs` | 115 | `wenlan-types` 0.9.2 scan |

## Compatibility Gates

| Gate | Required before | Check | Fallback behavior |
|---|---|---|---|
| Daemon reachability | Any app UI data load | `GET /api/health` typed as `HealthResponse` | show daemon-offline state; do not assume local DB access |
| Daemon capability set | P0 feature UI | `GET /api/status` plus route probes for optional surfaces | hide unsupported panels; show "requires newer daemon" diagnostics |
| Typed wire contract | API wrapper migration | `wenlan-types` dependency and serde smoke tests for critical envelopes | app-local DTO allowed only for Tauri-local or missing shared types |
| Sidecar availability | `cargo build`/Tauri bundle | target-specific `wenlan-server`, `wenlan-mcp`, and `cloudflared` sidecars present | fail loud with migration instructions; do not hand-copy ad hoc binaries |
| Origin bridge detection | Runtime identity rename | old LaunchAgents, app paths, MCP config keys, token/relay ids detected | preserve old state for bridge release; do not silently open empty Wenlan state |

## P0 Matrix

These block "caught up enough for full refactor run".

| Surface | Current source | App status | Action | Daemon compatibility |
|---|---|---|---|---|
| `/api/health` | `wenlan-server/src/router.rs`; `HealthResponse` | present in `WenlanClient.health` | keep typed daemon reachability gate | required; app cannot proceed without daemon |
| `/api/status` | `StatusResponse`, `RerankerStatus` | `get_index_status` now merges daemon count/source/reranker fields with app-local indexing activity; StatusBar surfaces reranker failures without treating daemon `is_running=true` as perpetual indexing | broaden settings diagnostics if operator detail needs more than the compact StatusBar signal | required for capability gate; tolerate absent fields by feature-gating |
| `/api/search` | `SearchResponse` in `wenlan-types` 0.9.2 | `search` consumes additive `supplemental_pages` and appends page-channel rows into the existing `SearchResult[]` Tauri surface; page rows route to Page detail instead of copy/file open | design a richer discriminated search response only when the UI needs a separate global Pages section | required for global search; tolerates older daemons that omit `supplemental_pages` |
| `/api/memory/search` | `SearchMemoryResponse` | `search_memory` consumes additive `supplemental_pages` and appends page-channel rows into the existing `SearchResult[]` Tauri surface; page rows route to Page detail instead of memory detail | keep existing TS surface until a UI-level result union is intentionally designed | required for memory search; tolerates older daemons that omit `supplemental_pages` |
| `/api/capture-stats` | daemon capture stats route | `get_capture_stats` calls the dedicated daemon route and maps `total_chunks` to the existing frontend `total` key | keep route-level regression test so it does not drift back to `/api/status` | required for capture-count UI |
| `/api/refinery/queue` | `ListRefinementsResponse` | typed Rust/Tauri/TS wrappers present; Home Worth-a-glance surfaces proposals | keep Home review lane; consider a dedicated review screen only if queue volume requires it | require route for review queue; hide panel if absent |
| `/api/refinery/queue/{id}/accept` | `AcceptRefinementResponse` | typed Rust/Tauri/TS wrappers present; Home Accept calls it and invalidates review/recent caches | reuse typed response for any future dedicated review queue | optional until queue route exists |
| `/api/refinery/queue/{id}/reject` | `RejectRefinementResponse` | typed Rust/Tauri/TS wrappers present; Home Dismiss calls it and invalidates the review queue | reuse typed response for any future dedicated review queue | optional until queue route exists |
| `/api/memory/pending-revisions` | daemon route | typed Tauri/TS wrapper present; Home Worth-a-glance surfaces pending revision cards | keep Home review lane; consider a dedicated review screen only if queue volume requires it | require for P0 review surface; hide list if absent |
| `/api/memory/pending-revision/{source_id}` | daemon route | present as per-memory `getPendingRevision` | keep but type through `wenlan-types` if available | optional detail route |
| `/api/memory/revision/{id}/accept` | `RevisionAcceptResponse` | present command accepts `sourceId`; Home review lane calls it and invalidates pending revision/recent memory caches | reuse typed response for any future dedicated review queue | require typed response before review UI |
| `/api/memory/revision/{id}/dismiss` | `RevisionDismissResponse` | present command accepts `sourceId`; Home review lane calls it and invalidates pending revision cache | reuse typed response for any future dedicated review queue | require typed response before review UI |
| `/api/memory/contradiction/{source_id}/dismiss` | `ContradictionDismissResponse` | present command returns typed response | surface in review queue | optional if contradictions route absent |
| `/api/memory/{source_id}/enrichment-status` | `EnrichmentStatusResponse` | typed Rust/Tauri/TS wrapper present; Memory Detail shows non-blocking status when available | keep hidden on old daemons and do not block memory render | optional per-memory route; show unknown state if absent |
| `/api/sources` | daemon source routes | Obsidian registered source list/add/remove/sync route through daemon-backed Tauri commands; local directory sources remain app-config/watcher compatibility entries merged into `list_registered_sources` | keep legacy source-name commands and directory watcher compatibility only where still needed | required for source management |
| `/api/spaces` + `/api/spaces/{name}` + `/api/spaces/{name}/star` | daemon space CRUD routes; create/update return `Space`, delete returns `{deleted}`, star returns `{starred}` | primary and legacy Tauri space wrappers now parse typed daemon envelopes while preserving existing frontend return surfaces | keep legacy alias wrappers until older UI callers are removed; move ad hoc delete/star envelopes into shared `wenlan-types` if the daemon stabilizes them there | required for sidebar Space list and filters |
| `/api/spaces/{from}/move-to/{to}` + delete-space memory actions | daemon exposes a move route returning `{ affected }`, but `DELETE /api/spaces/{name}` hardcodes core `delete_space(..., "keep")`; core-only `unassign`/`delete`/`move:target` actions are not an HTTP contract yet | typed Rust client, Tauri command, and TS wrapper present as `moveSpace`; delete UI remains one honest keep/delete action and no bulk-move UI is exposed yet | design affected-count confirmation before surfacing move UI; add a backend API contract before reintroducing unassign/delete/move delete choices | current daemon compatibility requires app honesty; typed move seam available |
| `DELETE /api/agents/{name}` | `{ deleted: name }` | typed local Rust deleted-name envelope; public Tauri/TS surface remains void | move to shared response type after daemon publishes a named-delete envelope | required for Settings agent management |
| `/api/config` | `ConfigResponse`, `UpdateConfigRequest` | switched to `wenlan-types`; wizard completion, model choice, external LLM settings, skip lists, capture toggles, private-browsing detection, and remote-access persistence now update daemon config with sparse patch bodies; startup runtime gates hydrate from daemon config after health; app-launched sidecar and launchd server plist get the app-selected data root as `WENLAN_DATA_DIR`; app-local sensor state mirrors successful daemon writes for clipboard/screen capture; directory-source compatibility saves preserve daemon-only JSON fields | keep local directory-source compatibility until daemon owns live directory watchers; remove local write helpers only after that compatibility path is gone | required for settings |
| `/api/setup/status` + `/api/setup/anthropic-key` | daemon setup routes; key set/clear return `SuccessResponse` | explicit setup-status DTO, Tauri command, TS wrapper, wizard gating, and typed key set/clear response helpers are present | keep setup daemon-owned; broaden settings diagnostics around mode/model/key state | required for setup correctness |
| MCP config bridge | `app/src/mcp_config.rs`, setup wizard | writes a typed `WenlanMcpEntry` for `wenlan`; detects and preserves legacy `origin` entries | keep bridge behavior through at least one compatibility release | bridge release required |
| Remote access bridge | `app/src/remote_access.rs` | token path is `~/.config/wenlan-mcp/token` with legacy `origin-mcp` import fallback; relay URL intentionally remains legacy Origin until endpoint strategy exists | define relay endpoint strategy before URL rename; do not import legacy relay ids blindly | bridge release required |
| Sidecar build contract | `app/tauri.conf.json`, `package.json`, `scripts/prepare-sidecars.sh` | uses `wenlan`, `wenlan-server`, `wenlan-mcp`, and `cloudflared`; `cargo build` passes after repeatable sidecar prep | keep repeatable sidecar prep; no hand-copying ad hoc binaries | required before build verification |

## P1 Matrix

These are required for feature parity but can follow the P0 review/status/setup surfaces.

| Surface | Current source | App status | Action | Daemon compatibility |
|---|---|---|---|---|
| `/api/memory/{id}/revisions` | `ListMemoryRevisionsResponse` | typed Rust/Tauri/TS wrapper present; Memory Detail shows non-blocking revision history | keep old version-chain wrapper only as legacy fallback until removed in a later cleanup | hide revisions panel if route absent |
| `/api/pages/{id}/revisions` | `ListPageRevisionsResponse` | typed Rust/Tauri/TS wrapper present; Page Detail shows non-blocking revision history | reuse for future page diff UI | hide revisions panel if route absent |
| `/api/pages/{id}/links` | `PageLinksResponse` | typed Rust/Tauri/TS wrappers present; PageDetail uses daemon outbound/inbound links and no longer infers links via `listPages` | keep non-blocking link UI; unresolved outbound labels stay inert | hide links section if route absent or errors |
| `/api/pages/orphan-links` | `OrphanLinksResponse` | typed Rust/Tauri/TS wrappers present; Page Detail shows repeated unresolved labels as a non-blocking diagnostics section | keep hidden on empty/error so older daemons do not block page render; expand to a dedicated review view only if volume requires it | optional diagnostics until route exists |
| `/api/pages/{id}/sources` | `PageSourceWithMemory` | present via `getPageSources` | keep and type through `wenlan-types` | optional |
| `/api/pages` + `/api/pages/search` | `SearchPagesResponse` | `list_pages` and `search_pages` now use the shared response envelope instead of a local `ListPagesWire` mirror | keep shared type as the daemon/app contract | optional page browsing/search |
| `/api/pages/{id}/archive` | `{ status: "archived" }` | typed local Rust status envelope; public Tauri/TS surface remains void | move to shared response type after `wenlan-types` publishes a matching page status envelope | route required for archive action |
| `DELETE /api/pages/{id}` | `{ status: "deleted" }` | typed local Rust status envelope; public Tauri/TS surface remains void | move to shared response type after `wenlan-types` publishes a matching page status envelope | route required for delete action |
| `/api/pages/export` | `ExportStats` | typed Rust/Tauri/TS wrapper present as `exportPagesToObsidian` | keep; rename "concept" UI/file wording where user-facing | optional |
| `/api/pages/{id}/export` | `ExportPageResponse` | typed Rust/Tauri/TS wrapper present as `exportPageToObsidian`; preserves daemon response envelope | keep; rename "concept" UI/file wording where user-facing | optional |
| `/api/ingest/webpage` | `IngestWebpageRequest` / `IngestResponse` | typed Rust client, Tauri command, and TS wrapper present; no visible URL-ingest flow yet | design URL ingest flow separately before surfacing in UI | optional ingest surface; typed seam available |
| `/api/distill` + `/api/distill/{page_id}` | global distill returns review payloads without daemon synthesis when called without `target`/`force`; both page re-distill entry points can clear `user_edited` before a safe desktop confirmation contract exists | global `/api/distill` review wrapper and Home review panel are present; it sends an empty request and exposes no per-page re-distill command/button | keep global review surface; fix backend page re-distill contract and add no-LLM `user_edited` regression tests before adding PageDetail rebuild UI | global review is safe on 0.9.x; page re-distill needs a safer daemon contract |
| remaining direct route gaps | `/api/config/skip-apps`, `/api/context`, `/api/memory/entities/{entity_id}/observations`, `/api/memory/link-entity`, `/api/memory/relations`, `/api/ping`, `/api/steep`, `/ws/updates` | generated route diff shows all 8 gaps classified; graph-authoring boundary proposed in `docs/superpowers/refactor/2026-06-29-wenlan-app-graph-authoring-design.md`; `/api/debug/pipeline` is now implemented through the Settings diagnostics panel | keep superseded, redundant, alternate, agent, operator-maintenance, graph-authoring, and WebSocket-architecture gaps classified; avoid raw write wrappers for graph and maintenance routes | not required for desktop P0/P1 parity |
| `/api/on-device-model` | server-owned model DTOs | Rust/Tauri/TS wrappers use local typed mirrors; `serde_json::Value` removed from the app route | move `OnDeviceModel*` DTOs into `wenlan-types` upstream, then replace local mirrors | optional settings section |
| `/api/llm/test` | `TestLlmRequest` / `TestLlmResponse` | typed Rust/Tauri/TS wrapper present as `testExternalLlm`; preserves daemon response envelope | keep response envelope; settings UI may display `response` if surfaced later | optional settings section |
| `/api/tags` | `TagsResponse` | `list_all_tags` reads the daemon global tag list and preserves additive `document_tags` maps when available; older daemons default to an empty map | merge daemon `document_tags` response support before relying on per-document tag filters in release notes | global tags available; per-document tag map gated by daemon version |
| `/api/memory/{id}/correct` | `{ corrected, source_id }` | typed local Rust correction envelope; public Tauri/TS surface remains corrected string | move to shared response type if the nurture correction route remains public | optional nurture action; requires LLM availability |
| `/api/memory/entity-suggestions` | daemon list route backed by refinement queue IDs; `suggest_entity` has no accept path in current daemon | Sidebar suggestions now expose Dismiss only; dismiss routes through typed refinery reject commands; accept wrappers fail closed instead of invoking the daemon | keep Create hidden until the daemon implements an accept path for `suggest_entity`; use Home-style `canAcceptRefinementAction` gating for any new review surface | list route available; dismiss uses `/api/refinery/queue/{id}/reject`; accept remains unsupported for `suggest_entity` |

## Taxonomy Matrix

| Surface | Current app state | Required action | Gate |
|---|---|---|---|
| `goal` memory type | removed from advertised TypeScript facet union; `lesson`/`gotcha` added; legacy `goal` remains only for old rows/maps | keep compatibility path only; do not reintroduce first-class Goal UI | `taxonomyCopy`, StructuredEditor, ProfilePage tests |
| `concept` copy | user-facing copy migrated to Page/wiki language in product-owned surfaces; remaining graph legend labels `concept` entity type as Theme while preserving wire keys | keep `Concept = Page`, `listConcepts`, `getConceptSources`, `ActivityKind = "concept"` compatibility wrappers | `taxonomyCopy` + page/onboarding/import tests |
| `domain` copy | product-owned visible labels use Space; daemon wire keys still use `domain`/`space` compatibility mapping | keep daemon back-compat field names and request keys | `taxonomyCopy`, domain wire tests |
| `Origin` app copy | product-owned visible copy and runtime names are Wenlan; legacy Origin references remain for bridge paths, tests, fixtures, and compatibility docs | keep legacy references explicit as bridge artifacts; do not globally replace | runtime identity and taxonomy copy tests |

## Runtime Identity Matrix

| Surface | Current app state | Required action | Gate |
|---|---|---|---|
| Bundle id | `com.wenlan.desktop`; runtime identity test asserts product name, identifier, updater endpoint, package/repo identity | keep legacy Origin state readable through bridge paths | runtime identity tests |
| Dock icon, bundle identity, and startup reveal | setup sets the full app icon before activation policy, keeps a delayed backend reveal pass if frontend readiness does not expose the main window, and `pnpm build:app:local` produces a local `Wenlan.app` bundle without updater signing keys | keep bundled `.app` launch as the stronger Dock/bundle evidence because dev mode runs an unbundled binary; production updater artifacts remain enabled in `app/tauri.conf.json` | runtime identity, local bundle validation, and macOS lib tests |
| LaunchAgents | current labels are `com.wenlan.desktop` and `com.wenlan.server`; legacy `com.origin.desktop`/`com.origin.server` are detected and cleaned only when owned | preserve owned legacy cleanup and foreign-file safety | lifecycle tests |
| LaunchAgent template | current `com.wenlan.desktop.plist` uses Wenlan placeholders; legacy `com.origin.desktop.plist` retained as bridge artifact | keep legacy template untouched until bridge cleanup | lifecycle template/install tests |
| Stable app path | accepts `/Applications/Wenlan.app`, `~/Applications/Wenlan.app`, and legacy `Origin.app` bridge paths | keep old app path detection until bridge cleanup | lifecycle tests |
| Lifecycle quit command | `quit_wenlan_full` primary; `quit_origin_full` legacy alias | keep Origin alias through bridge release | Tauri wrapper + Rust command tests |
| Daemon sidecar | `wenlan-server` sidecar, plus `wenlan` CLI for service management | keep legacy process cleanup only where bridge behavior requires it | `cargo build` reaches app code |
| MCP sidecar | `wenlan-mcp` sidecar and setup entry; legacy `origin` MCP config entries are preserved | bridge old config key/package through compatibility release | setup wizard and MCP config tests |
| Remote relay | `origin-relay.originmemory.workers.dev` intentionally retained | provision/decide Wenlan relay endpoint and relay-id migration before changing URL | remote access tests |
| Data/config paths | `WENLAN_DATA_DIR` preferred; `ORIGIN_DATA_DIR` legacy fallback | do not open fresh Wenlan path if legacy data exists | migration smoke |
| UI preference keys | `wenlan-*` / `wenlan:*` primary; Origin-prefixed localStorage keys import-only legacy fallback | preserve user UI state while new writes use Wenlan keys | preference helper + consumer tests |
| Daemon port env | `WENLAN_PORT` preferred; `ORIGIN_PORT` legacy fallback | keep legacy fallback through bridge release; default to `7878` | `api::tests::wenlan_client_*_port` |

## P1 Closure Audit

The 2026-06-30 UTC revalidation and PR #54 merge treat P0/P1 daemon API parity
as closed for the desktop migration checkpoint, with the following explicit
boundaries:

- PR #54 merged the closure checkpoint into `main` at
  `965721bdccb9f7cfae1d1efa8b6bcd53ffec9af7`.
- Raw route parity is green: `pnpm refactor:api-routes --json` reports 123
  backend routes, 115 app route strings, 8 missing app direct routes, all 8
  classified, 0 unclassified, and 0 app-only routes.
- The 8 remaining direct route gaps are not unimplemented desktop features:
  they are superseded compatibility routes, agent routes, alternate routes,
  graph-authoring writes, redundant health, operator maintenance, or a
  deliberate WebSocket architecture gap.
- Graph authoring remains backend-first and product-gated. The app must not
  add raw `/api/memory/link-entity` or `/api/memory/relations` wrappers until
  the daemon exposes typed provenance, validation, and recovery semantics.
- `/api/steep` remains deferred. A manual maintenance button needs a separate
  design covering cost, capability, progress, failure, and cancellation states.
- `/ws/updates` remains an architecture decision, not a wrapper task. The app
  currently uses Tauri events, query invalidation, and polling.
- Legacy Origin state remains a compatibility bridge. Do not delete old MCP
  config, token, relay id, LaunchAgent, app path, or data path state during the
  migration closure pass.

Baseline evidence from the closure worktree:

- `codegraph init -i .` indexed 179 files, 2,688 nodes, and 6,648 edges.
- `codegraph status .` reported the index up to date.
- `pnpm prepare:sidecars` prepared `wenlan`, `wenlan-server`, `wenlan-mcp`, and
  `cloudflared` sidecars from `/Users/lucian/Repos/wenlan/target/debug`.
- `pnpm test` passed on rerun: 53 files, 441 tests passed, 1 skipped. The first
  run showed one `SetupWizard` failure that did not reproduce in a single-test
  run, the full file, an adjacent subset, or the full rerun.
- `cargo test --manifest-path app/Cargo.toml` passed: 216 tests.
- `pnpm build` passed with the existing Vite dynamic-import and large-chunk
  warnings.
- `pnpm tauri dev` launched `target/debug/wenlan-app`; macOS reported process
  `wenlan-app` with window title `Wenlan`, and live `/api/health`,
  `/api/status`, and `/api/debug/pipeline` probes returned HTTP 200 against
  daemon `0.9.1`. The user-provided runtime screenshot in the Codex thread
  showed Home loaded with 10 pages and 6205 memories; its local temp path is
  not a durable repo artifact.

## Post-Closure Guardrails

The full `origin-app` -> `wenlan-app` migration checkpoint has landed. Keep
these checks as regression guardrails for future app refactors:

- Keep `bash scripts/refactor/inventory.sh` and the generated inventory summary
  current after structural app changes.
- Keep `pnpm refactor:api-routes` current and require every missing direct
  daemon route to be classified.
- Use CodeGraph for cross-cutting edits: run `codegraph sync` plus
  target-specific `query`, `impact`, or `affected` probes before changing
  behavior.
- Keep `pnpm test`, `pnpm build`, and `cargo test --manifest-path app/Cargo.toml`
  green for app changes.
- Preserve the repeatable `wenlan-server`/`wenlan-mcp` sidecar strategy; do not
  hand-copy binaries to make Tauri builds pass.
- Keep legacy Origin state readable until a release cleanup plan explicitly
  removes bridge paths, tokens, MCP config, LaunchAgents, or data roots.

Do not regress these migration boundaries:

- Do not replace text globally with `rg` before AST inventory surfaces are
  classified.
- Do not remove old MCP config or token paths automatically.
- Do not add raw desktop wrappers for graph-authoring writes or operator
  maintenance routes just to reduce the route-diff count.
