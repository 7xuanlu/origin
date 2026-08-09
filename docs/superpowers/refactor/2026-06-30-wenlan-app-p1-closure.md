# Wenlan App P1 Migration Closure Audit

- **Date:** 2026-06-30 UTC / 2026-06-29 PDT
- **Worktree:** `/Users/lucian/Repos/wenlan-app/.worktrees/wenlan-app-migration-revalidation`
- **Branch:** `codex/wenlan-app-migration-revalidation`
- **Backend source:** `/Users/lucian/Repos/wenlan`
- **Backend source revision:** `5f3ec0dc9abf03ae00bb9e7a26e773910288c06a`
- **Base:** `origin/main` at `557840e` after PR #53
- **Merged checkpoint:** PR #54, merge commit `965721bdccb9f7cfae1d1efa8b6bcd53ffec9af7`
- **Status:** P0/P1 migration closure checkpoint merged. Deferred graph-authoring, maintenance, WebSocket, and palette work remains out of scope for this closure pass.
- **Third-eye review:** completed after PRs #54, #55, and #56 merged; verdict was `ready with caveats`, with no critical issues.

## Scope

This audit checks whether the old `origin-app` codebase is caught up enough to
serve as the active `wenlan-app` desktop app against the current Wenlan daemon
API. It is not a visual redesign and not a product-expansion pass.

The closure target is:

- current Wenlan daemon/API parity for P0/P1 desktop surfaces;
- structural tooling coverage before broad refactor claims;
- explicit classification of routes that should not become raw desktop
  wrappers;
- bridge preservation for Origin-era paths, data, LaunchAgents, MCP config,
  token paths, and relay state;
- repeatable validation through tests, builds, route diff, sidecar prep, and a
  Tauri launch against the local daemon.

## Tool Boundary

| Tool | Current result | Boundary |
|---|---|---|
| CodeGraph | `codegraph init -i .` indexed 179 files, 2,688 nodes, and 6,648 edges. `codegraph status .` reports the index is up to date. Queries for `WenlanClient`, `DiagnosticsSection`, and `EntitySuggestions` resolved the relevant app/client surfaces. | Structural and blast-radius evidence only; not behavioral proof. |
| ast-grep | `sg outline` confirms the current Tauri, Rust client, diagnostics, entity suggestion, and identity detail surfaces. | Structural inventory only. |
| LSP/compiler | `cargo test --manifest-path app/Cargo.toml` compiles and runs the Tauri Rust test suite. | Type/build proof for Rust app code; not product proof by itself. |
| route diff | `pnpm refactor:api-routes --json` reports 123 backend routes, 115 app route strings, 8 missing app direct routes, all 8 classified, 0 unclassified, and 0 app-only routes. | Direct route-string coverage and classification guard; not a demand to expose every daemon route. |
| grep fallback | Bounded residual scans classify legacy Origin bridges, taxonomy compatibility, and deferred routes. | Used after structural tools to verify remaining named surfaces. |

## Current Closure Evidence

The fresh worktree was created from `origin/main`, not from the dirty base
checkout:

```text
git status --short --branch
## codex/wenlan-app-migration-revalidation...origin/main
```

Setup and structural probes:

```text
pnpm install --frozen-lockfile
Done

pnpm prepare:sidecars
Prepared sidecars in app/binaries for aarch64-apple-darwin

codegraph init -i .
Indexed 179 files; 2,688 nodes; 6,648 edges

codegraph status .
Index is up to date

pnpm refactor:api-routes --json
{"backendRoutes":123,"appSourceRoutes":115,"missingInApp":8,"classifiedMissingInApp":8,"unclassifiedMissingInApp":0,"appOnly":0}
```

Baseline tests and build:

```text
pnpm test
53 files passed; 441 passed; 1 skipped

cargo test --manifest-path app/Cargo.toml
216 passed

pnpm build
passed with existing Vite dynamic-import and large-chunk warnings
```

The first `pnpm test` run produced one `SetupWizard` failure:

```text
stays on connect step when MCP setup fails
expected writeMcpConfig to be called with ["cursor"]
Number of calls: 0
```

That failure did not reproduce in the single test, the full
`SetupWizard.test.tsx` file, an adjacent three-file subset, or the full suite
rerun. Treat it as a transient baseline flake unless it reappears.

## Runtime Launch Evidence

The app was launched from this closure worktree with:

```text
pnpm tauri dev
Running DevCommand (`cargo  run --no-default-features --color always --`)
Running `/Users/lucian/Repos/wenlan-app/.worktrees/wenlan-app-migration-revalidation/target/debug/wenlan-app`
```

Startup warning observed:

```text
tauri_plugin_mcp: [TAURI_MCP] WARNING: No auth token configured. Socket server is unauthenticated.
```

This warning is the local MCP socket auth warning and did not block app launch
or daemon-backed data loading. It remains a release/security validation item if
this checkpoint is used beyond migration closure.

Live daemon probes during the app session:

```text
GET /api/health
{"status":"ok","db_initialized":true,"version":"0.9.1"}
HTTP 200

GET /api/status
{"is_running":true,"files_indexed":9000,"files_total":0,"sources_connected":[],"reranker":{"state":"disabled"},"reranker_light":{"state":"disabled"},"reranker_mode":"off"}
HTTP 200

GET /api/debug/pipeline
{"enrichment":{"failed":1,"ok":3525,"raw":4989,"skipped":4114},"entity_linking":{"linked":2275,"unlinked":6692},"quality":{"high":1563,"low":271,"medium":178,"unclassified":6955},"recaps":797,"refinement_queue":[{"action":"detect_contradiction","count":10,"status":"awaiting_review"},{"action":"detect_contradiction","count":543,"status":"dismissed"},{"action":"detect_contradiction","count":723,"status":"pending"}],"types":{"decision":899,"fact":2708,"gotcha":472,"identity":162,"lesson":396,"null":3908,"preference":422}}
HTTP 200
```

macOS process/window evidence:

```text
System Events process names containing Wenlan/wenlan/Origin -> wenlan-app
System Events process "wenlan-app" window title -> Wenlan
System Events process "wenlan-app" window geometry -> 80, 120, 1280, 720, Wenlan
```

Visual evidence reviewed in the Codex thread:

```text
/var/folders/tf/w6fz2l2x4vg5nx_n4clfvpgh0000gn/T/TemporaryItems/NSIRD_screencaptureui_VjUyYG/Screenshot 2026-06-29 at 7.03.51 PM.png
```

The screenshot shows the Wenlan window open on Home with the profile avatar
resolved, Spaces loaded, Worth-a-glance cards visible, "Where AI looked"
entries visible, and real daemon-backed counts: 10 pages and 6205 memories.
That local `/var/folders/...` screenshot path was temporary and is not a
durable filesystem artifact; the durable runtime proof is the recorded daemon
probes plus process/window evidence above.

Backend checkout caveat:

```text
git -C /Users/lucian/Repos/wenlan status --short --branch
## fix/wenlan-pages-browse...origin/fix/wenlan-pages-browse [ahead 3]
 M plugin/skills/capture/SKILL.md
```

The dirty file is outside the daemon route/type surfaces used by this app
closure audit.

## Remaining Route Gaps

The generated diff has 8 backend routes without direct app source path strings:

| Route | Classification | Closure decision |
|---|---|---|
| `/api/config/skip-apps` | superseded compatibility route | No app work; app uses sparse `/api/config`. |
| `/api/context` | agent/MCP route | Keep hidden unless a diagnostics context panel is designed. |
| `/api/memory/entities/{entity_id}/observations` | alternate route | No direct wrapper; app already uses the generic observation route. |
| `/api/memory/link-entity` | graph-authoring write | Keep deferred; raw memory-to-entity links can corrupt provenance. |
| `/api/memory/relations` | graph-authoring write | Keep deferred until a provenance and undo-aware relation flow exists. |
| `/api/ping` | redundant health | No app work; app uses typed `/api/health` and `/api/status`. |
| `/api/steep` | operator maintenance | Keep deferred; needs cost, capability, progress, and failure design. |
| `/ws/updates` | architecture gap | Keep deferred unless daemon WebSocket events become an explicit app architecture goal. |

The route diff therefore has no unclassified migration blocker.

## Boundary Decisions

1. **Graph authoring is backend-first.** The app must not add raw wrappers for
   `/api/memory/link-entity` or `/api/memory/relations`. If graph authoring
   enters product scope, the daemon should expose a typed proposal-accept flow
   with validation, atomic writes, provenance, and failure semantics.
2. **Maintenance is not a button.** `/api/steep` may run LLM/refinery work and
   belongs behind a separately designed maintenance surface.
3. **WebSocket adoption is architecture, not parity.** The current app already
   uses Tauri events, polling, and query invalidation. `/ws/updates` should not
   be wrapped just to reduce the route-diff count.
4. **Origin bridge state is intentionally retained.** Legacy data roots, app
   paths, LaunchAgents, MCP config entries, token paths, and relay URLs remain
   migration bridges until a cleanup/release plan explicitly removes them.
5. **Visual palette work is out of scope.** The neutral theme is acceptable for
   migration closure; color direction can be revisited separately.

## Post-Merge Status

PR #54 merged this closure checkpoint into `main` after GitHub CI passed. No
unclassified P0/P1 desktop parity blocker remains in this audit.

The remaining deferred items are explicit product/backend architecture decisions,
not migration closure gaps:

- graph authoring needs daemon-level typed proposal, validation, provenance, and
  recovery semantics before desktop UI work;
- `/api/steep` needs a separate maintenance design before surfacing;
- `/ws/updates` needs an explicit app-event architecture decision before
  replacing the current Tauri event, polling, and query invalidation flow;
- visual palette exploration is separate from daemon/API migration closure.

## Third-Eye Validation

An independent post-merge reviewer rechecked the closure against the original
objective after PRs #54, #55, and #56 were merged. Verdict: `ready with
caveats`.

Findings resolved or bounded:

- **CodeGraph readiness:** reviewer hit `unable to open database file` in a
  restricted context. A follow-up run with filesystem access verified
  `codegraph status .`: 179 files, 2,688 nodes, 6,648 edges, index up to date.
- **Screenshot durability:** reviewer correctly flagged the `/var/folders/...`
  screenshot path as ephemeral. The doc now treats it as reviewed thread
  evidence, not a durable repository artifact.
- **PR metadata:** #54 recorded the closure checkpoint; #55 and #56 then merged
  post-merge status and guardrails. No open migration PRs remained in the
  follow-up audit.
- **Residual scope:** MCP socket auth warning, graph authoring, `/api/steep`,
  `/ws/updates`, and palette exploration remain future product/backend or
  release-security work, not P0/P1 migration closure blockers.

## Review Prompt

Historical prompt used for the pre-merge closure review:

```text
/boule:debate Review docs/superpowers/refactor/2026-06-30-wenlan-app-p1-closure.md and docs/superpowers/refactor/wenlan-app-parity-matrix.md for the origin-app to wenlan-app migration closure checkpoint.

Evaluate on the merits, not as forced optimism or forced pessimism. Check whether the closure claim preserves the original migration goal: current Wenlan daemon/API parity, structural tooling boundaries, bridge compatibility for Origin-era user state, no raw graph/maintenance wrappers, and runtime validation through Tauri launch against the v0.9 daemon.

Identify any missing evidence, unsafe scope narrowing, stale counts, false route classifications, weak tests, or ways this could strand user data/config or hide a real parity gap.
```
