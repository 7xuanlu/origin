# Wenlan App Migration Goal Context

- **Date:** 2026-06-26
- **App repo:** `/Users/lucian/Repos/wenlan-app`
- **Active worktree:** `/Users/lucian/Repos/wenlan-app`
- **Branch:** `codex/wenlan-app-runtime-validation`
- **Remote:** `https://github.com/7xuanlu/wenlan-app.git`
- **Backend source of truth:** `/Users/lucian/Repos/wenlan`

## Goal

Drive the old `origin-app` codebase to `wenlan-app` end to end. This is not a shallow rename. The app must keep up with current Wenlan daemon/API behavior while preserving bridge compatibility for users who still have Origin-era app paths, LaunchAgents, MCP config, token paths, and data.

## Current Evidence

The active app worktree is the renamed `wenlan-app` worktree, not the backend monorepo worktree.

Current runtime baseline after checkpoint PRs #3, #4, and #5:

```text
GET /api/health -> {"status":"ok","db_initialized":true,"version":"0.9.1"}
GET /api/status -> {"is_running":true,"files_indexed":8931,"files_total":0,"sources_connected":[],"reranker":{"state":"disabled"},"reranker_light":{"state":"disabled"},"reranker_mode":"off"}
```

Merged checkpoint commits on `origin/main`:

```text
1d9537d Merge pull request #3 from 7xuanlu/codex/wenlan-app-next-refactor
944980d fix: bridge app runtime identity to wenlan (#4)
ac263d6 fix: rename app product identity to wenlan (#5)
```

Runtime validation on 2026-06-27T04:44Z:

```text
pnpm tauri dev
Vite ready at http://localhost:1420/
Running /Users/lucian/Repos/wenlan-app/target/debug/wenlan-app

PID 98404: node /Users/lucian/Repos/wenlan-app/node_modules/.bin/../vite/bin/vite.js
PID 98649: /Users/lucian/Repos/wenlan-app/target/debug/wenlan-app
```

Window/process identity evidence:

```text
System Events process names containing "Wenlan" -> wenlan-app
System Events process names containing "Origin" -> <empty>
System Events window title for process "wenlan-app" -> Wenlan
Dock accessibility item -> {name: wenlan-app, description: application dock item, role: AXDockItem, subrole: AXApplicationDockItem}
```

Runtime screenshot:

```text
/private/tmp/wenlan-app-runtime-identity.png
```

The screenshot shows:

- macOS menu bar product name `Wenlan`.
- visible app window with Home selected.
- daemon-backed data loaded (`10 concepts`, `6165 memories`, Worth-a-glance cards, recent AI access rows).
- no blank first paint or framework overlay.

Runtime log evidence:

```text
/Users/lucian/Library/Logs/com.wenlan.desktop/wenlan.log
```

The exact `wenlan.log` file exists after launch. Non-fatal runtime warnings observed:

```text
tauri::protocol::asset: File does not exist at path: /Users/lucian/Library/Application Support/origin/avatars/57515813-4419-4116-bea6-21bc66e1a511.jpg
tauri_plugin_updater::updater: update endpoint did not respond with a successful status code
wenlan_lib::updater: update check failed: Could not fetch a valid release JSON from the remote
```

These warnings did not block page render or daemon-backed data loading.

Current structural inventory from `bash scripts/refactor/inventory.sh`:

| Surface | Count |
|---|---:|
| frontend `invoke(...)` calls | 126 |
| registered Tauri commands | 170 |
| Rust `origin_types` references | 0 |
| runtime identity references | 221 |
| stale taxonomy references | 239 |
| source files under `app/src` and `src` | 151 |

Current build evidence:

```text
curl -fsS http://127.0.0.1:7878/api/health -> {"status":"ok","db_initialized":true,"version":"0.9.1"}
cargo test -p wenlan-app --lib identity_paths -- --nocapture -> 9 passed
cargo test -p wenlan-app --lib remote_access::tests::token_path_imports_nonempty_legacy_token_when_current_missing -- --nocapture -> 1 passed
cargo test -p wenlan-app --lib lifecycle::tests::cleanup_legacy_app_plist_preserves_foreign_file -- --nocapture -> 1 passed
pnpm build -> passed, with existing Vite chunk/dynamic-import warnings
cargo build -> Finished `dev` profile [unoptimized + debuginfo]
cargo test -p wenlan-app --lib -> 165 passed
pnpm test -> 364 passed, 1 skipped
git diff --check -> passed
```

## Completed Migration Base

Already landed in this branch history:

| Area | Evidence |
|---|---|
| typed app client | `wenlan-types = "0.9.2"`, `WenlanClient`, `origin_types` residual count 0; `/api/search` and `/api/memory/search` preserve additive `supplemental_pages` rows |
| sidecar bridge | `wenlan-server` and `wenlan-mcp` external bins, `scripts/prepare-sidecars.sh`, direct `cargo build` passes |
| MCP config bridge | writes `wenlan`, detects and preserves legacy `origin` |
| stable app path bridge | accepts current `Wenlan.app` paths and legacy `Origin.app` paths |
| Dock visibility | app activation policy is regular; debug app shows as an app process |
| migrated avatar path | avatar loader handles migrated paths |
| revision signal bridge | `StoreMemoryResponse` preserves `triggered_revisions` and `auto_superseded`; revision/contradiction mutation wrappers return typed daemon responses; `listPendingRevisions` wraps `/api/memory/pending-revisions` |
| neutral theme baseline | failed palette experiment replaced with conservative graphite-gray tokens; revisit visual design later |
| pending revision Home review lane | `listPendingRevisions` feeds Worth-a-glance cards; Accept/Dismiss call typed daemon wrappers and invalidate relevant caches |
| refinery queue review bridge | `listRefinements`, `acceptRefinement`, and `rejectRefinement` wrap `/api/refinery/queue`; Home Worth-a-glance surfaces refinery proposals with Accept/Dismiss actions |
| setup status bridge | `getSetupStatus` wraps `/api/setup/status`; wizard gating and completion now use daemon-backed setup state instead of app-local config writes |
| config settings bridge | clipboard, screen-capture, private-browsing, and remote-access settings patch `/api/config` through `WenlanClient`; capture toggles mirror the returned daemon value into app-local sensor state; sidecar launch and launchd server plist both export the app-selected data root as `WENLAN_DATA_DIR` |

## Tool Boundaries

Use tools in this order for cross-cutting edits:

| Tool | Use for | Boundary |
|---|---|---|
| CodeGraph | symbol orientation, impact, affected-test hints | advisory only; not a proof oracle |
| ast-grep | repeatable syntax inventory and codemod surfaces | structural, not semantic |
| LSP/compiler | imports, types, signatures, semantic breakage | cannot prove product parity |
| tests/builds | behavior evidence for edited slices | cannot discover full migration scope |
| `rg` | residual checks and allowlists | use after graph/structural scope is known |
| `grep` | last-resort bounded fallback | only if stronger lanes are unavailable |

Current practical state:

- `rust-analyzer` is on `PATH`.
- `codegraph` and `sg` are available in the current shell, but this app worktree is outside the Codex writable root, so CodeGraph index access can require sandbox escalation.
- CodeGraph remains advisory: for the `codex/wenlan-app-search-supplements` checkpoint, `codegraph sync .` succeeded with escalation, but `codegraph query`/`impact` did not resolve `search_results_from_response` or `searchResultTarget`. Use target-specific CodeGraph probes when useful, then fall back to ast-grep, compiler diagnostics, tests, and bounded `rg` for proof.
- ast-grep is available as `sg`; the inventory harness can also use `npx -y -p @ast-grep/cli sg` when needed.

## Next Refactor Slices

### Slice B1: Daemon/API Parity Review Surface

Goal: make the app catch up to current Wenlan review/status/setup surfaces before public product rename.

Current daemon/API parity gaps from read-only subagent exploration:

| Gap | Priority | Action |
|---|---|---|
| setup/config source of truth is split | P0 | setup status, wizard completion, model/external-LLM settings, skip lists, privacy toggles, capture toggles, and remote-access persistence now use daemon `setup_status`/`update_config`; startup runtime gates hydrate from daemon config after health; sidecar launch and patched launchd server plist export the app-selected data root as `WENLAN_DATA_DIR`; remaining direct config file use is local directory-source compatibility, and app-local saves preserve daemon-only JSON keys |
| store/revision signals need UI surfacing | P0 | Rust and TypeScript wrappers now preserve `triggered_revisions` and `auto_superseded`; Home now surfaces pending revisions; post-store notification surfacing still remains |
| revision/contradiction accept/dismiss need consumer handling | P0 | wrappers return typed daemon responses; pending-revision Accept/Dismiss now consumes them through Home cache invalidation |
| pending revisions need central UI | P0 | Home Worth-a-glance now lists pending revisions with Accept/Dismiss; expand later only if volume requires a dedicated review screen |
| memory taxonomy is stale | P1 | remove first-class `goal`, add `lesson`/`gotcha`, and update colors, stability tiers, fixtures, and reclassification options |
| page links/revisions are under-wrapped | P1 | add `getPageLinks`, `listOrphanLinks`, `getPageRevisions`, and `getMemoryRevisions`; move `PageDetail` related links to `/api/pages/{id}/links` |
| source registry still mutates local config | P1 | route add/list/remove through `/api/sources`; keep sync as daemon proxy |
| frontend status shape is stale | P1 | add a daemon `getStatus` wrapper and surface reranker disabled/failed/active state in operator status UI |

Pre-edit structural probes:

```bash
CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 npx -y @colbymchenry/codegraph sync .
CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 npx -y @colbymchenry/codegraph query WenlanClient --json
CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 npx -y @colbymchenry/codegraph impact WenlanClient --json
npx -y -p @ast-grep/cli sg outline app/src/api.rs
npx -y -p @ast-grep/cli sg outline app/src/search.rs
npx -y -p @ast-grep/cli sg run -p 'invoke($CMD, $$$ARGS)' -l ts src
rg -n 'serde_json::Value|pending-revisions|refinery/queue|setup/status|enrichment-status|StatusResponse' app/src src
```

Required verification:

```bash
pnpm build
cargo build
cargo test -p origin-app --lib
pnpm vitest run <targeted UI tests>
curl -s http://127.0.0.1:7878/api/health
curl -s http://127.0.0.1:7878/api/status
```

### Slice B2: Runtime Identity Classification

Goal: decide which remaining Origin references are bridge state and which are real rename work.

Current classification from read-only subagent exploration:

| Residual group | Classification | Action |
|---|---|---|
| app bundle/product identity (`productName`, bundle id, crate/package/executable) | migrate later | cut over as one coordinated product/runtime identity migration |
| low-risk visible Origin labels/docs/menu text | migrate now | rename when not part of compatibility behavior |
| backend sidecars | already migrated | keep `wenlan`, `wenlan-server`, `wenlan-mcp` sidecar strategy |
| server LaunchAgent bridge | keep legacy bridge | keep cleanup/detection for `com.origin.server` |
| app LaunchAgent identity | investigate | decide whether app LaunchAgent follows old bundle id or new brand; if migrating, add old-plist cleanup |
| app config/data root | migrate now | prefer `WENLAN_DATA_DIR`, fallback to `ORIGIN_DATA_DIR`, default to `wenlan`, keep legacy read/import |
| app log paths | migrate now | move app logs to Wenlan naming or document explicitly as legacy |
| local MCP config bridge | keep legacy bridge | keep writing `wenlan` and preserving old `origin` entries |
| remote MCP token/config path | migrate now | prefer `~/.config/wenlan-mcp`, import/fallback from `~/.config/origin-mcp` |
| relay URL/domain | investigate | do not blind-rename until a Wenlan relay is provisioned and existing relay ids have a migration strategy |
| confusing comments/tests | migrate now | leave only comments that explicitly say "legacy Origin bridge" |

Rule: do not delete old state automatically. Detect, preserve, and migrate through an explicit bridge release.

Recommended order:

1. Fix app config/data/env first: align app config and daemon URL helpers with `WENLAN_*` plus legacy fallback.
2. Migrate remote token path to `~/.config/wenlan-mcp` with import fallback from `origin-mcp`.
3. Rename low-risk UI/docs/comments/tests from Origin to Wenlan.
4. Open a separate bundle-identity plan for product name, bundle id, executable/crate name, updater endpoint, app LaunchAgent label, and DMG naming.
5. Investigate relay migration before changing `origin-relay.originmemory.workers.dev`.

### Slice B3: Taxonomy and Product Copy

Goal: remove stale user-facing Origin taxonomy only after API parity is stable.

Candidate scope:

- first-class `goal` memory type in UI.
- visible `concept` copy where the daemon now models pages/wiki surfaces.
- visible `domain` copy where spaces/pages are the correct product language.
- product copy still saying Origin in setup, updater, README, tests, and UI.

## Boule Review Handoff

Use this prompt after the next design/spec draft is ready:

```text
/boule:debate Review the Wenlan app migration design. The target is a large origin-app to wenlan-app convergence, not a shallow rename. Evaluate the plan on the merits: daemon/API parity with Wenlan v0.9.x, typed-client correctness, bridge compatibility for Origin-era user state, runtime identity rename sequencing, structural-tool boundaries, and verification gates. Do not force optimism or pessimism. Identify missing requirements, false dependencies, unsafe ordering, insufficient tests, and any places where the plan could silently strand user data or config.
```

## Do Not Do Yet

- Do not globally replace `Origin` with `Wenlan`.
- Do not remove old MCP config, token, relay id, LaunchAgent, app path, or data path state.
- Do not use color/palette exploration as a blocker for the migration.
- Do not treat CodeGraph output as completion evidence.
- Do not mark the overall goal complete until the app product/runtime identity, parity surfaces, bridge behavior, and Tauri launch against the v0.9 daemon are all verified together.
