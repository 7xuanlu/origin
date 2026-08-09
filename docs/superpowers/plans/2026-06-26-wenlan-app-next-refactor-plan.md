# Wenlan App Next Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drive the remaining `origin-app` to `wenlan-app` convergence after the daemon-backed setup, pending revision, and refinery review bridges.

**Architecture:** Keep the desktop app as a thin Tauri client over the Wenlan daemon. Treat daemon wire types and routes as source of truth, classify legacy Origin state as compatibility bridge versus product rename, and land each migration slice behind typed wrappers, focused UI consumers, and live daemon/Tauri verification.

**Tech Stack:** Tauri 2, Rust, React 19, TypeScript, TanStack Query, pnpm, Cargo, CodeGraph via `npx -y @colbymchenry/codegraph` when available, ast-grep via `npx -y -p @ast-grep/cli sg`, rust-analyzer/compiler diagnostics, bounded `rg`/`grep` fallback.

---

## Current Baseline

- Active worktree: `/Users/lucian/Repos/wenlan-app/.worktrees/wenlan-app-convergence`
- Branch: `codex/wenlan-app-convergence`
- Backend source of truth: `/Users/lucian/Repos/wenlan`
- Current completed bridges: typed `wenlan-types` client, sidecars, MCP config bridge, avatar migration, conservative theme baseline, daemon-backed setup status, pending revision Home review lane, refinery queue Home review lane.
- Current inventory counts: 126 frontend invokes, 170 registered Tauri commands, 0 Rust `origin_types` references, 221 runtime identity references, 239 stale taxonomy references, 151 source files.

## Tool Boundaries

| Tool | Use | Boundary |
|---|---|---|
| CodeGraph | First-pass symbol orientation and blast-radius hints for cross-cutting Rust/TS surfaces | Advisory only; if unavailable, record failure and continue with ast-grep + LSP + `rg` |
| ast-grep | Repeatable structural inventory and codemod candidate discovery | Structural only; does not prove semantic correctness |
| LSP/compiler | Type/import/signature diagnostics | Does not prove product parity or migration safety |
| tests/builds | Behavioral evidence for the edited slice | Does not discover missing daemon routes |
| `rg` | Bounded residual checks and allowlists after structural scope is known | Fallback, not the first tool for broad refactors |
| `grep` | Last resort when stronger lanes are unavailable | Keep bounded with include/exclude filters |

For every task below, first run or consciously record the structural lane:

```bash
CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 npx -y @colbymchenry/codegraph sync .
CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 npx -y @colbymchenry/codegraph query WenlanClient --json
npx -y -p @ast-grep/cli sg outline app/src/api.rs
npx -y -p @ast-grep/cli sg outline app/src/search.rs
```

If CodeGraph is unavailable, write the exact error in the task note and proceed with ast-grep, compiler diagnostics, and bounded `rg`.

### Task 1: Settings Config Source-of-Truth Convergence

**Files:**
- Modify: `src/components/memory/SettingsPage.tsx`
- Modify: `src/lib/tauri.ts`
- Modify: `app/src/config.rs`
- Modify: `app/src/search.rs`
- Create: `src/components/memory/SettingsPage.daemon-config.test.tsx`
- Test: `src/lib/tauri.test.ts`

- [ ] **Step 1: Inventory local config reads and writes**

Run:

```bash
rg -n "getConfig|updateConfig|localStorage|setup_completed|anthropic|model|reranker|config" src/components/memory/SettingsPage.tsx src/lib app/src/config.rs app/src/search.rs
```

Expected: every settings value is classified as daemon-owned, app-local sensor, or legacy fallback.

- [ ] **Step 2: Add failing tests for daemon-owned settings**

Add or update tests so changing daemon-owned fields calls the typed `updateConfig` wrapper and invalidates the daemon-backed settings query. Keep app-local sensors out of daemon config writes.

- [ ] **Step 3: Implement the smallest settings changes**

Route daemon-owned fields through `/api/config`. Leave explicitly app-local values documented in component-local comments only where needed.

- [ ] **Step 4: Verify**

Run:

```bash
pnpm vitest run src/lib/tauri.test.ts src/components/memory/SettingsPage.daemon-config.test.tsx
pnpm build
cargo test -p origin-app --lib
```

- [ ] **Step 5: Commit**

```bash
git add src/components/memory/SettingsPage.tsx src/components/memory/SettingsPage.daemon-config.test.tsx src/lib/tauri.ts src/lib/tauri.test.ts app/src/config.rs app/src/search.rs
git commit -m "fix: converge settings on daemon config"
```

### Task 2: Enrichment Status Consumer

**Files:**
- Modify: `app/src/api.rs`
- Modify: `app/src/search.rs`
- Modify: `app/src/lib.rs`
- Modify: `src/lib/tauri.ts`
- Modify: `src/lib/tauri.test.ts`
- Modify: `src/components/memory/MemoryDetail*.tsx`

- [ ] **Step 1: Add failing wrapper tests**

Add tests for a Tauri command and TS wrapper for `/api/memory/{source_id}/enrichment-status`.

- [ ] **Step 2: Implement typed wrappers**

Use `wenlan_types::responses::EnrichmentStatusResponse` in Rust and mirror the response shape in TypeScript.

- [ ] **Step 3: Surface status in memory detail**

Show the status only when the route returns data. Do not block memory detail render on enrichment polling.

- [ ] **Step 4: Verify**

Run:

```bash
pnpm vitest run src/lib/tauri.test.ts src/components/memory
cargo test -p origin-app --lib api::tests search::tests -- --nocapture
curl -fsS "http://127.0.0.1:7878/api/health" | jq -e '.status == "ok" and (.version | startswith("0.9."))'
MEMORY_ID=$(curl -fsS "http://127.0.0.1:7878/api/memory/recent?limit=1" | jq -er '.[0].id')
test -n "${MEMORY_ID}" && test "${MEMORY_ID}" != "null"
export MEMORY_ID
curl -fsS "http://127.0.0.1:7878/api/memory/${MEMORY_ID}/enrichment-status" | jq -e '.source_id == env.MEMORY_ID and (.steps | type == "array")'
```

- [ ] **Step 5: Commit**

```bash
git add app/src/api.rs app/src/search.rs app/src/lib.rs src/lib/tauri.ts src/lib/tauri.test.ts src/components/memory
git commit -m "fix: surface memory enrichment status"
```

### Task 3: Page Links and Revision History

**Files:**
- Modify: `app/src/api.rs`
- Modify: `app/src/search.rs`
- Modify: `app/src/lib.rs`
- Modify: `src/lib/tauri.ts`
- Modify: `src/lib/tauri.test.ts`
- Modify: `src/components/memory/PageDetail*.tsx`

- [ ] **Step 1: Add failing tests for page graph wrappers**

Cover `getPageLinks`, `listOrphanLinks`, `getPageRevisions`, and `getMemoryRevisions` at the TS wrapper layer and Rust method-presence layer.

- [ ] **Step 2: Implement typed wrappers**

Use shared `wenlan-types` response structs where available. Avoid `serde_json::Value` unless the shared crate lacks a type; if so, document the gap in the parity matrix.

- [ ] **Step 3: Move related links UI to daemon page links**

Update Page Detail related-link affordances to use `/api/pages/{id}/links` rather than app-local inference.

- [ ] **Step 4: Verify**

Run:

```bash
pnpm vitest run src/lib/tauri.test.ts src/components/memory/PageDetail.test.tsx src/components/memory/__tests__/PageDetail.export.test.tsx
cargo test -p origin-app --lib
pnpm build
```

- [ ] **Step 5: Commit**

```bash
git add app/src/api.rs app/src/search.rs app/src/lib.rs src/lib/tauri.ts src/lib/tauri.test.ts src/components/memory/PageDetail.tsx src/components/memory/PageDetail.test.tsx src/components/memory/__tests__/PageDetail.export.test.tsx docs/superpowers/refactor/wenlan-app-parity-matrix.md
git commit -m "fix: use daemon page links and revisions"
```

### Task 4: Runtime Identity Classification Patch

**Files:**
- Modify: `docs/superpowers/refactor/2026-06-26-wenlan-app-goal-context.md`
- Modify: `docs/superpowers/refactor/wenlan-app-parity-matrix.md`
- Modify: low-risk UI/test/docs files identified by the residual scan
- Do not modify: `app/tauri.conf.json`, bundle id, updater endpoints, relay URL, token migration code, LaunchAgent cleanup code.

- [ ] **Step 1: Generate residual inventory**

Run:

```bash
bash scripts/refactor/inventory.sh
rg -n "Origin|origin-server|origin-mcp|com.origin|origin-relay|originmemory" app src docs README.md package.json
```

- [ ] **Step 2: Classify each residual group**

Mark every group as one of: legacy bridge, bundle identity, relay strategy, low-risk visible copy, test fixture, stale comment.

- [ ] **Step 3: Rename only low-risk visible copy**

Do not change bundle identity or migration bridge state in this task. Leave explicit `legacy Origin bridge` comments where old names remain intentionally.

- [ ] **Step 4: Verify**

Run:

```bash
pnpm test
pnpm build
cargo test -p origin-app --lib
bash scripts/refactor/inventory.sh
```

- [ ] **Step 5: Commit**

```bash
git add app src docs README.md package.json
git commit -m "fix: rename low-risk app copy to wenlan"
```

## Adversarial Review Notes

These are the attacks the next design review should try to break:

| Attack | Defense in current plan | Remaining risk |
|---|---|---|
| The app looks caught up but silently misses daemon review queues | Pending revisions and refinery queue now have typed wrappers and Home review consumers | Dedicated review screen may still be needed if queue volume exceeds Home capacity |
| Settings still split source of truth between app-local and daemon config | Task 1 forces field classification before edits | Easy to over-migrate app-local sensors into daemon config |
| Product rename can strand old users | Runtime identity stays separate from API parity and keeps legacy bridge state readable | Bundle id, updater, relay, and LaunchAgent need a coordinated bridge release |
| CodeGraph can be unavailable or misleading | Tool boundaries make CodeGraph advisory and require ast-grep/LSP/test fallback | Missing graph may increase read cost, not change correctness gates |
| Typed wrappers can compile while UI never calls them | Each wrapper task includes a focused UI consumer or explicit parity-matrix reason for deferral | Hidden route incompatibility still needs live daemon probes |
| Cache invalidation can leave stale Home review cards | Current Home invalidates refinery, pending-revision, recent memory/concept/change, and connection query families after refinery mutations | Dedicated mutation tests may be needed once a full review screen exists |

## Boule Handoff

Use this exact prompt for the user-invoked design council:

```text
/boule:debate Review the Wenlan app next-refactor plan in docs/superpowers/plans/2026-06-26-wenlan-app-next-refactor-plan.md and the parity matrix in docs/superpowers/refactor/wenlan-app-parity-matrix.md. The target is origin-app to wenlan-app convergence, not a shallow rename. Evaluate on the merits: daemon/API parity with Wenlan v0.9.x, typed-client correctness, bridge compatibility for Origin-era user state, runtime identity rename sequencing, structural-tool boundaries, and verification gates. Do not force optimism or pessimism. Identify missing requirements, false dependencies, unsafe ordering, insufficient tests, and places where the plan could silently strand user data, config, or review proposals.
```
