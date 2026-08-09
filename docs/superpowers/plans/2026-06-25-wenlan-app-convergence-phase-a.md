# Wenlan App Convergence Phase A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare `origin-app` for the Wenlan convergence refactor with repeatable structural inventory, baseline checks, and finite Phase A/B gates before changing behavior.

**Architecture:** Keep the app as a thin Tauri client. Use `wenlan-server` and `wenlan-types` as the source of truth, but first map the app's current Tauri command, HTTP wrapper, local DTO, sidecar, and identity surfaces with CodeGraph, ast-grep, LSP/compiler diagnostics, and residual checks in separate authority lanes.

**Tech Stack:** Tauri 2, Rust, React 19, TypeScript, pnpm, Cargo, CodeGraph via `npx -y @colbymchenry/codegraph`, ast-grep via `npx -p @ast-grep/cli`, rust-analyzer, tsserver.

---

## Execution Status

- Task 1 and Task 2 prerequisites were completed in commit `63ed0ac`.
- Task 3 prerequisite matrix is complete in `docs/superpowers/refactor/wenlan-app-parity-matrix.md`.
- CodeGraph evaluation is complete in `docs/superpowers/refactor/2026-06-25-codegraph-evaluation.md`; `.codegraph/` is ignored local cache.
- Task 4 typed-client convergence is complete in commit `7e32190` with follow-up doc pinning in `8e231c`; current `origin_types` residual count is `0`.
- Task 5 sidecar compatibility is complete in commit `c7f44fb` with follow-up sidecar bridge hardening through `9e1d10d`; current `cargo build` exits 0.
- Current renamed repo/worktree context is recorded in `docs/superpowers/refactor/2026-06-26-wenlan-app-goal-context.md`.
- Next implementation work should start from Slice B1/B2 in the goal-context document, not from the historical Task 4/5 checkboxes below.

## Tool Boundary Protocol

Use this split for every remaining cross-cutting task.

| Tool | Use for | Required before | Do not use for |
|---|---|---|---|
| CodeGraph | symbol navigation, dependency orientation, impact/blast-radius discovery, initial affected-test hints | typed-client edits, sidecar edits, MCP bridge edits, runtime identity edits | deterministic inventory counts, rewrite safety, type correctness |
| ast-grep | repeatable syntax inventory, structural command/DTO/wrapper lists, codemod candidate lists | bulk rename, local DTO removal, Tauri command inventory, frontend `invoke(...)` inventory | semantic call graph, compiler correctness |
| LSP/compiler | type/import/signature diagnostics and semantic correctness | after each narrow edit batch and before claiming typed migration is green | route parity discovery, product-surface classification |
| tests/builds | behavior and integration evidence | task completion | scope discovery |
| `rg` | residual text checks and allowlist enforcement | after graph and structural scopes are known | first-pass cross-file planning |
| `grep` | last-resort bounded text search | only when CodeGraph, ast-grep, LSP/compiler, tests/builds, and `rg` are unavailable or unsuitable | structural inventory, semantic correctness, rewrite safety |

Canonical sequence, shown for the typed-client task. Later tasks list their own exact target symbols and files.

```bash
CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 npx -y @colbymchenry/codegraph sync .
CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 npx -y @colbymchenry/codegraph query WenlanClient --json
CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 npx -y @colbymchenry/codegraph impact WenlanClient --json
npx -y -p @ast-grep/cli sg run -p 'pub struct $NAME { $$$FIELDS }' -l rs app/src
cargo check
pnpm build
rg -n 'origin-types|origin_types::|use origin_types|OriginClient|serde_json::Value' app/Cargo.toml app/src
grep -RIn --exclude-dir=node_modules --exclude-dir=target --exclude-dir=.codegraph 'OriginClient' app src
```

Boundary notes:

- CodeGraph output may guide what to read next, but it does not replace deterministic inventory or verification.
- ast-grep output can gate counts and residual syntax, but it does not prove type correctness or runtime behavior.
- LSP/compiler diagnostics can approve imports/types/signatures, but they do not prove feature parity or migration safety.
- `codegraph affected` test hints are advisory; run real targeted tests plus the task's required build/test command.
- If CodeGraph is unavailable, record the exact failure and fall back to ast-grep + LSP + `rg`; do not silently skip the blast-radius step.
- If those lanes also fail or are unavailable for the current surface, use bounded `grep` with include/exclude filters and record why every stronger tool was unavailable.

### Task 1: Lock Baseline Dependency Setup

**Files:**
- Create: `pnpm-workspace.yaml`
- Verify: `package.json`

- [ ] **Step 1: Keep pnpm build approval explicit**

```yaml
allowBuilds:
  esbuild: true
```

- [ ] **Step 2: Verify offline install**

Run:

```bash
pnpm install --frozen-lockfile --offline
```

Expected: exits 0 and does not print `ERR_PNPM_IGNORED_BUILDS`.

- [ ] **Step 3: Verify frontend baseline**

Run:

```bash
pnpm test
```

Expected: `35 passed`, `319 passed`, `1 skipped`.

- [ ] **Step 4: Record Rust sidecar baseline failure**

Run:

```bash
cargo build
```

Expected for this baseline phase: failure with:

```text
resource path `binaries/origin-server-aarch64-apple-darwin` doesn't exist
```

This failure is tracked as a migration prerequisite, not fixed by hand-copying binaries.

- [ ] **Step 5: Commit**

```bash
git add pnpm-workspace.yaml
git commit -m "chore: make pnpm build approval explicit"
```

### Task 2: Add Structural Inventory Harness

**Files:**
- Create: `scripts/refactor/inventory.sh`
- Create: `docs/superpowers/refactor/2026-06-25-wenlan-app-tooling.md`
- Generated: `docs/superpowers/refactor/wenlan-app-inventory/summary.md`

- [ ] **Step 1: Add the inventory script**

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
OUT="$ROOT/docs/superpowers/refactor/wenlan-app-inventory"
SG=(npx -y -p @ast-grep/cli sg)

mkdir -p "$OUT"

"${SG[@]}" outline "$ROOT/src/lib/tauri.ts" > "$OUT/tauri-ts-outline.txt"
"${SG[@]}" outline "$ROOT/app/src/api.rs" > "$OUT/api-rs-outline.txt"
"${SG[@]}" outline "$ROOT/app/src/search.rs" > "$OUT/search-rs-outline.txt"
"${SG[@]}" run -p 'invoke($CMD, $$$ARGS)' -l ts "$ROOT/src" > "$OUT/frontend-invokes.txt"
"${SG[@]}" run -p 'pub struct $NAME { $$$FIELDS }' -l rs "$ROOT/app/src" > "$OUT/rust-structs.txt"
```

- [ ] **Step 2: Make it executable**

Run:

```bash
chmod +x scripts/refactor/inventory.sh
```

- [ ] **Step 3: Generate the inventory**

Run:

```bash
bash scripts/refactor/inventory.sh
```

Expected: `Inventory written to .../docs/superpowers/refactor/wenlan-app-inventory`.

- [ ] **Step 4: Verify key counts in the generated summary**

Run:

```bash
sed -n '1,80p' docs/superpowers/refactor/wenlan-app-inventory/summary.md
```

Expected counts are allowed to change only when the related surface is intentionally refactored.

- [ ] **Step 5: Commit**

```bash
git add scripts/refactor/inventory.sh docs/superpowers/refactor
git commit -m "chore: add wenlan app structural inventory"
```

### Task 3: Produce the Parity Matrix

**Files:**
- Create: `docs/superpowers/refactor/wenlan-app-parity-matrix.md`
- Read: `src/lib/tauri.ts`
- Read: `app/src/search.rs`
- Read: `app/src/api.rs`
- Read: `/Users/lucian/Repos/wenlan/crates/wenlan-server/src/router.rs`
- Read: `/Users/lucian/Repos/wenlan/crates/wenlan-types/src`

- [ ] **Step 1: Generate current app inventory**

Run:

```bash
bash scripts/refactor/inventory.sh
```

Expected: generated inventory files exist under `docs/superpowers/refactor/wenlan-app-inventory/`.

- [ ] **Step 2: Extract current Wenlan routes**

Run from `/Users/lucian/Repos/wenlan`:

```bash
rg -n 'route\(|nest\(' crates/wenlan-server/src/router.rs
```

Expected: route list includes refinement queue, page graph/orphans, pending revisions, enrichment status, sources/config, status/reranker, and page export.

- [ ] **Step 3: Extract current Wenlan wire responses**

Run from `/Users/lucian/Repos/wenlan`:

```bash
rg -n 'struct (RerankerStatus|StatusResponse|EnrichmentStatusResponse|PageLinksResponse|OrphanLinksResponse|ListMemoryRevisionsResponse|ListPageRevisionsResponse|ListRefinementsResponse|AcceptRefinementResponse|RejectRefinementResponse|RevisionAcceptResponse|RevisionDismissResponse|ContradictionDismissResponse)' crates/wenlan-types/src
```

Expected: every named response resolves to a concrete `wenlan-types` definition.

- [ ] **Step 4: Write matrix rows**

Create rows with these columns:

```markdown
| Surface | Current source | App status | Action | Daemon compatibility |
|---|---|---|---|---|
| `/api/refinery/queue` | `wenlan-server` route + `ListRefinementsResponse` | missing UI | add P0 review queue | require route, hide UI if absent |
```

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/refactor/wenlan-app-parity-matrix.md
git commit -m "docs: add wenlan app parity matrix"
```

### Task 4: Plan Typed Client Convergence

**Files:**
- Modify: `app/Cargo.toml`
- Modify: `app/src/api.rs`
- Modify: `app/src/state.rs`
- Modify: `app/src/search.rs`
- Modify: `app/src/router/intent.rs`
- Modify: `app/src/events.rs`
- Modify: `app/src/system_info.rs`
- Modify: `app/src/config.rs`
- Modify: `app/src/sources/mod.rs`
- Modify: `app/src/sources/data_source.rs`
- Modify: `app/src/sources/local_files.rs`
- Modify: `src/lib/tauri.ts`

- [ ] **Step 1: Capture CodeGraph typed-client blast radius**

Run:

```bash
CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 npx -y @colbymchenry/codegraph sync .
CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 npx -y @colbymchenry/codegraph query OriginClient --json
CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 npx -y @colbymchenry/codegraph impact OriginClient --json
CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 npx -y @colbymchenry/codegraph query origin_types --json
```

Expected: output names `app/src/api.rs`, `app/src/state.rs`, `app/src/search.rs`, and `origin_types` import sites. Record any additional impacted files in the task notes before editing.

- [ ] **Step 2: Write the deterministic Rust import check**

Run:

```bash
rg -n 'origin-types|origin_types|OriginClient' app/Cargo.toml app/src
```

Expected before implementation: matches exist in `app/Cargo.toml`, `app/src/api.rs`, `app/src/state.rs`, `app/src/search.rs`, and supporting app modules that import shared wire types.

- [ ] **Step 3: Capture ast-grep structural surfaces**

Run:

```bash
npx -y -p @ast-grep/cli sg outline app/src/api.rs
npx -y -p @ast-grep/cli sg outline app/src/search.rs
npx -y -p @ast-grep/cli sg run -p 'pub struct $NAME { $$$FIELDS }' -l rs app/src
```

Expected: output identifies the HTTP client methods, Tauri commands, and local Rust DTO shadows. Use this as the deterministic edit surface; CodeGraph remains the orientation layer.

- [ ] **Step 4: Replace the crate dependency**

Target dependency:

```toml
wenlan-types = "0.9.1"
```

Use the version required by the current daemon release. This plan is pinned to the local Wenlan workspace version observed in `/Users/lucian/Repos/wenlan/Cargo.toml` on 2026-06-25.

- [ ] **Step 5: Rename the client seam**

Target names:

```rust
pub struct WenlanClient {
    client: Client,
    base_url: String,
}
```

```rust
pub client: WenlanClient,
```

- [ ] **Step 6: Run LSP/compiler semantic checks**

Run:

```bash
cargo check
pnpm build
```

Expected: `cargo check` may still fail on the known sidecar/build contract if Task 5 has not run, but it must not fail on unresolved `origin_types`, `OriginClient`, or request/response type names introduced by this task. `pnpm build` must not report TypeScript wrapper type errors from the typed-client rename.

- [ ] **Step 7: Verify no stale typed imports remain**

Run:

```bash
rg -n 'origin-types|origin_types::|use origin_types|OriginClient' app/Cargo.toml app/src
```

Expected after implementation: only intentional compatibility comments or fixtures remain.

- [ ] **Step 8: Commit**

```bash
git add app/Cargo.toml app/src/api.rs app/src/state.rs app/src/search.rs app/src/router/intent.rs app/src/events.rs app/src/system_info.rs app/src/config.rs app/src/sources/mod.rs app/src/sources/data_source.rs app/src/sources/local_files.rs src/lib/tauri.ts
git commit -m "refactor: converge app client on wenlan types"
```

### Task 5: Plan Sidecar Compatibility Before Public Rename

**Files:**
- Modify: `package.json`
- Modify: `app/tauri.conf.json`
- Modify: `app/capabilities/default.json`
- Modify: `app/src/lib.rs`
- Modify: `app/src/lifecycle.rs`

- [ ] **Step 1: Capture CodeGraph sidecar and lifecycle blast radius**

Run:

```bash
CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 npx -y @colbymchenry/codegraph sync .
CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 npx -y @colbymchenry/codegraph query spawn_sidecar --json
CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 npx -y @colbymchenry/codegraph query origin-server --json
CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 npx -y @colbymchenry/codegraph affected app/src/lib.rs --json
CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 npx -y @colbymchenry/codegraph affected app/src/lifecycle.rs --json
```

Expected: output or explicit no-match notes for sidecar spawn, sidecar names, lifecycle tests, and affected frontend/Rust tests. Record no-match output; it is useful evidence for choosing ast-grep/`rg` follow-up.

- [ ] **Step 2: Capture current failing sidecar contract**

Run:

```bash
cargo build
```

Expected before implementation:

```text
resource path `binaries/origin-server-aarch64-apple-darwin` doesn't exist
```

- [ ] **Step 3: Capture ast-grep and residual sidecar surfaces**

Run:

```bash
npx -y -p @ast-grep/cli sg outline app/src/lib.rs
npx -y -p @ast-grep/cli sg outline app/src/lifecycle.rs
rg -n 'origin-server|origin-mcp|cloudflared|externalBin|com\.origin|Origin\.app|Wenlan\.app' package.json app/tauri.conf.json app/capabilities/default.json app/src
```

Expected: deterministic list of config, sidecar, LaunchAgent, and stable app path references. This list gates the edit surface; do not rely on CodeGraph alone for string/config references.

- [ ] **Step 4: Define sidecar names without public app rename**

Target rule:

```text
Use wenlan-server and wenlan-mcp sidecar binaries for new builds, while detecting old Origin launch agents and config during bridge releases.
```

- [ ] **Step 5: Add tests around stable app target validation**

Required cases:

```text
/Applications/Wenlan.app accepted
~/Applications/Wenlan.app accepted
/Applications/Origin.app detected as legacy migration state
random Downloads app path rejected
```

- [ ] **Step 6: Run LSP/compiler semantic checks**

Run:

```bash
cargo check
```

Expected: no unresolved sidecar constants, lifecycle function names, or stale `origin-server` binary path references introduced by the edit. Remaining failures must be listed with exact error text.

- [ ] **Step 7: Verify sidecar build reaches app code**

Run:

```bash
cargo build
```

Expected after implementation: no missing `binaries/origin-server-*` error.

- [ ] **Step 8: Commit**

```bash
git add package.json app/tauri.conf.json app/capabilities/default.json app/src/lib.rs app/src/lifecycle.rs
git commit -m "refactor: prepare wenlan sidecar compatibility"
```
