# Default Save Space Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Wenlan one daemon-owned default save destination, truthful write receipts, and consistent strict-pin versus overridable-context behavior across the daemon, CLI, MCP, and both bundled plugins.

**Architecture:** `wenlan-types` owns the tri-state write target and additive receipt enums. `wenlan-core::space_context` resolves named candidates to stable Space IDs and `MemoryDB` persists one default Space plus a one-time legacy migration watermark. Axum handlers apply the resolver only to top-level writes; clients supply higher-priority local context and display the daemon-returned destination.

**Tech Stack:** Rust 2021, serde/serde_json, libSQL, Axum 0.8, reqwest, clap, rmcp, Bash 3.2-compatible resolver scripts.

## Global Constraints

- Default save space affects new Memory/import, Page, and Entity writes only; unscoped reads remain All Spaces.
- `WENLAN_SPACE` remains a strict official-client pin and is not an authorization boundary.
- `WENLAN_DEFAULT_SPACE` is overridable by an explicit per-call Space.
- Missing write `space` means inherit, JSON `null` means Uncategorized, and a non-empty string means a named registered Space.
- Request body beats header; header beats daemon default; daemon default beats Uncategorized.
- Stable Space ID survives rename; deletion of an explicit/header target fails, while deletion of the daemon default falls back to Uncategorized.
- Existing Entity/Page matches never move; receipts report their actual persisted destination.
- Pages persist one resolved destination to both `pages.space` and `pages.workspace`.
- Relation and Observation do not acquire independent Space arguments.
- No new dependency may be added to `wenlan-types`.
- `wenlan-app` is not edited in this plan; `/api/status` advertises the additive `default_save_space` capability for its separate Stage 3 PR.

---

### Task 1: Shared wire contract and default persistence

**Files:**
- Create: `crates/wenlan-types/src/space_context.rs`
- Modify: `crates/wenlan-types/src/lib.rs`
- Modify: `crates/wenlan-types/src/memory.rs`
- Modify: `crates/wenlan-types/src/requests.rs`
- Modify: `crates/wenlan-types/src/responses.rs`
- Modify: `crates/wenlan-core/src/lib.rs`
- Create: `crates/wenlan-core/src/space_context.rs`
- Modify: `crates/wenlan-core/src/db.rs`

**Interfaces:**
- Produces: `WriteSpaceTarget::{Inherit, Uncategorized, Named(String)}` with missing/null/string wire behavior.
- Produces: `WriteSpaceSource::{Request, Header, Default, Uncategorized, Existing}` and `WriteOutcome::{Created, ResolvedExisting, AttachedExisting, Unknown}`.
- Produces: `ResolvedWriteSpace { space_id: Option<String>, space_name: Option<String>, source: WriteSpaceSource }`.
- Produces: `MemoryDB::{get_default_space,set_default_space,clear_default_space,resolve_write_space,finalize_write_space,import_legacy_default_once}`.

- [ ] **Step 1: Write failing wire tests**

Add tests in `space_context.rs` that deserialize `{"space": omitted}`, `{"space": null}`, and `{"space":"Work"}` through a fixture struct and assert `Inherit`, `Uncategorized`, and `Named("Work")`; serialize the latter two as `null` and `"Work"`. Add response tests proving absent receipt fields deserialize to `None` and unknown enum strings deserialize to `Unknown`.

- [ ] **Step 2: Run the wire tests and verify RED**

Run:

```bash
cargo test -p wenlan-types space_context -- --nocapture
```

Expected: compilation fails because `space_context` and the new enums do not exist.

- [ ] **Step 3: Implement the shared wire types**

Create the enum module with custom `Serialize`/`Deserialize` for `WriteSpaceTarget`, `Default` returning `Inherit`, and `is_inherit()` for `skip_serializing_if`. Add optional `space`, `space_source`, and `write_outcome` receipt fields to the four affected response types; add `is_default` to `Space`; add `DefaultSpaceResponse` and `SetDefaultSpaceRequest`.

- [ ] **Step 4: Run the wire tests and verify GREEN**

Run:

```bash
cargo test -p wenlan-types space_context -- --nocapture
```

Expected: all `space_context` tests pass.

- [ ] **Step 5: Write failing database lifecycle tests**

In `db.rs`, add tests named:

```rust
default_space_set_replace_rename_delete_lifecycle
legacy_default_import_runs_once_without_rewriting_toml
write_space_resolution_carries_stable_id_through_rename
```

Use a temporary DB and temporary TOML path. Assert the Uncategorized sentinel cannot become default, exactly one registered row is marked, rename preserves the ID/default, delete clears it, the migration watermark prevents resurrection, and finalization follows rename.

- [ ] **Step 6: Run the database tests and verify RED**

Run:

```bash
cargo test -p wenlan-core --lib default_space_ -- --nocapture
cargo test -p wenlan-core --lib legacy_default_import_runs_once -- --nocapture
cargo test -p wenlan-core --lib write_space_resolution_carries -- --nocapture
```

Expected: compilation fails because the default-space methods and migration are absent.

- [ ] **Step 7: Implement migration 81 and the core resolver**

Add `spaces.is_default INTEGER NOT NULL DEFAULT 0`, a partial unique index excluding the unfiled sentinel, and a durable app-metadata watermark. Resolve names to stable IDs before async work; at finalization, query the current name by ID while holding the DB connection. Return validation failure for deleted explicit/header IDs and Uncategorized for a deleted default ID.

- [ ] **Step 8: Run focused core tests and verify GREEN**

Run:

```bash
cargo test -p wenlan-core --lib default_space_ -- --nocapture
cargo test -p wenlan-core --lib legacy_default_import_runs_once -- --nocapture
cargo test -p wenlan-core --lib write_space_resolution_carries -- --nocapture
```

Expected: all focused tests pass.

- [ ] **Step 9: Commit**

```bash
git add crates/wenlan-types crates/wenlan-core
git commit -m "fix: add default save space contract"
```

---

### Task 2: Daemon default API, capability, and Memory receipts

**Files:**
- Modify: `crates/wenlan-server/src/router.rs`
- Modify: `crates/wenlan-server/src/routes.rs`
- Modify: `crates/wenlan-server/src/memory_routes.rs`
- Modify: `crates/wenlan-server/src/ingest_batcher.rs`
- Modify: `crates/wenlan-server/src/state.rs`

**Interfaces:**
- Consumes: Task 1 wire enums and `MemoryDB` default/resolution methods.
- Produces: `GET|PUT|DELETE /api/spaces/default`.
- Produces: `/api/status.capabilities` containing `default_save_space`.
- Produces: Memory store receipts with actual `space`, `space_source`, and `write_outcome`.

- [ ] **Step 1: Write failing route tests**

Add router-level tests covering get-with-none, set, replace, clear, invalid sentinel, status capability, body named over header, explicit null over header/default, header over default, default over Uncategorized, unknown named error, and default deletion during a store.

- [ ] **Step 2: Run route tests and verify RED**

Run:

```bash
cargo test -p wenlan-server default_space -- --nocapture
cargo test -p wenlan-server store_space_precedence -- --nocapture
```

Expected: route tests fail with 404 or missing receipt fields.

- [ ] **Step 3: Implement routes and Memory write resolution**

Register the three default routes. Add `capabilities: vec!["default_save_space".into()]` to status. Resolve the `WriteSpaceTarget` before building `RawDocument`, carry `ResolvedWriteSpace` beside the coalesced request, and finalize the stable ID immediately before the batch transaction. Set the persisted name on `RawDocument.space`; return the finalized name/source instead of predicting from the request.

- [ ] **Step 4: Run focused daemon tests and verify GREEN**

Run:

```bash
cargo test -p wenlan-server default_space -- --nocapture
cargo test -p wenlan-server store_space_precedence -- --nocapture
```

Expected: all focused tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/wenlan-server
git commit -m "fix: apply daemon default save space"
```

---

### Task 3: Import, Entity, Page, and derived-write semantics

**Files:**
- Modify: `crates/wenlan-server/src/import_routes.rs`
- Modify: `crates/wenlan-server/src/memory_routes.rs`
- Modify: `crates/wenlan-core/src/importer.rs`
- Modify: `crates/wenlan-core/src/post_write.rs`
- Modify: `crates/wenlan-core/src/post_ingest.rs`
- Modify: `crates/wenlan-core/src/db.rs`

**Interfaces:**
- Consumes: Task 1 `ResolvedWriteSpace` and receipt enums.
- Produces: batch import destination receipts.
- Produces: Entity `created` versus `resolved_existing` receipts.
- Produces: Page `created` versus `attached_existing` receipts.
- Produces: extraction batches partitioned by persisted Space.

- [ ] **Step 1: Write failing behavior tests**

Add tests named:

```rust
import_uses_one_resolved_space_and_does_not_move_duplicates
create_entity_reports_existing_owner_without_moving
create_page_rejects_conflicting_space_aliases
create_page_mirrors_one_resolved_destination
recent_extraction_batch_never_mixes_spaces
```

Assert literal receipt values and persisted rows, not helper return values.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
cargo test -p wenlan-server import_uses_one_resolved_space -- --nocapture
cargo test -p wenlan-server create_entity_reports_existing_owner -- --nocapture
cargo test -p wenlan-server create_page_ -- --nocapture
cargo test -p wenlan-core --lib recent_extraction_batch_never_mixes_spaces -- --nocapture
```

Expected: tests fail because default resolution/receipts/partitioning are absent.

- [ ] **Step 3: Implement import and Entity receipts**

Delete the server-local duplicate import request/response structs and use `wenlan-types`. Resolve the batch destination once and pass its finalized name to the importer. For Entity resolve-or-create, return the persisted Entity Space and choose `Created` or `ResolvedExisting` without moving an existing row.

- [ ] **Step 4: Implement Page alias and receipt rules**

Normalize `workspace` and `space`: accept either, accept equal duplicates, reject mismatches, resolve once, and write the same final name to both columns. On dedup attachment, load the existing Page and return `AttachedExisting` plus its persisted Space.

- [ ] **Step 5: Partition derived extraction by Space**

Change `find_recent_batch` to accept the anchor's persisted Space and add a null-safe predicate so the query returns only rows in that same Space partition. Thread the anchor Space through `post_ingest`; newly created derived Entities receive that Space while global existing-Entity resolution remains unchanged.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run:

```bash
cargo test -p wenlan-server import_uses_one_resolved_space -- --nocapture
cargo test -p wenlan-server create_entity_reports_existing_owner -- --nocapture
cargo test -p wenlan-server create_page_ -- --nocapture
cargo test -p wenlan-core --lib recent_extraction_batch_never_mixes_spaces -- --nocapture
```

Expected: all focused tests pass.

- [ ] **Step 7: Commit**

```bash
git add crates/wenlan-server crates/wenlan-core
git commit -m "fix: preserve scoped write destinations"
```

---

### Task 4: CLI identity, local context, and daemon-backed default commands

**Files:**
- Create: `crates/wenlan-cli/src/space_context.rs`
- Modify: `crates/wenlan-cli/src/main.rs`
- Modify: `crates/wenlan-cli/src/client.rs`
- Modify: `crates/wenlan-cli/src/commands/space.rs`
- Modify: `crates/wenlan-cli/src/commands/store.rs`
- Modify: `crates/wenlan-cli/src/commands/search.rs`
- Modify: `crates/wenlan-cli/src/commands/recall.rs`
- Modify: `crates/wenlan-cli/src/commands/list.rs`

**Interfaces:**
- Produces: global `--agent-name`, `--space`, and `--all-spaces`.
- Produces: `resolve_cli_space(explicit, all_spaces, cwd, operation) -> Result<CliSpaceContext>`.
- Produces: daemon-backed `wenlan spaces default [name|--clear]`.

- [ ] **Step 1: Write failing CLI resolver tests**

Cover strict `WENLAN_SPACE`, conflicting flags, `WENLAN_DEFAULT_SPACE`, longest cwd mapping, registered repo basename, unregistered repo skip, write omission, read All Spaces, and agent-name precedence. Use temporary directories and a fake daemon response for registered Spaces.

- [ ] **Step 2: Run CLI tests and verify RED**

Run:

```bash
cargo test -p wenlan cli_space_ -- --nocapture
cargo test -p wenlan agent_name_ -- --nocapture
```

Expected: tests fail because the global arguments and resolver do not exist.

- [ ] **Step 3: Implement the resolver and common headers**

Parse global options once. Build a `WenlanClient` carrying optional `X-Agent-Name` and optional Space header. Resolve strict pin first; otherwise explicit, `WENLAN_DEFAULT_SPACE`, longest mapping, registered repo basename, then omission. Reads honor `--all-spaces`; writes reject it.

- [ ] **Step 4: Replace TOML default commands**

Remove `read_default_from_toml` and `set_default_in_toml`. Add client methods for get/set/clear default and make list/show render daemon `is_default`. Implement `--clear` as mutually exclusive with a name.

- [ ] **Step 5: Render truthful write receipts**

For human output, print the actual daemon-returned destination and outcome. JSON passes through literal receipt fields. Quiet output remains empty.

- [ ] **Step 6: Run CLI tests and verify GREEN**

Run:

```bash
cargo test -p wenlan cli_space_ -- --nocapture
cargo test -p wenlan agent_name_ -- --nocapture
cargo test -p wenlan commands::space -- --nocapture
```

Expected: all focused CLI tests pass.

- [ ] **Step 7: Commit**

```bash
git add crates/wenlan-cli
git commit -m "fix: unify CLI space context"
```

---

### Task 5: MCP fallback context and truthful tool receipts

**Files:**
- Modify: `crates/wenlan-mcp/src/lock_state.rs`
- Modify: `crates/wenlan-mcp/src/client.rs`
- Modify: `crates/wenlan-mcp/src/tools.rs`
- Modify: `crates/wenlan-mcp/src/main.rs`

**Interfaces:**
- Consumes: shared response enums from Task 1.
- Produces: strict `WENLAN_SPACE` behavior unchanged.
- Produces: explicit tool Space over `WENLAN_DEFAULT_SPACE`.
- Produces: capture/Entity/Page messages from persisted receipt fields.

- [ ] **Step 1: Write failing MCP tests**

Add tests proving strict-pin schema/runtime behavior remains, fallback leaves schemas visible, explicit tool Space beats fallback, fallback fills an omitted argument, and receipt formatters report default/Uncategorized/existing destinations.

- [ ] **Step 2: Run MCP tests and verify RED**

Run:

```bash
cargo test -p wenlan-mcp default_space -- --nocapture
cargo test -p wenlan-mcp receipt -- --nocapture
```

Expected: fallback and receipt tests fail.

- [ ] **Step 3: Implement fallback state and receipt formatting**

Read both environment variables at startup. Keep schema gating and cached-schema runtime enforcement tied only to strict `WENLAN_SPACE`. Make `effective_space` return strict pin, then explicit argument, then `WENLAN_DEFAULT_SPACE`. Use receipt enums and actual `space` for tool success prose.

- [ ] **Step 4: Run MCP tests and verify GREEN**

Run:

```bash
cargo test -p wenlan-mcp default_space -- --nocapture
cargo test -p wenlan-mcp receipt -- --nocapture
```

Expected: all focused MCP tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/wenlan-mcp
git commit -m "fix: add overridable MCP space context"
```

---

### Task 6: Claude/Codex resolver parity and skill copy

**Files:**
- Modify: `plugin/bin/resolve-space.sh`
- Modify: `plugin-codex/bin/resolve-space.sh`
- Modify: `plugin/bin/test/test-resolve-space.sh`
- Modify: `plugin-codex/bin/test/test-resolve-space.sh`
- Modify: `plugin/skills/README.md`
- Modify: space-aware skills under `plugin/skills/` and `plugin-codex/skills/` only where their source labels or receipts change
- Modify: `scripts/validate-plugin-contract.py`
- Modify: `scripts/validate-plugin-contract.test.sh`
- Modify: `scripts/validate-codex-plugin-slice.py`

**Interfaces:**
- Produces: identical resolver scripts with source labels `locked-env`, `arg`, `default-env`, `cwd-config`, `cwd-repo`, and `unscoped`.
- Removes: top-level TOML default and topic fallback as resolution layers.
- Requires: repo basename is returned only when daemon Space registration can be confirmed.

- [ ] **Step 1: Write failing resolver tests**

Add shell cases for strict pin over arg, explicit over `WENLAN_DEFAULT_SPACE`, longest mapping, ignored legacy default, no topic fallback, registered repo basename, and skipped unregistered basename. Provide a temporary fake `wenlan` executable that returns controlled `spaces list --json` data.

- [ ] **Step 2: Run resolver tests and verify RED**

Run:

```bash
bash plugin/bin/test/test-resolve-space.sh
bash plugin-codex/bin/test/test-resolve-space.sh
```

Expected: legacy default/topic and unvalidated repo cases fail.

- [ ] **Step 3: Implement both resolvers and update copy**

Keep Bash 3.2 compatibility. Resolve strict pin before explicit. Use `WENLAN_DEFAULT_SPACE` after explicit. Parse mappings by longest prefix. Validate repo basename against `wenlan spaces list --json`; on daemon/CLI failure, skip repo inference and return unscoped. Remove topic/default source labels and update skill receipts to use actual daemon response when available.

- [ ] **Step 4: Run plugin tests and validators**

Run:

```bash
bash plugin/bin/test/test-resolve-space.sh
bash plugin-codex/bin/test/test-resolve-space.sh
python3 scripts/validate-plugin-contract.py
bash scripts/validate-plugin-contract.test.sh
python3 scripts/validate-codex-plugin-slice.py
```

Expected: all commands exit 0.

- [ ] **Step 5: Commit**

```bash
git add plugin plugin-codex scripts
git commit -m "fix: align plugin space resolution"
```

---

### Task 7: Integrated verification, review, and PR publication

**Files:**
- Modify: `docs/superpowers/specs/2026-07-27-default-save-space-design.md`
- Add: `docs/superpowers/plans/2026-07-27-default-save-space-implementation.md`
- Modify: user-facing README/help text only when focused tests reveal stale semantics

**Interfaces:**
- Produces: one reviewed branch and one PR against `origin/main`.

- [ ] **Step 1: Reconcile the implementation against every spec section**

Check Default lifecycle, tri-state wire, precedence, strict/fallback envs, reads, receipts, Page aliasing, Entity behavior, batching, capability, migration, CLI, MCP, and both plugin slices. Add a focused regression test before correcting any discovered behavior gap.

- [ ] **Step 2: Run formatting and focused validators**

```bash
cargo fmt --all -- --check
python3 scripts/validate-plugin-contract.py
bash scripts/validate-plugin-contract.test.sh
python3 scripts/validate-codex-plugin-slice.py
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 3: Run the repository verification floor**

```bash
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --lib
cargo test -p wenlan-mcp --all-targets
```

Expected: every command exits 0 with zero failing tests.

- [ ] **Step 4: Run fresh integrated code review**

Review the complete diff from `1cdec7955aeec847fa9d799b7c5de52aed4f833b` with a read-only reviewer focused on correctness, migration/data loss, async rename/delete races, cross-Space leakage, wire compatibility, and spec drift. Fix every Critical/Important issue with RED-first tests and repeat the relevant floor.

- [ ] **Step 5: Commit documentation and final fixes**

```bash
git add -f docs/superpowers/specs/2026-07-27-default-save-space-design.md docs/superpowers/plans/2026-07-27-default-save-space-implementation.md
git add crates plugin plugin-codex scripts
git commit -m "docs: finalize default save space rollout"
```

- [ ] **Step 6: Verify the exact PR tree**

```bash
git status --short
git diff --check origin/main...HEAD
cargo test --workspace --lib
```

Expected: clean status, clean diff check, and zero failing tests.

- [ ] **Step 7: Push and create the PR**

```bash
git push -u origin codex/mcp-surface-consolidation
gh pr create --base main --head codex/mcp-surface-consolidation --title "fix: add default save space contract" --body "Adds one daemon-owned Default save space for new Memories, imports, Pages, and Entities while keeping reads global by default. Preserves the released WENLAN_SPACE strict pin, adds overridable WENLAN_DEFAULT_SPACE context, migrates the legacy TOML default once without dual writes, and returns truthful persisted-destination receipts across CLI/MCP/plugins. Verification: cargo fmt --all -- --check; cargo clippy --workspace --all-targets -- -D warnings; cargo test --workspace --lib; cargo test -p wenlan-mcp --all-targets; plugin contract validators. wenlan-app adoption remains a separate capability-gated Stage 3 PR."
```

The PR body must lead with user-visible behavior, list compatibility/migration decisions, enumerate exact verification commands, and state that `wenlan-app` support is a separate capability-gated Stage 3 PR.
