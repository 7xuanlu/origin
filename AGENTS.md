# AGENTS.md

This repo holds the **daemon** (`wenlan-server`), the **CLI** (`wenlan`), shared **wire types** (`wenlan-types`), the **business-logic core** (`wenlan-core`), the **MCP server** (`wenlan-mcp`), and the **Tauri desktop app** (`wenlan-app`, in `app/`). All six ship from this monorepo (the app was folded back in on 2026-07-20, reversing a 2026-05-07 split). Public product surface lives at [wenlan.app](https://wenlan.app) (marketing, docs at `/docs`, longer-form writing at `/learn`).

## Repo map

Where things live. Subtree `AGENTS.md` files load automatically when you work under them (the [agents.md](https://agents.md/) hierarchy), so this file stays the always-loaded, cross-cutting layer and the detail lives next to its code.

| Working on… | Start here |
|---|---|
| Cross-cutting rules — crate boundaries, async & SQL safety, dev gotchas, release policy | **this file** (loaded every session) |
| Business logic — DB, engine, classify/extract, rerank, pages, retrieval + the deep flag reference | `crates/wenlan-core/AGENTS.md` |
| HTTP daemon — router, routes, state, ingest batcher, scheduler, websocket | `crates/wenlan-server/AGENTS.md` |
| Eval discipline — fixtures, baselines, seed scripts, cache TTL, faithfulness benches | `app/eval/AGENTS.md` |
| Eval internals — runner conventions, paired-A/B apparatus, the G3 gate | `crates/wenlan-core/src/eval/AGENTS.md` |
| Test layers — what runs at L1-L8, where, when, whether it blocks | `docs/test-layers.md` |
| Platform code — per-OS data dirs, service registration, GPU backends, Windows verification | `docs/cross-platform.md` |
| Running & verifying against live surfaces — daemon launch/lifecycle, per-surface drive recipes, mutation audit / behavior trace / weekly sweep | `.claude/skills/run-wenlan/SKILL.md` → `.claude/skills/verify/SKILL.md` → `.claude/skills/prove/SKILL.md` (tracked in-repo) |

## Build & Dev Commands

Wenlan is a Cargo workspace with 6 crates: `wenlan-types`, `wenlan-core`, `wenlan-server`, `wenlan` (CLI in `crates/wenlan-cli`), `wenlan-mcp`, and `wenlan-app` (the desktop app, in `app/`). `default-members` covers the 5 daemon crates only, so routine `cargo build`/`test`/`clippy` never compile the desktop app; build it explicitly with `-p wenlan-app`.

```bash
# Daemon as a managed launchd/systemd/schtasks service (not a plain cargo run):
cargo build -p wenlan -p wenlan-server
./target/debug/wenlan setup --basic       # configure local memory
./target/debug/wenlan background on       # writes plist, launchctl load
./target/debug/wenlan background off      # when done

bash scripts/coverage.sh                  # HTML coverage, opens in browser
bash scripts/setup-hooks.sh               # one-time: .githooks/pre-commit + pre-push

# Eval baselines are #[ignore]d — they need Qwen 3.5-9B on a Metal GPU:
cargo test -p wenlan-core --test eval_harness save_locomo_baseline -- --ignored --nocapture
cargo test -p wenlan-core --test eval_harness save_longmemeval_baseline -- --ignored --nocapture
# Baselines land in <EVAL_BASELINES_DIR>/*.json (gitignored, default ~/.cache/origin-eval).
```

Pre-commit checks Rust formatting without changing the worktree and runs Clippy on directly changed crates. Pre-push uses the CI planner to run Clippy and tests over the affected reverse-dependency closure.

## Cross-platform

Supported: macOS arm64, Linux x86_64/aarch64 (glibc), Windows x86_64. macOS x86_64 is not a stock source-build target — the pinned ONNX Runtime has no prebuilt Intel macOS binary, so a custom build must compile it separately and point `ORT_LIB_LOCATION` at the result.

**Per-OS data dirs and service registration, the llama-cpp-2 GPU backends, ORT on Windows, the three manual Windows GPU verification legs, and the Linux-smoke-from-macOS recipe are in [`docs/cross-platform.md`](docs/cross-platform.md).** Read it before touching platform-conditional code or the release matrix.

## Test layers

Eight layers run, from the IDE to pre-release evals. The split answers three questions: can a hosted runner do this (no GPU, no keys, no cost), is it under 60s cold, and does it gate correctness or measure quality. **Quality measures never gate.**

**The full table — what runs at each layer, where, when, and whether it blocks — is in [`docs/test-layers.md`](docs/test-layers.md).** Read it before adding a test or changing CI routing.

Two rules to obey while writing code:

- **`_e2e.rs` is hermetic** — full internal pipeline, in-process, the LLM stubbed. Fast, deterministic, CI-safe. **`scripts/smoke-*.sh`** is an HTTP black-box check against a running daemon, and its depth varies.
- **Never write bare "smoke test" for a GPU-gated script.** If it touches the real on-device model it is a *live* smoke test and the filename folds the qualifier in (`live-smoke-doc-reconcile.sh`). The word alone does not signal depth — pair it with "live" in code comments and docs alike, so the non-hermetic tier is legible at a glance.

A required CI check failed intermittently? Follow [`docs/ci-flake-policy.md`](docs/ci-flake-policy.md) before rerunning, quarantining, rerouting, or reverting.

### Eval cache, baselines & faithfulness benches → `app/eval/AGENTS.md`

The eval-specific machinery — the baseline/scenario-DB cache (`EVAL_BASELINES_DIR`, TTL/purge policy, `EVAL_ENRICHMENT_CACHE_DIR` chaining, `migrate-eval-cache.sh`), cached scenario DBs (`~/.cache/origin-eval/scenario_seeded/{locomo_v1,lme_v1}/`, `seed-scenario-dbs.sh`, `cached_scenario_db_check.rs`), the `EVAL_LOCOMO_LIMIT`/`EVAL_LME_LIMIT` pre-flight subset, the full eval env-var table, the KG- and page-distillation faithfulness benches, fixture management, baseline layout, pre-flight checklist, and citation discipline — lives in **`app/eval/AGENTS.md`** and **`crates/wenlan-core/src/eval/AGENTS.md`**. Those subdir `AGENTS.md` files apply per the agents.md hierarchical-instruction convention when an agent is working under those subtrees.

## Releasing (release-please)

Releases are automated via [release-please](https://github.com/googleapis/release-please): every push to `main` maintains an open "release PR" that bumps the version and updates `CHANGELOG.md`; merging that PR publishes a GitHub release + `v*` tag, which triggers `.github/workflows/release.yml` to build and publish `wenlan`, `wenlan-server`, and `wenlan-mcp`. **The operator runbook — manual `bump-version.sh`, tag steps, the release-workflow breakdown, config files, and required secrets — lives in [`RELEASING.md`](RELEASING.md).** What an agent must keep in mind while coding:

**Every release bumps patch.** `release-please-config.json` sets `"versioning": "always-bump-patch"`, so the commit prefix decides only what lands in the changelog, never the bump size:

| Commit prefix | Version bump | Changelog |
|---|---|---|
| `fix:` | patch | Bug Fixes |
| `feat:` | patch | Features |
| `BREAKING CHANGE` | patch | Breaking |
| `chore:`, `ci:`, `docs:`, `refactor:`, `test:` | no bump on its own | hidden |

**Squash merge commit messages still matter — for the changelog.** When GitHub squash-merges a PR, the commit message defaults to the PR title. The prefix no longer changes the bump size (always patch), but it decides whether and where the change appears in `CHANGELOG.md`, and a title without a conventional `type:` prefix is invisible to release-please entirely. Keep PR titles valid conventional commits.

**Version files must stay in sync:** `version.txt`, `.release-please-manifest.json`, and the root workspace `Cargo.toml` (`# x-release-please-version` marker on the `[workspace.package]` version line; the 4 crates inherit it via `version.workspace = true`). Teeth #3 enforces this; the release-please workflow syncs them on the release branch, so any manual version edit must touch all three. The desktop app crate (`app/`) still carries its own version across `app/Cargo.toml`, `app/tauri.conf.json`, and `package.json`, lockstepped to `version.txt` and enforced by `scripts/release-version-sync.test.ts`.

### Branch protection

Main branch has: required CI (`conclusion` — aggregate gate over `fmt` + `lint` + `test`, rust-lang convention from cargo / rustup / rust-analyzer) before merge, no force pushes, no deletion. `enforce_admins: false` so the repo owner can push directly for hotfixes. Force push requires temporarily enabling it via API (remember to re-disable after).

### Git hooks

Manual setup: `bash scripts/setup-hooks.sh`. Hooks live under `.githooks/`.

- **Pre-commit:** checks formatting (`cargo fmt --all -- --check`) without modifying or staging files, then runs Clippy on directly changed crates.
- **Pre-push:** planner-selected Clippy + library tests for affected packages and reverse dependents. Direct integration-test edits and isolated unit-test owners run only that target/module. No coverage gate — [`docs/test-layers.md`](docs/test-layers.md) records why the 90% gate was tried and removed.

### Drift-defense (doc/flag/config drift)

Four fail-loud doc/flag/config-drift teeth live as `#[cfg(test)]` lib tests in `crates/wenlan-core/src/drift_guard.rs` (selected whenever the planner includes `wenlan-core`, and always by the full `main` backstop). The file also carries teeth #4 (root `AGENTS.md` byte budget) and #5 (FastEmbed CI cache), which guard non-drift concerns:

- **Teeth #1 — path resolver:** tracked markdown may not reference an in-repo path that doesn't exist on the branch. Skips `docs/plans/**`, `docs/superpowers/**`, and `*AUDIT.md` (historical/aspirational), and only checks file-like refs. Suppress an intentional ref with `<!-- drift-ok -->`.
- **Teeth #2 — flag doc contract (fail-closed):** every behavioral `WENLAN_*` flag read in `crates/*/src` must be documented in an `AGENTS.md`, else allowlisted (`FLAG_ALLOWLIST`, infra/test) or grandfathered (`BASELINE_UNDOCUMENTED`, the burn-down list of flags undocumented at introduction). A NEW undocumented flag fails the build.
- **Teeth #3 — version sync:** `version.txt`, `.release-please-manifest.json`, and the root workspace `Cargo.toml` must carry an identical version string.
- **Teeth #6 — section-heading resolver:** a cross-reference like ``See `crates/wenlan-core/AGENTS.md` "Some Heading".`` must resolve to a real heading in the target file (case-insensitively). Teeth #1 guards the *path*; this guards the *quoted heading*, so a doc-tiering move that relocates a section can't silently leave a dangling pointer. Same skips as teeth #1; suppress with `<!-- drift-ok -->`. <!-- drift-ok -->  (this bullet's own `"Some Heading"` example is illustrative, hence suppressed)

The fuzzy surfaces (eval numbers stale vs the current env-hash, design-doc/decision rot, memory→repo dangling pointers, stale worktrees) are covered by the read-only `doc-drift-auditor` subagent. Run weekly, locally:

- One-off: `bash scripts/drift-audit.sh`
- Recurring: `/loop 7d "bash scripts/drift-audit.sh"`, or a cron/launchd entry. Reports land in `docs/superpowers/drift-reports/` (gitignored working-doc space).

GitHub Actions caches are pruned daily by `ci-cache-maintenance.yml` to a 9 GB operating target (oldest unprotected entries first; the portable FastEmbed snapshot is protected). The workflow uploads a receipt and supports a manual dry run.

## Architecture

Wenlan is a **Personal Agent Memory Layer** — a local-first memory server on macOS where AI agents write what they learn and humans curate. Daemon-centric: a headless HTTP server owns all business logic and data; the desktop app, the CLI, and external MCP clients are all thin clients over its HTTP API.

### Database & events (owned by wenlan-core)

One libSQL database (`MemoryDB` in `crates/wenlan-core/src/db.rs`) holds document chunks + vectors, the knowledge graph, and FTS, combined via Reciprocal Rank Fusion. `wenlan-core` stays framework-agnostic by emitting UI events through an `EventEmitter` trait (`NoopEmitter` in the daemon, `TauriEmitter` in the desktop app) rather than depending on tauri. **Schema, connection/sharing patterns, and the trait definition live in `crates/wenlan-core/AGENTS.md`** (loaded when working under that crate, per the agents.md hierarchical convention).

## Key Modules

Per-crate module tables live in subtree `AGENTS.md` files (loaded when an agent works under that crate, per the agents.md hierarchical-instruction convention):

- `crates/wenlan-core/AGENTS.md` — all business logic (db, engine, classify, extract, rerank, refinery, pages, eval, ...).
- `crates/wenlan-server/AGENTS.md` — HTTP daemon (router, routes, state, ingest_batcher, scheduler, ...).

## Conventions

### Eval Citation Discipline

See `app/eval/AGENTS.md` "eval citation discipline" section for the full rules (single-run, schema-version, receipt-only, per-case visibility, layer attribution, commit policy). External-facing numbers MUST satisfy those rules.

### Crate boundaries
- **wenlan-core must have NO tauri or axum dependencies.** Verify with `grep -rn "use tauri\|use axum" crates/wenlan-core/src/` — expect zero hits. Any event emission goes through the `EventEmitter` trait.
- **wenlan-types must be lightweight.** Only serde + serde_json + anyhow. No chrono, no tokio, no heavy deps. These types are shared with `wenlan-mcp` (Apache-2.0) and `wenlan-app` (AGPL-3.0), both in this workspace — `wenlan-app` takes them as a path dep (`workspace = true`) — so adding heavy deps forces them downstream.
- **Don't add business logic to wenlan-server.** Route handlers should call `wenlan-core` functions with state snapshots — the server's job is HTTP framing, not logic.
- **Don't add new HTTP endpoints to the CLI.** Use existing daemon endpoints. If a CLI subcommand needs new data, add a daemon endpoint first.
- **MCP wrappers in `wenlan-mcp` always typed-deserialize.** Every `_impl` method in `crates/wenlan-mcp/src/tools.rs` deserializes the daemon response into a typed wire struct from `wenlan-types` (e.g. `SearchPagesResponse { pages: Vec<Page> }`), never into `serde_json::Value`. Untyped responses silently emit whatever shape the daemon returns; typed deserialization fails loud on envelope-key drift. Mirror commit `4f545869` and PR #77.

### Enrichment parity & eval-seed contract → `crates/wenlan-core/AGENTS.md`

All post-store enrichment goes through the ONE canonical path (`wenlan_core::ingest::run_canonical_enrichment`) so no consumer re-implements a divergent subset (the training-serving-skew fix), and the eval seed + eval read share ONE liveness contract (`seed_contract.rs`) so neither drifts onto a dead substrate. The full rationale, the `seed_scenario_dbs_complete` orchestrator, and the `SeedExpectations` teeth live in **`crates/wenlan-core/AGENTS.md`** (loaded when working under that crate).

### Async and locking
- **Never hold a `tokio::sync::RwLock` read or write guard across `.await`.** Holding a read guard during an LLM call (which can take seconds) blocks all writers. Pattern: snapshot what you need from the guard into a scoped block that ends before the await, then call the async function with the cloned values. See `crates/wenlan-server/src/memory_routes.rs` `handle_store_memory` for an example of the post-ingest enrichment pattern.
- **`Arc<MemoryDB>` is the sharing primitive.** `ServerState.db` is `Option<Arc<MemoryDB>>`. Clone the Arc out of the guard rather than borrowing through the guard.
- **Daemon is the single writer.** Only `wenlan-server` opens the libSQL database. The desktop app and CLI never touch the DB directly — they talk HTTP.
- **libSQL connection pattern**: `MemoryDB` holds `tokio::sync::Mutex<libsql::Connection>` internally. Never try to share a `libsql::Connection` across tasks directly (`Send` but not `Sync`).

### SQL, strings, data
- **LIKE patterns against JSON**: Quote the match target to avoid substring false positives — `%"{id}"%` not `%{id}%` (e.g., `mem_1` would otherwise match `mem_10`). See the fix in `crates/wenlan-core/src/db.rs` (the `%"{id}"%` quoting shown above) and the regression test.

### Dev environment gotchas

**Daemon lifecycle** — which binary owns :7878, stale binaries after `git pull`, `kill -9` verification, per-worktree `target/` dirs, restart-after-upgrade, reranker startup states: the full checklist lives in the launch-primitive skill `.claude/skills/run-wenlan/SKILL.md` (tracked in-repo; any agent can read it).

**Other:**
- **Metal/ggml on macOS Tahoe 26.x**: `ggml_metal_init` may fail even though native Metal works. The daemon auto-degrades and continues without LLM. Not a code bug. Check for competing GPU processes: `pgrep -la wenlan`.
- **Dev and prod share data by default**: Both use port 7878 and the platform data directory (on macOS, `~/Library/Application Support/wenlan/`). For isolated testing, override explicitly: `WENLAN_PORT=7879 WENLAN_DATA_DIR=/tmp/origin-test cargo run -p wenlan-server`.
- **Isolation has THREE axes, not two — `WENLAN_DATA_DIR` does not cover the page vault.** The projection directory comes from the `knowledge_path` config field, which `Config::knowledge_path_or_default()` (`crates/wenlan-core/src/config.rs:147`) resolves to `~/.wenlan/pages` when unset — it reads nothing from `WENLAN_DATA_DIR`. A fresh isolated data dir therefore has no override, so an "isolated" daemon that creates or exports pages writes real `.md` files into the user's live vault. Before starting an isolated daemon that will touch pages, seed its data dir with a `config.json` carrying `{"knowledge_path": "<scratch>/pages"}`, then confirm via `GET /api/knowledge/path` that the daemon reports the scratch path. Fingerprint the real vault before and after if the run is at all destructive.

### Worktree cleanup after squash-merge

Squash-merge lands a fresh SHA on `main`, so the branch's original commits still look unmerged locally. Two consequences:

- **`git cherry main feature/<name>` lies** — it compares SHAs, not patch content, and marks every squashed commit `+`. Confirm from the squash commit body instead: `git log -1 --format=%B <squash-sha>` lists each original PR commit message.
- **Removing a worktree destroys its gitignored files.** `app/eval/baselines/`, `.fastembed_cache/`, and build outputs are per-checkout. If a worktree is the only host of an eval baseline DB or downloaded model, move it to `~/.cache/origin-eval/` (via `scripts/migrate-eval-cache.sh`) first, then `git worktree remove --force .worktrees/<name>`, `git branch -D <branch>` (force needed for the same SHA reason), `git worktree prune`.

### Misc
- `WENLAN_BIND_ADDR=<host:port>`: override the daemon's bind address (default `127.0.0.1:7878`). Used inside Docker to listen on `0.0.0.0`.
- `WENLAN_AGENT_NAME=<name>`: default agent identity for CLI requests when `--agent-name` is not provided.
- `WENLAN_DEFAULT_SPACE=<name>`: overridable default save Space for CLI and MCP writes when no explicit Space is supplied. `WENLAN_SPACE` remains the strict lock and wins over explicit and default values.
- Log filter default is `warn` — add modules explicitly for `info` logs (e.g., `wenlan_core::db=info`, `wenlan_server=info`)
- All local data stored in the platform data directory (`dirs::data_local_dir()/origin/`; on macOS, `~/Library/Application Support/wenlan/`) — MemoryDB, config, activities, tags
- Crate names: `wenlan-types`, `wenlan-core`, `wenlan-server`, `wenlan` (CLI), `wenlan-mcp`, and `wenlan-app` (the `app/` desktop crate) — all in this workspace.
- **Licenses**: the five runtime crates (`wenlan-types`, `wenlan-core`, `wenlan-server`, `wenlan` CLI, `wenlan-mcp`) are **Apache-2.0** via workspace inheritance. The `wenlan-app` desktop crate (`app/`) is **AGPL-3.0-only** via its own `license` field, overriding the workspace default.
- `wenlan-mcp` is in-tree at `crates/wenlan-mcp/` (merged from the old `7xuanlu/wenlan-mcp` repo on 2026-05-09 via `git subtree`). It talks to the daemon via HTTP at runtime and is published to npm as a standalone binary (`npx -y wenlan-mcp`).

### Retrieval helpers location (PR-A, 2026-05-27)

`crates/wenlan-core/src/retrieval/` is the canonical home for retrieval helpers (`hard_filters`, `signals`). The old `composite/` namespace was deleted along with the dead `CompositeWeights` scaffolding when PR #200 closed. Future retrieval-channel additions (page-channel in PR-B, etc.) live in `retrieval/`.

### Retrieval / LLM / consolidation tuning

The deep per-flag reference — retrieval-channel flags (`WENLAN_RERANKER_MODEL`, `WENLAN_RERANKER_MODE`, `WENLAN_GRAPH_MEMORY_STREAM`, `WENLAN_ENABLE_TEMPORAL_SOFT_BOOST`, `WENLAN_TEMPORAL_BONUS`, `WENLAN_ENABLE_INTENT_LLM`, `WENLAN_RERANK_SKIP_PREFERENCE`, `WENLAN_ENABLE_ENTITY_SWEEP`), the on-device LLM throughput flags (`WENLAN_LLM_SLOT_BACKFILL`, `WENLAN_LLM_PREFIX_KV_CACHE`), and the always-on consolidation demotion (P3) — lives in **`crates/wenlan-core/AGENTS.md`**, which loads automatically when an agent works under that crate (agents.md hierarchical convention). `drift_guard` teeth #2 scans every tracked `*AGENTS.md`, so the flag-doc contract still holds.
