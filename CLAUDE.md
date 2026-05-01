# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Dev Commands

Origin is a Cargo workspace with 4 crates: `origin-types`, `origin-core`, `origin-server`, and the Tauri app (`origin`).

```bash
# Install frontend dependencies
pnpm install

# Run in development (recommended — single command handles everything):
pnpm dev:all
# This runs: (1) cargo build -p origin-server, (2) starts daemon on :7878,
# (3) pnpm tauri dev (Vite + cargo watch + app).
# Daemon runs as a background process, NOT as a Tauri sidecar.
# Sidecar mechanism only works in production builds (known Tauri limitation:
# github.com/tauri-apps/tauri/issues/1298, #3612, #4780, #13767).

# Or manually in two terminals:
cargo run -p origin-server            # terminal 1 — daemon
pnpm tauri dev                        # terminal 2 — app

# Or start the daemon as a managed launchd service:
cargo build -p origin-server
./target/debug/origin-server install   # writes plist, launchctl load
./target/debug/origin-server status
./target/debug/origin-server uninstall # when done

# Workspace-level builds
cargo check --workspace
cargo build --workspace
cargo test --workspace

# Per-crate builds (faster for iteration)
cargo check -p origin-types
cargo check -p origin-core
cargo check -p origin-server
cargo check -p origin                  # the Tauri app

# Run tests for a single crate
cargo test -p origin-core
cargo test -p origin-core --lib <module>::tests
cargo test -p origin-core <test_name>

# Frontend tests
pnpm test
pnpm test:watch
pnpm test:coverage
pnpm test:all                          # frontend + app crate tests

# Generate coverage reports (opens in browser)
bash scripts/coverage.sh

# Set up git hooks (one-time)
bash scripts/setup-hooks.sh

# Eval benchmarks (require GPU + model files, run manually)
# Unit tests for eval modules (fast, no GPU):
cargo test -p origin-core --lib locomo::tests
cargo test -p origin-core --lib longmemeval::tests

# Generate eval baselines (slow, needs Qwen 3.5-9B on Metal GPU):
cargo test -p origin --test eval_harness save_locomo_baseline -- --ignored --nocapture
cargo test -p origin --test eval_harness save_locomo_reranked_baseline -- --ignored --nocapture
cargo test -p origin --test eval_harness save_locomo_expanded_baseline -- --ignored --nocapture
cargo test -p origin --test eval_harness save_longmemeval_baseline -- --ignored --nocapture
cargo test -p origin --test eval_harness save_longmemeval_reranked_baseline -- --ignored --nocapture
cargo test -p origin --test eval_harness save_longmemeval_expanded_baseline -- --ignored --nocapture
# Baselines saved to app/eval/baselines/*.json (gitignored)
```

Frontend tests use Vitest + React Testing Library. Git hooks auto-activate on `pnpm install` -- pre-commit auto-formats and checks compilation, pre-push runs clippy + workspace tests (no coverage gate, see below).

## Local vs CI test responsibilities

Origin runs across several layers. The split is driven by three questions: **(1) Can a hosted runner do this?** (no GPU, no API keys, no cost). **(2) Is it under 60s on cold cache?** **(3) Does it gate correctness or measure quality?** Quality measures never gate.

| Layer | What runs | Where | When | Time | Blocks? |
|---|---|---|---|---|---|
| **L1 dev loop** | rust-analyzer / IDE | Local | Every save | <1s | No |
| **L2 pre-commit** | `cargo fmt --all`, clippy on staged crates, vitest if FE staged | Local | `git commit` | ~5s | Yes |
| **L3 pre-push** | `cargo clippy --workspace --all-targets`, `cargo test --workspace`, `pnpm vitest run --bail 1` | Local | `git push` | ~60-90s | Yes |
| **L4 CI on PR** | Same checks workspace-wide, plus `cargo test -p origin --lib`, `pnpm test` | GitHub (`ci.yml`) | Every PR | ~10min | Yes (required) |
| **L5 coverage on PR** | `cargo llvm-cov` on origin-core + origin-server only; vitest --coverage | GitHub (`coverage.yml`) | Every PR | ~10min | **No (informational)** |
| **L6 main canary** | Embedding-only eval (`cargo test -p origin-core --lib eval::token_efficiency -- --ignored`) | GitHub (`ci.yml`) | Push to `main` | ~10min | No (post-merge) |
| **L7 manual local** | `bash scripts/coverage.sh` (HTML coverage), GPU eval suite (`cargo test -- --ignored`), Anthropic batch judge (`ANTHROPIC_API_KEY=... cargo test ...`) | Your laptop | On demand | minutes-hours | No |
| **L8 pre-release** | Full eval suite vs saved baseline. Record deltas in vault/memory **never git** (AGPL public-repo rule) | Your laptop | Per release | hours | Soft gate |

### What does NOT run in CI and why

- **GPU evals (LongMemEval / LoCoMo runner functions, Qwen3.5-9B inference)** — GitHub macOS runners have no Metal acceleration. The tests are `#[ignore]`d so they don't accidentally run.
- **Anthropic API batch judge** — costs $0.35/run and requires `ANTHROPIC_API_KEY` which we don't expose to PR runs from forks.
- **Tauri app coverage** — `--package origin` (the Tauri app crate) is mostly command proxies that can't be exercised without a GUI runtime, and instrumented compilation peaks at 8-16GB RSS. Coverage is scoped to `origin-core + origin-server`.

### Why pre-push doesn't run coverage

Earlier versions of `.githooks/pre-push` enforced a 90% `cargo llvm-cov` gate. That violated the principles above:
- **Slow:** instrumented rebuild of the Tauri-app-pulling workspace took 5-15min and overloaded memory.
- **Not mirrored in CI:** the main `ci.yml` lane doesn't run coverage at all, so the gate added local friction without upstream protection.
- **Percentage gates rot:** any new untestable surface (Tauri commands, GPU-only eval) drops the percentage and forces busywork.

The current pre-push runs only clippy + non-instrumented tests. Coverage is L5 (informational on PR) or L7 (manual command on laptop).

### Eval baselines cache

The per-scenario DB cache (Phase 1 enrichment + Phase 3 answer cache + judge JSONLs) lives at `<baselines_dir>/`, where `baselines_dir` defaults to `app/eval/baselines/` per worktree. Override with `EVAL_BASELINES_DIR=<path>` to point at a shared, worktree-agnostic location:

```bash
export EVAL_BASELINES_DIR=$HOME/.cache/origin-eval
```

Path must be writable and local (network mounts not recommended). When set, also chains `EVAL_ENRICHMENT_CACHE_DIR` default to the same dir unless explicitly overridden. Migration: `bash scripts/migrate-eval-cache.sh <source-baselines>`.

## Releasing (release-please)

Releases are automated via [release-please](https://github.com/googleapis/release-please). The workflow runs on every push to `main`.

**How it works:**
1. Every push to `main`, release-please scans new commits and maintains an open "release PR" that accumulates changes and updates `CHANGELOG.md`.
2. When you're ready to ship, merge the release PR. That triggers a GitHub release (draft) + git tag.
3. The `v*` tag push triggers `.github/workflows/release.yml`, which builds the Tauri app, bundles sidecars, uploads DMG + standalone binaries to the release, and publishes it (draft -> public).
4. The release-please workflow also syncs `Cargo.toml` + `package.json` + `tauri.conf.json` versions on the release branch (release-please can't handle Cargo workspaces or JSON extra-files reliably with `simple` release type).

**Commit messages control version bumps.** Pre-1.0:

| Commit prefix | Version bump | Example |
|---|---|---|
| `fix:` | patch | 0.1.1 -> 0.1.2 |
| `feat:` | **minor** | 0.1.1 -> 0.2.0 |
| `BREAKING CHANGE` | minor (capped) | 0.1.1 -> 0.2.0 |
| `chore:`, `ci:`, `docs:`, `refactor:`, `test:` | no bump | (hidden in changelog) |

After 1.0, standard semver: `feat:` bumps minor, `BREAKING CHANGE` bumps major.

**IMPORTANT: `feat:` bumps minor, not patch.** Use `fix:` for small features, improvements, and bug fixes. Only use `feat:` when you intentionally want 0.x.0 -> 0.(x+1).0. The `bump-patch-for-minor-pre-major` and `release-as` config flags do NOT work with release-please v17 + simple release type. If a `feat:` commit lands on main, the only way to prevent the minor bump is to rewrite the commit message via `git filter-branch` and force-push.

**Squash merge commit messages matter.** When GitHub squash-merges a PR, the commit message defaults to the PR title. A PR titled `feat: ...` creates a `feat:` commit on main, triggering a minor version bump. Review PR titles before merging -- rename to `fix:` if a minor bump is not intended.

**Config files:**
- `release-please-config.json` -- release type, version bump behavior, extra files
- `.release-please-manifest.json` -- current version (0.1.1)
- `.github/workflows/release-please.yml` -- creates/updates release PRs, syncs version files
- `.github/workflows/release.yml` -- builds app + uploads artifacts on `v*` tag push

```bash
# To release: just merge the open release-please PR on GitHub.
# To check what version is pending:
cat .release-please-manifest.json
```

### Release pipeline gotchas (learned the hard way)

**Version files that must stay in sync:** `version.txt`, `.release-please-manifest.json`, `package.json`, `app/tauri.conf.json`, and all four `Cargo.toml` files (`# x-release-please-version` marker). The release-please workflow syncs these automatically on the release branch, but manual version changes must update all of them.

**release-please determines "last version" from merged PR commit messages**, not tags or manifest. It scans for commits matching `chore(main): release X.Y.Z`. Deleting a tag or GitHub Release is NOT enough to reset the version. You must also ensure no commit message in the history matches that pattern, or use `release-as` to force-override.

**Never delete a release tag without also cleaning up the commit history.** If you need to undo a release version, you must rewrite the commit message that release-please created (`git filter-branch --msg-filter`), delete the tag, delete the GitHub Release, and rename the merged PR title via API. Otherwise release-please will keep bumping from the old version.

**Code signing on CI.** The release workflow uses `APPLE_SIGNING_IDENTITY` with `"-"` fallback (ad-hoc signing) when no Apple Developer secrets are configured. The `APPLE_ID`, `APPLE_PASSWORD`, and `APPLE_TEAM_ID` env vars must be completely absent (not empty strings) when not configured, or Tauri attempts notarization and fails with "Team ID must be at least 3 characters".

**The `release.yml` workflow builds everything.** It handles: origin-server (cargo build), origin-mcp (cargo install from separate repo), cloudflared (download from Cloudflare), Tauri app (tauri-action), standalone binary uploads, and crates.io publishing. Do not duplicate build logic in release-please.yml.

**Draft releases.** `release-please-config.json` has `"draft": true`, so release-please creates draft releases. The `release.yml` tauri-action step sets `releaseDraft: false` to publish the release after artifacts are uploaded. This prevents users from seeing empty releases during the build window.

### Branch protection

Main branch has: required CI ("Test & Lint") before merge, no force pushes, no deletion. `enforce_admins: false` so the repo owner can push directly for hotfixes. Force push requires temporarily enabling it via API (remember to re-disable after).

### Git hooks (auto-activated)

Hooks activate automatically on `pnpm install` (postinstall script sets `core.hooksPath`). No manual setup needed.

Pre-commit: **auto-formats** Rust (`cargo fmt --all`, re-stages changed files) + `cargo check` + vitest. Formatting issues can never reach CI.
Pre-push: `cargo clippy -- -D warnings` + full test suite + coverage gate.
Manual setup: `bash scripts/setup-hooks.sh` (only needed if you skip `pnpm install`).

## Architecture

Origin is a **Personal Agent Memory Layer** — a local-first memory server on macOS where AI agents write what they learn and humans curate. Daemon-centric: a headless HTTP server owns all business logic and data, and a thin Tauri desktop app (plus external MCP clients) talk to it via HTTP.

### Workspace Layout

The repo is a Cargo workspace with 4 crates:

| Crate | Role | Key dependencies |
|---|---|---|
| `crates/origin-types` | Shared API boundary types (request/response, memory, entities). License: Apache-2.0 so `origin-mcp` (MIT) and other downstream consumers can use it without AGPL contamination. | serde only — no heavy deps |
| `crates/origin-core` | All business logic: DB, embeddings, LLM engine, search, classification, knowledge graph, refinery, concepts, export, eval. **Must have NO tauri or axum dependencies.** | libSQL, FastEmbed, llama-cpp-2, hf-hub |
| `crates/origin-server` | Headless HTTP daemon on `127.0.0.1:7878`. Depends on `origin-core`. Provides `install/uninstall/status` subcommands for launchd management. | axum, tower, clap |
| `app/` (crate name `origin`) | Thin Tauri desktop client. Depends on `origin-types` + `reqwest` (HTTP) + a minimal bit of `origin-core` for sensor utilities. All data commands proxy to the daemon. | tauri, reqwest |

The daemon (`origin-server`) is the single source of truth. The Tauri app process can come and go; memory storage continues in the daemon. External tools (`origin-mcp`, curl, etc.) talk to the same daemon.

### Stack

- **Daemon**: Rust, Axum 0.8 (HTTP), libSQL (Turso's SQLite fork — vectors, knowledge graph, documents), Tokio, FastEmbed (BGE-Base-EN-v1.5-Q, 768-dim, 512-token max), llama-cpp-2 (Qwen3-4B-Instruct-2507 via Metal GPU; Qwen3.5-9B optional), launchd for process management
- **App (thin client)**: Rust, Tauri 2, reqwest, small sensor/trigger modules for macOS-specific capture
- **Frontend**: React 19, TanStack React Query 5, Tailwind CSS 4, Vite
- **Package manager**: pnpm

### Database: libSQL (owned by origin-core)

One libSQL database at `~/Library/Application Support/origin/memorydb/origin_memory.db`, owned by `MemoryDB` in `crates/origin-core/src/db.rs`:
- **Document chunks**: `chunks` table with `F32_BLOB(768)` vector column, DiskANN indexing (768-dim, BGE-Base-EN-v1.5-Q)
- **Knowledge graph**: `entities`, `relations`, `observations` tables with FK cascades
- **Full-text search**: FTS5 virtual table (`chunks_fts`) auto-synced via triggers
- **Hybrid search**: Vector similarity + FTS combined with Reciprocal Rank Fusion (RRF)

**Connection pattern**: `tokio::sync::Mutex<libsql::Connection>` — `libsql::Connection` is `Send` but not `Sync`, so it's wrapped in an async `Mutex` inside `MemoryDB`.

**Sharing pattern**: `MemoryDB` is wrapped in `Arc<MemoryDB>` at the state layer (`ServerState.db: Option<Arc<MemoryDB>>`). This lets handlers clone the `Arc` out of the `RwLock<ServerState>` guard and drop the guard before performing long-running operations.

### Events: EventEmitter trait (no tauri::Emitter in core)

Instead of passing `tauri::AppHandle` into business logic, `origin-core` defines an `EventEmitter` trait:

```rust
// crates/origin-core/src/events.rs
pub trait EventEmitter: Send + Sync {
    fn emit(&self, event: &str, payload: &str) -> Result<()>;
}
pub struct NoopEmitter;
```

- The daemon uses `NoopEmitter` (no UI to notify)
- The Tauri app would use a `TauriEmitter` adapter (`app/src/events.rs`) that wraps `AppHandle::emit`
- `MemoryDB::new(db_path, emitter: Arc<dyn EventEmitter>)` takes the trait object

This keeps `origin-core` framework-agnostic and testable with `NoopEmitter` in unit tests.

### Multi-Window Architecture

Six Tauri windows from a single webview, routed by URL hash in `src/main.tsx`:
- **main** — Spotlight search / Memory view / Settings / Chunk viewer
- **toast** — Transparent overlay for capture notifications (non-activating panel)
- **snip** — Full-screen transparent overlay for region selection
- **quick-capture** — Popup for quick thought capture
- **ambient** — Ambient context overlay window
- **icon** — Small icon trigger overlay near the cursor/selection

### Unified Trigger Architecture

All input sources emit a `TriggerEvent` into a single mpsc channel (capacity 32), consumed by the **smart router** (`router/intent.rs`):

```
Sensors (focus, selection, ambient, hotkey, snip, thought)
    → TriggerEvent channel (32)
        → Smart Router (dedup, AFK gating, frame comparison, OCR)
            → ContextBundle channel (8)
                → Context Consumer (MemoryDB upsert, LLM queue)
```

The router applies per-window frame comparison, text dedup (bigram Jaccard, 0.85 threshold), AFK detection (60s idle), and PII redaction before storage.

**Note**: Ambient capture (clipboard, window_activity) is disabled by default as part of the memory-layer pivot. Screen capture infrastructure remains but is opt-in.

### Two-Pass Capture Pipeline

1. **Immediate**: Raw OCR → PII redaction → per-window chunking → MemoryDB upsert → toast notification
2. **LLM (async)**: Qwen3-4B-Instruct-2507 reformats focused window text → structured summary + category → MemoryDB update → UI refresh

### Thread Model

- **Tokio runtime** (Tauri-managed): Router, consumer, ambient ticker, IPC server, DB operations
- **Dedicated std::thread**: Focus sensor (10Hz cursor polling), LLM formatter (GPU inference)
- **Bridge pattern**: LLM thread captures `tokio::runtime::Handle` for async DB calls via `handle.block_on()`
- **IMPORTANT**: In `setup()` closure, `tokio::runtime::Handle::current()` panics — use `tauri::async_runtime::block_on(async { Handle::current() })` instead

### State Management

- **Backend**: `Arc<RwLock<AppState>>` managed by Tauri. Atomic flags (`Arc<AtomicBool>`) for feature toggles shared with sensor threads (lock-free reads from hot loops).
- **Frontend**: TanStack React Query for server state. Module-level stores (`useSyncExternalStore` pattern) for state that must survive React unmounts (`processingStore.ts`, `captureHeartbeat.ts`). localStorage for collapsed section state with 7-day TTL.

### IPC Surface

All data flows through the daemon's HTTP API. The Tauri app's `#[tauri::command]` functions are thin proxies that call `OriginClient` (`app/src/api.rs`).

- **HTTP API**: Axum on `127.0.0.1:7878`, served by `origin-server`. Used by the Tauri app, the `origin-mcp` MCP server (separate repo), and any external tool.
  - General: `/api/health`, `/api/status`, `/api/search`, `/api/context`, `/api/chat-context`, `/api/ping`
  - Ingest: `/api/ingest/text`, `/api/ingest/webpage`, `/api/ingest/memory`
  - Memory CRUD: `/api/memory/store`, `/api/memory/search`, `/api/memory/confirm/{id}`, `/api/memory/list`, `/api/memory/delete/{id}`
  - Knowledge graph: `/api/memory/entities`, `/api/memory/relations`, `/api/memory/observations`
  - Profile & Agents: `/api/profile`, `/api/agents`, `/api/agents/{name}`
  - WebSocket: `/ws/updates`
- **Tauri commands**: commands in `app/src/search.rs`, registered in `app/src/lib.rs`, called from frontend via `invoke()` wrappers in `src/lib/tauri.ts`. Most are one-line HTTP proxies via `OriginClient`; a handful of macOS-specific commands (quick-capture positioning, tray, sensor toggles) stay native.
- **Tauri events**: `capture-event` emitted globally for toast/UI updates via `TauriEmitter` adapter.
- **A legacy native command surface still exists**: many Tauri commands remain app-specific macOS helpers or thin wrappers that have not been migrated into daemon-backed HTTP endpoints. Add server endpoints incrementally as needed rather than assuming full parity.

## Key Modules — origin-core (`crates/origin-core/src/`)

All business logic lives here. No tauri, no axum. Framework-agnostic.

| Module | Purpose |
|---|---|
| `db.rs` | `MemoryDB` — libSQL storage, vectors, chunks, hybrid search, embeddings, knowledge graph, migrations. Three search methods: `search_memory` (embedding+FTS+RRF), `search_memory_reranked` (+ LLM reranking after), `search_memory_expanded` (+ LLM query expansion before). Uses `EventEmitter` trait for UI notifications (no tauri). |
| `events.rs` | `EventEmitter` trait and `NoopEmitter` |
| `engine.rs` | `LlmEngine` — llama-cpp-2 wrapper, model download, inference loop, format helpers |
| `classify.rs` | Memory/profile classification via `LlmEngine` |
| `extract.rs` | Knowledge-graph extraction (entities, relations) via `LlmEngine` |
| `rerank.rs` | LLM reranker |
| `merge.rs` | Memory merging, pattern extraction, contradiction detection |
| `llm_provider.rs` | `LlmProvider` trait + `ApiProvider` (Anthropic API) + `OnDeviceProvider` shim |
| `llm_classifier.rs` | Higher-level classification orchestration |
| `refinery.rs` | Background refinement queue — dedup, auto-linking, consolidation |
| `post_ingest.rs` | Post-ingest enrichment (dedup check, entity linking, title enrich, recap, concept growth) |
| `concepts.rs` | Type definitions for the Concept struct. Actual clustering + distillation live in `db.rs` + `refinery.rs` |
| `spaces.rs` | Spaces / tag store |
| `narrative.rs` | Profile narrative assembly (editorial prose) |
| `briefing.rs` | Daily briefing assembly |
| `working_memory.rs` | Working memory builder |
| `access_tracker.rs` | Memory access counts + time decay |
| `contradiction.rs` | Contradiction detection |
| `context_packager.rs` | Context bundle → prompt packaging |
| `importer.rs` | File importer pipeline |
| `quality_gate.rs` | Pre-store quality gate |
| `tuning.rs` | Tuning config (refinery, distillation, weights) |
| `schema.rs` | Memory schema definitions (formerly `memory_schema.rs`) |
| `prompts/` | Prompt registry (defaults + override dir loader) |
| `chunker/` | Code-aware, Markdown-aware, fixed-size chunking |
| `sources/` | `RawDocument`, file watchers, Obsidian importer. `RawDocument` and related types re-exported from `origin-types`. |
| `privacy.rs` | PII redaction |
| `router/classify.rs`, `content_score.rs` | Smart router scoring helpers (non-tauri parts) |
| `config.rs` | Persistent config at `~/Library/Application Support/origin/config.json` |
| `export/` | Markdown/JSON/zip/PDF exporters |
| `eval/` | Benchmark harness: LoCoMo, LongMemEval, LoCoMo-Plus, LifeBench. Each benchmark has base (embedding-only), reranked (LLM rescores after search), and expanded (LLM query expansion before search) variants. Baselines in `app/eval/baselines/` (gitignored). |
| `state.rs` | `CoreState` — shared state struct used by origin-server |

## Key Modules — origin-server (`crates/origin-server/src/`)

HTTP daemon — owns the Axum router + all routes. All handlers operate on `Arc<RwLock<ServerState>>` where `ServerState.db: Option<Arc<MemoryDB>>`.

| Module | Purpose |
|---|---|
| `main.rs` | Binary entry — clap subcommands (`install`/`uninstall`/`status`/daemon), tracing init, port binding with existing-daemon fallback, `MemoryDB::new`, LLM provider init, background tasks, `axum::serve` |
| `state.rs` | `ServerState` struct with `db: Option<Arc<MemoryDB>>`, `llm`, `prompts`, `tuning`, `quality_gate`, `space_store`, `access_tracker`, `llm_processing_ids`, `watch_paths`. `SharedState = Arc<RwLock<ServerState>>` |
| `router.rs` | `build_router(state) -> axum::Router` — all route registrations |
| `routes.rs` | General endpoints: health, search, context, chat-context, status, profile/agents |
| `memory_routes.rs` | Memory CRUD, knowledge graph, classification, entities, concepts |
| `ingest_routes.rs` | `/api/ingest/*` — text, webpage, memory |
| `ingest_batcher.rs` | Request-level coalescer for concurrent `/api/memory/store` — folds QualityGate in-line; async classify/extract; passes enrichment + hint through in the response |
| `knowledge_routes.rs` | Entity/relation/observation read paths + knowledge-graph queries |
| `source_routes.rs` | Source registry endpoints |
| `import_routes.rs` | Bulk import endpoints |
| `config_routes.rs` | Config read/write endpoints |
| `onboarding_routes.rs` | First-run wizard / milestone state |
| `scheduler.rs` | Background periodic tasks (refinery ticks, distillation, etc.) |
| `websocket.rs` | `/ws/updates` |
| `error.rs` | `ServerError` + axum `IntoResponse` impl |
| `resources/com.origin.server.plist` | launchd plist template (embedded via `include_str!`) |

## Key Modules — Tauri app (`app/src/`)

Thin client — only Tauri-specific code. Data commands proxy to the daemon.

| Module | Purpose |
|---|---|
| `lib.rs` | Tauri setup, plugin registration, window/tray/shortcut setup, daemon health check on startup, ~146 command registrations |
| `main.rs` | Tauri entry point |
| `state.rs` | Simplified `AppState` — `OriginClient`, `app_handle`, UI flags (clipboard_enabled, screen_capture_enabled, ambient_mode, etc.). No MemoryDB, no LLM, no embeddings. |
| `api.rs` | `OriginClient` — `reqwest`-based HTTP client wrapper, one method per endpoint, uses `origin-types` |
| `events.rs` | `TauriEmitter` — adapter that implements `origin_core::events::EventEmitter` by wrapping `tauri::AppHandle::emit` |
| `search.rs` | ~146 `#[tauri::command]` functions, most are HTTP proxies to `OriginClient` |
| `sensor/` | macOS-specific capture: vision, frame_compare, idle |
| `trigger/` | Focus sensor thread, ambient ticker |
| `router/intent.rs` | Smart router (trigger events → context bundles) — still in the app because it's entangled with sensors |
| `indexer.rs` | File watcher (notify-based, 2s debounce) |
| `remote_access.rs` | Cloudflare tunnel management for exposing the daemon externally |
| `ambient/` | Ambient overlay (disabled by default) |
| `mcp_config.rs` | MCP server configuration for external tools |

## Key Frontend Modules (src/)

| Module | Purpose |
|--------|---------|
| `main.tsx` | Hash-based multi-window entry point |
| `App.tsx` | Page navigation (spotlight, memory, settings, chunks), capture event listener |
| `lib/tauri.ts` | All `invoke()` wrappers + TypeScript interfaces for Tauri commands |
| `components/MemoryView.tsx` | Primary dashboard — time-grouped files, search, categories, activity timeline |
| `components/Spotlight.tsx` | Main search UI (Cmd+K) with source filters and working memory |
| `components/SourceManager.tsx` | Settings page — watch paths, feature toggles, categories |

## Conventions

### Crate boundaries
- **origin-core must have NO tauri or axum dependencies.** Verify with `grep -rn "use tauri\|use axum" crates/origin-core/src/` — expect zero hits. Any event emission goes through the `EventEmitter` trait.
- **origin-types must be lightweight.** Only serde + serde_json. No chrono, no tokio, no heavy deps. These types are shared with `origin-mcp` (MIT-licensed separate repo), so adding heavy deps forces them downstream.
- **Don't add business logic to origin-server.** Route handlers should call `origin-core` functions with state snapshots — the server's job is HTTP framing, not logic.
- **Don't add data logic to the Tauri app.** The app is a thin client. Every data command should either proxy via `OriginClient` or be a macOS-specific UI/sensor helper.

### Async and locking
- **Never hold a `tokio::sync::RwLock` read or write guard across `.await`.** Holding a read guard during an LLM call (which can take seconds) blocks all writers. Pattern: snapshot what you need from the guard into a scoped block that ends before the await, then call the async function with the cloned values. See `crates/origin-server/src/memory_routes.rs` `handle_store_memory` for an example of the post-ingest enrichment pattern.
- **`Arc<MemoryDB>` is the sharing primitive.** `ServerState.db` is `Option<Arc<MemoryDB>>`. Clone the Arc out of the guard rather than borrowing through the guard.
- **Daemon is the single writer.** Only `origin-server` opens the libSQL database. The Tauri app never touches the DB directly — it talks HTTP.
- **libSQL connection pattern**: `MemoryDB` holds `tokio::sync::Mutex<libsql::Connection>` internally. Never try to share a `libsql::Connection` across tasks directly (`Send` but not `Sync`).

### SQL, strings, data
- **SQL safety**: Always use parameterized queries — never interpolate user input into SQL strings
- **NULL semantics**: Store `Option<T>` as SQL NULL, not empty string — so IS NULL filters work correctly
- **UTF-8 safety**: Never byte-index Rust strings (`&s[..n]`) — use `chars().take(n)` or `strip_prefix`/`strip_suffix`. Exception: byte-slicing after a verified ASCII prefix check is safe (the boundary is guaranteed valid), but prefer the char-safe version anyway for consistency.
- **Batch SQL**: Wrap multi-row insert/delete loops in BEGIN/COMMIT transactions
- **LIKE patterns against JSON**: Quote the match target to avoid substring false positives — `%"{id}"%` not `%{id}%` (e.g., `mem_1` would otherwise match `mem_10`). See the fix at `crates/origin-core/src/db.rs:~9895` and the regression test.

### Dev environment gotchas

**Daemon lifecycle:**
- **Worktree daemon mismatch**: The daemon on port 7878 can be from launchd, main branch, a stale worktree, or a previous session. Always verify which binary is running: `lsof -i :7878` to get the PID, then `lsof -p <PID> | grep "txt.*origin-server"` to see the binary path and size. Kill and restart from the current working tree.
- **Stale binary after merge/pull**: `cargo build -p origin-server` may report "0.64s Finished" without recompiling if the source timestamps haven't changed (e.g., after `git pull` fast-forward). Touch a source file to force recompilation: `touch crates/origin-server/src/router.rs && cargo build -p origin-server`. Verify the binary timestamp matches: `ls -la target/debug/origin-server`.
- **kill vs kill -9**: `kill <PID>` may not terminate the daemon cleanly. Always use `kill -9 <PID>` and verify with `lsof -ti :7878` afterward. If the port is still in use, another process took over.
- **Worktree target directories are shared**: All worktrees share the same `target/` directory in the main repo. Building in one worktree overwrites binaries from another. After switching worktrees, always rebuild to ensure the binary matches the checked-out source.

**Tauri app:**
- **Never pipe Tauri binary output**: `./target/debug/origin 2>&1 | head -N` kills the process via SIGPIPE when head closes. Always redirect to file: `./target/debug/origin > /tmp/origin.log 2>&1 &`. This is the #1 reason the app appears to "exit silently".
- **Use `pnpm tauri dev` for development**: It handles Vite startup, cargo watch, and the full dev lifecycle. Running `./target/debug/origin` directly skips sidecar management and process lifecycle.
- **tauri.conf.json is compile-time**: Changes to `trafficLightPosition`, `visible`, `skipTaskbar`, window dimensions are baked into the binary. Requires `cargo build -p origin` and app restart (not HMR).
- **Vite required for dev binary**: `./target/debug/origin` loads the frontend from `devUrl` (localhost:1420). Without Vite running, the window shows a permanent white screen.
- **Window visibility**: The main window config has `visible: false`. The app-ready event from React calls `show()` + `center()` after mount. This prevents the white flash and position jump on startup.

**Cleanup scripts:**
```bash
pnpm clean:dev      # kills all origin/vite processes, frees ports 7878/1420
pnpm clean:release    # removes stale DMG temp files and old bundles
pnpm clean:all      # both
```

**Clean dev startup (canonical sequence):**
```bash
pnpm dev:all        # runs clean:dev first, then builds daemon + starts app
```
If post-merge and `pnpm dev:all` shows stale data, force a daemon rebuild first:
```bash
touch crates/origin-server/src/router.rs && pnpm dev:all
```

**Sidecar (dev vs production):**
- **Sidecar doesn't work in dev mode**: Tauri's `externalBin` sidecar resolution fails during `cargo run` / `tauri dev`. This is a known Tauri limitation (issues #1298, #3612, #4780, #13767). The `[init] Failed to spawn origin-server sidecar: No such file or directory` error is expected in dev. Use `pnpm dev:all` which starts the daemon directly instead.
- **Sidecar works in production builds**: `tauri build` bundles the sidecar binary correctly. The app's `lib.rs` sidecar spawn code is production-only in practice.

**Release builds:**
- **C++17 required for llama.cpp on macOS 26.x**: Release builds fail with `no template named 'optional' in namespace 'std'` because macOS Tahoe's SDK defaults to C++14 for the app crate (min version 26.4). Fix: `CXXFLAGS="-std=c++17" pnpm tauri build`. Debug builds are unaffected (different min version). This is a known upstream change: llama.cpp moved from C++11 to C++17 for `std::string_view` and `std::optional` in `common.h`.
- **DMG packaging**: Tauri's DMG bundler works if stale temp files (`rw.*.dmg`) in `target/release/bundle/macos/` are cleaned first. `release` runs `clean:release` automatically. If Tauri's bundler fails, `pnpm release:dmg` calls `hdiutil` directly as fallback (no external deps).
- **Dev vs production sidecar isolation**: `dev:daemon` copies the debug binary to `app/binaries/`. `release` copies the release binary. Always run `pnpm release` for production. Never ship a DMG after running `dev:daemon` without rebuilding release first.
- **Release build (single command)**:
  ```bash
  pnpm release          # builds release daemon, copies sidecar, builds .app
  pnpm release:dmg              # creates DMG from .app (requires create-dmg)
  # Output: target/release/bundle/macos/Origin.app
  #         target/release/bundle/dmg/Origin_0.1.0_aarch64.dmg
  ```
- **Runtime data separation**: The daemon stores data at `~/Library/Application Support/origin/`. This is shared between dev and production by default (one memory database). The Tauri app's own data goes to `~/Library/Application Support/com.origin.desktop/` (via bundle identifier). Build artifacts live in `target/` (debug vs release).
- **Dev and prod share data by default**: Both use port 7878 and `~/Library/Application Support/origin/`. This follows Tauri's official pattern (same identifier = same data dir in dev and prod). Your memories are your memories regardless of which build you're running. For isolated testing (onboarding flow, clean slate), override explicitly:
  ```bash
  ORIGIN_PORT=7879 ORIGIN_DATA_DIR=/tmp/origin-test pnpm dev:all
  ```
- **Code signing (not yet set up)**: The app is currently unsigned. Users must `xattr -cr /Applications/Origin.app`. For proper distribution: Apple Developer account ($99/yr), `signingIdentity` in tauri.conf.json, `Entitlements.plist` for WebView JIT, and notarization via `xcrun notarytool`. Deferred to post-v0.1.0.
- **Bundle identifier**: `com.origin.desktop` (not `.app`, which conflicts with macOS bundle extension).

**Other:**
- **Metal/ggml on macOS Tahoe 26.x**: `ggml_metal_init` may fail even though native Metal works. The daemon auto-degrades and continues without LLM. Not a code bug. Check for competing GPU processes: `pgrep -la origin`.

### Misc
- Log filter default is `warn` — add modules explicitly for `info` logs (e.g., `origin_core::db=info`, `origin_server=info`)
- macOS-specific code uses CoreGraphics/AppKit FFI via `cocoa` and `objc` crates, only in the Tauri app (`app/src/sensor`, `app/src/trigger`)
- All local data stored in `~/Library/Application Support/origin/` — MemoryDB, config, activities, tags
- Crate names: `origin-types`, `origin-core`, `origin-server`, `origin` (the Tauri app). The legacy name `origin_lib` still appears as the library name of the `origin` crate in some log filters.
- **Licenses**: `origin-types`, `origin-core`, and `origin-server` are **Apache-2.0** (permissive — lets downstream tools like `origin-mcp` consume them without AGPL contamination). The Tauri app (`origin` crate in `app/`) is **AGPL-3.0-only** (the shipped desktop product).
- `origin-mcp` lives in a separate repo (`~/Repos/origin-mcp`), is MIT-licensed, and talks to the daemon via HTTP. It can depend on `origin-types` (Apache-2.0) without license issues.
