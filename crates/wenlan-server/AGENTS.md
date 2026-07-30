# crates/wenlan-server

Applies to agents working under `crates/wenlan-server/`. Read alongside root `AGENTS.md`, which takes precedence on any topic not covered here.

HTTP daemon — owns the Axum router + all routes. All handlers operate on `Arc<RwLock<ServerState>>` where `ServerState.db: Option<Arc<MemoryDB>>`.

## Key Modules (`crates/wenlan-server/src/`)

| Module | Purpose |
|---|---|
| `main.rs` | Binary entry — daemon startup plus internal maintenance commands, tracing init, port binding with existing-daemon fallback, `MemoryDB::new`, LLM provider init, background tasks, `axum::serve` |
| `state.rs` | `ServerState` struct with `db: Option<Arc<MemoryDB>>`, `llm`, `prompts`, `tuning`, `quality_gate`, `space_store`, `access_tracker`, `llm_processing_ids`, `watch_paths`. `SharedState = Arc<RwLock<ServerState>>` |
| `router.rs` | Axum composition root — assembles module-owned registration helpers plus the remaining inline registrations, then applies the truth/security/lifecycle layers |
| `routes.rs` | General endpoints and their `TrackedRouter` registration helper: health, status, search/context, diagnostics, recent activity, steep/distill |
| `memory_routes.rs` | Remaining memory CRUD/search/enrichment, classification, statistics, scheduler-handoff, rerank, attribution, update, revision, and contradiction tests |
| `page_routes.rs` | Page list/get/search, source/link/revision reads, create/update/refresh/archive/delete, projection export, and their two position-preserving `TrackedRouter` registration helpers |
| `activity_tag_routes.rs` | Activity feed and tag list/suggestion/mutation handlers plus their `TrackedRouter` registration helper |
| `briefing_routes.rs` | Scoped daily briefing route plus its `TrackedRouter` registration helper |
| `memory_detail_routes.rs` | Global capture count and scoped single/batch memory-detail readers plus their `TrackedRouter` registration helper |
| `memory_revision_routes.rs` | Scoped memory revision history and pending-revision readers; two registration helpers preserve their separated composition positions |
| `decisions_routes.rs` | Scoped decision listing and global decision-space compatibility response plus their `TrackedRouter` registration helper |
| `entity_graph_routes.rs` | `/api/memory` entity, relation, observation, linking, suggestion, and scoped entity-read handlers; four registration helpers preserve their separated composition positions |
| `indexed_files_routes.rs` | Indexed-file and chunk read/update/delete handlers plus their `TrackedRouter` registration helper |
| `profile_agents_routes.rs` | Profile and agent CRUD handlers plus their `TrackedRouter` registration helper |
| `profile_narrative_routes.rs` | Cache-first profile narrative read and forced regeneration plus their `TrackedRouter` registration helper |
| `pinned_memory_routes.rs` | Scoped pinned-memory listing and pin/unpin mutations plus their `TrackedRouter` registration helper |
| `spaces_routes.rs` | Space CRUD/default/order/state handlers and document reassignment; core and extended registration helpers preserve their separated composition positions |
| `snapshot_routes.rs` | Global snapshot listing/deletion and parent-filtered snapshot capture readers plus their `TrackedRouter` registration helper |
| `ingest_routes.rs` | `/api/ingest/*` — text, webpage, memory |
| `ingest_batcher.rs` | Request-level coalescer for concurrent `/api/memory/store` — folds QualityGate in-line; async classify/extract; passes enrichment + hint through in the response |
| `knowledge_routes.rs` | Knowledge-directory path/count plus the recent-relations feed |
| `source_routes.rs` | Source registry endpoints |
| `import_routes.rs` | Bulk import endpoints |
| `config_routes.rs` | Config read/write endpoints |
| `onboarding_routes.rs` | First-run wizard / milestone state |
| `scheduler.rs` | Background periodic tasks (distill cycles, distillation, etc.) |
| `websocket.rs` | `/ws/updates` |
| `error.rs` | `ServerError` + axum `IntoResponse` impl |
| `resources/com.wenlan.server.plist` | launchd plist template (embedded via `include_str!`) |

## Manual RB-01 profiling flags

These flags control ignored, target-Mac profiling tests; they are not daemon runtime settings and must not be set in normal service configuration.

| Flag | Contract |
|---|---|
| `WENLAN_RB01_BASELINE` | Set to `1` to opt into the five-minute daemon-off resource baseline test. |
| `WENLAN_RB01_THERMAL_HELPER` | Optional path to the frozen helper executable that prints the macOS `ProcessInfo.thermalState` raw value; the test falls back to `/usr/bin/swift` when absent. |
| `WENLAN_RB01_CALIBRATION_LOAD_DUTIES` | Comma-separated synthetic-load duty percentages, each `1..=100`, with a total cap of `300`; must be supplied together with the CPU band. |
| `WENLAN_RB01_CALIBRATION_CPU_BAND` | Required `min:max` observed system-CPU percentage band for a calibrated profile; must be supplied together with load duties. Outside the band, the test records a skipped calibration and performs no inference. |
