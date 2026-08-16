# crates/wenlan-server

Applies to agents working under `crates/wenlan-server/`. Read alongside root `AGENTS.md`, which takes precedence on any topic not covered here.

HTTP daemon — owns the Axum router + all routes. All handlers operate on `Arc<RwLock<ServerState>>` where `ServerState.db: Option<Arc<MemoryDB>>`.

## Key Modules (`crates/wenlan-server/src/`)

Route modules follow the file name: `*_routes.rs` owns the handlers for that surface and exposes one or more `TrackedRouter` registration helpers, which `router.rs` composes. Where a module has two or more helpers it is because their composition positions are separated and must stay that way. Read the directory for the current set.

The modules whose job is not evident from the name:

| Module | Purpose |
|---|---|
| `main.rs` | Binary entry — daemon startup plus internal maintenance commands, tracing init, port binding with existing-daemon fallback, `MemoryDB::new`, LLM provider init, background tasks, `axum::serve` |
| `state.rs` | `ServerState` with `db: Option<Arc<MemoryDB>>`, `llm`, `prompts`, `tuning`, `quality_gate`, `space_store`, `access_tracker`, `llm_processing_ids`, `watch_paths`. `SharedState = Arc<RwLock<ServerState>>` |
| `router.rs` | Axum composition root — assembles the module-owned registration helpers plus the remaining inline registrations, then applies the truth/security/lifecycle layers |
| `routes.rs` | General endpoints: health, status, search/context, diagnostics, recent activity, steep/distill |
| `memory_routes.rs` | Memory CRUD/search/enrichment, classification, statistics, scheduler-handoff, rerank, attribution, update, revision, contradiction tests |
| `ingest_batcher.rs` | Request-level coalescer for concurrent `/api/memory/store` — folds QualityGate in-line, async classify/extract, passes enrichment + hint through in the response |
| `scheduler.rs` | Background periodic tasks (distill cycles, distillation, the reconcile/backfill sweeps gated by the `WENLAN_ENABLE_*` flags) |

## Adding or rescoping a route

- If the route reads user data, register it in `crates/wenlan-core/src/lint/serving/routes.rs`.
- Add it to `SCOPED` or `GLOBAL` and update the count in
  `crates/wenlan-core/src/lint/serving_review_test.rs`.
- Run `python3 scripts/m5-reader-sweep.py --update-inventory` and stage the
  regenerated inventory.

Pre-commit and pre-push run these two checks first, so a missed step fails in
seconds, not after the full suite.

## Manual RB-01 profiling flags

These flags control ignored, target-Mac profiling tests; they are not daemon runtime settings and must not be set in normal service configuration.

| Flag | Contract |
|---|---|
| `WENLAN_RB01_BASELINE` | Set to `1` to opt into the five-minute daemon-off resource baseline test. |
| `WENLAN_RB01_THERMAL_HELPER` | Optional path to the frozen helper executable that prints the macOS `ProcessInfo.thermalState` raw value; the test falls back to `/usr/bin/swift` when absent. |
| `WENLAN_RB01_CALIBRATION_LOAD_DUTIES` | Comma-separated synthetic-load duty percentages, each `1..=100`, with a total cap of `300`; must be supplied together with the CPU band. |
| `WENLAN_RB01_CALIBRATION_CPU_BAND` | Required `min:max` observed system-CPU percentage band for a calibrated profile; must be supplied together with load duties. Outside the band, the test records a skipped calibration and performs no inference. |
