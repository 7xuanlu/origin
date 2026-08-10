# crates/wenlan-core

Applies under `crates/wenlan-core/`. The root `AGENTS.md` still applies.

This crate owns framework-agnostic business logic. Do not add Tauri or Axum dependencies,
and do not move HTTP framing into the core. Use `REFERENCE.md` only when a task needs the
detailed module map, database layout, feature-flag wiring, or historical measurement.

## Core invariants

- `MemoryDB` owns storage and holds its libSQL connection behind the established async
  mutex. Share the database through `Arc<MemoryDB>`; do not hold outer state guards
  across `.await`.
- UI notifications go through `EventEmitter`; daemon code uses `NoopEmitter` and the app
  supplies its adapter.
- All write consumers call `ingest::run_canonical_enrichment`. Add new write-time
  behavior there rather than recreating a subset in the server, importer, or eval seed.
- A new channel must have an executable substrate/liveness check. Do not report a null
  experiment over an empty or stale substrate as product evidence.
- Retrieval feature defaults and experiment receipts live in `REFERENCE.md` and
  `src/retrieval/REFERENCE.md`, not in this instruction file.

## Eval seed + eval read: ONE route, ONE contract (no drift)

Re-seed scenario databases through `seed_scenario_dbs_complete`; individual `seed_*`
steps are its internals, not an operator runbook. `eval/seed_contract.rs` is the shared
producer/consumer liveness contract. When adding a write-time channel, wire its seed
step, expectation, consumer assertion, and focused unit test together.

Runner conventions and experiment methodology live in `src/eval/REFERENCE.md`. Fixture,
artifact, and citation rules live in `app/eval/REFERENCE.md`.
