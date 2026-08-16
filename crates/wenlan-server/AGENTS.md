# crates/wenlan-server

Applies to agents working under `crates/wenlan-server/`. Read alongside root `AGENTS.md`, which takes precedence on any topic not covered here.

HTTP daemon — owns the Axum router + all routes. All handlers operate on `Arc<RwLock<ServerState>>` where `ServerState.db: Option<Arc<MemoryDB>>`.

The module map and the RB-01 profiling env-flag table live in `REFERENCE.md`; read it
when a task needs the module list or those flags.

## Adding or rescoping a route

- If the route reads user data, register it in `crates/wenlan-core/src/lint/serving/routes.rs`.
- Add it to `SCOPED` or `GLOBAL` in
  `crates/wenlan-core/src/lint/serving_review_test.rs`.
- The reader inventory is regenerated and staged by the pre-commit hook;
  review it in your diff. If hooks are not installed, run
  `python3 scripts/m5-reader-sweep.py --update-inventory` yourself.

Pre-commit and pre-push run these two checks first, so a missed step fails in
seconds, not after the full suite.
