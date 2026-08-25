# crates/wenlan-server

Applies to agents working under `crates/wenlan-server/`. Read alongside root `AGENTS.md`, which takes precedence on any topic not covered here.

HTTP daemon — owns the Axum router + all routes. All handlers operate on `Arc<RwLock<ServerState>>` where `ServerState.db: Option<Arc<MemoryDB>>`.

The module map and the RB-01 profiling env-flag table live in `REFERENCE.md`; read it
when a task needs the module list or those flags.

## Agent `trust_level` gates a write, not only visibility

`agent_connections.trust_level` (`full`, `review`, `unknown`) used to govern only
page-read visibility and confidence scoring. A `/api/memory/store` request that
carries `supersedes` also stages instead of taking effect when the storing
agent's `trust_level` is not `"full"` — see `memory_routes.rs`, the
`pending_revision` decision next to where `trust_level` is resolved. The
superseded memory stays live and searchable until a human accepts the staged
row from the existing pending-revisions queue; nothing else about the store
path changes.

## Adding or rescoping a route

- If the route reads user data, register it in `crates/wenlan-core/src/lint/serving/routes.rs`.
- Add it to `SCOPED` or `GLOBAL` in
  `crates/wenlan-core/src/lint/serving_review_test.rs`.
- The reader inventory is regenerated and staged by the pre-commit hook;
  review it in your diff. If hooks are not installed, run
  `python3 scripts/m5-reader-sweep.py --update-inventory` yourself.

Pre-commit and pre-push run the reader-inventory check first, so a missed step
fails in seconds. The SCOPED/GLOBAL route-catalog test is a `wenlan-core` lib
test: CI always runs it, and pre-push runs it locally only under
`WENLAN_PUSH_FULL=1`.
