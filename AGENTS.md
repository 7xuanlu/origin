# AGENTS.md

Wenlan is a local-first agent memory system. This monorepo ships the daemon, CLI,
shared wire types, business-logic core, MCP server, and Tauri desktop app.

Keep this file to repository-wide behavior. Detailed facts, flag receipts, eval
methodology, and historical decisions belong in the owning reference or runbook and
are read only when the task needs them.

## Start here

| Work | Read |
|---|---|
| Business logic, storage, retrieval, enrichment | `crates/wenlan-core/AGENTS.md` |
| HTTP daemon and runtime workers | `crates/wenlan-server/AGENTS.md` |
| Desktop app and frontend | `app/AGENTS.md` or the closest nested `AGENTS.md` |
| React frontend (Tauri UI) | `src/AGENTS.md` |
| Browser e2e tests | `e2e/AGENTS.md` |
| Browser-only preview harness | `preview/AGENTS.md` |
| Release, sidecar, and repo-inventory scripts | `scripts/AGENTS.md` |
| Eval fixtures and artifacts | `app/eval/AGENTS.md` |
| Rust eval runners and experiment design | `crates/wenlan-core/src/eval/AGENTS.md` |
| Test routing and CI layers | `docs/test-layers.md` |
| Platform and release work | `docs/cross-platform.md`, `RELEASING.md` |
| Live daemon/app verification | `.claude/skills/run-wenlan/SKILL.md`, then the relevant verify/prove skill |

## Build and verification

The default Cargo workspace covers the five daemon crates; the desktop app must be
selected explicitly with `-p wenlan-app`.

- During implementation, run the smallest test that exercises the changed behavior.
- Before publishing a stable head, run the affected-package and reverse-dependency
  closure selected by the existing CI planner, or locally via `WENLAN_PUSH_FULL=1
  git push` (pre-push is fast/non-compiling by default; CI is the required gate).
- The full workspace suite is the CI/main backstop, not a per-worker or per-review-pass
  default. Do not rerun an unchanged, trustworthy receipt without a concrete reason.
- Eval quality runs, live GPU smokes, and deployed checks are measurement lanes. Run
  them only when the change or requested evidence requires them.
- A missing fixture or unavailable gate is `unchecked`, never a pass. Follow
  `docs/ci-flake-policy.md` before rerunning or quarantining an intermittent CI failure.

Hooks install automatically via `package.json`'s `postinstall`
(`scripts/install-git-hooks.mjs`) when you run `pnpm install`. The manual fallback for
a checkout without `pnpm install` is `git config core.hooksPath .githooks`. Do not
claim hooks are active without checking the current worktree.

## Repository invariants

- `wenlan-core` stays framework-agnostic: no Tauri or Axum dependencies.
- `wenlan-types` stays lightweight: its production dependencies are only `anyhow`,
  `serde`, and `serde_json`. Shared wire shapes live there; do not add `chrono`,
  `tokio`, or other heavy dependencies.
- `wenlan-server` owns HTTP framing and the single database writer. Business logic
  belongs in `wenlan-core`; the app and CLI use daemon APIs.
- MCP wrappers typed-deserialize daemon responses; do not pass arbitrary
  `serde_json::Value` envelopes through the boundary.
- Never hold a `tokio::sync::RwLock` guard across `.await`; snapshot state and drop the
  guard first.
- Canonical write-time enrichment routing: see `crates/wenlan-core/AGENTS.md` ("Core
  invariants").
- Dev and production use the same daemon port and data roots by default. Isolated tests
  that can create pages must isolate the daemon port, `WENLAN_DATA_DIR`, and the
  configured knowledge/page path, then verify the live daemon reports those scratch
  paths. For a scratch daemon, the CLI reads `WENLAN_HOST` (a full URL such as
  `http://127.0.0.1:17917`, not a port), `wenlan-mcp` takes `--origin-url`, and the
  default pages folder is `.wenlan/pages` under the OS user-home directory, not under
  `WENLAN_DATA_DIR` — a portable scratch setup sets `knowledge_path` explicitly.
- Request defaults: `WENLAN_AGENT_NAME` supplies the CLI identity when no explicit
  agent name is given; `WENLAN_DEFAULT_SPACE` is an overridable CLI/MCP fallback, while
  the strict `WENLAN_SPACE` lock wins over both explicit and default values.
- Daemon recovery: on a connect failure to a loopback host the CLI starts the registered
  background service once and re-polls health before failing. `WENLAN_NO_AUTOSTART=1`
  turns that off — every test harness and CI job must set it so a run can never start the
  developer's daemon. The `autostart.off` marker file in the data root is the persistent
  equivalent (`wenlan background off` writes it).
- Preserve cross-platform behavior on macOS arm64, Linux x86_64/aarch64, and Windows
  x86_64. Read `docs/cross-platform.md` before changing platform conditionals or release
  matrices.

## Git and release

Use conventional PR titles because release-please uses the squash title for the
changelog. Version changes must keep `version.txt`, `.release-please-manifest.json`, the
workspace version, and the app version surfaces in lockstep; the existing verifier is
the authority.

Removing a worktree also removes its gitignored artifacts. Before cleanup, check for
the only copy of eval databases, model downloads, or other valuable caches and migrate
them to their documented external cache location.

## Documentation ownership

An `AGENTS.md` instruction must change agent behavior in that directory. Module maps,
environment-flag wiring, experimental receipts, benchmark statistics, and long-form
rationale go in the closest `REFERENCE.md` or existing `docs/` page. Link to them from
the short instruction file; do not copy them back into always-loaded context.
