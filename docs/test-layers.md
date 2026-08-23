# Test layers: local vs CI responsibilities

Extracted from the root `AGENTS.md` so it loads when you are adding a test or
changing CI routing, rather than in every session.

Wenlan runs across several layers. The split is driven by three questions: **(1) Can a hosted runner do this?** (no GPU, no API keys, no cost). **(2) Is it under 60s on cold cache?** **(3) Does it gate correctness or measure quality?** Quality measures never gate.

### Terminology: e2e / smoke / live

- **`_e2e.rs`** (`chat_import_e2e`, `doc_reconcile_e2e`, `page_citations_e2e`, ...) = hermetic: full internal pipeline, in-process, external deps (the LLM) faked/stubbed. Fast, deterministic, CI-safe (L4).
- **`scripts/smoke-*.sh`** = HTTP black-box check against a running daemon. Depth varies — check whether the script actually invokes the real on-device model:
  - No real model touched (`smoke-folder-ingest.sh`, `smoke-linux.sh`, `smoke-windows.ps1`) → plain **smoke test**, CI-safe (L4).
  - Real on-device model touched → **live smoke test**, filename folds the qualifier in (`live-smoke-doc-reconcile.sh`, `live-smoke-page-citations.sh`, `live-smoke-distill.sh`), L7 manual-only (needs the qwen3-4b GGUF cached; GitHub runners have no Metal/GPU).
- Never write bare "smoke test" for a GPU-gated script — the word alone doesn't signal depth. Always pair it with "live" so the non-hermetic tier is legible at a glance, in code comments and docs alike.

| Layer | What runs | Where | When | Time | Blocks? |
|---|---|---|---|---|---|
| **L1 dev loop** | rust-analyzer / IDE | Local | Every save | <1s | No |
| **L2 pre-commit** | `cargo fmt --all -- --check`; Clippy on directly changed crates only | Local | `git commit` | ~5s | Yes |
| **L3 pre-push** | Reader-inventory freshness check (Python, no compile); `WENLAN_PUSH_FULL=1` opts into the planner-selected Clippy + lib-test closure | Local | `git push` | ~4s when core/server changed, ~0.1s otherwise (opt-in run: change-dependent) | Yes (fast checks only) |
| **L4 CI on PR/main** | Fail-closed differential plan: affected lib, integration, contract, platform, and HTTP smoke owners only; aggregate `conclusion` verifies every expected job. Pushes to `main` reuse the same source-owned routing; release-sensitive pushes retain the Windows release-profile cache warmer, while CI-only pushes skip it. Manual dispatch is the full backstop. An exact same-repository Release PR whose current-main diff passes the semantic validator omits duplicate base-tree Rust/platform lanes only after independently proving that base's main CI succeeded; release-managed plugin/npm/docs checks and all four shipped-target preflights remain. | GitHub (`ci.yml`) | Every PR/main push | target ≤20min | Yes (required) |
| **L5 coverage** | `cargo llvm-cov` on wenlan-core + wenlan-server only | GitHub (`coverage.yml`) | Relevant source-owner push to `main`, or manual dispatch | ~30min | **No (informational)** |
| **L6 main canary** | Exact retrieval-quality + ranking-drift pair (`test_run_quality_cost_eval_basic`, `ranking_drift_vs_golden`) | GitHub (`main-canary.yml`) | Relevant core/eval-owner push to `main`, or manual dispatch | <20min | No (post-merge) |
| **L7 manual local** | `bash scripts/coverage.sh` (HTML coverage), GPU eval suite (`cargo test -- --ignored`), Anthropic batch judge (`ANTHROPIC_API_KEY=... cargo test ...`), live smokes with a real on-device judge (`bash scripts/live-smoke-doc-reconcile.sh`, `bash scripts/live-smoke-page-citations.sh`, `bash scripts/live-smoke-distill.sh`) — run the matching live smoke before merging a feature whose e2e stubs the LLM or never boots the daemon | Your laptop | On demand | minutes-hours | No |
| **L8 pre-release** | Full eval suite vs saved baseline. Commit a **curated, env-stamped snapshot** of headline numbers to a results doc/README (single-run tagged "scaffold"; headline claims need N≥3 + stddev). Raw per-run baselines + history series stay gitignored. See "Commit policy" under Eval Citation Discipline. | Your laptop | Per release | hours | Soft gate |

Windows release preflight treats cache availability as an optimization: a measured host+target cold miss waits 25 seconds and makes one pinned restore-only retry before Cargo. The final probe fails on partial or exact-but-empty state, records exact/fallback/cold plus the selected Cargo job bound, and continues a coherent cold build with two jobs.

A required CI check failed intermittently? Follow [`ci-flake-policy.md`](ci-flake-policy.md) before rerunning, quarantining, rerouting, or reverting.

### What does NOT run in CI and why

- **GPU evals (LongMemEval / LoCoMo runner functions, Qwen3.5-9B inference)** — GitHub macOS runners have no Metal acceleration. The tests are `#[ignore]`d so they don't accidentally run.
- **Anthropic API batch judge** — costs $0.35/run and requires `ANTHROPIC_API_KEY` which we don't expose to PR runs from forks.
- **Tauri / desktop coverage** — the desktop app (`app/`) is format/clippy/test-checked and e2e-run by the path-filtered `app-check` CI job on macos-14. Line/branch coverage (`.github/workflows/coverage.yml`) stays scoped to `wenlan-core + wenlan-server`.

### Why pre-push doesn't run coverage

Tried 90% `cargo llvm-cov` gate in pre-push, removed because:
- **Slow:** instrumented rebuild 5-15min, memory pressure.
- **Not mirrored in CI:** `ci.yml` has no coverage gate, so local-only friction.
- **Percentage gates rot:** new untestable surface forces busywork.

Pre-push now runs planner-selected Clippy + non-instrumented tests only. Coverage = L5 (main/manual, informational) or L7 (manual HTML).

### Why pre-push doesn't run the full workspace lib suite

The planner is fail-closed, so a workflow file, an unowned path, or any shared
build input widens the plan to the whole workspace. Running that locally cost
more than it caught:

- **Slow, and worst exactly where it is least informative:** over an hour on
  Windows for changes — a YAML edit, a new script — that are the least likely to
  break a specific package.
- **Already owned by CI:** `workspace-lib-required` runs the same suite on every
  PR and main push, so the local run was duplicate work, not extra coverage.
- **It made whole platforms unpushable:** any Windows-only test failure turned
  every broad change into a hard block, regardless of what the change touched.

A widened plan now prints `local checks: plan widened to the full workspace; CI
owns that suite` and stops at Clippy. Narrow plans still run their own lib and
integration tests locally, which is where the fast local signal is worth paying
for.

**What CI does not own: Windows-gated unit tests.** `workspace-lib-required`
runs on Linux and macOS. The Windows job deliberately skips the workspace lib
suite — see the "Workspace lib tests skipped on Windows" step comment in
`ci.yml` — because Linux and macOS catch the same cross-platform unit logic in
a third of the wall-clock. A `#[cfg(windows)]` unit test therefore has no
required-CI execution on any platform; `projection_state_mode_tracks_windows_readonly_attribute`
is one such test.

Trimming the widened pre-push plan removed the one lane that did run them.
Before that change a widened plan scheduled `cargo test --workspace --exclude
wenlan-app --lib` and the `wenlan-server` binary tests, so a Windows developer
pushing a broad change executed every `#[cfg(windows)]` unit test on the only
platform that compiles them. Nothing runs them now — not a required check, not
a local hook, on any platform.

That is the cost of the trade the section above describes, taken knowingly: the
same run took over an hour on Windows and blocked every broad change on a
failure unrelated to it. Windows behaviour is still covered at a coarser grain
by the Windows CLI/server integration step and the schtasks install round-trip
in the same job. Closing the unit-level gap means paying the ~25 minute Windows
lib run on every PR, which is a separate decision from this one.

### Why pre-push no longer compiles

The route-catalog `cargo test` and the `ci_test_plan.py` planner call each
compiled `wenlan-core` from scratch: ~5 min warm, 19–45 min cold, and both
contended with rust-analyzer for the `target/` directory lock. Change
detection was also wrong — it diffed against the stdin `remote_sha`, the *old*
remote tip, so a rebase onto a newer `main` plus a force-push made every
upstream commit look like a local change, pulling the Rust closure into a
docs- or plugin-only push. CI already owns clippy, the lib suites, and the
route-catalog test (it's a `wenlan-core` lib test, so the nextest slices run
it). Community consensus puts a local hook's ceiling at 30 s–2 min; past that,
routine `--no-verify` is a hook defect, not developer misconduct. The heavy
affected-closure run stays available on demand via `WENLAN_PUSH_FULL=1`.

