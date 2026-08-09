# Review flavor and Wiki/Space refinement implementation plan

> **Navigation contract superseded (2026-07-17):** Later user review made the
> Wiki row itself the Page destination and kept `Kind` as non-interactive
> metadata. Space remains the sole secondary row link. The durable Wiki kinds
> are now `Page | Entity`; semantic prose must not be guessed into
> Topic/Decision/Recap schemas.

> **For agentic workers:** Execute in this worktree with RED-first tests, then verify the rendered native Review bundle. Do not commit, push, publish, or touch production data.

**Goal:** Deliver every approved Review-isolation and Wiki/Space design correction in one verified pass.

**Architecture:** Add a compile-time `review-fixtures` Rust entry point plus a dedicated Tauri/Vite flavor. The Review frontend is bound directly to the in-memory Tauri mock runtime. Keep product UI changes inside the existing navigation/pages/spaces seams and lock visual semantics with focused tests.

**Tech Stack:** Tauri 2, Rust, React 19, TypeScript, Vite 6, Vitest, Playwright, Tailwind v4/CSS.

## Global constraints

- Preserve all unrelated dirty-worktree changes.
- Never launch the old `.omo` Review bundle or connect Review to `127.0.0.1:7878`.
- No new dependency for icons; use exact, source-backed standard geometry inline.
- All visible copy is localized in all three existing locales.
- Unit/build checks are necessary but not sufficient: inspect fresh rendered desktop and compact captures and smoke-test the native process.

## Task 1 — RED contracts for the safe Review flavor

- [x] Extend the runtime identity tests to require the Review name, bundle identifier, compile-time feature, fixture-only Vite aliases, sidecar/updater/tray removal, and dedicated scripts.
- [x] Add Review runtime tests for deterministic startup, reset-on-new-runtime, known fixture commands, and rejection of unknown IPC.
- [x] Add badge tests proving production-hidden, Review-visible, localized copy, and explicit reset behavior.
- [x] Run the focused tests and record the expected failures.

## Task 2 — Compile-time fixture-only Review shell

- [x] Add `review-fixtures` to `app/Cargo.toml` and select `run_review()` from `app/src/main.rs` only when compiled with that feature.
- [x] Implement a minimal Tauri builder in a gated review module with no production plugins, state, setup hook, invoke handler, tray, sidecar, daemon, sync, tunnel, updater, MCP, or global shortcut path.
- [x] Add `app/tauri.review.conf.json` with `Wenlan Review`, `com.wenlan.desktop.review`, Review-only frontend assets, no external binaries, no updater, no tray, and a minimal inline capability.
- [x] Add `vite.review.config.ts` and `review/tauri-core.ts` so the declared Review capability allowlist is served by a fresh in-memory fixture and commands outside that bounded surface fail closed.
- [x] Add build/open scripts and a bundle smoke verifier.

## Task 3 — Review proof-stamp and fixture coverage

- [x] Add a persistent Wenlan-styled `TEST DATA` proof-stamp and reset control to the shell header, gated by the compile-time build constant.
- [x] Add English, Simplified Chinese, and Traditional Chinese copy with key-parity tests.
- [x] Expand the Review fixture with Entity, ordinary Page, varied content, and no-Space examples while preserving existing navigation test requirements. The later `Page | Entity` decision supersedes heuristic Topic/Decision/Recap kinds.

## Task 4 — Wiki/Space hierarchy and links

- [x] Write failing tests for the three agreed title scales.
- [x] Replace the earlier chip-destination contract with tests for one full-row Page destination plus the sole secondary Space link in desktop and compact rows.
- [x] Apply the overview/detail title scales in the final cascade locations.
- [x] Keep Wiki rows routed to Page detail and Space metadata routed to Space detail; `Kind` remains non-interactive metadata under the later navigation decision.

## Task 5 — Memories glyph with the design lead

- [x] Ask Fable to select the exact standard brain geometry in the context of the rendered Wenlan sidebar.
- [x] Write a failing exact-geometry test, replace only the Memories SVG, and preserve every locked neighbouring navigation mark.

## Task 6 — Verification and handoff

- [x] Run focused Vitest suites, i18n tests, TypeScript, production build, Review web build, Rust tests, and Review feature compilation.
- [x] Build the native Review app; inspect its Info.plist and bundle contents.
- [x] Launch only the new Review bundle and prove no port-7878 connection, daemon child process, or production-log mutation.
- [x] Capture fresh Wiki, Spaces overview, Space detail, Entity navigation, and compact screenshots; perform two inline visual-QA passes and fix any critical issue.
- [x] Run the final regression floor and leave `Wenlan Review.app` open for user inspection until explicitly asked to close it; it was closed after that explicit request.
