---
name: run-wenlan-app
description: Build, launch, screenshot, and drive the wenlan-app Tauri desktop app in dev mode on macOS. Use when asked to run or start the app, verify a UI change in the real running app (not just tests), or take a screenshot of it.
---

# Run wenlan-app (dev)

Tauri 2 desktop app (Rust `app/` + React/Vite frontend on :1420). Drive it
with `.claude/skills/run-wenlan-app/driver.sh` — all paths below are relative
to the unit root (repo or worktree root). macOS host only.

A debug build refuses to start without a complete isolated runtime identity
(`validate_debug_runtime_isolation` in `app/src/lib.rs`), so `launch` takes the
worktree-scoped config from `scripts/dev-runtime.sh` and starts its **own**
daemon on the worktree port. The user's live `wenlan-server` on `:7878` is
never the daemon under test and **must never be killed**; `stop` only ends the
daemon this worktree recorded.

## Prerequisites

- `pnpm install` done in the unit root (worktrees start without `node_modules`).
- No sibling checkout needed: the monorepo checkout is the backend, and
  `scripts/resolve-backend-dir.sh` probes the repo root first (an explicit
  `WENLAN_BACKEND_DIR` overrides it; a sibling checkout is only a legacy
  fallback). The driver defaults `CARGO_TARGET_DIR` to the main checkout (a
  warm cargo cache) via `git rev-parse --git-common-dir`, so worktrees work
  out of the box.
- Terminal needs macOS Screen Recording permission (for `shot`).

## Run (agent path)

```bash
.claude/skills/run-wenlan-app/driver.sh build    # sidecars + cargo build (decoupled from tauri dev)
.claude/skills/run-wenlan-app/driver.sh launch   # isolated daemon + vite up + launch target/debug/wenlan-app directly → "APP UP"
.claude/skills/run-wenlan-app/driver.sh shot /tmp/shot.png   # window PNG, works while occluded
.claude/skills/run-wenlan-app/driver.sh stop     # kills only the dev app, vite, and this worktree's daemon; never :7878
```

Logs land in `$TMPDIR/wenlan-app.log` and `$TMPDIR/wenlan-vite.log`.
**Look at the screenshot** after `shot` — a blank frame means the frontend
did not load.

### Driving UI state

Synthetic events (`CGEvent.postToPid`) are **not handled** by Tauri's
WKWebView, and global coordinate clicks are forbidden here (other live agent
sessions share this desktop; a stray click once hit the updater's Install
button and it consumed `target/debug/wenlan-app`). Instead, drive UI state
through **vite HMR**: make a temporary `import.meta.env.DEV`-guarded edit
(e.g. force a section expanded, early-return a toast), wait ~3s, `shot`,
revert the edit. Mark such edits `// TEMP (do not commit)`.

## Run (human path)

`pnpm dev:all` — the supported entry point: an isolated worktree daemon plus
`pnpm tauri dev` with a dev bundle identifier and the worktree UI port. Agents
use `driver.sh` instead because `tauri dev` dies on cold caches (see Gotchas).
Ctrl-C to stop.

## Test

`pnpm test` (Vitest), `cd app && cargo test`.

## Gotchas

- **`launch` must run unsandboxed.** The app inherits the shell's seatbelt
  sandbox, which denies the mach lookup for the ViewBridge XPC service behind
  `NSOpenPanel`. The app starts fine and then hard-crashes on the *first*
  file/folder picker — `+[NSOpenPanel openPanel]` returns NULL and objc2
  panics — hours after launch, looking exactly like a product bug. It is not:
  the same call returns a panel normally outside the sandbox. `launch` now
  refuses when it detects a sandboxed shell (canary: `pbpaste`).
- `tauri dev` compiles sidecars inside `beforeDevCommand`, then waits only
  180s for vite — on a cold cache it dies with
  `Error Could not connect to `http://localhost:1420/` after 180s`.
  The driver decouples: sidecars → vite → `cargo build` → direct launch.
- `app/Cargo.toml` has `default = []` (no `custom-protocol`), so a plain
  debug `cargo build` produces the dev binary that loads `devUrl` :1420.
  That URL is baked in at build time and no runtime env var overrides it, so
  `driver.sh vite` pins vite to :1420 even though the isolated runtime config
  also carries a worktree-scoped `WENLAN_DEV_UI_PORT` (which `pnpm dev:all`
  does use, since the tauri CLI can rewrite `devUrl` via `--config`).
- The dev updater toast covers the lower-left sidebar and its Install button
  is live. Never click it. Suppress during verification with a TEMP
  `if (import.meta.env.DEV) return null;` at the top of
  `src/components/UpdaterDialog.tsx`.
- The app owns a secondary ~500×500 window; `wincap.swift` sorts by area to
  capture the main one.
- Full-screen `screencapture` shows whatever is frontmost — on a desktop with
  a live user this races window focus. `shot` uses ScreenCaptureKit's
  desktop-independent window capture instead.

## Troubleshooting

- `initializing rolling file appender failed ... PermissionDenied` in the app
  log → the binary was spawned through a sandboxed pnpm/tauri chain; launch
  `target/debug/wenlan-app` directly (what `driver.sh launch` does).
- `could not create image from window` / `could not create image from rect`
  from `screencapture -l`/`-R` → use `driver.sh shot`.
- `Assertion failed: (did_initialize), function CGS_REQUIRE_INIT` from a
  Swift capture script → touch `NSApplication.shared` before creating an
  `SCContentFilter` (already done in `wincap.swift`).
- `failed to restart app: No such file or directory` after an updater
  Install → the updater consumed the dev binary; rebuild with
  `driver.sh build`.
- `vitest: command not found` in a fresh worktree → `pnpm install`.
