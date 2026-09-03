---
name: run-wenlan-app
description: Build, launch, screenshot, and drive the wenlan-app Tauri desktop app in dev mode on macOS (WKWebView) or Windows (WebView2). Use when asked to run or start the app, verify a UI change in the real running app (not just tests), or take a screenshot of it.
---

# Run wenlan-app (dev)

Tauri 2 desktop app (Rust `app/` + React/Vite frontend on :1420). Drive it
with `.claude/skills/run-wenlan-app/driver.sh` — all paths below are relative
to the unit root (repo or worktree root).

**Prerequisites, both Run sections, Gotchas, and Troubleshooting below are
macOS only** (`driver.sh`, vite, ScreenCaptureKit). On Windows the webview is
WebView2 and the recipe is different — jump to
[Windows (WebView2)](#windows-webview2).

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

## Windows (WebView2)

`driver.sh`, vite, and `wincap.swift` are macOS-only. On Windows the app hosts
WebView2, is driven over WebDriver, and is captured with `PrintWindow`. There
is no committed Windows driver script yet; the recipe below is what a live run
on Windows 11 actually required.

### 1. Build

```bash
export CARGO_TARGET_DIR=C:/wl-target                 # warm cache; keeps target/ off the worktree
export PATH="/c/Strawberry/perl/bin:$PATH"           # see gotcha: openssl-sys needs a real perl
cargo build -p wenlan-app --features tauri/custom-protocol
```

`--features tauri/custom-protocol` is the important part: it flips
`tauri-macros`' `dev` flag off, so the binary **embeds `dist/`** instead of
loading `devUrl`. That removes vite and :1420 from the picture entirely — a
plain `cargo build -p wenlan-app` gives you the macOS-style dev binary that
needs a vite server on :1420. Run `pnpm build` first so `dist/` is current.

`wenlan-app` depends on `wenlan-types`, **not** `wenlan-core`, so the app
builds even on a host where the daemon crates cannot (see the Vulkan gotcha).

### 2. Isolated runtime identity

A debug build refuses to start without one (`validate_debug_runtime_isolation`,
`app/src/lib.rs`). Take the worktree-scoped values from
`bash scripts/dev-runtime.sh print-config` and export `WENLAN_PORT`,
`WENLAN_DEV_UI_PORT`, `WENLAN_DEV_REMOTE_PORT_START`, `WENLAN_DEV_APP_ID`,
`WENLAN_DEV_STATE_DIR`, `WENLAN_DATA_DIR`, `WENLAN_DEV_TAURI_MCP_SOCKET`, plus
`WENLAN_NO_AUTOSTART=1`.

Two things that are easy to miss:

- **The bundle identifier must be overridden at build time**, not runtime.
  `tauri_plugin_single_instance` keys on it, so a debug build carrying the
  production `com.wenlan.desktop` will hand its argv to the user's real running
  app instead of starting. The only lever without the tauri CLI is `TAURI_CONFIG`
  (json_patch-merged by both `tauri-build` and `tauri-codegen`):
  ```bash
  export TAURI_CONFIG='{"identifier":"com.wenlan.desktop.dev.<worktree-hash>"}'
  ```
  Set it for the build and confirm it landed: `strings` the binary for the id.
- **`knowledge_path` must be written explicitly** into
  `$WENLAN_DATA_DIR/config.json`. It defaults to `.wenlan/pages` under the OS
  user home — *not* under `WENLAN_DATA_DIR` — so without it an "isolated" run
  writes pages into the developer's real memory.

### 3. Daemon

The app **does** spawn a sidecar daemon on Windows, and it is the
`cfg!(target_os = "macos")` flag that makes it do so: `app/src/lib.rs:1118`
reads `if !launch_agent_startup { … spawn_daemon_sidecar(…) }`, so the branch
always fires off macOS. The macOS path hands ownership to a launchd LaunchAgent
instead. On Windows the app *tries* to bind the sidecar into a kill-on-close job
object (`app/src/daemon_start.rs:356`). That is best-effort, not a guarantee:
the bind can fail (a restricted token, an assignment the OS refuses), and
startup deliberately continues on failure rather than leaving the user with no
daemon. The outcome is recorded on the handle as `JobBinding::Bound` /
`Unbound { reason }` / `NotSupported` and read back through
`sidecar_job_binding()` — ask it, never assume. So: a **bound** sidecar dies
with the app, hard kill included. An **unbound** one is ended only by the exit
paths that run app code (tray Quit, SIGTERM, the quit-path kill verified against
the recorded pid *and* process start time); a hard kill of the app runs no app
code, so an unbound sidecar survives it and holds the port. Confirm the binding
before reporting a clean teardown, and kill by image path either way.

Start your own isolated daemon first anyway. Not because the app cannot, but
because you need one whose port, data dir and pid you recorded: the app's
sidecar, finding yours already healthy on the port, logs
`[daemon] Existing healthy daemon on port <n> — exiting cleanly` and forgets the
child (`app/src/daemon_start.rs:371`), which is the state the ownership records
in `WENLAN_DEV_STATE_DIR` describe.

`scripts/dev-runtime.sh start` is the supported way and it runs
`cargo build -p wenlan-server`. That needs the **Vulkan SDK**, a build-time
prerequisite `wenlan-core` pulls in through `llama-cpp-2`'s `vulkan` feature;
without it `llama-cpp-sys-2`'s build script panics and no daemon crate can be
rebuilt. Install it exactly as CI does — `scripts/setup-vulkan-sdk-windows.ps1`,
which pins LunarG 1.4.350.0 by SHA-256 and installs copy-only under
`%LOCALAPPDATA%\wenlan-build` — then export `VULKAN_SDK` in the build shell.
See `docs/windows-vulkan.md`. The installer ships the loader, and the published
ZIP and NSIS channels are measured *starting* and answering `/api/health` on a
driverless hosted runner (`first-run-gauntlet.yml`). What no gate has ever done
is load a GGUF on a machine with no vendor ICD — the CPU plan sits downstream of
`LlamaBackend::init()` (`crates/wenlan-core/src/engine.rs:33`), so end-user
impact is unproven in both directions. Do not repeat "end users are unaffected"
as fact; `docs/windows-vulkan.md` names the driverless-VM gate that would settle
it.

Falling back to a prebuilt `wenlan-server.exe` is a last resort: stage it and
write the same ownership records (`wenlan-server.{pid,path,port,data-dir}` in
`WENLAN_DEV_STATE_DIR`) so `dev-runtime.sh stop` still recognizes it — and say
plainly in any report that the daemon was prebuilt, not built by that run.

### 4. Launch and drive (WebDriver)

```bash
tauri-driver --port 17942 --native-port 17943 --native-driver /path/to/msedgedriver.exe
```

**`msedgedriver.exe` must match the installed WebView2 runtime exactly**
(`151.0.4129.107` ↔ `151.0.4129.107`). Check the runtime with the version of
`C:\Program Files (x86)\Microsoft\EdgeWebView\Application\<version>\`; a
mismatched driver fails the session with a version error, not a useful one.

Create the session with the Tauri capability; msedgedriver launches the app,
so **the app inherits the driver's environment** — export the isolated identity
in the shell that starts `tauri-driver`.

```json
{"capabilities":{"alwaysMatch":{"tauri:options":{"application":"C:/wl-target/debug/wenlan-app.exe"}}}}
```

The app exposes **three** window handles — main (`location.hash === ""`),
`#quick-capture`, and `#toast`. Enumerate handles and pick by hash; do not
assume the first. A capture → search round-trip is: switch to main → click
`button[title^='Quick Capture']` → switch to the `#quick-capture` handle → type
into its `textarea` → click Save → switch back to main → type into
`[data-wenlan-search-input]`.

### 5. Screenshot

WebDriver `/screenshot` captures the webview only. For the real Win32 window
(frame included, and it works while occluded — the analogue of macOS
ScreenCaptureKit) use `PrintWindow` with `PW_RENDERFULLCONTENT` (`0x2`):
enumerate top-level windows with `EnumWindows`, keep the visible ones whose
`GetWindowThreadProcessId` matches the app pid, and take the **largest by
area** — the app owns secondary windows, same rule as `wincap.swift`. To
photograph a *secondary* window (the quick-capture composer, say), select it by
title instead; largest-by-area would silently hand back the main window.

**Declare per-monitor DPI awareness first, or the capture crops.**
`powershell.exe` is DPI-unaware, so on a scaled display Windows virtualizes its
coordinates: at 125% scaling `GetWindowRect` on the 1600x900 main window returns
1294x758, `PrintWindow` renders into a DC built from those numbers, and the
bottom-right fifth of the window — window controls, account chip, Save button —
is cropped away. It reads as an app layout bug and is not one. Call
`SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)`
(`-4`) before reading any geometry, and treat a NULL return as fatal. **That
fatal NULL check is the only thing that catches virtualization**, because a
DPI-unaware process is self-consistent: it reads a virtualized `GetWindowRect`,
renders into a DC built from those same shrunken numbers, and every measurement
it can take of itself agrees at the wrong size. Nothing downstream can tell.

The separate cross-check — that the bitmap you got is the rectangle you asked
for — is against **measured geometry, never against the other PNG**. The two
captures photograph different rectangles and are *supposed* to differ:
WebDriver `/screenshot` returns the **client surface**; `PrintWindow` returns the
**outer window, frame included**. So:

| PNG | must equal (DPI-aware) |
| --- | --- |
| WebDriver `/screenshot` | `GetClientRect` |
| `PrintWindow` | `GetWindowRect` |

Comparing the two PNGs to each other is **not** a valid check in either
direction. Requiring them to match rejects correct captures: this app's real
ones are 500x200 client against 518x210 outer for the undecorated quick-capture
window, and 1600x900 against 1618x947 for the decorated main window. Those
deltas are the frame — 18x10 with no title bar, 18x47 with one — and they are
what a correct capture looks like. **A frame difference is not a crop; do not
"fix" it.**

And requiring them to match would never have caught the original bug either.
Virtualization scales *every* rect a DPI-unaware process reads by the same
factor, so at 125% the client rect and the window rect shrink together and the
frame delta between them shrinks with them: both PNGs come out wrong, in the
same proportion, and a PNG-to-PNG comparison — equality, ratio, or delta —
passes on the cropped capture. No arrangement of the two bitmaps detects it.
The fatal NULL check on `SetThreadDpiAwarenessContext` is the only guard, which
is why the cross-check is against DPI-aware measured geometry and never against
the other PNG.

`wincap.ps1` enforces the `PrintWindow` half itself: it re-reads the DPI-aware
`GetWindowRect` after the capture and exits **8** if the PNG's pixel dimensions
disagree, and prints the `GetClientRect` dimensions beside it so a caller can
check the WebDriver half. A rect it cannot read is COULD-NOT-MEASURE (exit
**9**) — never a pass. Measure the PNG from its own IHDR bytes, not from the
in-memory `Bitmap`: re-reporting the object you just encoded checks nothing.

**Distinct exit codes only survive if the script never uses `Write-Error` to
report them.** Under `$ErrorActionPreference = 'Stop'`, `Write-Error` is
*terminating*: it unwinds the script before the `exit N` on the next line ever
runs, and `powershell.exe` reports **1**. Every outcome — cropped capture,
unreadable rect, no such window, DPI awareness refused — then arrives at the
caller as the same anonymous 1, which is the "failed measurement that looks
like every other failure" defect wearing a different hat. Write the message
with `[Console]::Error.WriteLine($msg)` and then `exit` the code you mean.

Then check the frame is not near-uniform and report a flat frame as *unmeasured*
rather than as a blank UI.

### 6. Teardown

Kill by `ExecutablePath`, never by image name — `msedgedriver.exe` and
`wenlan-server.exe` may also belong to the developer's real session. Deleting
the WebDriver session closes the app gracefully; then stop the drivers and the
daemon, and **measure** the result: no process per image path, and no LISTENING
socket on the isolated ports. `Stop-Process` is not synchronous — poll for
actual exit rather than asserting it.

Filter in PowerShell, not in WQL. `Get-CimInstance Win32_Process -Filter
"Name='x'"` needs single quotes around the value, and inside a bash
single-quoted `-Command` block every `''` collapses to nothing, so the query
arrives as `Name=x` — invalid. The error goes to stderr and the empty result
then reads as "that process is not running": the teardown reports a clean kill
it never performed, and the "is it gone?" check answers NONE because the query
failed. Use `Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object {
$_.ExecutablePath -ieq $want }`, which needs no quoting the shell can eat, and
print the row count beside a negative so an unreadable table is distinguishable
from an empty one.

### Windows gotchas

- **A search miss contains the query.** The empty state renders
  `main.search.noResults` = `0 results for "<query>"`, so asserting
  `main.innerText.includes(sentinel)` **passes on a miss**. Assert on the hit
  header instead (`main.search.memories_*` → `N memories for "<query>"`, only
  rendered when `memoryResults.length > 0`) plus a result card containing the
  sentinel.
- **Search is debounced.** Reading the DOM immediately after typing measures
  the pre-request state. Poll for the result, don't assert once.
- `GET /api/search` returns **405** with `allow: POST`. It is
  `POST {"query": "..."}`.
- `openssl-sys` vendored build fails with
  `Can't locate Locale/Maketext/Simple.pm` — Git Bash puts
  `C:\Program Files\Git\usr\bin\perl.exe` ahead of Strawberry Perl in the PATH
  that `CreateProcess` searches. Prepend `/c/Strawberry/perl/bin` **only**;
  adding Strawberry's `c/bin` shadows MSVC's cmake.
- `llama-cpp-sys-2` build script panics with
  `Please install Vulkan SDK and ensure that VULKAN_SDK env variable is set`.
  `wenlan-core` pulls `llama-cpp-2` with the `vulkan` feature on Windows, so
  without the SDK **no daemon crate can be rebuilt** — `-p wenlan-server` alone
  is no exception. `C:\Windows\System32\vulkan-1.dll` is only the loader, not
  the SDK. A panicking build script never writes its `output`/`stderr` files,
  so a stale-but-present artifact is *not* evidence of a cache hit — read
  cargo's actual output, and never through a pipe (`| tail` gives you *tail's*
  exit code, not cargo's).

## Test

`pnpm test` (Vitest), `cd app && cargo test`.

## Gotchas (macOS)

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

## Troubleshooting (macOS)

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
