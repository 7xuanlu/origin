# Wenlan App Bundled Runtime Checkpoint

- **Date:** 2026-06-29 UTC / 2026-06-28 PDT
- **Worktree:** `/Users/lucian/Repos/wenlan-app/.worktrees/wenlan-app-api-parity-audit`
- **Branch:** `codex/wenlan-app-bundled-runtime`
- **Daemon probe:** `127.0.0.1:7878`, version `0.9.1`

## Scope

This checkpoint follows the runtime-asset PR and validates the stronger macOS runtime path: a built `Wenlan.app` bundle, not the unbundled `pnpm tauri dev` binary.

The immediate migration gap was that `pnpm tauri build --bundles app` produced `target/release/bundle/macos/Wenlan.app` but exited 1 when updater signing keys were absent:

```text
A public key has been found, but no private key. Make sure to set TAURI_SIGNING_PRIVATE_KEY environment variable.
```

That is correct for release artifacts, but too strict for local Dock/window/bundle validation. The fix adds `pnpm build:app:local`, which builds a local app bundle with updater artifact creation disabled only through `app/tauri.local-bundle.conf.json`. Production `app/tauri.conf.json` still has `bundle.createUpdaterArtifacts = true`, and release scripts do not reference the local override.

## Structural Tool Boundary

| Tool | Boundary |
|---|---|
| CodeGraph | `sync .` was already up to date; `query run`, `query set_macos_application_icon_once`, and `query stable_launch_agent_target` bounded the runtime identity surfaces before changing scripts/config. |
| ast-grep | `outline app/src/lib.rs` and `outline app/src/lifecycle.rs` confirmed no Rust runtime changes were needed for this checkpoint. |
| LSP | No callable LSP tool is available in this Codex session, so compiler/test output plus bounded `rg` served as the fallback for symbol and config checks. |
| `rg`/file reads | Used after structural probes to inspect `bundle`, `icon.icns`, `Info.plist`, updater, activation policy, and launch path references. |

## Red/Green Evidence

RED:

```text
pnpm vitest run src/runtimeIdentity.test.ts
1 failed: ENOENT app/tauri.local-bundle.conf.json
```

GREEN:

```text
pnpm vitest run src/runtimeIdentity.test.ts
9 passed
```

Local bundle validation:

```text
pnpm build:app:local
Finished 1 bundle at target/release/bundle/macos/Wenlan.app
```

The bundle command ad-hoc signed the app/sidecars and skipped notarization because Apple signing credentials are absent. It exited 0.

## Bundle Metadata

```text
CFBundleDisplayName = Wenlan
CFBundleExecutable = wenlan-app
CFBundleIconFile = icon.icns
CFBundleIdentifier = com.wenlan.desktop
CFBundleName = Wenlan
CFBundlePackageType = APPL
```

```text
codesign Identifier=com.wenlan.desktop
codesign Format=app bundle with Mach-O thin (arm64)
codesign Signature=adhoc
```

```text
Contents/Resources/icon.icns: Mac OS X icon
Contents/MacOS/wenlan-app: Mach-O 64-bit executable arm64
Contents/MacOS/wenlan: Mach-O 64-bit executable arm64
Contents/MacOS/wenlan-server: Mach-O 64-bit executable arm64
Contents/MacOS/wenlan-mcp: Mach-O 64-bit executable arm64
Contents/MacOS/cloudflared: Mach-O 64-bit executable arm64
```

## Runtime Evidence

Launched with:

```text
open -n target/release/bundle/macos/Wenlan.app
```

macOS Accessibility reported:

```text
bundle identifier: com.wenlan.desktop
displayed name: Wenlan
title: Wenlan
windows: 1
window name: Wenlan
AXMain: true
AXFocused: true
position: 224, 199
size: 1280, 720
```

Dock item list included:

```text
Wenlan
```

Daemon probe:

```json
{"status":"ok","db_initialized":true,"version":"0.9.1"}
```

App logs showed the bundle launch path:

```text
target/release/bundle/macos/Wenlan.app/Contents/MacOS/wenlan-app
```

The only repeated warnings were the known updater endpoint warning and first-run LaunchAgent skip for a non-stable worktree bundle path.

## Visual Capture Caveat

macOS `screencapture` repeatedly captured Codex/Cursor rather than the Wenlan window even when Accessibility reported the Wenlan app and window as present and focused. Treat AX, Dock, bundle metadata, daemon health, and logs as the authoritative evidence for this checkpoint; do not use those screenshots as proof.
