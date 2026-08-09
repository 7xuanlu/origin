# Wenlan App Runtime Asset Checkpoint

- **Date:** 2026-06-29 UTC / 2026-06-28 PDT
- **Worktree:** `/Users/lucian/Repos/wenlan-app/.worktrees/wenlan-app-api-parity-audit`
- **Branch:** `codex/wenlan-app-runtime-assets`
- **Daemon probe:** `127.0.0.1:7878`, version `0.9.1`

## Scope

Focused runtime-identity follow-up after the sidecar and API-parity checkpoints:

- Dock/app identity should not depend on the frontend `app-ready` event.
- The app should set the full Wenlan/Origin ring icon before applying macOS activation policy.
- A delayed backend reveal pass should recover launches where the frontend-ready event does not make the main window visible.
- Legacy avatar paths should keep resolving through the existing Wenlan avatar copy bridge.

## Structural Tool Boundary

| Tool | Result |
|---|---|
| CodeGraph | `sync .` returned already up to date; `query ProfileAvatar`, `impact ProfileAvatar`, `query set_main_window_dock_visibility`, and `impact set_main_window_dock_visibility` bounded the avatar and Dock/window surfaces. |
| ast-grep | `outline app/src/search.rs`, `outline app/src/lib.rs`, and `run -p 'invoke($CMD, $$$ARGS)' -l ts src` confirmed the avatar commands, runtime window helper, and frontend call sites. |
| live probes | `/api/profile` still returns a legacy Origin avatar path, while the corresponding file exists under the Wenlan avatar root. |
| `rg`/file reads | Used only after structural probes to inspect concrete avatar, icon, and window-visibility call sites. |

## Findings

The live daemon profile still stores:

```text
/Users/lucian/Library/Application Support/origin/avatars/57515813-4419-4116-bea6-21bc66e1a511.jpg
```

The legacy Origin avatar directory no longer exists, but the same filename exists in:

```text
/Users/lucian/Library/Application Support/wenlan/avatars/
```

Current `resolve_profile_avatar_path` already maps this missing legacy path to the Wenlan copy or suppresses the path when no migrated copy exists, so this checkpoint did not add another avatar-path shim.

The Dock icon issue is lower in the runtime identity path: `pnpm tauri dev` runs a bare debug binary with no bundle identifier. The app already had a full 512px app icon and a 22px tray template icon, but the full app icon was only set via later window-visibility paths. Startup now sets the app icon before activation policy, then performs one delayed reveal pass.

## Red/Green Evidence

RED:

```text
pnpm vitest run src/runtimeIdentity.test.ts
1 failed: expected app/src/lib.rs to contain startup_reveal_fallback_delay
```

GREEN:

```text
pnpm vitest run src/runtimeIdentity.test.ts
8 passed

cargo test --manifest-path app/Cargo.toml --lib startup_reveal_fallback -- --nocapture
1 passed

cargo test --manifest-path app/Cargo.toml --lib dock_icon_uses_full_app_icon_asset_not_tray_template -- --nocapture
1 passed
```

Full checkpoint verification:

```text
pnpm test
49 passed; 419 passed, 1 skipped

pnpm build
passed with existing Vite dynamic-import and large-chunk warnings

cargo test --manifest-path app/Cargo.toml --lib
205 passed

cargo fmt --manifest-path app/Cargo.toml --check
passed

CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 codegraph affected app/src/lib.rs src/runtimeIdentity.test.ts --json
affectedTests: ["src/runtimeIdentity.test.ts"]

pnpm tauri build --no-bundle
Built application at target/release/wenlan-app
```

## Runtime Notes

After this change, the app process and Dock item are still visible as `wenlan-app` in dev mode because Tauri dev runs the unbundled debug binary. The code now sets the app icon from `app/icons/icon.png` immediately at startup and again through the reveal paths. A bundled `.app` launch remains the stronger Dock/bundle evidence because it carries the bundle identifier and `.icns` metadata.
