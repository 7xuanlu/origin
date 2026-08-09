# Wenlan App Runtime Identity Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the desktop app runtime identity from Origin to Wenlan without stranding existing Origin-era app config, data, LaunchAgents, MCP token files, relay IDs, logs, or install paths.

**Architecture:** Use bridge-first migration: prefer new Wenlan paths and labels for new writes, read/import old Origin paths when present, and clean up only owned LaunchAgents after explicit tests prove detection. Keep daemon/API behavior unchanged; this plan only changes app-local runtime identity surfaces and compatibility glue.

**Tech Stack:** Tauri 2, Rust, React 19, TypeScript, Vitest, Cargo, CodeGraph (`npx -y @colbymchenry/codegraph`), ast-grep (`npx -y -p @ast-grep/cli sg`), rust-analyzer/compiler diagnostics, bounded `rg` fallback.

---

## Current Baseline

- Repo: `/Users/lucian/Repos/wenlan-app`
- Local branch after merge: `main`
- Local state: `main...origin/main [ahead 5]`
- Completed before this plan: daemon parity slice for enrichment status, source registry, page links, orphan links, and memory/page revisions.
- Runtime identity remains mixed by design:
  - `app/tauri.conf.json` still has `productName: "Origin"` and `identifier: "com.origin.desktop"`.
  - `app/src/lifecycle.rs` uses `APP_PLIST_LABEL = "com.origin.desktop"`, but already accepts `Wenlan.app` and legacy `Origin.app` paths.
  - `app/src/remote_access.rs` runs `wenlan-mcp`, but token and relay ID paths are still `~/.config/origin-mcp`.
  - `app/src/config.rs` still writes default app config to an `origin` data directory and honors only `ORIGIN_DATA_DIR`.
  - The relay endpoint remains `https://origin-relay.originmemory.workers.dev`; do not rename it until a Wenlan relay and ID migration strategy exist.

## Refreshed Tool Evidence

Run before Task 1 and after every cross-cutting task:

```bash
CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 npx -y @colbymchenry/codegraph sync .
npx -y -p @ast-grep/cli sg outline app/src/lifecycle.rs
npx -y -p @ast-grep/cli sg outline app/src/remote_access.rs
bash scripts/refactor/inventory.sh
```

`scripts/refactor/inventory.sh` rewrites tracked files under `docs/superpowers/refactor/wenlan-app-inventory/`. After running it, either commit those refreshed artifacts with the relevant plan/task or revert them before continuing. Do not leave generated inventory drift unclassified in the worktree.

Observed on merged `main`:

```text
CodeGraph initialized in /Users/lucian/Repos/wenlan-app
Indexed 159 files
2,187 nodes, 5,315 edges in 962ms

frontend invoke calls: 131
registered Tauri commands: 175
origin_types references in Rust app code: 0
runtime identity references: 222
stale taxonomy references: 241
source files under app/src and src: 153
```

Target-specific CodeGraph caveat:

```text
codegraph query remote_access --json
```

found `app/src/remote_access.rs`, `RemoteAccessPanel`, config tests, and Tauri commands. But:

```text
codegraph impact remote_access --json
```

returned only the file node. Treat CodeGraph as orientation here, not impact proof. Use ast-grep outlines, compiler diagnostics, targeted tests, and bounded residual `rg` scans as the completion gates.

## File Structure

- `app/src/identity_paths.rs`: new focused module for Wenlan-vs-Origin app-local path resolution. Owns data dir and MCP config dir decisions; no Tauri state.
- `app/src/config.rs`: use `identity_paths::app_data_dir()` for app-local config; preserve legacy `watch_paths` migration and `Config` schema.
- `app/src/search.rs`: use the same data-dir resolver for avatar storage and any app-local path reads.
- `app/src/lifecycle.rs`: use the same data-dir resolver for the app-owned auto-start opt-out sentinel, plus LaunchAgent identity work in Task 3.
- `app/src/lifecycle.rs`: migrate app LaunchAgent label/log directory to Wenlan while detecting and cleaning old owned app plist state.
- `app/resources/com.wenlan.desktop.plist`: new app LaunchAgent template.
- `app/src/remote_access.rs`: prefer `~/.config/wenlan-mcp`, import only non-empty legacy tokens from `~/.config/origin-mcp`, generate current relay IDs without copying legacy `relay_id`, keep current relay URL until a Wenlan relay exists.
- `app/src/lib.rs`: register any new lifecycle cleanup command only if the UI needs it; otherwise keep cleanup internal to first-run.
- `app/tauri.conf.json`: final product identity cutover after bridge tests are green.
- `package.json`: release DMG names and volume label after product identity cutover.
- `README.md`, `src/components/SetupWizard.tsx`, `src/components/onboarding/*`, `src/lib/agents.ts`: visible copy rename after runtime bridge tests are green.
- `docs/superpowers/refactor/wenlan-app-parity-matrix.md`: update matrix counts and classify remaining intentional legacy Origin references.

## Bridge Safety Policy

- Explicit `WENLAN_DATA_DIR` wins because it is an operator override. If both `WENLAN_DATA_DIR` and `ORIGIN_DATA_DIR` are set, the app must log which root it selected.
- Explicit `ORIGIN_DATA_DIR` remains a supported legacy override when `WENLAN_DATA_DIR` is unset.
- With no env override, the app must not create a fresh empty Wenlan app-data root when the old default Origin app-data root is populated. It must select the old root for this bridge release or explicitly import selected state before returning the new root.
- The old default app-data root is considered populated if any of these exist: `config.json`, `avatars/`, `auto_start_disabled.flag`.
- Existing Wenlan app-data wins over old Origin app-data. Both populated means the user has already crossed the bridge; do not merge automatically.
- `~/.config/origin-mcp/token` may be imported to `~/.config/wenlan-mcp/token` when current token is absent and the legacy token is non-empty.
- `~/.config/origin-mcp/relay_id` must not be blindly copied. Relay IDs are tied to the current legacy relay service and have had stale-route failures. Generate/register a new Wenlan-side relay ID unless a future relay migration proves old IDs are valid.
- Legacy LaunchAgents may be removed only when the plist content proves app ownership: expected legacy label plus an executable path inside `/Applications/Origin.app` or `~/Applications/Origin.app`. Malformed, foreign, or user-edited plists must be preserved.

## Task 1: Add Focused App Identity Path Resolver

**Files:**
- Create: `app/src/identity_paths.rs`
- Modify: `app/src/lib.rs`
- Modify: `app/src/config.rs`
- Modify: `app/src/search.rs`
- Modify: `app/src/lifecycle.rs`

- [ ] **Step 1: Write failing resolver tests**

Create `app/src/identity_paths.rs` with only tests and helper scaffolding first:

```rust
// SPDX-License-Identifier: AGPL-3.0-only
use std::path::PathBuf;

pub fn app_data_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("origin")
}

pub fn legacy_app_data_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("origin")
}

fn path_has_app_state(path: &std::path::Path) -> bool {
    path.join("config.json").exists()
        || path.join("avatars").exists()
        || path.join("auto_start_disabled.flag").exists()
}

pub fn mcp_config_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".config")
        .join("origin-mcp")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    struct EnvGuard {
        home: Option<std::ffi::OsString>,
        wenlan: Option<std::ffi::OsString>,
        origin: Option<std::ffi::OsString>,
    }

    impl EnvGuard {
        fn capture() -> Self {
            Self {
                home: std::env::var_os("HOME"),
                wenlan: std::env::var_os("WENLAN_DATA_DIR"),
                origin: std::env::var_os("ORIGIN_DATA_DIR"),
            }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.home {
                Some(value) => std::env::set_var("HOME", value),
                None => std::env::remove_var("HOME"),
            }
            match &self.wenlan {
                Some(value) => std::env::set_var("WENLAN_DATA_DIR", value),
                None => std::env::remove_var("WENLAN_DATA_DIR"),
            }
            match &self.origin {
                Some(value) => std::env::set_var("ORIGIN_DATA_DIR", value),
                None => std::env::remove_var("ORIGIN_DATA_DIR"),
            }
        }
    }

    #[test]
    fn app_data_dir_prefers_wenlan_env() {
        let _guard = env_lock();
        let _env = EnvGuard::capture();
        std::env::set_var("WENLAN_DATA_DIR", "/tmp/wenlan-app-test");
        std::env::set_var("ORIGIN_DATA_DIR", "/tmp/origin-app-test");

        assert_eq!(app_data_dir(), PathBuf::from("/tmp/wenlan-app-test"));
    }

    #[test]
    fn app_data_dir_falls_back_to_origin_env() {
        let _guard = env_lock();
        let _env = EnvGuard::capture();
        std::env::remove_var("WENLAN_DATA_DIR");
        std::env::set_var("ORIGIN_DATA_DIR", "/tmp/origin-app-test");

        assert_eq!(app_data_dir(), PathBuf::from("/tmp/origin-app-test"));
    }

    #[test]
    fn app_data_dir_uses_legacy_default_when_current_absent_and_legacy_has_config() {
        let _guard = env_lock();
        let _env = EnvGuard::capture();
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        std::env::remove_var("WENLAN_DATA_DIR");
        std::env::remove_var("ORIGIN_DATA_DIR");
        let legacy = dirs::data_local_dir().unwrap().join("origin");
        std::fs::create_dir_all(&legacy).unwrap();
        std::fs::write(legacy.join("config.json"), "{}").unwrap();

        assert_eq!(app_data_dir(), legacy);
    }

    #[test]
    fn app_data_dir_uses_legacy_default_when_current_empty_and_legacy_has_config() {
        let _guard = env_lock();
        let _env = EnvGuard::capture();
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        std::env::remove_var("WENLAN_DATA_DIR");
        std::env::remove_var("ORIGIN_DATA_DIR");
        let current = dirs::data_local_dir().unwrap().join("wenlan");
        let legacy = dirs::data_local_dir().unwrap().join("origin");
        std::fs::create_dir_all(&current).unwrap();
        std::fs::create_dir_all(&legacy).unwrap();
        std::fs::write(legacy.join("config.json"), "{}").unwrap();

        assert_eq!(app_data_dir(), legacy);
    }

    #[test]
    fn app_data_dir_uses_wenlan_default_when_current_has_app_state() {
        let _guard = env_lock();
        let _env = EnvGuard::capture();
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        std::env::remove_var("WENLAN_DATA_DIR");
        std::env::remove_var("ORIGIN_DATA_DIR");
        let current = dirs::data_local_dir().unwrap().join("wenlan");
        let legacy = dirs::data_local_dir().unwrap().join("origin");
        std::fs::create_dir_all(&current).unwrap();
        std::fs::write(current.join("config.json"), "{}").unwrap();
        std::fs::create_dir_all(&legacy).unwrap();
        std::fs::write(legacy.join("config.json"), "{}").unwrap();

        assert_eq!(app_data_dir(), current);
    }

    #[test]
    fn app_data_dir_uses_wenlan_default_when_neither_exists() {
        let _guard = env_lock();
        let _env = EnvGuard::capture();
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        std::env::remove_var("WENLAN_DATA_DIR");
        std::env::remove_var("ORIGIN_DATA_DIR");

        assert_eq!(app_data_dir(), dirs::data_local_dir().unwrap().join("wenlan"));
    }
}
```

Expose the module in `app/src/lib.rs`:

```rust
mod identity_paths;
```

Run:

```bash
cargo test -p origin-app --lib identity_paths -- --nocapture
```

Expected: FAIL because `app_data_dir()` still ignores `WENLAN_DATA_DIR` and cannot select the Wenlan default when no legacy state exists.

- [ ] **Step 2: Add failing config/avatar path tests**

In `app/src/config.rs`, add helper and tests so `config_path()` prefers `WENLAN_DATA_DIR` and falls back to `ORIGIN_DATA_DIR`:

```rust
struct EnvGuard {
    wenlan: Option<std::ffi::OsString>,
    origin: Option<std::ffi::OsString>,
}

impl EnvGuard {
    fn capture() -> Self {
        Self {
            wenlan: std::env::var_os("WENLAN_DATA_DIR"),
            origin: std::env::var_os("ORIGIN_DATA_DIR"),
        }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        match &self.wenlan {
            Some(value) => std::env::set_var("WENLAN_DATA_DIR", value),
            None => std::env::remove_var("WENLAN_DATA_DIR"),
        }
        match &self.origin {
            Some(value) => std::env::set_var("ORIGIN_DATA_DIR", value),
            None => std::env::remove_var("ORIGIN_DATA_DIR"),
        }
    }
}

#[test]
#[serial_test::serial]
fn config_path_prefers_wenlan_data_dir() {
    let _env = EnvGuard::capture();
    std::env::set_var("WENLAN_DATA_DIR", "/tmp/wenlan-config-test");
    std::env::set_var("ORIGIN_DATA_DIR", "/tmp/origin-config-test");

    assert_eq!(config_path(), PathBuf::from("/tmp/wenlan-config-test/config.json"));
}

#[test]
#[serial_test::serial]
fn config_path_falls_back_to_origin_data_dir() {
    let _env = EnvGuard::capture();
    std::env::remove_var("WENLAN_DATA_DIR");
    std::env::set_var("ORIGIN_DATA_DIR", "/tmp/origin-config-test");

    assert_eq!(config_path(), PathBuf::from("/tmp/origin-config-test/config.json"));
}
```

In `app/src/search.rs` avatar path tests, keep the existing legacy fallback test and add:

```rust
#[test]
#[serial_test::serial]
fn avatar_storage_dir_prefers_wenlan_data_dir_when_both_are_set() {
    let old_wenlan = std::env::var_os("WENLAN_DATA_DIR");
    let old_origin = std::env::var_os("ORIGIN_DATA_DIR");
    std::env::set_var("WENLAN_DATA_DIR", "/tmp/wenlan-avatar-test");
    std::env::set_var("ORIGIN_DATA_DIR", "/tmp/origin-avatar-test");

    assert_eq!(
        avatar_storage_dir(),
        PathBuf::from("/tmp/wenlan-avatar-test/avatars")
    );

    restore_env("WENLAN_DATA_DIR", old_wenlan);
    restore_env("ORIGIN_DATA_DIR", old_origin);
}
```

In `app/src/lifecycle.rs`, add:

```rust
#[test]
#[serial_test::serial]
fn opt_out_prefers_wenlan_data_dir() {
    struct EnvGuard {
        wenlan: Option<std::ffi::OsString>,
        origin: Option<std::ffi::OsString>,
    }
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.wenlan {
                Some(value) => std::env::set_var("WENLAN_DATA_DIR", value),
                None => std::env::remove_var("WENLAN_DATA_DIR"),
            }
            match &self.origin {
                Some(value) => std::env::set_var("ORIGIN_DATA_DIR", value),
                None => std::env::remove_var("ORIGIN_DATA_DIR"),
            }
        }
    }
    let _env = EnvGuard {
        wenlan: std::env::var_os("WENLAN_DATA_DIR"),
        origin: std::env::var_os("ORIGIN_DATA_DIR"),
    };
    let tmp = tempfile::tempdir().unwrap();
    let legacy = tempfile::tempdir().unwrap();
    std::env::set_var("WENLAN_DATA_DIR", tmp.path());
    std::env::set_var("ORIGIN_DATA_DIR", legacy.path());

    assert!(!user_opted_out());
    set_user_opted_out(true).unwrap();
    assert!(tmp.path().join("auto_start_disabled.flag").exists());
    assert!(!legacy.path().join("auto_start_disabled.flag").exists());
}
```

Run:

```bash
cargo test -p origin-app --lib identity_paths -- --nocapture
cargo test -p origin-app --lib config::tests -- --nocapture
cargo test -p origin-app --lib avatar_path_tests -- --nocapture
cargo test -p origin-app --lib lifecycle::tests::opt_out_prefers_wenlan_data_dir -- --nocapture
```

Expected: FAIL until `config.rs` and `search.rs` use `identity_paths::app_data_dir()`.

- [ ] **Step 3: Implement the resolver**

Replace `app_data_dir()` in `app/src/identity_paths.rs`:

```rust
pub fn app_data_dir() -> PathBuf {
    if let Some(custom) = std::env::var_os("WENLAN_DATA_DIR") {
        log::info!("[identity] using WENLAN_DATA_DIR for app data");
        return PathBuf::from(custom);
    }
    if let Some(custom) = std::env::var_os("ORIGIN_DATA_DIR") {
        log::info!("[identity] using legacy ORIGIN_DATA_DIR for app data");
        return PathBuf::from(custom);
    }
    let current = dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("wenlan");
    let legacy = legacy_app_data_dir();
    if path_has_app_state(&current) {
        return current;
    }
    if path_has_app_state(&legacy) {
        log::warn!(
            "[identity] using populated legacy Origin app data root for bridge release: {}",
            legacy.display()
        );
        return legacy;
    }
    current
}
```

Update `app/src/config.rs` `config_path()`:

```rust
fn config_path() -> PathBuf {
    crate::identity_paths::app_data_dir().join("config.json")
}
```

Update `app/src/search.rs` `avatar_storage_dir()` to use:

```rust
fn avatar_storage_dir() -> PathBuf {
    crate::identity_paths::app_data_dir().join("avatars")
}
```

Update `app/src/lifecycle.rs` `data_dir()` to use:

```rust
fn data_dir() -> Result<PathBuf> {
    Ok(crate::identity_paths::app_data_dir())
}
```

Run:

```bash
cargo test -p origin-app --lib identity_paths -- --nocapture
cargo test -p origin-app --lib config::tests -- --nocapture
cargo test -p origin-app --lib avatar_path_tests -- --nocapture
cargo test -p origin-app --lib lifecycle::tests::opt_out_prefers_wenlan_data_dir -- --nocapture
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add app/src/identity_paths.rs app/src/lib.rs app/src/config.rs app/src/search.rs app/src/lifecycle.rs
git commit -m "fix: bridge app data paths to wenlan"
```

## Task 2: Bridge Remote MCP Token and Relay ID Paths

**Files:**
- Modify: `app/src/identity_paths.rs`
- Modify: `app/src/remote_access.rs`

- [ ] **Step 1: Write failing MCP path tests**

Extend `app/src/identity_paths.rs`:

```rust
pub fn legacy_mcp_config_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".config")
        .join("origin-mcp")
}

#[cfg(test)]
mod mcp_tests {
    use super::*;

    #[test]
    fn mcp_config_dir_uses_wenlan_mcp() {
        let dir = mcp_config_dir();
        assert!(dir.ends_with(".config/wenlan-mcp"));
    }

    #[test]
    fn legacy_mcp_config_dir_uses_origin_mcp() {
        let dir = legacy_mcp_config_dir();
        assert!(dir.ends_with(".config/origin-mcp"));
    }
}
```

Run:

```bash
cargo test -p origin-app --lib identity_paths::mcp_tests -- --nocapture
```

Expected: FAIL because `mcp_config_dir()` still returns `origin-mcp`.

- [ ] **Step 2: Write failing remote-access import tests**

In `app/src/remote_access.rs`, add pure path helper tests without touching the real home directory:

```rust
#[test]
fn token_path_imports_nonempty_legacy_token_when_current_missing() {
    let temp = tempfile::tempdir().unwrap();
    let current = temp.path().join("wenlan-mcp");
    let legacy = temp.path().join("origin-mcp");
    std::fs::create_dir_all(&legacy).unwrap();
    std::fs::write(legacy.join("token"), "legacy-token\n").unwrap();

    let path = token_file_path_for_dirs(&current, &legacy).unwrap();

    assert_eq!(path, current.join("token"));
    assert_eq!(std::fs::read_to_string(path).unwrap(), "legacy-token\n");
}

#[test]
fn token_path_keeps_current_token_when_present() {
    let temp = tempfile::tempdir().unwrap();
    let current = temp.path().join("wenlan-mcp");
    let legacy = temp.path().join("origin-mcp");
    std::fs::create_dir_all(&current).unwrap();
    std::fs::create_dir_all(&legacy).unwrap();
    std::fs::write(current.join("token"), "current-token\n").unwrap();
    std::fs::write(legacy.join("token"), "legacy-token\n").unwrap();

    let path = token_file_path_for_dirs(&current, &legacy).unwrap();

    assert_eq!(path, current.join("token"));
    assert_eq!(std::fs::read_to_string(path).unwrap(), "current-token\n");
}

#[test]
fn token_path_does_not_import_empty_legacy_token() {
    let temp = tempfile::tempdir().unwrap();
    let current = temp.path().join("wenlan-mcp");
    let legacy = temp.path().join("origin-mcp");
    std::fs::create_dir_all(&legacy).unwrap();
    std::fs::write(legacy.join("token"), " \n").unwrap();

    let path = token_file_path_for_dirs(&current, &legacy).unwrap();

    assert_eq!(path, current.join("token"));
    assert!(!path.exists(), "empty legacy token must not be copied");
}

#[test]
fn relay_id_path_does_not_import_legacy_relay_id() {
    let temp = tempfile::tempdir().unwrap();
    let current = temp.path().join("wenlan-mcp");
    let legacy = temp.path().join("origin-mcp");
    std::fs::create_dir_all(&legacy).unwrap();
    std::fs::write(legacy.join("relay_id"), "stale-relay-id").unwrap();

    let path = relay_id_path_for_dirs(&current);

    assert_eq!(path, current.join("relay_id"));
    assert!(!path.exists(), "legacy relay ID must not be copied blindly");
}
```

Add placeholder helpers above the tests:

```rust
fn token_file_path_for_dirs(
    current_dir: &std::path::Path,
    legacy_dir: &std::path::Path,
) -> Result<std::path::PathBuf, String> {
    let _ = current_dir;
    Ok(legacy_dir.join("token"))
}

fn relay_id_path_for_dirs(current_dir: &std::path::Path) -> std::path::PathBuf {
    current_dir.join("relay_id")
}
```

Run:

```bash
cargo test -p origin-app --lib remote_access::tests::token_path_imports_nonempty_legacy_token_when_current_missing -- --nocapture
cargo test -p origin-app --lib remote_access::tests::relay_id_path_does_not_import_legacy_relay_id -- --nocapture
```

Expected: FAIL because the token helper still returns the legacy path and does not copy the token into the current Wenlan MCP dir.

- [ ] **Step 3: Implement current/legacy MCP path helpers**

In `app/src/identity_paths.rs`:

```rust
pub fn mcp_config_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".config")
        .join("wenlan-mcp")
}
```

In `app/src/remote_access.rs`, replace direct `origin-mcp` joins with helpers:

```rust
fn import_nonempty_legacy_file(
    current_dir: &std::path::Path,
    legacy_dir: &std::path::Path,
    file_name: &str,
) -> Result<std::path::PathBuf, String> {
    let current = current_dir.join(file_name);
    if current.exists() {
        return Ok(current);
    }
    let legacy = legacy_dir.join(file_name);
    if legacy.exists() {
        let contents = std::fs::read_to_string(&legacy)
            .map_err(|e| format!("Failed to read legacy {} at {}: {}", file_name, legacy.display(), e))?;
        if contents.trim().is_empty() {
            return Ok(current);
        }
        if let Some(parent) = current.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create {}: {}", parent.display(), e))?;
        }
        std::fs::write(&current, contents)
            .map_err(|e| format!("Failed to import legacy {} to {}: {}", file_name, current.display(), e))?;
    }
    Ok(current)
}

fn token_file_path_for_dirs(
    current_dir: &std::path::Path,
    legacy_dir: &std::path::Path,
) -> Result<std::path::PathBuf, String> {
    import_nonempty_legacy_file(current_dir, legacy_dir, "token")
}

fn relay_id_path_for_dirs(current_dir: &std::path::Path) -> std::path::PathBuf {
    current_dir.join("relay_id")
}

fn token_file_path() -> Result<std::path::PathBuf, String> {
    token_file_path_for_dirs(
        &crate::identity_paths::mcp_config_dir(),
        &crate::identity_paths::legacy_mcp_config_dir(),
    )
}

fn relay_id_path() -> std::path::PathBuf {
    relay_id_path_for_dirs(&crate::identity_paths::mcp_config_dir())
}
```

Update `get_or_create_relay_id()` to use `relay_id_path()` instead of rebuilding `~/.config/origin-mcp/relay_id`. Do not consult or copy the legacy relay ID. Update every `token_file_path()` call site to handle `Result<PathBuf, String>`:

```rust
pub fn read_token() -> Result<String, String> {
    let path = token_file_path()?;
    std::fs::read_to_string(&path)
        .map(|s| s.trim().to_string())
        .map_err(|e| format!("Failed to read token at {}: {}", path.display(), e))
}
```

For token generation call sites, use:

```rust
let token_path = token_file_path()?;
```

Update the existing `test_token_generate_args_include_legacy_output_path` test to the current Wenlan path and rename it:

```rust
#[test]
#[serial_test::serial]
fn test_token_generate_args_include_wenlan_output_path() {
    let tmp = tempfile::tempdir().unwrap();
    std::env::set_var("HOME", tmp.path());
    let path = token_file_path().unwrap();
    let expected_path = tmp.path().join(".config/wenlan-mcp/token");

    assert_eq!(
        token_generate_args(&path),
        vec![
            "token".to_string(),
            "generate".to_string(),
            "--output".to_string(),
            expected_path.to_string_lossy().into_owned()
        ]
    );
}
```

Run:

```bash
cargo test -p origin-app --lib identity_paths -- --nocapture
cargo test -p origin-app --lib remote_access::tests -- --nocapture
```

Expected: PASS.

- [ ] **Step 4: Keep relay URL intentionally legacy**

Add this comment above `RELAY_URL` in `app/src/remote_access.rs`:

```rust
// Intentionally still the legacy Origin relay. Do not rename this constant
// until a Wenlan relay endpoint exists and existing relay IDs have a migration
// strategy.
const RELAY_URL: &str = "https://origin-relay.originmemory.workers.dev";
```

Run:

```bash
rg -n "origin-relay|originmemory|origin-mcp|wenlan-mcp" app/src/remote_access.rs
cargo test -p origin-app --lib remote_access::tests -- --nocapture
```

Expected: only the relay URL/comment and legacy fallback helper mention `origin-*`.

- [ ] **Step 5: Commit**

```bash
git add app/src/identity_paths.rs app/src/remote_access.rs
git commit -m "fix: bridge remote mcp identity paths"
```

## Task 3: Cut Over App LaunchAgent Identity with Legacy Cleanup

**Files:**
- Modify: `app/Cargo.toml`
- Modify: `app/src/lifecycle.rs`
- Create: `app/resources/com.wenlan.desktop.plist`
- Keep until cleanup: `app/resources/com.origin.desktop.plist`

- [ ] **Step 1: Write failing LaunchAgent label tests**

In `app/src/lifecycle.rs`, update tests or add:

```rust
#[test]
fn app_plist_path_uses_wenlan_label() {
    let path = app_plist_path().unwrap();
    assert!(path.ends_with("Library/LaunchAgents/com.wenlan.desktop.plist"));
}

#[test]
fn legacy_app_plist_path_uses_origin_label() {
    let path = legacy_app_plist_path().unwrap();
    assert!(path.ends_with("Library/LaunchAgents/com.origin.desktop.plist"));
}
```

Add placeholder function near `legacy_server_plist_path()`:

```rust
pub fn legacy_app_plist_path() -> Result<PathBuf> {
    app_plist_path()
}
```

Run:

```bash
cargo test -p origin-app --lib lifecycle::tests::app_plist_path_uses_wenlan_label -- --nocapture
cargo test -p origin-app --lib lifecycle::tests::legacy_app_plist_path_uses_origin_label -- --nocapture
```

Expected: FAIL because `APP_PLIST_LABEL` is still `com.origin.desktop`.

- [ ] **Step 2: Add legacy app ownership and cleanup tests**

In lifecycle tests, add:

```rust
use std::path::Path;

#[test]
#[serial_test::serial]
fn legacy_app_plist_ownership_accepts_owned_origin_app_path() {
    let content = r#"
    <plist><dict>
      <key>Label</key><string>com.origin.desktop</string>
      <key>ProgramArguments</key>
      <array><string>/Applications/Origin.app/Contents/MacOS/origin</string></array>
    </dict></plist>
    "#;

    assert!(legacy_app_plist_is_owned(content));
}

#[test]
#[serial_test::serial]
fn legacy_app_plist_ownership_rejects_foreign_path() {
    let content = r#"
    <plist><dict>
      <key>Label</key><string>com.origin.desktop</string>
      <key>ProgramArguments</key>
      <array><string>/tmp/Other.app/Contents/MacOS/origin</string></array>
    </dict></plist>
    "#;

    assert!(!legacy_app_plist_is_owned(content));
}

#[test]
#[serial_test::serial]
fn legacy_server_plist_ownership_accepts_owned_origin_server_path() {
    let content = r#"
    <plist><dict>
      <key>Label</key><string>com.origin.server</string>
      <key>ProgramArguments</key>
      <array><string>/Applications/Origin.app/Contents/MacOS/origin-server</string></array>
    </dict></plist>
    "#;

    assert!(legacy_server_plist_is_owned(content));
}

#[test]
#[serial_test::serial]
fn legacy_server_plist_ownership_rejects_foreign_path() {
    let content = r#"
    <plist><dict>
      <key>Label</key><string>com.origin.server</string>
      <key>ProgramArguments</key>
      <array><string>/tmp/origin-server</string></array>
    </dict></plist>
    "#;

    assert!(!legacy_server_plist_is_owned(content));
}

#[test]
#[serial_test::serial]
fn cleanup_legacy_app_plist_unloads_and_removes_owned_file() {
    let temp = tempfile::tempdir().unwrap();
    std::env::set_var("HOME", temp.path());
    let plist = legacy_app_plist_path().unwrap();
    std::fs::create_dir_all(plist.parent().unwrap()).unwrap();
    std::fs::write(
        &plist,
        r#"
        <plist><dict>
          <key>Label</key><string>com.origin.desktop</string>
          <key>ProgramArguments</key>
          <array><string>/Applications/Origin.app/Contents/MacOS/origin</string></array>
        </dict></plist>
        "#,
    )
    .unwrap();
    let mock = MockLaunchctl::default();

    cleanup_legacy_app_plist(&mock).unwrap();

    assert!(!plist.exists(), "legacy app plist removed");
    let calls = mock.calls.lock().unwrap();
    assert!(
        calls
            .iter()
            .any(|c| c[0] == "unload" && c[1] == plist.to_string_lossy()),
        "legacy app plist unloaded before removal"
    );
}

#[test]
#[serial_test::serial]
fn cleanup_legacy_app_plist_preserves_foreign_file() {
    let temp = tempfile::tempdir().unwrap();
    std::env::set_var("HOME", temp.path());
    let plist = legacy_app_plist_path().unwrap();
    std::fs::create_dir_all(plist.parent().unwrap()).unwrap();
    std::fs::write(
        &plist,
        r#"
        <plist><dict>
          <key>Label</key><string>com.origin.desktop</string>
          <key>ProgramArguments</key>
          <array><string>/tmp/Other.app/Contents/MacOS/origin</string></array>
        </dict></plist>
        "#,
    )
    .unwrap();
    let mock = MockLaunchctl::default();

    cleanup_legacy_app_plist(&mock).unwrap();

    assert!(plist.exists(), "foreign legacy plist must be preserved");
    assert!(mock.calls.lock().unwrap().is_empty());
}

#[test]
#[serial_test::serial]
fn cleanup_legacy_server_plist_preserves_foreign_file() {
    let temp = tempfile::tempdir().unwrap();
    std::env::set_var("HOME", temp.path());
    let plist = legacy_server_plist_path().unwrap();
    std::fs::create_dir_all(plist.parent().unwrap()).unwrap();
    std::fs::write(
        &plist,
        r#"
        <plist><dict>
          <key>Label</key><string>com.origin.server</string>
          <key>ProgramArguments</key>
          <array><string>/tmp/origin-server</string></array>
        </dict></plist>
        "#,
    )
    .unwrap();
    let mock = MockLaunchctl::default();

    cleanup_legacy_server_plist(&mock).unwrap();

    assert!(plist.exists(), "foreign legacy server plist must be preserved");
    assert!(mock.calls.lock().unwrap().is_empty());
}
```

Run:

```bash
cargo test -p origin-app --lib lifecycle::tests::legacy_app_plist_ownership_accepts_owned_origin_app_path -- --nocapture
cargo test -p origin-app --lib lifecycle::tests::cleanup_legacy_app_plist_preserves_foreign_file -- --nocapture
cargo test -p origin-app --lib lifecycle::tests::legacy_server_plist_ownership_accepts_owned_origin_server_path -- --nocapture
cargo test -p origin-app --lib lifecycle::tests::cleanup_legacy_server_plist_preserves_foreign_file -- --nocapture
```

Expected: FAIL because `legacy_app_plist_is_owned`, `legacy_server_plist_is_owned`, and `cleanup_legacy_app_plist` do not exist and server cleanup is still unconditional.

- [ ] **Step 3: Implement new current and legacy labels**

In `app/Cargo.toml`, add a plist parser instead of ad hoc XML matching:

```toml
plist = "1"
```

In `app/src/lifecycle.rs`:

```rust
pub const APP_PLIST_LABEL: &str = "com.wenlan.desktop";
pub const LEGACY_APP_PLIST_LABEL: &str = "com.origin.desktop";

const APP_PLIST_TEMPLATE: &str = include_str!("../resources/com.wenlan.desktop.plist");
```

Add:

```rust
fn plist_string(content: &str, key: &str) -> Option<String> {
    let value = plist::Value::from_reader_xml(content.as_bytes()).ok()?;
    let dict = value.as_dictionary()?;
    dict.get(key)?.as_string().map(str::to_owned)
}

fn plist_first_program(content: &str) -> Option<String> {
    let value = plist::Value::from_reader_xml(content.as_bytes()).ok()?;
    let dict = value.as_dictionary()?;
    if let Some(program) = dict.get("Program").and_then(|v| v.as_string()) {
        return Some(program.to_owned());
    }
    dict.get("ProgramArguments")?
        .as_array()?
        .first()?
        .as_string()
        .map(str::to_owned)
}

fn path_is_legacy_origin_app_exe(path: &str) -> bool {
    path == "/Applications/Origin.app/Contents/MacOS/origin"
        || path == "/Applications/Origin.app/Contents/MacOS/origin-app"
        || dirs::home_dir()
            .map(|home| {
                let origin = home.join("Applications/Origin.app/Contents/MacOS/origin");
                let origin_app = home.join("Applications/Origin.app/Contents/MacOS/origin-app");
                path == origin.to_string_lossy().as_ref()
                    || path == origin_app.to_string_lossy().as_ref()
            })
            .unwrap_or(false)
}

fn path_is_legacy_origin_server_exe(path: &str) -> bool {
    path == "/Applications/Origin.app/Contents/MacOS/origin-server"
        || dirs::home_dir()
            .map(|home| {
                let server = home.join("Applications/Origin.app/Contents/MacOS/origin-server");
                path == server.to_string_lossy().as_ref()
            })
            .unwrap_or(false)
}

fn legacy_app_plist_is_owned(content: &str) -> bool {
    plist_string(content, "Label").as_deref() == Some(LEGACY_APP_PLIST_LABEL)
        && plist_first_program(content)
            .as_deref()
            .is_some_and(path_is_legacy_origin_app_exe)
}

fn legacy_server_plist_is_owned(content: &str) -> bool {
    plist_string(content, "Label").as_deref() == Some(LEGACY_SERVER_PLIST_LABEL)
        && plist_first_program(content)
            .as_deref()
            .is_some_and(path_is_legacy_origin_server_exe)
}

pub fn legacy_app_plist_path() -> Result<PathBuf> {
    Ok(home_dir()?
        .join("Library/LaunchAgents")
        .join(format!("{}.plist", LEGACY_APP_PLIST_LABEL)))
}

pub fn cleanup_legacy_app_plist(launchctl: &dyn LaunchctlExec) -> Result<()> {
    let plist = legacy_app_plist_path()?;
    if !plist.exists() {
        return Ok(());
    }
    let content = std::fs::read_to_string(&plist)?;
    if !legacy_app_plist_is_owned(&content) {
        log::warn!(
            "[first-run] preserving legacy app plist that is not clearly owned by Wenlan/Origin: {}",
            plist.display()
        );
        return Ok(());
    }
    let unload = launchctl.run(&["unload", &plist.to_string_lossy()])?;
    if !unload.status.success() {
        anyhow::bail!(
            "launchctl unload legacy app plist failed: {}",
            String::from_utf8_lossy(&unload.stderr)
        );
    }
    std::fs::remove_file(&plist)?;
    Ok(())
}

pub fn cleanup_legacy_server_plist(launchctl: &dyn LaunchctlExec) -> Result<()> {
    let plist = legacy_server_plist_path()?;
    if !plist.exists() {
        return Ok(());
    }
    let content = std::fs::read_to_string(&plist)?;
    if !legacy_server_plist_is_owned(&content) {
        log::warn!(
            "[first-run] preserving legacy server plist that is not clearly owned by Wenlan/Origin: {}",
            plist.display()
        );
        return Ok(());
    }
    let unload = launchctl.run(&["unload", &plist.to_string_lossy()])?;
    if !unload.status.success() {
        anyhow::bail!(
            "launchctl unload legacy server plist failed: {}",
            String::from_utf8_lossy(&unload.stderr)
        );
    }
    std::fs::remove_file(&plist)?;
    Ok(())
}
```

Create `app/resources/com.wenlan.desktop.plist` by copying the legacy template and replacing only the label:

```xml
<key>Label</key>
<string>com.wenlan.desktop</string>
```

Keep `__ORIGIN_APP_PATH__` placeholder unchanged only if changing it would require broader template code churn. Rename the placeholder in a later cleanup task if it becomes the last confusing token in the file.

Remove the previous unconditional `cleanup_legacy_server_plist` implementation when adding the ownership-gated version.

Run:

```bash
cargo test -p origin-app --lib lifecycle::tests -- --nocapture
```

Expected: PASS.

- [ ] **Step 4: Wire cleanup into first-run path**

In `first_run_install_if_needed`, clean up owned legacy app/server plists before the opt-out return and before installing the new app plist:

```rust
cleanup_legacy_app_plist(launchctl)?;
cleanup_legacy_server_plist(launchctl)?;
if user_opted_out() {
    return Ok(());
}
```

Do not leave an earlier `if user_opted_out() { return Ok(()); }` above these cleanup calls; otherwise an opted-out user can keep an old `com.origin.desktop` LaunchAgent running forever.

In `set_run_at_login(false, ...)`, call both ownership-gated legacy cleanup helpers after uninstalling the current app plist and before returning:

```rust
cleanup_legacy_app_plist(launchctl)?;
cleanup_legacy_server_plist(launchctl)?;
```

In `quit_origin`, log but attempt both ownership-gated legacy cleanup helpers:

```rust
if let Err(e) = cleanup_legacy_app_plist(&launchctl) {
    log::warn!("[quit] cleanup_legacy_app_plist failed: {e}");
}
if let Err(e) = cleanup_legacy_server_plist(&launchctl) {
    log::warn!("[quit] cleanup_legacy_server_plist failed: {e}");
}
```

Run:

```bash
cargo test -p origin-app --lib lifecycle::tests -- --nocapture
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/Cargo.toml app/src/lifecycle.rs app/resources/com.wenlan.desktop.plist
git commit -m "fix: migrate app launch agent identity"
```

## Task 4: Product Bundle and Release Artifact Identity Cutover

**Files:**
- Modify: `app/tauri.conf.json`
- Modify: `app/Cargo.toml`
- Modify: `app/src/main.rs`
- Modify: `app/src/lib.rs`
- Modify: `.github/workflows/ci.yml`
- Modify: `package.json`
- Modify: `README.md`
- Modify: relevant tests under `src/components` and `src/lib`

- [ ] **Step 1: Inventory Tauri-derived path surfaces**

Run:

```bash
CODEGRAPH_TELEMETRY=0 DO_NOT_TRACK=1 npx -y @colbymchenry/codegraph query path --json
npx -y -p @ast-grep/cli sg run -p 'dirs::$FN($$$ARGS)' -l rs app/src
npx -y -p @ast-grep/cli sg run -p '$APP.path().$METHOD($$$ARGS)' -l rs app/src
rg -n "app_config_dir|app_data_dir|app_local_data_dir|data_dir\\(|data_local_dir|config_dir|path\\(\\)\\." app/src src app/tauri.conf.json
```

Expected: every app-local path that can be affected by product name or bundle identifier is either routed through `identity_paths`, explicitly daemon-owned, or listed as an intentional legacy bridge in `docs/superpowers/refactor/wenlan-app-parity-matrix.md`. Do not edit `app/tauri.conf.json` until this inventory is reviewed.

- [ ] **Step 2: Write failing identity assertions**

Create `src/runtimeIdentity.test.ts` with assertions that read `app/tauri.conf.json` and `package.json`:

```ts
import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

describe("runtime product identity", () => {
  it("uses Wenlan app product identity", () => {
    const tauri = JSON.parse(
      fs.readFileSync(path.join(process.cwd(), "app/tauri.conf.json"), "utf8")
    );

    expect(tauri.productName).toBe("Wenlan");
    expect(tauri.identifier).toBe("com.wenlan.desktop");
    expect(tauri.app.windows[0].title).toBe("Wenlan");
  });

  it("uses Wenlan release artifact names", () => {
    const pkg = JSON.parse(fs.readFileSync(path.join(process.cwd(), "package.json"), "utf8"));
    expect(pkg.name).toBe("wenlan-app");
    expect(pkg.repository.url).toBe("https://github.com/7xuanlu/wenlan-app");
    expect(pkg.scripts["release:dmg"]).toContain("Wenlan_");
    expect(pkg.scripts["release:dmg"]).toContain("-volname Wenlan");
    expect(pkg.scripts["clean:release"]).toContain("Wenlan_*.dmg");
  });

  it("uses Wenlan Rust package and executable identity", () => {
    const cargo = fs.readFileSync(path.join(process.cwd(), "app/Cargo.toml"), "utf8");
    expect(cargo).toContain('name = "wenlan-app"');
    expect(cargo).toContain('default-run = "wenlan-app"');
    expect(cargo).toContain('name = "wenlan_lib"');
  });

  it("uses Wenlan updater endpoint", () => {
    const tauri = JSON.parse(
      fs.readFileSync(path.join(process.cwd(), "app/tauri.conf.json"), "utf8")
    );
    expect(tauri.plugins.updater.endpoints[0]).toBe(
      "https://github.com/7xuanlu/wenlan-app/releases/latest/download/latest.json"
    );
  });
});
```

Run:

```bash
pnpm vitest run src/runtimeIdentity.test.ts
```

Expected: FAIL while product identity still says Origin.

- [ ] **Step 3: Update bundle identity**

Edit `app/tauri.conf.json`:

```json
{
  "productName": "Wenlan",
  "version": "0.3.1",
  "identifier": "com.wenlan.desktop",
  "app": {
    "windows": [
      {
        "title": "Wenlan"
      }
    ]
  },
  "plugins": {
    "updater": {
      "endpoints": [
        "https://github.com/7xuanlu/wenlan-app/releases/latest/download/latest.json"
      ]
    }
  }
}
```

Preserve the existing version (`0.3.1` at plan time) and all unrelated existing config keys exactly.

Edit `app/Cargo.toml`:

```toml
[package]
name = "wenlan-app"
default-run = "wenlan-app"

[lib]
name = "wenlan_lib"
```

Edit `app/src/main.rs`:

```rust
fn main() {
    wenlan_lib::run()
}
```

Edit `.github/workflows/ci.yml` Rust package filters from:

```yaml
cargo test -p origin-app --lib
```

to:

```yaml
cargo test -p wenlan-app --lib
```

Edit `package.json` release scripts:

```json
"name": "wenlan-app",
"repository": {
  "url": "https://github.com/7xuanlu/wenlan-app"
},
"release:dmg": "rm -f target/release/bundle/macos/rw.*.dmg && mkdir -p target/release/bundle/dmg && node -e \"const v=require('./package.json').version; const f='target/release/bundle/dmg/Wenlan_'+v+'_aarch64.dmg'; require('fs').rmSync(f,{force:true}); const {execSync}=require('child_process'); execSync('hdiutil create -volname Wenlan -srcfolder target/release/bundle/macos -ov -format UDZO '+f,{stdio:'inherit'}); console.log('Created',f);\"",
"clean:release": "rm -f target/release/bundle/macos/rw.*.dmg; rm -rf target/release/bundle/dmg/Wenlan_*.dmg; echo 'Release artifacts cleaned'"
```

Run:

```bash
pnpm vitest run src/runtimeIdentity.test.ts
pnpm build
cargo test -p wenlan-app --lib lifecycle::tests -- --nocapture
```

Expected: PASS.

- [ ] **Step 4: Rename low-risk visible product copy**

Change visible product text from Origin to Wenlan in:

```text
README.md
app/src/lib.rs log path, log filename, startup message, Tauri MCP plugin name, tray menu labels
src/components/SetupWizard.tsx
src/components/onboarding/WhatHappensNextCard.tsx
src/components/onboarding/GhostPagesRow.tsx
src/components/onboarding/MilestoneToaster.tsx
src/lib/agents.ts
src/components/SearchInput.tsx alt text
```

Do not change:

```text
origin-relay.originmemory.workers.dev
legacy Origin bridge comments
tests asserting legacy fallback behavior
```

In `app/src/lib.rs`, add small testable helpers for logging identity:

```rust
fn app_log_dir() -> std::path::PathBuf {
    dirs::home_dir()
        .map(|h| h.join("Library/Logs/com.wenlan.desktop"))
        .unwrap_or_else(std::env::temp_dir)
}

fn app_log_file_name() -> &'static str {
    "wenlan.log"
}
```

Use them in `run()`:

```rust
let log_dir = app_log_dir();
let file_appender = tracing_appender::rolling::daily(&log_dir, app_log_file_name());
```

Add Rust tests:

```rust
#[test]
fn app_log_identity_uses_wenlan() {
    assert!(app_log_dir().ends_with("Library/Logs/com.wenlan.desktop"));
    assert_eq!(app_log_file_name(), "wenlan.log");
}
```

Run:

```bash
pnpm vitest run src/components/SetupWizard.test.tsx src/components/onboarding/__tests__/WhatHappensNextCard.test.tsx src/components/onboarding/__tests__/FirstPageModal.test.tsx src/lib/tauri.test.ts
pnpm build
cargo test -p wenlan-app --lib app_log_identity_uses_wenlan -- --nocapture
```

Expected: PASS with updated Wenlan visible-copy assertions.

- [ ] **Step 5: Commit**

```bash
git add app/tauri.conf.json app/Cargo.toml app/src/main.rs app/src/lib.rs .github/workflows/ci.yml package.json README.md src/runtimeIdentity.test.ts src/components src/lib
git commit -m "fix: rename app product identity to wenlan"
```

## Task 5: Runtime Validation Against v0.9 Daemon

**Files:**
- Modify only if validation exposes a bug.

- [ ] **Step 1: Confirm daemon and app build**

```bash
curl -fsS "http://127.0.0.1:7878/api/health" | jq -e '.status == "ok" and (.version | startswith("0.9."))'
cargo test -p wenlan-app --lib identity_paths -- --nocapture
cargo test -p wenlan-app --lib remote_access::tests::token_path_imports_nonempty_legacy_token_when_current_missing -- --nocapture
cargo test -p wenlan-app --lib lifecycle::tests::cleanup_legacy_app_plist_preserves_foreign_file -- --nocapture
pnpm build
cargo build
cargo test -p wenlan-app --lib
pnpm test
git diff --check
```

Expected:

```text
health route returns ok
targeted bridge tests exit 0
pnpm build exits 0
cargo build exits 0
cargo test exits 0
pnpm test exits 0
git diff --check exits 0
```

The targeted bridge tests must include the old-only fixture cases:

```text
no WENLAN_DATA_DIR
no ORIGIN_DATA_DIR
old default Origin app data exists with config.json or avatars
new default Wenlan app data absent
resolver selects legacy root or imports selected state before choosing Wenlan
legacy MCP token imports only when non-empty
legacy relay_id is not copied
foreign legacy app LaunchAgent is preserved
owned legacy app LaunchAgent is unloaded before removal
```

- [ ] **Step 2: Launch Tauri app**

Kill old dev processes first:

```bash
pgrep -af "tauri dev|target/debug/origin-app|target/debug/wenlan-app|localhost:1420" || true
```

Stop only processes from this repo/worktree. Then launch:

```bash
pnpm tauri dev
```

Expected:

```text
Vite ready at http://localhost:1420/
Running target/debug/wenlan-app
```

Expected after Task 4: `target/debug/wenlan-app`. If the executable remains `origin-app`, the crate/executable rename task is incomplete.

- [ ] **Step 3: Visual/runtime checks**

Confirm:

```text
Dock shows the app icon, not a terminal icon.
App window title/product surface says Wenlan.
Home loads daemon-backed memory/page data.
No blank first paint.
No framework overlay.
Missing legacy avatar file logs do not break page render.
Updater fetch failures do not break page render.
```

Capture a screenshot to:

```text
/private/tmp/wenlan-app-runtime-identity.png
```

- [ ] **Step 4: Commit validation doc update**

Update `docs/superpowers/refactor/2026-06-26-wenlan-app-goal-context.md` with the exact daemon version, app path, process evidence, and screenshot path.

```bash
git add docs/superpowers/refactor/2026-06-26-wenlan-app-goal-context.md
git commit -m "docs: record wenlan app runtime identity validation"
```

## Deferred Follow-Up: Taxonomy and Copy Cleanup

Do not mix this into the runtime identity bridge implementation unless Tasks 1-5 are green.

Next plan should cover:

- Remove first-class `goal` from the visible `MemoryType` union and filters while preserving legacy data display.
- Rename visible `concept` UI to page/wiki language while retaining compatibility wrappers such as `getConceptSources`.
- Reclassify user-facing `domain` language to spaces where appropriate while keeping daemon field compatibility.
- Update tests with residual allowlists for intentionally preserved `goal`, `concept`, and `domain` wire fields.

## Verification Gates

Tasks 1-3, before the Cargo package rename, must end with:

```bash
cargo test -p origin-app --lib
pnpm build
git diff --check
```

After Task 4 changes the Cargo package identity, use:

```bash
cargo test -p wenlan-app --lib
pnpm build
git diff --check
```

Before merging the plan implementation branch:

```bash
pnpm test
cargo build
cargo test -p wenlan-app --lib
curl -fsS "http://127.0.0.1:7878/api/health" | jq -e '.status == "ok" and (.version | startswith("0.9."))'
```

## Boule Review Prompt

```text
/boule:debate Review docs/superpowers/plans/2026-06-26-wenlan-app-runtime-identity-bridge-plan.md. The target is bridge-safe runtime identity migration from Origin.app to Wenlan.app, not a shallow text rename. Attack and defend the plan on user-data safety, old Origin state detection, LaunchAgent label sequencing, MCP token and relay ID migration, relay URL deferral, product bundle identity order, test sufficiency, CodeGraph/ast-grep/LSP boundaries, and Tauri runtime validation against the v0.9 daemon. Identify missing requirements, false dependencies, unsafe ordering, insufficient tests, and any path that could silently open a fresh empty Wenlan state while old Origin data still exists.
```
