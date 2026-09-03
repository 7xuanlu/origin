// SPDX-License-Identifier: AGPL-3.0-only
//! Test-only RAII guard for process environment variables.
//!
//! Restoring by hand at the end of a test leaks the mutation whenever an
//! assertion panics first, which then poisons every later `#[serial]` test in
//! the same process. `Drop` runs during unwind, so this guard restores either
//! way.

use std::ffi::OsString;

pub(crate) struct EnvGuard {
    saved: Vec<(String, Option<OsString>)>,
}

impl EnvGuard {
    /// Snapshot `keys` now; restore their original values (including absence)
    /// when the guard drops. Bind it to a named local — `let _ = ...` drops it
    /// immediately and restores before the test body runs.
    pub(crate) fn capture(keys: &[&str]) -> Self {
        Self {
            saved: keys
                .iter()
                .map(|key| ((*key).to_string(), std::env::var_os(key)))
                .collect(),
        }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (key, value) in &self.saved {
            match value {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
        }
    }
}

/// Point every root the app *writes* to inside `dir`, and restore on drop.
///
/// Setting `HOME` does not do this. On Windows `dirs` resolves
/// `FOLDERID_LocalAppData` and `FOLDERID_Profile` and ignores `HOME`, so a
/// test that sets `HOME` to a tempdir still writes into the developer's real
/// `%LOCALAPPDATA%\wenlan` and `%USERPROFILE%\.config\wenlan-mcp`. Set `HOME`
/// as well when the test needs `~/Library/LaunchAgents` — the two answer
/// different questions and neither substitutes for the other.
///
/// `identity_paths` panics under `cfg(test)` when these are unset, so a test
/// that forgets this fails loudly instead of quietly editing the machine.
pub(crate) fn isolate_app_roots(dir: &std::path::Path) -> EnvGuard {
    let guard = EnvGuard::capture(&[
        crate::identity_paths::TEST_DATA_LOCAL_DIR_ENV,
        crate::identity_paths::TEST_HOME_DIR_ENV,
    ]);
    // Both are *bases*, mirroring `dirs::data_local_dir()` and
    // `dirs::home_dir()`: the `wenlan` / `origin` root selection and the
    // `.config/wenlan-mcp` join still run against them, so a test can stage a
    // legacy root exactly as it would on disk and the tests that pin those
    // layouts still test the layout rather than this override.
    std::env::set_var(crate::identity_paths::TEST_DATA_LOCAL_DIR_ENV, dir);
    std::env::set_var(crate::identity_paths::TEST_HOME_DIR_ENV, dir);
    guard
}
