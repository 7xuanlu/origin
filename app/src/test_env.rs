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
