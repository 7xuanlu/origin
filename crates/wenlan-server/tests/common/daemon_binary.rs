// SPDX-License-Identifier: Apache-2.0
//! Resolve the daemon binary for Cargo and relocatable nextest archives.

use std::ffi::OsString;
use std::path::PathBuf;

pub fn resolve_wenlan_server_binary(nextest_binary: Option<OsString>) -> PathBuf {
    let path = PathBuf::from(
        nextest_binary.unwrap_or_else(|| OsString::from(env!("CARGO_BIN_EXE_wenlan-server"))),
    );
    assert!(
        !path.as_os_str().is_empty(),
        "wenlan-server binary path must not be empty"
    );
    path
}

pub fn wenlan_server_binary() -> PathBuf {
    resolve_wenlan_server_binary(std::env::var_os("NEXTEST_BIN_EXE_wenlan_server"))
}
