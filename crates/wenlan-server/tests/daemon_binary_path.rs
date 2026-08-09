// SPDX-License-Identifier: Apache-2.0
//! Contract tests for Cargo/nextest daemon binary resolution.

#[allow(dead_code)]
#[path = "common/daemon_binary.rs"]
mod daemon_binary;

use std::ffi::OsString;
use std::path::PathBuf;

#[test]
fn nextest_runtime_binary_overrides_the_cargo_build_path() {
    let runtime = PathBuf::from("relocated-nextest-bin/wenlan-server");
    assert_eq!(
        daemon_binary::resolve_wenlan_server_binary(Some(runtime.clone().into_os_string())),
        runtime
    );
}

#[test]
fn cargo_build_path_is_the_non_nextest_fallback() {
    assert_eq!(
        daemon_binary::resolve_wenlan_server_binary(None),
        PathBuf::from(env!("CARGO_BIN_EXE_wenlan-server"))
    );
}

#[test]
#[should_panic(expected = "wenlan-server binary path must not be empty")]
fn empty_runtime_binary_path_is_rejected() {
    let _ = daemon_binary::resolve_wenlan_server_binary(Some(OsString::new()));
}
