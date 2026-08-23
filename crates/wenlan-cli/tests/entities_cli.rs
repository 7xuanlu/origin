// SPDX-License-Identifier: Apache-2.0
//! Integration tests for `wenlan entities` subcommands.
//!
//! Daemon behavior (merge/alias, resolution) is covered by wenlan-server
//! integration tests. These tests keep the process-level CLI contract
//! focused on argument parsing.

use assert_cmd::Command;

fn cli() -> Command {
    let mut command = Command::cargo_bin("wenlan").expect("origin binary built");
    command.env("WENLAN_NO_AUTOSTART", "1");
    command
}

#[test]
fn entities_subcommands_help_exits_zero() {
    for args in [
        &["entities", "--help"][..],
        &["entities", "merge", "--help"][..],
        &["entities", "alias", "--help"][..],
    ] {
        cli().args(args).assert().success();
    }
}

#[test]
fn entities_merge_requires_into() {
    cli()
        .args(["entities", "merge", "Alicia"])
        .assert()
        .failure()
        .stderr(predicates::str::contains("--into"));
}

#[test]
fn entities_alias_requires_alias_arg() {
    cli()
        .args(["entities", "alias", "Alicia"])
        .assert()
        .failure();
}
