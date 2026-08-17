// SPDX-License-Identifier: Apache-2.0
//! Integration tests for `wenlan spaces` subcommands.
//!
//! Daemon behavior is covered by wenlan-server integration tests. These tests
//! keep the process-level CLI contract focused on argument parsing, including
//! the daemon-backed Default clear surface.

use assert_cmd::Command;

fn cli() -> Command {
    let mut command = Command::cargo_bin("wenlan").expect("origin binary built");
    command.env("WENLAN_NO_AUTOSTART", "1");
    command
}

// ---------------------------------------------------------------------------
// Argument parsing — all `spaces` subcommands must accept --help
// ---------------------------------------------------------------------------

#[test]
fn spaces_subcommands_help_exits_zero() {
    for args in [
        &["spaces", "--help"][..],
        &["spaces", "list", "--help"][..],
        &["spaces", "add", "--help"][..],
        &["spaces", "default", "--help"][..],
        &["spaces", "move", "--help"][..],
        &["spaces", "show", "--help"][..],
    ] {
        cli().args(args).assert().success();
    }
}

#[test]
fn spaces_default_clear_conflicts_with_name() {
    cli()
        .args(["spaces", "default", "work", "--clear"])
        .assert()
        .failure()
        .stderr(predicates::str::contains("cannot be used with"));
}

#[test]
fn global_space_and_all_spaces_conflict() {
    cli()
        .args(["--space", "work", "--all-spaces", "search", "canary"])
        .assert()
        .failure()
        .stderr(predicates::str::contains("cannot be used with"));
}

#[test]
fn global_agent_and_space_options_are_visible() {
    cli()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicates::str::contains("--agent-name"))
        .stdout(predicates::str::contains("--all-spaces"))
        .stdout(predicates::str::contains("--space"));
}
