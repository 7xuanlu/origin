// SPDX-License-Identifier: Apache-2.0
use assert_cmd::Command;
use predicates::prelude::PredicateBooleanExt;
use serde_json::{json, Value};
use wenlan_types::lint::{
    LintAgentSubmission, LintAgentVerdict, LintDigest, LintGateEffect, LintOutcome,
    LintSemanticDecision, LintSemanticReasonCode,
};

#[path = "lint_cli/support.rs"]
mod support;
use support::{
    closed_host, report, report_with_evidence_count, report_with_gates, spawn_error,
    spawn_oversized, spawn_report, spawn_report_capture, spawn_value,
};

fn cli() -> Command {
    let mut command = Command::cargo_bin("wenlan").expect("wenlan binary built");
    command.env("WENLAN_NO_AUTOSTART", "1");
    command
}

#[test]
fn lint_json_accepts_global_flags_before_command_and_uses_remote_scope() {
    let expected = report(&[("memories.lifecycle_integrity", LintOutcome::Pass)]);
    let (base, request) = spawn_report(&expected);

    let output = cli()
        .env("WENLAN_HOST", base)
        .args(["--format", "json", "lint", "--space", "work"])
        .assert()
        .code(0)
        .stderr("")
        .get_output()
        .stdout
        .clone();

    assert_eq!(
        serde_json::from_slice::<Value>(&output).unwrap(),
        serde_json::to_value(expected).unwrap()
    );
    assert_eq!(
        request.recv().unwrap(),
        "GET /api/lint?space=work HTTP/1.1\r\n"
    );
}

#[test]
fn lint_deep_profile_is_forwarded_as_the_canonical_query() {
    let expected = report(&[("memories.lifecycle_integrity", LintOutcome::Pass)]);
    let (base, request) = spawn_report(&expected);

    cli()
        .env("WENLAN_HOST", base)
        .args(["--format", "json", "lint", "--profile", "deep"])
        .assert()
        .code(0)
        .stderr("");

    assert_eq!(
        request.recv().unwrap(),
        "GET /api/lint?profile=deep HTTP/1.1\r\n"
    );
}

#[test]
fn lint_deep_external_egress_consent_is_forwarded_without_selecting_a_provider_slot() {
    let expected = report(&[("memories.lifecycle_integrity", LintOutcome::Pass)]);
    let (base, request) = spawn_report(&expected);

    cli()
        .env("WENLAN_HOST", base)
        .args([
            "--format",
            "json",
            "lint",
            "--profile",
            "deep",
            "--allow-external",
        ])
        .assert()
        .code(0)
        .stderr("");

    assert_eq!(
        request.recv().unwrap(),
        "GET /api/lint?profile=deep&external_egress=true HTTP/1.1\r\n"
    );
}

#[test]
fn lint_external_egress_requires_deep_before_transport() {
    cli()
        .env("WENLAN_HOST", closed_host())
        .args(["lint", "--allow-external"])
        .assert()
        .code(2)
        .stdout("")
        .stderr("wenlan lint: --allow-external requires --profile deep\n");
}

#[test]
fn lint_agent_assist_is_deep_only_and_coexists_with_external_consent() {
    cli()
        .env("WENLAN_HOST", closed_host())
        .args(["lint", "--agent-assist"])
        .assert()
        .code(2)
        .stdout("")
        .stderr("wenlan lint: --agent-assist requires --profile deep\n");

    let expected = report(&[("memories.lifecycle_integrity", LintOutcome::Pass)]);
    let (base, request) = spawn_report(&expected);
    cli()
        .env("WENLAN_HOST", base)
        .args([
            "--format",
            "json",
            "lint",
            "--profile",
            "deep",
            "--allow-external",
            "--agent-assist",
        ])
        .assert()
        .code(0)
        .stderr("");
    assert_eq!(
        request.recv().unwrap(),
        "GET /api/lint?profile=deep&external_egress=true&agent_assist=true HTTP/1.1\r\n"
    );
}

#[test]
fn lint_agent_submission_posts_typed_json_to_the_same_endpoint() {
    let expected = report(&[("memories.lifecycle_integrity", LintOutcome::Pass)]);
    let submission = agent_submission();
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("submission.json");
    std::fs::write(&path, serde_json::to_vec(&submission).unwrap()).unwrap();
    let (base, request) = spawn_report_capture(&expected);

    cli()
        .env("WENLAN_HOST", base)
        .args([
            "--format",
            "json",
            "lint",
            "--profile",
            "deep",
            "--agent-submission",
            path.to_str().unwrap(),
        ])
        .assert()
        .code(0)
        .stderr("");

    let captured = request.recv().unwrap();
    assert_eq!(
        captured.request_line,
        "POST /api/lint?profile=deep&agent_assist=true HTTP/1.1\r\n"
    );
    assert_eq!(
        serde_json::from_slice::<LintAgentSubmission>(&captured.body).unwrap(),
        submission
    );
}

#[test]
fn lint_rejects_oversized_agent_submission_before_transport() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("oversized.json");
    std::fs::write(&path, vec![b' '; 65_537]).unwrap();

    cli()
        .env("WENLAN_HOST", closed_host())
        .args([
            "lint",
            "--profile",
            "deep",
            "--agent-submission",
            path.to_str().unwrap(),
        ])
        .assert()
        .code(2)
        .stdout("")
        .stderr(predicates::str::contains(
            "agent submission exceeds 65536-byte limit",
        ));
}

fn agent_submission() -> LintAgentSubmission {
    let verdicts = vec![LintAgentVerdict::try_new(
        1,
        LintSemanticDecision::Pass,
        None,
        LintSemanticReasonCode::ClassificationMismatch,
        9000,
        Vec::new(),
    )
    .unwrap()];
    LintAgentSubmission::try_new(LintDigest::from_u64(1), verdicts).unwrap()
}

#[test]
fn lint_quiet_json_suppresses_output_without_changing_exit() {
    let expected = report(&[("pages.projection.identity", LintOutcome::Finding)]);
    let (base, _) = spawn_report(&expected);

    cli()
        .env("WENLAN_HOST", base)
        .args(["lint", "--format", "json", "--quiet"])
        .assert()
        .code(1)
        .stdout("")
        .stderr("");
}

#[test]
fn lint_human_clean_report_summarizes_groups_and_exits_zero() {
    let expected = report(&[("memories.lifecycle_integrity", LintOutcome::Pass)]);
    let (base, _) = spawn_report(&expected);

    cli()
        .env("WENLAN_HOST", base)
        .args(["lint", "--format", "table"])
        .assert()
        .code(0)
        .stdout(predicates::str::contains(
            "memories: 10 checks, 0 findings, 0 incomplete",
        ))
        .stdout(predicates::str::contains("Findings: none"))
        .stdout(predicates::str::contains("Incomplete: none"))
        .stderr("");
}

#[test]
fn lint_human_finding_is_actionable_and_exits_one() {
    let expected = report(&[("pages.projection.identity", LintOutcome::Finding)]);
    let (base, _) = spawn_report(&expected);

    cli()
        .env("WENLAN_HOST", base)
        .args(["lint", "--format", "table"])
        .assert()
        .code(1)
        .stdout(predicates::str::contains("Findings (1):"))
        .stdout(predicates::str::contains(
            "pages.projection.identity: finding_detected; recommendation: review_finding",
        ))
        .stdout(predicates::str::contains(
            "affected=1; evaluated=3/3; evidence=opaque:1,reason:invalid_catalog_state; truncated=false",
        ))
        .stderr("");
}

#[test]
fn lint_human_uses_the_seven_canonical_catalog_groups() {
    let expected = report(&[
        ("entities.structural_integrity", LintOutcome::Finding),
        ("memory_entities.integrity", LintOutcome::Finding),
        ("relations.integrity", LintOutcome::Finding),
    ]);
    let (base, _) = spawn_report(&expected);

    let output = cli()
        .env("WENLAN_HOST", base)
        .args(["lint", "--format", "table"])
        .assert()
        .code(1)
        .get_output()
        .stdout
        .clone();
    let output = String::from_utf8(output).unwrap();

    assert!(output.contains("knowledge_graph: 8 checks, 3 findings, 0 incomplete"));
    assert!(!output.contains("\n  entities:"));
    assert!(!output.contains("\n  memory_entities:"));
    assert!(!output.contains("\n  relations:"));
}

#[test]
fn lint_human_caps_evidence_without_hiding_the_population() {
    let expected = report_with_evidence_count("pages.projection.identity", 12);
    let (base, _) = spawn_report(&expected);

    cli()
        .env("WENLAN_HOST", base)
        .args(["lint", "--format", "table"])
        .assert()
        .code(1)
        .stdout(predicates::str::contains(
            "evidence=opaque:1,opaque:2,opaque:3,opaque:4,opaque:5,opaque:6,opaque:7,opaque:8,+4_more",
        ))
        .stdout(predicates::str::contains("evaluated=12/12"))
        .stdout(predicates::str::contains("opaque:9").not());
}

#[test]
fn lint_advisory_only_is_visible_and_exits_zero() {
    let expected = report_with_gates(&[(
        "memories.semantic.contradiction",
        LintOutcome::Finding,
        LintGateEffect::Advisory,
    )]);
    let (base, _) = spawn_report(&expected);

    cli()
        .env("WENLAN_HOST", base)
        .args(["lint", "--format", "table"])
        .assert()
        .code(0)
        .stdout(predicates::str::contains(
            "0 actionable findings, 1 advisory",
        ))
        .stdout(predicates::str::contains("Advisories (1):"))
        .stderr("");
}

#[test]
fn lint_incomplete_precedes_findings_and_exits_two() {
    let expected = report(&[
        ("pages.projection.identity", LintOutcome::Finding),
        (
            "runtime.provider_inventory",
            LintOutcome::NotRunPrerequisite,
        ),
        ("runtime.schema_contract", LintOutcome::InconsistentSnapshot),
        ("runtime.status_parity", LintOutcome::FailedToRun),
    ]);
    let (base, _) = spawn_report(&expected);

    cli()
        .env("WENLAN_HOST", base)
        .args(["lint", "--format", "table"])
        .assert()
        .code(2)
        .stdout(predicates::str::contains("Findings (1):"))
        .stdout(predicates::str::contains("Incomplete (3):"))
        .stdout(predicates::str::contains(
            "runtime.provider_inventory: prerequisite_unavailable; recommendation: restore_prerequisite",
        ))
        .stdout(predicates::str::contains(
            "runtime.schema_contract: snapshot_inconsistent; recommendation: rerun_after_snapshot_stabilizes",
        ))
        .stdout(predicates::str::contains(
            "runtime.status_parity: execution_failed; recommendation: inspect_runtime",
        ))
        .stderr("");
}

#[test]
fn lint_quiet_suppresses_human_output_without_changing_exit() {
    let expected = report(&[("pages.projection.identity", LintOutcome::Finding)]);
    let (base, _) = spawn_report(&expected);

    cli()
        .env("WENLAN_HOST", base)
        .args(["--quiet", "lint", "--format", "table"])
        .assert()
        .code(1)
        .stdout("")
        .stderr("");
}

#[test]
fn lint_transport_and_schema_failures_exit_two_on_stderr_only() {
    let host = closed_host();
    cli()
        .env("WENLAN_HOST", &host)
        .arg("lint")
        .assert()
        .code(2)
        .stdout("")
        .stderr(predicates::str::contains(format!(
            "GET {host}/api/lint failed"
        )));

    let mut unsupported = serde_json::to_value(report(&[(
        "memories.lifecycle_integrity",
        LintOutcome::Pass,
    )]))
    .expect("serialize canonical report");
    unsupported["report_schema_version"] = json!(999);
    let (base, _) = spawn_value(&unsupported);
    cli()
        .env("WENLAN_HOST", base)
        .arg("lint")
        .assert()
        .code(2)
        .stdout("")
        .stderr(predicates::str::contains("unsupported_lint_report_schema"));
}

#[test]
fn lint_invalid_scope_preserves_typed_daemon_diagnostic() {
    let (base, _) = spawn_error(422, &json!({"error": "invalid_scope"}));

    cli()
        .env("WENLAN_HOST", base)
        .args(["--format", "json", "lint", "--space", "missing"])
        .assert()
        .code(2)
        .stdout("")
        .stderr("wenlan lint: invalid_scope\n");
}

#[test]
fn lint_rejects_oversized_daemon_response() {
    let (base, _) = spawn_oversized();

    cli()
        .env("WENLAN_HOST", base)
        .args(["--format", "json", "lint"])
        .assert()
        .code(2)
        .stdout("")
        .stderr(predicates::str::contains(
            "lint response exceeds 8388608 bytes",
        ));
}

#[test]
fn lint_does_not_create_wiki_command_or_change_pages_doctor_query() {
    cli()
        .args(["wiki", "check"])
        .assert()
        .code(2)
        .stdout("")
        .stderr(predicates::str::contains("unrecognized subcommand 'wiki'"));

    let home = tempfile::tempdir().expect("isolated home");
    cli()
        .env("HOME", home.path())
        .env("WENLAN_DATA_DIR", home.path())
        .args(["pages", "doctor", "--format", "table"])
        .assert()
        .code(0)
        .stdout("")
        .stderr(predicates::str::contains("no page matches: doctor"));
}
