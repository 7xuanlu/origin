use wenlan_mcp::client::WenlanClient;
use wenlan_mcp::tools::{
    LintAgentSubmissionParam, LintAgentVerdictParam, LintParams, LintProfileParam,
    LintSemanticDecisionParam, LintSemanticReasonParam, TransportMode, WenlanMcpServer,
};
use wenlan_types::lint::{
    canonical_check_ids, canonical_gate_effect, LintApplicability, LintCapabilityContext,
    LintCheckResult, LintCheckResultInput, LintConfigFingerprint, LintCoverage, LintDbSnapshotMode,
    LintDbSnapshotReceipt, LintDigest, LintOutcome, LintPageSnapshotMode, LintPageSnapshotReceipt,
    LintPrecondition, LintProducerReceipt, LintProfile, LintReport, LintScope, LintSeverity,
    LintSnapshotReceipts, LintSummaryCode, LintValidationMethod, LINT_MAX_EVIDENCE_PER_CHECK,
};
use wiremock::matchers::{body_json, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn fixture_for(profile: LintProfile) -> LintReport {
    let digest = || LintDigest::from_u64(1);
    LintReport::try_new_for_profile(
        profile,
        LintScope::global(),
        LintCapabilityContext::daemon_operator_endpoint(),
        LintSnapshotReceipts::new(
            LintDbSnapshotReceipt::new(
                LintDbSnapshotMode::TransactionalReadOnly,
                digest(),
                Some(digest()),
            ),
            LintPageSnapshotReceipt::new(
                LintPageSnapshotMode::BestEffort,
                digest(),
                Some(digest()),
            ),
        ),
        LintConfigFingerprint::from_effective_config(&[]),
        LintProducerReceipt::new(None),
        canonical_check_ids(profile)
            .map(|check_id| {
                LintCheckResult::try_new_with_gate_effect(
                    LintCheckResultInput {
                        check_id: check_id.to_string(),
                        outcome: LintOutcome::Pass,
                        severity: LintSeverity::Info,
                        applicability: LintApplicability::Inventory,
                        precondition: LintPrecondition::Ready,
                        coverage: LintCoverage::new(
                            LintValidationMethod::FullEnumeration,
                            0,
                            0,
                            LINT_MAX_EVIDENCE_PER_CHECK,
                            false,
                            0,
                        )
                        .unwrap(),
                        metrics: Vec::new(),
                        summary_code: LintSummaryCode::CheckPassed,
                        recommendation_code: None,
                        evidence: Vec::new(),
                        duration_ms: 0,
                    },
                    canonical_gate_effect(profile, check_id).unwrap(),
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap()
}

fn fixture() -> LintReport {
    fixture_for(LintProfile::Deep)
}

#[tokio::test]
async fn mcp_general_lint_returns_the_cli_text_and_no_structured_report() {
    let mock = MockServer::start().await;
    let report = fixture_for(LintProfile::General);
    Mock::given(method("GET"))
        .and(path("/api/lint"))
        .and(query_param("profile", "general"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&report))
        .expect(1)
        .mount(&mock)
        .await;
    let server = WenlanMcpServer::new(
        WenlanClient::new(mock.uri()),
        TransportMode::Stdio,
        "test-agent".to_string(),
        None,
    );

    let result = server
        .lint_impl(LintParams {
            profile: Some(LintProfileParam::General),
            space: None,
            agent_assist: false,
            agent_submission: None,
        })
        .await
        .unwrap();

    assert!(
        result.structured_content.is_none(),
        "General carries no typed report: clients drop the text when one is present"
    );
    assert_eq!(result.content.len(), 1);
    assert_eq!(
        result.content[0].as_text().map(|text| text.text.as_str()),
        Some(report.render_text().as_str())
    );
}

#[tokio::test]
async fn mcp_lint_forwards_typed_profile_and_scope_and_returns_canonical_report() {
    let mock = MockServer::start().await;
    let report = fixture();
    Mock::given(method("GET"))
        .and(path("/api/lint"))
        .and(query_param("profile", "deep"))
        .and(query_param("space", "work"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&report))
        .expect(1)
        .mount(&mock)
        .await;
    let server = WenlanMcpServer::new(
        WenlanClient::new(mock.uri()),
        TransportMode::Stdio,
        "test-agent".to_string(),
        None,
    );

    let result = server
        .lint_impl(LintParams {
            profile: Some(LintProfileParam::Deep),
            space: Some("work".to_string()),
            agent_assist: false,
            agent_submission: None,
        })
        .await
        .unwrap();

    assert_eq!(
        result.structured_content,
        Some(serde_json::to_value(&report).unwrap())
    );
    assert_eq!(result.is_error, Some(false));
}

#[tokio::test]
async fn mcp_lint_rejects_unknown_report_schema_via_typed_deserialization() {
    let mock = MockServer::start().await;
    let mut body = serde_json::to_value(fixture()).unwrap();
    body["report_schema_version"] = serde_json::json!(999);
    body["private_canary"] = serde_json::json!("RAW_PAGE_TEXT_MUST_NOT_LEAK");
    Mock::given(method("GET"))
        .and(path("/api/lint"))
        .respond_with(ResponseTemplate::new(200).set_body_json(body))
        .expect(1)
        .mount(&mock)
        .await;
    let server = WenlanMcpServer::new(
        WenlanClient::new(mock.uri()),
        TransportMode::Stdio,
        "test-agent".to_string(),
        None,
    );

    let result = server
        .lint_impl(LintParams {
            profile: None,
            space: None,
            agent_assist: false,
            agent_submission: None,
        })
        .await
        .unwrap();

    assert_eq!(result.is_error, Some(true));
    assert!(result.structured_content.is_none());
    assert!(!serde_json::to_string(&result)
        .unwrap()
        .contains("RAW_PAGE_TEXT_MUST_NOT_LEAK"));
}

#[tokio::test]
async fn mcp_lint_rejects_oversized_daemon_response_without_echoing_it() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/lint"))
        .respond_with(ResponseTemplate::new(200).set_body_string("x".repeat(8 * 1024 * 1024 + 1)))
        .expect(1)
        .mount(&mock)
        .await;
    let server = WenlanMcpServer::new(
        WenlanClient::new(mock.uri()),
        TransportMode::Stdio,
        "test-agent".to_string(),
        None,
    );

    let result = server
        .lint_impl(LintParams {
            profile: None,
            space: None,
            agent_assist: false,
            agent_submission: None,
        })
        .await
        .unwrap();

    assert_eq!(result.is_error, Some(true));
    assert!(result.structured_content.is_none());
    assert!(serde_json::to_string(&result)
        .unwrap()
        .contains("size limit"));
}

#[tokio::test]
async fn mcp_lint_agent_prepare_and_submission_use_one_typed_tool_and_endpoint() {
    let prepare_mock = MockServer::start().await;
    let report = fixture();
    Mock::given(method("GET"))
        .and(path("/api/lint"))
        .and(query_param("profile", "deep"))
        .and(query_param("agent_assist", "true"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&report))
        .expect(1)
        .mount(&prepare_mock)
        .await;
    let prepare_server = WenlanMcpServer::new(
        WenlanClient::new(prepare_mock.uri()),
        TransportMode::Stdio,
        "test-agent".to_string(),
        None,
    );
    prepare_server
        .lint_impl(LintParams {
            profile: Some(LintProfileParam::Deep),
            space: None,
            agent_assist: true,
            agent_submission: None,
        })
        .await
        .unwrap();

    let submission = LintAgentSubmissionParam {
        work_digest: "0000000000000001".to_string(),
        verdicts: vec![LintAgentVerdictParam {
            candidate_ref: 1,
            decision: LintSemanticDecisionParam::Pass,
            second_decision: None,
            reason_code: LintSemanticReasonParam::ClassificationMismatch,
            confidence_basis_points: 9000,
            counterevidence_refs: Vec::new(),
        }],
    };
    let expected_body = serde_json::to_value(&submission).unwrap();
    let submit_mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/lint"))
        .and(query_param("profile", "deep"))
        .and(query_param("agent_assist", "true"))
        .and(body_json(expected_body))
        .respond_with(ResponseTemplate::new(200).set_body_json(&report))
        .expect(1)
        .mount(&submit_mock)
        .await;
    let submit_server = WenlanMcpServer::new(
        WenlanClient::new(submit_mock.uri()),
        TransportMode::Stdio,
        "test-agent".to_string(),
        None,
    );
    submit_server
        .lint_impl(LintParams {
            profile: Some(LintProfileParam::Deep),
            space: None,
            agent_assist: false,
            agent_submission: Some(submission),
        })
        .await
        .unwrap();
}
