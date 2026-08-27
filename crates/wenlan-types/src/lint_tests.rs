// SPDX-License-Identifier: Apache-2.0
use super::*;
use serde::Deserialize;
use serde_json::json;

fn coverage(authorized_denominator: u64) -> LintCoverage {
    LintCoverage::new(
        LintValidationMethod::FullEnumeration,
        authorized_denominator,
        authorized_denominator,
        100,
        false,
        if authorized_denominator == 0 { 0 } else { 1 },
    )
    .unwrap()
}

fn check(
    outcome: LintOutcome,
    severity: LintSeverity,
) -> Result<LintCheckResult, LintContractError> {
    check_with_id("catalog.open_check", outcome, severity)
}

fn check_with_id(
    check_id: &str,
    outcome: LintOutcome,
    severity: LintSeverity,
) -> Result<LintCheckResult, LintContractError> {
    let (applicability, precondition, summary_code, recommendation_code) = match outcome {
        LintOutcome::Pass => (
            LintApplicability::Applicable,
            LintPrecondition::Ready,
            LintSummaryCode::CheckPassed,
            None,
        ),
        LintOutcome::Finding => (
            LintApplicability::Applicable,
            LintPrecondition::Ready,
            LintSummaryCode::FindingDetected,
            Some(LintRecommendationCode::ReviewFinding),
        ),
        LintOutcome::NotRunPrerequisite => (
            LintApplicability::NotApplicable,
            LintPrecondition::MissingPrerequisite,
            LintSummaryCode::PrerequisiteUnavailable,
            Some(LintRecommendationCode::RestorePrerequisite),
        ),
        LintOutcome::InconsistentSnapshot => (
            LintApplicability::Applicable,
            LintPrecondition::SnapshotUnstable,
            LintSummaryCode::SnapshotInconsistent,
            Some(LintRecommendationCode::RerunAfterSnapshotStabilizes),
        ),
        LintOutcome::FailedToRun => (
            LintApplicability::Applicable,
            LintPrecondition::Ready,
            LintSummaryCode::ExecutionFailed,
            Some(LintRecommendationCode::InspectRuntime),
        ),
    };

    LintCheckResult::try_new(LintCheckResultInput {
        check_id: check_id.to_string(),
        outcome,
        severity,
        applicability,
        precondition,
        coverage: coverage(3),
        metrics: vec![LintMetric::new(
            LintMetricCode::ObservedRecords,
            LintMetricValue::CatalogCode {
                code: LintMetricStringCode::Ready,
            },
        )],
        summary_code,
        recommendation_code,
        evidence: vec![LintEvidenceRef::OpaqueId {
            opaque_id: LintOpaqueId::from_sorted_position(0).unwrap(),
        }],
        duration_ms: 4,
    })
}

fn wire_report(scope: LintScope) -> LintReport {
    report(
        scope,
        canonical_check_ids(LintProfile::General)
            .map(|check_id| check_with_id(check_id, LintOutcome::Pass, LintSeverity::Info).unwrap())
            .collect(),
    )
}

fn snapshots() -> LintSnapshotReceipts {
    LintSnapshotReceipts::new(
        LintDbSnapshotReceipt::new(
            LintDbSnapshotMode::TransactionalReadOnly,
            LintDigest::from_u64(1),
            Some(LintDigest::from_u64(2)),
        ),
        LintPageSnapshotReceipt::new(
            LintPageSnapshotMode::BestEffort,
            LintDigest::from_u64(3),
            Some(LintDigest::from_u64(4)),
        ),
    )
}

fn report(scope: LintScope, checks: Vec<LintCheckResult>) -> LintReport {
    LintReport::try_new(
        scope,
        LintCapabilityContext::daemon_operator_endpoint(),
        snapshots(),
        LintConfigFingerprint::from_effective_config(&[
            LintConfigSelection::new(LintConfigSetting::RerankerEnabled, LintConfigValue::Enabled),
            LintConfigSelection::new(
                LintConfigSetting::PageProjectionEnabled,
                LintConfigValue::Disabled,
            ),
        ]),
        LintProducerReceipt::new(None),
        checks,
    )
    .unwrap()
}

#[test]
fn report_roundtrips_current_schema_for_each_applied_scope_kind() {
    let scopes = [
        LintScope::global(),
        LintScope::registered(LintOpaqueId::from_sorted_position(0).unwrap()),
        LintScope::uncategorized(),
    ];

    for scope in scopes {
        let report = wire_report(scope);

        let encoded = serde_json::to_value(&report).unwrap();
        let decoded: LintReport = serde_json::from_value(encoded.clone()).unwrap();

        assert!(decoded.complete());
        assert_eq!(
            encoded["report_schema_version"],
            json!(LINT_REPORT_SCHEMA_VERSION)
        );
        assert_eq!(
            encoded["check_catalog_version"],
            json!(LINT_CHECK_CATALOG_VERSION)
        );
        assert_eq!(encoded["profile"], json!("general"));
        assert_eq!(
            encoded["capability_context"],
            json!("daemon_operator_endpoint_unauthenticated_unverified")
        );
        assert_eq!(
            encoded["snapshots"]["db"]["mode"],
            json!("transactional_read_only")
        );
        assert_eq!(encoded["snapshots"]["pages"]["mode"], json!("best_effort"));
        assert!(encoded["producer_receipt"]["runtime_commit"].is_null());
    }
}

#[test]
fn opaque_digest_evidence_roundtrips_without_ordinal_semantics() {
    assert_eq!(LINT_REPORT_SCHEMA_VERSION, 5);
    assert_eq!(LINT_CHECK_CATALOG_VERSION, 3);
    let evidence = LintEvidenceRef::OpaqueDigest {
        opaque_digest: LintOpaqueDigest::from_hex(
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        )
        .unwrap(),
    };
    let result = LintCheckResult::try_new(LintCheckResultInput {
        check_id: "identity.tag_integrity".to_string(),
        outcome: LintOutcome::Finding,
        severity: LintSeverity::Error,
        applicability: LintApplicability::Applicable,
        precondition: LintPrecondition::Ready,
        coverage: coverage(1),
        metrics: vec![LintMetric::new(
            LintMetricCode::AffectedRecords,
            LintMetricValue::Count { value: 1 },
        )],
        summary_code: LintSummaryCode::FindingDetected,
        recommendation_code: Some(LintRecommendationCode::ReviewFinding),
        evidence: vec![evidence.clone()],
        duration_ms: 1,
    })
    .unwrap();
    let value = serde_json::to_value(&result).unwrap();
    assert_eq!(value["evidence"][0]["kind"], "opaque_digest");
    assert_eq!(
        serde_json::from_value::<LintCheckResult>(value)
            .unwrap()
            .evidence(),
        &[evidence]
    );
}

#[test]
fn result_accepts_only_legal_outcome_severity_pairs() {
    let legal_pairs = [
        (LintOutcome::Pass, LintSeverity::Info),
        (LintOutcome::Finding, LintSeverity::Warning),
        (LintOutcome::Finding, LintSeverity::Error),
        (LintOutcome::NotRunPrerequisite, LintSeverity::Error),
        (LintOutcome::InconsistentSnapshot, LintSeverity::Error),
        (LintOutcome::FailedToRun, LintSeverity::Error),
    ];

    for outcome in [
        LintOutcome::Pass,
        LintOutcome::Finding,
        LintOutcome::NotRunPrerequisite,
        LintOutcome::InconsistentSnapshot,
        LintOutcome::FailedToRun,
    ] {
        for severity in [
            LintSeverity::Info,
            LintSeverity::Warning,
            LintSeverity::Error,
        ] {
            let result = check(outcome, severity);
            assert_eq!(result.is_ok(), legal_pairs.contains(&(outcome, severity)));
        }
    }

    let expected_empty = LintCheckResult::try_new(LintCheckResultInput {
        check_id: "catalog.empty".to_string(),
        outcome: LintOutcome::Pass,
        severity: LintSeverity::Info,
        applicability: LintApplicability::ExpectedEmpty,
        precondition: LintPrecondition::ConfiguredOff,
        coverage: coverage(0),
        metrics: vec![],
        summary_code: LintSummaryCode::ExpectedEmpty,
        recommendation_code: None,
        evidence: vec![],
        duration_ms: 0,
    });
    assert!(expected_empty.is_ok());
}

#[test]
fn report_derives_completeness_and_totals_from_outcomes() {
    let complete = report(
        LintScope::global(),
        vec![
            check(LintOutcome::Pass, LintSeverity::Info).unwrap(),
            check(LintOutcome::Finding, LintSeverity::Warning).unwrap(),
        ],
    );
    assert!(complete.complete());
    assert_eq!(complete.totals().checks(), 2);
    assert_eq!(complete.totals().findings(), 1);
    assert_eq!(complete.totals().incomplete(), 0);

    let incomplete = report(
        LintScope::global(),
        vec![check(LintOutcome::FailedToRun, LintSeverity::Error).unwrap()],
    );
    assert!(!incomplete.complete());
    assert_eq!(incomplete.totals().incomplete(), 1);

    let mut mismatched = serde_json::to_value(&incomplete).unwrap();
    mismatched["complete"] = json!(true);
    assert!(serde_json::from_value::<LintReport>(mismatched).is_err());
}

#[test]
fn report_separates_actionable_and_advisory_findings() {
    let actionable = check(LintOutcome::Finding, LintSeverity::Error).unwrap();
    let mut advisory_wire =
        serde_json::to_value(check(LintOutcome::Finding, LintSeverity::Warning).unwrap()).unwrap();
    advisory_wire["gate_effect"] = json!("advisory");
    let advisory = serde_json::from_value(advisory_wire).unwrap();
    let report = report(LintScope::global(), vec![actionable, advisory]);

    assert_eq!(report.totals().findings(), 2);
    assert_eq!(report.totals().actionable_findings(), 1);
    assert_eq!(report.totals().advisory_findings(), 1);
}

#[test]
fn rejects_unknown_enums_and_unsupported_schema_versions() {
    let report = wire_report(LintScope::global());

    let mut unknown_enum = serde_json::to_value(&report).unwrap();
    unknown_enum["checks"][0]["outcome"] = json!("future_outcome");
    assert!(serde_json::from_value::<LintReport>(unknown_enum).is_err());

    let mut unsupported_schema = serde_json::to_value(&report).unwrap();
    unsupported_schema["report_schema_version"] = json!(LINT_REPORT_SCHEMA_VERSION + 1);
    assert!(serde_json::from_value::<LintReport>(unsupported_schema).is_err());
}

#[test]
fn permits_additive_object_fields_but_rejects_schema_drift() {
    let report = wire_report(LintScope::global());
    let mut additive = serde_json::to_value(&report).unwrap();
    additive["future_top_level"] = json!({ "added": true });
    additive["checks"][0]["future_check_field"] = json!(1);

    assert!(serde_json::from_value::<LintReport>(additive).is_ok());
}

#[test]
fn deserialization_rejects_incomplete_or_duplicate_catalog_shapes() {
    let report = wire_report(LintScope::global());
    let mut missing = serde_json::to_value(&report).unwrap();
    missing["checks"].as_array_mut().unwrap().pop();
    missing["totals"]["checks"] = json!(LINT_GENERAL_CHECK_COUNT - 1);
    assert!(serde_json::from_value::<LintReport>(missing).is_err());

    let mut duplicate = serde_json::to_value(&report).unwrap();
    duplicate["checks"][1] = duplicate["checks"][0].clone();
    assert!(serde_json::from_value::<LintReport>(duplicate).is_err());
}

#[test]
fn deserialization_rejects_replaced_check_ids_and_gate_downgrades() {
    let report = wire_report(LintScope::global());

    let mut replaced = serde_json::to_value(&report).unwrap();
    replaced["checks"][0]["check_id"] = json!("entities.future_replacement");
    assert!(serde_json::from_value::<LintReport>(replaced).is_err());

    let mut downgraded = serde_json::to_value(&report).unwrap();
    downgraded["checks"][0]["gate_effect"] = json!("advisory");
    assert!(serde_json::from_value::<LintReport>(downgraded).is_err());
}

#[test]
fn bounds_evidence_and_rejects_raw_identifier_or_path_attempts() {
    let too_many = vec![
        LintEvidenceRef::ReasonCode {
            reason_code: LintReasonCode::MissingArtifact,
        };
        101
    ];
    let bounded = LintCheckResult::try_new(LintCheckResultInput {
        check_id: "catalog.bounded".to_string(),
        outcome: LintOutcome::Pass,
        severity: LintSeverity::Info,
        applicability: LintApplicability::Applicable,
        precondition: LintPrecondition::Ready,
        coverage: coverage(101),
        metrics: vec![],
        summary_code: LintSummaryCode::CheckPassed,
        recommendation_code: None,
        evidence: too_many,
        duration_ms: 0,
    });
    assert!(bounded.is_err());

    let arbitrary_path = json!({
        "kind": "safe_root_relative_path",
        "safe_root_relative_path": "../../private/raw-content-canary"
    });
    assert!(serde_json::from_value::<LintEvidenceRef>(arbitrary_path).is_err());

    let arbitrary_identifier = json!({ "kind": "opaque_id", "opaque_id": "person-42" });
    assert!(serde_json::from_value::<LintEvidenceRef>(arbitrary_identifier).is_err());

    let encoded = serde_json::to_string(&report(
        LintScope::registered(LintOpaqueId::from_sorted_position(0).unwrap()),
        vec![check(LintOutcome::Pass, LintSeverity::Info).unwrap()],
    ))
    .unwrap();
    for forbidden in [
        "raw-content-canary",
        "requested_space",
        "config_values",
        "error_message",
        "arbitrary_identifier",
        "\"path\"",
    ] {
        assert!(
            !encoded.contains(forbidden),
            "serialized forbidden value: {forbidden}"
        );
    }
}

#[test]
fn fingerprints_are_deterministic_and_string_metrics_are_closed() {
    let forward = LintConfigFingerprint::from_effective_config(&[
        LintConfigSelection::new(LintConfigSetting::RerankerEnabled, LintConfigValue::Enabled),
        LintConfigSelection::new(
            LintConfigSetting::PageProjectionEnabled,
            LintConfigValue::Disabled,
        ),
    ]);
    let reverse = LintConfigFingerprint::from_effective_config(&[
        LintConfigSelection::new(
            LintConfigSetting::PageProjectionEnabled,
            LintConfigValue::Disabled,
        ),
        LintConfigSelection::new(LintConfigSetting::RerankerEnabled, LintConfigValue::Enabled),
    ]);
    assert_eq!(forward, reverse);

    let report = report(
        LintScope::global(),
        vec![check(LintOutcome::Pass, LintSeverity::Info).unwrap()],
    );
    let mut invalid_metric = serde_json::to_value(&report).unwrap();
    invalid_metric["checks"][0]["metrics"][0]["value"] =
        json!({ "kind": "catalog_code", "code": "untrusted-string" });
    assert!(serde_json::from_value::<LintReport>(invalid_metric).is_err());

    let encoded = serde_json::to_string(&report).unwrap();
    assert!(!encoded.contains("reranker_enabled"));
    assert!(!encoded.contains("page_projection_enabled"));
}

#[test]
fn query_defaults_to_general_and_accepts_only_closed_profiles() {
    let query: LintQuery = serde_json::from_value(json!({ "space": "requested-space" })).unwrap();
    assert_eq!(query.space.as_deref(), Some("requested-space"));
    assert_eq!(query.applied_profile(), LintProfile::General);

    let deep: LintQuery = serde_json::from_value(json!({ "profile": "deep" })).unwrap();
    assert_eq!(deep.applied_profile(), LintProfile::Deep);
    assert!(serde_json::from_value::<LintQuery>(json!({ "profile": "future" })).is_err());

    let general = report(
        LintScope::global(),
        vec![check(LintOutcome::Pass, LintSeverity::Info).unwrap()],
    );
    assert_eq!(general.profile(), LintProfile::General);

    let request: LintRequestQuery = serde_json::from_value(json!({
        "profile": "deep",
        "space": "requested-space",
        "external_egress": true,
        "agent_assist": true
    }))
    .unwrap();
    assert_eq!(request.lint().applied_profile(), LintProfile::Deep);
    assert_eq!(request.lint().space.as_deref(), Some("requested-space"));
    assert!(request.external_egress());
    assert!(request.agent_assist());
    assert_eq!(
        serde_json::to_value(LintRequestQuery::new(
            LintQuery::new(Some(LintProfile::Deep), None),
            false,
            false,
        ))
        .unwrap(),
        json!({ "profile": "deep" })
    );
}

#[test]
fn agent_work_and_submission_are_bounded_typed_contracts() {
    let populations = LintSemanticCheckId::ALL
        .into_iter()
        .map(|check_id| {
            LintSemanticPopulation::try_new(
                check_id,
                1,
                u64::from(check_id == LintSemanticCheckId::MemoryContradiction),
                u64::from(check_id == LintSemanticCheckId::MemoryContradiction),
                false,
            )
            .unwrap()
        })
        .collect();
    let work = LintAgentWork::try_new(
        LintDigest::from_u64(9),
        populations,
        vec![
            LintAgentRecord::try_new(
                1,
                LintAgentRecordKind::Memory,
                "bounded memory excerpt".to_string(),
                Some("fact".to_string()),
                None,
                None,
            )
            .unwrap(),
            LintAgentRecord::try_new(
                2,
                LintAgentRecordKind::Page,
                "bounded page excerpt".to_string(),
                None,
                Some(1),
                Some("bounded source excerpt".to_string()),
            )
            .unwrap(),
        ],
        vec![LintAgentCandidate::try_new(
            1,
            LintSemanticCheckId::MemoryContradiction,
            LintSemanticCandidateKind::PairReview,
            vec![1, 2],
            vec![],
            LintSemanticAction::ReviewContradiction,
            LintSemanticReasonCode::PotentialContradiction,
        )
        .unwrap()],
    )
    .unwrap();
    assert_eq!(work.records().len(), 2);

    let verdicts = vec![LintAgentVerdict::try_new(
        1,
        LintSemanticDecision::Finding,
        None,
        LintSemanticReasonCode::PotentialContradiction,
        8_500,
        vec![],
    )
    .unwrap()];
    let submission = LintAgentSubmission::try_new(work.work_digest().clone(), verdicts).unwrap();
    let encoded = serde_json::to_value(&submission).unwrap();
    let decoded: LintAgentSubmission = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded, submission);

    let report = LintReport::try_new_for_profile_with_agent_work(
        LintProfile::Deep,
        LintScope::global(),
        LintCapabilityContext::daemon_operator_endpoint(),
        snapshots(),
        LintConfigFingerprint::from_effective_config(&[]),
        LintProducerReceipt::new(None),
        LINT_CANONICAL_CHECK_IDS
            .iter()
            .map(|check_id| {
                let check = check_with_id(check_id, LintOutcome::Pass, LintSeverity::Info).unwrap();
                let mut wire = serde_json::to_value(check).unwrap();
                wire["gate_effect"] = serde_json::to_value(
                    canonical_gate_effect(LintProfile::Deep, check_id).unwrap(),
                )
                .unwrap();
                serde_json::from_value(wire).unwrap()
            })
            .collect(),
        Some(work.clone()),
    )
    .unwrap();
    let decoded: LintReport =
        serde_json::from_value(serde_json::to_value(&report).unwrap()).unwrap();
    assert_eq!(decoded.agent_work(), Some(&work));

    assert_eq!(
        LintReport::try_new_for_profile_with_agent_work(
            LintProfile::General,
            LintScope::global(),
            LintCapabilityContext::daemon_operator_endpoint(),
            snapshots(),
            LintConfigFingerprint::from_effective_config(&[]),
            LintProducerReceipt::new(None),
            (0..LINT_GENERAL_CHECK_COUNT)
                .map(|index| {
                    check_with_id(
                        &format!("catalog.open_general_check_{index:02}"),
                        LintOutcome::Pass,
                        LintSeverity::Info,
                    )
                    .unwrap()
                })
                .collect(),
            Some(work.clone()),
        ),
        Err(LintContractError::InvalidAgentWork)
    );

    assert!(LintAgentRecord::try_new(
        1,
        LintAgentRecordKind::Memory,
        "x".repeat(LINT_AGENT_EXCERPT_CHAR_CAP + 1),
        None,
        None,
        None,
    )
    .is_err());
    assert!(LintAgentVerdict::try_new(
        1,
        LintSemanticDecision::Finding,
        None,
        LintSemanticReasonCode::PotentialContradiction,
        8_500,
        vec![2, 1],
    )
    .is_err());
}

#[test]
fn candidate_semantic_contract_carries_population_action_and_disagreement() {
    let records = vec![
        LintAgentRecord::try_new(
            1,
            LintAgentRecordKind::Memory,
            "Project Atlas moved to the work space".to_string(),
            Some("fact".to_string()),
            None,
            None,
        )
        .unwrap(),
        LintAgentRecord::try_new(
            2,
            LintAgentRecordKind::Entity,
            "Project Atlas".to_string(),
            None,
            None,
            None,
        )
        .unwrap(),
    ];
    let populations = LintSemanticCheckId::ALL
        .into_iter()
        .map(|check_id| {
            if check_id == LintSemanticCheckId::MemoryEntityLinks {
                LintSemanticPopulation::try_new(check_id, 20, 3, 1, true)
            } else {
                LintSemanticPopulation::try_new(check_id, 0, 0, 0, false)
            }
            .unwrap()
        })
        .collect();
    let candidate = LintAgentCandidate::try_new(
        1,
        LintSemanticCheckId::MemoryEntityLinks,
        LintSemanticCandidateKind::MissingLink,
        vec![1, 2],
        vec![],
        LintSemanticAction::AddMemoryEntityLink,
        LintSemanticReasonCode::MentionWithoutLink,
    )
    .unwrap();
    let work = LintAgentWork::try_new(
        LintDigest::from_u64(10),
        populations,
        records,
        vec![candidate],
    )
    .unwrap();
    assert_eq!(work.candidates().len(), 1);
    assert!(work
        .populations()
        .iter()
        .find(|population| population.check_id() == LintSemanticCheckId::MemoryEntityLinks)
        .unwrap()
        .truncated());

    let verdict = LintAgentVerdict::try_new(
        1,
        LintSemanticDecision::Finding,
        Some(LintSemanticDecision::Pass),
        LintSemanticReasonCode::MentionWithoutLink,
        9_200,
        vec![],
    )
    .unwrap();
    assert!(verdict.has_unresolved_disagreement());
    let submission =
        LintAgentSubmission::try_new(work.work_digest().clone(), vec![verdict]).unwrap();
    let decoded: LintAgentSubmission =
        serde_json::from_value(serde_json::to_value(&submission).unwrap()).unwrap();
    assert_eq!(decoded, submission);

    let finding = LintSemanticFinding::try_new(
        LintOpaqueId::from_sorted_position(1).unwrap(),
        LintSemanticAction::AddMemoryEntityLink,
        LintSemanticReasonCode::MentionWithoutLink,
        9_200,
        LintSemanticProviderRoute::CallingAgent,
        vec![LintDigest::from_u64(1)],
        vec![],
    )
    .unwrap();
    let evidence = LintEvidenceRef::SemanticFinding { finding };
    let round_trip: LintEvidenceRef =
        serde_json::from_value(serde_json::to_value(&evidence).unwrap()).unwrap();
    assert_eq!(round_trip, evidence);
}

#[test]
fn check_group_resolver_uses_the_canonical_seven_groups() {
    assert_eq!(LintCheckGroup::ALL.len(), 7);
    assert_eq!(
        LintCheckGroup::for_check_id("memory_entities.integrity"),
        Some(LintCheckGroup::KnowledgeGraph)
    );
    assert_eq!(
        LintCheckGroup::for_check_id("pages.projection.identity"),
        Some(LintCheckGroup::Pages)
    );
    assert_eq!(LintCheckGroup::for_check_id("fixture.padding"), None);
}

#[test]
fn full_enumeration_rejects_partial_population() {
    let coverage = LintCoverage::new(
        LintValidationMethod::FullEnumeration,
        10,
        1,
        LINT_MAX_EVIDENCE_PER_CHECK,
        false,
        0,
    );

    assert_eq!(coverage, Err(LintContractError::InvalidCoverage));
}

#[test]
fn full_enumeration_accepts_complete_and_legitimately_empty_populations() {
    for (authorized_denominator, evaluated) in [(10, 10), (0, 0)] {
        let coverage = LintCoverage::new(
            LintValidationMethod::FullEnumeration,
            authorized_denominator,
            evaluated,
            LINT_MAX_EVIDENCE_PER_CHECK,
            false,
            0,
        );

        assert!(coverage.is_ok());
    }
}

#[test]
fn intrinsic_sample_preserves_partial_coverage() {
    let coverage = LintCoverage::new(
        LintValidationMethod::IntrinsicSample,
        10,
        1,
        LINT_MAX_EVIDENCE_PER_CHECK,
        false,
        0,
    );

    assert!(coverage.is_ok());
}

#[test]
fn coverage_deserialization_rejects_partial_full_enumeration() {
    let result = serde_json::from_value::<LintCoverage>(json!({
        "method": "full_enumeration",
        "authorized_denominator": 10,
        "evaluated": 1,
        "evidence_cap": 100,
        "truncated": false,
        "evidence_returned": 0
    }));

    let error = result.expect_err("partial full enumeration must fail typed deserialization");
    assert_eq!(error.to_string(), "invalid_lint_coverage");
}

#[test]
fn coverage_deserialization_preserves_valid_methods_and_additive_fields() {
    let cases = [
        ("full_enumeration", 10, 10),
        ("full_enumeration", 0, 0),
        ("exact_aggregate", 10, 1),
        ("intrinsic_sample", 10, 1),
    ];

    for (method, authorized_denominator, evaluated) in cases {
        let coverage = serde_json::from_value::<LintCoverage>(json!({
            "method": method,
            "authorized_denominator": authorized_denominator,
            "evaluated": evaluated,
            "evidence_cap": 100,
            "truncated": false,
            "evidence_returned": 0,
            "future_coverage_field": { "additive": true }
        }));

        assert!(coverage.is_ok());
    }
}

const ALL_SUMMARY_CODES: [LintSummaryCode; 6] = [
    LintSummaryCode::CheckPassed,
    LintSummaryCode::FindingDetected,
    LintSummaryCode::PrerequisiteUnavailable,
    LintSummaryCode::SnapshotInconsistent,
    LintSummaryCode::ExecutionFailed,
    LintSummaryCode::ExpectedEmpty,
];

const ALL_RECOMMENDATION_CODES: [LintRecommendationCode; 4] = [
    LintRecommendationCode::ReviewFinding,
    LintRecommendationCode::RestorePrerequisite,
    LintRecommendationCode::RerunAfterSnapshotStabilizes,
    LintRecommendationCode::InspectRuntime,
];

const ALL_ACTION_CODES: [LintActionCode; 1] = [LintActionCode::ChooseModelSource];

// The match arms are exhaustive by the compiler. What a test has to hold is
// that nobody satisfied the compiler with a shared placeholder: every code says
// its own thing, in a full sentence, so a rendered line reads as English.
#[test]
fn every_summary_and_recommendation_code_carries_its_own_plain_sentence() {
    let meanings = ALL_SUMMARY_CODES
        .iter()
        .map(|code| code.meaning())
        .collect::<Vec<_>>();
    let actions = ALL_RECOMMENDATION_CODES
        .iter()
        .map(|code| code.action())
        .chain(ALL_ACTION_CODES.iter().map(|code| code.action()))
        .collect::<Vec<_>>();

    for sentence in meanings.iter().chain(actions.iter()) {
        assert!(sentence.len() > 20, "{sentence:?} is not a sentence");
        assert!(sentence.ends_with('.'), "{sentence:?} is not a sentence");
        assert!(
            !sentence.to_ascii_lowercase().contains("unknown"),
            "{sentence:?} is a fallback, not a sentence"
        );
    }

    assert_eq!(
        meanings
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        meanings.len()
    );
    assert_eq!(
        actions
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        actions.len()
    );
    assert!(LintActionCode::ChooseModelSource
        .action()
        .contains("model source"));
}

/// The fresh-store shape: a passing, expected-empty check that still has one
/// thing to ask of the owner. Nothing is wrong, so `recommendation_code` stays
/// empty; the ask travels in the additive `action_code` field.
fn fresh_store_check() -> LintCheckResult {
    LintCheckResult::try_new(LintCheckResultInput {
        check_id: "kg.substrate_liveness".to_string(),
        outcome: LintOutcome::Pass,
        severity: LintSeverity::Info,
        applicability: LintApplicability::ExpectedEmpty,
        precondition: LintPrecondition::ExpectedEmpty,
        coverage: coverage(0),
        metrics: vec![],
        summary_code: LintSummaryCode::ExpectedEmpty,
        recommendation_code: None,
        evidence: vec![],
        duration_ms: 0,
    })
    .expect("expected-empty pass with an action is a legal check result")
    .with_action_code(Some(LintActionCode::ChooseModelSource))
}

// A passing, expected-empty check is allowed to carry advice: nothing is wrong,
// the owner simply has to choose a model source before the substrate can exist.
// This shape must stay legal and must round-trip on the wire.
#[test]
fn expected_empty_pass_may_carry_a_choose_model_source_action() {
    let result = fresh_store_check();

    let encoded = serde_json::to_value(&result).unwrap();
    assert_eq!(encoded["action_code"], json!("choose_model_source"));
    assert_eq!(encoded["precondition"], json!("expected_empty"));
    // The frozen schema-5 field stays empty: a pass is not a recommendation.
    assert!(encoded
        .get("recommendation_code")
        .is_none_or(serde_json::Value::is_null));
    assert_eq!(
        serde_json::from_value::<LintCheckResult>(encoded).unwrap(),
        result
    );
}

// A check with nothing to ask of the owner must serialize exactly as it did
// before `action_code` existed -- the key is absent, not null.
#[test]
fn a_check_without_an_action_does_not_grow_a_wire_key() {
    let result = LintCheckResult::try_new(LintCheckResultInput {
        check_id: "kg.substrate_liveness".to_string(),
        outcome: LintOutcome::Pass,
        severity: LintSeverity::Info,
        applicability: LintApplicability::ExpectedEmpty,
        precondition: LintPrecondition::ConfiguredOff,
        coverage: coverage(0),
        metrics: vec![],
        summary_code: LintSummaryCode::ExpectedEmpty,
        recommendation_code: None,
        evidence: vec![],
        duration_ms: 0,
    })
    .expect("configured-off pass is a legal check result");

    let encoded = serde_json::to_value(&result).unwrap();
    assert!(
        encoded.get("action_code").is_none(),
        "absent action must not appear on the wire: {encoded}"
    );
}

/// A byte-for-byte copy of the `recommendation_code` enum as it stands in the
/// released schema-5 wire shape (base `43c00470`). Frozen on purpose: it must
/// NOT gain variants when the live enum does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum FrozenRecommendationCode {
    ReviewFinding,
    RestorePrerequisite,
    RerunAfterSnapshotStabilizes,
    InspectRuntime,
}

/// A copy of `LintCheckResultInput` as an older CLI or MCP client parses it.
/// Like the real one it does not deny unknown fields, so an added field is
/// ignored -- but serde rejects an unknown *enum variant*, which is exactly the
/// break this test exists to catch.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct FrozenReaderCheck {
    check_id: String,
    outcome: LintOutcome,
    severity: LintSeverity,
    applicability: LintApplicability,
    precondition: LintPrecondition,
    coverage: LintCoverage,
    metrics: Vec<LintMetric>,
    summary_code: LintSummaryCode,
    recommendation_code: Option<FrozenRecommendationCode>,
    evidence: Vec<LintEvidenceRef>,
    duration_ms: u64,
}

// Patch-ahead daemons are explicitly compatible with older clients
// (`wenlan-mcp/src/version_check.rs`), so a daemon on this build must not emit
// a report that a client on the released schema-5 shape cannot parse. Putting
// `choose_model_source` into `recommendation_code` would do exactly that.
#[test]
fn a_frozen_schema_five_reader_still_parses_the_fresh_store_check() {
    let encoded = serde_json::to_value(fresh_store_check()).unwrap();
    assert_eq!(encoded["action_code"], json!("choose_model_source"));

    let old = serde_json::from_value::<FrozenReaderCheck>(encoded)
        .expect("a released schema-5 reader must still parse this check");

    assert_eq!(old.check_id, "kg.substrate_liveness");
    assert_eq!(old.outcome, LintOutcome::Pass);
    assert_eq!(old.precondition, LintPrecondition::ExpectedEmpty);
    assert_eq!(old.summary_code, LintSummaryCode::ExpectedEmpty);
    // The old reader simply does not see the action; it does not choke on it.
    assert_eq!(old.recommendation_code, None);
}

/// The two checks a fresh store with no model source used to fail on, as the
/// daemon now produces them: complete, passing, expected-empty, each asking
/// the owner to pick a model source.
fn fresh_store_report() -> LintReport {
    report(
        LintScope::global(),
        vec![
            model_source_check("kg.substrate_liveness"),
            model_source_check("serving.channel.graph"),
        ],
    )
}

fn model_source_check(check_id: &str) -> LintCheckResult {
    LintCheckResult::try_new(LintCheckResultInput {
        check_id: check_id.to_string(),
        outcome: LintOutcome::Pass,
        severity: LintSeverity::Info,
        applicability: LintApplicability::ExpectedEmpty,
        precondition: LintPrecondition::ExpectedEmpty,
        coverage: coverage(0),
        metrics: vec![],
        summary_code: LintSummaryCode::ExpectedEmpty,
        recommendation_code: None,
        evidence: vec![],
        duration_ms: 0,
    })
    .expect("expected-empty pass is a legal check result")
    .with_action_code(Some(LintActionCode::ChooseModelSource))
}

#[test]
fn the_model_source_checks_render_with_a_plain_sentence_and_no_findings() {
    let rendered = fresh_store_report().render_text();

    assert!(rendered.contains("0 actionable findings"), "{rendered}");
    assert!(rendered.contains("Findings: none\n"), "{rendered}");
    assert!(rendered.contains("Waiting on you (2):"), "{rendered}");
    for check_id in ["kg.substrate_liveness", "serving.channel.graph"] {
        assert!(
            rendered.contains(&format!(
                "  {check_id}: expected_empty; action: choose_model_source\n"
            )),
            "{rendered}"
        );
    }
    assert!(
        rendered.contains(
            "    There was nothing here to check, which is the expected state right now. \
             Run `wenlan setup` and choose a model source so Wenlan can build this in the \
             background.\n"
        ),
        "{rendered}"
    );
}

// Every rendered line carries the sentence, not just the new one -- the
// codes stay for scripts, the English is for the person reading.
#[test]
fn a_rendered_finding_carries_both_the_code_and_the_plain_sentence() {
    let rendered = report(
        LintScope::global(),
        vec![check_with_id(
            "pages.projection.identity",
            LintOutcome::Finding,
            LintSeverity::Error,
        )
        .unwrap()],
    )
    .render_text();

    assert!(
        rendered.contains(
            "  pages.projection.identity: finding_detected; recommendation: review_finding\n"
        ),
        "{rendered}"
    );
    assert!(
        rendered.contains(
            "    This check found records that do not match what Wenlan expects. \
             Look at the listed records and fix or dismiss them.\n"
        ),
        "{rendered}"
    );
    assert!(rendered.contains("Waiting on you: none\n"), "{rendered}");
}
