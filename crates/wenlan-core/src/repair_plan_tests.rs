// SPDX-License-Identifier: Apache-2.0
use super::{
    build_repair_plan, deterministic_resolution, prepare_repair_plan, semantic_action_route,
    SemanticActionRoute,
};
use crate::lint::catalog::catalog_for_profile;
use wenlan_types::{
    lint::{
        canonical_gate_effect, LintApplicability, LintCapabilityContext, LintCheckResult,
        LintCheckResultInput, LintConfigFingerprint, LintCoverage, LintDbSnapshotMode,
        LintDbSnapshotReceipt, LintDigest, LintEvidenceRef, LintGateEffect, LintMetric,
        LintOpaqueDigest, LintOutcome, LintPageSnapshotMode, LintPageSnapshotReceipt,
        LintPrecondition, LintProducerReceipt, LintProfile, LintRecommendationCode, LintReport,
        LintScope, LintSemanticAction, LintSeverity, LintSnapshotReceipts, LintSummaryCode,
        LintValidationMethod, LINT_GENERAL_CHECK_COUNT, LINT_MAX_EVIDENCE_PER_CHECK,
    },
    repair::{RepairDigest, RepairLintScope},
    repair_plan::{
        RepairBlocked, RepairBlockedReasonCode, RepairFindingKind, RepairPlan, RepairPlanDraft,
        RepairPlanEntriesRequest, RepairPlanEntry, RepairPlanReportReceipt, RepairPlanRequest,
        RepairResolution, RepairSystemActionKind, REPAIR_PLAN_PAGE_MAX_BYTES,
    },
};

fn snapshots(seed: u64) -> LintSnapshotReceipts {
    LintSnapshotReceipts::new(
        LintDbSnapshotReceipt::new(
            LintDbSnapshotMode::TransactionalReadOnly,
            LintDigest::from_u64(seed),
            Some(LintDigest::from_u64(seed)),
        ),
        LintPageSnapshotReceipt::new(
            LintPageSnapshotMode::BestEffort,
            LintDigest::from_u64(seed + 1),
            Some(LintDigest::from_u64(seed + 1)),
        ),
    )
}

fn result(profile: LintProfile, check_id: &str, outcome: LintOutcome) -> LintCheckResult {
    let (severity, applicability, precondition, summary, recommendation) = match outcome {
        LintOutcome::Pass => (
            LintSeverity::Info,
            LintApplicability::Applicable,
            LintPrecondition::Ready,
            LintSummaryCode::CheckPassed,
            None,
        ),
        LintOutcome::Finding => (
            LintSeverity::Warning,
            LintApplicability::Applicable,
            LintPrecondition::Ready,
            LintSummaryCode::FindingDetected,
            Some(LintRecommendationCode::ReviewFinding),
        ),
        LintOutcome::NotRunPrerequisite => (
            LintSeverity::Error,
            LintApplicability::NotApplicable,
            LintPrecondition::MissingPrerequisite,
            LintSummaryCode::PrerequisiteUnavailable,
            Some(LintRecommendationCode::RestorePrerequisite),
        ),
        LintOutcome::InconsistentSnapshot => (
            LintSeverity::Error,
            LintApplicability::Applicable,
            LintPrecondition::SnapshotUnstable,
            LintSummaryCode::SnapshotInconsistent,
            Some(LintRecommendationCode::RerunAfterSnapshotStabilizes),
        ),
        LintOutcome::FailedToRun => (
            LintSeverity::Error,
            LintApplicability::Applicable,
            LintPrecondition::Ready,
            LintSummaryCode::ExecutionFailed,
            Some(LintRecommendationCode::InspectRuntime),
        ),
    };
    LintCheckResult::try_new_with_gate_effect(
        LintCheckResultInput {
            check_id: check_id.to_string(),
            outcome,
            severity,
            applicability,
            precondition,
            coverage: LintCoverage::new(
                LintValidationMethod::FullEnumeration,
                0,
                0,
                LINT_MAX_EVIDENCE_PER_CHECK,
                false,
                0,
            )
            .unwrap(),
            metrics: Vec::<LintMetric>::new(),
            summary_code: summary,
            recommendation_code: recommendation,
            evidence: Vec::<LintEvidenceRef>::new(),
            duration_ms: 0,
        },
        canonical_gate_effect(profile, check_id).unwrap_or(LintGateEffect::Actionable),
    )
    .unwrap()
}

fn report(profile: LintProfile, outcomes: &[(&str, LintOutcome)], seed: u64) -> LintReport {
    let checks = catalog_for_profile(profile)
        .map(|entry| {
            let outcome = outcomes
                .iter()
                .find_map(|(check_id, outcome)| (*check_id == entry.id).then_some(*outcome))
                .unwrap_or(LintOutcome::Pass);
            result(profile, entry.id, outcome)
        })
        .collect();
    LintReport::try_new_for_profile(
        profile,
        LintScope::global(),
        LintCapabilityContext::daemon_operator_endpoint(),
        snapshots(seed),
        LintConfigFingerprint::from_effective_config(&[]),
        LintProducerReceipt::new(None),
        checks,
    )
    .unwrap()
}

fn byte_heavy_blocked_plan(plan_id: &str, entry_count: usize, detail_bytes: usize) -> RepairPlan {
    let entries = (0..entry_count)
        .map(|index| {
            RepairPlanEntry::try_new(
                RepairFindingKind::Deterministic,
                "identity.registry_integrity".to_string(),
                RepairDigest::parse(&format!("{:064x}", index + 1)).unwrap(),
                vec![],
                RepairResolution::Blocked {
                    blocked: RepairBlocked::try_new(
                        RepairBlockedReasonCode::UnsupportedDeterministicWriter,
                        "x".repeat(detail_bytes),
                        "add a typed adapter".to_string(),
                    )
                    .unwrap(),
                },
            )
            .unwrap()
        })
        .collect();
    let general = report(LintProfile::General, &[], 1);
    let draft = RepairPlanDraft::try_new(
        plan_id.to_string(),
        RepairLintScope::global(),
        RepairPlanReportReceipt::from_report(&general),
        None,
        true,
        false,
        entries,
    )
    .unwrap();
    let digest = crate::repair::repair_digest(&draft.canonical_bytes().unwrap());
    RepairPlan::try_new(draft, digest, |canonical, expected| {
        crate::repair::repair_digest(canonical) == *expected
    })
    .unwrap()
}

#[test]
fn plan_entry_pages_are_byte_bounded_and_preserve_all_entries() {
    let plan = byte_heavy_blocked_plan(
        "repair_plan_550e8400-e29b-41d4-a716-446655440001",
        8,
        20 * 1024,
    );
    let repair_root = tempfile::tempdir().unwrap();
    let store = crate::repair::RepairArtifactStore::new(repair_root.path().to_path_buf());
    store.persist_plan(&plan).unwrap();

    let mut offset = 0;
    let mut gathered = Vec::new();
    loop {
        let page = store
            .load_plan_entries_page(
                &RepairPlanEntriesRequest::try_new(
                    plan.plan_id().to_string(),
                    plan.plan_digest().clone(),
                    offset,
                    100,
                )
                .unwrap(),
            )
            .unwrap();
        let serialized = serde_json::to_vec(&page).unwrap();
        assert!(serialized.len() <= REPAIR_PLAN_PAGE_MAX_BYTES);
        assert!(!page.entries().is_empty());
        gathered.extend(
            page.entries()
                .iter()
                .map(|entry| entry.occurrence_digest().as_str().to_string()),
        );
        let value = serde_json::to_value(&page).unwrap();
        let Some(next_offset) = value["next_offset"].as_u64() else {
            break;
        };
        let next_offset = usize::try_from(next_offset).unwrap();
        assert!(next_offset > offset);
        offset = next_offset;
    }

    assert_eq!(
        gathered,
        plan.entries()
            .iter()
            .map(|entry| entry.occurrence_digest().as_str().to_string())
            .collect::<Vec<_>>()
    );
}

#[test]
fn single_oversized_plan_entry_fails_closed() {
    let plan = byte_heavy_blocked_plan(
        "repair_plan_550e8400-e29b-41d4-a716-446655440002",
        1,
        REPAIR_PLAN_PAGE_MAX_BYTES,
    );
    let repair_root = tempfile::tempdir().unwrap();
    let store = crate::repair::RepairArtifactStore::new(repair_root.path().to_path_buf());
    let final_path = store.plan_path(plan.plan_id()).unwrap();

    let error = store.persist_plan(&plan).unwrap_err();
    assert!(matches!(
        error,
        crate::error::WenlanError::Validation(message)
            if message == "repair_plan_entry_too_large"
    ));
    assert!(!final_path.exists());
}

#[test]
fn oversized_plan_artifact_is_not_published() {
    let plan = byte_heavy_blocked_plan(
        "repair_plan_550e8400-e29b-41d4-a716-446655440003",
        400,
        44 * 1024,
    );
    let repair_root = tempfile::tempdir().unwrap();
    let store = crate::repair::RepairArtifactStore::new(repair_root.path().to_path_buf());
    let final_path = store.plan_path(plan.plan_id()).unwrap();

    let error = store.persist_plan(&plan).unwrap_err();
    assert!(matches!(
        error,
        crate::error::WenlanError::Validation(message)
            if message == "repair_plan_artifact_too_large"
    ));
    assert!(!final_path.exists());
}

fn with_semantic_finding(
    report: LintReport,
    check_id: &str,
    finding: wenlan_types::lint::LintSemanticFinding,
) -> LintReport {
    let mut checks = report.checks().to_vec();
    let target = checks
        .iter_mut()
        .find(|check| check.check_id() == check_id)
        .expect("semantic check in Deep catalog");
    *target = LintCheckResult::try_new_with_gate_effect(
        LintCheckResultInput {
            check_id: check_id.to_string(),
            outcome: LintOutcome::Finding,
            severity: LintSeverity::Warning,
            applicability: LintApplicability::Applicable,
            precondition: LintPrecondition::Ready,
            coverage: LintCoverage::new(
                LintValidationMethod::FullEnumeration,
                1,
                1,
                LINT_MAX_EVIDENCE_PER_CHECK,
                false,
                1,
            )
            .unwrap(),
            metrics: vec![],
            summary_code: LintSummaryCode::FindingDetected,
            recommendation_code: Some(LintRecommendationCode::ReviewFinding),
            evidence: vec![LintEvidenceRef::SemanticFinding { finding }],
            duration_ms: 0,
        },
        canonical_gate_effect(LintProfile::Deep, check_id).unwrap(),
    )
    .unwrap();
    LintReport::try_new_for_profile(
        LintProfile::Deep,
        report.scope().clone(),
        report.capability_context(),
        report.snapshots().clone(),
        report.config_fingerprint().clone(),
        report.producer_receipt().clone(),
        checks,
    )
    .unwrap()
}

fn with_completed_agent_work(report: LintReport) -> LintReport {
    let populations = wenlan_types::lint::LintSemanticCheckId::ALL
        .into_iter()
        .map(|check_id| {
            wenlan_types::lint::LintSemanticPopulation::try_new(check_id, 0, 0, 0, false).unwrap()
        })
        .collect();
    LintReport::try_new_for_profile_with_agent_work(
        report.profile(),
        report.scope().clone(),
        report.capability_context(),
        report.snapshots().clone(),
        report.config_fingerprint().clone(),
        report.producer_receipt().clone(),
        report.checks().to_vec(),
        Some(
            wenlan_types::lint::LintAgentWork::try_new(
                LintDigest::from_u64(9_001),
                populations,
                vec![],
                vec![],
            )
            .unwrap(),
        ),
    )
    .unwrap()
}

fn with_large_non_target_evidence(report: LintReport) -> LintReport {
    let mut checks = report.checks().to_vec();
    for (check_index, target) in checks
        .iter_mut()
        .filter(|check| {
            check.outcome() == LintOutcome::Pass
                && check.applicability() == LintApplicability::Applicable
        })
        .take(6)
        .enumerate()
    {
        let evidence = (0..usize::from(LINT_MAX_EVIDENCE_PER_CHECK))
            .map(|evidence_index| LintEvidenceRef::OpaqueDigest {
                opaque_digest: LintOpaqueDigest::from_hex(&format!(
                    "{:064x}",
                    1 + (check_index * usize::from(LINT_MAX_EVIDENCE_PER_CHECK)) + evidence_index
                ))
                .unwrap(),
            })
            .collect::<Vec<_>>();
        *target = LintCheckResult::try_new_with_gate_effect(
            LintCheckResultInput {
                check_id: target.check_id().to_string(),
                outcome: target.outcome(),
                severity: target.severity(),
                applicability: target.applicability(),
                precondition: target.precondition(),
                coverage: LintCoverage::new(
                    LintValidationMethod::FullEnumeration,
                    u64::from(LINT_MAX_EVIDENCE_PER_CHECK),
                    u64::from(LINT_MAX_EVIDENCE_PER_CHECK),
                    LINT_MAX_EVIDENCE_PER_CHECK,
                    false,
                    u64::from(LINT_MAX_EVIDENCE_PER_CHECK),
                )
                .unwrap(),
                metrics: target.metrics().to_vec(),
                summary_code: target.summary_code(),
                recommendation_code: target.recommendation_code(),
                evidence,
                duration_ms: target.duration_ms(),
            },
            target.gate_effect(),
        )
        .unwrap();
    }
    LintReport::try_new_for_profile_with_agent_work(
        report.profile(),
        report.scope().clone(),
        report.capability_context(),
        report.snapshots().clone(),
        report.config_fingerprint().clone(),
        report.producer_receipt().clone(),
        checks,
        report.agent_work().cloned(),
    )
    .unwrap()
}

fn with_check_outcome(report: LintReport, check_id: &str, outcome: LintOutcome) -> LintReport {
    let mut checks = report.checks().to_vec();
    let target = checks
        .iter_mut()
        .find(|check| check.check_id() == check_id)
        .expect("check in report catalog");
    *target = result(report.profile(), check_id, outcome);
    LintReport::try_new_for_profile(
        report.profile(),
        report.scope().clone(),
        report.capability_context(),
        report.snapshots().clone(),
        report.config_fingerprint().clone(),
        report.producer_receipt().clone(),
        checks,
    )
    .unwrap()
}

#[test]
fn all_general_findings_have_a_visible_resolution() {
    let outcomes = catalog_for_profile(LintProfile::General)
        .map(|entry| (entry.id, LintOutcome::Finding))
        .collect::<Vec<_>>();
    let request = RepairPlanRequest::try_new(
        RepairLintScope::global(),
        report(LintProfile::General, &outcomes, 1),
        None,
    )
    .unwrap();

    let plan = build_repair_plan(request).unwrap();
    assert_eq!(plan.entries().len(), LINT_GENERAL_CHECK_COUNT);
    assert_eq!(
        plan.totals().deterministic(),
        u64::try_from(LINT_GENERAL_CHECK_COUNT).unwrap()
    );
    assert_eq!(plan.totals().semantic(), 0);
    assert!(plan.deterministic_complete());
    assert!(!plan.semantic_complete());
}

#[test]
fn unsupported_deterministic_writer_is_blocked_not_filtered() {
    let request = RepairPlanRequest::try_new(
        RepairLintScope::global(),
        report(
            LintProfile::General,
            &[("identity.registry_integrity", LintOutcome::Finding)],
            1,
        ),
        None,
    )
    .unwrap();

    let plan = build_repair_plan(request).unwrap();
    assert_eq!(plan.entries().len(), 1);
    assert_eq!(plan.entries()[0].check_id(), "identity.registry_integrity");
    assert!(matches!(
        plan.entries()[0].resolution(),
        RepairResolution::Blocked { blocked }
            if blocked.reason_code() == RepairBlockedReasonCode::UnsupportedDeterministicWriter
    ));
}

#[test]
fn scoped_tag_finding_directs_repair_to_global_scope() {
    let resolution = deterministic_resolution(
        &result(
            LintProfile::General,
            "identity.tag_integrity",
            LintOutcome::Finding,
        ),
        &RepairLintScope::registered("work".to_string()).unwrap(),
    )
    .unwrap();

    assert!(matches!(
        resolution,
        RepairResolution::Blocked { blocked }
            if blocked.reason_code() == RepairBlockedReasonCode::MissingPrerequisite
                && blocked.next_action().contains("/lint repair global")
    ));
}

#[test]
fn incomplete_semantic_deep_never_hides_complete_deterministic_entries() {
    let general = report(
        LintProfile::General,
        &[("identity.registry_integrity", LintOutcome::Finding)],
        1,
    );
    let deep = report(
        LintProfile::Deep,
        &[(
            "memories.semantic.classification",
            LintOutcome::NotRunPrerequisite,
        )],
        3,
    );
    let request =
        RepairPlanRequest::try_new(RepairLintScope::global(), general, Some(deep)).unwrap();

    let plan = build_repair_plan(request).unwrap();
    assert!(plan.deterministic_complete());
    assert!(!plan.semantic_complete());
    assert!(plan
        .entries()
        .iter()
        .any(|entry| entry.check_id() == "identity.registry_integrity"));
    assert!(plan.entries().iter().any(|entry| {
        entry.check_id() == "memories.semantic.classification"
            && matches!(
                entry.resolution(),
                RepairResolution::Blocked { blocked }
                    if blocked.reason_code() == RepairBlockedReasonCode::SourceIncomplete
            )
    }));
}

#[test]
fn deep_deterministic_finding_is_not_hidden_by_general_pass() {
    let general = report(LintProfile::General, &[], 1);
    let deep = report(
        LintProfile::Deep,
        &[("identity.registry_integrity", LintOutcome::Finding)],
        3,
    );
    let request =
        RepairPlanRequest::try_new(RepairLintScope::global(), general, Some(deep)).unwrap();

    let plan = build_repair_plan(request).unwrap();

    assert!(plan.entries().iter().any(|entry| {
        entry.check_id() == "identity.registry_integrity"
            && matches!(
                entry.resolution(),
                RepairResolution::Blocked { blocked }
                    if blocked.reason_code()
                        == RepairBlockedReasonCode::UnsupportedDeterministicWriter
            )
    }));
}

#[test]
fn schema_and_route_contracts_are_system_actions_not_data_manifests() {
    let request = RepairPlanRequest::try_new(
        RepairLintScope::global(),
        report(
            LintProfile::General,
            &[
                ("runtime.schema_contract", LintOutcome::Finding),
                ("serving.route_scope_contracts", LintOutcome::Finding),
            ],
            1,
        ),
        None,
    )
    .unwrap();

    let plan = build_repair_plan(request).unwrap();
    let schema = plan
        .entries()
        .iter()
        .find(|entry| entry.check_id() == "runtime.schema_contract")
        .unwrap();
    assert!(matches!(
        schema.resolution(),
        RepairResolution::Blocked { blocked }
            if blocked.reason_code() == RepairBlockedReasonCode::MissingPrerequisite
    ));
    let route = plan
        .entries()
        .iter()
        .find(|entry| entry.check_id() == "serving.route_scope_contracts")
        .unwrap();
    assert!(matches!(
        route.resolution(),
        RepairResolution::SystemAction { system_action }
            if matches!(
                system_action.kind(),
                RepairSystemActionKind::CorrectRouteScopeContract
                    | RepairSystemActionKind::UpdateDaemon
            ) && !system_action.evidence().is_empty()
    ));
}

#[tokio::test]
async fn current_schema_diagnostic_distinguishes_old_and_future_versions() {
    use crate::lint::{
        context::{CancellationToken, LintClock},
        runner::LintRunner,
    };
    use wenlan_types::lint::LintQuery;

    for (version, expected_kind, expected_blocked) in [
        (
            i64::from(crate::db::SCHEMA_VERSION) - 1,
            Some(RepairSystemActionKind::RunSchemaMigration),
            None,
        ),
        (
            i64::from(crate::db::SCHEMA_VERSION) + 1,
            None,
            Some(RepairBlockedReasonCode::UnknownSchemaShape),
        ),
    ] {
        let (db, _dir) = crate::db::tests::test_db().await;
        db.test_primary_session()
            .await
            .execute(&format!("PRAGMA user_version={version}"), ())
            .await
            .unwrap();
        let general = LintRunner::new(LintClock::fixed(), CancellationToken::new())
            .run(
                &db,
                &LintQuery::new(Some(LintProfile::General), None),
                None,
                false,
            )
            .await
            .unwrap();
        let repair_root = tempfile::tempdir().unwrap();
        let store = crate::repair::RepairArtifactStore::new(repair_root.path().to_path_buf());
        let plan = prepare_repair_plan(
            &db,
            &store,
            RepairPlanRequest::try_new(RepairLintScope::global(), general, None).unwrap(),
            None,
            1_721_000_000,
        )
        .await
        .unwrap();
        let schema = plan
            .entries()
            .iter()
            .find(|entry| entry.check_id() == "runtime.schema_contract")
            .unwrap();
        match (expected_kind, expected_blocked, schema.resolution()) {
            (Some(expected), None, RepairResolution::SystemAction { system_action }) => {
                assert_eq!(system_action.kind(), expected)
            }
            (None, Some(expected), RepairResolution::Blocked { blocked }) => {
                assert_eq!(blocked.reason_code(), expected)
            }
            (_, _, other) => panic!("unexpected schema resolution: {other:?}"),
        }
    }
}

#[tokio::test]
async fn missing_search_object_selects_the_canonical_rebuild_action() {
    use crate::lint::{
        context::{CancellationToken, LintClock},
        runner::LintRunner,
    };
    use wenlan_types::lint::LintQuery;

    let (db, _dir) = crate::db::tests::test_db().await;
    db.test_primary_session()
        .await
        .execute("DROP TRIGGER memories_fts_update", ())
        .await
        .unwrap();
    let general = LintRunner::new(LintClock::fixed(), CancellationToken::new())
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::General), None),
            None,
            false,
        )
        .await
        .unwrap();
    let repair_root = tempfile::tempdir().unwrap();
    let store = crate::repair::RepairArtifactStore::new(repair_root.path().to_path_buf());
    let plan = prepare_repair_plan(
        &db,
        &store,
        RepairPlanRequest::try_new(RepairLintScope::global(), general, None).unwrap(),
        None,
        1_721_000_000,
    )
    .await
    .unwrap();
    let search = plan
        .entries()
        .iter()
        .find(|entry| entry.check_id() == "runtime.search_index_contract")
        .unwrap();

    assert!(matches!(
        search.resolution(),
        RepairResolution::SystemAction { system_action }
            if system_action.kind() == RepairSystemActionKind::RebuildSearchIndex
                && system_action
                    .evidence()
                    .iter()
                    .any(|item| item.contains("memories_fts_update"))
    ));
}

#[tokio::test]
async fn general_only_plan_refuses_a_stale_source_report() {
    use crate::lint::{
        context::{CancellationToken, LintClock},
        runner::LintRunner,
    };
    use wenlan_types::lint::LintQuery;

    let (db, _dir) = crate::db::tests::test_db().await;
    let general = LintRunner::new(LintClock::fixed(), CancellationToken::new())
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::General), None),
            None,
            false,
        )
        .await
        .unwrap();
    db.test_primary_session()
        .await
        .execute(
            "INSERT INTO spaces (id,name,created_at,updated_at) VALUES ('stale','stale',1,1)",
            (),
        )
        .await
        .unwrap();
    let repair_root = tempfile::tempdir().unwrap();
    let store = crate::repair::RepairArtifactStore::new(repair_root.path().to_path_buf());

    let error = prepare_repair_plan(
        &db,
        &store,
        RepairPlanRequest::try_new(RepairLintScope::global(), general, None).unwrap(),
        None,
        1_721_000_000,
    )
    .await
    .unwrap_err();

    assert!(error.to_string().contains("repair_source_reports_stale"));
    assert!(!repair_root.path().join("plans").exists());
}

#[test]
fn every_semantic_action_has_an_explicit_resolution_route() {
    use LintSemanticAction::*;

    let routes = [
        ReclassifyMemory,
        ReviewContradiction,
        ReviewStaleness,
        SupersedeMemory,
        AddMemoryEntityLink,
        RemoveMemoryEntityLink,
        AddEntityRelation,
        RemoveEntityRelation,
        ReviewPageClaim,
        AddPageEvidence,
        RemovePageEvidence,
        ReviewRetrieval,
    ]
    .map(semantic_action_route);
    assert_eq!(routes[0], SemanticActionRoute::ExactCandidate);
    assert_eq!(
        routes
            .iter()
            .filter(|route| **route == SemanticActionRoute::Review)
            .count(),
        4
    );
    assert_eq!(
        routes
            .iter()
            .filter(|route| **route == SemanticActionRoute::MutationCandidate)
            .count(),
        7
    );
}

#[tokio::test]
async fn lint_review_enqueue_is_idempotent_and_does_not_resurrect_terminal_items() {
    let (db, _dir) = crate::db::tests::test_db().await;
    let review_id = format!(
        "lint_review_{}",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    );
    let source_ids = vec!["mem_a".to_string()];
    let payload = serde_json::json!({
        "action": "lint_repair_review",
        "check_id": "identity.memory_state_integrity",
        "occurrence_digest": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "owner_binding_digest": crate::repair::lint_review_owner_binding_digest(
            &wenlan_types::repair::RepairDigest::parse(
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            )
            .unwrap(),
            &source_ids,
        )
        .unwrap(),
        "issue": "ambiguous state",
        "choices": ["confirm", "unpin"],
        "suggested_research_queries": []
    })
    .to_string();

    for invalid_source_ids in [
        vec![],
        vec!["mem_b".to_string(), "mem_a".to_string()],
        vec!["mem_a".to_string(), "mem_a".to_string()],
        vec![" mem_a ".to_string()],
    ] {
        assert!(db
            .insert_lint_review_if_absent(&review_id, &invalid_source_ids, &payload)
            .await
            .is_err());
    }
    assert!(db
        .insert_lint_review_if_absent(&review_id, &source_ids, &payload)
        .await
        .unwrap());
    assert!(!db
        .insert_lint_review_if_absent(&review_id, &source_ids, &payload)
        .await
        .unwrap());
    db.test_primary_session()
        .await
        .execute(
            "UPDATE refinement_queue SET status = 'dismissed' WHERE id = ?1",
            libsql::params![review_id.clone()],
        )
        .await
        .unwrap();
    assert!(!db
        .insert_lint_review_if_absent(&review_id, &source_ids, &payload)
        .await
        .unwrap());
    assert_eq!(
        db.get_refinement_proposal(&review_id)
            .await
            .unwrap()
            .unwrap()
            .status,
        "dismissed"
    );
}

#[tokio::test]
async fn prepare_plan_emits_exact_manifest_without_mutating_canonical_memory() {
    use crate::lint::{
        context::{CancellationToken, LintClock},
        runner::LintRunner,
    };
    use wenlan_types::lint::{LintProfile, LintQuery};

    let (db, _dir) = crate::db::tests::test_db().await;
    db.test_primary_session()
        .await
        .execute_batch(
            "INSERT INTO memories
                 (id,content,source,source_id,title,chunk_index,last_modified,chunk_type,
                  pending_revision,is_recap,supersede_mode,memory_type,source_agent,
                  pinned,confirmed,stability)
             VALUES ('row_exact','exact','memory','mem_exact','exact',0,1,'text',
                     0,0,'hide','fact','   ',0,0,'new');",
        )
        .await
        .unwrap();
    let runner = || LintRunner::new(LintClock::fixed(), CancellationToken::new());
    let general = runner()
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::General), None),
            None,
            false,
        )
        .await
        .unwrap();
    let deep = runner()
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::Deep), None),
            None,
            false,
        )
        .await
        .unwrap();
    let request =
        RepairPlanRequest::try_new(RepairLintScope::global(), general, Some(deep)).unwrap();
    let repair_root = tempfile::tempdir().unwrap();

    let plan = prepare_repair_plan(
        &db,
        &crate::repair::RepairArtifactStore::new(repair_root.path().to_path_buf()),
        request,
        None,
        1_721_000_000,
    )
    .await
    .unwrap();
    assert!(plan.entries().iter().any(|entry| matches!(
        entry.resolution(),
        RepairResolution::Ready { manifest }
            if manifest.writer() == wenlan_types::repair::RepairWriter::NormalizeMemorySourceAgent
    )));
    let source_agent = db
        .test_primary_session()
        .await
        .query(
            "SELECT source_agent FROM memories WHERE source_id='mem_exact'",
            (),
        )
        .await
        .unwrap()
        .next()
        .await
        .unwrap()
        .unwrap()
        .get::<Option<String>>(0)
        .unwrap();
    assert_eq!(source_agent.as_deref(), Some("   "));
}

#[tokio::test]
async fn deep_plan_keeps_deterministic_manifest_general_only_and_pageable() {
    use crate::lint::{
        context::{CancellationToken, LintClock},
        runner::LintRunner,
    };
    use wenlan_types::{
        lint::{LintProfile, LintQuery},
        repair::RepairWriter,
    };

    let (db, _dir) = crate::db::tests::test_db().await;
    db.test_primary_session()
        .await
        .execute_batch(
            "INSERT INTO memories
                 (id,content,source,source_id,title,chunk_index,last_modified,chunk_type,
                  pending_revision,is_recap,supersede_mode,memory_type,source_agent,
                  pinned,confirmed,stability)
             VALUES ('row_pageable','pageable','memory','mem_pageable','pageable',0,1,'text',
                     0,0,'hide','fact','   ',0,0,'new');",
        )
        .await
        .unwrap();
    let runner = || LintRunner::new(LintClock::fixed(), CancellationToken::new());
    let general = runner()
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::General), None),
            None,
            false,
        )
        .await
        .unwrap();
    let deep = with_large_non_target_evidence(
        runner()
            .run(
                &db,
                &LintQuery::new(Some(LintProfile::Deep), None),
                None,
                false,
            )
            .await
            .unwrap(),
    );
    assert!(
        serde_json::to_vec(&crate::repair::repair_check_baseline(&deep).unwrap())
            .unwrap()
            .len()
            > REPAIR_PLAN_PAGE_MAX_BYTES
    );
    let repair_root = tempfile::tempdir().unwrap();
    let store = crate::repair::RepairArtifactStore::new(repair_root.path().to_path_buf());
    let plan = prepare_repair_plan(
        &db,
        &store,
        RepairPlanRequest::try_new(RepairLintScope::global(), general, Some(deep)).unwrap(),
        None,
        1_721_000_000,
    )
    .await
    .unwrap();
    let manifest = plan
        .entries()
        .iter()
        .find_map(|entry| match entry.resolution() {
            RepairResolution::Ready { manifest }
                if manifest.writer() == RepairWriter::NormalizeMemorySourceAgent =>
            {
                Some(manifest)
            }
            _ => None,
        })
        .expect("normalize source-agent manifest");
    assert!(manifest.source().is_general_only_deterministic());
    assert!(manifest
        .post_assertions()
        .verification_policy()
        .is_general_only());
    store
        .load_plan_entries_page(
            &RepairPlanEntriesRequest::try_new(
                plan.plan_id().to_string(),
                plan.plan_digest().clone(),
                0,
                plan.entries().len().min(100),
            )
            .unwrap(),
        )
        .unwrap();
}

#[tokio::test]
async fn deep_only_deterministic_finding_is_visible_but_not_general_only_ready() {
    use crate::lint::{
        context::{CancellationToken, LintClock},
        runner::LintRunner,
    };
    use wenlan_types::{
        lint::{LintProfile, LintQuery},
        repair::RepairWriter,
    };

    let (db, _dir) = crate::db::tests::test_db().await;
    db.test_primary_session()
        .await
        .execute_batch(
            "INSERT INTO memories
                 (id,content,source,source_id,title,chunk_index,last_modified,chunk_type,
                  pending_revision,is_recap,supersede_mode,memory_type,source_agent,
                  pinned,confirmed,stability)
             VALUES ('row_deep_only','deep only','memory','mem_deep_only','deep only',0,1,'text',
                     0,0,'hide','fact','   ',0,0,'new');",
        )
        .await
        .unwrap();
    let runner = || LintRunner::new(LintClock::fixed(), CancellationToken::new());
    let general = with_check_outcome(
        runner()
            .run(
                &db,
                &LintQuery::new(Some(LintProfile::General), None),
                None,
                false,
            )
            .await
            .unwrap(),
        "identity.memory_state_integrity",
        LintOutcome::Pass,
    );
    let deep = runner()
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::Deep), None),
            None,
            false,
        )
        .await
        .unwrap();
    let repair_root = tempfile::tempdir().unwrap();
    let plan = prepare_repair_plan(
        &db,
        &crate::repair::RepairArtifactStore::new(repair_root.path().to_path_buf()),
        RepairPlanRequest::try_new(RepairLintScope::global(), general, Some(deep)).unwrap(),
        None,
        1_721_000_000,
    )
    .await
    .unwrap();
    assert!(!plan.entries().iter().any(|entry| matches!(
        entry.resolution(),
        RepairResolution::Ready { manifest }
            if manifest.writer() == RepairWriter::NormalizeMemorySourceAgent
    )));
    assert!(plan.entries().iter().any(|entry| {
        entry.check_id() == "identity.memory_state_integrity"
            && matches!(
                entry.resolution(),
                RepairResolution::Blocked { blocked }
                    if blocked.reason_code() == RepairBlockedReasonCode::MissingPrerequisite
                        && blocked.detail().contains("only backed by Deep")
            )
    }));
}

#[tokio::test]
#[cfg_attr(not(unix), ignore = "repair artifacts are unix-only")]
async fn approved_general_only_deterministic_manifest_applies_and_verifies_exact_field() {
    use crate::lint::{
        context::{CancellationToken, LintClock},
        runner::LintRunner,
    };
    use wenlan_types::{
        lint::{LintProfile, LintQuery},
        repair::{ApplyRepairRequest, RepairDigest, RepairWriter, VerifyRepairRequest},
        repair_plan::RepairPlanEntriesRequest,
    };

    let (db, _dir) = crate::db::tests::test_db().await;
    db.test_primary_session().await
        .execute_batch(
            "INSERT INTO memories
                 (id,content,source,source_id,title,chunk_index,last_modified,chunk_type,
                  pending_revision,is_recap,supersede_mode,memory_type,source_agent,
                  pinned,confirmed,stability)
             VALUES ('row_apply','unchanged content','memory','mem_apply','unchanged title',0,7,'text',
                     0,0,'hide','fact','   ',0,0,'new');",
        )
        .await
        .unwrap();
    let runner = || LintRunner::new(LintClock::fixed(), CancellationToken::new());
    let general = runner()
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::General), None),
            None,
            false,
        )
        .await
        .unwrap();
    let prepared_general = general.clone();
    let repair_root = tempfile::tempdir().unwrap();
    let store = crate::repair::RepairArtifactStore::new(repair_root.path().to_path_buf());
    let plan = prepare_repair_plan(
        &db,
        &store,
        RepairPlanRequest::try_new(RepairLintScope::global(), general, None).unwrap(),
        None,
        1_721_000_000,
    )
    .await
    .unwrap();
    let page_len = plan.entries().len().min(2);
    assert!(page_len > 0);
    let first_page = store
        .load_plan_entries_page(
            &RepairPlanEntriesRequest::try_new(
                plan.plan_id().to_string(),
                plan.plan_digest().clone(),
                0,
                page_len,
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(first_page.entries(), &plan.entries()[..page_len]);
    assert!(store
        .load_plan_entries_page(
            &RepairPlanEntriesRequest::try_new(
                plan.plan_id().to_string(),
                RepairDigest::parse(
                    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                )
                .unwrap(),
                0,
                page_len,
            )
            .unwrap(),
        )
        .is_err());
    let artifact = store.plan_path(plan.plan_id()).unwrap();
    let lines = std::fs::read_to_string(&artifact).unwrap();
    assert_eq!(lines.lines().count(), plan.entries().len() + 2);
    let header: serde_json::Value = serde_json::from_str(lines.lines().next().unwrap()).unwrap();
    assert_eq!(header["scope"], serde_json::to_value(plan.scope()).unwrap());
    assert_eq!(
        header["general_report_receipt"],
        serde_json::to_value(plan.general_report_receipt()).unwrap()
    );
    assert_eq!(
        header["deep_report_receipt"],
        serde_json::to_value(plan.deep_report_receipt()).unwrap()
    );
    assert!(store.persist_plan(&plan).is_err());
    let mut artifact_records = lines
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .collect::<Vec<_>>();
    artifact_records
        .last_mut()
        .unwrap()
        .as_object_mut()
        .unwrap()
        .insert(
            "entry_count".to_string(),
            serde_json::json!(plan.entries().len() + 1),
        );
    std::fs::write(
        &artifact,
        format!(
            "{}\n",
            artifact_records
                .iter()
                .map(serde_json::Value::to_string)
                .collect::<Vec<_>>()
                .join("\n")
        ),
    )
    .unwrap();
    assert!(store
        .load_plan_entries_page(
            &RepairPlanEntriesRequest::try_new(
                plan.plan_id().to_string(),
                plan.plan_digest().clone(),
                0,
                page_len,
            )
            .unwrap(),
        )
        .is_err());
    let manifest = plan
        .entries()
        .iter()
        .find_map(|entry| match entry.resolution() {
            RepairResolution::Ready { manifest }
                if manifest.writer() == RepairWriter::NormalizeMemorySourceAgent =>
            {
                Some(manifest.as_ref().clone())
            }
            _ => None,
        })
        .expect("normalize source-agent manifest");
    assert!(manifest.source().is_general_only_deterministic());
    assert!(manifest
        .post_assertions()
        .verification_policy()
        .is_general_only());
    let request = ApplyRepairRequest::try_new(
        manifest.manifest_id().to_string(),
        manifest.manifest_digest().clone(),
        format!(
            "apply repair {} {}",
            manifest.manifest_id(),
            manifest.manifest_digest().as_str()
        ),
    )
    .unwrap();

    let apply_receipt = crate::repair::apply_repair(&db, &store, request, 1_721_000_001)
        .await
        .unwrap();

    let session = db.test_primary_session().await;
    let mut rows = session
        .query(
            "SELECT content,title,last_modified,memory_type,source_agent
             FROM memories WHERE source_id='mem_apply'",
            (),
        )
        .await
        .unwrap();
    let row = rows.next().await.unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "unchanged content");
    assert_eq!(row.get::<String>(1).unwrap(), "unchanged title");
    assert_eq!(row.get::<i64>(2).unwrap(), 7);
    assert_eq!(row.get::<String>(3).unwrap(), "fact");
    assert_eq!(row.get::<Option<String>>(4).unwrap(), None);
    drop(rows);
    drop(session);

    assert_eq!(
        manifest.post_assertions().target_check_id(),
        "identity.memory_state_integrity"
    );
    assert!(crate::repair::deterministic_target_assertion_supported(
        &manifest
    ));
    let general = runner()
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::General), None),
            None,
            false,
        )
        .await
        .unwrap();
    let mut unchanged_value = serde_json::to_value(&general).unwrap();
    let prepared_value = serde_json::to_value(&prepared_general).unwrap();
    let prepared_target = prepared_value["checks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|check| check["check_id"] == manifest.post_assertions().target_check_id())
        .unwrap()
        .clone();
    let checks = unchanged_value["checks"].as_array_mut().unwrap();
    let target = checks
        .iter_mut()
        .find(|check| check["check_id"] == manifest.post_assertions().target_check_id())
        .unwrap();
    *target = prepared_target;
    let passed = checks
        .iter()
        .filter(|check| check["outcome"] == "pass")
        .count();
    let findings = checks
        .iter()
        .filter(|check| check["outcome"] == "finding")
        .count();
    let actionable_findings = checks
        .iter()
        .filter(|check| check["outcome"] == "finding" && check["gate_effect"] == "actionable")
        .count();
    let incomplete = checks
        .iter()
        .filter(|check| {
            matches!(
                check["outcome"].as_str(),
                Some("not_run_prerequisite" | "inconsistent_snapshot" | "failed_to_run")
            )
        })
        .count();
    unchanged_value["totals"] = serde_json::json!({
        "checks": checks.len(),
        "passed": passed,
        "findings": findings,
        "actionable_findings": actionable_findings,
        "advisory_findings": findings - actionable_findings,
        "incomplete": incomplete,
    });
    let unchanged_general = serde_json::from_value(unchanged_value).unwrap();
    let unchanged_result = crate::repair::record_repair_verification(
        &db,
        &store,
        VerifyRepairRequest::try_new_general_only(
            manifest.manifest_id().to_string(),
            manifest.manifest_digest().clone(),
            apply_receipt.receipt_digest().clone(),
            unchanged_general,
        )
        .unwrap(),
        None,
        1_721_000_002,
    )
    .await;
    assert!(matches!(
        unchanged_result,
        Err(crate::error::WenlanError::Validation(message))
            if message == "repair_target_assertion_failed"
    ));
    crate::repair::record_repair_verification(
        &db,
        &store,
        VerifyRepairRequest::try_new_general_only(
            manifest.manifest_id().to_string(),
            manifest.manifest_digest().clone(),
            apply_receipt.receipt_digest().clone(),
            general,
        )
        .unwrap(),
        None,
        1_721_000_003,
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn incomplete_tag_source_check_cannot_be_replaced_by_a_ready_manifest() {
    use crate::lint::{
        context::{CancellationToken, LintClock},
        runner::LintRunner,
    };
    use wenlan_types::{
        lint::{LintOutcome, LintProfile, LintQuery},
        repair::RepairWriter,
    };

    let (db, _dir) = crate::db::tests::test_db().await;
    db.test_primary_session()
        .await
        .execute(
            "INSERT INTO document_tags(source,source_id,tag)
             VALUES ('future','missing','stale')",
            (),
        )
        .await
        .unwrap();
    let general = LintRunner::new(LintClock::fixed(), CancellationToken::new())
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::General), None),
            None,
            false,
        )
        .await
        .unwrap();
    assert_eq!(
        general
            .checks()
            .iter()
            .find(|check| check.check_id() == "identity.tag_integrity")
            .unwrap()
            .outcome(),
        LintOutcome::Finding
    );
    let incomplete =
        with_check_outcome(general, "identity.tag_integrity", LintOutcome::FailedToRun);
    let repair_root = tempfile::tempdir().unwrap();
    let store = crate::repair::RepairArtifactStore::new(repair_root.path().to_path_buf());

    let plan = prepare_repair_plan(
        &db,
        &store,
        RepairPlanRequest::try_new(RepairLintScope::global(), incomplete, None).unwrap(),
        None,
        1_721_000_000,
    )
    .await
    .unwrap();

    assert!(!plan.entries().iter().any(|entry| matches!(
        entry.resolution(),
        RepairResolution::Ready { manifest }
            if manifest.writer() == RepairWriter::DeleteTagRow
    )));
    assert!(plan.entries().iter().any(|entry| {
        entry.check_id() == "identity.tag_integrity"
            && matches!(
                entry.resolution(),
                RepairResolution::Blocked { blocked }
                    if blocked.reason_code() == RepairBlockedReasonCode::SourceIncomplete
            )
    }));
}

#[tokio::test]
#[cfg_attr(not(unix), ignore = "repair artifacts are unix-only")]
async fn tag_manifests_from_global_plan_verify_sequentially_across_ordinal_shifts() {
    use crate::db::repair_verification::{
        with_repair_verification_test_control, RepairVerificationTestControl,
    };
    use crate::lint::{
        context::{CancellationToken, LintClock},
        runner::LintRunner,
    };
    use wenlan_types::{
        lint::{LintOutcome, LintProfile, LintQuery},
        repair::{ApplyRepairRequest, RepairTarget, RepairWriter, VerifyRepairRequest},
    };

    let (db, _dir) = crate::db::tests::test_db().await;
    db.test_primary_session()
        .await
        .execute_batch(
            "INSERT INTO spaces (id,name,created_at,updated_at)
             VALUES ('work','work',1,1);
             INSERT INTO memories
                 (id,content,source,source_id,title,chunk_index,last_modified,chunk_type,
                  pending_revision,is_recap,supersede_mode,memory_type,source_agent,
                  pinned,confirmed,stability)
             VALUES ('row_tag_owner','valid','memory','mem_tag_owner','valid',0,7,'text',
                     0,0,'hide','fact','test',0,0,'new');
             INSERT INTO document_tags(source,source_id,tag)
             VALUES
                 ('aaa','missing_first','invalid-first'),
                 ('memory','mem_tag_owner','valid-middle'),
                 ('zzz','missing_second','invalid-second');",
        )
        .await
        .unwrap();
    let runner = || LintRunner::new(LintClock::fixed(), CancellationToken::new());
    let prepared_general = runner()
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::General), None),
            None,
            false,
        )
        .await
        .unwrap();
    let repair_root = tempfile::tempdir().unwrap();
    let store = crate::repair::RepairArtifactStore::new(repair_root.path().to_path_buf());
    let plan = prepare_repair_plan(
        &db,
        &store,
        RepairPlanRequest::try_new(RepairLintScope::global(), prepared_general.clone(), None)
            .unwrap(),
        None,
        1_721_000_000,
    )
    .await
    .unwrap();
    let tag_manifests = plan
        .entries()
        .iter()
        .filter_map(|entry| match entry.resolution() {
            RepairResolution::Ready { manifest }
                if manifest.writer() == RepairWriter::DeleteTagRow =>
            {
                Some(manifest.as_ref().clone())
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(tag_manifests.len(), 2);
    let first = tag_manifests
        .iter()
        .find(|manifest| {
            matches!(
                manifest.target(),
                RepairTarget::Tag { source, .. } if source == "aaa"
            )
        })
        .unwrap();
    let second = tag_manifests
        .iter()
        .find(|manifest| {
            matches!(
                manifest.target(),
                RepairTarget::Tag { source, .. } if source == "zzz"
            )
        })
        .unwrap();
    let apply_request = |manifest: &wenlan_types::repair::RepairManifest| {
        ApplyRepairRequest::try_new(
            manifest.manifest_id().to_string(),
            manifest.manifest_digest().clone(),
            format!(
                "apply repair {} {}",
                manifest.manifest_id(),
                manifest.manifest_digest().as_str()
            ),
        )
        .unwrap()
    };

    let first_apply = crate::repair::apply_repair(&db, &store, apply_request(first), 1_721_000_001)
        .await
        .unwrap();
    let after_first = runner()
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::General), None),
            None,
            false,
        )
        .await
        .unwrap();
    let tag_lock_path = store.root().join(format!(
        ".tag-record-set-{}.lock",
        first
            .post_assertions()
            .target_record_set()
            .expect("tag record set")
            .digest()
            .as_str()
    ));
    with_repair_verification_test_control(
        RepairVerificationTestControl {
            after_commit_before_pending_clear: Some(Box::new(move || {
                let lock = std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(&tag_lock_path)
                    .unwrap();
                assert_eq!(
                    fs2::FileExt::try_lock_exclusive(&lock).unwrap_err().kind(),
                    std::io::ErrorKind::WouldBlock
                );
            })),
            ..Default::default()
        },
        crate::repair::record_repair_verification(
            &db,
            &store,
            VerifyRepairRequest::try_new_general_only(
                first.manifest_id().to_string(),
                first.manifest_digest().clone(),
                first_apply.receipt_digest().clone(),
                after_first,
            )
            .unwrap(),
            None,
            1_721_000_002,
        ),
    )
    .await
    .unwrap();

    let second_apply =
        crate::repair::apply_repair(&db, &store, apply_request(second), 1_721_000_003)
            .await
            .unwrap();
    let after_second = runner()
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::General), None),
            None,
            false,
        )
        .await
        .unwrap();
    assert_eq!(
        after_second
            .checks()
            .iter()
            .find(|check| check.check_id() == "identity.tag_integrity")
            .unwrap()
            .outcome(),
        LintOutcome::Pass
    );
    crate::repair::record_repair_verification(
        &db,
        &store,
        VerifyRepairRequest::try_new_general_only(
            second.manifest_id().to_string(),
            second.manifest_digest().clone(),
            second_apply.receipt_digest().clone(),
            after_second,
        )
        .unwrap(),
        None,
        1_721_000_004,
    )
    .await
    .unwrap();
}

#[derive(Clone, Copy)]
enum TagDriftAfterPrepare {
    None,
    Add,
    Replace,
}

async fn verify_large_tag_window_repair(
    target_source_id: &str,
    drift: TagDriftAfterPrepare,
) -> Result<(), crate::error::WenlanError> {
    use crate::lint::{
        context::{CancellationToken, LintClock},
        runner::LintRunner,
    };
    use wenlan_types::{
        lint::{LintProfile, LintQuery},
        repair::{ApplyRepairRequest, RepairTarget, RepairWriter, VerifyRepairRequest},
    };

    let (db, _dir) = crate::db::tests::test_db().await;
    db.test_primary_session()
        .await
        .execute_batch(
            "WITH RECURSIVE sequence(value) AS (
                 VALUES(0)
                 UNION ALL
                 SELECT value + 1 FROM sequence WHERE value < 101
             )
             INSERT INTO document_tags(source,source_id,tag)
             SELECT 'aaa', printf('missing-%03d', value), 'invalid'
             FROM sequence;",
        )
        .await
        .unwrap();
    let runner = || LintRunner::new(LintClock::fixed(), CancellationToken::new());
    let prepared_general = runner()
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::General), None),
            None,
            false,
        )
        .await
        .unwrap();
    let repair_root = tempfile::tempdir().unwrap();
    let store = crate::repair::RepairArtifactStore::new(repair_root.path().to_path_buf());
    let plan = prepare_repair_plan(
        &db,
        &store,
        RepairPlanRequest::try_new(RepairLintScope::global(), prepared_general, None).unwrap(),
        None,
        1_721_200_000,
    )
    .await
    .unwrap();
    let manifest = plan
        .entries()
        .iter()
        .find_map(|entry| match entry.resolution() {
            RepairResolution::Ready { manifest }
                if manifest.writer() == RepairWriter::DeleteTagRow
                    && matches!(
                        manifest.target(),
                        RepairTarget::Tag { source_id, .. }
                            if source_id == target_source_id
                    ) =>
            {
                Some(manifest.as_ref())
            }
            _ => None,
        })
        .unwrap();

    if matches!(drift, TagDriftAfterPrepare::Replace) {
        db.test_primary_session()
            .await
            .execute(
                "DELETE FROM document_tags
                 WHERE source='aaa' AND source_id='missing-100' AND tag='invalid'",
                (),
            )
            .await
            .unwrap();
    }
    if !matches!(drift, TagDriftAfterPrepare::None) {
        db.test_primary_session()
            .await
            .execute(
                "INSERT INTO document_tags(source,source_id,tag)
                 VALUES ('zzz','missing-new','invalid-new')",
                (),
            )
            .await
            .unwrap();
    }
    let apply_result = crate::repair::apply_repair(
        &db,
        &store,
        ApplyRepairRequest::try_new(
            manifest.manifest_id().to_string(),
            manifest.manifest_digest().clone(),
            format!(
                "apply repair {} {}",
                manifest.manifest_id(),
                manifest.manifest_digest().as_str()
            ),
        )
        .unwrap(),
        1_721_200_001,
    )
    .await;
    let apply = match apply_result {
        Ok(apply) => apply,
        Err(error) => {
            let connection = db.test_primary_session().await;
            let mut rows = connection
                .query(
                    "SELECT COUNT(*) FROM document_tags
                     WHERE source='aaa' AND source_id=?1 AND tag='invalid'",
                    libsql::params![target_source_id.to_string()],
                )
                .await
                .unwrap();
            assert_eq!(
                rows.next().await.unwrap().unwrap().get::<i64>(0).unwrap(),
                1,
                "stale set rejection must happen before the approved target mutates"
            );
            return Err(error);
        }
    };
    let after = runner()
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::General), None),
            None,
            false,
        )
        .await
        .unwrap();

    crate::repair::record_repair_verification(
        &db,
        &store,
        VerifyRepairRequest::try_new_general_only(
            manifest.manifest_id().to_string(),
            manifest.manifest_digest().clone(),
            apply.receipt_digest().clone(),
            after,
        )
        .unwrap(),
        None,
        1_721_200_002,
    )
    .await
    .map(|_| ())
}

#[tokio::test]
#[cfg_attr(not(unix), ignore = "repair artifacts are unix-only")]
async fn tag_repair_inside_capped_evidence_window_allows_natural_window_shift() {
    verify_large_tag_window_repair("missing-000", TagDriftAfterPrepare::None)
        .await
        .unwrap();
}

#[tokio::test]
#[cfg_attr(not(unix), ignore = "repair artifacts are unix-only")]
async fn tag_repair_outside_capped_evidence_window_is_still_proven_resolved() {
    verify_large_tag_window_repair("missing-101", TagDriftAfterPrepare::None)
        .await
        .unwrap();
}

#[tokio::test]
#[cfg_attr(not(unix), ignore = "repair artifacts are unix-only")]
async fn same_check_finding_added_after_prepare_is_rejected() {
    let error = verify_large_tag_window_repair("missing-101", TagDriftAfterPrepare::Add)
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        crate::error::WenlanError::Conflict(message)
            if message == "repair_tag_set_changed"
    ));
}

#[tokio::test]
#[cfg_attr(not(unix), ignore = "repair artifacts are unix-only")]
async fn count_neutral_same_check_replacement_after_prepare_is_rejected() {
    let error = verify_large_tag_window_repair("missing-101", TagDriftAfterPrepare::Replace)
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        crate::error::WenlanError::Conflict(message)
            if message == "repair_tag_set_changed"
    ));
}

#[tokio::test]
async fn semantic_finding_becomes_one_stable_review_item() {
    use crate::lint::{
        context::{CancellationToken, LintClock},
        runner::LintRunner,
    };
    use wenlan_types::lint::{
        LintOpaqueId, LintQuery, LintSemanticAction, LintSemanticFinding,
        LintSemanticProviderRoute, LintSemanticReasonCode,
    };

    let (db, _dir) = crate::db::tests::test_db().await;
    db.test_primary_session()
        .await
        .execute_batch(
            "INSERT INTO memories
                 (id,content,source,source_id,title,chunk_index,last_modified,chunk_type,
                  pending_revision,is_recap,supersede_mode,memory_type,source_agent,
                  pinned,confirmed,stability)
             VALUES ('row_semantic','semantic','memory','mem_semantic','semantic',0,7,'text',
                     0,0,'hide','fact',NULL,0,0,'new');",
        )
        .await
        .unwrap();
    let finding = LintSemanticFinding::try_new(
        LintOpaqueId::from_sorted_position(0).unwrap(),
        LintSemanticAction::ReviewStaleness,
        LintSemanticReasonCode::PotentialStaleness,
        8_000,
        LintSemanticProviderRoute::CallingAgent,
        vec![crate::lint::semantic_record_digest(
            "memory",
            "mem_semantic",
        )],
        vec![],
    )
    .unwrap();
    let repair_root = tempfile::tempdir().unwrap();
    let store = crate::repair::RepairArtifactStore::new(repair_root.path().to_path_buf());
    let mut review_ids = Vec::new();

    for prepared_at in [1_721_000_000, 1_721_000_001] {
        let runner = || LintRunner::new(LintClock::fixed(), CancellationToken::new());
        let general = runner()
            .run(
                &db,
                &LintQuery::new(Some(LintProfile::General), None),
                None,
                false,
            )
            .await
            .unwrap();
        let deep = runner()
            .run(
                &db,
                &LintQuery::new(Some(LintProfile::Deep), None),
                None,
                false,
            )
            .await
            .unwrap();
        let plan = prepare_repair_plan(
            &db,
            &store,
            RepairPlanRequest::try_new(
                RepairLintScope::global(),
                general,
                Some(with_semantic_finding(
                    deep,
                    "memories.semantic.staleness",
                    finding.clone(),
                )),
            )
            .unwrap(),
            None,
            prepared_at,
        )
        .await
        .unwrap();
        let review_id = plan
            .entries()
            .iter()
            .find_map(|entry| match entry.resolution() {
                RepairResolution::Review { review_item }
                    if entry.check_id() == "memories.semantic.staleness" =>
                {
                    Some(review_item.review_id().to_string())
                }
                _ => None,
            })
            .expect("semantic review item");
        review_ids.push(review_id);
    }

    assert_eq!(review_ids[0], review_ids[1]);
    let proposal = db
        .get_refinement_proposal(&review_ids[0])
        .await
        .unwrap()
        .unwrap();
    let payload: wenlan_types::RefinementPayload =
        serde_json::from_str(proposal.payload.as_deref().unwrap()).unwrap();
    let wenlan_types::RefinementPayload::LintRepairReview {
        occurrence_digest,
        owner_binding_digest,
        ..
    } = payload
    else {
        panic!("expected lint repair review payload");
    };
    assert_eq!(proposal.source_ids, vec!["mem_semantic".to_string()]);
    assert_eq!(
        owner_binding_digest,
        crate::repair::lint_review_owner_binding_digest(&occurrence_digest, &proposal.source_ids)
            .unwrap()
    );
    let count = db
        .test_primary_session()
        .await
        .query(
            "SELECT COUNT(*) FROM refinement_queue WHERE id=?1",
            libsql::params![review_ids[0].clone()],
        )
        .await
        .unwrap()
        .next()
        .await
        .unwrap()
        .unwrap()
        .get::<i64>(0)
        .unwrap();
    assert_eq!(count, 1);
}

#[tokio::test]
#[cfg_attr(not(unix), ignore = "repair artifacts are unix-only")]
async fn review_item_contract_reclassification_prepare_requires_plan_item() {
    use crate::{
        lint::{
            context::{CancellationToken, LintClock},
            runner::LintRunner,
        },
        repair::{prepare_memory_reclassification, RepairArtifactStore},
    };
    use wenlan_types::{
        lint::{
            LintOpaqueId, LintQuery, LintSemanticAction, LintSemanticFinding,
            LintSemanticProviderRoute, LintSemanticReasonCode,
        },
        repair::{PrepareRepairRequest, RepairLintScope},
        MemoryType,
    };

    let (db, _dir) = crate::db::tests::test_db().await;
    db.test_primary_session()
        .await
        .execute_batch(
            "INSERT INTO memories
                 (id,content,source,source_id,title,chunk_index,last_modified,chunk_type,
                  pending_revision,is_recap,supersede_mode,memory_type,source_agent,
                  pinned,confirmed,stability)
             VALUES ('row_reclass','semantic','memory','mem_reclass','semantic',0,7,'text',
                     0,0,'hide','fact',NULL,0,0,'new');",
        )
        .await
        .unwrap();
    let finding = LintSemanticFinding::try_new(
        LintOpaqueId::from_sorted_position(0).unwrap(),
        LintSemanticAction::ReclassifyMemory,
        LintSemanticReasonCode::ClassificationMismatch,
        9_000,
        LintSemanticProviderRoute::CallingAgent,
        vec![crate::lint::semantic_record_digest("memory", "mem_reclass")],
        vec![],
    )
    .unwrap();
    let runner = || LintRunner::new(LintClock::fixed(), CancellationToken::new());
    let general = runner()
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::General), None),
            None,
            false,
        )
        .await
        .unwrap();
    let deep = with_semantic_finding(
        runner()
            .run(
                &db,
                &LintQuery::new(Some(LintProfile::Deep), None),
                None,
                false,
            )
            .await
            .unwrap(),
        "memories.semantic.classification",
        finding.clone(),
    );
    let repair_root = tempfile::tempdir().unwrap();
    let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
    let plan = prepare_repair_plan(
        &db,
        &store,
        RepairPlanRequest::try_new(
            RepairLintScope::global(),
            general.clone(),
            Some(deep.clone()),
        )
        .unwrap(),
        None,
        1_721_000_000,
    )
    .await
    .unwrap();
    let review_id = plan
        .entries()
        .iter()
        .find_map(|entry| match entry.resolution() {
            RepairResolution::Review { review_item }
                if entry.check_id() == "memories.semantic.classification" =>
            {
                Some(review_item.review_id().to_string())
            }
            _ => None,
        })
        .expect("classification Review Item");
    db.test_primary_session()
        .await
        .execute(
            "DELETE FROM refinement_queue WHERE id=?1",
            libsql::params![review_id],
        )
        .await
        .unwrap();

    let general = runner()
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::General), None),
            None,
            false,
        )
        .await
        .unwrap();
    let deep = with_completed_agent_work(with_semantic_finding(
        runner()
            .run(
                &db,
                &LintQuery::new(Some(LintProfile::Deep), None),
                None,
                false,
            )
            .await
            .unwrap(),
        "memories.semantic.classification",
        finding.clone(),
    ));
    let error = prepare_memory_reclassification(
        &db,
        &store,
        PrepareRepairRequest::try_new(
            RepairLintScope::global(),
            general,
            deep,
            finding,
            MemoryType::Decision,
        )
        .unwrap(),
        1_721_000_001,
    )
    .await
    .unwrap_err();
    assert!(matches!(
        error,
        crate::error::WenlanError::Conflict(message)
            if message == "repair_target_stale"
    ));
}

#[tokio::test]
async fn deterministic_review_payload_binds_the_canonical_owner_set() {
    use crate::lint::{
        context::{CancellationToken, LintClock},
        runner::LintRunner,
    };
    use wenlan_types::{
        lint::LintQuery,
        repair::RepairLintScope,
        repair_plan::{RepairPlanRequest, RepairResolution},
    };

    let (db, _dir) = crate::db::tests::test_db().await;
    let page_root = tempfile::tempdir().unwrap();
    let now = "2026-07-15T00:00:00Z";
    for (id, title, content) in [
        ("source-link", "Source Link", "see [[Shared Target]]"),
        ("target-a", "Shared Target", "first"),
        ("target-b", "Shared Target", "second"),
    ] {
        db.insert_page(id, title, None, content, None, Some("work"), &[], now)
            .await
            .unwrap();
    }
    let runner = || LintRunner::new(LintClock::fixed(), CancellationToken::new());
    let general = runner()
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::General), None),
            Some(page_root.path()),
            true,
        )
        .await
        .unwrap();
    let orphan_check = general
        .checks()
        .iter()
        .find(|check| check.check_id() == "pages.links.orphan_labels")
        .unwrap();
    assert_eq!(
        orphan_check.outcome(),
        LintOutcome::Finding,
        "the source report must authorize the deterministic review: {orphan_check:?}"
    );
    let deep = runner()
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::Deep), None),
            Some(page_root.path()),
            true,
        )
        .await
        .unwrap();
    let repair_root = tempfile::tempdir().unwrap();
    let store = crate::repair::RepairArtifactStore::new(repair_root.path().to_path_buf());
    let plan = prepare_repair_plan(
        &db,
        &store,
        RepairPlanRequest::try_new(RepairLintScope::global(), general, Some(deep)).unwrap(),
        Some(page_root.path()),
        1_721_000_000,
    )
    .await
    .unwrap();
    let review_id = plan
        .entries()
        .iter()
        .find_map(|entry| match entry.resolution() {
            RepairResolution::Review { review_item }
                if entry.check_id() == "pages.links.orphan_labels" =>
            {
                Some(review_item.review_id().to_string())
            }
            _ => None,
        })
        .expect("deterministic review item");
    let proposal = db
        .get_refinement_proposal(&review_id)
        .await
        .unwrap()
        .unwrap();
    let payload: wenlan_types::RefinementPayload =
        serde_json::from_str(proposal.payload.as_deref().unwrap()).unwrap();
    let wenlan_types::RefinementPayload::LintRepairReview {
        occurrence_digest,
        owner_binding_digest,
        ..
    } = payload
    else {
        panic!("expected lint repair review payload");
    };
    assert_eq!(
        proposal.source_ids,
        vec!["source-link:shared target".to_string()]
    );
    assert_eq!(
        owner_binding_digest,
        crate::repair::lint_review_owner_binding_digest(&occurrence_digest, &proposal.source_ids)
            .unwrap()
    );
}

#[tokio::test]
#[cfg_attr(not(unix), ignore = "repair artifacts are unix-only")]
async fn review_item_contract_duplicate_title_plan_can_prepare_manifest() {
    use crate::{
        export::knowledge::KnowledgeProjectionWrite,
        lint::{
            context::{CancellationToken, LintClock},
            runner::LintRunner,
        },
        repair::{prepare_memory_reclassification_with_pages, RepairArtifactStore},
    };
    use wenlan_types::{
        lint::LintQuery,
        repair::{PrepareRepairRequest, RepairChoice, RepairLintScope, RepairWriter},
        repair_plan::{RepairPlanRequest, RepairResolution},
    };

    let (db, _dir) = crate::db::tests::test_db().await;
    let page_root = tempfile::tempdir().unwrap();
    let now = "2026-07-17T00:00:00Z";
    for (id, title) in [("page-a", "Origin"), ("page-b", "origin")] {
        db.insert_page(id, title, None, "body", None, Some("work"), &[], now)
            .await
            .unwrap();
        let page = db.get_page(id).await.unwrap().unwrap();
        KnowledgeProjectionWrite::new(page_root.path().to_path_buf(), &db)
            .write_page(&page)
            .unwrap();
    }
    let runner = || LintRunner::new(LintClock::fixed(), CancellationToken::new());
    let general = runner()
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::General), None),
            Some(page_root.path()),
            true,
        )
        .await
        .unwrap();
    let repair_root = tempfile::tempdir().unwrap();
    let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
    let plan = prepare_repair_plan(
        &db,
        &store,
        RepairPlanRequest::try_new(RepairLintScope::global(), general, None).unwrap(),
        Some(page_root.path()),
        1_721_000_000,
    )
    .await
    .unwrap();
    let review_id = plan
        .entries()
        .iter()
        .find_map(|entry| match entry.resolution() {
            RepairResolution::Review { review_item }
                if entry.check_id() == "pages.duplicate_active_titles"
                    && entry
                        .affected_records()
                        .iter()
                        .any(|record| record.durable_id() == "page-a") =>
            {
                Some(review_item.review_id().to_string())
            }
            _ => None,
        })
        .expect("duplicate title Review Item");

    let fresh_general = runner()
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::General), None),
            Some(page_root.path()),
            true,
        )
        .await
        .unwrap();
    let manifest = prepare_memory_reclassification_with_pages(
        &db,
        &store,
        PrepareRepairRequest::try_new_with_choice(
            RepairLintScope::global(),
            fresh_general,
            None,
            RepairChoice::rename_page_title(
                review_id,
                "page-a".to_string(),
                "Origin".to_string(),
                "Origin Memory".to_string(),
            )
            .unwrap(),
        )
        .unwrap(),
        Some(page_root.path()),
        1_721_000_001,
    )
    .await
    .unwrap();
    assert_eq!(manifest.writer(), RepairWriter::RenamePageTitle);
}

#[tokio::test]
#[cfg_attr(not(unix), ignore = "repair artifacts are unix-only")]
async fn review_item_contract_failed_entity_plan_can_prepare_manifest() {
    use crate::{
        lint::{
            context::{CancellationToken, LintClock},
            runner::LintRunner,
        },
        repair::{prepare_memory_reclassification_with_pages, RepairArtifactStore},
    };
    use wenlan_types::{
        lint::LintQuery,
        repair::{PrepareRepairRequest, RepairChoice, RepairLintScope, RepairWriter},
        repair_plan::{RepairPlanRequest, RepairResolution},
    };

    let (db, _dir) = crate::db::tests::test_db().await;
    db.test_primary_session()
        .await
        .execute_batch(
            "INSERT INTO spaces (id,name,created_at,updated_at)
             VALUES ('space-work','work',1,1);
             INSERT INTO memories
                 (id,content,source,source_id,title,chunk_index,last_modified,chunk_type,
                  pending_revision,is_recap,supersede_mode,memory_type,space)
             VALUES ('row-entity','Target memory','memory','mem-entity','target',0,10,
                     'text',0,0,'hide','fact','work');
             INSERT INTO enrichment_steps
                 (source_id,step_name,status,error,attempts,updated_at)
             VALUES ('mem-entity','entity_extract','failed','transient',2,1721000000);",
        )
        .await
        .unwrap();
    // G6 Stage 3: the selected-entity existence guard reads the canonical
    // shadow page (`entity_page_map` JOIN `pages`), and migration 123 dropped
    // `entities`, so the shadow IS the seed.
    db.test_seed_entity_shadow_page(
        crate::db::TestEntity::new("ent-new", "New", "concept").space("work"),
    )
    .await
    .unwrap();
    let runner = || LintRunner::new(LintClock::fixed(), CancellationToken::new());
    let general = runner()
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::General), None),
            None,
            false,
        )
        .await
        .unwrap();
    let repair_root = tempfile::tempdir().unwrap();
    let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
    let plan = prepare_repair_plan(
        &db,
        &store,
        RepairPlanRequest::try_new(RepairLintScope::global(), general, None).unwrap(),
        None,
        1_721_000_000,
    )
    .await
    .unwrap();
    let review_id = plan
        .entries()
        .iter()
        .find_map(|entry| match entry.resolution() {
            RepairResolution::Review { review_item }
                if entry.check_id() == "memories.enrichment_failures"
                    && entry
                        .affected_records()
                        .iter()
                        .any(|record| record.durable_id() == "mem-entity") =>
            {
                Some(review_item.review_id().to_string())
            }
            _ => None,
        })
        .expect("failed entity extraction Review Item");

    let fresh_general = runner()
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::General), None),
            None,
            false,
        )
        .await
        .unwrap();
    let manifest = prepare_memory_reclassification_with_pages(
        &db,
        &store,
        PrepareRepairRequest::try_new_with_choice(
            RepairLintScope::global(),
            fresh_general,
            None,
            RepairChoice::complete_entity_extraction(
                review_id,
                "mem-entity".to_string(),
                vec!["ent-new".to_string()],
            )
            .unwrap(),
        )
        .unwrap(),
        None,
        1_721_000_001,
    )
    .await
    .unwrap();
    assert_eq!(manifest.writer(), RepairWriter::CompleteEntityExtraction);
}

#[tokio::test]
#[cfg_attr(not(unix), ignore = "repair artifacts are unix-only")]
async fn deterministic_db_writers_apply_exactly_and_orphan_binding_fails_when_ambiguous() {
    use crate::lint::{
        context::{CancellationToken, LintClock},
        runner::LintRunner,
    };
    use std::collections::{BTreeMap, BTreeSet};
    use wenlan_types::{
        lint::LintQuery,
        repair::{ApplyRepairRequest, RepairManifest, RepairWriter},
    };

    let (db, _dir) = crate::db::tests::test_db().await;
    let page_root = tempfile::tempdir().unwrap();
    db.test_primary_session()
        .await
        .execute_batch(
            "INSERT INTO memories
                 (id,content,source,source_id,title,chunk_index,last_modified,chunk_type,
                  pending_revision,is_recap,supersede_mode,memory_type,source_agent,
                  supersedes,pinned,confirmed,stability)
             VALUES ('row_self','self','memory','mem_self','self',0,7,'text',
                     0,0,'hide','fact',NULL,'mem_self',0,0,'new');
             INSERT INTO document_tags(source,source_id,tag)
             VALUES ('memory','missing_tag_owner','stale');",
        )
        .await
        .unwrap();
    let now = "2026-07-15T00:00:00Z";
    db.insert_page(
        "source_link",
        "Source Link",
        None,
        "see [[Unique Link Target]]",
        None,
        Some("work"),
        &[],
        now,
    )
    .await
    .unwrap();
    db.insert_page(
        "target_link_a",
        "Unique Link Target",
        None,
        "target",
        None,
        Some("work"),
        &[],
        now,
    )
    .await
    .unwrap();
    let runner = || LintRunner::new(LintClock::fixed(), CancellationToken::new());
    let general = runner()
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::General), None),
            Some(page_root.path()),
            true,
        )
        .await
        .unwrap();
    for check_id in [
        "identity.memory_state_integrity",
        "memories.supersession_integrity",
        "identity.tag_integrity",
        "pages.links.orphan_labels",
    ] {
        assert_eq!(
            general
                .checks()
                .iter()
                .find(|check| check.check_id() == check_id)
                .unwrap()
                .outcome(),
            LintOutcome::Finding,
            "the source report must authorize {check_id}"
        );
    }
    let deep = runner()
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::Deep), None),
            Some(page_root.path()),
            true,
        )
        .await
        .unwrap();
    let repair_root = tempfile::tempdir().unwrap();
    let store = crate::repair::RepairArtifactStore::new(repair_root.path().to_path_buf());
    let plan = prepare_repair_plan(
        &db,
        &store,
        RepairPlanRequest::try_new(RepairLintScope::global(), general, Some(deep)).unwrap(),
        Some(page_root.path()),
        1_721_000_000,
    )
    .await
    .unwrap();
    let ready_tuples = plan
        .entries()
        .iter()
        .filter_map(|entry| match entry.resolution() {
            RepairResolution::Ready { manifest } => Some((
                manifest.manifest_id().to_string(),
                manifest.manifest_digest().as_str().to_string(),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    let unique_ready_tuples = ready_tuples.iter().cloned().collect::<BTreeSet<_>>();
    assert_eq!(
        ready_tuples.len(),
        unique_ready_tuples.len(),
        "one exact mutation must produce one approvable manifest tuple even when it resolves multiple checks"
    );
    assert_eq!(
        plan.totals().ready(),
        unique_ready_tuples.len() as u64,
        "ready totals must count approval tuples, not source-check aliases"
    );
    let manifests = plan
        .entries()
        .iter()
        .filter_map(|entry| match entry.resolution() {
            RepairResolution::Ready { manifest } => Some((
                manifest.manifest_id().to_string(),
                manifest.as_ref().clone(),
            )),
            _ => None,
        })
        .collect::<BTreeMap<String, RepairManifest>>();
    let by_writer = |writer| {
        manifests
            .values()
            .find(|manifest| manifest.writer() == writer)
            .cloned()
            .expect("writer manifest")
    };
    let exact_request = |manifest: &RepairManifest| {
        ApplyRepairRequest::try_new(
            manifest.manifest_id().to_string(),
            manifest.manifest_digest().clone(),
            format!(
                "apply repair {} {}",
                manifest.manifest_id(),
                manifest.manifest_digest().as_str()
            ),
        )
        .unwrap()
    };

    for writer in [
        RepairWriter::ClearMemorySupersedes,
        RepairWriter::DeleteTagRow,
    ] {
        let manifest = by_writer(writer);
        crate::repair::apply_repair(&db, &store, exact_request(&manifest), 1_721_000_001)
            .await
            .unwrap();
    }
    let supersedes = db
        .test_primary_session()
        .await
        .query(
            "SELECT supersedes FROM memories WHERE source_id='mem_self'",
            (),
        )
        .await
        .unwrap()
        .next()
        .await
        .unwrap()
        .unwrap()
        .get::<Option<String>>(0)
        .unwrap();
    assert_eq!(supersedes, None);
    let tag_count = db
        .test_primary_session()
        .await
        .query(
            "SELECT COUNT(*) FROM document_tags
             WHERE source='memory' AND source_id='missing_tag_owner' AND tag='stale'",
            (),
        )
        .await
        .unwrap()
        .next()
        .await
        .unwrap()
        .unwrap()
        .get::<i64>(0)
        .unwrap();
    assert_eq!(tag_count, 0);

    db.insert_page(
        "target_link_b",
        "Unique Link Target",
        None,
        "second target",
        None,
        Some("work"),
        &[],
        now,
    )
    .await
    .unwrap();
    let link_manifest = by_writer(RepairWriter::BindPageLink);
    let error =
        crate::repair::apply_repair(&db, &store, exact_request(&link_manifest), 1_721_000_002)
            .await
            .unwrap_err();
    assert!(error.to_string().contains("repair_target_stale"));
    let target = db
        .test_primary_session()
        .await
        .query(
            "SELECT target_page_id FROM page_links
             WHERE source_page_id='source_link' AND label='Unique Link Target'",
            (),
        )
        .await
        .unwrap()
        .next()
        .await
        .unwrap()
        .unwrap()
        .get::<Option<String>>(0)
        .unwrap();
    assert_eq!(target, None);
}

/// The repair tool's orphan-bind (`RepairWriter::BindPageLink`) is a second
/// writer of `page_links.target_page_id` beside `resolve_orphan_page_links`;
/// a successful bind must mint the now-implied canonical `links` edge, or the
/// row is permanent "missing" parity drift (G6 Stage 0).
#[tokio::test]
#[cfg_attr(not(unix), ignore = "repair artifacts are unix-only")]
async fn bind_page_link_repair_mints_the_links_edge() {
    use crate::lint::{
        context::{CancellationToken, LintClock},
        runner::LintRunner,
    };
    use wenlan_types::{
        lint::LintQuery,
        repair::{ApplyRepairRequest, RepairWriter},
    };

    let (db, _dir) = crate::db::tests::test_db().await;
    let page_root = tempfile::tempdir().unwrap();
    let now = "2026-07-15T00:00:00Z";
    db.insert_page(
        "source_link",
        "Source Link",
        None,
        "see [[Unique Link Target]]",
        None,
        Some("work"),
        &[],
        now,
    )
    .await
    .unwrap();
    db.insert_page(
        "target_link_a",
        "Unique Link Target",
        None,
        "target",
        None,
        Some("work"),
        &[],
        now,
    )
    .await
    .unwrap();
    let runner = || LintRunner::new(LintClock::fixed(), CancellationToken::new());
    let general = runner()
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::General), None),
            Some(page_root.path()),
            true,
        )
        .await
        .unwrap();
    let deep = runner()
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::Deep), None),
            Some(page_root.path()),
            true,
        )
        .await
        .unwrap();
    let repair_root = tempfile::tempdir().unwrap();
    let store = crate::repair::RepairArtifactStore::new(repair_root.path().to_path_buf());
    let plan = prepare_repair_plan(
        &db,
        &store,
        RepairPlanRequest::try_new(RepairLintScope::global(), general, Some(deep)).unwrap(),
        Some(page_root.path()),
        1_721_000_000,
    )
    .await
    .unwrap();
    let manifest = plan
        .entries()
        .iter()
        .find_map(|entry| match entry.resolution() {
            RepairResolution::Ready { manifest }
                if manifest.writer() == RepairWriter::BindPageLink =>
            {
                Some(manifest.as_ref().clone())
            }
            _ => None,
        })
        .expect("bind manifest ready");
    let request = ApplyRepairRequest::try_new(
        manifest.manifest_id().to_string(),
        manifest.manifest_digest().clone(),
        format!(
            "apply repair {} {}",
            manifest.manifest_id(),
            manifest.manifest_digest().as_str()
        ),
    )
    .unwrap();
    crate::repair::apply_repair(&db, &store, request, 1_721_000_001)
        .await
        .unwrap();

    // G6 Stage 2 PR 2b (item 3): resolving a bind stops writing the
    // resolved row into `page_links` -- the row is deleted once bound, not
    // updated in place. The edge minted below is now the sole canonical
    // representation of "row bound".
    let orphan_row_gone = db
        .test_primary_session()
        .await
        .query(
            "SELECT 1 FROM page_links WHERE source_page_id='source_link'",
            (),
        )
        .await
        .unwrap()
        .next()
        .await
        .unwrap()
        .is_none();
    assert!(
        orphan_row_gone,
        "resolved orphan row is deleted, not updated in place"
    );

    // G6 Stage 2 PR 2b: parity oracle retired, correctness carried by per-writer regression tests (item 7).
    let edge_id = crate::provenance::compute_edge_id(
        "links",
        "page",
        "source_link",
        "page",
        "target_link_a",
        "unique link target",
    );
    let valid_until = db
        .test_primary_session()
        .await
        .query(
            "SELECT valid_until FROM edges WHERE edge_id = ?1",
            libsql::params![edge_id.as_str()],
        )
        .await
        .unwrap()
        .next()
        .await
        .unwrap()
        .expect("minted links edge exists")
        .get::<Option<i64>>(0)
        .unwrap();
    assert!(valid_until.is_none(), "minted links edge is active");
}

#[tokio::test]
#[cfg_attr(not(unix), ignore = "repair artifacts are unix-only")]
async fn page_projection_manifest_repairs_only_the_named_page_projection() {
    use crate::db::repair_verification::{
        assert_repair_verification_transaction_reusable, with_repair_verification_test_control,
        RepairVerificationTestControl,
    };
    use crate::lint::{
        context::{CancellationToken, LintClock},
        runner::LintRunner,
    };
    use wenlan_types::{
        lint::{LintProfile, LintQuery},
        repair::{ApplyRepairRequest, RepairWriter, VerifyRepairRequest},
    };

    let (db, _dir) = crate::db::tests::test_db().await;
    let page_root = tempfile::tempdir().unwrap();
    let now = "2026-07-15T00:00:00Z";
    db.insert_page(
        "page_projection_repair",
        "Projection Repair",
        None,
        "canonical body",
        None,
        Some("work"),
        &[],
        now,
    )
    .await
    .unwrap();
    let page = db
        .get_page("page_projection_repair")
        .await
        .unwrap()
        .unwrap();
    crate::export::knowledge::KnowledgeProjectionWrite::new(page_root.path().to_path_buf(), &db)
        .write_page(&page)
        .unwrap();
    let projected_path = page_root.path().join("projection-repair.md");
    let before_projection = std::fs::read(&projected_path).unwrap();
    std::fs::write(page_root.path().join("unrelated.txt"), b"leave me alone").unwrap();
    db.test_primary_session()
        .await
        .execute(
            "UPDATE pages SET version=version+1 WHERE id='page_projection_repair'",
            (),
        )
        .await
        .unwrap();

    let runner = || LintRunner::new(LintClock::fixed(), CancellationToken::new());
    let general = runner()
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::General), None),
            Some(page_root.path()),
            true,
        )
        .await
        .unwrap();
    let deep = runner()
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::Deep), None),
            Some(page_root.path()),
            true,
        )
        .await
        .unwrap();
    let repair_root = tempfile::tempdir().unwrap();
    let store = crate::repair::RepairArtifactStore::new(repair_root.path().to_path_buf());
    let plan = prepare_repair_plan(
        &db,
        &store,
        RepairPlanRequest::try_new(RepairLintScope::global(), general, Some(deep)).unwrap(),
        Some(page_root.path()),
        1_721_000_000,
    )
    .await
    .unwrap();
    let manifest = plan
        .entries()
        .iter()
        .find_map(|entry| match entry.resolution() {
            RepairResolution::Ready { manifest }
                if manifest.writer() == RepairWriter::RegeneratePageProjection =>
            {
                Some(manifest.as_ref().clone())
            }
            _ => None,
        })
        .expect("page projection manifest");
    let exact_request = || {
        ApplyRepairRequest::try_new(
            manifest.manifest_id().to_string(),
            manifest.manifest_digest().clone(),
            format!(
                "apply repair {} {}",
                manifest.manifest_id(),
                manifest.manifest_digest().as_str()
            ),
        )
        .unwrap()
    };

    let state_path = page_root.path().join(".wenlan/state.json");
    let mut aliased_state =
        serde_json::from_slice::<serde_json::Value>(&std::fs::read(&state_path).unwrap()).unwrap();
    aliased_state["pages"]["page_projection_alias"] =
        aliased_state["pages"]["page_projection_repair"].clone();
    std::fs::write(
        &state_path,
        serde_json::to_vec_pretty(&aliased_state).unwrap(),
    )
    .unwrap();
    let alias_error = crate::repair::apply_repair_with_pages(
        &db,
        &store,
        exact_request(),
        Some(page_root.path()),
        1_721_000_001,
    )
    .await
    .unwrap_err();
    assert!(alias_error.to_string().contains("repair_target_stale"));
    assert_eq!(std::fs::read(&projected_path).unwrap(), before_projection);
    aliased_state["pages"]
        .as_object_mut()
        .unwrap()
        .remove("page_projection_alias");
    std::fs::write(
        &state_path,
        serde_json::to_vec_pretty(&aliased_state).unwrap(),
    )
    .unwrap();

    std::fs::write(&projected_path, b"user changed the target after planning").unwrap();
    let stale = crate::repair::apply_repair_with_pages(
        &db,
        &store,
        exact_request(),
        Some(page_root.path()),
        1_721_000_001,
    )
    .await
    .unwrap_err();
    assert!(stale.to_string().contains("repair_target_stale"));
    assert_eq!(
        std::fs::read(&projected_path).unwrap(),
        b"user changed the target after planning"
    );
    std::fs::write(&projected_path, &before_projection).unwrap();

    let apply_receipt = crate::repair::apply_repair_with_pages(
        &db,
        &store,
        exact_request(),
        Some(page_root.path()),
        1_721_000_001,
    )
    .await
    .unwrap();

    assert_eq!(
        std::fs::read(page_root.path().join("unrelated.txt")).unwrap(),
        b"leave me alone"
    );
    let projected = std::fs::read_to_string(&projected_path).unwrap();
    assert!(projected.contains("origin_version: 2"));
    assert!(projected.contains("canonical body"));

    std::fs::write(
        page_root.path().join("unrelated.txt"),
        b"changed after apply",
    )
    .unwrap();
    let general = runner()
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::General), None),
            Some(page_root.path()),
            true,
        )
        .await
        .unwrap();
    let manifest_dir = store.manifest_dir(manifest.manifest_id()).unwrap();
    let verification_receipt = manifest_dir.join("verification-receipt.json");
    let receipt_at_begin = verification_receipt.clone();
    let pending = manifest_dir.join(".apply-receipt.json.pending");
    std::fs::hard_link(manifest_dir.join("apply-receipt.json"), &pending).unwrap();
    let error = with_repair_verification_test_control(
        RepairVerificationTestControl {
            after_begin: Some(Box::new(move || {
                assert!(!receipt_at_begin.exists());
            })),
            ..Default::default()
        },
        crate::repair::record_repair_verification(
            &db,
            &store,
            VerifyRepairRequest::try_new_general_only(
                manifest.manifest_id().to_string(),
                manifest.manifest_digest().clone(),
                apply_receipt.receipt_digest().clone(),
                general,
            )
            .unwrap(),
            Some(page_root.path()),
            1_721_000_002,
        ),
    )
    .await
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("repair_non_target_state_changed"));
    assert!(!verification_receipt.exists());
    assert!(pending.is_file());
    assert_repair_verification_transaction_reusable(&db).await;
}

#[tokio::test]
#[cfg_attr(not(unix), ignore = "repair artifacts are unix-only")]
async fn page_projection_manifests_from_one_plan_apply_sequentially() {
    use crate::db::repair_verification::{
        with_repair_verification_test_control, RepairVerificationDbContender,
        RepairVerificationTestControl,
    };
    use crate::lint::{
        context::{CancellationToken, LintClock},
        runner::LintRunner,
    };
    use wenlan_types::{
        lint::{LintProfile, LintQuery},
        repair::{ApplyRepairRequest, RepairTarget, RepairWriter, VerifyRepairRequest},
    };

    let (db, _dir) = crate::db::tests::test_db().await;
    let page_root = tempfile::tempdir().unwrap();
    let now = "2026-07-15T00:00:00Z";
    let shared_sources = ["mem_projection_shared"];
    for (id, title) in [
        ("page_projection_first", "Projection First"),
        ("page_projection_second", "Projection Second"),
    ] {
        db.insert_page(
            id,
            title,
            None,
            "canonical body",
            None,
            Some("work"),
            &shared_sources,
            now,
        )
        .await
        .unwrap();
        let page = db.get_page(id).await.unwrap().unwrap();
        crate::export::knowledge::KnowledgeProjectionWrite::new(
            page_root.path().to_path_buf(),
            &db,
        )
        .write_page(&page)
        .unwrap();
    }
    let shared_stub = page_root
        .path()
        .join("_sources")
        .join("mem_projection_shared.md");
    std::fs::write(&shared_stub, b"shared provenance canary").unwrap();
    db.test_primary_session()
        .await
        .execute(
            "UPDATE pages SET version=version+1
              WHERE id IN ('page_projection_first','page_projection_second')",
            (),
        )
        .await
        .unwrap();

    let runner = || LintRunner::new(LintClock::fixed(), CancellationToken::new());
    let general = runner()
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::General), None),
            Some(page_root.path()),
            true,
        )
        .await
        .unwrap();
    let deep = runner()
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::Deep), None),
            Some(page_root.path()),
            true,
        )
        .await
        .unwrap();
    let repair_root = tempfile::tempdir().unwrap();
    let store = crate::repair::RepairArtifactStore::new(repair_root.path().to_path_buf());
    let plan = prepare_repair_plan(
        &db,
        &store,
        RepairPlanRequest::try_new(RepairLintScope::global(), general, Some(deep)).unwrap(),
        Some(page_root.path()),
        1_721_000_000,
    )
    .await
    .unwrap();
    let mut manifests = plan
        .entries()
        .iter()
        .filter_map(|entry| match entry.resolution() {
            RepairResolution::Ready { manifest }
                if manifest.writer() == RepairWriter::RegeneratePageProjection =>
            {
                Some(manifest.as_ref().clone())
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    manifests.sort_by_key(|manifest| match manifest.target() {
        RepairTarget::PageProjection { page_id, .. } => page_id.clone(),
        _ => unreachable!("filtered page projection manifest"),
    });
    assert_eq!(manifests.len(), 2);

    for (offset, manifest) in manifests.iter().enumerate() {
        let request = ApplyRepairRequest::try_new(
            manifest.manifest_id().to_string(),
            manifest.manifest_digest().clone(),
            format!(
                "apply repair {} {}",
                manifest.manifest_id(),
                manifest.manifest_digest().as_str()
            ),
        )
        .unwrap();
        let receipt = crate::repair::apply_repair_with_pages(
            &db,
            &store,
            request,
            Some(page_root.path()),
            1_721_000_001 + i64::try_from(offset).unwrap(),
        )
        .await
        .unwrap();
        let general = runner()
            .run(
                &db,
                &LintQuery::new(Some(LintProfile::General), None),
                Some(page_root.path()),
                true,
            )
            .await
            .unwrap();
        let projection_lock = page_root.path().join(".wenlan/.projection.lock");
        let before_lock = projection_lock.clone();
        let contender = RepairVerificationDbContender::new(&db);
        let start_contender = contender.clone();
        let contender_before_persist = contender.clone();
        let contender_after_persist = contender.clone();
        with_repair_verification_test_control(
            RepairVerificationTestControl {
                after_begin: Some(Box::new(move || {
                    start_contender.start_and_observe_pending();
                })),
                before_receipt_persist: Some(Box::new(move || {
                    contender_before_persist.assert_pending();
                    let lock = std::fs::OpenOptions::new()
                        .read(true)
                        .write(true)
                        .open(&before_lock)
                        .unwrap();
                    assert_eq!(
                        fs2::FileExt::try_lock_exclusive(&lock).unwrap_err().kind(),
                        std::io::ErrorKind::WouldBlock
                    );
                })),
                after_receipt_persist: Some(Box::new(move || {
                    contender_after_persist.assert_pending();
                    let lock = std::fs::OpenOptions::new()
                        .read(true)
                        .write(true)
                        .open(&projection_lock)
                        .unwrap();
                    assert_eq!(
                        fs2::FileExt::try_lock_exclusive(&lock).unwrap_err().kind(),
                        std::io::ErrorKind::WouldBlock
                    );
                })),
                ..Default::default()
            },
            crate::repair::record_repair_verification(
                &db,
                &store,
                VerifyRepairRequest::try_new_general_only(
                    manifest.manifest_id().to_string(),
                    manifest.manifest_digest().clone(),
                    receipt.receipt_digest().clone(),
                    general,
                )
                .unwrap(),
                Some(page_root.path()),
                1_721_000_003 + i64::try_from(offset).unwrap(),
            ),
        )
        .await
        .unwrap();
        contender.assert_entered_after_verification();
    }

    for path in ["projection-first.md", "projection-second.md"] {
        let projected = std::fs::read_to_string(page_root.path().join(path)).unwrap();
        assert!(projected.contains("origin_version: 2"));
    }
    assert_eq!(
        std::fs::read(shared_stub).unwrap(),
        b"shared provenance canary"
    );
}

#[tokio::test]
#[cfg_attr(not(unix), ignore = "repair artifacts are unix-only")]
async fn page_projection_apply_preserves_non_target_shared_state_on_effect_escape() {
    use crate::lint::{
        context::{CancellationToken, LintClock},
        runner::LintRunner,
    };
    use wenlan_types::{
        lint::{LintProfile, LintQuery},
        repair::{ApplyRepairRequest, RepairTarget, RepairWriter},
    };

    let (db, _dir) = crate::db::tests::test_db().await;
    let page_root = tempfile::tempdir().unwrap();
    let now = "2026-07-15T00:00:00Z";
    for (id, title) in [
        ("page_projection_target", "Projection Target"),
        ("page_projection_other", "Projection Other"),
    ] {
        db.insert_page(
            id,
            title,
            None,
            "canonical body",
            None,
            Some("work"),
            &[],
            now,
        )
        .await
        .unwrap();
        let page = db.get_page(id).await.unwrap().unwrap();
        crate::export::knowledge::KnowledgeProjectionWrite::new(
            page_root.path().to_path_buf(),
            &db,
        )
        .write_page(&page)
        .unwrap();
    }
    db.test_primary_session()
        .await
        .execute(
            "UPDATE pages SET version=version+1 WHERE id='page_projection_target'",
            (),
        )
        .await
        .unwrap();

    let runner = || LintRunner::new(LintClock::fixed(), CancellationToken::new());
    let general = runner()
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::General), None),
            Some(page_root.path()),
            true,
        )
        .await
        .unwrap();
    let deep = runner()
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::Deep), None),
            Some(page_root.path()),
            true,
        )
        .await
        .unwrap();
    let repair_root = tempfile::tempdir().unwrap();
    let store = crate::repair::RepairArtifactStore::new(repair_root.path().to_path_buf());
    let plan = prepare_repair_plan(
        &db,
        &store,
        RepairPlanRequest::try_new(RepairLintScope::global(), general, Some(deep)).unwrap(),
        Some(page_root.path()),
        1_721_000_000,
    )
    .await
    .unwrap();
    let manifest = plan
        .entries()
        .iter()
        .find_map(|entry| match entry.resolution() {
            RepairResolution::Ready { manifest }
                if manifest.writer() == RepairWriter::RegeneratePageProjection
                    && matches!(
                        manifest.target(),
                        RepairTarget::PageProjection { page_id, .. }
                            if page_id == "page_projection_target"
                    ) =>
            {
                Some(manifest.as_ref().clone())
            }
            _ => None,
        })
        .expect("target projection manifest");

    let state_path = page_root.path().join(".wenlan/state.json");
    let mut state =
        serde_json::from_slice::<serde_json::Value>(&std::fs::read(&state_path).unwrap()).unwrap();
    state["pages"]["page_projection_other"]["canary"] =
        serde_json::Value::String("preserve me".to_string());
    std::fs::write(&state_path, serde_json::to_vec_pretty(&state).unwrap()).unwrap();

    let request = ApplyRepairRequest::try_new(
        manifest.manifest_id().to_string(),
        manifest.manifest_digest().clone(),
        format!(
            "apply repair {} {}",
            manifest.manifest_id(),
            manifest.manifest_digest().as_str()
        ),
    )
    .unwrap();
    let error = crate::repair::apply_repair_with_pages(
        &db,
        &store,
        request,
        Some(page_root.path()),
        1_721_000_001,
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("repair_effect_escape"));

    let restored =
        serde_json::from_slice::<serde_json::Value>(&std::fs::read(&state_path).unwrap()).unwrap();
    assert_eq!(
        restored["pages"]["page_projection_other"]["canary"],
        serde_json::Value::String("preserve me".to_string())
    );
    let target = std::fs::read_to_string(page_root.path().join("projection-target.md")).unwrap();
    assert!(target.contains("origin_version: 1"));
}

#[tokio::test]
#[cfg_attr(not(unix), ignore = "repair artifacts are unix-only")]
async fn page_projection_invalid_pending_with_changed_target_fails_closed() {
    use crate::lint::{
        context::{CancellationToken, LintClock},
        runner::LintRunner,
    };
    use wenlan_types::{
        lint::{LintProfile, LintQuery},
        repair::{ApplyRepairRequest, RepairTarget, RepairWriter},
    };

    let (db, _dir) = crate::db::tests::test_db().await;
    let page_root = tempfile::tempdir().unwrap();
    let now = "2026-07-15T00:00:00Z";
    for (id, title) in [
        ("page_projection_first", "Projection First"),
        ("page_projection_second", "Projection Second"),
    ] {
        db.insert_page(
            id,
            title,
            None,
            "canonical body",
            None,
            Some("work"),
            &[],
            now,
        )
        .await
        .unwrap();
        let page = db.get_page(id).await.unwrap().unwrap();
        crate::export::knowledge::KnowledgeProjectionWrite::new(
            page_root.path().to_path_buf(),
            &db,
        )
        .write_page(&page)
        .unwrap();
    }
    db.test_primary_session()
        .await
        .execute(
            "UPDATE pages SET version=version+1
              WHERE id IN ('page_projection_first','page_projection_second')",
            (),
        )
        .await
        .unwrap();

    let runner = || LintRunner::new(LintClock::fixed(), CancellationToken::new());
    let general = runner()
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::General), None),
            Some(page_root.path()),
            true,
        )
        .await
        .unwrap();
    let deep = runner()
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::Deep), None),
            Some(page_root.path()),
            true,
        )
        .await
        .unwrap();
    let repair_root = tempfile::tempdir().unwrap();
    let store = crate::repair::RepairArtifactStore::new(repair_root.path().to_path_buf());
    let plan = prepare_repair_plan(
        &db,
        &store,
        RepairPlanRequest::try_new(RepairLintScope::global(), general, Some(deep)).unwrap(),
        Some(page_root.path()),
        1_721_000_000,
    )
    .await
    .unwrap();
    let mut manifests = plan
        .entries()
        .iter()
        .filter_map(|entry| match entry.resolution() {
            RepairResolution::Ready { manifest }
                if manifest.writer() == RepairWriter::RegeneratePageProjection =>
            {
                Some(manifest.as_ref().clone())
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    manifests.sort_by_key(|manifest| match manifest.target() {
        RepairTarget::PageProjection { page_id, .. } => page_id.clone(),
        _ => unreachable!("filtered page projection manifest"),
    });
    assert_eq!(manifests.len(), 2);
    let exact_request = |manifest: &wenlan_types::repair::RepairManifest| {
        ApplyRepairRequest::try_new(
            manifest.manifest_id().to_string(),
            manifest.manifest_digest().clone(),
            format!(
                "apply repair {} {}",
                manifest.manifest_id(),
                manifest.manifest_digest().as_str()
            ),
        )
        .unwrap()
    };

    crate::repair::apply_repair_with_pages(
        &db,
        &store,
        exact_request(&manifests[0]),
        Some(page_root.path()),
        1_721_000_001,
    )
    .await
    .unwrap();

    let second_page = db
        .get_page("page_projection_second")
        .await
        .unwrap()
        .unwrap();
    crate::export::knowledge::KnowledgeProjectionWrite::new(page_root.path().to_path_buf(), &db)
        .write_page(&second_page)
        .unwrap();
    let second_path = page_root.path().join("projection-second.md");
    let user_edit = b"user edit after interrupted apply";
    std::fs::write(&second_path, user_edit).unwrap();
    let pending_path = store
        .manifest_dir(manifests[1].manifest_id())
        .unwrap()
        .join(".apply-receipt.json.pending");
    std::fs::write(&pending_path, b"interrupted").unwrap();

    let error = crate::repair::apply_repair_with_pages(
        &db,
        &store,
        exact_request(&manifests[1]),
        Some(page_root.path()),
        1_721_000_002,
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("repair_apply_recovery_required"));

    let state = serde_json::from_slice::<serde_json::Value>(
        &std::fs::read(page_root.path().join(".wenlan/state.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(state["pages"]["page_projection_first"]["version"], 2);
    assert_eq!(state["pages"]["page_projection_second"]["version"], 2);
    assert_eq!(std::fs::read(second_path).unwrap(), user_edit);
    assert!(pending_path.exists());
}
