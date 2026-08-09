use super::catalog::{
    catalog, catalog_entry, catalog_for_profile, catalog_group, has_valid_owner, LintCheckGroup,
};
use super::context::{CancellationToken, ExecutionGate, LintClock, PopulationAccounting};
use super::runner::{
    configured_off_results, validate_catalog_results, LintRunError, LintRunner, TestScenario,
};
use crate::db::tests::test_db;
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::Path;
use std::time::Duration;
use wenlan_types::lint::{
    LintApplicability, LintCheckResult, LintCheckResultInput, LintCoverage, LintEvidenceRef,
    LintMetric, LintOutcome, LintPrecondition, LintQuery, LintRecommendationCode, LintSeverity,
    LintSummaryCode, LintValidationMethod, LINT_MAX_EVIDENCE_PER_CHECK,
};

fn result(check_id: &str, outcome: LintOutcome) -> LintCheckResult {
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
    LintCheckResult::try_new(LintCheckResultInput {
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
    })
    .unwrap()
}

#[test]
fn catalog_is_static_ordered_unique_and_group_owned() {
    let entries = catalog();
    let ids = entries.iter().map(|entry| entry.id).collect::<Vec<_>>();
    assert!(ids.windows(2).all(|pair| pair[0] < pair[1]));
    assert_eq!(
        ids.iter().copied().collect::<BTreeSet<_>>().len(),
        ids.len()
    );
    assert!(entries.iter().all(has_valid_owner));
    assert_eq!(
        catalog_for_profile(wenlan_types::lint::LintProfile::General).count(),
        wenlan_types::lint::LINT_GENERAL_CHECK_COUNT
    );
    assert_eq!(
        catalog_for_profile(wenlan_types::lint::LintProfile::Deep).count(),
        wenlan_types::lint::LINT_DEEP_CHECK_COUNT
    );
    for profile in [
        wenlan_types::lint::LintProfile::General,
        wenlan_types::lint::LintProfile::Deep,
    ] {
        assert!(catalog_for_profile(profile).all(|entry| {
            wenlan_types::lint::canonical_gate_effect(profile, entry.id) == Some(entry.gate_effect)
        }));
    }
    let policies: BTreeSet<_> = entries.iter().map(|entry| entry.scope_policy).collect();
    assert_eq!(policies.len(), 4);
}

#[test]
fn catalog_result_bijection_rejects_missing_duplicate_and_unknown_ids() {
    let mut complete = catalog_for_profile(wenlan_types::lint::LintProfile::General)
        .map(|entry| result(entry.id, LintOutcome::Pass))
        .collect::<Vec<_>>();
    validate_catalog_results(&mut complete).unwrap();
    assert!(complete
        .windows(2)
        .all(|pair| pair[0].check_id() < pair[1].check_id()));

    let mut missing = complete.clone();
    missing.pop();
    assert!(validate_catalog_results(&mut missing).is_err());
    let mut duplicate = complete.clone();
    duplicate.push(complete[0].clone());
    assert!(validate_catalog_results(&mut duplicate).is_err());
    let mut unknown = complete;
    unknown[0] = result("pages.unknown", LintOutcome::Pass);
    assert!(validate_catalog_results(&mut unknown).is_err());
}

#[test]
fn fixed_budgets_and_cancellation_fail_closed() {
    assert_eq!(ExecutionGate::RUN_BUDGET, Duration::from_secs(15));
    assert_eq!(ExecutionGate::PAGE_BUDGET, Duration::from_secs(5));
    assert_eq!(ExecutionGate::DEEP_RUN_BUDGET, Duration::from_secs(120));
    assert_eq!(ExecutionGate::DEEP_PAGE_BUDGET, Duration::from_secs(30));
    let token = CancellationToken::new();
    let gate = ExecutionGate::new(token.clone());
    assert!(gate.check(Duration::from_secs(4)).is_ok());
    assert!(gate.check(Duration::from_secs(6)).is_err());
    token.cancel();
    assert!(gate.check(Duration::ZERO).is_err());
}

#[test]
fn population_validation_is_independent_from_bounded_evidence() {
    let mut population = PopulationAccounting::new(101);
    for ordinal in 1..=101 {
        population.validate(ordinal, ordinal <= 100);
    }
    let coverage = population.coverage().unwrap();
    assert_eq!(coverage.denominator(), 101);
    assert_eq!(coverage.evaluated(), 101);
    assert_eq!(population.evidence_ordinals().len(), 1);
    assert!(!coverage.truncated());

    let mut all_defective = PopulationAccounting::new(101);
    for ordinal in 1..=101 {
        all_defective.validate(ordinal, false);
    }
    assert_eq!(all_defective.coverage().unwrap().evaluated(), 101);
    assert_eq!(all_defective.evidence_ordinals().len(), 100);
    assert!(all_defective.coverage().unwrap().truncated());

    let mut interrupted = PopulationAccounting::new(101);
    for ordinal in 1..=100 {
        interrupted.validate(ordinal, true);
    }
    assert!(interrupted.coverage().is_err());
}

#[test]
fn configured_off_checks_are_expected_empty_passes() {
    let mut checks = configured_off_results(LintClock::fixed());
    validate_catalog_results(&mut checks).unwrap();
    assert!(checks.iter().all(|check| {
        check.outcome() == LintOutcome::Pass
            && check.severity() == LintSeverity::Info
            && check.applicability() == LintApplicability::ExpectedEmpty
            && check.precondition() == LintPrecondition::ConfiguredOff
    }));
}

#[tokio::test]
async fn invalid_scope_fails_before_page_scan() {
    let (db, _dir) = test_db().await;
    let query = LintQuery {
        profile: None,
        space: Some("does-not-exist".to_string()),
    };
    let error = LintRunner::new(LintClock::fixed(), CancellationToken::new())
        .run(
            &db,
            &query,
            Some(Path::new("/definitely/missing/page-root")),
            false,
        )
        .await
        .unwrap_err();
    assert!(matches!(error, LintRunError::InvalidScope));
}

#[tokio::test]
async fn fixed_clock_report_is_deterministic() {
    let (db, _dir) = test_db().await;
    let runner = LintRunner::new(LintClock::fixed(), CancellationToken::new());
    let query = LintQuery {
        profile: None,
        space: None,
    };
    let first = runner.run(&db, &query, None, false).await.unwrap();
    let second = runner.run(&db, &query, None, false).await.unwrap();
    let first: Value = serde_json::to_value(first).unwrap();
    let second: Value = serde_json::to_value(second).unwrap();
    assert_eq!(first, second);
    assert!(first["complete"].as_bool().unwrap());
}

#[tokio::test]
async fn deep_profile_is_a_registered_superset_without_duplicate_general_checks() {
    let (db, _dir) = test_db().await;
    let runner = LintRunner::new(LintClock::fixed(), CancellationToken::new());
    let general = runner
        .run(
            &db,
            &LintQuery {
                profile: None,
                space: None,
            },
            None,
            false,
        )
        .await
        .unwrap();
    let deep = runner
        .run(
            &db,
            &LintQuery {
                profile: Some(wenlan_types::lint::LintProfile::Deep),
                space: None,
            },
            None,
            false,
        )
        .await
        .unwrap();

    assert_eq!(general.profile(), wenlan_types::lint::LintProfile::General);
    assert_eq!(deep.profile(), wenlan_types::lint::LintProfile::Deep);
    let general_ids = general
        .checks()
        .iter()
        .map(LintCheckResult::check_id)
        .collect::<BTreeSet<_>>();
    let deep_ids = deep
        .checks()
        .iter()
        .map(LintCheckResult::check_id)
        .collect::<BTreeSet<_>>();
    assert!(general_ids.is_subset(&deep_ids));
    assert!(deep_ids.len() > general_ids.len());
    assert_eq!(deep_ids.len(), deep.checks().len());
    assert!(!deep.complete());
}

#[tokio::test]
async fn cancellation_returns_an_incomplete_report() {
    let (db, _dir) = test_db().await;
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let report = LintRunner::new(LintClock::fixed(), cancellation)
        .run(
            &db,
            &LintQuery {
                profile: None,
                space: None,
            },
            None,
            false,
        )
        .await
        .unwrap();
    assert!(!report.complete());
    let general_checks = catalog_for_profile(wenlan_types::lint::LintProfile::General).count();
    assert_eq!(report.totals().incomplete(), general_checks as u32);
    assert!(report
        .checks()
        .iter()
        .all(|check| check.outcome() == LintOutcome::FailedToRun));
}

#[tokio::test]
async fn cancellation_interrupts_an_in_flight_group() {
    let (db, _dir) = test_db().await;
    let cancellation = CancellationToken::new();
    let cancel = cancellation.clone();
    let task = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(10)).await;
        cancel.cancel();
    });
    let runner = LintRunner::new(LintClock::capture(), cancellation)
        .with_test_scenario(TestScenario::BlockedGroup);
    let run = runner.run(
        &db,
        &LintQuery {
            profile: None,
            space: None,
        },
        None,
        false,
    );
    let report = tokio::time::timeout(Duration::from_millis(500), run)
        .await
        .expect("in-flight cancellation must interrupt the active group")
        .unwrap();
    task.await.unwrap();
    assert!(!report.complete());
    assert!(report
        .checks()
        .iter()
        .all(|check| check.outcome() == LintOutcome::FailedToRun));
}

#[tokio::test]
async fn run_budget_interrupts_an_in_flight_group() {
    let (db, _dir) = test_db().await;
    let runner = LintRunner::new(LintClock::capture(), CancellationToken::new())
        .with_test_scenario(TestScenario::BlockedGroup)
        .with_test_run_timeout(Duration::from_millis(10));
    let report = tokio::time::timeout(
        Duration::from_millis(500),
        runner.run(
            &db,
            &LintQuery {
                profile: None,
                space: None,
            },
            None,
            false,
        ),
    )
    .await
    .expect("run budget must interrupt the active group")
    .unwrap();
    assert!(!report.complete());
    assert!(report
        .checks()
        .iter()
        .all(|check| check.outcome() == LintOutcome::FailedToRun));
}

#[tokio::test]
async fn deep_terminal_paths_preserve_advisory_catalog_gates() {
    let (db, _dir) = test_db().await;
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let report = LintRunner::new(LintClock::fixed(), cancellation)
        .run(
            &db,
            &LintQuery {
                profile: Some(wenlan_types::lint::LintProfile::Deep),
                space: None,
            },
            None,
            false,
        )
        .await
        .expect("deep cancellation must return a typed incomplete report");
    assert!(!report.complete());
    assert_eq!(
        report.checks().len(),
        wenlan_types::lint::LINT_DEEP_CHECK_COUNT
    );
    assert!(report
        .checks()
        .iter()
        .filter(|check| check.check_id().contains(".semantic."))
        .all(|check| check.gate_effect() == wenlan_types::lint::LintGateEffect::Advisory));

    let report = LintRunner::new(LintClock::fixed(), CancellationToken::new())
        .run(
            &db,
            &LintQuery {
                profile: Some(wenlan_types::lint::LintProfile::Deep),
                space: None,
            },
            Some(Path::new("/definitely/missing/page-root")),
            true,
        )
        .await
        .expect("deep Page scan failure must stay inside the report");
    assert!(!report.complete());
}

#[tokio::test]
async fn runner_mixes_warning_and_real_query_failure_without_leaking_error() {
    let (db, _dir) = test_db().await;
    let report = LintRunner::new(LintClock::fixed(), CancellationToken::new())
        .with_test_scenario(TestScenario::MixedQueryFailure)
        .run(
            &db,
            &LintQuery {
                profile: None,
                space: None,
            },
            None,
            false,
        )
        .await
        .unwrap();
    let general_checks = catalog_for_profile(wenlan_types::lint::LintProfile::General).count();
    assert_eq!(report.totals().checks(), general_checks as u32);
    assert_eq!(
        report.totals().passed(),
        u32::try_from(general_checks - 2).unwrap()
    );
    assert_eq!(report.totals().findings(), 1);
    assert_eq!(report.totals().incomplete(), 1);
    assert!(!report.complete());
    let json = serde_json::to_string(&report).unwrap();
    assert!(!json.contains("CANARY_RAW_QUERY_ERROR_7f31"));
}

#[tokio::test]
async fn page_group_timeout_with_incomplete_enumeration_fails_the_report() {
    let (db, _dir) = test_db().await;
    let report = LintRunner::new(LintClock::fixed(), CancellationToken::new())
        .with_test_scenario(TestScenario::PageGroupTimeout)
        .run(
            &db,
            &LintQuery {
                profile: None,
                space: None,
            },
            None,
            false,
        )
        .await
        .unwrap();
    assert!(!report.complete());
    assert_eq!(
        report.totals().incomplete(),
        u32::try_from(catalog_group(LintCheckGroup::Pages).count()).unwrap()
    );
    assert!(report.checks().iter().all(|check| {
        let page = catalog_entry(check.check_id()).unwrap().group == LintCheckGroup::Pages;
        (check.outcome() == LintOutcome::FailedToRun) == page
    }));
    assert!(report
        .checks()
        .iter()
        .all(|check| check.coverage().denominator() == 101));
}

#[tokio::test]
async fn selected_scope_enforces_scoped_and_global_denominators() {
    let (db, _dir) = test_db().await;
    db.test_primary_session()
        .await
        .execute(
            "INSERT INTO spaces (id, name, created_at, updated_at) VALUES ('lint-space', 'alpha', 1, 1)",
            (),
        )
        .await
        .unwrap();
    let report = LintRunner::new(LintClock::fixed(), CancellationToken::new())
        .with_test_scenario(TestScenario::ScopedDenominators)
        .run(
            &db,
            &LintQuery {
                profile: None,
                space: Some("alpha".to_string()),
            },
            None,
            false,
        )
        .await
        .unwrap();
    assert_eq!(
        report.scope().kind(),
        wenlan_types::lint::LintScopeKind::Registered
    );
    let scoped = report
        .checks()
        .iter()
        .find(|check| check.check_id() == "pages.db.partitions")
        .unwrap();
    let global = report
        .checks()
        .iter()
        .find(|check| check.check_id() == "pages.projection.manifest_inventory")
        .unwrap();
    assert_eq!(scoped.coverage().denominator(), 3);
    assert_eq!(global.coverage().denominator(), 9);
    assert!(global.evidence().is_empty());
}

#[test]
fn every_non_complete_outcome_makes_the_report_incomplete() {
    for outcome in [
        LintOutcome::NotRunPrerequisite,
        LintOutcome::InconsistentSnapshot,
        LintOutcome::FailedToRun,
    ] {
        let report = super::runner::synthetic_report(vec![result(catalog()[0].id, outcome)]);
        assert!(!report.complete());
        assert_eq!(report.totals().incomplete(), 1);
    }
}
