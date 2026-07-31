use super::config::{RuntimeConfigSnapshot, RuntimeRunConfig};
use super::{
    ProviderClass, RerankerPath, RuntimeObservation, RuntimeReadiness, StatusFilesObservation,
    WorkingMemoryObservation,
};
use crate::db::tests::test_db;
use crate::lint::context::{CancellationToken, LintClock};
use crate::lint::runner::LintRunner;
use wenlan_types::lint::{
    LintApplicability, LintMetricCode, LintMetricValue, LintOutcome, LintPrecondition, LintQuery,
    LintSummaryCode,
};

const SCHEMA: &str = "runtime.schema_contract";
const INDEXES: &str = "runtime.search_index_contract";
const PROVIDERS: &str = "runtime.provider_inventory";
const STATUS: &str = "runtime.status_parity";
const WORKER: &str = "runtime.ingest_worker_liveness";

#[tokio::test]
async fn schema_and_index_fixtures_fail_loud_without_status_rescue() {
    let (db, _temp) = test_db().await;
    let baseline = run(&db, config(RuntimeObservation::open(1))).await;
    assert_eq!(check(&baseline, SCHEMA).outcome(), LintOutcome::Pass);
    assert_eq!(check(&baseline, INDEXES).outcome(), LintOutcome::Pass);

    db.test_primary_session()
        .await
        .execute_batch("DROP TRIGGER memories_fts_insert; DROP INDEX child_vectors_vec_idx;")
        .await
        .unwrap();
    let damaged = run(&db, config(RuntimeObservation::open(1))).await;
    assert_eq!(check(&damaged, INDEXES).outcome(), LintOutcome::Finding);
    assert_eq!(metric(check(&damaged, INDEXES)), 2);

    let failed = run(
        &db,
        config(RuntimeObservation::open(0)).with_query_failure(),
    )
    .await;
    for id in [SCHEMA, INDEXES, STATUS] {
        assert_eq!(
            check(&failed, id).outcome(),
            LintOutcome::FailedToRun,
            "{id}"
        );
    }
    assert_eq!(check(&failed, PROVIDERS).outcome(), LintOutcome::Pass);
    assert_eq!(check(&failed, WORKER).outcome(), LintOutcome::Pass);
}

#[tokio::test]
async fn missing_fts_and_malformed_same_name_search_objects_are_findings() {
    let (db, _temp) = test_db().await;
    db.test_primary_session()
        .await
        .execute_batch(
            "DROP TABLE pages_fts;
         DROP TRIGGER memories_fts_insert;
         CREATE TRIGGER memories_fts_insert AFTER INSERT ON memories BEGIN SELECT 1; END;
         DROP INDEX child_vectors_vec_idx;
         CREATE INDEX child_vectors_vec_idx ON child_vectors(parent_id);",
        )
        .await
        .unwrap();

    let report = run(&db, config(RuntimeObservation::open(0))).await;
    assert_eq!(check(&report, INDEXES).outcome(), LintOutcome::Finding);
    assert!(metric(check(&report, INDEXES)) >= 3);
}

#[tokio::test]
async fn missing_memories_is_schema_finding_with_only_dependent_status_incomplete() {
    let (db, _temp) = test_db().await;
    db.test_primary_session()
        .await
        .execute_batch("DROP TABLE memories;")
        .await
        .unwrap();

    let report = run(&db, config(RuntimeObservation::open(0))).await;
    assert_eq!(check(&report, SCHEMA).outcome(), LintOutcome::Finding);
    assert_eq!(check(&report, INDEXES).outcome(), LintOutcome::Finding);
    assert_eq!(check(&report, STATUS).outcome(), LintOutcome::FailedToRun);
    assert_eq!(check(&report, PROVIDERS).outcome(), LintOutcome::Pass);
    assert_eq!(check(&report, WORKER).outcome(), LintOutcome::Pass);
}

#[tokio::test]
async fn provider_off_and_worker_observations_are_non_mutating_inventory() {
    let (db, _temp) = test_db().await;
    let off = run(&db, config(RuntimeObservation::open(1))).await;
    assert_eq!(check(&off, PROVIDERS).outcome(), LintOutcome::Pass);
    assert_eq!(check(&off, WORKER).outcome(), LintOutcome::Pass);
    assert_eq!(
        metric_value(
            check(&off, WORKER),
            LintMetricCode::WorkingMemoryTelemetryUnavailable,
        ),
        1
    );

    let requested = RuntimeConfigSnapshot::disabled()
        .with_provider_request(ProviderClass::OnDevice, "model-a")
        .with_reranker_request(RerankerPath::Light, "reranker-a");
    let unavailable = RuntimeObservation::closed(1);
    let report = run(
        &db,
        RuntimeRunConfig::for_test(requested, unavailable, None),
    )
    .await;
    assert_eq!(check(&report, PROVIDERS).outcome(), LintOutcome::Finding);
    assert_eq!(check(&report, WORKER).outcome(), LintOutcome::Finding);
}

#[tokio::test]
async fn repair_recovery_suspends_optional_workers_without_inventing_provider_failure() {
    let (db, _temp) = test_db().await;
    let requested = RuntimeConfigSnapshot::disabled()
        .with_provider_request(ProviderClass::OnDevice, "model-a")
        .with_reranker_request(RerankerPath::Light, "reranker-a");
    let suspended = RuntimeObservation::open(1).with_optional_workers_suspended();

    let report = run(&db, RuntimeRunConfig::for_test(requested, suspended, None)).await;
    let provider_inventory = check(&report, PROVIDERS);

    assert_eq!(provider_inventory.outcome(), LintOutcome::Pass);
    assert_eq!(
        provider_inventory.applicability(),
        LintApplicability::ExpectedEmpty
    );
    assert_eq!(
        provider_inventory.precondition(),
        LintPrecondition::ConfiguredOff
    );
    assert_eq!(
        provider_inventory.summary_code(),
        LintSummaryCode::ExpectedEmpty
    );
    assert_eq!(metric(provider_inventory), 0);
    assert_eq!(
        metric_value(provider_inventory, LintMetricCode::ObservedRecords),
        0
    );
}

#[tokio::test]
async fn repair_recovery_reports_an_optional_worker_that_is_still_running() {
    let (db, _temp) = test_db().await;
    let requested =
        RuntimeConfigSnapshot::disabled().with_provider_request(ProviderClass::OnDevice, "model-a");
    let suspended = RuntimeObservation::open(1)
        .with_optional_workers_suspended()
        .with_provider(ProviderClass::OnDevice, "model-a", RuntimeReadiness::Ready);

    let report = run(&db, RuntimeRunConfig::for_test(requested, suspended, None)).await;

    assert_eq!(check(&report, PROVIDERS).outcome(), LintOutcome::Finding);
    assert_eq!(metric(check(&report, PROVIDERS)), 1);
}

#[tokio::test]
async fn repair_recovery_reports_an_optional_worker_that_failed_to_start() {
    let (db, _temp) = test_db().await;
    let requested =
        RuntimeConfigSnapshot::disabled().with_provider_request(ProviderClass::OnDevice, "model-a");
    let suspended = RuntimeObservation::open(1)
        .with_optional_workers_suspended()
        .with_provider(ProviderClass::OnDevice, "model-a", RuntimeReadiness::Failed);

    let report = run(&db, RuntimeRunConfig::for_test(requested, suspended, None)).await;

    assert_eq!(check(&report, PROVIDERS).outcome(), LintOutcome::Finding);
    assert_eq!(metric(check(&report, PROVIDERS)), 1);
}

#[tokio::test]
async fn producer_commit_is_nullable_receipt_inventory() {
    let (db, _temp) = test_db().await;
    let absent = run(&db, config(RuntimeObservation::open(1))).await;
    assert!(absent.producer_receipt().runtime_commit().is_none());

    let present = run(
        &db,
        RuntimeRunConfig::for_test(
            RuntimeConfigSnapshot::disabled(),
            RuntimeObservation::open(1),
            Some("0123456789abcdef0123456789abcdef01234567"),
        ),
    )
    .await;
    assert_eq!(
        present
            .producer_receipt()
            .runtime_commit()
            .unwrap()
            .as_str(),
        "0123456789abcdef0123456789abcdef01234567"
    );
}

fn config(observation: RuntimeObservation) -> RuntimeRunConfig {
    RuntimeRunConfig::for_test(RuntimeConfigSnapshot::disabled(), observation, None)
}

async fn run(db: &crate::db::MemoryDB, config: RuntimeRunConfig) -> wenlan_types::lint::LintReport {
    run_at(db, config, 0).await
}

async fn run_at(
    db: &crate::db::MemoryDB,
    config: RuntimeRunConfig,
    epoch_seconds: i64,
) -> wenlan_types::lint::LintReport {
    LintRunner::new(LintClock::fixed_at(epoch_seconds), CancellationToken::new())
        .with_test_runtime_config(config)
        .run(
            db,
            &LintQuery {
                profile: None,
                space: None,
            },
            None,
            false,
        )
        .await
        .unwrap()
}

fn check<'a>(
    report: &'a wenlan_types::lint::LintReport,
    id: &str,
) -> &'a wenlan_types::lint::LintCheckResult {
    report
        .checks()
        .iter()
        .find(|check| check.check_id() == id)
        .unwrap()
}

fn metric(result: &wenlan_types::lint::LintCheckResult) -> u64 {
    metric_value(result, LintMetricCode::AffectedRecords)
}

fn metric_value(result: &wenlan_types::lint::LintCheckResult, code: LintMetricCode) -> u64 {
    result
        .metrics()
        .iter()
        .find_map(|metric| match (metric.code(), metric.value()) {
            (observed, LintMetricValue::Count { value }) if observed == code => Some(*value),
            _ => None,
        })
        .unwrap()
}

#[path = "runtime_readiness_test.rs"]
mod readiness;
