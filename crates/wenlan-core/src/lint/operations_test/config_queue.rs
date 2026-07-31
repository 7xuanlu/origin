use super::{check, metric, run, run_at, source, NOW, QUEUE, SOURCE_CONFIG};
use crate::db::tests::test_db;
use crate::lint::test_support::assert_no_privacy_canaries;
use wenlan_types::lint::{LintMetricCode, LintOutcome};
use wenlan_types::sources::SyncStatus;

#[tokio::test]
async fn reclaimed_paused_row_retains_retry_diagnostics_and_is_valid() {
    let (db, _tmp) = test_db().await;
    db.enqueue_document("source", "document", Some("hash"))
        .await
        .unwrap();
    db.claim_next_pending().await.unwrap().unwrap();
    db.checkpoint_chunk("source", "document", 2).await.unwrap();
    db.mark_paused("source", "document", "transient", Some(0))
        .await
        .unwrap();
    let reclaimed = db.claim_next_pending().await.unwrap().unwrap();
    assert_eq!(reclaimed.status, "in_progress");
    assert_eq!(reclaimed.next_retry_at, Some(0));
    assert_eq!(reclaimed.error_detail.as_deref(), Some("transient"));

    let report = run_at(&db, &[], chrono::Utc::now().timestamp()).await;

    assert_eq!(check(&report, QUEUE).outcome(), LintOutcome::Pass);
    assert_eq!(
        metric(
            check(&report, QUEUE),
            LintMetricCode::OperationInvalidStates
        ),
        0
    );
}

#[tokio::test]
async fn source_config_failures_are_findings_with_redacted_evidence() {
    // Given
    let (db, _tmp) = test_db().await;
    let sources = vec![
        source(
            "duplicate",
            "/private/source/CANARY_IMPORT_7f31.md",
            SyncStatus::Active,
        ),
        source(
            "duplicate",
            "private-host-7f31.invalid",
            SyncStatus::Error("nested error at /private/CANARY_ERROR_7f31".into()),
        ),
    ];

    // When
    let report = run(&db, &sources).await;

    // Then
    let result = check(&report, SOURCE_CONFIG);
    assert_eq!(result.outcome(), LintOutcome::Finding);
    assert_eq!(
        metric(result, LintMetricCode::SourceInvalidConfigurations),
        2
    );
    assert_eq!(metric(result, LintMetricCode::SourceTerminalFailures), 1);
    assert_no_privacy_canaries(&serde_json::to_string(&report).unwrap());
}

#[tokio::test]
async fn source_snapshot_identity_is_canonical_and_semantically_complete() {
    let (db, _tmp) = test_db().await;
    let mut left = source("b", "", SyncStatus::Active);
    left.last_sync_errors = 1;
    let right = source("a", "/configured", SyncStatus::Paused);

    let first = run(&db, &[left.clone(), right.clone()]).await;
    let reordered = run(&db, &[right.clone(), left.clone()]).await;
    assert_eq!(first.config_fingerprint(), reordered.config_fingerprint());
    assert_eq!(
        check(&first, SOURCE_CONFIG).evidence(),
        check(&reordered, SOURCE_CONFIG).evidence()
    );

    let changed_id = run(&db, &[source("c", "", SyncStatus::Active), right.clone()]).await;
    let changed_path = run(&db, &[left.clone(), source("a", "", SyncStatus::Paused)]).await;
    let changed_status = run(&db, &[left, source("a", "/configured", SyncStatus::Active)]).await;
    for changed in [changed_id, changed_path, changed_status] {
        assert_ne!(first.config_fingerprint(), changed.config_fingerprint());
    }
}

#[tokio::test]
async fn retry_boundary_and_queue_ages_use_closed_buckets() {
    // Given
    let (db, _tmp) = test_db().await;
    let conn = db.test_primary_session().await;
    for (id, retry_at, enqueued_at) in
        [("active", NOW + 1, NOW - 10), ("expired", NOW, NOW - 3_600)]
    {
        conn.execute(
            "INSERT INTO document_enrichment_queue
             (source_id,file_path,status,last_completed_chunk,attempt_count,next_retry_at,error_detail,enqueued_at,updated_at)
             VALUES (?1,?1,'paused',0,1,?2,'CANARY_ENV_VALUE_7f31',?3,?3)",
            libsql::params![id, retry_at, enqueued_at],
        ).await.unwrap();
    }
    conn.execute(
        "INSERT INTO document_enrichment_queue
         (source_id,file_path,status,last_completed_chunk,attempt_count,enqueued_at,updated_at)
         VALUES ('old','old','pending',-1,0,?1,?1)",
        libsql::params![NOW - 604_800],
    )
    .await
    .unwrap();
    drop(conn);

    // When
    let report = run(&db, &[]).await;

    // Then
    let result = check(&report, QUEUE);
    assert_eq!(result.outcome(), LintOutcome::Finding);
    assert_eq!(metric(result, LintMetricCode::OperationActiveRetries), 1);
    assert_eq!(metric(result, LintMetricCode::OperationExpiredRetries), 1);
    assert_eq!(metric(result, LintMetricCode::OperationAgeUnderHour), 1);
    assert_eq!(metric(result, LintMetricCode::OperationAgeOneTo24Hours), 1);
    assert_eq!(
        metric(result, LintMetricCode::OperationAgeSevenDaysOrMore),
        1
    );
    let json = serde_json::to_string(&report).unwrap();
    assert!(!json.contains("\"age_seconds\""));
    assert_no_privacy_canaries(&json);
}

#[tokio::test]
async fn queue_affected_records_count_unique_rows() {
    let (db, _tmp) = test_db().await;
    db.test_primary_session().await.execute(
        "INSERT INTO document_enrichment_queue
         (source_id,file_path,status,last_completed_chunk,attempt_count,next_retry_at,error_detail,enqueued_at,updated_at)
         VALUES ('one','one','paused',-2,1,?1,'error',?2,?2)",
        libsql::params![NOW, NOW - 10],
    ).await.unwrap();

    let report = run(&db, &[]).await;
    let result = check(&report, QUEUE);
    assert_eq!(metric(result, LintMetricCode::ObservedRecords), 1);
    assert_eq!(metric(result, LintMetricCode::AffectedRecords), 1);
    assert_eq!(metric(result, LintMetricCode::OperationExpiredRetries), 1);
    assert_eq!(metric(result, LintMetricCode::OperationInvalidStates), 1);
}
