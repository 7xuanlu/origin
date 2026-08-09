use super::{uses_cross_store, uses_filesystem};
use crate::db::tests::test_db;
use crate::lint::context::{CancellationToken, LintClock};
use crate::lint::runner::{LintRunner, TestSyncPoint, TestSynchronization};
use crate::lint::test_support::{DbBytesFingerprint, PageBytesFingerprint};
use std::sync::Arc;
use std::time::Duration;
use wenlan_types::lint::{LintOutcome, LintQuery};

#[path = "integration_test_support.rs"]
mod support;
use support::{
    assert_selective_inconsistency, insert_page, page, run_receipt_drift, semantic_fingerprint,
    ReceiptDrift,
};

#[tokio::test]
async fn markdown_first_partial_write_is_incomplete_without_blocking() {
    let (db, _tmp) = test_db().await;
    let page_root = tempfile::tempdir().unwrap();
    let projection = crate::export::knowledge::KnowledgeProjectionWrite::new(
        page_root.path().to_path_buf(),
        &db,
    );
    projection.write_page(&page("page-md-first")).unwrap();
    let report = tokio::time::timeout(
        Duration::from_secs(2),
        LintRunner::new(LintClock::fixed(), CancellationToken::new()).run(
            &db,
            &LintQuery {
                profile: None,
                space: None,
            },
            Some(page_root.path()),
            true,
        ),
    )
    .await
    .expect("lint must not wait for an active Page writer")
    .unwrap();
    assert_selective_inconsistency(&report, uses_cross_store);
    drop(projection);
    assert!(!db.page_projection_tracker().sample().has_active_writes());
}

#[tokio::test]
async fn db_first_partial_write_is_incomplete_without_blocking() {
    let (db, _tmp) = test_db().await;
    let page_root = tempfile::tempdir().unwrap();
    let projection = crate::export::knowledge::KnowledgeProjectionWrite::new(
        page_root.path().to_path_buf(),
        &db,
    );
    insert_page(&db, "page-db-first").await;
    let report = tokio::time::timeout(
        Duration::from_secs(2),
        LintRunner::new(LintClock::fixed(), CancellationToken::new()).run(
            &db,
            &LintQuery {
                profile: None,
                space: None,
            },
            Some(page_root.path()),
            true,
        ),
    )
    .await
    .expect("lint must not wait for a DB-first Page writer")
    .unwrap();
    assert_selective_inconsistency(&report, uses_cross_store);
    drop(projection);
}

#[tokio::test]
async fn completed_real_writer_between_receipts_is_detected() {
    let (db, _tmp) = test_db().await;
    let db = Arc::new(db);
    let page_root = tempfile::tempdir().unwrap();
    let root = page_root.path().to_path_buf();
    let (synchronization, control) = TestSynchronization::new(TestSyncPoint::AfterTrackerSample);
    let runner_db = Arc::clone(&db);
    let runner_root = root.clone();
    let task = tokio::spawn(async move {
        LintRunner::new(LintClock::fixed(), CancellationToken::new())
            .with_test_synchronization(synchronization)
            .run(
                &runner_db,
                &LintQuery {
                    profile: None,
                    space: None,
                },
                Some(&runner_root),
                true,
            )
            .await
    });
    control.wait_until_reached().await;
    let projection = crate::export::knowledge::KnowledgeProjectionWrite::new(root, &db);
    projection.write_page(&page("page-complete")).unwrap();
    insert_page(&db, "page-complete").await;
    drop(projection);
    control.resume().await;
    let report = task.await.unwrap().unwrap();
    assert_selective_inconsistency(&report, uses_cross_store);
}

#[tokio::test]
async fn page_only_and_db_only_drift_have_distinct_suppression_sets() {
    let page_report = run_receipt_drift(ReceiptDrift::Page).await;
    assert_selective_inconsistency(&page_report, uses_filesystem);

    let db_report = run_receipt_drift(ReceiptDrift::Database).await;
    assert_selective_inconsistency(&db_report, |_| true);
    assert_ne!(
        db_report.snapshots().db().analysis_digest(),
        db_report.snapshots().db().post_run_digest().unwrap()
    );
}

#[tokio::test]
async fn database_drift_is_incomplete_when_page_projection_is_disabled() {
    let (db, _tmp) = test_db().await;
    let db = Arc::new(db);
    insert_page(&db, "same-count-update").await;
    let (synchronization, control) = TestSynchronization::new(TestSyncPoint::BeforeReceipts);
    let runner_db = Arc::clone(&db);
    let task = tokio::spawn(async move {
        LintRunner::new(LintClock::fixed(), CancellationToken::new())
            .with_test_synchronization(synchronization)
            .run(
                &runner_db,
                &LintQuery {
                    profile: None,
                    space: None,
                },
                None,
                false,
            )
            .await
    });
    control.wait_until_reached().await;
    db.test_primary_session()
        .await
        .execute(
            "UPDATE pages SET title='changed' WHERE id='same-count-update'",
            (),
        )
        .await
        .unwrap();
    control.resume().await;

    let report = task.await.unwrap().unwrap();
    assert_selective_inconsistency(&report, |_| true);
    assert_ne!(
        report.snapshots().db().analysis_digest(),
        report.snapshots().db().post_run_digest().unwrap()
    );
}

#[tokio::test]
async fn whole_run_budget_includes_after_scan_receipts() {
    let (db, _tmp) = test_db().await;
    let db = Arc::new(db);
    let page_root = tempfile::tempdir().unwrap();
    let root = page_root.path().to_path_buf();
    let (synchronization, control) = TestSynchronization::new(TestSyncPoint::BeforeReceipts);
    let runner_db = Arc::clone(&db);
    let task = tokio::spawn(async move {
        LintRunner::new(LintClock::capture(), CancellationToken::new())
            .with_test_run_timeout(Duration::from_millis(50))
            .with_test_synchronization(synchronization)
            .run(
                &runner_db,
                &LintQuery {
                    profile: None,
                    space: None,
                },
                Some(&root),
                true,
            )
            .await
    });
    control.wait_until_reached().await;
    tokio::time::sleep(Duration::from_millis(75)).await;
    control.resume().await;

    let report = task.await.unwrap().unwrap();
    assert!(!report.complete());
    assert!(report
        .checks()
        .iter()
        .all(|check| check.outcome() == LintOutcome::FailedToRun));
}

#[tokio::test]
async fn non_page_work_before_receipts_does_not_consume_page_budget() {
    let (db, _tmp) = test_db().await;
    let db = Arc::new(db);
    let page_root = tempfile::tempdir().unwrap();
    let root = page_root.path().to_path_buf();
    let (synchronization, control) = TestSynchronization::new(TestSyncPoint::BeforeReceipts);
    let runner_db = Arc::clone(&db);
    let task = tokio::spawn(async move {
        LintRunner::new(LintClock::capture(), CancellationToken::new())
            .with_test_run_timeout(Duration::from_secs(2))
            .with_test_page_timeout(Duration::from_millis(100))
            .with_test_synchronization(synchronization)
            .run(
                &runner_db,
                &LintQuery {
                    profile: None,
                    space: None,
                },
                Some(&root),
                true,
            )
            .await
    });
    control.wait_until_reached().await;
    tokio::time::sleep(Duration::from_millis(150)).await;
    control.resume().await;

    let report = task.await.unwrap().unwrap();
    assert!(report.checks().iter().all(|check| {
        !uses_filesystem(check.check_id()) || check.outcome() != LintOutcome::FailedToRun
    }));
}

#[tokio::test]
async fn real_writer_completion_clears_active_state_within_two_seconds() {
    let (db, _tmp) = test_db().await;
    let tracker = db.page_projection_tracker();
    let page_root = tempfile::tempdir().unwrap();
    let projection = crate::export::knowledge::KnowledgeProjectionWrite::new(
        page_root.path().to_path_buf(),
        &db,
    );
    let task = tokio::task::spawn_blocking(move || projection.write_page(&page("page-fast")));
    tokio::time::timeout(Duration::from_secs(2), task)
        .await
        .expect("real writer completes within two seconds")
        .unwrap()
        .unwrap();
    assert!(!tracker.sample().has_active_writes());
    assert_eq!(tracker.sample().generation(), 1);
}

#[tokio::test]
async fn stable_repeated_run_is_byte_identical_and_keeps_all_checks_conclusive() {
    let (db, _tmp) = test_db().await;
    let page_root = tempfile::tempdir().unwrap();
    let first = LintRunner::new(LintClock::fixed(), CancellationToken::new())
        .run(
            &db,
            &LintQuery {
                profile: None,
                space: None,
            },
            Some(page_root.path()),
            true,
        )
        .await
        .unwrap();
    let second = LintRunner::new(LintClock::fixed(), CancellationToken::new())
        .run(
            &db,
            &LintQuery {
                profile: None,
                space: None,
            },
            Some(page_root.path()),
            true,
        )
        .await
        .unwrap();
    assert!(first
        .checks()
        .iter()
        .all(|check| check.outcome() != LintOutcome::InconsistentSnapshot));
    assert_eq!(
        serde_json::to_vec(&first).unwrap(),
        serde_json::to_vec(&second).unwrap()
    );
}

#[tokio::test]
async fn full_page_group_does_not_mutate_database_or_projection_tree() {
    let (db, db_root) = test_db().await;
    let page_root = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(page_root.path().join(".wenlan")).unwrap();
    std::fs::write(
        page_root.path().join(".wenlan/state.json"),
        b"{\"schema_version\":2,\"pages\":{}}",
    )
    .unwrap();
    std::fs::create_dir_all(page_root.path().join("_sources")).unwrap();
    std::fs::write(
        page_root.path().join("_sources/.manifest.json"),
        b"{\"pages\":{}}",
    )
    .unwrap();
    let db_before = DbBytesFingerprint::capture(db_root.path()).unwrap();
    let semantic_before = semantic_fingerprint(&db).await;
    let page_before = PageBytesFingerprint::capture(page_root.path()).unwrap();
    let projection_before = db.page_projection_tracker().sample();
    assert!(!projection_before.has_active_writes());

    for _ in 0..2 {
        LintRunner::new(LintClock::fixed(), CancellationToken::new())
            .run(
                &db,
                &LintQuery {
                    profile: None,
                    space: None,
                },
                Some(page_root.path()),
                true,
            )
            .await
            .unwrap();
    }

    assert_eq!(
        db_before,
        DbBytesFingerprint::capture(db_root.path()).unwrap()
    );
    assert_eq!(
        page_before,
        PageBytesFingerprint::capture(page_root.path()).unwrap()
    );
    assert_eq!(semantic_before, semantic_fingerprint(&db).await);
    let projection_after = db.page_projection_tracker().sample();
    assert!(!projection_after.has_active_writes());
    assert_eq!(
        projection_after.generation(),
        projection_before.generation()
    );
}

#[tokio::test]
async fn knowledge_writer_rejects_a_guard_from_another_tracker() {
    let (db, _tmp) = test_db().await;
    let page_root = tempfile::tempdir().unwrap();
    let writer =
        crate::export::knowledge::KnowledgeWriter::new(page_root.path().to_path_buf(), &db);
    let other = crate::page_projection_tracker::PageProjectionTracker::new();
    let wrong_guard = other.begin_write();
    let error = writer
        .remove_page(&wrong_guard, "page-private")
        .unwrap_err();
    assert!(error.to_string().contains("guard"));
}

#[test]
fn page_check_store_classification_is_closed_over_the_catalog() {
    for id in [
        "pages.projection.identity",
        "pages.projection.version_alignment",
        "pages.project.artifact_inventory",
    ] {
        assert!(uses_cross_store(id));
        assert!(uses_filesystem(id));
    }
    for id in [
        "pages.projection.state_contract",
        "pages.projection.manifest_inventory",
    ] {
        assert!(!uses_cross_store(id));
        assert!(uses_filesystem(id));
    }
    for id in [
        "pages.db.partitions",
        "pages.citations.partitions",
        "pages.links.orphan_labels",
    ] {
        assert!(!uses_cross_store(id));
        assert!(!uses_filesystem(id));
    }
}
