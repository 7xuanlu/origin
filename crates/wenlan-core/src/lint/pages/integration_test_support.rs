use crate::db::tests::test_db;
use crate::lint::context::{CancellationToken, LintClock};
use crate::lint::runner::{LintRunner, TestSyncPoint, TestSynchronization};
use crate::lint::snapshot::LintReadSnapshot;
use crate::lint::test_support::DbSemanticFingerprint;
use crate::pages::Page;
use std::sync::Arc;
use wenlan_types::lint::{LintOutcome, LintQuery};

#[derive(Clone, Copy)]
pub(super) enum ReceiptDrift {
    Page,
    Database,
}

pub(super) async fn run_receipt_drift(kind: ReceiptDrift) -> wenlan_types::lint::LintReport {
    let (db, _tmp) = test_db().await;
    let db = Arc::new(db);
    let page_root = tempfile::tempdir().unwrap();
    let root = page_root.path().to_path_buf();
    let (synchronization, control) = TestSynchronization::new(TestSyncPoint::BeforeReceipts);
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
    match kind {
        ReceiptDrift::Page => std::fs::write(root.join("manual.md"), "# changed").unwrap(),
        ReceiptDrift::Database => insert_page(&db, "page-db-drift").await,
    }
    control.resume().await;
    task.await.unwrap().unwrap()
}

pub(super) fn assert_selective_inconsistency(
    report: &wenlan_types::lint::LintReport,
    affected: impl Fn(&str) -> bool,
) {
    assert!(!report.complete());
    for check in report.checks() {
        assert_eq!(
            check.outcome() == LintOutcome::InconsistentSnapshot,
            affected(check.check_id()),
            "unexpected receipt classification for {}",
            check.check_id()
        );
    }
}

pub(super) async fn insert_page(db: &crate::db::MemoryDB, id: &str) {
    db.conn
        .lock()
        .await
        .execute(
            "INSERT INTO pages (id, title, content, source_memory_ids, version, status, created_at, last_compiled, last_modified, creation_kind, review_status) VALUES (?1, ?1, 'body', '[]', 1, 'active', 'now', 'now', 'now', 'distilled', 'confirmed')",
            libsql::params![id],
        )
        .await
        .unwrap();
}

pub(super) async fn semantic_fingerprint(db: &crate::db::MemoryDB) -> DbSemanticFingerprint {
    let snapshot = LintReadSnapshot::open(&db._db).await.unwrap();
    let fingerprint = DbSemanticFingerprint::capture(&snapshot).await.unwrap();
    snapshot.finish().await.unwrap();
    fingerprint
}

pub(super) fn page(id: &str) -> Page {
    Page {
        id: id.to_string(),
        title: id.to_string(),
        summary: None,
        content: "body".to_string(),
        entity_id: None,
        space: None,
        source_memory_ids: Vec::new(),
        version: 1,
        status: "active".to_string(),
        created_at: "2026-07-10T00:00:00Z".to_string(),
        last_compiled: "2026-07-10T00:00:00Z".to_string(),
        last_modified: "2026-07-10T00:00:00Z".to_string(),
        sources_updated_count: 0,
        stale_reason: None,
        pending_rebuild: None,
        user_edited: false,
        relevance_score: 0.0,
        last_edited_by: None,
        last_edited_at: None,
        last_delta_summary: None,
        changelog: None,
        creation_kind: "distilled".to_string(),
        review_status: "confirmed".to_string(),
        workspace: None,
        citations: Vec::new(),
        kind: "concept".to_string(),
    }
}
