use crate::db::tests::test_db;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Barrier;

#[tokio::test]
async fn lint_snapshot_baseline_primary_connection_answers_independently() {
    // Given: a separate read-only test session is open.
    let (db, _dir) = test_db().await;
    let transaction = db
        .test_secondary_session()
        .expect("secondary connection opens")
        .begin_read_only()
        .await
        .expect("read-only transaction starts");

    // When: the primary MemoryDB connection answers while that transaction remains open.
    let count = db.count().await.expect("primary connection answers");

    // Then: the primary connection remains independently usable.
    assert_eq!(count, 0);
    transaction
        .rollback()
        .await
        .expect("read-only transaction rolls back");
}

#[tokio::test]
async fn snapshot_keeps_multiple_reads_coherent_while_writer_commits() {
    // Given: an open lint snapshot and a writer held at a barrier.
    let (db, _dir) = test_db().await;
    let db = Arc::new(db);
    let snapshot = db.open_lint_snapshot().await.expect("lint snapshot opens");
    let writer_ready = Arc::new(Barrier::new(2));
    let writer_release = Arc::new(Barrier::new(2));
    let writer_db = Arc::clone(&db);
    let writer_ready_for_task = Arc::clone(&writer_ready);
    let writer_release_for_task = Arc::clone(&writer_release);
    let writer = tokio::spawn(async move {
        writer_ready_for_task.wait().await;
        writer_release_for_task.wait().await;
        writer_db
            .create_entity("lint snapshot writer", "test", None)
            .await
    });
    writer_ready.wait().await;

    // When: a writer commits between two reads from the open snapshot.
    let mut before_rows = snapshot
        .query(
            "SELECT COUNT(*) FROM entities",
            libsql::params::Params::None,
        )
        .await
        .expect("first lint read succeeds");
    let before = before_rows
        .next()
        .await
        .expect("first lint row succeeds")
        .expect("first lint row exists")
        .get::<i64>(0)
        .expect("first lint count is an integer");
    assert_eq!(
        db.count()
            .await
            .expect("primary connection remains available"),
        0
    );
    writer_release.wait().await;
    tokio::time::timeout(Duration::from_secs(2), writer)
        .await
        .expect("writer completes within two seconds")
        .expect("writer task joins")
        .expect("writer commits");
    let mut after_rows = snapshot
        .query(
            "SELECT COUNT(*) FROM entities",
            libsql::params::Params::None,
        )
        .await
        .expect("second lint read succeeds");
    let after = after_rows
        .next()
        .await
        .expect("second lint row succeeds")
        .expect("second lint row exists")
        .get::<i64>(0)
        .expect("second lint count is an integer");
    let receipt = snapshot.finish().await.expect("snapshot receipt succeeds");

    // Then: the snapshot stays coherent and its fresh receipt reports drift.
    assert_eq!(before, after);
    assert!(!receipt.is_consistent(), "post-run receipt detects drift");
}

#[tokio::test]
async fn snapshot_receipt_detects_same_row_count_update() {
    // Given: an entity, an open snapshot, and an UPDATE writer held at a barrier.
    let (db, _dir) = test_db().await;
    let db = Arc::new(db);
    let entity_id = db
        .create_entity("before update", "test", None)
        .await
        .expect("entity seed succeeds");
    let snapshot = db.open_lint_snapshot().await.expect("lint snapshot opens");
    let writer_ready = Arc::new(Barrier::new(2));
    let writer_release = Arc::new(Barrier::new(2));
    let writer_db = Arc::clone(&db);
    let writer_id = entity_id.clone();
    let writer_ready_for_task = Arc::clone(&writer_ready);
    let writer_release_for_task = Arc::clone(&writer_release);
    let writer = tokio::spawn(async move {
        writer_ready_for_task.wait().await;
        writer_release_for_task.wait().await;
        writer_db
            .conn
            .lock()
            .await
            .execute(
                "UPDATE entities SET name = ?1 WHERE id = ?2",
                libsql::params!["after update", writer_id],
            )
            .await
    });
    writer_ready.wait().await;

    // When: the UPDATE commits between two reads without changing a table count.
    let before = entity_name(&snapshot, &entity_id).await;
    writer_release.wait().await;
    tokio::time::timeout(Duration::from_secs(2), writer)
        .await
        .expect("update writer completes within two seconds")
        .expect("update writer task joins")
        .expect("update writer commits");
    let after = entity_name(&snapshot, &entity_id).await;
    let receipt = snapshot.finish().await.expect("snapshot receipt succeeds");

    // Then: snapshot reads stay coherent and the receipt still detects the UPDATE.
    assert_eq!(before, "before update");
    assert_eq!(after, before);
    assert!(
        !receipt.is_consistent(),
        "same-row-count UPDATE must invalidate the receipt"
    );
}

#[tokio::test]
async fn completed_receipt_is_invalidated_by_a_later_same_row_update() {
    // Given: a complete receipt from the daemon's shared freshness observer.
    let (db, _dir) = test_db().await;
    db.bootstrap_profile()
        .await
        .expect("profile fixture exists");
    let before = db
        .open_lint_snapshot()
        .await
        .expect("first snapshot opens")
        .finish()
        .await
        .expect("first snapshot finishes");

    // When: a canonical UPDATE commits without changing schema or row counts.
    let changed = db
        .conn
        .lock()
        .await
        .execute("UPDATE profiles SET updated_at = updated_at + 1", ())
        .await
        .expect("same-row update commits");
    assert_eq!(changed, 1);
    let after = db
        .open_lint_snapshot()
        .await
        .expect("second snapshot opens")
        .finish()
        .await
        .expect("second snapshot finishes");

    // Then: a prepare/verify freshness comparison rejects the older report.
    assert_ne!(
        before.analysis_receipt_digest(),
        after.analysis_receipt_digest()
    );
}

#[tokio::test]
async fn structural_snapshot_work_is_independent_of_table_payload_bytes() {
    // Given: an ordinary table whose schema is fixed before the receipt work starts.
    let (db, _dir) = test_db().await;
    let connection = db
        .test_secondary_session()
        .expect("receipt probe connection opens");
    connection
        .execute(
            "CREATE TABLE lint_payload_probe (payload BLOB NOT NULL)",
            (),
        )
        .await
        .expect("receipt probe table is created");
    let before = connection
        .structural_digest()
        .await
        .expect("pre-payload structural digest succeeds");

    // When: the table gains a payload large enough to expose any row-byte scan.
    connection
        .execute(
            "INSERT INTO lint_payload_probe(payload) VALUES (zeroblob(8388608))",
            (),
        )
        .await
        .expect("large payload is inserted");
    let after = connection
        .structural_digest()
        .await
        .expect("post-payload structural digest succeeds");

    // Then: structural receipt work is bounded by schema objects, not data bytes.
    // Committed data freshness is carried separately by the freshness sample.
    assert_eq!(before, after);
}

#[tokio::test]
async fn replacing_the_freshness_observer_invalidates_prior_receipts() {
    // Given: a receipt produced under one daemon-lifetime freshness observer.
    let (db, _dir) = test_db().await;
    db.bootstrap_profile()
        .await
        .expect("profile fixture exists");
    let old = db
        .open_isolated_lint_snapshot_for_test()
        .await
        .expect("old snapshot opens")
        .finish()
        .await
        .expect("old snapshot finishes");

    // When: a same-row canonical write commits and a new daemon lifetime opens
    // a fresh observer connection over the same database.
    let changed = db
        .conn
        .lock()
        .await
        .execute("UPDATE profiles SET updated_at = updated_at + 1", ())
        .await
        .expect("same-row update commits");
    assert_eq!(
        changed, 1,
        "the probe must update exactly one canonical row"
    );
    let new = db
        .open_isolated_lint_snapshot_for_test()
        .await
        .expect("new snapshot opens")
        .finish()
        .await
        .expect("new snapshot finishes");

    // Then: receipts cannot compare equal across observer/process lifetimes,
    // even though PRAGMA data_version counters are connection-relative.
    assert_ne!(old.analysis_receipt_digest(), new.analysis_receipt_digest());
}

#[tokio::test]
async fn post_run_receipt_uses_one_version_while_writer_commits() {
    // Given: an open snapshot and a writer waiting for the post-run snapshot to be pinned.
    let (db, _dir) = test_db().await;
    let db = Arc::new(db);
    let snapshot = db.open_lint_snapshot().await.expect("lint snapshot opens");
    let post_snapshot_pinned = Arc::new(Barrier::new(2));
    let writer_committed = Arc::new(Barrier::new(2));
    let writer_db = Arc::clone(&db);
    let writer_pinned = Arc::clone(&post_snapshot_pinned);
    let writer_done = Arc::clone(&writer_committed);
    let writer = tokio::spawn(async move {
        writer_pinned.wait().await;
        let result = writer_db
            .create_entity("post receipt writer", "test", None)
            .await;
        writer_done.wait().await;
        result
    });

    // When: the writer commits after the post-run transaction pins its read version.
    let receipt_pinned = Arc::clone(&post_snapshot_pinned);
    let receipt_writer_done = Arc::clone(&writer_committed);
    let receipt = snapshot
        .finish_with_post_snapshot_hook(|| async move {
            receipt_pinned.wait().await;
            tokio::time::timeout(Duration::from_secs(2), receipt_writer_done.wait())
                .await
                .expect("writer commits during post-receipt collection");
        })
        .await
        .expect("snapshot receipt succeeds");
    tokio::time::timeout(Duration::from_secs(2), writer)
        .await
        .expect("post-receipt writer completes within two seconds")
        .expect("post-receipt writer task joins")
        .expect("post-receipt writer commits");

    // Then: the post digest is one coherent version while the observer reports drift.
    assert_eq!(receipt.analysis_digest(), receipt.post_run_digest());
    assert!(
        !receipt.is_consistent(),
        "observer detects the concurrent commit"
    );
}

#[tokio::test]
async fn deferred_step_error_maps_to_snapshot_error() {
    // Given: an open snapshot and SQL whose runtime error occurs while stepping rows.
    let (db, _dir) = test_db().await;
    let snapshot = db.open_lint_snapshot().await.expect("lint snapshot opens");

    // When: query preparation succeeds and the first row is stepped.
    let mut rows: super::LintRows<'_> = snapshot
        .query(
            "SELECT json_extract('not-json', '$')",
            libsql::params::Params::None,
        )
        .await
        .expect("deferred-error query prepares");
    let step = rows.next().await;

    // Then: the deferred libSQL error remains a typed snapshot failure.
    assert!(matches!(step, Err(super::SnapshotError::Database(_))));
}

#[tokio::test]
async fn snapshot_read_only_transaction_rejects_mutation() {
    // Given: an open lint read-only snapshot.
    let (db, _dir) = test_db().await;
    let snapshot = db.open_lint_snapshot().await.expect("lint snapshot opens");

    // When: a lint query attempts a write.
    let write_result = snapshot
        .query(
            "INSERT INTO entities (id, name, entity_type, confirmed, created_at, updated_at) VALUES ('lint-write', 'lint-write', 'test', 0, 1, 1) RETURNING id",
            libsql::params::Params::None,
        )
        .await;
    let write_result = match write_result {
        Ok(mut rows) => rows.next().await.map(|_| ()),
        Err(error) => Err(error),
    };

    // Then: the mutation fails and the primary database remains unchanged.
    assert!(
        matches!(write_result, Err(super::SnapshotError::Database(_))),
        "read-only transaction rejects writes with a typed database failure"
    );
    snapshot.finish().await.expect("snapshot cleanup succeeds");
    assert_eq!(db.count().await.expect("primary connection answers"), 0);
}

#[tokio::test]
async fn snapshot_query_failure_cleans_up_without_passing() {
    // Given: an open lint snapshot.
    let (db, _dir) = test_db().await;
    let snapshot = db.open_lint_snapshot().await.expect("lint snapshot opens");

    // When: a malformed lint query fails and the snapshot is dropped.
    let query_result = snapshot
        .query(
            "SELECT missing_column FROM entities",
            libsql::params::Params::None,
        )
        .await;
    assert!(
        matches!(query_result, Err(super::SnapshotError::Database(_))),
        "query failure is a typed database failure, never a pass"
    );
    drop(snapshot);

    // Then: cleanup releases the read transaction for the next snapshot.
    let next_snapshot = db
        .open_lint_snapshot()
        .await
        .expect("next lint snapshot opens after failure");
    next_snapshot
        .finish()
        .await
        .expect("next snapshot finishes");
}

async fn entity_name(snapshot: &super::LintReadSnapshot<'_>, entity_id: &str) -> String {
    let mut rows = snapshot
        .query(
            "SELECT name FROM entities WHERE id = ?1",
            libsql::params::Params::Positional(vec![libsql::Value::Text(entity_id.to_owned())]),
        )
        .await
        .expect("entity-name query succeeds");
    rows.next()
        .await
        .expect("entity-name row succeeds")
        .expect("entity-name row exists")
        .get::<String>(0)
        .expect("entity name is text")
}
