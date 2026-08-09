use super::{check, fingerprint, run, QUEUE};
use crate::db::tests::test_db;

#[tokio::test]
async fn full_population_is_checked_through_read_only_capability_without_db_mutation() {
    // Given
    let (db, _tmp) = test_db().await;
    let conn = db.test_primary_session().await;
    for index in 0..101 {
        conn.execute(
            "INSERT INTO document_enrichment_queue
             (source_id,file_path,status,last_completed_chunk,attempt_count,enqueued_at,updated_at)
             VALUES (?1,?1,'impossible',-1,0,1,1)",
            libsql::params![format!("row-{index:03}")],
        )
        .await
        .unwrap();
    }
    drop(conn);
    let before = fingerprint(&db).await;

    // When
    let report = run(&db, &[]).await;

    // Then
    let result = check(&report, QUEUE);
    assert_eq!(result.coverage().denominator(), 101);
    assert_eq!(result.coverage().evaluated(), 101);
    assert!(result.coverage().truncated());
    assert_eq!(result.evidence().len(), 100);
    assert_eq!(before, fingerprint(&db).await);
    let source = concat!(
        include_str!("../operations.rs"),
        include_str!("../operations/config.rs"),
        include_str!("../operations/query.rs"),
        include_str!("../operations/query/imports.rs"),
        include_str!("../operations/query/maintenance.rs"),
        include_str!("../operations/query/queue.rs"),
        include_str!("../operations/query/refinement.rs"),
        include_str!("../operations/query/reviews.rs"),
        include_str!("../operations/query/source.rs"),
        include_str!("../operations/read_context.rs"),
        include_str!("../operations/result.rs"),
    )
    .to_ascii_lowercase();
    for forbidden in [
        "memorydb",
        "std::fs",
        "reqwest",
        "enqueue_document",
        "mark_paused",
        "mark_done",
        "insert_refinement",
        "resolve_refinement",
        "emit(",
        "llmprovider",
        "gpu",
        "update ",
        "insert ",
        "delete ",
    ] {
        assert!(
            !source.contains(forbidden),
            "mutating seam present: {forbidden}"
        );
    }
}
