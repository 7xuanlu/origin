// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;
use std::sync::Arc;

async fn insert_page(
    db: &MemoryDB,
    id: &str,
    title: &str,
    legacy_sources: &[&str],
    embedding: Option<&str>,
) {
    let conn = db.conn.lock().await;
    conn.execute(
        "INSERT INTO pages (
             id, title, content, source_memory_ids, status, created_at, last_compiled,
             last_modified, review_status, workspace, space, embedding
         ) VALUES (?1, ?2, 'body', ?3, 'active', 'now', 'now', 'now',
                   'confirmed', 'work', 'work',
                   CASE WHEN ?4 IS NULL THEN NULL ELSE vector32(?4) END)",
        libsql::params![
            id,
            title,
            serde_json::to_string(legacy_sources).unwrap(),
            embedding,
        ],
    )
    .await
    .unwrap();
}

fn unit_vector(axis: usize) -> String {
    let mut values = vec![0.0; 768];
    values[axis] = 1.0;
    serde_json::to_string(&values).unwrap()
}

#[tokio::test]
async fn near_duplicate_reader_preserves_pair_order_cursor_and_raw_sources() {
    let (db, _tmp) = crate::db::tests::test_db().await;
    for id in ["a", "b", "c"] {
        let legacy_source = format!("legacy-{id}");
        insert_page(&db, id, id, &[legacy_source.as_str()], None).await;
    }
    let conn = db.conn.lock().await;
    conn.execute(
        "INSERT INTO page_sources (page_id, memory_source_id, linked_at, link_reason)
         VALUES ('b', 'normalized-b-3', 3, 'test'),
                ('b', 'normalized-b-1', 1, 'test'),
                ('b', 'normalized-b-2', 2, 'test')",
        (),
    )
    .await
    .unwrap();
    conn.execute("UPDATE pages SET status = 'archived' WHERE id = 'c'", ())
        .await
        .unwrap();
    drop(conn);

    let reader = db.begin_near_duplicate_slice_reader().await;
    let first = reader.scan_near_duplicate_slice(None, 2).await.unwrap();
    assert_eq!(
        first
            .iter()
            .map(|row| (row.left_id.as_str(), row.right_id.as_str()))
            .collect::<Vec<_>>(),
        vec![("a", "b"), ("a", "c")]
    );
    assert_eq!(first[0].left_fallback_sources, vec!["legacy-a"]);
    assert_eq!(first[0].right_fallback_sources, vec!["legacy-b"]);
    assert!(first[0].eligible);
    assert!(!first[1].eligible);
    assert_eq!(
        reader.load_bounded_page_source_ids("b", 2).await.unwrap(),
        vec!["normalized-b-1".to_string(), "normalized-b-2".to_string()]
    );
    assert!(reader
        .load_bounded_page_source_ids("a", 257)
        .await
        .unwrap()
        .is_empty());

    let resumed = reader
        .scan_near_duplicate_slice(Some(("a", "b")), 1)
        .await
        .unwrap();
    assert_eq!(resumed[0].right_id, "c");
    assert!(!resumed[0].eligible);
}

#[tokio::test]
async fn near_duplicate_reader_keeps_connection_locked_until_reader_drop() {
    let (db, _tmp) = crate::db::tests::test_db().await;
    let db = Arc::new(db);
    let reader = db.begin_near_duplicate_slice_reader().await;
    reader.scan_near_duplicate_slice(None, 1).await.unwrap();

    let started = Arc::new(tokio::sync::Notify::new());
    let mut probe = tokio::spawn({
        let db = Arc::clone(&db);
        let started = Arc::clone(&started);
        async move {
            started.notify_one();
            db.get_memory_count().await
        }
    });
    started.notified().await;
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(30), &mut probe)
            .await
            .is_err(),
        "a scoped reader must retain the DB mutex through caller policy"
    );
    drop(reader);
    tokio::time::timeout(std::time::Duration::from_secs(2), probe)
        .await
        .expect("probe resumes after reader drop")
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn embedding_pair_reader_preserves_distance_order_and_sql_limit() {
    let (db, _tmp) = crate::db::tests::test_db().await;
    let first = unit_vector(1);
    let other = unit_vector(2);
    insert_page(&db, "a", "A", &[], Some(&first)).await;
    insert_page(&db, "b", "B", &[], Some(&first)).await;
    insert_page(&db, "c", "C", &[], Some(&other)).await;
    insert_page(&db, "overview", "OvErViEw", &[], Some(&first)).await;

    let limited = db
        .embedding_near_duplicate_pairs(1.0, Some(1))
        .await
        .unwrap();
    assert_eq!(limited.len(), 1);
    assert_eq!(
        (limited[0].left_id.as_str(), limited[0].right_id.as_str()),
        ("a", "b")
    );
    assert!(limited[0].distance.abs() < f64::EPSILON);

    let all = db.embedding_near_duplicate_pairs(1.0, None).await.unwrap();
    assert_eq!(all.len(), 3);
    assert!(all
        .windows(2)
        .all(|rows| rows[0].distance <= rows[1].distance));
    assert!(all
        .iter()
        .all(|row| row.left_id != "overview" && row.right_id != "overview"));
}
