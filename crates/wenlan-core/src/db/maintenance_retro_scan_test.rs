// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;

async fn insert_page(
    db: &MemoryDB,
    id: &str,
    title: &str,
    legacy_sources: &[&str],
    status: &str,
    creation_kind: &str,
    user_edited: bool,
) {
    let source_memory_ids = serde_json::to_string(legacy_sources).unwrap();
    let conn = db.conn.lock().await;
    conn.execute(
        "INSERT INTO pages (
             id, title, content, source_memory_ids, status, created_at,
             last_compiled, last_modified, creation_kind, user_edited
         ) VALUES (?1, ?2, 'body', ?3, ?4, 'now', 'now', 'now', ?5, ?6)",
        libsql::params![
            id,
            title,
            source_memory_ids,
            status,
            creation_kind,
            user_edited as i64,
        ],
    )
    .await
    .unwrap();
}

async fn insert_normalized_sources(db: &MemoryDB, page_id: &str, source_ids: &[&str]) {
    let conn = db.conn.lock().await;
    for (linked_at, source_id) in source_ids.iter().enumerate() {
        conn.execute(
            "INSERT INTO page_sources (page_id, memory_source_id, linked_at, link_reason) \
             VALUES (?1, ?2, ?3, 'test')",
            libsql::params![page_id, *source_id, linked_at as i64],
        )
        .await
        .unwrap();
        // G6 Stage 1.3: the retro-scan reader migrated onto `edges` --
        // mirror the dual-write here so this raw-SQL fixture still drives
        // the reader it's meant to test (`lineage='legacy'` exempts the
        // cross-space fence trigger).
        conn.execute(
            "INSERT INTO edges (edge_id, src_id, src_kind, dst_id, dst_kind, edge_type, \
                                 lineage, grounded, space, created_at) \
             VALUES (?1, ?2, 'page', ?3, 'memory', 'cites', 'legacy', 0, 'default', ?4)",
            libsql::params![
                format!("retro-scan-{page_id}-{source_id}"),
                page_id,
                *source_id,
                linked_at as i64,
            ],
        )
        .await
        .unwrap();
    }
}

#[tokio::test]
async fn automatic_retro_scan_reports_empty_and_skips_ineligible_pages() {
    let (db, _tmp) = crate::db::tests::test_db().await;
    let empty = db.scan_automatic_retro_stub_slice(None, 3).await.unwrap();
    assert!(empty.next_cursor.is_none());
    assert!(empty.eligible_source_count.is_none());
    assert!(!empty.more);

    for (id, title, status, creation_kind, user_edited) in [
        ("a_archived", "Archived", "archived", "distilled", false),
        ("b_authored", "Authored", "active", "authored", false),
        ("c_edited", "Edited", "active", "distilled", true),
        ("d_overview", "OvErViEw", "active", "distilled", false),
    ] {
        insert_page(
            &db,
            id,
            title,
            &["legacy-control"],
            status,
            creation_kind,
            user_edited,
        )
        .await;
    }

    let mut cursor = None;
    for expected_id in ["a_archived", "b_authored", "c_edited", "d_overview"] {
        let scan = db
            .scan_automatic_retro_stub_slice(cursor.as_deref(), 3)
            .await
            .unwrap();
        assert_eq!(scan.next_cursor.as_deref(), Some(expected_id));
        assert!(scan.eligible_source_count.is_none());
        cursor = scan.next_cursor;
    }
}

#[tokio::test]
async fn automatic_retro_scan_bounds_normalized_source_rows_at_three() {
    for (suffix, normalized_sources, expected_source_count) in [
        ("zero", Vec::<&str>::new(), Some(0)),
        ("one", vec!["source-1"], Some(1)),
        (
            "many",
            vec!["source-4", "source-2", "source-1", "source-3"],
            Some(3),
        ),
    ] {
        let (db, _tmp) = crate::db::tests::test_db().await;
        let page_id = format!("normalized-{suffix}");
        insert_page(&db, &page_id, "Eligible", &[], "active", "distilled", false).await;
        insert_normalized_sources(&db, &page_id, &normalized_sources).await;

        let scan = db.scan_automatic_retro_stub_slice(None, 3).await.unwrap();
        assert_eq!(scan.next_cursor.as_deref(), Some(page_id.as_str()));
        assert_eq!(scan.eligible_source_count, expected_source_count);
    }
}

#[tokio::test]
async fn automatic_retro_scan_uses_legacy_sources_only_when_normalized_rows_are_empty() {
    for (legacy_sources, expected_source_count) in [
        (vec!["legacy-1"], Some(1)),
        (
            vec!["legacy-4", "legacy-2", "legacy-1", "legacy-3"],
            Some(3),
        ),
    ] {
        let (db, _tmp) = crate::db::tests::test_db().await;
        insert_page(
            &db,
            "legacy-page",
            "Eligible",
            &legacy_sources,
            "active",
            "distilled",
            false,
        )
        .await;

        let scan = db.scan_automatic_retro_stub_slice(None, 3).await.unwrap();
        assert_eq!(scan.eligible_source_count, expected_source_count);
    }

    let (db, _tmp) = crate::db::tests::test_db().await;
    insert_page(
        &db,
        "normalized-wins",
        "Eligible",
        &["legacy-1", "legacy-2", "legacy-3"],
        "active",
        "distilled",
        false,
    )
    .await;
    insert_normalized_sources(&db, "normalized-wins", &["normalized-1"]).await;
    let scan = db.scan_automatic_retro_stub_slice(None, 3).await.unwrap();
    assert_eq!(
        scan.eligible_source_count,
        Some(1),
        "a non-empty normalized source set suppresses the legacy list"
    );
}

#[tokio::test]
async fn automatic_retro_scan_handles_malformed_legacy_json_and_null_user_edited() {
    let (malformed_db, _tmp) = crate::db::tests::test_db().await;
    insert_page(
        &malformed_db,
        "malformed-json",
        "Eligible",
        &[],
        "active",
        "distilled",
        false,
    )
    .await;
    let conn = malformed_db.conn.lock().await;
    conn.execute(
        "UPDATE pages SET source_memory_ids = '{not-json' WHERE id = 'malformed-json'",
        (),
    )
    .await
    .unwrap();
    drop(conn);
    let malformed = malformed_db
        .scan_automatic_retro_stub_slice(None, 3)
        .await
        .unwrap();
    assert_eq!(malformed.eligible_source_count, Some(0));

    let (db, _tmp) = crate::db::tests::test_db().await;
    insert_page(
        &db,
        "null-user-edited",
        "Eligible",
        &[],
        "active",
        "distilled",
        false,
    )
    .await;
    let conn = db.conn.lock().await;
    conn.execute(
        "UPDATE pages SET user_edited = NULL WHERE id = 'null-user-edited'",
        (),
    )
    .await
    .unwrap();
    drop(conn);

    let scan = db.scan_automatic_retro_stub_slice(None, 3).await.unwrap();
    assert_eq!(
        scan.eligible_source_count,
        Some(0),
        "COALESCE(user_edited, 0) keeps legacy NULL rows eligible"
    );
}

#[tokio::test]
async fn automatic_retro_scan_uses_second_row_only_for_more_and_advances_cursor() {
    let (db, _tmp) = crate::db::tests::test_db().await;
    for id in ["page-a", "page-b"] {
        insert_page(&db, id, "Eligible", &[], "active", "distilled", false).await;
    }

    let first = db.scan_automatic_retro_stub_slice(None, 3).await.unwrap();
    assert_eq!(first.next_cursor.as_deref(), Some("page-a"));
    assert!(first.more);

    let second = db
        .scan_automatic_retro_stub_slice(first.next_cursor.as_deref(), 3)
        .await
        .unwrap();
    assert_eq!(second.next_cursor.as_deref(), Some("page-b"));
    assert!(!second.more);
}
