// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;

async fn insert_queue_row(db: &MemoryDB, id: &str, action: &str, status: &str) {
    let conn = db.conn.lock().await;
    conn.execute(
        "INSERT INTO refinement_queue (id, action, source_ids, status) VALUES (?1, ?2, '[]', ?3)",
        libsql::params![id, action, status],
    )
    .await
    .unwrap();
}

async fn clear_queue(db: &MemoryDB) {
    let conn = db.conn.lock().await;
    conn.execute("DELETE FROM refinement_queue", ())
        .await
        .unwrap();
}

#[tokio::test]
async fn retro_probe_matches_only_open_retro_actions() {
    let (db, _tmp) = crate::db::tests::test_db().await;

    insert_queue_row(&db, "dismissed-merge", "page_merge", "dismissed").await;
    insert_queue_row(&db, "dismissed-keep", "page_keep_or_archive", "dismissed").await;
    insert_queue_row(&db, "other-open-action", "cross_space_discovery", "pending").await;
    assert!(
        !db.has_pending_retro_review().await.unwrap(),
        "dismissed retro rows and open non-retro rows are controls against blanket true"
    );

    for action in ["page_merge", "page_keep_or_archive"] {
        for status in ["pending", "awaiting_review"] {
            clear_queue(&db).await;
            insert_queue_row(&db, "control-dismissed", action, "dismissed").await;
            insert_queue_row(&db, "control-resolved", action, "resolved").await;
            insert_queue_row(&db, "control-auto-applied", action, "auto_applied").await;
            insert_queue_row(&db, "control-other", "cross_space_discovery", status).await;
            assert!(!db.has_pending_retro_review().await.unwrap());

            insert_queue_row(&db, "matching-open", action, status).await;
            assert!(
                db.has_pending_retro_review().await.unwrap(),
                "{action}/{status} must block retro maintenance"
            );
        }
    }
}

#[tokio::test]
async fn cross_space_probe_matches_only_open_cross_space_action() {
    let (db, _tmp) = crate::db::tests::test_db().await;

    insert_queue_row(
        &db,
        "dismissed-discovery",
        "cross_space_discovery",
        "dismissed",
    )
    .await;
    insert_queue_row(&db, "other-open-action", "page_merge", "pending").await;
    assert!(
        !db.has_pending_cross_space_discovery().await.unwrap(),
        "a dismissed discovery row and open other action are controls against blanket true"
    );

    for status in ["pending", "awaiting_review"] {
        clear_queue(&db).await;
        insert_queue_row(
            &db,
            "control-dismissed",
            "cross_space_discovery",
            "dismissed",
        )
        .await;
        insert_queue_row(&db, "control-resolved", "cross_space_discovery", "resolved").await;
        insert_queue_row(
            &db,
            "control-auto-applied",
            "cross_space_discovery",
            "auto_applied",
        )
        .await;
        insert_queue_row(&db, "control-other", "page_merge", status).await;
        assert!(!db.has_pending_cross_space_discovery().await.unwrap());

        insert_queue_row(&db, "matching-open", "cross_space_discovery", status).await;
        assert!(
            db.has_pending_cross_space_discovery().await.unwrap(),
            "cross_space_discovery/{status} must block discovery"
        );
    }
}
