// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;

async fn seed_temporal(db: &MemoryDB) {
    let conn = db.conn.lock().await;
    conn.execute(
        "INSERT INTO memories (
             id, content, source, source_id, title, chunk_index, last_modified,
             chunk_type, event_date
         ) VALUES ('memory-temporal', 'body', 'memory', 'source-temporal',
                   'Temporal', 0, 1, 'text', 1)",
        (),
    )
    .await
    .unwrap();
}

async fn seed_graph(db: &MemoryDB) {
    let conn = db.conn.lock().await;
    conn.execute(
        "INSERT INTO entities (id, name, entity_type, created_at, updated_at)
         VALUES ('entity-graph', 'Graph', 'concept', 1, 1)",
        (),
    )
    .await
    .unwrap();
    conn.execute(
        "INSERT INTO memory_entities (memory_id, entity_id)
         VALUES ('source-temporal', 'entity-graph')",
        (),
    )
    .await
    .unwrap();
}

async fn seed_page(db: &MemoryDB) {
    let conn = db.conn.lock().await;
    conn.execute(
        "INSERT INTO pages (
             id, title, content, status, created_at, last_compiled, last_modified
         ) VALUES ('page-live', 'Live', 'body', 'active', 'now', 'now', 'now')",
        (),
    )
    .await
    .unwrap();
}

fn assert_refused(error: crate::error::WenlanError, substrate: &str) {
    let message = error.to_string();
    assert!(
        message.contains(&format!("{substrate} substrate empty")),
        "expected {substrate} refusal, got: {message}"
    );
}

#[tokio::test]
async fn feature_guard_delegates_graph_temporal_page_aliases_and_unmodeled_passes() {
    let (db, _tmp) = crate::db::tests::test_db().await;

    db.assert_eval_feature_substrate_live("reranker")
        .await
        .unwrap();

    assert_refused(
        db.assert_eval_feature_substrate_live("graph_ab")
            .await
            .unwrap_err(),
        "graph",
    );
    seed_temporal(&db).await;
    seed_graph(&db).await;
    db.assert_eval_feature_substrate_live("graph_ab")
        .await
        .unwrap();

    let (temporal_db, _tmp) = crate::db::tests::test_db().await;
    assert_refused(
        temporal_db
            .assert_eval_feature_substrate_live("temporal_ab")
            .await
            .unwrap_err(),
        "temporal",
    );
    seed_temporal(&temporal_db).await;
    temporal_db
        .assert_eval_feature_substrate_live("temporal_ab")
        .await
        .unwrap();

    let (page_db, _tmp) = crate::db::tests::test_db().await;
    assert_refused(
        page_db
            .assert_eval_feature_substrate_live("page_channel")
            .await
            .unwrap_err(),
        "page",
    );
    seed_page(&page_db).await;
    page_db
        .assert_eval_feature_substrate_live("page_channel")
        .await
        .unwrap();
}

#[tokio::test]
async fn migration_guard_refuses_in_temporal_graph_page_order_then_passes() {
    let (db, _tmp) = crate::db::tests::test_db().await;

    assert_refused(
        db.assert_eval_migration_substrates_live()
            .await
            .unwrap_err(),
        "temporal",
    );

    seed_temporal(&db).await;
    assert_refused(
        db.assert_eval_migration_substrates_live()
            .await
            .unwrap_err(),
        "graph",
    );

    seed_graph(&db).await;
    assert_refused(
        db.assert_eval_migration_substrates_live()
            .await
            .unwrap_err(),
        "page",
    );

    seed_page(&db).await;
    db.assert_eval_migration_substrates_live().await.unwrap();
}
