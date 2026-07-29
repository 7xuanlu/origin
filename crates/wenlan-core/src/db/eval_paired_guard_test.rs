// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;

async fn drop_summary_nodes(db: &MemoryDB) {
    let conn = db.conn.lock().await;
    conn.execute_batch(
        "DROP TABLE IF EXISTS summary_node_sources;
         DROP TABLE IF EXISTS summary_nodes;",
    )
    .await
    .unwrap();
}

async fn create_summary_nodes(db: &MemoryDB, with_row: bool) {
    let conn = db.conn.lock().await;
    conn.execute("CREATE TABLE summary_nodes (id TEXT)", ())
        .await
        .unwrap();
    if with_row {
        conn.execute("INSERT INTO summary_nodes (id) VALUES ('one')", ())
            .await
            .unwrap();
    }
}

#[tokio::test]
async fn summary_node_count_errors_when_missing_then_reads_zero_and_one() {
    let (db, _tmp) = crate::db::tests::test_db().await;
    drop_summary_nodes(&db).await;

    assert!(db.eval_paired_summary_node_count().await.is_err());
    crate::eval::paired::assert_summary_nodes_empty(&db).await;

    create_summary_nodes(&db, false).await;
    assert_eq!(db.eval_paired_summary_node_count().await.unwrap(), 0);
    {
        let conn = db.conn.lock().await;
        conn.execute("INSERT INTO summary_nodes (id) VALUES ('one')", ())
            .await
            .unwrap();
    }
    assert_eq!(db.eval_paired_summary_node_count().await.unwrap(), 1);
}

#[tokio::test]
#[should_panic(expected = "summary_nodes is non-empty (1 rows): the global-prelude prepend")]
async fn paired_caller_panics_with_existing_diagnostic_prefix_for_one_row() {
    let (db, _tmp) = crate::db::tests::test_db().await;
    drop_summary_nodes(&db).await;
    create_summary_nodes(&db, true).await;

    crate::eval::paired::assert_summary_nodes_empty(&db).await;
}
