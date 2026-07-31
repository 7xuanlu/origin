// SPDX-License-Identifier: Apache-2.0

use super::{EvalTemporalSeed, MemoryDB};

async fn insert_memory(db: &MemoryDB, id: &str, source_id: &str, event_date: Option<i64>) {
    let conn = db.conn.lock().await;
    conn.execute(
        "INSERT INTO memories (
             id, content, source, source_id, title, chunk_index, last_modified,
             chunk_type, event_date
         ) VALUES (?1, 'body', 'memory', ?2, 'title', 0, 1, 'text', ?3)",
        libsql::params![id, source_id, event_date],
    )
    .await
    .unwrap();
}

async fn event_dates(db: &MemoryDB, source_id: &str) -> Vec<Option<i64>> {
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT event_date FROM memories WHERE source_id = ?1 ORDER BY id",
            libsql::params![source_id],
        )
        .await
        .unwrap();
    let mut dates = Vec::new();
    while let Some(row) = rows.next().await.unwrap() {
        dates.push(row.get::<Option<i64>>(0).unwrap());
    }
    dates
}

async fn temporal_seed_audit(db: &MemoryDB) -> Vec<(String, String, i64)> {
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT memory_id, source_id, event_date
             FROM temporal_seed_audit
             ORDER BY sequence",
            (),
        )
        .await
        .unwrap();
    let mut audit = Vec::new();
    while let Some(row) = rows.next().await.unwrap() {
        audit.push((
            row.get::<String>(0).unwrap(),
            row.get::<String>(1).unwrap(),
            row.get::<i64>(2).unwrap(),
        ));
    }
    audit
}

fn seed(source_id: &str, event_date: i64) -> EvalTemporalSeed {
    EvalTemporalSeed {
        source_id: source_id.to_string(),
        event_date,
    }
}

#[tokio::test]
async fn temporal_seed_updates_continue_after_failure_without_rollback_or_deduplication() {
    let (db, _tmp) = crate::db::tests::test_db().await;
    insert_memory(&db, "before", "before", None).await;
    insert_memory(&db, "shared-a", "shared", None).await;
    insert_memory(&db, "shared-b", "shared", None).await;
    insert_memory(&db, "failure", "failure", Some(9)).await;
    insert_memory(&db, "after", "after", None).await;
    insert_memory(&db, "unrelated", "unrelated", Some(77)).await;
    {
        let conn = db.conn.lock().await;
        conn.execute_batch(
            "CREATE TABLE temporal_seed_audit (
                 sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                 memory_id TEXT NOT NULL,
                 source_id TEXT NOT NULL,
                 event_date INTEGER NOT NULL
             );
             CREATE TRIGGER fail_selected_temporal_seed
             BEFORE UPDATE OF event_date ON memories
             WHEN OLD.source_id = 'failure'
             BEGIN
                 SELECT RAISE(ABORT, 'selected temporal seed failure');
             END;
             CREATE TRIGGER audit_successful_temporal_seed
             AFTER UPDATE OF event_date ON memories
             WHEN NEW.id = (
                 SELECT MIN(candidate.id)
                 FROM memories candidate
                 WHERE candidate.source_id = NEW.source_id
             )
             BEGIN
                 INSERT INTO temporal_seed_audit (memory_id, source_id, event_date)
                 VALUES (NEW.id, NEW.source_id, NEW.event_date);
             END;",
        )
        .await
        .unwrap();
    }

    db.apply_eval_temporal_seeds(&[
        seed("before", 700),
        seed("shared", 300),
        seed("failure", 900),
        seed("unmatched", 100),
        seed("shared", 800),
        seed("after", 200),
    ])
    .await;

    assert_eq!(
        temporal_seed_audit(&db).await,
        vec![
            ("before".to_string(), "before".to_string(), 700),
            ("shared-a".to_string(), "shared".to_string(), 300),
            ("shared-a".to_string(), "shared".to_string(), 800),
            ("after".to_string(), "after".to_string(), 200),
        ],
        "audit order must match successful supplied statements exactly; failed and unmatched seeds emit nothing"
    );
    assert_eq!(event_dates(&db, "before").await, vec![Some(700)]);
    assert_eq!(
        event_dates(&db, "shared").await,
        vec![Some(800), Some(800)],
        "both physical rows update and the last duplicate seed wins"
    );
    assert_eq!(event_dates(&db, "failure").await, vec![Some(9)]);
    assert_eq!(event_dates(&db, "after").await, vec![Some(200)]);
    assert_eq!(event_dates(&db, "unrelated").await, vec![Some(77)]);
    assert!(event_dates(&db, "unmatched").await.is_empty());
}

#[tokio::test]
async fn empty_temporal_seed_slice_completes_without_mutation() {
    let (db, _tmp) = crate::db::tests::test_db().await;
    insert_memory(&db, "untouched", "untouched", Some(71)).await;

    db.apply_eval_temporal_seeds(&[]).await;

    assert_eq!(event_dates(&db, "untouched").await, vec![Some(71)]);
}

#[tokio::test]
async fn temporal_seed_swallows_missing_memories_table_error() {
    let (db, _tmp) = crate::db::tests::test_db().await;
    {
        let conn = db.conn.lock().await;
        conn.execute_batch("PRAGMA foreign_keys = OFF; DROP TABLE memories;")
            .await
            .unwrap();
    }

    db.apply_eval_temporal_seeds(&[seed("missing", 81)]).await;
}
