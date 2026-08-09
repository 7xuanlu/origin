// SPDX-License-Identifier: Apache-2.0
//! Regression tests for the shared-connection transaction leak (issue #389,
//! Finding 2).
//!
//! `MemoryDB` owns ONE `tokio::sync::Mutex<libsql::Connection>`. A function
//! shaped `BEGIN -> fallible ops with ? -> COMMIT` that returns early without a
//! `ROLLBACK` leaves that single connection inside an open transaction. Callers
//! of these functions swallow the error, so the daemon keeps serving while
//! every later write is uncommitted: its own reads see them, nothing reaches
//! the WAL, and process death discards the lot.
//!
//! Each test forces a failure in the middle of one such transaction and asserts
//! the connection is back in autocommit afterwards. The forcing lever is a
//! `RAISE(ABORT)` trigger on the table the transaction body writes: SQLite's
//! ABORT rolls back the failing statement only and leaves the surrounding
//! transaction open, which is exactly the leak this fixes.

use super::MemoryDB;
use crate::events::NoopEmitter;
use std::sync::Arc;

/// A `MemoryDB` on a temp file. Deliberately not `db::tests::test_db` — nothing
/// here embeds, so this skips the shared fastembed model entirely.
async fn tx_test_db() -> (MemoryDB, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("tx_rollback.db");
    let db = MemoryDB::new(&db_path, Arc::new(NoopEmitter))
        .await
        .expect("open MemoryDB");
    (db, dir)
}

/// Make every `event` on `table` fail, so the next transaction body touching it
/// returns an error mid-transaction.
async fn force_failure_on(db: &MemoryDB, event: &str, table: &str) {
    let conn = db.conn.lock().await;
    conn.execute_batch(&format!(
        "CREATE TRIGGER tx_leak_probe BEFORE {event} ON {table} \
         BEGIN SELECT RAISE(ABORT, 'forced mid-transaction failure'); END;"
    ))
    .await
    .expect("install abort trigger");
}

/// Read the connection's transaction state directly — deliberately NOT through
/// `assert_autocommit_or_recover`, which would heal the leak it is meant to
/// detect and make every assertion below vacuously true.
async fn is_autocommit(db: &MemoryDB) -> bool {
    let conn = db.conn.lock().await;
    conn.is_autocommit()
}

/// Insert one `source='memory'` row directly. Column list mirrors the seeding
/// in `db/main_tests.rs`'s migration-55 tests.
async fn seed_memory(db: &MemoryDB, chunk_id: &str, source_id: &str, content: &str) {
    let conn = db.conn.lock().await;
    conn.execute(
        "INSERT INTO memories (id, content, source, source_id, title, chunk_index, last_modified, chunk_type) \
         VALUES (?1, ?2, 'memory', ?3, 't', 0, 0, 'text')",
        libsql::params![chunk_id, content, source_id],
    )
    .await
    .expect("seed memory row");
}

// ── The watchdog itself ───────────────────────────────────────────────────

#[tokio::test]
async fn watchdog_reports_and_heals_a_leaked_transaction() {
    let (db, _dir) = tx_test_db().await;

    // Simulate exactly what a missing ROLLBACK leaves behind.
    {
        let conn = db.conn.lock().await;
        conn.execute("BEGIN", ()).await.expect("begin");
    }
    assert!(
        !is_autocommit(&db).await,
        "precondition: the connection must be inside a transaction"
    );

    assert!(
        db.assert_autocommit_or_recover("unit test").await,
        "watchdog must report the leak"
    );
    assert!(
        is_autocommit(&db).await,
        "watchdog must roll the connection back to autocommit"
    );
    assert!(
        !db.assert_autocommit_or_recover("unit test").await,
        "watchdog must report nothing on a healthy connection"
    );
}

// ── Tier 1: reachable from requests or background sweeps ──────────────────

#[tokio::test]
async fn link_memory_entities_rolls_back_on_failure() {
    let (db, _dir) = tx_test_db().await;
    force_failure_on(&db, "INSERT", "memory_entities").await;

    let result = db.link_memory_entities("mem_leak", &["ent_leak"]).await;

    assert!(result.is_err(), "forced failure must propagate");
    assert!(
        is_autocommit(&db).await,
        "link_memory_entities must ROLLBACK before returning the error"
    );
}

#[tokio::test]
async fn decay_update_confidence_rolls_back_on_failure() {
    let (db, _dir) = tx_test_db().await;
    seed_memory(&db, "c_decay", "mem_decay", "a memory that will be decayed").await;
    force_failure_on(&db, "UPDATE", "memories").await;

    let result = db.decay_update_confidence().await;

    assert!(result.is_err(), "forced failure must propagate");
    assert!(
        is_autocommit(&db).await,
        "decay_update_confidence must ROLLBACK before returning the error"
    );
}

#[tokio::test]
async fn insert_summary_node_rolls_back_on_failure() {
    let (db, _dir) = tx_test_db().await;
    force_failure_on(&db, "INSERT", "summary_node_sources").await;

    let result = db
        .insert_summary_node(
            "sn_leak",
            0,
            Some("bucket"),
            "title",
            "body",
            &vec![0.0f32; crate::db::EMBEDDING_DIM],
            1,
            0,
            &["mem_leak".to_string()],
        )
        .await;

    assert!(result.is_err(), "forced failure must propagate");
    assert!(
        is_autocommit(&db).await,
        "insert_summary_node must ROLLBACK before returning the error"
    );
}

#[tokio::test]
async fn evict_stale_rolls_back_on_failure() {
    // `WENLAN_ENABLE_EVICTION` is process-global; share the one crate lock every
    // reader and mutator of that flag takes (see `db/main_tests.rs`).
    let _serial = crate::db::tests::EVICT_ENV_LOCK.lock().await;
    let (db, _dir) = tx_test_db().await;

    // One evictable row: unconfirmed, unpinned, low confidence, long unread.
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "INSERT INTO memories (id, content, source, source_id, title, chunk_index, last_modified, chunk_type, memory_type, space, confirmed, pinned, effective_confidence, last_accessed) \
             VALUES ('c_evict', 'stale body', 'memory', 'mem_evict', 't', 0, 0, 'text', 'fact', 'engineering', 0, 0, 0.05, datetime('now','-120 days'))",
            (),
        )
        .await
        .expect("seed evictable row");
    }
    force_failure_on(&db, "UPDATE", "memories").await;

    let result = temp_env::async_with_vars([("WENLAN_ENABLE_EVICTION", Some("1"))], async {
        db.evict_stale(&crate::tuning::EvictionConfig::default())
            .await
    })
    .await;

    assert!(result.is_err(), "forced failure must propagate");
    assert!(
        is_autocommit(&db).await,
        "evict_stale must ROLLBACK before returning the error"
    );
}

#[tokio::test]
async fn flush_access_counts_rolls_back_on_failure() {
    let (db, _dir) = tx_test_db().await;
    seed_memory(&db, "c_flush", "mem_flush", "an accessed memory").await;
    force_failure_on(&db, "UPDATE", "memories").await;

    let result = db.flush_access_counts(&["mem_flush".to_string()]).await;

    assert!(result.is_err(), "forced failure must propagate");
    assert!(
        is_autocommit(&db).await,
        "flush_access_counts must ROLLBACK before returning the error"
    );
}

#[tokio::test]
async fn log_accesses_rolls_back_on_failure() {
    let (db, _dir) = tx_test_db().await;
    force_failure_on(&db, "INSERT", "access_log").await;

    let result = db.log_accesses(&["mem_logged".to_string()]).await;

    assert!(result.is_err(), "forced failure must propagate");
    assert!(
        is_autocommit(&db).await,
        "log_accesses must ROLLBACK before returning the error"
    );
}

// ── Tier 2: boot-reachable ────────────────────────────────────────────────

#[tokio::test]
async fn migration_55_pass_a_rolls_back_on_failure() {
    let (db, _dir) = tx_test_db().await;
    // "in Q2 2024" is the cue `extract_cue_for_content` matches, so pass A
    // reaches its UPDATE (a row with no cue would never fire the trigger).
    seed_memory(&db, "c_m55a", "mem_m55a", "we shipped in Q2 2024").await;
    force_failure_on(&db, "UPDATE", "memories").await;

    let result = db.run_migration_55_pass_a().await;

    assert!(result.is_err(), "forced failure must propagate");
    assert!(
        is_autocommit(&db).await,
        "run_migration_55_pass_a must ROLLBACK before returning the error"
    );
}

#[tokio::test]
async fn migration_55_pass_b_rolls_back_on_failure() {
    let (db, _dir) = tx_test_db().await;
    // An `entity_id` with no live entity behind it, so pass B's self-heal
    // UPDATE — the first statement inside its transaction — matches a row.
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "INSERT INTO memories (id, content, source, source_id, title, chunk_index, last_modified, chunk_type, entity_id) \
             VALUES ('c_m55b', '', 'memory', 'mem_m55b', '', 0, 0, 'text', 'ent_gone')",
            (),
        )
        .await
        .expect("seed orphan-entity row");
    }
    force_failure_on(&db, "UPDATE", "memories").await;

    let result = db.run_migration_55_pass_b().await;

    assert!(result.is_err(), "forced failure must propagate");
    assert!(
        is_autocommit(&db).await,
        "run_migration_55_pass_b must ROLLBACK before returning the error"
    );
}

// ── Tier 3: narrow or request-only ────────────────────────────────────────

#[tokio::test]
async fn set_event_dates_by_source_id_rolls_back_on_failure() {
    let (db, _dir) = tx_test_db().await;
    seed_memory(&db, "c_dates", "mem_dates", "a memory with a date").await;
    force_failure_on(&db, "UPDATE", "memories").await;

    let result = db
        .set_event_dates_by_source_id(&[("mem_dates".to_string(), 1_700_000_000)])
        .await;

    assert!(result.is_err(), "forced failure must propagate");
    assert!(
        is_autocommit(&db).await,
        "set_event_dates_by_source_id must ROLLBACK before returning the error"
    );
}

#[tokio::test]
async fn reorder_space_rolls_back_on_failure() {
    let (db, _dir) = tx_test_db().await;
    db.create_space("leak-a", None, false)
        .await
        .expect("create space a");
    db.create_space("leak-b", None, false)
        .await
        .expect("create space b");
    let target = db
        .get_space("leak-b")
        .await
        .expect("read space b")
        .expect("space b exists");
    force_failure_on(&db, "UPDATE", "spaces").await;

    let result = db.reorder_space("leak-b", target.sort_order - 1).await;

    assert!(result.is_err(), "forced failure must propagate");
    assert!(
        is_autocommit(&db).await,
        "reorder_space must ROLLBACK before returning the error"
    );
}

#[tokio::test]
async fn mark_captures_packaged_rolls_back_on_failure() {
    let (db, _dir) = tx_test_db().await;
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "INSERT INTO capture_refs (source_id, activity_id, snapshot_id, app_name, window_title, timestamp, source) \
             VALUES ('cap_leak', 'act_leak', NULL, 'app', 'window', 0, 'test')",
            (),
        )
        .await
        .expect("seed capture ref");
    }
    force_failure_on(&db, "UPDATE", "capture_refs").await;

    let result = db.mark_captures_packaged(&["cap_leak"], "snap_leak").await;

    assert!(result.is_err(), "forced failure must propagate");
    assert!(
        is_autocommit(&db).await,
        "mark_captures_packaged must ROLLBACK before returning the error"
    );
}
