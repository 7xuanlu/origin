// SPDX-License-Identifier: Apache-2.0
//! Pre-conditions for the M5 `edges` rebuild (artifact 3 §7).
//!
//! SQLite cannot alter a CHECK constraint, so widening `edges` for the
//! `claim_revision`/`root` endpoint kinds and the `attests` edge type requires
//! dropping and recreating the table. Everything attached to it — every index
//! and every trigger — goes with it.
//!
//! These tests record what is actually attached, read from `sqlite_master` on a
//! freshly migrated database rather than from a hand-written list. The rebuild
//! must restore exactly this set.

use super::tests::test_db;
use super::MemoryDB;

async fn schema_objects(kind: &str) -> Vec<String> {
    let (db, _temp) = test_db().await;
    let conn = db.conn.lock().await;
    // `sql IS NOT NULL` skips SQLite's implicit index for `edge_id TEXT PRIMARY
    // KEY`: it is derived from the PK declaration, has no `CREATE` statement of
    // its own, and returns with the `CREATE TABLE`. Same predicate the capture
    // helper uses, so the census and the rebuild cannot disagree about what
    // counts as attached.
    let mut rows = conn
        .query(
            "SELECT name FROM sqlite_master
             WHERE type = ?1 AND tbl_name = 'edges' AND sql IS NOT NULL
             ORDER BY name",
            libsql::params![kind],
        )
        .await
        .unwrap();
    let mut names = Vec::new();
    while let Some(row) = rows.next().await.unwrap() {
        names.push(row.get::<String>(0).unwrap());
    }
    names
}

/// Weakening: recreate only the triggers a spec paragraph happens to name.
///
/// `docs/plans/2026-07-27-m5-edge-rebuild-matrix.md` §7 step 6 says "recreate
/// triggers (space fence, both twins)" — two names. The live schema carries
/// eight. A rebuild that trusts the prose silently drops the six M4 triggers
/// that keep community grouping and page-community route inputs invalidated,
/// and nothing downstream would fail loudly: the tables stay correct-looking
/// while their invalidation stops firing.
///
/// The rebuild must diff against `sqlite_master`, not against a list.
#[tokio::test]
async fn every_trigger_on_edges_is_recorded_for_the_rebuild() {
    assert_eq!(
        schema_objects("trigger").await,
        [
            "edges_space_fence",
            "edges_space_fence_update",
            "m4_grouping_edge_delete",
            "m4_grouping_edge_insert",
            "m4_grouping_edge_update",
            "m4_page_community_edge_delete_invalidate",
            "m4_page_community_edge_insert_invalidate",
            "m4_page_community_edge_update_invalidate",
        ],
        "the edges rebuild must restore every one of these; update this list \
         and the rebuild together, never one alone"
    );
}

/// Same weakening, index half. Dropping `edges` drops all of these too.
#[tokio::test]
async fn every_index_on_edges_is_recorded_for_the_rebuild() {
    assert_eq!(
        schema_objects("index").await,
        [
            "idx_edges_active_grounded_space_type",
            "idx_edges_dst",
            "idx_edges_operation",
            "idx_edges_root",
            "idx_edges_src",
            "idx_edges_superseded",
        ],
        "the edges rebuild must restore every one of these, plus the four \
         partial support/attest indexes it adds (artifact 3 §6)"
    );
}

/// Weakening: hand-maintain the recreate list instead of reading the schema.
///
/// The round trip is the whole point — capture, drop, recreate the table,
/// replay — because it stays correct when a later rung attaches a ninth
/// trigger without touching the rebuild.
#[tokio::test]
async fn capture_and_replay_round_trip_restores_every_attached_object() {
    let (db, _temp) = test_db().await;
    let conn = db.conn.lock().await;
    conn.execute_batch(
        "CREATE TABLE scratch (id TEXT PRIMARY KEY, v INTEGER NOT NULL);
         CREATE INDEX idx_scratch_v ON scratch(v);
         CREATE INDEX idx_scratch_positive ON scratch(v) WHERE v > 0;
         CREATE TRIGGER scratch_guard AFTER INSERT ON scratch
         BEGIN
             SELECT RAISE(ABORT, 'negative') WHERE NEW.v < 0;
         END;",
    )
    .await
    .unwrap();

    let tx = conn.transaction().await.unwrap();
    let captured = MemoryDB::capture_attached_objects(&tx, "scratch")
        .await
        .unwrap();
    tx.commit().await.unwrap();
    assert_eq!(
        captured.len(),
        3,
        "two indexes and one trigger: {captured:?}"
    );

    conn.execute_batch(
        "DROP TABLE scratch;
         CREATE TABLE scratch (id TEXT PRIMARY KEY, v INTEGER NOT NULL);",
    )
    .await
    .unwrap();

    let tx = conn.transaction().await.unwrap();
    assert!(
        MemoryDB::capture_attached_objects(&tx, "scratch")
            .await
            .unwrap()
            .is_empty(),
        "DROP TABLE must take the attached objects with it"
    );
    MemoryDB::replay_attached_objects(&tx, &captured)
        .await
        .unwrap();
    let restored = MemoryDB::capture_attached_objects(&tx, "scratch")
        .await
        .unwrap();
    tx.commit().await.unwrap();
    assert_eq!(restored, captured, "replay must restore the exact set");

    // The trigger is not merely present, it fires.
    let rejected = conn
        .execute("INSERT INTO scratch VALUES ('a', -1)", ())
        .await;
    assert!(rejected.is_err(), "replayed trigger must still abort");
}

/// Weakening: filter implicit indexes by an `sqlite_autoindex%` name prefix.
///
/// The property that makes them unreplayable is that they have no statement,
/// not that they have a particular name — and replaying a NULL is what a
/// name-prefix filter eventually lets through.
#[tokio::test]
async fn capture_skips_the_implicit_primary_key_index() {
    let (db, _temp) = test_db().await;
    let conn = db.conn.lock().await;
    conn.execute("CREATE TABLE scratch (id TEXT PRIMARY KEY)", ())
        .await
        .unwrap();

    let tx = conn.transaction().await.unwrap();
    let captured = MemoryDB::capture_attached_objects(&tx, "scratch")
        .await
        .unwrap();
    tx.commit().await.unwrap();

    assert!(
        captured.is_empty(),
        "the PK's implicit index has no CREATE statement to replay: {captured:?}"
    );
}

/// Weakening: swallow a replay failure. That leaves a widened table with part
/// of its fence missing — artifact 3 §8's "after rename, before triggers"
/// window, which must refuse to serve rather than accept writes.
#[tokio::test]
async fn replay_fails_loud_on_a_statement_that_does_not_apply() {
    let (db, _temp) = test_db().await;
    let conn = db.conn.lock().await;
    let tx = conn.transaction().await.unwrap();

    let result = MemoryDB::replay_attached_objects(
        &tx,
        &["CREATE INDEX idx_missing ON no_such_table(col)".to_string()],
    )
    .await;

    assert!(result.is_err(), "replay must not swallow a failure");
}
