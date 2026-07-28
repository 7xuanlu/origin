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
            "idx_edges_attests_fwd",
            "idx_edges_attests_rev",
            "idx_edges_dst",
            "idx_edges_operation",
            "idx_edges_root",
            "idx_edges_src",
            "idx_edges_superseded",
            "idx_edges_supports_fwd",
            "idx_edges_supports_rev",
        ],
        "six pre-M5 indexes the rebuild replays, plus the four partial \
         support/attest indexes it adds (artifact 3 §6)"
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

/// Rewind `user_version` to 96 and drive the real migration dispatch, the way
/// `migration_58_idempotent` does, so the test exercises the shipped ordering
/// rather than a hand-assembled copy of it. 97 is the rebuild; whatever follows
/// it runs too, which is the point — the rebuild has to leave a substrate the
/// rest of the chain can still migrate.
///
/// The end state is asserted against `SCHEMA_VERSION`, never a literal. An
/// earlier draft of this helper pinned 97 and went red the moment 98 landed,
/// which is the same rot this commit's sibling fix removed from
/// `migration_96_bootstraps_cutover_control_plane_and_first_entity_space`.
async fn rerun_migrations_from_96(db: &MemoryDB) {
    {
        let conn = db.conn.lock().await;
        conn.execute("PRAGMA user_version = 96", ()).await.unwrap();
    }
    db.run_migrations(&crate::events::NoopEmitter)
        .await
        .expect("migration 97 must be re-runnable");
    let conn = db.conn.lock().await;
    let mut rows = conn.query("PRAGMA user_version", ()).await.unwrap();
    let version: i64 = rows.next().await.unwrap().unwrap().get(0).unwrap();
    assert_eq!(
        version,
        i64::from(crate::db::SCHEMA_VERSION),
        "the chain must run to completion, not stall inside the rebuild"
    );
}

/// Weakening: assume the standard SQLite table-rebuild ordering just works
/// here, because it works on a table nothing else references.
///
/// It does not. `m4_page_community_memory_insert_invalidate` (`db.rs:11162`)
/// and its delete twin (`db.rs:11180`) are triggers **on `memories`** whose
/// bodies read `FROM edges e`. They are not attached to `edges`, so
/// `DROP TABLE edges` does not take them — they survive, naming a table that
/// no longer exists. Since SQLite 3.25 `ALTER TABLE edges_new RENAME TO edges`
/// reparses every trigger in the schema to rewrite references to the renamed
/// object, that reparse resolves table names, and it trips over them:
///
/// ```text
/// error in trigger m4_page_community_memory_insert_invalidate:
///   no such table: main.edges
/// ```
///
/// That is measured, not predicted — it is what this test reported before
/// `rename_into_place` suppressed the reparse. Removing the suppression turns
/// it red again.
#[tokio::test]
async fn the_rebuild_survives_the_window_where_edges_does_not_exist() {
    let (db, _temp) = test_db().await;
    let before = {
        let conn = db.conn.lock().await;
        schema_names(&conn, "trigger").await
    };

    rerun_migrations_from_96(&db).await;

    let conn = db.conn.lock().await;
    assert_eq!(
        schema_names(&conn, "trigger").await,
        before,
        "every trigger on edges must come back"
    );

    // The surviving triggers on `memories` must still resolve `edges`. A
    // reparse that quietly repointed them elsewhere would leave this insert
    // working and the invalidation dead, so the assertion is that the trigger
    // body runs at all.
    conn.execute(
        "INSERT INTO memories (id, content, source, source_id, title, chunk_index,
                              last_modified, chunk_type, space)
         VALUES ('m-fence', 'body', 'memory', 'm-fence', 't', 0, 0, 'text', 'unfiled')",
        (),
    )
    .await
    .expect("m4 memories triggers must still find edges");
}

/// Weakening: verify the rebuild by row count alone.
///
/// A count survives a column-order mistake in the copy; the row contents do
/// not. Seeded with `lineage='legacy'`, which the space fence skips by design,
/// so the fixture needs no page or memory endpoints to exist.
#[tokio::test]
async fn the_rebuild_preserves_every_row_unchanged() {
    let (db, _temp) = test_db().await;
    let before = {
        let conn = db.conn.lock().await;
        for (id, weight) in [("e1", "0.5"), ("e2", "NULL")] {
            conn.execute(
                &format!(
                    "INSERT INTO edges (edge_id, src_id, src_kind, dst_id, dst_kind, edge_type,
                                        lineage, grounded, space, weight, payload, created_at)
                     VALUES ('{id}', 'p1', 'page', 'm1', 'memory', 'cites', 'legacy', 0,
                             'unfiled', {weight}, '{{\"k\":1}}', 7)"
                ),
                (),
            )
            .await
            .unwrap();
        }
        dump_edges(&conn).await
    };
    assert_eq!(before.len(), 2, "fixture seeded");

    rerun_migrations_from_96(&db).await;

    let conn = db.conn.lock().await;
    assert_eq!(
        dump_edges(&conn).await,
        before,
        "rows must survive verbatim"
    );
}

/// Weakening: widen the CHECK enumerations and call the fail-open lane closed.
///
/// Artifact 3 §4 is explicit that it is not: both fence triggers carry
/// `WHEN NEW.lineage != 'legacy'`, so a `claim_revision → memory` row written
/// as legacy skips the fence body entirely and lands in the trusted support set
/// with only writer discipline in the way. A CHECK has no lineage exemption,
/// which is why the rebuild is the one chance to add one.
#[tokio::test]
async fn a_legacy_lineage_claim_revision_edge_is_refused_by_the_check() {
    let (db, _temp) = test_db().await;
    let conn = db.conn.lock().await;

    let refused = conn
        .execute(
            "INSERT INTO edges (edge_id, src_id, src_kind, dst_id, dst_kind, edge_type,
                                lineage, grounded, space, created_at)
             VALUES ('bypass', 'cr1', 'claim_revision', 'm1', 'memory', 'supports',
                     'legacy', 1, 'unfiled', 0)",
            (),
        )
        .await;
    assert!(
        refused.is_err(),
        "a legacy-lineage claim_revision edge bypasses the fence, so storage must refuse it"
    );
}

/// Weakening: accept any attestation whose columns are individually legal.
///
/// Artifact 3 §5 requires `attests.root_id` to equal `src_id`. Without it an
/// attestation could claim one root's authority while pointing at another's
/// identity — the two roots are both real, both permitted, and nothing else in
/// the row is wrong.
#[tokio::test]
async fn an_attestation_may_not_name_a_root_other_than_its_source() {
    let (db, _temp) = test_db().await;
    let conn = db.conn.lock().await;
    for root in ["r-src", "r-other"] {
        conn.execute(
            &format!(
                "INSERT INTO provenance_roots (root_id, identity_version, identity_digest,
                                               root_kind, independence_group_id, created_at)
                 VALUES ('{root}', 1, '{root}', 'human_capture', 'g', 0)"
            ),
            (),
        )
        .await
        .unwrap();
    }

    let refused = conn
        .execute(
            "INSERT INTO edges (edge_id, src_id, src_kind, dst_id, dst_kind, edge_type,
                                lineage, grounded, root_id, space, created_at)
             VALUES ('a1', 'r-src', 'root', 'cr1', 'claim_revision', 'attests',
                     'assertion', 0, 'r-other', 'unfiled', 0)",
            (),
        )
        .await;
    assert!(
        refused.is_err(),
        "attests.root_id must equal src_id (artifact 3 §5)"
    );
}

/// Weakening: let a human attestation set `grounded=1`.
///
/// Artifact 3 §3: human presence is provenance, not grounding. A human
/// clicking approve does not make an ungrounded claim grounded — that is
/// exactly the D2 axis collapse this rung exists to forbid.
#[tokio::test]
async fn an_attestation_may_not_assert_grounding() {
    let (db, _temp) = test_db().await;
    let conn = db.conn.lock().await;
    conn.execute(
        "INSERT INTO provenance_roots (root_id, identity_version, identity_digest,
                                       root_kind, independence_group_id, created_at)
         VALUES ('r1', 1, 'r1', 'human_capture', 'g', 0)",
        (),
    )
    .await
    .unwrap();

    let refused = conn
        .execute(
            "INSERT INTO edges (edge_id, src_id, src_kind, dst_id, dst_kind, edge_type,
                                lineage, grounded, root_id, space, created_at)
             VALUES ('a1', 'r1', 'root', 'cr1', 'claim_revision', 'attests',
                     'assertion', 1, 'r1', 'unfiled', 0)",
            (),
        )
        .await;
    assert!(refused.is_err(), "attestation is never grounding");
}

async fn schema_names(conn: &libsql::Connection, kind: &str) -> Vec<String> {
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

async fn dump_edges(conn: &libsql::Connection) -> Vec<String> {
    let mut rows = conn
        .query(
            "SELECT edge_id || '|' || src_id || '|' || src_kind || '|' || dst_id || '|'
                 || dst_kind || '|' || edge_type || '|' || lineage || '|' || grounded || '|'
                 || coalesce(root_id,'~') || '|' || space || '|' || coalesce(weight,'~') || '|'
                 || coalesce(payload,'~') || '|' || coalesce(provenance,'~') || '|'
                 || coalesce(operation_id,'~') || '|' || created_at || '|'
                 || coalesce(superseded_by,'~') || '|' || coalesce(valid_until,'~')
               FROM edges ORDER BY edge_id",
            (),
        )
        .await
        .unwrap();
    let mut dumped = Vec::new();
    while let Some(row) = rows.next().await.unwrap() {
        dumped.push(row.get::<String>(0).unwrap());
    }
    dumped
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
