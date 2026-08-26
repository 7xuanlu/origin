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

use super::tests::{test_db, test_db_at};
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
            "edges_attests_requires_human_root",
            "edges_attests_requires_human_root_update",
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
        .expect("migration 98 must be re-runnable");
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
/// Artifact 3 §4 is explicit that the lane is real: both fence triggers carry
/// `WHEN NEW.lineage != 'legacy'`, so a `claim_revision → memory` row written
/// as legacy skips the fence body entirely and would land in the trusted
/// support set with only writer discipline in the way. The rebuild is the one
/// chance to close it in storage.
///
/// This asserts the PROPERTY — storage refuses the row — and deliberately not
/// the mechanism. Review found that the lineage tooth is never the sole
/// refuser: brute-forcing its whole domain (every `src_kind` × `dst_kind` ×
/// `edge_type` × `grounded` × `root_id` nullity with `lineage='legacy'` and a
/// `claim_revision`/`root` endpoint) against the widened table with the tooth
/// REMOVED accepted 0 of 480, because CHECK #4 and the `supports`/`attests`
/// shape CHECKs already pin `lineage`. The tooth is defense-in-depth, and a
/// test that named it as the refuser would be claiming a protection it is not
/// the one providing. The lane staying shut is what matters and is what this
/// locks; which constraint shuts it is free to change.
#[tokio::test]
async fn a_legacy_lineage_claim_revision_edge_is_refused_by_storage() {
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
    // Without a real `cr1`, the space fence resolves the destination space to
    // NULL and aborts the row on its own — the test would pass with the §5 rule
    // deleted, proving the fence instead of the rule it names.
    seed_attestable_revision(&db).await;
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
    // Same reason as the §5 test above: an unseeded `cr1` lets the space fence
    // do the refusing, and the §3 rule under test never fires.
    seed_attestable_revision(&db).await;
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

/// Seed `pg1 → c1 → cr1`, a claim revision in space `unfiled` that an
/// attestation can legally point at.
///
/// Every attestation-refusal test needs this, and the reason is the whole
/// lesson of the mutation that found it missing. Without a real revision at
/// `cr1`, the space fence resolves the destination to NULL and aborts the row
/// on its own — so the test went green with the human-root trigger deleted, and
/// was proving the fence rather than the rule it names. A refusal test only
/// means something when exactly one rule can do the refusing.
async fn seed_attestable_revision(db: &MemoryDB) {
    db.insert_page(
        "pg1",
        "pg1",
        None,
        "",
        None,
        Some("unfiled"),
        &[],
        "2026-07-27T00:00:00Z",
    )
    .await
    .unwrap();
    let conn = db.conn.lock().await;
    conn.execute_batch(
        "INSERT INTO claims (claim_id, page_id, created_at) VALUES ('c1', 'pg1', 0);
         INSERT INTO claim_revisions (claim_revision_id, claim_id, predecessor_revision_id,
                                      canonical_text, canonical_text_digest, claim_kind,
                                      extractor_version, created_at)
         VALUES ('cr1', 'c1', '', 'a sentence', 'digest', 'observation', 1, 0);",
    )
    .await
    .unwrap();
}

/// Weakening: allow a `generated` root to attest.
///
/// Artifact 3 §5: "The attesting root's `root_kind` must be `human_capture` or
/// `human_edit_delta`. A `generated` root can never attest." This is the rule
/// that decides whether a machine can manufacture human presence, and it is the
/// one §5 rule no CHECK can carry — a CHECK sees only the row being written,
/// and the root's kind lives in another table. Every other column of this row is
/// individually legal, which is exactly why the gap was invisible.
#[tokio::test]
async fn a_generated_root_may_not_attest() {
    let (db, _temp) = test_db().await;
    seed_attestable_revision(&db).await;
    let conn = db.conn.lock().await;
    conn.execute(
        "INSERT INTO provenance_roots (root_id, identity_version, identity_digest,
                                       root_kind, independence_group_id, created_at)
         VALUES ('rg', 1, 'rg', 'generated', 'g', 0)",
        (),
    )
    .await
    .unwrap();

    let refused = conn
        .execute(
            "INSERT INTO edges (edge_id, src_id, src_kind, dst_id, dst_kind, edge_type,
                                lineage, grounded, root_id, space, created_at)
             VALUES ('a1', 'rg', 'root', 'cr1', 'claim_revision', 'attests',
                     'assertion', 0, 'rg', 'unfiled', 0)",
            (),
        )
        .await;
    assert!(
        refused.is_err(),
        "a generated root can never attest (artifact 3 §5)"
    );
}

/// An attestation naming a root that does not exist at all is refused — and
/// review established that the refuser here is the `root_id REFERENCES
/// provenance_roots(root_id)` foreign key, not the human-root trigger.
///
/// Kept, and renamed to say so. The property is real and worth locking: drop
/// that FK and "no such root" starts reading as "nothing to object to". But
/// the trigger's fail-closed `NOT EXISTS` arm is NOT what this exercises, and
/// it cannot be — isolating that arm needs a root the FK accepts and the
/// trigger rejects, which is exactly what `a_generated_root_may_not_attest`
/// above already does with a real-but-machine-generated root. Naming the
/// trigger here would have this test claim a protection its sibling provides.
#[tokio::test]
async fn an_attestation_naming_an_unknown_root_is_refused_by_the_foreign_key() {
    let (db, _temp) = test_db().await;
    seed_attestable_revision(&db).await;
    let conn = db.conn.lock().await;
    let refused = conn
        .execute(
            "INSERT INTO edges (edge_id, src_id, src_kind, dst_id, dst_kind, edge_type,
                                lineage, grounded, root_id, space, created_at)
             VALUES ('a1', 'ghost', 'root', 'cr1', 'claim_revision', 'attests',
                     'assertion', 0, 'ghost', 'unfiled', 0)",
            (),
        )
        .await;
    assert!(refused.is_err(), "an unknown root cannot attest");
}

/// Weakening: enforce the human-root rule on INSERT only.
///
/// An UPDATE can flip `edge_type` to `attests` or repoint `src_id` at a machine
/// root. A rule enforced on one statement is a rule with a documented way
/// around it — the same reason the space fence carries an UPDATE twin.
/// This is also the first test that inserts a **valid** attestation, which the
/// refusal tests above cannot do: they only ever prove a row is rejected, and a
/// fence that rejected everything would satisfy every one of them. Seeding a
/// real page → claim → revision chain is what makes the happy path reachable,
/// and therefore what makes the refusals mean something.
#[tokio::test]
async fn an_update_may_not_repoint_an_attestation_at_a_machine_root() {
    let (db, _temp) = test_db().await;
    seed_attestable_revision(&db).await;
    let conn = db.conn.lock().await;
    conn.execute_batch(
        "INSERT INTO provenance_roots (root_id, identity_version, identity_digest,
                                       root_kind, independence_group_id, created_at)
         VALUES ('rh', 1, 'rh', 'human_capture', 'g', 0);
         INSERT INTO provenance_roots (root_id, identity_version, identity_digest,
                                       root_kind, independence_group_id, created_at)
         VALUES ('rg', 1, 'rg', 'generated', 'g', 0);
         INSERT INTO edges (edge_id, src_id, src_kind, dst_id, dst_kind, edge_type,
                            lineage, grounded, root_id, space, created_at)
         VALUES ('a1', 'rh', 'root', 'cr1', 'claim_revision', 'attests',
                 'assertion', 0, 'rh', 'unfiled', 0);",
    )
    .await
    .expect("a human root attesting a same-space revision is the valid shape");

    let refused = conn
        .execute(
            "UPDATE edges SET src_id = 'rg', root_id = 'rg' WHERE edge_id = 'a1'",
            (),
        )
        .await;
    assert!(
        refused.is_err(),
        "an UPDATE must not move an attestation onto a machine root"
    );
}

/// Weakening: let the migration widen past a pre-M5 `supports` row.
///
/// Artifact 3 §2 reserves `supports` and records that nothing writes it today.
/// A row of that type in an existing database therefore has unknown provenance,
/// and `supports` is precisely the shape M5 reads as truth — so the migration
/// must HALT rather than carry it across into the trusted set.
///
/// The halt mechanism is the copy itself: the row cannot enter the widened
/// table, so the `INSERT ... SELECT` aborts, so the transaction rolls back and
/// `user_version` is never stamped. This test drives that mechanism at the table
/// level rather than through `run_migrations`, because it cannot do otherwise
/// and say anything true: `test_db()` hands back a database already migrated
/// past 97, so there is no pre-97 `edges` left to seed. Hand-forging one would
/// mean writing the old schema into the test — making the fixture a copy of the
/// very thing under test, which proves nothing when the two drift.
///
/// So: build the widened shape the migration builds, and show the row bounces
/// off it. Migration 98 wraps that in a transaction, which is what turns a
/// bounced row into a rolled-back migration.
#[tokio::test]
async fn a_pre_m5_supports_row_cannot_enter_the_widened_table() {
    let (db, _temp) = test_db().await;
    let conn = db.conn.lock().await;
    let tx = conn.transaction().await.unwrap();
    tx.execute(super::edges_rebuild::EDGES_WIDENED_DDL, ())
        .await
        .unwrap();

    let refused = tx
        .execute(
            "INSERT INTO edges_new (edge_id, src_id, src_kind, dst_id, dst_kind, edge_type,
                                    lineage, grounded, space, created_at)
             VALUES ('legacy-sup', 'p1', 'page', 'm1', 'memory', 'supports',
                     'legacy', 0, 'unfiled', 0)",
            (),
        )
        .await;
    assert!(
        refused.is_err(),
        "a supports row of unknown provenance must stop the rebuild, not ride it \
         into the set M5 reads as truth"
    );
}

/// Weakening: skip the copy verification entirely.
///
/// Artifact 3 §9 names this one and names why it is hard: "a clean corpus stays
/// green either way, so the oracle must corrupt something". Every other rebuild
/// test copies a healthy table, so none of them can tell a working verify from
/// an absent one. This one corrupts the copy on purpose, in both directions a
/// copy can be wrong — a row that went missing, and a row that arrived altered.
#[tokio::test]
async fn the_copy_verifier_catches_a_corrupted_copy() {
    let (db, _temp) = test_db().await;
    let conn = db.conn.lock().await;
    conn.execute(
        "INSERT INTO edges (edge_id, src_id, src_kind, dst_id, dst_kind, edge_type,
                            lineage, grounded, space, created_at)
         VALUES ('e1', 'p1', 'page', 'p2', 'page', 'links', 'legacy', 0, 'unfiled', 7)",
        (),
    )
    .await
    .unwrap();

    let tx = conn.transaction().await.unwrap();
    tx.execute(super::edges_rebuild::EDGES_WIDENED_DDL, ())
        .await
        .unwrap();
    let columns = super::edges_rebuild::EDGE_COLUMNS;

    // A faithful copy passes, so a later failure is the corruption talking and
    // not the fixture.
    tx.execute(
        &format!("INSERT INTO edges_new ({columns}) SELECT {columns} FROM edges"),
        (),
    )
    .await
    .unwrap();
    MemoryDB::assert_copy_is_faithful(&tx)
        .await
        .expect("an exact copy is faithful");

    // Corruption 1: a row that never arrived.
    tx.execute("DELETE FROM edges_new WHERE edge_id = 'e1'", ())
        .await
        .unwrap();
    assert!(
        MemoryDB::assert_copy_is_faithful(&tx).await.is_err(),
        "a dropped row must fail the verify"
    );

    // Corruption 2: a row that arrived with a column changed. Row counts match
    // here, so a count-only check would sail straight past it.
    tx.execute(
        &format!("INSERT INTO edges_new ({columns}) SELECT {columns} FROM edges"),
        (),
    )
    .await
    .unwrap();
    tx.execute(
        "UPDATE edges_new SET created_at = 8 WHERE edge_id = 'e1'",
        (),
    )
    .await
    .unwrap();
    assert!(
        MemoryDB::assert_copy_is_faithful(&tx).await.is_err(),
        "an altered row must fail the verify even though the counts agree"
    );
}

/// The §2 restore drill, **executed** rather than promised: "pre-migration
/// online backup + integrity receipt + restore drill executed, before the
/// rebuild — a backup never restored is a hope, not a rollback plan"
/// (`docs/plans/2026-07-27-m5-migration-state-machine.md` §2).
///
/// `online_backup_is_a_restorable_snapshot` already drills the generic backup
/// primitive end to end. What it does not cover, and §2 names, is migration
/// **97's own** restore point: that the rebuild takes one, records its receipt,
/// and leaves a file that opens and genuinely predates the rebuild.
///
/// The last clause is the load-bearing one, and it is asserted structurally
/// rather than by trusting the filename. A snapshot taken *after* the migration
/// would still exist, still pass `integrity_check`, and still open — and would
/// be worthless as a rollback target. So the drill asserts the M5 tables are
/// ABSENT from it: `claims`, `claim_revisions`, and `entailment_cache` are
/// created BY migration 98, so a restore point that contains them was taken too
/// late. That check cannot be satisfied by a backup of the post-rebuild state,
/// which is exactly the failure a rollback plan must not have.
///
/// A row-survival assertion was tried first and removed: `test_db` seeds its
/// schema and migrates during setup, so 97's backup is already taken by the
/// time a test can insert anything, and `backup_before_migration` then
/// deliberately preserves that original restore point instead of
/// re-snapshotting. The skip is correct behavior (a re-snapshot after a failed
/// attempt would copy partially-migrated state over the pristine file), so the
/// drill verifies what the real restore point contains rather than fighting the
/// fixture to plant a row in it.
#[tokio::test]
async fn the_pre_migration_97_backup_is_a_usable_restore_point() {
    // A database that already exists at 96 is what gets a restore point; a
    // store created in this boot takes none.
    let (db, _temp) = test_db_at(96).await;
    db.run_migrations(&crate::events::NoopEmitter)
        .await
        .unwrap();
    let source_path = {
        let conn = db.conn.lock().await;
        MemoryDB::main_db_path(&conn).await.unwrap()
    };
    let dest = source_path
        .parent()
        .unwrap()
        .join("pre_migration_97_backup.db");
    assert!(
        dest.exists(),
        "migration 98 must leave a restore point at {}",
        dest.display()
    );

    let receipt = db
        .get_app_metadata("backup_before_migration_97")
        .await
        .unwrap()
        .expect("the backup receipt must be recorded, or the drill has nothing to verify");
    let receipt: serde_json::Value = serde_json::from_str(&receipt).unwrap();
    assert_eq!(
        receipt["integrity_ok"], true,
        "a restore point that failed integrity_check is not a rollback plan"
    );

    // Open the snapshot directly -- never through `MemoryDB`, which would
    // migrate the very file whose pre-migration state is the thing under test.
    let snapshot = libsql::Builder::new_local(dest.to_str().unwrap())
        .build()
        .await
        .unwrap();
    let conn = snapshot.connect().unwrap();

    let mut rows = conn.query("PRAGMA user_version", ()).await.unwrap();
    let stamped: i64 = rows.next().await.unwrap().unwrap().get(0).unwrap();
    assert!(
        stamped < 97,
        "the restore point must predate the rebuild it protects against, got {stamped}"
    );

    for table in ["claims", "claim_revisions", "entailment_cache"] {
        let mut rows = conn
            .query(
                "SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name = ?1",
                libsql::params![table],
            )
            .await
            .unwrap();
        let present: i64 = rows.next().await.unwrap().unwrap().get(0).unwrap();
        assert_eq!(
            present, 0,
            "{table} is created BY migration 98, so a restore point holding it was \
             taken after the rebuild and rolls back to nothing"
        );
    }

    // The pre-migration schema is intact and queryable, so the file is a real
    // database rather than a valid-but-empty shell.
    let mut rows = conn.query("SELECT count(*) FROM edges", ()).await.unwrap();
    let _: i64 = rows.next().await.unwrap().unwrap().get(0).unwrap();
}

/// Weakening: guard the rebuild with a bare `PRAGMA foreign_key_check`.
///
/// Review found this one, and it was the ship blocker. The bare pragma walks
/// EVERY foreign key in the database, not just the rebuilt table. Migrations
/// have suspended FK enforcement before (`db.rs:4657`, `6584`, `6929`), so a
/// pre-existing orphan in some unrelated table is a real historical
/// possibility — and it would abort 97, roll back, and leave `user_version` at
/// 96. The daemon would then fail on every subsequent boot with
/// "m98 rebuild left a dangling foreign-key reference", blaming the rebuild for
/// damage it did not cause, with no way forward short of hand-surgery on the
/// user's database.
///
/// The guard exists to catch a reference THE REBUILD broke. Scoping it to
/// `edges` is what makes it answer that question and no other.
#[tokio::test]
async fn a_pre_existing_orphan_elsewhere_does_not_block_the_rebuild() {
    let (db, _temp) = test_db().await;
    {
        let conn = db.conn.lock().await;
        // An orphan in a table the rebuild never touches, planted the way a
        // historical migration could have: enforcement off, row in, back on.
        conn.execute("PRAGMA foreign_keys = OFF", ()).await.unwrap();
        conn.execute(
            // `page_id REFERENCES pages(id)` is the FK that goes dangling.
            "INSERT INTO page_sources (page_id, memory_source_id, linked_at)
             VALUES ('no_such_page', 'no_such_memory', 0)",
            (),
        )
        .await
        .unwrap();
        conn.execute("PRAGMA foreign_keys = ON", ()).await.unwrap();

        // Precondition: the bare pragma really does see it. Without this the
        // test would pass on a database where nothing was ever wrong.
        let mut rows = conn.query("PRAGMA foreign_key_check", ()).await.unwrap();
        assert!(
            rows.next().await.unwrap().is_some(),
            "the orphan must be visible to the unscoped pragma, or this proves nothing"
        );
    }

    // The whole chain still runs to completion.
    rerun_migrations_from_96(&db).await;
}

/// Weakening: let `EDGE_COLUMNS` and the live `edges` drift apart.
///
/// Review found this one. Naming the columns explicitly catches a column in the
/// LIST that the table lacks — that is a SQL error. The reverse is silent: a
/// column the TABLE has and the list omits is never selected, never created on
/// `edges_new`, and then compared away by `assert_copy_is_faithful`, which is
/// built from the same list. The verifier reads its own input and agrees with
/// itself while the data is gone.
///
/// Not reachable today — the shipped DDL and the list match column for column.
/// It becomes reachable the moment a later migration adds an `edges` column,
/// which is exactly when nobody will be looking at this file.
#[tokio::test]
async fn a_column_missing_from_the_list_fails_loudly_instead_of_vanishing() {
    let (db, _temp) = test_db().await;
    {
        let conn = db.conn.lock().await;
        conn.execute("ALTER TABLE edges ADD COLUMN future_column TEXT", ())
            .await
            .unwrap();
        conn.execute("PRAGMA user_version = 96", ()).await.unwrap();
    }

    let refused = db.run_migrations(&crate::events::NoopEmitter).await;
    let error = refused.expect_err("a column the rebuild would silently drop must abort it");
    assert!(
        format!("{error}").contains("EDGE_COLUMNS is stale")
            && format!("{error}").contains("future_column"),
        "the failure must name the column that would have been dropped: {error}"
    );
}
