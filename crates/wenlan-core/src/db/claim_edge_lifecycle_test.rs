// SPDX-License-Identifier: Apache-2.0
//! F5: what happens to a support or attestation edge when the row underneath it
//! moves or disappears.
//!
//! `valid_until IS NULL` is a live assertion that a claim is still supported by
//! a piece of evidence, in a space where both are visible. Ordinary curation
//! falsifies that without touching the edge: a page moves to another space, an
//! evidence memory moves, a page is deleted and its claims cascade away while
//! the polymorphic edges — which carry no foreign key — survive pointing at
//! nothing. Every one of these leaves a row that still reads as a live, fenced,
//! space-consistent support.
//!
//! Each test asserts the PROPERTY (the edge stops being active), not the
//! mechanism, so the triggers may be replaced by something better without these
//! going stale. The negative controls are the other half: a rule that retracts
//! too much is as wrong as one that retracts too little, and only the pairing
//! can tell the two apart.

use super::tests::test_db;
use super::MemoryDB;

const SPACE_A: &str = "space-a";
const SPACE_B: &str = "space-b";

/// A page with one claim revision, one evidence memory, and a human root — the
/// smallest world in which both a `supports` and an `attests` edge are legal.
///
/// Everything lands in `SPACE_A` because the space fence refuses any edge whose
/// endpoints disagree with it; a fixture that skipped this would be testing the
/// fence rather than the lifecycle.
async fn seed(db: &MemoryDB) {
    db.insert_page(
        "pg1",
        "pg1",
        None,
        "",
        None,
        Some(SPACE_A),
        &[],
        "2026-07-27T00:00:00Z",
    )
    .await
    .unwrap();
    let conn = db.conn.lock().await;
    conn.execute_batch(&format!(
        "INSERT INTO memories (id, content, source, source_id, title, chunk_index,
                               last_modified, chunk_type, space)
         VALUES ('m1', 'evidence', 'memory', 'mem1', 'title', 0, 0, 'text', '{SPACE_A}');

         INSERT INTO claims (claim_id, page_id, created_at) VALUES ('c1', 'pg1', 0);
         INSERT INTO claim_revisions (claim_revision_id, claim_id, predecessor_revision_id,
                                      canonical_text, canonical_text_digest, claim_kind,
                                      extractor_version, created_at)
         VALUES ('cr1', 'c1', '', 'a sentence', 'digest', 'observation', 1, 0);

         INSERT INTO provenance_roots (root_id, identity_version, identity_digest,
                                       root_kind, independence_group_id, created_at)
         VALUES ('r1', 1, 'r1', 'human_capture', 'g', 0);

         INSERT INTO edges (edge_id, src_id, src_kind, dst_id, dst_kind, edge_type,
                            lineage, grounded, root_id, space, created_at)
         VALUES ('s1', 'cr1', 'claim_revision', 'mem1', 'memory', 'supports',
                 'evidence', 1, 'r1', '{SPACE_A}', 0);
         INSERT INTO edges (edge_id, src_id, src_kind, dst_id, dst_kind, edge_type,
                            lineage, grounded, root_id, space, created_at)
         VALUES ('a1', 'r1', 'root', 'cr1', 'claim_revision', 'attests',
                 'assertion', 0, 'r1', '{SPACE_A}', 0);"
    ))
    .await
    .unwrap();
}

async fn is_active(db: &MemoryDB, edge_id: &str) -> bool {
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT valid_until IS NULL FROM edges WHERE edge_id = ?1",
            libsql::params![edge_id],
        )
        .await
        .unwrap();
    let row = rows.next().await.unwrap().expect("edge row");
    row.get::<i64>(0).unwrap() == 1
}

/// Weakening: let the page move and leave the support edge alone.
///
/// The edge keeps `space = SPACE_A` while its source claim now lives in
/// `SPACE_B`. Nothing complains — the fence only guards writes — so a
/// space-scoped reader keeps counting a support that crosses a boundary it is
/// not allowed to cross.
///
/// Goes through `set_page_workspace` rather than raw SQL: the invariant is
/// worth nothing if the production path is the one that bypasses it.
#[tokio::test]
async fn moving_a_page_retracts_the_support_for_its_claims() {
    let (db, _temp) = test_db().await;
    seed(&db).await;
    assert!(
        is_active(&db, "s1").await,
        "seeded support must start active"
    );

    db.set_page_workspace("pg1", Some(SPACE_B)).await.unwrap();

    assert!(
        !is_active(&db, "s1").await,
        "the claim moved to another space; its support cannot still be live"
    );
}

/// Same move, attestation half. `attests` points the other way — root into
/// revision — so a retraction keyed only on `src_id` would miss it entirely.
#[tokio::test]
async fn moving_a_page_retracts_attestations_over_its_claims() {
    let (db, _temp) = test_db().await;
    seed(&db).await;

    db.set_page_workspace("pg1", Some(SPACE_B)).await.unwrap();

    assert!(
        !is_active(&db, "a1").await,
        "a human approved this claim where it used to live, not where it now is"
    );
}

/// Weakening: delete the page and trust the foreign keys.
///
/// There are none to trust. `claims` and `claim_revisions` cascade from
/// `pages`, but an edge endpoint is `(kind, id)` with no FK behind it, so both
/// edges survive the delete — active, and now naming a revision that no longer
/// exists.
#[tokio::test]
async fn deleting_a_page_retracts_both_edge_directions() {
    let (db, _temp) = test_db().await;
    seed(&db).await;

    db.delete_page("pg1").await.unwrap();

    assert!(
        !is_active(&db, "s1").await,
        "the supported claim is gone; the support cannot outlive it"
    );
    assert!(
        !is_active(&db, "a1").await,
        "the attested claim is gone; the attestation cannot outlive it"
    );
    // Retraction, not deletion: the support was true until the page went away,
    // and when it stopped being true is the part worth keeping.
    let conn = db.conn.lock().await;
    let mut rows = conn.query("SELECT COUNT(*) FROM edges", ()).await.unwrap();
    let surviving: i64 = rows.next().await.unwrap().unwrap().get(0).unwrap();
    assert_eq!(
        surviving, 2,
        "edges are soft-invalidated, never hard-deleted"
    );
}

/// The evidence side of the same hazard: the memory moves, the claim does not.
#[tokio::test]
async fn moving_the_evidence_memory_retracts_the_support() {
    let (db, _temp) = test_db().await;
    seed(&db).await;

    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE memories SET space = ?1 WHERE source_id = 'mem1'",
            libsql::params![SPACE_B],
        )
        .await
        .unwrap();
    }

    assert!(
        !is_active(&db, "s1").await,
        "the evidence left the space the support was stamped with"
    );
    assert!(
        is_active(&db, "a1").await,
        "the attestation is over the claim, not the evidence -- it must survive"
    );
}

/// Deleting the evidence, the mirror of deleting the page.
///
/// Not one of the four paths F5 named — found while closing them, and the same
/// defect: `memories` carries no foreign key from `edges` either, so the
/// support survives naming a memory that no longer exists.
#[tokio::test]
async fn deleting_the_evidence_memory_retracts_the_support() {
    let (db, _temp) = test_db().await;
    seed(&db).await;

    {
        let conn = db.conn.lock().await;
        conn.execute("DELETE FROM memories WHERE source_id = 'mem1'", ())
            .await
            .unwrap();
    }

    assert!(
        !is_active(&db, "s1").await,
        "the evidence is gone; the support cannot outlive it"
    );
    assert!(
        is_active(&db, "a1").await,
        "the attestation is over the claim, not the evidence"
    );
}

/// Weakening: retract as soon as any row of a chunked memory goes.
///
/// One logical memory is several `memories` rows sharing a `source_id`.
/// Deleting one chunk leaves the memory there, so the support is still true —
/// and a rule that fired anyway would quietly drop support every time a long
/// memory was re-chunked.
#[tokio::test]
async fn deleting_one_chunk_of_a_multi_chunk_memory_retracts_nothing() {
    let (db, _temp) = test_db().await;
    seed(&db).await;
    {
        let conn = db.conn.lock().await;
        conn.execute(
            &format!(
                "INSERT INTO memories (id, content, source, source_id, title, chunk_index,
                                       last_modified, chunk_type, space)
                 VALUES ('m1b', 'more', 'memory', 'mem1', 'title', 1, 0, 'text', '{SPACE_A}')"
            ),
            (),
        )
        .await
        .unwrap();
        conn.execute("DELETE FROM memories WHERE id = 'm1b'", ())
            .await
            .unwrap();
    }

    assert!(
        is_active(&db, "s1").await,
        "chunk 1 went; the memory the support names is still there"
    );
}

/// The fourth F5 path, closed by refusal rather than repair.
///
/// If an attesting root could be flipped from human to `generated`, the active
/// attestation over it would become a machine-manufactured human approval — and
/// no trigger could retract it afterwards, because the human-root rule guards
/// every UPDATE naming `attests` and would refuse the repair itself.
#[tokio::test]
async fn an_attesting_roots_kind_cannot_be_rewritten() {
    let (db, _temp) = test_db().await;
    seed(&db).await;
    let conn = db.conn.lock().await;

    let refused = conn
        .execute(
            "UPDATE provenance_roots SET root_kind = 'generated' WHERE root_id = 'r1'",
            (),
        )
        .await;

    assert!(
        refused.is_err(),
        "root_kind is part of the root's own content address; it cannot be rewritten"
    );
}

/// Weakening: fire on any write to `pages`, or on any space column mention.
///
/// A trigger that retracted whenever the row was touched would invalidate the
/// whole support set on every unrelated page edit — the failure mode that looks
/// like "the system forgot everything" rather than like a bug.
#[tokio::test]
async fn an_unrelated_page_write_retracts_nothing() {
    let (db, _temp) = test_db().await;
    seed(&db).await;

    // Same space written again, plus a genuinely unrelated column.
    db.set_page_workspace("pg1", Some(SPACE_A)).await.unwrap();
    db.set_page_review_status("pg1", "confirmed").await.unwrap();

    assert!(is_active(&db, "s1").await, "no endpoint moved");
    assert!(is_active(&db, "a1").await, "no endpoint moved");
}

/// Weakening: retract by page, not by claim.
///
/// A second page's move must not touch the first page's edges. Scoping the
/// retraction through `claims.page_id` is what makes this hold; a coarser rule
/// (retract everything in the space being left) would pass every other test
/// here and fail this one.
#[tokio::test]
async fn moving_a_different_page_leaves_this_pages_edges_alone() {
    let (db, _temp) = test_db().await;
    seed(&db).await;
    db.insert_page(
        "pg2",
        "pg2",
        None,
        "",
        None,
        Some(SPACE_A),
        &[],
        "2026-07-27T00:00:00Z",
    )
    .await
    .unwrap();

    db.set_page_workspace("pg2", Some(SPACE_B)).await.unwrap();

    assert!(is_active(&db, "s1").await, "pg1's claim did not move");
    assert!(is_active(&db, "a1").await, "pg1's claim did not move");
}

/// Weakening: re-stamp `valid_until` on every firing.
///
/// The moment an edge stopped being true is a fact about the past, and a later
/// move of the same page must not overwrite it. The `valid_until IS NULL` guard
/// is what holds that.
///
/// The retraction timestamp is pinned to a sentinel rather than read from the
/// first move, because `strftime('%s','now')` has one-second resolution: two
/// moves in the same test land on the same second, so comparing real timestamps
/// would pass whether the guard existed or not. Confirmed by mutation — with
/// the guard removed, the timestamp version of this test stayed green.
#[tokio::test]
async fn a_second_move_does_not_rewrite_the_original_retraction() {
    let (db, _temp) = test_db().await;
    seed(&db).await;

    db.set_page_workspace("pg1", Some(SPACE_B)).await.unwrap();
    {
        let conn = db.conn.lock().await;
        conn.execute("UPDATE edges SET valid_until = 42 WHERE edge_id = 's1'", ())
            .await
            .unwrap();
    }

    db.set_page_workspace("pg1", Some(SPACE_A)).await.unwrap();

    assert_eq!(
        retracted_at(&db, "s1").await,
        42,
        "the edge was already retracted; when it happened does not change"
    );
}

async fn retracted_at(db: &MemoryDB, edge_id: &str) -> i64 {
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT valid_until FROM edges WHERE edge_id = ?1",
            libsql::params![edge_id],
        )
        .await
        .unwrap();
    rows.next()
        .await
        .unwrap()
        .expect("edge row")
        .get::<i64>(0)
        .expect("retracted edges carry a timestamp")
}
