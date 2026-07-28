// SPDX-License-Identifier: Apache-2.0
//! Teeth for the M5 claim-identity substrate. Each test is one row of a Stage 0
//! mutation table; the weakening it guards is named in the doc comment.

use super::tests::test_db;
use super::MemoryDB;

/// Build the substrate on a fully migrated database, so the test also proves
/// the DDL composes with the real schema. Pages `p1`/`p2` are seeded because
/// foreign keys are enforced here — which is itself part of the contract: a
/// claim, marker, or truth row for a page that does not exist is not a state
/// M5 should be able to represent.
async fn db_with_substrate() -> (MemoryDB, tempfile::TempDir) {
    let (db, temp) = test_db().await;
    for id in ["p1", "p2"] {
        db.insert_page(id, id, None, "", None, None, &[], "2026-07-27T00:00:00Z")
            .await
            .unwrap();
    }
    {
        let conn = db.conn.lock().await;
        let tx = conn.transaction().await.unwrap();
        MemoryDB::ensure_claim_identity_tables(&tx).await.unwrap();
        tx.commit().await.unwrap();
    }
    (db, temp)
}

#[tokio::test]
async fn substrate_is_idempotent() {
    // A resumed or re-fired migration must converge, not fail.
    let (db, _temp) = db_with_substrate().await;
    let conn = db.conn.lock().await;
    let tx = conn.transaction().await.unwrap();
    MemoryDB::ensure_claim_identity_tables(&tx).await.unwrap();
    tx.commit().await.unwrap();
}

#[tokio::test]
async fn every_substrate_table_exists() {
    let (db, _temp) = db_with_substrate().await;
    let conn = db.conn.lock().await;
    for table in [
        "claims",
        "claim_revisions",
        "claim_anchors",
        "page_version_claims",
        "claim_derivation_markers",
        "entailment_cache",
        "page_truth_state",
        "claim_derivation_jobs",
        "presence_nonces",
        "presence_receipts",
    ] {
        let mut rows = conn
            .query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?1",
                libsql::params![table],
            )
            .await
            .unwrap();
        assert!(
            rows.next().await.unwrap().is_some(),
            "missing substrate table: {table}"
        );
    }
}

/// Weakening: let a revision's text be edited in place. That would move every
/// support and attestation onto text nobody judged (artifact 1 §1).
#[tokio::test]
async fn claim_revisions_reject_update() {
    let (db, _temp) = db_with_substrate().await;
    let conn = db.conn.lock().await;
    conn.execute_batch(
        "INSERT INTO claims VALUES ('c1','p1',1);
         INSERT INTO claim_revisions
            VALUES ('r1','c1','','the alarm is armed','digest','assertion',1,1);",
    )
    .await
    .unwrap();

    let result = conn
        .execute(
            "UPDATE claim_revisions SET canonical_text='something else' WHERE claim_revision_id='r1'",
            (),
        )
        .await;

    assert!(result.is_err(), "revision UPDATE must abort");
}

/// Weakening: let `human_reviewed` be set without binding the exact version and
/// digest. Approval would then float free of the text it approved
/// (artifact 2 §2).
#[tokio::test]
async fn human_reviewed_requires_version_and_digest() {
    let (db, _temp) = db_with_substrate().await;
    let conn = db.conn.lock().await;

    let unbound = conn
        .execute(
            "INSERT INTO page_truth_state
                (page_id,page_version,support_status,human_reviewed,updated_at)
             VALUES ('p1',1,'provisional',1,1)",
            (),
        )
        .await;
    assert!(
        unbound.is_err(),
        "human_reviewed=1 without version+digest must be rejected"
    );

    conn.execute(
        "INSERT INTO page_truth_state
            (page_id,page_version,support_status,human_reviewed,
             reviewed_page_version,reviewed_page_digest,updated_at)
         VALUES ('p1',1,'provisional',1,1,'digest',1)",
        (),
    )
    .await
    .expect("bound review must be accepted");
}

/// Weakening: treat any state other than `supported` as a free-text field.
/// The machine axis is a whitelist, so an unanticipated value must fail rather
/// than pass through as a third truth state (artifact 2 §1).
#[tokio::test]
async fn support_status_is_a_whitelist() {
    let (db, _temp) = db_with_substrate().await;
    let conn = db.conn.lock().await;
    let result = conn
        .execute(
            "INSERT INTO page_truth_state
                (page_id,page_version,support_status,human_reviewed,updated_at)
             VALUES ('p1',1,'probably-fine',0,1)",
            (),
        )
        .await;
    assert!(result.is_err(), "support_status must reject unknown values");
}

/// Weakening: key the entailment cache on fewer than five parts. Two weight-sets
/// of one `model_id` would then share cached scores under a single threshold
/// (artifact 6 §2).
#[tokio::test]
async fn entailment_cache_key_is_all_five_parts() {
    let (db, _temp) = db_with_substrate().await;
    let conn = db.conn.lock().await;
    conn.execute_batch(
        "INSERT INTO entailment_cache
            VALUES ('claim','span','qwen','v1','p1',0.9,0.5,'metal',1);
         INSERT INTO entailment_cache
            VALUES ('claim','span','qwen','v2','p1',0.1,0.5,'metal',1);",
    )
    .await
    .expect("rows differing only in model_version are distinct cache entries");

    let mut rows = conn
        .query("SELECT COUNT(*) FROM entailment_cache", ())
        .await
        .unwrap();
    let count: i64 = rows.next().await.unwrap().unwrap().get(0).unwrap();
    assert_eq!(count, 2);

    let dup = conn
        .execute(
            "INSERT INTO entailment_cache
                VALUES ('claim','span','qwen','v1','p1',0.7,0.5,'metal',2)",
            (),
        )
        .await;
    assert!(dup.is_err(), "a full five-part key collision must conflict");
}

/// Weakening: refuse to record a marker for a page that derives to zero claims.
/// That leaves the page in the 'never derived' state -- an unknown, not an
/// outcome -- and holds readiness under 100% forever (artifact 9 §5).
#[tokio::test]
async fn a_zero_claim_page_can_carry_a_derivation_marker() {
    let (db, _temp) = db_with_substrate().await;
    let conn = db.conn.lock().await;
    conn.execute(
        "INSERT INTO claim_derivation_markers VALUES ('p1',1,'digest',1,0,1)",
        (),
    )
    .await
    .expect("zero-claim marker must be storable");

    let negative = conn
        .execute(
            "INSERT INTO claim_derivation_markers VALUES ('p2',1,'digest',1,-1,1)",
            (),
        )
        .await;
    assert!(negative.is_err(), "inventory_count must reject negatives");
}

/// Weakening: dedupe derivation jobs by page alone. A page would then be unable
/// to hold a job for a new version while an old one is still queued.
#[tokio::test]
async fn derivation_jobs_dedupe_per_page_version() {
    let (db, _temp) = db_with_substrate().await;
    let conn = db.conn.lock().await;
    conn.execute_batch(
        "INSERT INTO claim_derivation_jobs
            (job_id,page_id,page_version,status,created_at,updated_at)
            VALUES ('j1','p1',1,'pending',1,1);
         INSERT INTO claim_derivation_jobs
            (job_id,page_id,page_version,status,created_at,updated_at)
            VALUES ('j2','p1',2,'pending',1,1);",
    )
    .await
    .expect("two versions of one page are two jobs");

    let dup = conn
        .execute(
            "INSERT INTO claim_derivation_jobs
                (job_id,page_id,page_version,status,created_at,updated_at)
                VALUES ('j3','p1',1,'pending',1,1)",
            (),
        )
        .await;
    assert!(dup.is_err(), "same page+version must not queue twice");
}

/// Weakening: migrate a page's existing state into `support_status` because
/// something about it looks vetted.
///
/// Nothing about a pre-M5 page carries the claim `support_status` makes. It
/// says the D8 finalizer evaluated this exact version and found supporting
/// evidence, and that has never run. A page with citations, a page distilled
/// from many sources, a page a human once edited — none of those are that.
#[tokio::test]
async fn the_backfill_infers_nothing_and_covers_every_page() {
    let (db, _temp) = db_with_substrate().await;
    let conn = db.conn.lock().await;
    // Dress p1 up as something a lenient migration might read as approved.
    conn.execute(
        "UPDATE pages SET citations = '[{\"n\":1,\"source_id\":\"m1\"}]' WHERE id = 'p1'",
        (),
    )
    .await
    .unwrap();

    let tx = conn.transaction().await.unwrap();
    let filled = MemoryDB::backfill_page_truth_state(&tx).await.unwrap();
    tx.commit().await.unwrap();
    assert_eq!(filled, 2, "both seeded pages must be covered");

    let mut rows = conn
        .query(
            "SELECT page_id, support_status, human_reviewed, reviewed_page_version
               FROM page_truth_state ORDER BY page_id",
            (),
        )
        .await
        .unwrap();
    let mut seen = Vec::new();
    while let Some(row) = rows.next().await.unwrap() {
        seen.push((
            row.get::<String>(0).unwrap(),
            row.get::<String>(1).unwrap(),
            row.get::<i64>(2).unwrap(),
            row.get::<Option<i64>>(3).unwrap(),
        ));
    }
    assert_eq!(
        seen,
        vec![
            ("p1".to_string(), "provisional".to_string(), 0, None),
            ("p2".to_string(), "provisional".to_string(), 0, None),
        ],
        "citations are not evidence of evaluation, and nothing is a human review"
    );
}

/// Weakening: make the backfill an unconditional INSERT OR REPLACE, so a
/// re-run is "safe".
///
/// It would be the opposite. The backfill has to be re-runnable — that is how
/// it resumes — and a re-run that overwrites is a re-run that resets a real
/// evaluation to `provisional` and drops a human review. Only the gap gets
/// filled.
#[tokio::test]
async fn the_backfill_never_overwrites_an_existing_evaluation() {
    let (db, _temp) = db_with_substrate().await;
    let conn = db.conn.lock().await;
    {
        let tx = conn.transaction().await.unwrap();
        MemoryDB::backfill_page_truth_state(&tx).await.unwrap();
        tx.commit().await.unwrap();
    }
    conn.execute(
        "UPDATE page_truth_state
            SET support_status='supported', human_reviewed=1,
                reviewed_page_version=1, reviewed_page_digest='d'
          WHERE page_id='p1'",
        (),
    )
    .await
    .unwrap();

    let tx = conn.transaction().await.unwrap();
    let filled = MemoryDB::backfill_page_truth_state(&tx).await.unwrap();
    tx.commit().await.unwrap();
    assert_eq!(filled, 0, "no gap left to fill");

    let mut rows = conn
        .query(
            "SELECT support_status, human_reviewed FROM page_truth_state WHERE page_id='p1'",
            (),
        )
        .await
        .unwrap();
    let row = rows.next().await.unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "supported");
    assert_eq!(row.get::<i64>(1).unwrap(), 1, "a human review must survive");
}

/// Weakening: trust that the INSERT covered everything and report success.
///
/// Coverage is the migration's whole postcondition, so it is read back from
/// the data rather than inferred from the statement not erroring. A page added
/// between the INSERT and the check would fail here — correctly, since it has
/// no truth row.
#[tokio::test]
async fn the_backfill_checks_coverage_rather_than_assuming_it() {
    let (db, _temp) = db_with_substrate().await;
    db.insert_page(
        "p3",
        "p3",
        None,
        "",
        None,
        None,
        &[],
        "2026-07-27T00:00:00Z",
    )
    .await
    .unwrap();
    let conn = db.conn.lock().await;
    let tx = conn.transaction().await.unwrap();
    MemoryDB::backfill_page_truth_state(&tx).await.unwrap();
    tx.commit().await.unwrap();

    let mut rows = conn
        .query(
            "SELECT count(*) FROM pages p
              WHERE NOT EXISTS (SELECT 1 FROM page_truth_state t WHERE t.page_id = p.id)",
            (),
        )
        .await
        .unwrap();
    let uncovered: i64 = rows.next().await.unwrap().unwrap().get(0).unwrap();
    assert_eq!(uncovered, 0, "every page carries a truth row");
}

// ===== The human-root minter (artifact 3 §5, artifact 6 §2a) =====

const BASE_PROSE: &str = "Kestrels hunt by hovering.\nThey favour rough grassland.";

/// Seed a page that already carries prose, and return its exact base digest.
/// `insert_page` writes the matching immutable `page_history` row, so the base
/// the minter checks against is the real one rather than a hand-built fixture.
async fn page_with_prose(db: &MemoryDB, id: &str) -> String {
    db.insert_page(
        id,
        id,
        None,
        BASE_PROSE,
        None,
        None,
        &[],
        "2026-07-27T00:00:00Z",
    )
    .await
    .unwrap();
    crate::provenance::revision_content_digest(BASE_PROSE)
}

async fn root_kind_of(db: &MemoryDB, root_id: &str) -> String {
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT root_kind FROM provenance_roots WHERE root_id = ?1",
            libsql::params![root_id],
        )
        .await
        .unwrap();
    rows.next().await.unwrap().unwrap().get(0).unwrap()
}

async fn human_delta_root_count(db: &MemoryDB) -> i64 {
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT count(*) FROM provenance_roots WHERE root_kind = 'human_edit_delta'",
            (),
        )
        .await
        .unwrap();
    rows.next().await.unwrap().unwrap().get(0).unwrap()
}

/// The happy path, and the shape artifact 3 §4a's "human-delta destination"
/// requires: the delta is reachable as a `memory` whose span can be cited, and
/// its root carries the human kind that §5 demands of an attesting root.
///
/// Weakening guarded: minting the root without storing the delta, or storing it
/// under any other root kind.
#[tokio::test]
async fn an_exact_base_save_mints_a_human_root_and_a_span_addressable_memory() {
    let (db, _temp) = db_with_substrate().await;
    let digest = page_with_prose(&db, "hp1").await;

    let minted = db
        .mint_human_edit_delta(
            "hp1",
            1,
            &digest,
            &format!("{BASE_PROSE}\nThey nest in old crow nests."),
        )
        .await
        .unwrap()
        .expect("an exact-base save that adds prose mints a delta");

    assert_eq!(
        minted.delta_text, "They nest in old crow nests.",
        "the delta is the added prose, not the whole page"
    );
    assert_eq!(
        root_kind_of(&db, &minted.root_id).await,
        "human_edit_delta",
        "the root must carry the kind §5 accepts from an attesting root"
    );

    // Addressable the way an edge addresses a memory: by source_id, which is
    // what the rebuilt space fence resolves on.
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT content FROM memories WHERE source_id = ?1",
            libsql::params![minted.memory_source_id.clone()],
        )
        .await
        .unwrap();
    let stored: String = rows.next().await.unwrap().unwrap().get(0).unwrap();
    assert_eq!(
        stored, minted.delta_text,
        "the delta must be stored verbatim, or its spans address nothing"
    );
}

/// D4's stale row: "conflict, nothing written". Not a partial write, not a
/// warning — nothing.
///
/// Weakening guarded: comparing the base loosely (canonically, by version
/// alone, or not at all), which would attribute another writer's text to the
/// human as new prose.
#[tokio::test]
async fn a_stale_base_mints_nothing() {
    let (db, _temp) = db_with_substrate().await;
    page_with_prose(&db, "hp2").await;

    let error = db
        .mint_human_edit_delta(
            "hp2",
            1,
            &crate::provenance::revision_content_digest("some other text entirely"),
            &format!("{BASE_PROSE}\nThey nest in old crow nests."),
        )
        .await
        .expect_err("a stale base is a conflict");
    assert!(
        format!("{error}").contains("human_delta_base_stale"),
        "the conflict must name itself: {error}"
    );
    assert_eq!(
        human_delta_root_count(&db).await,
        0,
        "a stale base writes nothing at all"
    );
}

/// Whitespace tolerance is the specific loosening that must not creep in. A
/// reflow changes every line boundary, so a canonical comparison here would let
/// the reflowed text be attributed to the human.
#[tokio::test]
async fn a_whitespace_only_difference_is_still_a_stale_base() {
    let (db, _temp) = db_with_substrate().await;
    page_with_prose(&db, "hp3").await;

    // Word spacing, which canonicalization collapses. A blank-line change would
    // NOT work here: canonicalization preserves blank lines, so the precondition
    // below would be false and the test would prove nothing.
    let reflowed = BASE_PROSE.replace(' ', "  ");
    assert_eq!(
        crate::provenance::canonical_content_digest(&reflowed),
        crate::provenance::canonical_content_digest(BASE_PROSE),
        "precondition: canonical digests DO converge here, which is why the \
         exact digest has to be the one that binds"
    );

    let error = db
        .mint_human_edit_delta(
            "hp3",
            1,
            &crate::provenance::revision_content_digest(&reflowed),
            &format!("{BASE_PROSE}\nAdded."),
        )
        .await
        .expect_err("an inexact base is stale even when it canonicalizes the same");
    assert!(
        format!("{error}").contains("human_delta_base_stale"),
        "{error}"
    );
}

/// A base version that never existed cannot be verified, so it cannot mint.
#[tokio::test]
async fn an_unknown_base_version_mints_nothing() {
    let (db, _temp) = db_with_substrate().await;
    let digest = page_with_prose(&db, "hp4").await;

    let error = db
        .mint_human_edit_delta("hp4", 99, &digest, "anything")
        .await
        .expect_err("there is no version 99 to have been written against");
    assert!(
        format!("{error}").contains("human_delta_base_unknown"),
        "{error}"
    );
    assert_eq!(human_delta_root_count(&db).await, 0);
}

/// A save that adds no prose has no evidence to ground. That is an ordinary
/// edit, not a conflict — refusing it would make deleting a sentence fail.
#[tokio::test]
async fn a_save_that_adds_no_prose_mints_no_delta() {
    let (db, _temp) = db_with_substrate().await;
    let digest = page_with_prose(&db, "hp5").await;

    let minted = db
        .mint_human_edit_delta("hp5", 1, &digest, "Kestrels hunt by hovering.")
        .await
        .unwrap();
    assert!(minted.is_none(), "a deletion mints nothing");
    assert_eq!(human_delta_root_count(&db).await, 0);
}

/// The same prose saved twice is one piece of evidence, not two. Roots are
/// content-addressed and the memory ids derive from the root, so a retry
/// converges instead of duplicating the delta.
///
/// Weakening guarded: a random root or memory id, which would let one sentence
/// be counted repeatedly.
#[tokio::test]
async fn the_same_delta_twice_converges_on_one_root_and_one_memory() {
    let (db, _temp) = db_with_substrate().await;
    let digest = page_with_prose(&db, "hp6").await;
    let saved = format!("{BASE_PROSE}\nThey nest in old crow nests.");

    let first = db
        .mint_human_edit_delta("hp6", 1, &digest, &saved)
        .await
        .unwrap()
        .unwrap();
    let second = db
        .mint_human_edit_delta("hp6", 1, &digest, &saved)
        .await
        .unwrap()
        .unwrap();

    assert_eq!(first.root_id, second.root_id, "one root for one text");
    assert_eq!(first.memory_source_id, second.memory_source_id);
    assert_eq!(human_delta_root_count(&db).await, 1);

    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT count(*) FROM memories WHERE source_id = ?1",
            libsql::params![first.memory_source_id.clone()],
        )
        .await
        .unwrap();
    let copies: i64 = rows.next().await.unwrap().unwrap().get(0).unwrap();
    assert_eq!(copies, 1, "one memory for one delta");
}

/// Artifact 6 §2a: a human can never corroborate themselves. Two unrelated
/// deltas, on two different pages, must land in ONE independence group — one
/// person is one source, whatever they write and wherever they write it.
///
/// Weakening guarded: giving each delta its own group (per-save, per-page, or a
/// fresh uuid), which would let a user manufacture independent support for a
/// claim by restating it.
#[tokio::test]
async fn every_human_delta_shares_one_independence_group() {
    let (db, _temp) = db_with_substrate().await;
    let d1 = page_with_prose(&db, "hp7").await;
    let d2 = page_with_prose(&db, "hp8").await;

    let a = db
        .mint_human_edit_delta(
            "hp7",
            1,
            &d1,
            &format!("{BASE_PROSE}\nThey nest in old crow nests."),
        )
        .await
        .unwrap()
        .unwrap();
    let b = db
        .mint_human_edit_delta(
            "hp8",
            1,
            &d2,
            &format!("{BASE_PROSE}\nRainfall in March determines the vole population."),
        )
        .await
        .unwrap()
        .unwrap();
    assert_ne!(a.root_id, b.root_id, "precondition: two distinct roots");

    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT count(DISTINCT independence_group_id) FROM provenance_roots
              WHERE root_kind = 'human_edit_delta'",
            (),
        )
        .await
        .unwrap();
    let groups: i64 = rows.next().await.unwrap().unwrap().get(0).unwrap();
    assert_eq!(
        groups, 1,
        "one human is one source; distinct groups would inflate independent support"
    );
}

/// The delta memory must inherit the page's space. The rebuilt space fence
/// compares a support edge's claim-side space against its memory-side space, so
/// a delta filed anywhere else is unciteable — and unciteable in the worst way,
/// refused at edge-write time rather than here.
#[tokio::test]
async fn the_delta_memory_takes_the_page_space_so_the_fence_admits_it() {
    let (db, _temp) = db_with_substrate().await;
    db.insert_page(
        "hp9",
        "hp9",
        None,
        BASE_PROSE,
        None,
        Some("birds"),
        &[],
        "2026-07-27T00:00:00Z",
    )
    .await
    .unwrap();

    let minted = db
        .mint_human_edit_delta(
            "hp9",
            1,
            &crate::provenance::revision_content_digest(BASE_PROSE),
            &format!("{BASE_PROSE}\nThey nest in old crow nests."),
        )
        .await
        .unwrap()
        .unwrap();

    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT (SELECT space FROM pages WHERE id='hp9')
                  = (SELECT space FROM memories WHERE source_id = ?1)",
            libsql::params![minted.memory_source_id.clone()],
        )
        .await
        .unwrap();
    let same: i64 = rows.next().await.unwrap().unwrap().get(0).unwrap();
    assert_eq!(same, 1, "delta memory space must equal the page space");
}
