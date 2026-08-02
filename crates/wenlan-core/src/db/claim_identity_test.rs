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
    // Since migration 105 the enqueue trigger has already queued p1 at its
    // current version, so clear the queue first: this test is about the shape
    // of the unique index, not about who filled the table.
    conn.execute("DELETE FROM claim_derivation_jobs", ())
        .await
        .unwrap();
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

// ===== The supports writer (artifact 3 §4a) =====

const ADDED_PROSE: &str = "They nest in old crow nests.";
const CLAIM_TEXT: &str = "Kestrels nest in old crow nests.";

/// Everything §4a's two invariants need to agree about: a real human-delta
/// memory, a real claim revision, and a cache row that actually recorded the
/// verdict. Built through the production minter rather than by hand, so the
/// test cannot accidentally agree with itself.
struct SupportScenario {
    claim_revision_id: String,
    memory_source_id: String,
    /// The evidence's provenance root, resolved by the caller the way the
    /// derivation worker resolves it. For a human delta the minter already
    /// returns it; nothing has to parse it back out of the memory id.
    root_id: String,
    verdict: super::claim_identity::SupportVerdict,
}

async fn support_scenario(db: &MemoryDB, page_id: &str) -> SupportScenario {
    support_scenario_saving(db, page_id, &format!("{BASE_PROSE}\n{ADDED_PROSE}")).await
}

/// The same scenario, over an arbitrary saved body — so a test can control what
/// the delta memory actually contains (two copies of one sentence, say) rather
/// than only ever seeing the single-occurrence shape.
async fn support_scenario_saving(db: &MemoryDB, page_id: &str, saved: &str) -> SupportScenario {
    let base = page_with_prose(db, page_id).await;
    let minted = db
        .mint_human_edit_delta(page_id, 1, &base, saved)
        .await
        .unwrap()
        .expect("the scenario needs a real delta to cite");

    let span_digest = crate::provenance::revision_content_digest(ADDED_PROSE);
    let claim_digest = crate::provenance::revision_content_digest(CLAIM_TEXT);
    let claim_revision_id = format!("cr_{page_id}");
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "INSERT INTO claims (claim_id, page_id, created_at) VALUES (?1, ?2, 0)",
            libsql::params![format!("c_{page_id}"), page_id],
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO claim_revisions (claim_revision_id, claim_id, predecessor_revision_id,
                                          canonical_text, canonical_text_digest, claim_kind,
                                          extractor_version, created_at)
             VALUES (?1, ?2, '', ?3, ?4, 'observation', 1, 0)",
            libsql::params![
                claim_revision_id.clone(),
                format!("c_{page_id}"),
                CLAIM_TEXT,
                claim_digest.clone(),
            ],
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO entailment_cache (claim_text_digest, source_span_digest, model_id,
                                           model_version, prompt_version, score,
                                           threshold_at_write, backend, scored_at)
             VALUES (?1, ?2, 'qwen3-4b', 'v1', 'p1', 0.91, 0.7, 'on_device', 0)",
            libsql::params![claim_digest, span_digest.clone()],
        )
        .await
        .unwrap();
    }

    SupportScenario {
        claim_revision_id,
        memory_source_id: minted.memory_source_id,
        root_id: minted.root_id,
        verdict: super::claim_identity::SupportVerdict {
            chunk_index: 0,
            source_version: 1,
            span_start: 0,
            span_end: ADDED_PROSE.len() as i64,
            span_digest,
            model_id: "qwen3-4b".to_string(),
            model_version: "v1".to_string(),
            prompt_version: "p1".to_string(),
            score: 0.91,
            threshold_at_write: 0.7,
        },
    }
}

async fn edge_row(db: &MemoryDB, edge_id: &str) -> (i64, String, String, String) {
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT grounded, lineage, root_id, payload FROM edges WHERE edge_id = ?1",
            libsql::params![edge_id],
        )
        .await
        .unwrap();
    let row = rows.next().await.unwrap().expect("the edge must exist");
    (
        row.get(0).unwrap(),
        row.get(1).unwrap(),
        row.get(2).unwrap(),
        row.get(3).unwrap(),
    )
}

/// Weakening: write the edge without freezing the span/verdict payload.
///
/// An edge that names only a memory cannot prove later that the thing judged is
/// the thing cited — the whole reason §4a exists.
#[tokio::test]
async fn a_faithful_verdict_writes_an_edge_that_names_its_span_and_its_judge() {
    let (db, _temp) = db_with_substrate().await;
    let scenario = support_scenario(&db, "sp1").await;

    let edge_id = db
        .write_support_edge(
            &scenario.claim_revision_id,
            &scenario.memory_source_id,
            &scenario.root_id,
            &scenario.verdict,
        )
        .await
        .expect("a verdict that agrees with the bytes and the cache is the valid shape");

    let (grounded, lineage, root_id, payload) = edge_row(&db, &edge_id).await;
    assert_eq!(
        lineage, "evidence",
        "§2 fixes the lineage of a support edge"
    );
    assert_eq!(
        root_id,
        scenario.memory_source_id.strip_prefix("hed_").unwrap(),
        "§5: root_id is the provenance root of the evidence cited"
    );
    assert_eq!(
        grounded, 0,
        "§3: a support edge may never manufacture grounding"
    );

    let frozen: serde_json::Value = serde_json::from_str(&payload).unwrap();
    for field in [
        "source_version",
        "span_start",
        "span_end",
        "span_digest",
        "model_id",
        "model_version",
        "prompt_version",
        "score",
        "threshold_at_write",
    ] {
        assert!(
            !frozen[field].is_null(),
            "§4a's table requires {field} on the edge itself"
        );
    }
}

/// Weakening: trust the stored offsets instead of re-reading the bytes.
///
/// Artifact 1 §3: offsets alone are never trusted. A span whose digest no
/// longer matches must invalidate the edge, not silently re-point at whatever
/// text now occupies those offsets.
#[tokio::test]
async fn a_span_that_moved_invalidates_the_support_edge() {
    let (db, _temp) = db_with_substrate().await;
    let scenario = support_scenario(&db, "sp2").await;

    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE memories SET content = ?1 WHERE source_id = ?2",
            libsql::params![
                "They nest on bare cliff ledges.",
                scenario.memory_source_id.clone()
            ],
        )
        .await
        .unwrap();
    }

    let refused = db
        .write_support_edge(
            &scenario.claim_revision_id,
            &scenario.memory_source_id,
            &scenario.root_id,
            &scenario.verdict,
        )
        .await;
    assert!(
        refused.is_err(),
        "the offsets still resolve, but to text this verdict never judged"
    );
}

/// Weakening: write the edge without checking the cache at all.
///
/// §4a invariant 2 — a support edge may never be written from a verdict it
/// cannot name.
#[tokio::test]
async fn a_verdict_the_cache_never_recorded_cannot_be_cited() {
    let (db, _temp) = db_with_substrate().await;
    let scenario = support_scenario(&db, "sp3").await;

    {
        let conn = db.conn.lock().await;
        conn.execute("DELETE FROM entailment_cache", ())
            .await
            .unwrap();
    }

    assert!(
        db.write_support_edge(
            &scenario.claim_revision_id,
            &scenario.memory_source_id,
            &scenario.root_id,
            &scenario.verdict,
        )
        .await
        .is_err(),
        "with no cache row there is no verdict to name"
    );
}

/// Weakening: check that a cache row exists, but not that it says what the
/// edge claims. A score the cache does not record is the exact shape of a
/// verdict laundered from a different judgment.
#[tokio::test]
async fn a_score_the_cache_does_not_record_is_refused() {
    let (db, _temp) = db_with_substrate().await;
    let mut scenario = support_scenario(&db, "sp4").await;
    scenario.verdict.score = 0.99;

    assert!(
        db.write_support_edge(
            &scenario.claim_revision_id,
            &scenario.memory_source_id,
            &scenario.root_id,
            &scenario.verdict,
        )
        .await
        .is_err(),
        "the cache recorded 0.91; an edge may not claim it cleared a different bar"
    );
}

/// Weakening: resolve a missing provenance root to some default rather than
/// refusing. §5 requires root_id on every support edge, and inventing one
/// would attribute evidence to a root that never produced it.
///
/// The weakening is unchanged; only the mechanism that guards it moved. PR-A
/// enforced this by refusing any memory id without an `hed_` prefix, which
/// guarded the rule by excluding almost all evidence. The root is now named by
/// the caller, so the same weakening is guarded by checking that the named root
/// exists at all.
#[tokio::test]
async fn evidence_with_no_provenance_root_is_refused_by_name() {
    let (db, _temp) = db_with_substrate().await;
    let scenario = support_scenario(&db, "sp5").await;

    let error = db
        .write_support_edge(
            &scenario.claim_revision_id,
            &scenario.memory_source_id,
            "no-such-root",
            &scenario.verdict,
        )
        .await
        .expect_err("a root that does not exist cannot back a claim");
    assert!(
        format!("{error}").contains("support_root_unknown"),
        "the refusal must name the gap rather than surfacing as a foreign-key error: {error}"
    );
}

/// Weakening: discriminate the edge id by memory alone. Two spans of one
/// memory would then collapse onto one edge, and the second verdict would be
/// silently dropped by ON CONFLICT DO NOTHING.
#[tokio::test]
async fn two_spans_of_one_memory_are_two_distinct_edges() {
    let (db, _temp) = db_with_substrate().await;
    let scenario = support_scenario(&db, "sp6").await;

    let first = db
        .write_support_edge(
            &scenario.claim_revision_id,
            &scenario.memory_source_id,
            &scenario.root_id,
            &scenario.verdict,
        )
        .await
        .unwrap();

    // A second, shorter span of the same memory, with its own cache row.
    // char-safe rather than `&ADDED_PROSE[..12]`: ASCII today, but the repo's
    // UTF-8 rule is a rule about the habit, not about this literal.
    let short: String = ADDED_PROSE.chars().take(12).collect();
    let short_digest = crate::provenance::revision_content_digest(&short);
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "INSERT INTO entailment_cache (claim_text_digest, source_span_digest, model_id,
                                           model_version, prompt_version, score,
                                           threshold_at_write, backend, scored_at)
             VALUES (?1, ?2, 'qwen3-4b', 'v1', 'p1', 0.91, 0.7, 'on_device', 0)",
            libsql::params![
                crate::provenance::revision_content_digest(CLAIM_TEXT),
                short_digest.clone(),
            ],
        )
        .await
        .unwrap();
    }
    let second = db
        .write_support_edge(
            &scenario.claim_revision_id,
            &scenario.memory_source_id,
            &scenario.root_id,
            &super::claim_identity::SupportVerdict {
                span_end: short.len() as i64,
                span_digest: short_digest,
                ..scenario.verdict.clone()
            },
        )
        .await
        .unwrap();

    assert_ne!(
        first, second,
        "the span digest must discriminate, or one verdict silently replaces the other"
    );
}

/// Weakening: keep `ON CONFLICT(edge_id) DO NOTHING` as the whole conflict
/// story, and let a re-judged span return `Ok` while the stored edge still
/// names the verdict it replaced.
///
/// Review found this one. `compute_edge_id` deliberately does NOT hash the
/// judge — one span of one memory is one support edge, not one per model that
/// ever looked at it — which is right, and which is exactly what makes a bare
/// `DO NOTHING` a silent discard. `entailment_cache`'s five-part key admits a
/// second row for the same span under a new `model_version`, so both §4a
/// invariants pass on the second write: the live bytes still match the span
/// digest, and the cache genuinely records that verdict. The edge id is
/// byte-identical, the insert does nothing, and the caller is told it
/// succeeded. Three records disagreeing with nothing forcing them to agree is
/// the drift §4a exists to close, so the conflict is read and refused.
#[tokio::test]
async fn a_second_verdict_for_one_span_is_refused_rather_than_silently_dropped() {
    let (db, _temp) = db_with_substrate().await;
    let scenario = support_scenario(&db, "sp7").await;
    let first = db
        .write_support_edge(
            &scenario.claim_revision_id,
            &scenario.memory_source_id,
            &scenario.root_id,
            &scenario.verdict,
        )
        .await
        .unwrap();

    // Re-judged: same claim, same span, a newer model and a lower score. The
    // cache records it honestly, so nothing upstream objects.
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "INSERT INTO entailment_cache (claim_text_digest, source_span_digest, model_id,
                                           model_version, prompt_version, score,
                                           threshold_at_write, backend, scored_at)
             VALUES (?1, ?2, 'qwen3-4b', 'v2', 'p1', 0.83, 0.7, 'on_device', 0)",
            libsql::params![
                crate::provenance::revision_content_digest(CLAIM_TEXT),
                scenario.verdict.span_digest.clone(),
            ],
        )
        .await
        .unwrap();
    }

    let error = db
        .write_support_edge(
            &scenario.claim_revision_id,
            &scenario.memory_source_id,
            &scenario.root_id,
            &super::claim_identity::SupportVerdict {
                model_version: "v2".to_string(),
                score: 0.83,
                ..scenario.verdict.clone()
            },
        )
        .await
        .expect_err("a changed verdict must refuse rather than return Ok");
    assert!(
        format!("{error}").contains("support_verdict_supersedes_existing"),
        "the refusal must name the superseding verdict: {error}"
    );

    // And the stored edge is untouched -- still the verdict that was actually
    // written, not a half-applied blend of the two.
    let payload = edge_row(&db, &first).await.3;
    assert!(
        payload.contains("\"model_version\":\"v1\"") && payload.contains("0.91"),
        "the original verdict must survive the refused write: {payload}"
    );
}

/// Weakening: check only that the three records AGREE, never that what they
/// agree on means "supported".
///
/// Review found this one too, and it is the sharpest of them: every check in
/// `write_support_edge` is an equality against the cache, and an honestly
/// recorded FAILURE satisfies all of them. A verdict that scored 0.55 against
/// its own 0.7 bar is the entailment judge saying no — and it would have
/// written a `supports` edge, which is the shape M5 reads as truth. Comparing
/// against the cache cannot catch it, because the cache faithfully recorded
/// the failure.
#[tokio::test]
async fn a_verdict_that_failed_its_own_threshold_is_not_support() {
    let (db, _temp) = db_with_substrate().await;
    let scenario = support_scenario(&db, "sp8").await;

    {
        let conn = db.conn.lock().await;
        conn.execute(
            "INSERT INTO entailment_cache (claim_text_digest, source_span_digest, model_id,
                                           model_version, prompt_version, score,
                                           threshold_at_write, backend, scored_at)
             VALUES (?1, ?2, 'qwen3-4b', 'v3', 'p1', 0.55, 0.7, 'on_device', 0)",
            libsql::params![
                crate::provenance::revision_content_digest(CLAIM_TEXT),
                scenario.verdict.span_digest.clone(),
            ],
        )
        .await
        .unwrap();
    }

    let error = db
        .write_support_edge(
            &scenario.claim_revision_id,
            &scenario.memory_source_id,
            &scenario.root_id,
            &super::claim_identity::SupportVerdict {
                model_version: "v3".to_string(),
                score: 0.55,
                threshold_at_write: 0.7,
                ..scenario.verdict.clone()
            },
        )
        .await
        .expect_err("a verdict below its own bar is not support");
    assert!(
        format!("{error}").contains("support_verdict_below_threshold"),
        "the refusal must name the threshold: {error}"
    );
}

/// Weakening: verify the base against `page_history` alone, and let a save
/// written against a superseded version mint as if it were current.
///
/// Cross-model review found this one. `page_history` is immutable, so every
/// version that ever existed answers the digest question correctly forever —
/// which means the digest check proves only "version N really said this", never
/// "N is the version you were editing". A human who opened the page at version
/// 1, went to lunch while a distill cycle rewrote it to version 2, and then
/// saved would mint a delta whose prose was computed against text no longer on
/// the page, silently discarding the intervening edit. D4's table calls that
/// row "stale base -> Err(Conflict), nothing written"; it was the one row the
/// code did not implement.
#[tokio::test]
async fn a_save_against_a_superseded_version_is_refused() {
    let (db, _temp) = db_with_substrate().await;
    let base = page_with_prose(&db, "hp11").await;

    // Someone else edits the page through the ordinary path, which bumps
    // `pages.version` and appends the matching history row.
    db.update_page_content(
        "hp11",
        &format!("{BASE_PROSE}\nThey also take beetles."),
        &[],
        "test",
    )
    .await
    .unwrap();

    let error = db
        .mint_human_edit_delta(
            "hp11",
            1,
            &base,
            &format!("{BASE_PROSE}\nThey nest in old crow nests."),
        )
        .await
        .expect_err("version 1 is no longer the text this page carries");
    assert!(
        format!("{error}").contains("human_delta_base_stale"),
        "the refusal must name the stale base: {error}"
    );
    assert_eq!(
        human_delta_root_count(&db).await,
        0,
        "a refused save mints nothing"
    );
}

/// Weakening: copy the caller's `source_version` into the payload without ever
/// checking it against the memory.
///
/// Cross-model review found this one. The digest check proves the BYTES are the
/// judged ones; it says nothing about the version number they are filed under,
/// because a span can survive an edit elsewhere in the memory unchanged. The
/// edge then carries a provenance record that is false in the single field a
/// later reader would use to fetch the source text back — and false
/// undetectably, since the digest agrees.
#[tokio::test]
async fn a_verdict_naming_a_version_the_memory_has_moved_past_is_refused() {
    let (db, _temp) = db_with_substrate().await;
    let scenario = support_scenario(&db, "sp9").await;

    // The memory moves to version 2 with its judged span byte-identical: an
    // edit elsewhere in the same row. Every other check still passes.
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE memories SET version = 2 WHERE source_id = ?1",
            libsql::params![scenario.memory_source_id.clone()],
        )
        .await
        .unwrap();
    }

    let error = db
        .write_support_edge(
            &scenario.claim_revision_id,
            &scenario.memory_source_id,
            &scenario.root_id,
            &scenario.verdict,
        )
        .await
        .expect_err("a verdict may not name a version the memory has moved past");
    assert!(
        format!("{error}").contains("support_source_version_stale"),
        "the refusal must name the version disagreement: {error}"
    );
}

/// Weakening: discriminate the edge id by span DIGEST alone, dropping the
/// offsets.
///
/// Cross-model review found this one in a fix rather than in the original
/// code — the conflict refusal above turns a digest-only id from a silent drop
/// into a LOUD WRONG ANSWER. A memory may say the same sentence twice, and the
/// two occurrences share a digest; under a digest-only discriminator the second
/// citation is rejected as "superseding" the first, which misdescribes it
/// completely. Nothing is superseding anything: they are two different places
/// in the text, judged identically, and both are true.
#[tokio::test]
async fn the_same_sentence_twice_is_two_citations_not_a_superseding_verdict() {
    let (db, _temp) = db_with_substrate().await;
    let scenario = support_scenario_saving(
        &db,
        "sp10",
        &format!("{BASE_PROSE}\n{ADDED_PROSE}\n{ADDED_PROSE}"),
    )
    .await;

    // Precondition: the delta really does carry the sentence twice, so the two
    // spans below are genuinely distinct places with one shared digest.
    let doubled = format!("{ADDED_PROSE}\n{ADDED_PROSE}");
    {
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT content FROM memories WHERE source_id = ?1",
                libsql::params![scenario.memory_source_id.clone()],
            )
            .await
            .unwrap();
        let content: String = rows.next().await.unwrap().unwrap().get(0).unwrap();
        assert_eq!(content, doubled, "precondition: two identical occurrences");
    }

    let second_start = (ADDED_PROSE.len() + 1) as i64;
    let first = db
        .write_support_edge(
            &scenario.claim_revision_id,
            &scenario.memory_source_id,
            &scenario.root_id,
            &scenario.verdict,
        )
        .await
        .expect("the first occurrence is an ordinary citation");
    let second = db
        .write_support_edge(
            &scenario.claim_revision_id,
            &scenario.memory_source_id,
            &scenario.root_id,
            &super::claim_identity::SupportVerdict {
                span_start: second_start,
                span_end: second_start + ADDED_PROSE.len() as i64,
                ..scenario.verdict.clone()
            },
        )
        .await
        .expect("a second occurrence is a second citation, not a superseding verdict");

    assert_ne!(
        first, second,
        "the LOCATION must discriminate, or one true citation is refused as the other's replacement"
    );
}

/// Weakening: `INSERT OR IGNORE` the delta memory and report success without
/// looking at the row that already occupies its id.
///
/// Cross-model review found this one. A provenance root is content-addressed
/// over the prose ALONE, and the memory id derives from the root, so the same
/// sentence added to a page in space A and a page in space B resolves to ONE
/// memory — filed in whichever space got there first. The minter returns Ok
/// either way, and the truth only emerges much later, when the space fence
/// refuses a support edge nobody can trace back to this call.
///
/// The refusal moves the failure to where the cause is. Whether one sentence
/// written twice in two spaces should be one piece of evidence or two is a
/// separate, deferred question — see the follow-ups doc.
#[tokio::test]
async fn the_same_prose_in_a_second_space_is_refused_rather_than_aliased() {
    let (db, _temp) = db_with_substrate().await;
    for (id, space) in [("hp12", "birds"), ("hp13", "falconry")] {
        db.insert_page(
            id,
            id,
            None,
            BASE_PROSE,
            None,
            Some(space),
            &[],
            "2026-07-27T00:00:00Z",
        )
        .await
        .unwrap();
    }
    let base = crate::provenance::revision_content_digest(BASE_PROSE);
    let saved = format!("{BASE_PROSE}\n{ADDED_PROSE}");

    let first = db
        .mint_human_edit_delta("hp12", 1, &base, &saved)
        .await
        .unwrap()
        .expect("the first space mints normally");

    let error = db
        .mint_human_edit_delta("hp13", 1, &base, &saved)
        .await
        .expect_err("the identical line in another space may not silently alias the first");
    assert!(
        format!("{error}").contains("human_delta_space_conflict"),
        "the refusal must name the space conflict: {error}"
    );

    // And the first space's evidence is untouched -- the refusal does not
    // retro-file the existing memory into the second space.
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT space FROM memories WHERE source_id = ?1",
            libsql::params![first.memory_source_id.clone()],
        )
        .await
        .unwrap();
    let space: String = rows.next().await.unwrap().unwrap().get(0).unwrap();
    assert_eq!(space, "birds", "the refused save must change nothing");
}

const EVIDENCE_PROSE: &str = "Kestrels nest in old crow nests, usually reusing them.";

/// Seed an ORDINARY evidence memory — one that arrived through ingest rather
/// than through a human page edit — in `space`, and mint the content-addressed
/// provenance root that identifies it. Returns `(memory_source_id, root_id)`.
///
/// This is the shape the derivation worker actually meets. `mint_human_edit_delta`
/// produces the other shape, and PR-A could only resolve that one.
async fn ordinary_evidence(db: &MemoryDB, source_id: &str, space: &str) -> (String, String) {
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "INSERT INTO memories (id, content, source, source_id, title, chunk_index,
                                   last_modified, chunk_type, space)
             VALUES (?1, ?2, 'memory', ?3, 'evidence', 0, 0, 'text', ?4)",
            libsql::params![format!("m_{source_id}"), EVIDENCE_PROSE, source_id, space],
        )
        .await
        .unwrap();
    }
    let root_id = db
        .acquire_provenance_root(
            "document_ingest",
            EVIDENCE_PROSE,
            &crate::provenance::IndependenceSignals {
                source_identity: Some("file:///evidence.txt"),
                agent_turn: None,
                import_batch: None,
            },
        )
        .await
        .expect("ordinary evidence must be able to acquire its own root");
    (source_id.to_string(), root_id)
}

/// Seed a page, a claim revision on it, and the cache row recording the verdict
/// that judged `EVIDENCE_PROSE` against that claim. Returns the verdict.
async fn ordinary_scenario(
    db: &MemoryDB,
    page_id: &str,
    space: &str,
) -> super::claim_identity::SupportVerdict {
    db.insert_page(
        page_id,
        page_id,
        None,
        BASE_PROSE,
        None,
        Some(space),
        &[],
        "2026-07-27T00:00:00Z",
    )
    .await
    .unwrap();
    let claim_digest = crate::provenance::revision_content_digest(CLAIM_TEXT);
    let span_digest = crate::provenance::revision_content_digest(EVIDENCE_PROSE);
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "INSERT INTO claims (claim_id, page_id, created_at) VALUES (?1, ?2, 0)",
            libsql::params![format!("c_{page_id}"), page_id],
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO claim_revisions (claim_revision_id, claim_id, predecessor_revision_id,
                                          canonical_text, canonical_text_digest, claim_kind,
                                          extractor_version, created_at)
             VALUES (?1, ?2, '', ?3, ?4, 'observation', 1, 0)",
            libsql::params![
                format!("cr_{page_id}"),
                format!("c_{page_id}"),
                CLAIM_TEXT,
                claim_digest.clone(),
            ],
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO entailment_cache (claim_text_digest, source_span_digest, model_id,
                                           model_version, prompt_version, score,
                                           threshold_at_write, backend, scored_at)
             VALUES (?1, ?2, 'qwen3-4b', 'v1', 'p1', 0.91, 0.7, 'on_device', 0)",
            libsql::params![claim_digest, span_digest.clone()],
        )
        .await
        .unwrap();
    }
    super::claim_identity::SupportVerdict {
        chunk_index: 0,
        source_version: 1,
        span_start: 0,
        span_end: EVIDENCE_PROSE.len() as i64,
        span_digest,
        model_id: "qwen3-4b".to_string(),
        model_version: "v1".to_string(),
        prompt_version: "p1".to_string(),
        score: 0.91,
        threshold_at_write: 0.7,
    }
}

/// Weakening: refuse evidence that did not arrive through a human page edit.
///
/// This is the M5 blocker itself. PR-A resolved the evidence root by parsing an
/// `hed_` prefix off the memory id, so the only citable evidence was prose the
/// human typed into that page. A distilled page — synthesized from ingested
/// memories — could never cite anything, which is why `support_status` matched
/// zero rows on every install and why the cutover was a no-op by construction.
///
/// The root is now resolved by the caller and VERIFIED against the evidence's
/// own bytes, so the binding is checked rather than trusted to a naming
/// convention.
#[tokio::test]
async fn ordinary_ingested_evidence_can_back_a_claim() {
    let (db, _temp) = db_with_substrate().await;
    let verdict = ordinary_scenario(&db, "og1", "birds").await;
    let (memory_source_id, root_id) = ordinary_evidence(&db, "mem_og1", "birds").await;

    let edge_id = db
        .write_support_edge("cr_og1", &memory_source_id, &root_id, &verdict)
        .await
        .expect("ingested evidence with a resolved root is citable");

    let (grounded, lineage, stored_root, _) = edge_row(&db, &edge_id).await;
    assert_eq!(
        lineage, "evidence",
        "§2 fixes the lineage of a support edge"
    );
    assert_eq!(
        stored_root, root_id,
        "§5: the edge carries the provenance root of the evidence it cites"
    );
    assert_eq!(
        grounded, 0,
        "§3: a support edge may never manufacture grounding"
    );
}

/// Weakening: accept whatever root the caller names.
///
/// Dropping the `hed_` prefix removes a (narrow) structural guarantee, so
/// something has to replace it or the caller could hand over any root at all and
/// the edge would claim evidence it does not have. The replacement is stronger
/// than what it replaces: the root's stored `identity_digest` must equal the
/// digest recomputed from the cited memory's OWN bytes, so the binding is
/// checked rather than asserted.
#[tokio::test]
async fn a_root_minted_over_other_content_is_not_this_evidences_root() {
    let (db, _temp) = db_with_substrate().await;
    let verdict = ordinary_scenario(&db, "og2", "birds").await;
    let (memory_source_id, _) = ordinary_evidence(&db, "mem_og2", "birds").await;

    // A real, active root — but minted over different prose entirely.
    let foreign_root = db
        .acquire_provenance_root(
            "document_ingest",
            "Peregrines stoop at over two hundred miles an hour.",
            &crate::provenance::IndependenceSignals {
                source_identity: Some("file:///unrelated.txt"),
                agent_turn: None,
                import_batch: None,
            },
        )
        .await
        .unwrap();

    let error = db
        .write_support_edge("cr_og2", &memory_source_id, &foreign_root, &verdict)
        .await
        .expect_err("a root minted over other content does not identify this evidence");
    assert!(
        format!("{error}").contains("support_root_not_evidence_root"),
        "the refusal must name the broken binding: {error}"
    );
}

/// Weakening: let a claim cite evidence filed in another space.
///
/// `the_same_prose_in_a_second_space_is_refused_rather_than_aliased` states in
/// prose that the truth "emerges much later, when the space fence refuses a
/// support edge" — but nothing anywhere asserted that it does. This is the
/// constraint that decides which memories a page may cite at all, so the
/// derivation worker's candidate query must encode it: a worker that gathers
/// evidence across spaces would spend judge budget on candidates the fence
/// rejects at the final step, and read the resulting zero as "no support found"
/// rather than "the query was wrong".
///
/// The fence resolves a `claim_revision` source through its claim to the owning
/// page's space (`edges_rebuild.rs` `FENCE_BODY`), while `write_support_edge`
/// stamps the edge with the *evidence memory's* space. Those two agreeing is
/// exactly the same-space condition; a mismatch aborts.
#[tokio::test]
async fn a_support_edge_may_not_cite_evidence_from_another_space() {
    let (db, _temp) = db_with_substrate().await;
    db.insert_page(
        "xs1",
        "xs1",
        None,
        BASE_PROSE,
        None,
        Some("birds"),
        &[],
        "2026-07-27T00:00:00Z",
    )
    .await
    .unwrap();
    let base = crate::provenance::revision_content_digest(BASE_PROSE);
    let minted = db
        .mint_human_edit_delta("xs1", 1, &base, &format!("{BASE_PROSE}\n{ADDED_PROSE}"))
        .await
        .unwrap()
        .expect("the scenario needs a real delta to cite");

    let span_digest = crate::provenance::revision_content_digest(ADDED_PROSE);
    let claim_digest = crate::provenance::revision_content_digest(CLAIM_TEXT);
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "INSERT INTO claims (claim_id, page_id, created_at) VALUES ('c_xs1', 'xs1', 0)",
            (),
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO claim_revisions (claim_revision_id, claim_id, predecessor_revision_id,
                                          canonical_text, canonical_text_digest, claim_kind,
                                          extractor_version, created_at)
             VALUES ('cr_xs1', 'c_xs1', '', ?1, ?2, 'observation', 1, 0)",
            libsql::params![CLAIM_TEXT, claim_digest.clone()],
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO entailment_cache (claim_text_digest, source_span_digest, model_id,
                                           model_version, prompt_version, score,
                                           threshold_at_write, backend, scored_at)
             VALUES (?1, ?2, 'qwen3-4b', 'v1', 'p1', 0.91, 0.7, 'on_device', 0)",
            libsql::params![claim_digest, span_digest.clone()],
        )
        .await
        .unwrap();
        // The evidence moves out from under the page. Nothing else about the
        // verdict changes: same bytes, same cached score, same span.
        conn.execute(
            "UPDATE memories SET space = 'falconry' WHERE source_id = ?1",
            libsql::params![minted.memory_source_id.clone()],
        )
        .await
        .unwrap();
    }

    let verdict = super::claim_identity::SupportVerdict {
        chunk_index: 0,
        source_version: 1,
        span_start: 0,
        span_end: ADDED_PROSE.len() as i64,
        span_digest,
        model_id: "qwen3-4b".to_string(),
        model_version: "v1".to_string(),
        prompt_version: "p1".to_string(),
        score: 0.91,
        threshold_at_write: 0.7,
    };
    let error = db
        .write_support_edge(
            "cr_xs1",
            &minted.memory_source_id,
            &minted.root_id,
            &verdict,
        )
        .await
        .expect_err("evidence in another space may not back this page's claim");
    assert!(
        format!("{error}").contains("edges_space_fence"),
        "the refusal must be the space fence, not an incidental failure: {error}"
    );

    // And nothing was left behind: a fence abort must not leave a half-written
    // support edge that a later read would count as evidence.
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT COUNT(*) FROM edges WHERE edge_type = 'supports' AND src_id = 'cr_xs1'",
            (),
        )
        .await
        .unwrap();
    let count: i64 = rows.next().await.unwrap().unwrap().get(0).unwrap();
    assert_eq!(count, 0, "a fenced write leaves no edge behind");
}

/// Weakening: resolve evidence by document instead of by chunk.
///
/// `source_id` names a DOCUMENT, and one document is several `memories` rows
/// sharing that id — `store_raw_import_memories_batch` writes exactly that
/// shape. A lookup on `source_id` alone therefore resolves to whichever chunk
/// the query happens to return, and BOTH §4a checks then run against text the
/// verdict never read: the root digest is recomputed over the wrong bytes, and
/// the span offsets index into the wrong string.
///
/// The failure is fail-closed — valid support becomes unwritable rather than
/// wrong — which is the same dead-substrate reading this rung exists to end, one
/// layer down: the worker sees zero support edges and reports "no evidence
/// found" when the truth is "the write was refused". The edge also could not
/// have said which chunk was ever verified.
#[tokio::test]
async fn support_resolves_the_chunk_the_verdict_names() {
    let (db, _temp) = db_with_substrate().await;
    let mut scenario = support_scenario(&db, "sp_chunked").await;
    {
        let conn = db.conn.lock().await;
        // The real evidence becomes chunk 1; a decoy paragraph of the same
        // document takes chunk 0 — the row a document-wide lookup reaches first.
        conn.execute(
            "UPDATE memories SET chunk_index = 1 WHERE source_id = ?1",
            libsql::params![scenario.memory_source_id.clone()],
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO memories (id, content, source, source_id, title, chunk_index,
                                   last_modified, chunk_type, space)
             SELECT 'decoy_' || id, 'A different paragraph of the same document.',
                    source, source_id, title, 0, last_modified, chunk_type, space
               FROM memories WHERE source_id = ?1 AND chunk_index = 1",
            libsql::params![scenario.memory_source_id.clone()],
        )
        .await
        .unwrap();
    }
    scenario.verdict.chunk_index = 1;

    db.write_support_edge(
        &scenario.claim_revision_id,
        &scenario.memory_source_id,
        &scenario.root_id,
        &scenario.verdict,
    )
    .await
    .expect("the chunk the verdict names is the chunk that must be verified");
}

/// And when the pair still cannot resolve to one row, refuse by name rather than
/// pick. Nothing in the schema makes `(source_id, chunk_index)` unique — only a
/// plain index on `source_id` exists — so the pair is a convention the rest of
/// the tree keeps, not a guarantee this code may lean on. Choosing between two
/// candidates would be the same silent arbitrary resolution the chunk index was
/// added to remove.
#[tokio::test]
async fn support_refuses_evidence_that_resolves_to_two_rows() {
    let (db, _temp) = db_with_substrate().await;
    let scenario = support_scenario(&db, "sp_ambiguous").await;
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "INSERT INTO memories (id, content, source, source_id, title, chunk_index,
                                   last_modified, chunk_type, space)
             SELECT 'twin_' || id, content, source, source_id, title, chunk_index,
                    last_modified, chunk_type, space
               FROM memories WHERE source_id = ?1",
            libsql::params![scenario.memory_source_id.clone()],
        )
        .await
        .unwrap();
    }

    let error = db
        .write_support_edge(
            &scenario.claim_revision_id,
            &scenario.memory_source_id,
            &scenario.root_id,
            &scenario.verdict,
        )
        .await
        .expect_err("an unresolvable chunk is not something a verdict may cite");
    assert!(
        error.to_string().contains("support_evidence_ambiguous"),
        "the refusal has to name the ambiguity: {error}"
    );
}
