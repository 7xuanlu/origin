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
