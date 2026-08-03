// SPDX-License-Identifier: Apache-2.0
//! Teeth for the M5 claim-derivation queue. Each test names the weakening it
//! guards; deleting the rule in the doc comment must turn the test RED.

use super::claim_derivation::{
    CandidateLocator, DerivationJob, DerivationWrite, PromotionTurn, SupportOutcome,
    EXTRACTOR_VERSION, M5_PRODUCTION_JUDGE_MODEL_ID, M5_PRODUCTION_JUDGE_MODEL_VERSION,
    M5_PRODUCTION_JUDGE_THRESHOLD, MAX_ATTEMPTS, SUPPORT_THRESHOLD,
};
use super::tests::test_db;
use super::MemoryDB;
use crate::llm_provider::{LlmBackend, LlmError, LlmProvider, LlmRequest};
use async_trait::async_trait;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

/// A fully migrated database with NO pages. `test_db` runs the real migration
/// chain, so migration 105 has already installed the triggers here — re-running
/// the installer is itself the idempotence check a resumed migration needs.
async fn db_with_queue() -> (MemoryDB, tempfile::TempDir) {
    let (db, temp) = test_db().await;
    {
        let conn = db.conn.lock().await;
        let tx = conn.transaction().await.unwrap();
        MemoryDB::ensure_claim_identity_tables(&tx).await.unwrap();
        MemoryDB::ensure_judge_eligibility_tables(&tx)
            .await
            .unwrap();
        MemoryDB::ensure_claim_derivation_triggers(&tx)
            .await
            .unwrap();
        tx.commit().await.unwrap();
    }
    set_test_judge_eligibility(&db, "active", SUPPORT_THRESHOLD).await;
    (db, temp)
}

/// The state of a real vault the moment before migration 105: the substrate
/// tables exist (they shipped in 98) but nothing enqueues. Reached by dropping
/// the triggers back off a migrated database, which is the only way to write a
/// page that the triggers never saw.
async fn db_before_the_worker() -> (MemoryDB, tempfile::TempDir) {
    let (db, temp) = test_db().await;
    {
        let conn = db.conn.lock().await;
        conn.execute_batch(
            "DROP TRIGGER IF EXISTS m5_page_insert_enqueues_derivation;
             DROP TRIGGER IF EXISTS m5_page_update_enqueues_derivation;",
        )
        .await
        .unwrap();
        conn.execute("DELETE FROM claim_derivation_jobs", ())
            .await
            .unwrap();
    }
    set_test_judge_eligibility(&db, "active", SUPPORT_THRESHOLD).await;
    (db, temp)
}

/// Change the test judge's policy-independent eligibility row and advance the
/// one global generation in the same transaction. This is the storage shape a
/// later policy-owning transition API must preserve.
async fn set_test_judge_eligibility(db: &MemoryDB, state: &str, threshold: f64) -> i64 {
    let conn = db.conn.lock().await;
    let tx = conn
        .transaction_with_behavior(libsql::TransactionBehavior::Immediate)
        .await
        .unwrap();
    let generation = {
        let mut rows = tx
            .query(
                "UPDATE claim_judge_eligibility_generation
                    SET last_generation = last_generation + 1
                  WHERE id = 1
                  RETURNING last_generation",
                (),
            )
            .await
            .unwrap();
        rows.next()
            .await
            .unwrap()
            .expect("the eligibility generation singleton must exist")
            .get::<i64>(0)
            .unwrap()
    };
    tx.execute(
        "INSERT INTO claim_judge_eligibility
             (model_id, model_version, prompt_version, state, threshold, generation)
         VALUES ('judge', 'v1', 'p1', ?1, ?2, ?3)
         ON CONFLICT(model_id, model_version, prompt_version) DO UPDATE SET
             state = excluded.state,
             threshold = excluded.threshold,
             generation = excluded.generation",
        libsql::params![state, threshold, generation],
    )
    .await
    .unwrap();
    tx.commit().await.unwrap();
    generation
}

async fn test_eligibility_generation(db: &MemoryDB) -> i64 {
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT last_generation FROM claim_judge_eligibility_generation WHERE id = 1",
            (),
        )
        .await
        .unwrap();
    rows.next().await.unwrap().unwrap().get(0).unwrap()
}

async fn eligibility_schema_fingerprint(conn: &libsql::Connection) -> (Vec<String>, i64, i64, i64) {
    let ddl = {
        let mut rows = conn
            .query(
                "SELECT sql FROM sqlite_schema
                  WHERE type = 'table'
                    AND name IN ('claim_judge_eligibility',
                                 'claim_judge_eligibility_generation')
                  ORDER BY name",
                (),
            )
            .await
            .unwrap();
        let mut ddl = Vec::new();
        while let Some(row) = rows.next().await.unwrap() {
            ddl.push(row.get::<String>(0).unwrap());
        }
        ddl
    };
    let job_column = {
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM pragma_table_info('claim_derivation_jobs')
                  WHERE name = 'eligibility_generation' AND type = 'INTEGER'
                    AND \"notnull\" = 1 AND dflt_value = '0'",
                (),
            )
            .await
            .unwrap();
        rows.next().await.unwrap().unwrap().get(0).unwrap()
    };
    let generation = {
        let mut rows = conn
            .query(
                "SELECT last_generation FROM claim_judge_eligibility_generation WHERE id = 1",
                (),
            )
            .await
            .unwrap();
        rows.next().await.unwrap().unwrap().get(0).unwrap()
    };
    let registry_rows = {
        let mut rows = conn
            .query("SELECT COUNT(*) FROM claim_judge_eligibility", ())
            .await
            .unwrap();
        rows.next().await.unwrap().unwrap().get(0).unwrap()
    };
    (ddl, job_column, generation, registry_rows)
}

#[tokio::test]
async fn migration_110_from_104_installs_eligibility_before_the_105_drift_scan() {
    let (db, _temp) = test_db().await;
    let fresh = {
        let conn = db.conn.lock().await;
        eligibility_schema_fingerprint(&conn).await
    };
    {
        let conn = db.conn.lock().await;
        conn.execute_batch(
            "DROP TABLE claim_judge_eligibility;
             DROP TABLE claim_judge_eligibility_generation;
             ALTER TABLE claim_derivation_jobs DROP COLUMN eligibility_generation;
             ALTER TABLE claim_derivation_jobs DROP COLUMN run_generation;
             PRAGMA user_version = 104;",
        )
        .await
        .unwrap();
    }

    db.run_migrations(&crate::events::NoopEmitter)
        .await
        .unwrap();

    let conn = db.conn.lock().await;
    let user_version: i64 = {
        let mut rows = conn.query("PRAGMA user_version", ()).await.unwrap();
        rows.next().await.unwrap().unwrap().get(0).unwrap()
    };
    assert_eq!(user_version, 110);
    assert_eq!(eligibility_schema_fingerprint(&conn).await, fresh);
}

#[tokio::test]
async fn migration_110_from_109_converges_with_the_fresh_schema() {
    let (db, _temp) = test_db().await;
    let fresh = {
        let conn = db.conn.lock().await;
        let fresh = eligibility_schema_fingerprint(&conn).await;
        assert_eq!(fresh.0.len(), 2);
        assert_eq!(fresh.1, 1);
        assert_eq!(fresh.2, 0);
        assert_eq!(fresh.3, 0, "a fresh registry must fail closed");
        fresh
    };
    {
        let conn = db.conn.lock().await;
        conn.execute_batch(
            "DROP TABLE claim_judge_eligibility;
             DROP TABLE claim_judge_eligibility_generation;
             ALTER TABLE claim_derivation_jobs DROP COLUMN eligibility_generation;
             PRAGMA user_version = 109;",
        )
        .await
        .unwrap();
    }

    db.run_migrations(&crate::events::NoopEmitter)
        .await
        .unwrap();

    let conn = db.conn.lock().await;
    let user_version: i64 = {
        let mut rows = conn.query("PRAGMA user_version", ()).await.unwrap();
        rows.next().await.unwrap().unwrap().get(0).unwrap()
    };
    assert_eq!(user_version, 110);
    let upgraded = eligibility_schema_fingerprint(&conn).await;
    assert_eq!(upgraded, fresh, "fresh and 109 -> 110 must converge");
    assert!(conn
        .execute(
            "INSERT INTO claim_judge_eligibility
                 VALUES ('judge', 'v1', 'p1', 'unknown', 0.75, 1)",
            (),
        )
        .await
        .is_err());
    assert!(conn
        .execute(
            "INSERT INTO claim_judge_eligibility
                 VALUES ('judge', 'v1', 'p1', 'active', 1.01, 1)",
            (),
        )
        .await
        .is_err());
}

async fn install_triggers(db: &MemoryDB) {
    let conn = db.conn.lock().await;
    let tx = conn.transaction().await.unwrap();
    MemoryDB::ensure_claim_derivation_triggers(&tx)
        .await
        .unwrap();
    tx.commit().await.unwrap();
}

async fn add_page(db: &MemoryDB, id: &str) {
    db.insert_page(
        id,
        id,
        None,
        "prose",
        None,
        None,
        &[],
        "2026-07-27T00:00:00Z",
    )
    .await
    .unwrap();
}

async fn set_kind(db: &MemoryDB, id: &str, kind: &str) {
    let conn = db.conn.lock().await;
    conn.execute(
        "UPDATE pages SET kind = ?1 WHERE id = ?2",
        libsql::params![kind, id],
    )
    .await
    .unwrap();
}

/// (job_id, status, attempts) for a page, or None when it is not queued.
async fn job_for(db: &MemoryDB, page_id: &str) -> Option<(String, String, i64)> {
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT job_id, status, attempts FROM claim_derivation_jobs WHERE page_id = ?1",
            libsql::params![page_id],
        )
        .await
        .unwrap();
    let row = rows.next().await.unwrap()?;
    Some((
        row.get::<String>(0).unwrap(),
        row.get::<String>(1).unwrap(),
        row.get::<i64>(2).unwrap(),
    ))
}

async fn mark_derived(db: &MemoryDB, page_id: &str, page_version: i64, extractor: i64) {
    let conn = db.conn.lock().await;
    conn.execute(
        "INSERT INTO claim_derivation_markers
             (page_id, page_version, page_version_digest, extractor_version,
              inventory_count, created_at)
         VALUES (?1, ?2, 'digest', ?3, 0, 0)",
        libsql::params![page_id, page_version, extractor],
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn a_new_page_enqueues_itself() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;

    let job = job_for(&db, "p1").await.expect("new page must be queued");
    assert_eq!(job.0, "p1:1", "job id is page id and version");
    assert_eq!(job.1, "pending");
}

/// An entity shadow page is a projection of an `entities` row, not distilled
/// prose. Deleting `kind <> 'entity'` from the triggers queues thousands of
/// pages that contain no authored claim to derive.
#[tokio::test]
async fn an_entity_shadow_page_is_not_queued() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "shadow").await;
    // The insert trigger already fired for the default 'concept' kind, so clear
    // the queue before flipping the kind — this test is about what the UPDATE
    // trigger declines to re-queue, and what the backlog scan declines to find.
    {
        let conn = db.conn.lock().await;
        conn.execute("DELETE FROM claim_derivation_jobs", ())
            .await
            .unwrap();
    }
    set_kind(&db, "shadow", "entity").await;
    assert!(
        job_for(&db, "shadow").await.is_none(),
        "flipping a page to entity kind must not queue it"
    );

    db.enqueue_stale_derivation_jobs(100).await.unwrap();
    assert!(
        job_for(&db, "shadow").await.is_none(),
        "the backlog scan must skip entity shadow pages too"
    );
}

/// THE test that separates a live queue from one that merely exists. On any
/// real install every page predates the worker, so no insert and no update ever
/// fires. Without the backlog scan the queue reads empty forever, and that zero
/// gets misread as "nothing needs derivation" rather than "nothing is wired".
#[tokio::test]
async fn the_backlog_scan_finds_pages_that_predate_the_worker() {
    let (db, _temp) = db_before_the_worker().await;
    for id in ["old1", "old2", "old3"] {
        add_page(&db, id).await;
    }
    install_triggers(&db).await;

    assert!(
        job_for(&db, "old1").await.is_none(),
        "triggers cannot retroactively fire for pages that already existed"
    );

    let enqueued = db.enqueue_stale_derivation_jobs(100).await.unwrap();
    assert_eq!(enqueued, 3, "every undelivered page joins the queue");
    for id in ["old1", "old2", "old3"] {
        assert_eq!(job_for(&db, id).await.unwrap().1, "pending");
    }
}

/// A zero from the scan must mean "derived", not "not looked at". A page with a
/// current marker is done; re-queueing it would re-spend judge budget on a
/// conclusion already reached.
#[tokio::test]
async fn a_derived_page_is_not_re_enqueued() {
    let (db, _temp) = db_before_the_worker().await;
    add_page(&db, "done").await;
    mark_derived(&db, "done", 1, EXTRACTOR_VERSION).await;
    install_triggers(&db).await;

    let enqueued = db.enqueue_stale_derivation_jobs(100).await.unwrap();
    assert_eq!(enqueued, 0);
    assert!(job_for(&db, "done").await.is_none());
}

/// Identical page text under a changed extractor yields a different claim set,
/// so the old marker no longer describes the page. Dropping the extractor
/// component from the scan's marker check would leave the whole vault frozen at
/// whatever the first extractor concluded.
#[tokio::test]
async fn an_extractor_bump_re_enqueues_an_already_derived_page() {
    let (db, _temp) = db_before_the_worker().await;
    add_page(&db, "stale").await;
    mark_derived(&db, "stale", 1, EXTRACTOR_VERSION - 1).await;
    install_triggers(&db).await;

    let enqueued = db.enqueue_stale_derivation_jobs(100).await.unwrap();
    assert_eq!(
        enqueued, 1,
        "a marker from an older extractor is not a pass"
    );
}

/// The done-job sweep is what makes the previous test work a second time: on an
/// extractor bump the marker check already fails, but a leftover `done` row
/// would still occupy the unique (page_id, page_version) slot and block the
/// re-enqueue. Deleting the sweep leaves the vault permanently un-re-derivable.
#[tokio::test]
async fn a_done_job_from_an_older_extractor_does_not_block_re_enqueue() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    mark_derived(&db, "p1", 1, EXTRACTOR_VERSION - 1).await;
    {
        let conn = db.conn.lock().await;
        conn.execute("UPDATE claim_derivation_jobs SET status = 'done'", ())
            .await
            .unwrap();
    }

    let enqueued = db.enqueue_stale_derivation_jobs(100).await.unwrap();
    assert_eq!(enqueued, 1);
    assert_eq!(job_for(&db, "p1").await.unwrap().1, "pending");
}

/// A page whose text moved has a marker describing text that is gone. The
/// upsert must reopen the finished job rather than ignore the conflict.
#[tokio::test]
async fn an_edit_reopens_a_finished_job() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    {
        let conn = db.conn.lock().await;
        conn.execute("UPDATE claim_derivation_jobs SET status = 'done'", ())
            .await
            .unwrap();
        conn.execute(
            "UPDATE pages SET content = 'rewritten prose' WHERE id = 'p1'",
            (),
        )
        .await
        .unwrap();
    }

    assert_eq!(
        job_for(&db, "p1").await.unwrap().1,
        "pending",
        "an edit must un-finish the derivation of the text it replaced"
    );
}

/// A leased job is off the board. Without the status guard in the lease
/// subquery two workers derive the same page at once and race at finalization.
#[tokio::test]
async fn a_leased_job_is_not_handed_out_twice() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;

    let first = db.lease_next_derivation_job("worker-a").await.unwrap();
    assert_eq!(first.unwrap().page_id, "p1");
    let second = db.lease_next_derivation_job("worker-b").await.unwrap();
    assert!(second.is_none(), "the only job is already leased");
}

/// The reason the lease columns exist. A worker that dies holding a job must
/// not park that page forever.
#[tokio::test]
async fn an_expired_lease_is_reclaimable() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;

    let job = db
        .lease_next_derivation_job("crashed")
        .await
        .unwrap()
        .unwrap();
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE claim_derivation_jobs SET lease_expires_at = 1 WHERE job_id = ?1",
            libsql::params![job.job_id.clone()],
        )
        .await
        .unwrap();
    }

    let reclaimed = db.lease_next_derivation_job("rescuer").await.unwrap();
    assert_eq!(
        reclaimed.expect("expired lease must be reclaimable").job_id,
        job.job_id
    );
}

/// A stalled worker that wakes after its lease was reclaimed must not be able
/// to declare the job finished — the new owner is mid-derivation. Deleting the
/// `lease_owner` guard lets the zombie retire a job that is still in flight.
#[tokio::test]
async fn a_reclaimed_job_cannot_be_finished_by_its_old_owner() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;

    let job = db
        .lease_next_derivation_job("zombie")
        .await
        .unwrap()
        .unwrap();
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE claim_derivation_jobs SET lease_expires_at = 1 WHERE job_id = ?1",
            libsql::params![job.job_id.clone()],
        )
        .await
        .unwrap();
    }
    let reclaimed = db
        .lease_next_derivation_job("rescuer")
        .await
        .unwrap()
        .unwrap();

    assert!(
        !db.finish_derivation_job(&job, "zombie").await.unwrap(),
        "the old owner's finish must be a no-op"
    );
    assert_eq!(job_for(&db, "p1").await.unwrap().1, "leased");
    assert!(db
        .finish_derivation_job(&reclaimed, "rescuer")
        .await
        .unwrap());
    assert_eq!(job_for(&db, "p1").await.unwrap().1, "done");
}

/// Owner names are labels, not run identities. A daemon may restart with the
/// same configured owner after its old lease expires, so the successor needs a
/// generation fence even though `lease_owner` did not change.
#[tokio::test]
async fn a_same_owner_reclaim_rejects_the_old_runs_finish() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;

    let stale = db
        .lease_next_derivation_job("worker")
        .await
        .unwrap()
        .unwrap();
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE claim_derivation_jobs SET lease_expires_at = 1 WHERE job_id = ?1",
            libsql::params![stale.job_id.clone()],
        )
        .await
        .unwrap();
    }
    let current = db
        .lease_next_derivation_job("worker")
        .await
        .unwrap()
        .unwrap();
    assert_ne!(stale.run_generation, current.run_generation);

    assert!(
        !db.finish_derivation_job(&stale, "worker").await.unwrap(),
        "the old run must not finish a same-owner successor"
    );
    assert_eq!(job_for(&db, "p1").await.unwrap().1, "leased");
    assert!(db.finish_derivation_job(&current, "worker").await.unwrap());
}

/// Candidate inventory is an assertion about one exact run, so a worker whose
/// lease was reclaimed must not be able to stamp its old candidate set onto the
/// new owner's generation. Looking up the job's current generation inside the
/// recorder turns the stale worker into a writer for a run it never performed.
#[tokio::test]
async fn a_reclaimed_worker_cannot_record_inventory_for_the_new_run() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 1).await;
    let stale = db
        .lease_next_derivation_job("stale")
        .await
        .unwrap()
        .unwrap();
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE claim_derivation_jobs SET lease_expires_at = 1 WHERE job_id = ?1",
            libsql::params![stale.job_id.clone()],
        )
        .await
        .unwrap();
    }
    let current = db
        .lease_next_derivation_job("current")
        .await
        .unwrap()
        .unwrap();
    assert_ne!(stale.run_generation, current.run_generation);

    let result = db
        .record_run_candidate_inventory(
            &stale,
            "stale",
            &[(
                revisions[0].clone(),
                CandidateLocator {
                    source_id: "stale-source".to_string(),
                    chunk_index: 0,
                    span_start: 0,
                    span_end: 1,
                    span_digest: "stale-digest".to_string(),
                },
            )],
        )
        .await
        .unwrap();
    assert!(
        !result,
        "a stale worker must not write inventory for the current owner's run"
    );

    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT COUNT(*) FROM claim_run_candidates
              WHERE page_id = 'p1' AND page_version = 1 AND run_generation = ?1",
            libsql::params![current.run_generation],
        )
        .await
        .unwrap();
    assert_eq!(
        rows.next().await.unwrap().unwrap().get::<i64>(0).unwrap(),
        0,
        "the refusal must leave the current owner's inventory untouched"
    );
}

/// The production recorder writes the exact candidate set and its completeness
/// seal together for the lease that owns the run.
#[tokio::test]
async fn the_current_worker_records_and_seals_its_candidate_inventory() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 1).await;
    let job = db
        .lease_next_derivation_job("worker")
        .await
        .unwrap()
        .unwrap();
    let locator = CandidateLocator {
        source_id: "source-a".to_string(),
        chunk_index: 2,
        span_start: 3,
        span_end: 8,
        span_digest: "digest-a".to_string(),
    };

    assert!(db
        .record_run_candidate_inventory(&job, "worker", &[(revisions[0].clone(), locator.clone())],)
        .await
        .unwrap());

    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT candidate_source_id, candidate_chunk_index,
                    candidate_span_start, candidate_span_end, candidate_digest
               FROM claim_run_candidates
              WHERE page_id = ?1 AND page_version = ?2 AND run_generation = ?3",
            libsql::params![job.page_id.as_str(), job.page_version, job.run_generation],
        )
        .await
        .unwrap();
    let row = rows.next().await.unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), locator.source_id);
    assert_eq!(row.get::<i64>(1).unwrap(), locator.chunk_index);
    assert_eq!(row.get::<i64>(2).unwrap(), locator.span_start);
    assert_eq!(row.get::<i64>(3).unwrap(), locator.span_end);
    assert_eq!(row.get::<String>(4).unwrap(), locator.span_digest);

    let mut seal = conn
        .query(
            "SELECT COUNT(*) FROM claim_run_inventories
              WHERE page_id = ?1 AND page_version = ?2 AND run_generation = ?3",
            libsql::params![job.page_id.as_str(), job.page_version, job.run_generation],
        )
        .await
        .unwrap();
    assert_eq!(
        seal.next().await.unwrap().unwrap().get::<i64>(0).unwrap(),
        1
    );
}

/// Attempts increment at LEASE time, not at failure time. A worker that dies
/// mid-job never reaches its failure handler, so counting failures would let a
/// hard-crashing page retry without limit and starve the queue.
#[tokio::test]
async fn a_job_that_keeps_crashing_its_worker_parks() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "poison").await;

    for _ in 0..MAX_ATTEMPTS {
        let job = db
            .lease_next_derivation_job("worker")
            .await
            .unwrap()
            .expect("job must stay claimable until attempts run out");
        db.release_derivation_job(&job, "worker", "boom")
            .await
            .unwrap();
    }

    assert!(
        db.lease_next_derivation_job("worker")
            .await
            .unwrap()
            .is_none(),
        "an exhausted job is no longer claimable"
    );
    assert_eq!(
        db.park_exhausted_derivation_jobs().await.unwrap(),
        0,
        "lease acquisition already parks exhausted pending jobs"
    );
    assert_eq!(job_for(&db, "poison").await.unwrap().1, "parked");
}

/// Release has the same causal fence as finish: a stale attempt from a daemon
/// incarnation with the same owner string may not put the current run back on
/// the queue or overwrite its error.
#[tokio::test]
async fn a_same_owner_reclaim_rejects_the_old_runs_release() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;

    let stale = db
        .lease_next_derivation_job("worker")
        .await
        .unwrap()
        .unwrap();
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE claim_derivation_jobs SET lease_expires_at = 1 WHERE job_id = ?1",
            libsql::params![stale.job_id.clone()],
        )
        .await
        .unwrap();
    }
    let current = db
        .lease_next_derivation_job("worker")
        .await
        .unwrap()
        .unwrap();
    assert_ne!(stale.run_generation, current.run_generation);

    assert!(
        !db.release_derivation_job(&stale, "worker", "stale failure")
            .await
            .unwrap(),
        "the old run must not release a same-owner successor"
    );
    assert_eq!(job_for(&db, "p1").await.unwrap().1, "leased");
    assert!(db
        .release_derivation_job(&current, "worker", "current failure")
        .await
        .unwrap());
    assert_eq!(job_for(&db, "p1").await.unwrap().1, "pending");
}

/// A worker that cannot finish must never be the reason a page is called
/// Unsupported. Parking retires the job; it must not touch truth state.
#[tokio::test]
async fn parking_a_job_leaves_the_pages_truth_state_alone() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "poison").await;
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "INSERT INTO page_truth_state
                 (page_id, page_version, support_status, evaluated_at, updated_at)
             VALUES ('poison', 1, 'provisional', NULL, 0)",
            (),
        )
        .await
        .unwrap();
        conn.execute(
            "UPDATE claim_derivation_jobs SET attempts = ?1",
            libsql::params![MAX_ATTEMPTS],
        )
        .await
        .unwrap();
    }

    db.park_exhausted_derivation_jobs().await.unwrap();

    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT support_status, evaluated_at FROM page_truth_state WHERE page_id = 'poison'",
            (),
        )
        .await
        .unwrap();
    let row = rows.next().await.unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "provisional");
    assert!(
        row.get::<Option<i64>>(1).unwrap().is_none(),
        "a parked job leaves the page unevaluated, never judged-and-failed"
    );
}

/// The migration's sweep is bounded so a large vault does not hold every
/// foreground request behind a full scan at boot.
#[tokio::test]
async fn the_backlog_scan_respects_its_limit() {
    let (db, _temp) = db_before_the_worker().await;
    for i in 0..5 {
        add_page(&db, &format!("p{i}")).await;
    }
    install_triggers(&db).await;

    assert_eq!(db.enqueue_stale_derivation_jobs(2).await.unwrap(), 2);
    assert_eq!(db.enqueue_stale_derivation_jobs(2).await.unwrap(), 2);
    assert_eq!(db.enqueue_stale_derivation_jobs(2).await.unwrap(), 1);
    assert_eq!(db.enqueue_stale_derivation_jobs(2).await.unwrap(), 0);
}

/// Replacing stale `done` jobs is part of the same bounded scan. Cleanup must
/// not delete a tail job before that page is captured for re-enqueue, or a
/// nominally two-page constructor turn still mutates the whole vault.
#[tokio::test]
async fn stale_done_job_cleanup_respects_the_captured_batch() {
    let (db, _temp) = db_with_queue().await;
    for id in ["stale_0", "stale_1", "stale_2"] {
        add_page(&db, id).await;
        mark_derived(&db, id, 1, EXTRACTOR_VERSION - 1).await;
    }
    {
        let conn = db.conn.lock().await;
        conn.execute("UPDATE claim_derivation_jobs SET status = 'done'", ())
            .await
            .unwrap();
    }

    assert_eq!(db.enqueue_stale_derivation_jobs(2).await.unwrap(), 2);
    assert_eq!(job_for(&db, "stale_0").await.unwrap().1, "pending");
    assert_eq!(job_for(&db, "stale_1").await.unwrap().1, "pending");
    assert_eq!(
        job_for(&db, "stale_2").await.unwrap().1,
        "done",
        "the uncaptured tail job must remain untouched until its turn"
    );

    assert_eq!(db.enqueue_stale_derivation_jobs(2).await.unwrap(), 1);
    assert_eq!(job_for(&db, "stale_2").await.unwrap().1, "pending");
    assert_eq!(db.enqueue_stale_derivation_jobs(2).await.unwrap(), 0);
}

// ---------------------------------------------------------------------------
// §1 predicate + phase-3 finalization
// ---------------------------------------------------------------------------

/// Give a page a completed derivation: N claims, N revisions, membership rows,
/// and a marker whose digest matches the page's live text.
async fn derive_page(db: &MemoryDB, page_id: &str, claim_count: usize) -> Vec<String> {
    let conn = db.conn.lock().await;
    let content: String = {
        let mut rows = conn
            .query(
                "SELECT content FROM pages WHERE id = ?1",
                libsql::params![page_id],
            )
            .await
            .unwrap();
        rows.next().await.unwrap().unwrap().get(0).unwrap()
    };
    let digest = crate::provenance::revision_content_digest(&content);
    let mut revisions = Vec::new();
    for i in 0..claim_count {
        let claim_id = format!("{page_id}_c{i}");
        let rev_id = format!("{page_id}_r{i}");
        conn.execute(
            "INSERT INTO claims (claim_id, page_id, created_at) VALUES (?1, ?2, 0)",
            libsql::params![claim_id.clone(), page_id],
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO claim_revisions
                 (claim_revision_id, claim_id, predecessor_revision_id, canonical_text,
                  canonical_text_digest, claim_kind, extractor_version, created_at)
             VALUES (?1, ?2, '', ?3, ?4, 'fact', ?5, 0)",
            // Page-specific text and digest. `entailment_cache` is keyed by
            // claim TEXT digest, so two pages sharing a digest would share
            // judgements — true of the real cache, and a fixture that leaks
            // that way silently tests the wrong page.
            libsql::params![
                rev_id.clone(),
                claim_id,
                format!("claim {i} of {page_id}"),
                format!("d_{page_id}_{i}"),
                EXTRACTOR_VERSION
            ],
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO page_version_claims (page_id, page_version, claim_revision_id, ordinal)
             VALUES (?1, 1, ?2, ?3)",
            libsql::params![page_id, rev_id.clone(), i as i64],
        )
        .await
        .unwrap();
        revisions.push(rev_id);
    }
    conn.execute(
        "INSERT INTO claim_derivation_markers
             (page_id, page_version, page_version_digest, extractor_version,
              inventory_count, created_at)
         VALUES (?1, 1, ?2, ?3, ?4, 0)",
        libsql::params![page_id, digest, EXTRACTOR_VERSION, claim_count as i64],
    )
    .await
    .unwrap();
    revisions
}

/// A judged candidate, identified the way a support edge identifies one: which
/// memory, which chunk, which bytes. A digest alone is not an identity — one
/// document can hold the same sentence twice.
#[derive(Clone)]
struct Candidate {
    source_id: String,
    chunk_index: i64,
    span_start: i64,
    span_end: i64,
    span_digest: String,
}

impl Candidate {
    /// The candidate [`support_claim_at_chunk`] writes an edge for: the whole of
    /// [`EVIDENCE_TEXT`] in chunk `chunk_index` of this revision's evidence
    /// memory. The attempt row and the edge have to agree byte for byte, or
    /// evaluation reads the edge as one this run never weighed.
    fn at_chunk(revision_id: &str, chunk_index: i64) -> Self {
        Self {
            source_id: format!("mem_{revision_id}"),
            chunk_index,
            span_start: 0,
            span_end: EVIDENCE_TEXT.len() as i64,
            span_digest: evidence_span_digest(),
        }
    }

    /// A candidate that exists only as a judgement — no edge, no memory row.
    /// For tests about how many candidates a run owed a conclusion on.
    fn named(revision_id: &str, span_digest: &str) -> Self {
        Self {
            source_id: format!("mem_{revision_id}"),
            chunk_index: 0,
            span_start: 0,
            span_end: 0,
            span_digest: span_digest.to_string(),
        }
    }

    /// Same bytes, different place. The collision case: two occurrences of one
    /// sentence share a digest and are still two separate obligations.
    fn at_offset(mut self, span_start: i64, span_end: i64) -> Self {
        self.span_start = span_start;
        self.span_end = span_end;
        self
    }
}

/// The run a page's job is currently on. Fixtures stamp their attempt rows with
/// it for the same reason a worker does: a conclusion belongs to the attempt
/// that reached it, and evaluation reads no other.
async fn current_run(conn: &libsql::Connection, page_id: &str, page_version: i64) -> i64 {
    let mut rows = conn
        .query(
            "SELECT run_generation FROM claim_derivation_jobs
              WHERE page_id = ?1 AND page_version = ?2",
            libsql::params![page_id, page_version],
        )
        .await
        .unwrap();
    match rows.next().await.unwrap() {
        Some(row) => row.get::<i64>(0).unwrap(),
        None => 0,
    }
}

/// Record that this run weighed a candidate for this claim and reached
/// `outcome`, plus the cache row a real judge would have left behind.
///
/// The attempt row is what conditions 3 and 4 read. The cache row is not — it is
/// written because a judgement that left no cache entry is a state the real
/// pipeline does not produce, and a fixture that omitted it would be testing a
/// DB that cannot exist.
async fn attempt_on(db: &MemoryDB, revision_id: &str, candidate: &Candidate, outcome: &str) {
    let conn = db.conn.lock().await;
    let (digest, page_id, page_version): (String, String, i64) = {
        let mut rows = conn
            .query(
                "SELECT cr.canonical_text_digest, pvc.page_id, pvc.page_version
                   FROM claim_revisions cr
                   JOIN page_version_claims pvc
                     ON pvc.claim_revision_id = cr.claim_revision_id
                  WHERE cr.claim_revision_id = ?1",
                libsql::params![revision_id],
            )
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        (
            row.get(0).unwrap(),
            row.get(1).unwrap(),
            row.get(2).unwrap(),
        )
    };
    let generation = current_run(&conn, &page_id, page_version).await;
    // A real run records the candidate it owes a conclusion on BEFORE it
    // judges, and seals the set when it has recorded all of them. Seeding the
    // attempt without the obligation would be seeding a state the pipeline
    // cannot produce -- and the one the evaluator now refuses to publish on.
    conn.execute(
        "INSERT OR REPLACE INTO claim_run_candidates
             (page_id, page_version, run_generation, claim_revision_id,
              candidate_source_id, candidate_chunk_index, candidate_span_start,
              candidate_span_end, candidate_digest)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        libsql::params![
            page_id.clone(),
            page_version,
            generation,
            revision_id,
            candidate.source_id.clone(),
            candidate.chunk_index,
            candidate.span_start,
            candidate.span_end,
            candidate.span_digest.clone(),
        ],
    )
    .await
    .unwrap();
    conn.execute(
        "INSERT OR REPLACE INTO claim_run_inventories
             (page_id, page_version, run_generation, sealed_at)
         VALUES (?1, ?2, ?3, 0)",
        libsql::params![page_id.clone(), page_version, generation],
    )
    .await
    .unwrap();
    conn.execute(
        "INSERT OR REPLACE INTO claim_judgment_attempts
             (page_id, page_version, run_generation, claim_revision_id,
              candidate_source_id, candidate_chunk_index, candidate_span_start,
              candidate_span_end, candidate_digest, outcome, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, 0)",
        libsql::params![
            page_id,
            page_version,
            generation,
            revision_id,
            candidate.source_id.clone(),
            candidate.chunk_index,
            candidate.span_start,
            candidate.span_end,
            candidate.span_digest.clone(),
            outcome
        ],
    )
    .await
    .unwrap();
    conn.execute(
        "INSERT OR IGNORE INTO entailment_cache
             (claim_text_digest, source_span_digest, model_id, model_version,
              prompt_version, score, threshold_at_write, backend, scored_at)
         VALUES (?1, ?2, 'judge', 'v1', 'p1', 0.1, 0.75, ?3, 0)",
        libsql::params![
            digest,
            candidate.span_digest.clone(),
            crate::claim_judge::M5_CLAIM_ENTAILMENT_CACHE_BACKEND,
        ],
    )
    .await
    .unwrap();
}

/// This run weighed every candidate for the claim and none cleared the bar —
/// as distinct from nobody ever looking, and from looking and not finishing.
async fn judged_but_short(db: &MemoryDB, revision_id: &str) {
    add_live_candidate_chunk(db, revision_id, 0).await;
    attempt_on(
        db,
        revision_id,
        &Candidate::at_chunk(revision_id, 0),
        "concluded",
    )
    .await;
}

/// The evidence prose every support fixture cites, and the digest of the span
/// covering all of it.
const EVIDENCE_TEXT: &str = "evidence prose";

fn evidence_span_digest() -> String {
    crate::provenance::revision_content_digest(EVIDENCE_TEXT)
}

/// A `supports` edge at `score`, written straight to `edges` — these tests are
/// about what the predicate READS, not about the write path's own refusals.
/// The matching `entailment_cache` row goes in too, because a support edge that
/// cannot name the verdict behind it is a state `write_support_edge` refuses to
/// create, and a fixture that produced one would be testing an impossible DB.
///
/// The edge is fully formed for the same reason: evaluation now revalidates the
/// chunk, version, span, and root a verdict named, so an edge missing any of
/// them is not a shortcut but a DIFFERENT test — one about a malformed edge
/// rather than about the condition under test. Cheaper to seed it right.
async fn support_claim(db: &MemoryDB, page_id: &str, revision_id: &str, score: f64) {
    support_claim_at_chunk(db, page_id, revision_id, score, 0).await;
}

/// [`support_claim`] where the cited evidence is some chunk other than the
/// first, so a test can delete or rewrite the chunk a verdict actually names
/// while its document lives on.
async fn support_claim_at_chunk(
    db: &MemoryDB,
    page_id: &str,
    revision_id: &str,
    score: f64,
    chunk_index: i64,
) {
    // The attempt row has to name the SAME location the edge below cites. A
    // conclusion recorded against chunk 0 does not discharge an edge into
    // chunk 3 — that is the whole point of locating a candidate.
    attempt_on(
        db,
        revision_id,
        &Candidate::at_chunk(revision_id, chunk_index),
        "concluded",
    )
    .await;
    let conn = db.conn.lock().await;
    let space: String = {
        let mut rows = conn
            .query(
                "SELECT space FROM pages WHERE id = ?1",
                libsql::params![page_id],
            )
            .await
            .unwrap();
        rows.next().await.unwrap().unwrap().get(0).unwrap()
    };
    let mem_id = format!("mem_{revision_id}");
    conn.execute(
        // `source_id`, not `id`: the space fence resolves a memory endpoint by
        // source_id, so a row without one reads as spaceless and is rejected.
        "INSERT INTO memories (id, content, source, source_id, title, chunk_index,
                               last_modified, chunk_type, space)
         VALUES (?1, ?4, 'memory', ?2, 'evidence', ?5, 0, 'text', ?3)",
        libsql::params![
            format!("m_{mem_id}"),
            mem_id.clone(),
            space.clone(),
            EVIDENCE_TEXT,
            chunk_index
        ],
    )
    .await
    .unwrap();
    // `edges` CHECKs that a supports edge carries a root — the schema enforcing
    // the same rule `write_support_edge` verifies. The identity digest is the
    // real one over the evidence bytes, because revalidation recomputes it.
    //
    // ONE root for every fixture edge, not one per revision: roots are
    // content-addressed and `provenance_roots` is UNIQUE on
    // (identity_version, identity_digest), so identical evidence bytes converge
    // on a single row — which is what `acquire_provenance_root` does in
    // production too.
    let root_id = "root_evidence_prose".to_string();
    conn.execute(
        "INSERT OR IGNORE INTO provenance_roots
             (root_id, identity_version, identity_digest, root_kind,
              independence_group_id, status, created_at)
         VALUES (?1, ?3, ?2, 'document_ingest', 'group_evidence_prose', 'active', 0)",
        libsql::params![
            root_id.clone(),
            crate::provenance::identity_digest("document_ingest", EVIDENCE_TEXT),
            crate::provenance::IDENTITY_VERSION
        ],
    )
    .await
    .unwrap();
    let payload = serde_json::json!({
        "chunk_index": chunk_index,
        "source_version": 1,
        "span_start": 0,
        "span_end": EVIDENCE_TEXT.len(),
        "span_digest": evidence_span_digest(),
        "model_id": "judge",
        "model_version": "v1",
        "prompt_version": "p1",
        "score": score,
        "threshold_at_write": 0.75,
    })
    .to_string();
    conn.execute(
        "INSERT INTO edges (edge_id, src_id, src_kind, dst_id, dst_kind, edge_type, lineage,
                            grounded, root_id, space, weight, payload, provenance,
                            operation_id, created_at, superseded_by, valid_until)
         VALUES (?1, ?2, 'claim_revision', ?3, 'memory', 'supports', 'evidence',
                 0, ?4, ?5, NULL, ?6, NULL, NULL, 0, NULL, NULL)",
        libsql::params![
            format!("e_{revision_id}"),
            revision_id,
            mem_id,
            root_id,
            space,
            payload
        ],
    )
    .await
    .unwrap();
}

/// Re-file the previous run's judgements under `generation`, standing in for a
/// worker that reached those same conclusions during the run it now holds.
///
/// Evaluation only reads the current run's attempt rows, so without this a
/// fixture that seeded judgements before taking a lease would evaluate as a page
/// nobody has weighed yet. Tests that are ABOUT run isolation deliberately skip
/// this and let the older rows stay where they are.
async fn carry_attempts_forward(
    conn: &libsql::Connection,
    page_id: &str,
    page_version: i64,
    generation: i64,
) {
    conn.execute(
        "INSERT OR REPLACE INTO claim_judgment_attempts
             (page_id, page_version, run_generation, claim_revision_id,
              candidate_source_id, candidate_chunk_index, candidate_span_start,
              candidate_span_end, candidate_digest, outcome, updated_at)
         SELECT page_id, page_version, ?3, claim_revision_id,
                candidate_source_id, candidate_chunk_index, candidate_span_start,
                candidate_span_end, candidate_digest, outcome, updated_at
           FROM claim_judgment_attempts
          WHERE page_id = ?1 AND page_version = ?2
            AND run_generation = (SELECT MAX(run_generation)
                                    FROM claim_judgment_attempts
                                   WHERE page_id = ?1 AND page_version = ?2
                                     AND run_generation < ?3)",
        libsql::params![page_id, page_version, generation],
    )
    .await
    .unwrap();
    // The obligations come forward with the conclusions. A run holding
    // judgements it did not record owing is a run that cannot publish, so
    // carrying one without the other would model a worker no fixture means.
    conn.execute(
        "INSERT OR REPLACE INTO claim_run_candidates
             (page_id, page_version, run_generation, claim_revision_id,
              candidate_source_id, candidate_chunk_index, candidate_span_start,
              candidate_span_end, candidate_digest)
         SELECT page_id, page_version, ?3, claim_revision_id,
                candidate_source_id, candidate_chunk_index, candidate_span_start,
                candidate_span_end, candidate_digest
           FROM claim_run_candidates
          WHERE page_id = ?1 AND page_version = ?2
            AND run_generation = (SELECT MAX(run_generation)
                                    FROM claim_run_candidates
                                   WHERE page_id = ?1 AND page_version = ?2
                                     AND run_generation < ?3)",
        libsql::params![page_id, page_version, generation],
    )
    .await
    .unwrap();
    seal_run_inventory(conn, page_id, page_version, generation).await;
}

/// Mark this run's candidate inventory complete, standing in for the worker
/// call that records the set before judging any of it.
///
/// Sealing an EMPTY inventory is a real state and a deliberate one: a run whose
/// claims drew no candidates recorded that fact and finished. Without the seal
/// the same page reads as a run that died before recording anything, which is
/// `NoPublish` rather than a verdict.
async fn seal_run_inventory(
    conn: &libsql::Connection,
    page_id: &str,
    page_version: i64,
    generation: i64,
) {
    conn.execute(
        "INSERT OR REPLACE INTO claim_run_inventories
             (page_id, page_version, run_generation, sealed_at)
         VALUES (?1, ?2, ?3, 0)",
        libsql::params![page_id, page_version, generation],
    )
    .await
    .unwrap();
}

/// Record that this run owed a conclusion on a candidate and never reached it —
/// no attempt row, and no support edge either.
///
/// The state a worker leaves by dying between recording its inventory and
/// judging the last of it, and the one neither the attempt rows nor the live
/// edges can show on their own.
async fn owed_but_unjudged(db: &MemoryDB, revision_id: &str, candidate: &Candidate) {
    let conn = db.conn.lock().await;
    let (page_id, page_version): (String, i64) = {
        let mut rows = conn
            .query(
                "SELECT page_id, page_version FROM page_version_claims
                  WHERE claim_revision_id = ?1",
                libsql::params![revision_id],
            )
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        (row.get(0).unwrap(), row.get(1).unwrap())
    };
    let generation = current_run(&conn, &page_id, page_version).await;
    conn.execute(
        "INSERT OR REPLACE INTO claim_run_candidates
             (page_id, page_version, run_generation, claim_revision_id,
              candidate_source_id, candidate_chunk_index, candidate_span_start,
              candidate_span_end, candidate_digest)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        libsql::params![
            page_id.clone(),
            page_version,
            generation,
            revision_id,
            candidate.source_id.clone(),
            candidate.chunk_index,
            candidate.span_start,
            candidate.span_end,
            candidate.span_digest.clone(),
        ],
    )
    .await
    .unwrap();
    seal_run_inventory(&conn, &page_id, page_version, generation).await;
}

/// [`carry_attempts_forward`] for a test that took its lease through the REAL
/// `lease_next_derivation_job` — which allocates a run of its own and carries
/// nothing, exactly as a worker starting fresh should. Tests that seeded their
/// judgements up front call this to say "this run reached them too".
async fn carry_seeded_judgements(db: &MemoryDB, page_id: &str, page_version: i64) {
    let conn = db.conn.lock().await;
    let generation = current_run(&conn, page_id, page_version).await;
    carry_attempts_forward(&conn, page_id, page_version, generation).await;
}

/// Put one page's job in the state a worker holds it in, and return its exact
/// leased run handle.
///
/// Publication is bound to the lease, so every test that finalizes has to hold
/// one. Leasing by `(page_id, page_version)` rather than through
/// `lease_next_derivation_job` keeps a multi-page test from having to guess the
/// queue order, and keeps the version straight on a page that has jobs for two.
/// The real lease path has its own teeth above; this is a fixture.
async fn lease_page(db: &MemoryDB, page_id: &str, page_version: i64, owner: &str) -> DerivationJob {
    let conn = db.conn.lock().await;
    let generation = MemoryDB::allocate_run_generation(&conn).await.unwrap();
    let eligibility_generation = {
        let mut rows = conn
            .query(
                "SELECT last_generation FROM claim_judge_eligibility_generation WHERE id = 1",
                (),
            )
            .await
            .unwrap();
        rows.next().await.unwrap().unwrap().get::<i64>(0).unwrap()
    };
    let job = {
        let mut rows = conn
            .query(
                "UPDATE claim_derivation_jobs
                    SET status = 'leased', lease_owner = ?3,
                        lease_expires_at = ?4, attempts = attempts + 1,
                        run_generation = ?5, eligibility_generation = ?6,
                        updated_at = ?4
                  WHERE page_id = ?1 AND page_version = ?2
                  RETURNING job_id, page_id, page_version",
                libsql::params![
                    page_id,
                    page_version,
                    owner,
                    chrono::Utc::now().timestamp() + 600,
                    generation,
                    eligibility_generation
                ],
            )
            .await
            .unwrap();
        let row = rows
            .next()
            .await
            .unwrap()
            .expect("the page must have a queued job to lease");
        DerivationJob {
            job_id: row.get::<String>(0).unwrap(),
            page_id: row.get::<String>(1).unwrap(),
            page_version: row.get::<i64>(2).unwrap(),
            run_generation: generation,
            eligibility_generation,
        }
    };
    carry_attempts_forward(&conn, page_id, page_version, generation).await;
    job
}

async fn truth_row(db: &MemoryDB, page_id: &str) -> Option<(String, Option<i64>, i64)> {
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT support_status, evaluated_at, human_reviewed
               FROM page_truth_state WHERE page_id = ?1",
            libsql::params![page_id],
        )
        .await
        .unwrap();
    let row = rows.next().await.unwrap()?;
    Some((
        row.get::<String>(0).unwrap(),
        row.get::<Option<i64>>(1).unwrap(),
        row.get::<i64>(2).unwrap(),
    ))
}

#[tokio::test]
async fn a_fully_supported_page_is_supported() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 2).await;
    for rev in &revisions {
        support_claim(&db, "p1", rev, 0.9).await;
    }

    assert_eq!(
        db.evaluate_page_support("p1", 1).await.unwrap(),
        SupportOutcome::Supported
    );
    let job = lease_page(&db, "p1", 1, "worker").await;
    assert!(db
        .finalize_page_support("p1", 1, &job, "worker", &SupportOutcome::Supported)
        .await
        .unwrap());
    let (status, evaluated, reviewed) = truth_row(&db, "p1").await.unwrap();
    assert_eq!(status, "supported");
    assert!(evaluated.is_some());
    assert_eq!(reviewed, 0, "machine support never manufactures review");
}

/// A production writer makes the judge identity reachable, so the evaluator
/// may no longer accept every well-formed support edge indiscriminately. The
/// M5 contract requires eligibility to be keyed by the exact
/// `(model_id, model_version, prompt_version)` triple; an unknown triple is
/// evidence from a judge this install has never authorized and must fail
/// closed even when its stored score clears the numeric threshold.
#[tokio::test]
async fn an_unregistered_judge_cannot_make_a_page_supported() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 1).await;
    support_claim(&db, "p1", &revisions[0], 0.9).await;

    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE entailment_cache
                SET model_id = 'unregistered-judge',
                    model_version = 'unregistered-version',
                    prompt_version = 'unregistered-prompt'",
            (),
        )
        .await
        .unwrap();
        conn.execute(
            "UPDATE edges
                SET payload = json_set(
                    payload,
                    '$.model_id', 'unregistered-judge',
                    '$.model_version', 'unregistered-version',
                    '$.prompt_version', 'unregistered-prompt'
                )
              WHERE edge_type = 'supports'",
            (),
        )
        .await
        .unwrap();
    }

    assert!(matches!(
        db.evaluate_page_support("p1", 1).await.unwrap(),
        SupportOutcome::Refuted { .. }
    ));
}

/// Rolling upgrades keep old evidence live while its judge drains. Treating
/// draining as retired would mass-demote pages before replacement judgments
/// have had time to arrive.
#[tokio::test]
async fn a_draining_judge_remains_eligible_for_existing_support() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 1).await;
    support_claim(&db, "p1", &revisions[0], 0.9).await;

    set_test_judge_eligibility(&db, "draining", SUPPORT_THRESHOLD).await;

    assert_eq!(
        db.evaluate_page_support("p1", 1).await.unwrap(),
        SupportOutcome::Supported
    );
}

#[tokio::test]
async fn a_retired_judge_cannot_support_a_page() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 1).await;
    support_claim(&db, "p1", &revisions[0], 0.9).await;

    set_test_judge_eligibility(&db, "retired", SUPPORT_THRESHOLD).await;

    assert!(matches!(
        db.evaluate_page_support("p1", 1).await.unwrap(),
        SupportOutcome::Refuted { .. }
    ));
}

#[tokio::test]
async fn raising_a_judges_threshold_demotes_stored_support() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 1).await;
    support_claim(&db, "p1", &revisions[0], 0.9).await;
    let job = lease_page(&db, "p1", 1, "worker").await;
    let supported = db.evaluate_page_support("p1", 1).await.unwrap();
    assert_eq!(supported, SupportOutcome::Supported);
    assert!(db
        .finalize_page_support("p1", 1, &job, "worker", &supported)
        .await
        .unwrap());

    set_test_judge_eligibility(&db, "active", 0.95).await;

    assert!(matches!(
        db.evaluate_page_support("p1", 1).await.unwrap(),
        SupportOutcome::Refuted { .. }
    ));
    assert_eq!(db.reconcile_supported_pages(100).await.unwrap().demoted, 1);
    assert_eq!(truth_row(&db, "p1").await.unwrap().0, "provisional");
}

#[tokio::test]
async fn eligibility_generation_change_between_lease_and_finalize_writes_nothing() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 1).await;
    support_claim(&db, "p1", &revisions[0], 0.9).await;
    let leased_under = test_eligibility_generation(&db).await;
    let job = db
        .lease_next_derivation_job("worker")
        .await
        .unwrap()
        .expect("the page must be claimable");
    assert_eq!(job.eligibility_generation, leased_under);
    carry_seeded_judgements(&db, "p1", 1).await;
    let outcome = db.evaluate_page_support("p1", 1).await.unwrap();
    assert_eq!(outcome, SupportOutcome::Supported);
    let before = truth_row(&db, "p1").await;

    let changed_generation = set_test_judge_eligibility(&db, "draining", SUPPORT_THRESHOLD).await;
    assert!(changed_generation > job.eligibility_generation);
    assert_eq!(test_eligibility_generation(&db).await, changed_generation);
    assert_eq!(
        db.evaluate_page_support("p1", 1).await.unwrap(),
        SupportOutcome::Supported,
        "the generation CAS must bite even when the new rules reach the same verdict"
    );

    assert!(!db
        .finalize_page_support("p1", 1, &job, "worker", &outcome)
        .await
        .unwrap());
    assert_eq!(truth_row(&db, "p1").await, before);
}

/// Weakening: let "unfiled" be spelled two ways.
///
/// Every other test here reads an edge the fixture seeded, and a fixture that
/// copies `pages.space` onto its own edge agrees with itself by construction —
/// it cannot see a disagreement between the two sides. This one drives the
/// PRODUCTION writer end to end on a page with no space at all, which is the
/// only place the question can actually be asked: `write_support_edge` stamps
/// the edge with `COALESCE(memories.space, UNFILED_SPACE_ID)` while the fence
/// resolves the claim side through to `pages.space`. If unfiled were NULL on
/// one side and the sentinel on the other, every unfiled page in the vault
/// would be permanently un-supportable, and the worker would read the resulting
/// zero as "no evidence found" rather than "the write was refused" — the
/// dead-substrate misread one layer down, and invisible for exactly the reason
/// the cross-space case is.
///
/// It doubles as the shape check the seeded fixtures cannot give: the predicate
/// reads an edge this crate's writer actually wrote.
#[tokio::test]
async fn an_unfiled_page_can_be_supported_by_unfiled_evidence() {
    const ADDED: &str = "They nest in old crow nests.";
    let (db, _temp) = db_with_queue().await;
    // `add_page` passes no space, so the page is unfiled — the case at issue.
    add_page(&db, "uf1").await;
    let base = crate::provenance::revision_content_digest("prose");
    let minted = db
        .mint_human_edit_delta("uf1", 1, &base, &format!("prose\n{ADDED}"))
        .await
        .unwrap()
        .expect("the scenario needs a real delta to cite");
    let revisions = derive_page(&db, "uf1", 1).await;

    let span_digest = crate::provenance::revision_content_digest(ADDED);
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "INSERT INTO entailment_cache
                 (claim_text_digest, source_span_digest, model_id, model_version,
                  prompt_version, score, threshold_at_write, backend, scored_at)
             SELECT canonical_text_digest, ?2, 'judge', 'v1', 'p1', 0.91, 0.75, ?3, 0
               FROM claim_revisions WHERE claim_revision_id = ?1",
            libsql::params![
                revisions[0].clone(),
                span_digest.clone(),
                crate::claim_judge::M5_CLAIM_ENTAILMENT_CACHE_BACKEND,
            ],
        )
        .await
        .unwrap();
    }
    let verdict = super::claim_identity::SupportVerdict {
        chunk_index: 0,
        source_version: 1,
        span_start: 0,
        span_end: ADDED.len() as i64,
        span_digest,
        model_id: "judge".to_string(),
        model_version: "v1".to_string(),
        prompt_version: "p1".to_string(),
        score: 0.91,
        threshold_at_write: 0.75,
    };
    db.write_support_edge(
        &revisions[0],
        &minted.memory_source_id,
        &minted.root_id,
        &verdict,
    )
    .await
    .expect("an unfiled page and its own unfiled evidence are the same space");
    // The judgement behind that edge, filed where the judge would have filed it.
    // Writing the edge without it would leave the run owing a conclusion on its
    // own evidence — the state the inventory check exists to catch.
    attempt_on(
        &db,
        &revisions[0],
        &Candidate {
            source_id: minted.memory_source_id.clone(),
            chunk_index: verdict.chunk_index,
            span_start: verdict.span_start,
            span_end: verdict.span_end,
            span_digest: verdict.span_digest.clone(),
        },
        "concluded",
    )
    .await;

    // Both sides agree, and they agree on the sentinel rather than on NULL —
    // asserted rather than inferred from the write succeeding, because a fence
    // written with `IS NOT` would also pass NULL against NULL, and that DB
    // would break the moment one side started defaulting.
    {
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT (SELECT space FROM pages WHERE id = 'uf1'),
                        (SELECT space FROM memories WHERE source_id = ?1)",
                libsql::params![minted.memory_source_id.clone()],
            )
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        let page_space: Option<String> = row.get(0).unwrap();
        let memory_space: Option<String> = row.get(1).unwrap();
        assert_eq!(
            page_space.as_deref(),
            Some(super::UNFILED_SPACE_ID),
            "an unfiled page stores the sentinel, not NULL"
        );
        assert_eq!(
            memory_space, page_space,
            "the evidence must land in the same spelling of unfiled as its page"
        );
    }

    assert_eq!(
        db.evaluate_page_support("uf1", 1).await.unwrap(),
        SupportOutcome::Supported,
        "the predicate must count an edge the production writer wrote"
    );
}

/// Condition 3, and the only failure that is a verdict: the claims are real, a
/// judge scored them, and nothing cleared the bar.
#[tokio::test]
async fn a_claim_judged_and_found_wanting_refutes_the_page() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 2).await;
    support_claim(&db, "p1", &revisions[0], 0.9).await;
    judged_but_short(&db, &revisions[1]).await;

    let outcome = db.evaluate_page_support("p1", 1).await.unwrap();
    let SupportOutcome::Refuted { ref reason } = outcome else {
        panic!("got {outcome:?}");
    };
    assert!(
        reason.contains("judged and fell short"),
        "the stored reason must say the evidence was weighed: {reason}"
    );
    let job = lease_page(&db, "p1", 1, "worker").await;
    db.finalize_page_support("p1", 1, &job, "worker", &outcome)
        .await
        .unwrap();
    let (status, evaluated, _) = truth_row(&db, "p1").await.unwrap();
    assert_eq!(status, "provisional");
    assert!(
        evaluated.is_some(),
        "a real verdict stamps evaluated_at — this is the one case that may cost a file"
    );
}

/// The widened reason set, and the safety call inside it. A claim no judge has
/// ever scored means OUR pipeline did not run, not that the page is unsupported
/// — so it must not stamp `evaluated_at`, and the stored reason must say which
/// of the two situations it was. Collapsing these back into one `provisional`
/// makes a gathering gap indistinguishable from a real refutation at exactly
/// the moment a human is deciding whether to trust the page.
#[tokio::test]
async fn a_claim_nobody_judged_leaves_the_page_unevaluated() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 2).await;
    support_claim(&db, "p1", &revisions[0], 0.9).await;
    // revisions[1] is never judged at all — no edge, no cache row.

    let outcome = db.evaluate_page_support("p1", 1).await.unwrap();
    let SupportOutcome::Unevaluated { ref reason } = outcome else {
        panic!("a claim nobody judged is not a verdict; got {outcome:?}");
    };
    assert!(
        reason.contains("no candidate evidence"),
        "the stored reason must name the gathering gap: {reason}"
    );

    let job = lease_page(&db, "p1", 1, "worker").await;
    db.finalize_page_support("p1", 1, &job, "worker", &outcome)
        .await
        .unwrap();
    let (status, evaluated, _) = truth_row(&db, "p1").await.unwrap();
    assert_eq!(status, "provisional");
    assert!(
        evaluated.is_none(),
        "an ungathered claim must never cost the page its file"
    );
}

/// The two reasons must be distinguishable in the stored text, not merely in
/// the enum a caller has already thrown away.
#[tokio::test]
async fn the_two_provisional_reasons_are_distinguishable_on_disk() {
    let (db, _temp) = db_with_queue().await;
    for id in ["ungathered", "refuted"] {
        add_page(&db, id).await;
    }
    let a = derive_page(&db, "ungathered", 1).await;
    let b = derive_page(&db, "refuted", 1).await;
    let _ = a;
    judged_but_short(&db, &b[0]).await;

    for id in ["ungathered", "refuted"] {
        let outcome = db.evaluate_page_support(id, 1).await.unwrap();
        let job = lease_page(&db, id, 1, "worker").await;
        db.finalize_page_support(id, 1, &job, "worker", &outcome)
            .await
            .unwrap();
    }

    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT page_id, provisional_reason FROM page_truth_state ORDER BY page_id",
            (),
        )
        .await
        .unwrap();
    let mut seen = Vec::new();
    while let Some(row) = rows.next().await.unwrap() {
        seen.push((
            row.get::<String>(0).unwrap(),
            row.get::<Option<String>>(1).unwrap().unwrap_or_default(),
        ));
    }
    assert_eq!(seen.len(), 2);
    let refuted = &seen.iter().find(|r| r.0 == "refuted").unwrap().1;
    let ungathered = &seen.iter().find(|r| r.0 == "ungathered").unwrap().1;
    assert!(refuted.contains("judged and fell short"), "{refuted}");
    assert!(ungathered.contains("no candidate evidence"), "{ungathered}");
    assert_ne!(refuted, ungathered);
}

/// Row 15. Comparing each edge against its own `threshold_at_write` instead of
/// the live threshold would make every stored verdict permanently
/// self-certifying, and raising the bar would demote nothing.
#[tokio::test]
async fn support_below_the_current_threshold_does_not_count() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 1).await;
    support_claim(&db, "p1", &revisions[0], SUPPORT_THRESHOLD - 0.01).await;

    assert!(matches!(
        db.evaluate_page_support("p1", 1).await.unwrap(),
        SupportOutcome::Refuted { .. }
    ));
}

/// An invalidated edge is evidence withdrawn, not evidence.
#[tokio::test]
async fn an_invalidated_support_edge_does_not_count() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 1).await;
    support_claim(&db, "p1", &revisions[0], 0.9).await;
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE edges SET valid_until = 1 WHERE edge_type = 'supports'",
            (),
        )
        .await
        .unwrap();
    }

    assert!(matches!(
        db.evaluate_page_support("p1", 1).await.unwrap(),
        SupportOutcome::Refuted { .. }
    ));
}

/// THE mass-flip tooth. A worker that cannot resolve any evidence must leave
/// every page exactly where it was — provisional AND unevaluated — because only
/// `evaluated_at` being set turns `provisional` into `Unsupported`, and only
/// `Unsupported` costs a page its projected file. The 511-file precedent is
/// what this test exists to prevent recurring.
#[tokio::test]
async fn a_worker_that_resolves_no_evidence_flips_nothing() {
    let (db, _temp) = db_with_queue().await;
    for id in ["p1", "p2", "p3"] {
        add_page(&db, id).await;
    }

    for id in ["p1", "p2", "p3"] {
        let outcome = db.evaluate_page_support(id, 1).await.unwrap();
        assert!(
            matches!(outcome, SupportOutcome::Unevaluated { .. }),
            "{id} got {outcome:?}"
        );
        let job = lease_page(&db, id, 1, "worker").await;
        db.finalize_page_support(id, 1, &job, "worker", &outcome)
            .await
            .unwrap();
    }

    for id in ["p1", "p2", "p3"] {
        let (status, evaluated, _) = truth_row(&db, id).await.unwrap();
        assert_eq!(status, "provisional");
        assert!(
            evaluated.is_none(),
            "{id} was never judged, so it must never read as Unsupported"
        );
    }
}

/// Condition 2. `forall x in {}` is trivially true in every natural
/// implementation, so an empty inventory has to be refused by name.
#[tokio::test]
async fn an_empty_inventory_is_never_supported() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    derive_page(&db, "p1", 0).await;

    let outcome = db.evaluate_page_support("p1", 1).await.unwrap();
    assert!(
        matches!(outcome, SupportOutcome::Unevaluated { .. }),
        "got {outcome:?}"
    );
}

/// And the second half of that call: a page the extractor could not cut into
/// claims must not be ARCHIVED for it either. Stamping `evaluated_at` on an
/// empty inventory would take the file off every stub, list, and table page in
/// the vault at the next cutover.
#[tokio::test]
async fn an_empty_inventory_is_unevaluated_not_refuted() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    derive_page(&db, "p1", 0).await;

    let outcome = db.evaluate_page_support("p1", 1).await.unwrap();
    let job = lease_page(&db, "p1", 1, "worker").await;
    db.finalize_page_support("p1", 1, &job, "worker", &outcome)
        .await
        .unwrap();
    let (status, evaluated, _) = truth_row(&db, "p1").await.unwrap();
    assert_eq!(status, "provisional");
    assert!(evaluated.is_none());
}

/// Condition 1, extractor half. Same page text, bumped extractor: the marker
/// must be rejected, because identical prose under a changed extractor yields a
/// different claim set.
#[tokio::test]
async fn a_marker_from_another_extractor_is_not_a_derivation() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 1).await;
    support_claim(&db, "p1", &revisions[0], 0.9).await;
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE claim_derivation_markers SET extractor_version = ?1",
            libsql::params![EXTRACTOR_VERSION + 1],
        )
        .await
        .unwrap();
    }

    assert!(matches!(
        db.evaluate_page_support("p1", 1).await.unwrap(),
        SupportOutcome::Unevaluated { .. }
    ));
}

/// Condition 1, digest half. A marker describing text the page no longer holds
/// is not a judgement about the page as it is.
#[tokio::test]
async fn a_marker_against_different_text_is_not_a_derivation() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 1).await;
    support_claim(&db, "p1", &revisions[0], 0.9).await;
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE claim_derivation_markers SET page_version_digest = 'other'",
            (),
        )
        .await
        .unwrap();
    }

    assert!(matches!(
        db.evaluate_page_support("p1", 1).await.unwrap(),
        SupportOutcome::Unevaluated { .. }
    ));
}

/// Row 6. A run that finished some claims and lost others publishes NOTHING —
/// not `supported` on the strength of the ones that landed, and not
/// `provisional` either. The page keeps whatever state it had.
#[tokio::test]
async fn a_partial_derivation_publishes_nothing() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 2).await;
    for rev in &revisions {
        support_claim(&db, "p1", rev, 0.9).await;
    }
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "DELETE FROM page_version_claims WHERE claim_revision_id = ?1",
            libsql::params![revisions[1].clone()],
        )
        .await
        .unwrap();
    }

    let outcome = db.evaluate_page_support("p1", 1).await.unwrap();
    assert!(
        matches!(outcome, SupportOutcome::NoPublish { .. }),
        "got {outcome:?}"
    );
    let job = lease_page(&db, "p1", 1, "worker").await;
    assert!(
        !db.finalize_page_support("p1", 1, &job, "worker", &outcome)
            .await
            .unwrap(),
        "a no-publish outcome must write nothing at all"
    );
    assert!(truth_row(&db, "p1").await.is_none());
}

/// Model work happens outside every lock, so the page may move under a verdict
/// in flight. Publishing against the stale version would attach a judgement to
/// text nobody read.
#[tokio::test]
async fn a_verdict_for_a_superseded_version_is_not_published() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    derive_page(&db, "p1", 1).await;
    {
        let conn = db.conn.lock().await;
        conn.execute("UPDATE pages SET version = 2 WHERE id = 'p1'", ())
            .await
            .unwrap();
    }

    assert!(matches!(
        db.evaluate_page_support("p1", 1).await.unwrap(),
        SupportOutcome::NoPublish { .. }
    ));
    let job = lease_page(&db, "p1", 1, "worker").await;
    assert!(
        !db.finalize_page_support("p1", 1, &job, "worker", &SupportOutcome::Supported)
            .await
            .unwrap(),
        "finalization re-checks the version under its own transaction"
    );
}

/// The axes are independent. A machine verdict may never set `human_reviewed`,
/// and may not disturb an approval that names the version being published.
#[tokio::test]
async fn publishing_support_leaves_a_current_human_review_intact() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 1).await;
    support_claim(&db, "p1", &revisions[0], 0.9).await;
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "INSERT INTO page_truth_state
                 (page_id, page_version, support_status, evaluated_at, human_reviewed,
                  reviewed_page_version, reviewed_page_digest, updated_at)
             VALUES ('p1', 1, 'provisional', NULL, 1, 1, 'digest', 0)",
            (),
        )
        .await
        .unwrap();
    }

    let job = lease_page(&db, "p1", 1, "worker").await;
    db.finalize_page_support("p1", 1, &job, "worker", &SupportOutcome::Supported)
        .await
        .unwrap();
    let (status, _, reviewed) = truth_row(&db, "p1").await.unwrap();
    assert_eq!(status, "supported");
    assert_eq!(reviewed, 1, "a current approval survives a support write");
}

// ---------------------------------------------------------------------------
// Publication is bound to the lease and to evidence that has not moved
// ---------------------------------------------------------------------------

/// The reclaim scenario end to end. Worker A evaluates the page as Supported and
/// stalls past its lease. B reclaims it, finds the support edge retracted,
/// evaluates Refuted, and publishes. A wakes and tries to publish its stale
/// Supported — and the PAGE VERSION NEVER MOVED, so the re-read `finalize` used
/// to do could not tell the two apart. Its `finish` failing afterwards is no
/// comfort: the truth row is already wrong by then.
#[tokio::test]
async fn a_reclaimed_job_cannot_publish_its_old_owners_verdict() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 1).await;
    support_claim(&db, "p1", &revisions[0], 0.9).await;

    let job = db
        .lease_next_derivation_job("worker-a")
        .await
        .unwrap()
        .unwrap();
    carry_seeded_judgements(&db, "p1", 1).await;
    let stale = db.evaluate_page_support("p1", 1).await.unwrap();
    assert_eq!(stale, SupportOutcome::Supported);

    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE claim_derivation_jobs SET lease_expires_at = 1 WHERE job_id = ?1",
            libsql::params![job.job_id.clone()],
        )
        .await
        .unwrap();
        conn.execute(
            "UPDATE edges SET valid_until = 1 WHERE edge_type = 'supports'",
            (),
        )
        .await
        .unwrap();
    }
    let reclaimed = db
        .lease_next_derivation_job("worker-b")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(reclaimed.job_id, job.job_id);
    carry_seeded_judgements(&db, "p1", 1).await;
    let fresh = db.evaluate_page_support("p1", 1).await.unwrap();
    assert!(
        matches!(fresh, SupportOutcome::Refuted { .. }),
        "got {fresh:?}"
    );
    assert!(db
        .finalize_page_support("p1", 1, &reclaimed, "worker-b", &fresh)
        .await
        .unwrap());

    assert!(
        !db.finalize_page_support("p1", 1, &job, "worker-a", &stale)
            .await
            .unwrap(),
        "a worker whose lease was reclaimed must not publish over its successor"
    );
    let (status, evaluated, _) = truth_row(&db, "p1").await.unwrap();
    assert_eq!(status, "provisional", "worker B's verdict has to survive");
    assert!(evaluated.is_some());
}

/// The harder reclaim: both runs use the same stable owner string and the
/// evidence stays byte-identical, so owner, job id, page version, and verdict
/// all agree. Only the lease generation can distinguish the stale computation
/// from the successor that currently holds the job.
#[tokio::test]
async fn a_same_owner_reclaim_rejects_the_old_runs_publication() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 1).await;
    support_claim(&db, "p1", &revisions[0], 0.9).await;

    let stale = db
        .lease_next_derivation_job("worker")
        .await
        .unwrap()
        .unwrap();
    carry_seeded_judgements(&db, "p1", 1).await;
    let stale_outcome = db.evaluate_page_support("p1", 1).await.unwrap();
    assert_eq!(stale_outcome, SupportOutcome::Supported);
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE claim_derivation_jobs SET lease_expires_at = 1 WHERE job_id = ?1",
            libsql::params![stale.job_id.clone()],
        )
        .await
        .unwrap();
    }

    let current = db
        .lease_next_derivation_job("worker")
        .await
        .unwrap()
        .unwrap();
    assert_ne!(stale.run_generation, current.run_generation);
    carry_seeded_judgements(&db, "p1", 1).await;
    let current_outcome = db.evaluate_page_support("p1", 1).await.unwrap();
    assert_eq!(current_outcome, stale_outcome);

    assert!(
        !db.finalize_page_support("p1", 1, &stale, "worker", &stale_outcome)
            .await
            .unwrap(),
        "the old run must not publish through a same-owner successor's lease"
    );
    assert!(truth_row(&db, "p1").await.is_none());
    assert!(db
        .finalize_page_support("p1", 1, &current, "worker", &current_outcome)
        .await
        .unwrap());
    assert_eq!(truth_row(&db, "p1").await.unwrap().0, "supported");
}

/// The lease guard, isolated. Here the evidence never changes, so the verdict
/// recomputed at finalization is byte-identical to the one worker A holds —
/// nothing but the job state can refuse this write. A parked job is retired for
/// a human; its former worker publishing into it is the same intrusion as the
/// reclaim case with none of the other signals present.
#[tokio::test]
async fn a_parked_jobs_former_worker_cannot_publish() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 1).await;
    support_claim(&db, "p1", &revisions[0], 0.9).await;

    let job = db
        .lease_next_derivation_job("worker")
        .await
        .unwrap()
        .unwrap();
    carry_seeded_judgements(&db, "p1", 1).await;
    let outcome = db.evaluate_page_support("p1", 1).await.unwrap();
    assert_eq!(outcome, SupportOutcome::Supported);

    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE claim_derivation_jobs SET attempts = ?1, lease_expires_at = 1",
            libsql::params![MAX_ATTEMPTS],
        )
        .await
        .unwrap();
    }
    assert_eq!(db.park_exhausted_derivation_jobs().await.unwrap(), 1);

    assert!(
        !db.finalize_page_support("p1", 1, &job, "worker", &outcome)
            .await
            .unwrap(),
        "a parked job has no holder, so nobody may publish through it"
    );
    assert!(
        truth_row(&db, "p1").await.is_none(),
        "and the page keeps the state parking promised to leave alone"
    );
}

/// The evidence guard, isolated. The worker holds a live, unexpired lease the
/// whole time — the job state is impeccable — but a support edge was retracted
/// between phase 2 and phase 3. The page version cannot see that: support moves
/// without the prose moving, which is exactly why re-reading the version was
/// never enough to make a publication safe.
#[tokio::test]
async fn evidence_that_moved_after_evaluation_is_not_published() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 1).await;
    support_claim(&db, "p1", &revisions[0], 0.9).await;

    let job = db
        .lease_next_derivation_job("worker")
        .await
        .unwrap()
        .unwrap();
    carry_seeded_judgements(&db, "p1", 1).await;
    let outcome = db.evaluate_page_support("p1", 1).await.unwrap();
    assert_eq!(outcome, SupportOutcome::Supported);

    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE edges SET valid_until = 1 WHERE edge_type = 'supports'",
            (),
        )
        .await
        .unwrap();
    }

    assert!(
        !db.finalize_page_support("p1", 1, &job, "worker", &outcome)
            .await
            .unwrap(),
        "a verdict whose evidence is gone must not be published on the strength \
         of an unchanged page version"
    );
    assert!(truth_row(&db, "p1").await.is_none());
}

/// F2's full surface, not merely its easy half. The extractor writes two claims
/// and the judge reaches only one of them — a timeout, a deferral, a malformed
/// response, anything that ends without a score. The judged one fell short, so
/// there IS a real verdict in this inventory, and the tempting reading is
/// "something was judged and failed, so the page is Refuted". It is not: one
/// unscored claim means the derivation never ran to completion over this page,
/// and Refuted stamps `evaluated_at` and costs the page its file for a gap in
/// OUR pipeline. The unjudged claim has to win.
#[tokio::test]
async fn a_judged_short_claim_beside_an_unjudged_one_is_unevaluated() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 2).await;
    judged_but_short(&db, &revisions[0]).await;
    // revisions[1] is never scored at all — no edge, no cache row.

    let outcome = db.evaluate_page_support("p1", 1).await.unwrap();
    let SupportOutcome::Unevaluated { ref reason } = outcome else {
        panic!("an incomplete derivation is not a verdict; got {outcome:?}");
    };
    assert!(
        reason.contains("never been scored"),
        "the stored reason must name the gap rather than the shortfall: {reason}"
    );

    let job = lease_page(&db, "p1", 1, "worker").await;
    db.finalize_page_support("p1", 1, &job, "worker", &outcome)
        .await
        .unwrap();
    assert!(
        truth_row(&db, "p1").await.unwrap().1.is_none(),
        "an incomplete derivation must never cost the page its file"
    );
}

/// F4. `page_truth_state` holds one row per PAGE, so a v1 verdict's timestamp
/// outliving a v2 publish answers "has this been judged" about text that is
/// gone. v2 here is genuinely unjudged, and keeping v1's stamp exposes it as
/// Unsupported and takes its file — the precise inversion of what `Unevaluated`
/// exists to give.
#[tokio::test]
async fn an_unevaluated_publish_clears_the_previous_verdicts_timestamp() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 1).await;
    judged_but_short(&db, &revisions[0]).await;

    let refuted = db.evaluate_page_support("p1", 1).await.unwrap();
    assert!(matches!(refuted, SupportOutcome::Refuted { .. }));
    let job1 = lease_page(&db, "p1", 1, "worker").await;
    assert!(db
        .finalize_page_support("p1", 1, &job1, "worker", &refuted)
        .await
        .unwrap());
    assert!(
        truth_row(&db, "p1").await.unwrap().1.is_some(),
        "the v1 verdict is a real one and does stamp"
    );

    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE pages SET version = 2, content = 'rewritten prose' WHERE id = 'p1'",
            (),
        )
        .await
        .unwrap();
    }

    let outcome = db.evaluate_page_support("p1", 2).await.unwrap();
    assert!(
        matches!(outcome, SupportOutcome::Unevaluated { .. }),
        "v2 has no marker at all; got {outcome:?}"
    );
    let job2 = lease_page(&db, "p1", 2, "worker").await;
    assert!(db
        .finalize_page_support("p1", 2, &job2, "worker", &outcome)
        .await
        .unwrap());

    let (status, evaluated, _) = truth_row(&db, "p1").await.unwrap();
    assert_eq!(status, "provisional");
    assert!(
        evaluated.is_none(),
        "v1's judgement must not answer for v2's text"
    );
}

/// F5. Attempts count at LEASE time, so the worker holding the last one sits at
/// `attempts == MAX_ATTEMPTS` from its first instant. Parking on the count alone
/// retires a run that is healthy and still going: four daemon restarts followed
/// by one good run is an ordinary history, not a poison item, and the worker
/// then discovers its work was thrown away when its finish silently fails.
#[tokio::test]
async fn a_healthy_fifth_lease_is_not_parked() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    for _ in 0..MAX_ATTEMPTS - 1 {
        let job = db
            .lease_next_derivation_job("worker")
            .await
            .unwrap()
            .unwrap();
        db.release_derivation_job(&job, "worker", "daemon restart")
            .await
            .unwrap();
    }
    let job = db
        .lease_next_derivation_job("worker")
        .await
        .unwrap()
        .expect("the last attempt is still claimable");
    assert_eq!(job_for(&db, "p1").await.unwrap().2, MAX_ATTEMPTS);

    assert_eq!(
        db.park_exhausted_derivation_jobs().await.unwrap(),
        0,
        "an unexpired lease is a run in progress, not an exhausted job"
    );
    assert_eq!(job_for(&db, "p1").await.unwrap().1, "leased");
    assert!(
        db.finish_derivation_job(&job, "worker").await.unwrap(),
        "and the healthy run's finish must still be accepted"
    );
}

/// The other half of F5, so the fix cannot degrade into "never park a leased
/// job". Once the last lease expires it IS a crash, and parking is exactly
/// right — otherwise a job that dies on its final attempt is claimable by
/// nobody and parked by nothing, and sits leased forever.
#[tokio::test]
async fn an_expired_fifth_lease_still_parks() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    for _ in 0..MAX_ATTEMPTS - 1 {
        let job = db
            .lease_next_derivation_job("worker")
            .await
            .unwrap()
            .unwrap();
        db.release_derivation_job(&job, "worker", "boom")
            .await
            .unwrap();
    }
    db.lease_next_derivation_job("worker")
        .await
        .unwrap()
        .unwrap();
    {
        let conn = db.conn.lock().await;
        conn.execute("UPDATE claim_derivation_jobs SET lease_expires_at = 1", ())
            .await
            .unwrap();
    }

    assert_eq!(db.park_exhausted_derivation_jobs().await.unwrap(), 1);
    assert_eq!(job_for(&db, "p1").await.unwrap().1, "parked");
}

#[tokio::test]
async fn an_expired_final_lease_is_auto_parked_before_reclaim() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    for _ in 0..MAX_ATTEMPTS - 1 {
        let job = db
            .lease_next_derivation_job("worker")
            .await
            .unwrap()
            .unwrap();
        db.release_derivation_job(&job, "worker", "daemon restart")
            .await
            .unwrap();
    }
    let final_lease = db
        .lease_next_derivation_job("worker")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(job_for(&db, "p1").await.unwrap().2, MAX_ATTEMPTS);
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE claim_derivation_jobs SET lease_expires_at = 1 WHERE job_id = ?1",
            libsql::params![final_lease.job_id.as_str()],
        )
        .await
        .unwrap();
    }

    assert!(
        db.lease_next_derivation_job("rescuer")
            .await
            .unwrap()
            .is_none(),
        "an expired max-attempt lease must be parked, not handed to a new run"
    );
    assert_eq!(job_for(&db, "p1").await.unwrap().1, "parked");
}

// ---------------------------------------------------------------------------
// Rows 13 and 15: evidence that stops qualifying after the derivation ran
// ---------------------------------------------------------------------------

/// Row 13. A supporting memory is deleted. Its edge is retracted, but nothing
/// used to touch the page: `page_truth_state` kept saying `supported` and no job
/// was created, so the page could stay supported indefinitely on evidence that
/// no longer exists. Queueing the work and demoting the claim answer different
/// questions — when will we know again, and what do we say meanwhile — so both
/// have to happen, and the demotion has to be synchronous because on a substrate
/// with no producer "eventually" is never.
#[tokio::test]
async fn withdrawing_evidence_demotes_the_page_and_requeues_it() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 1).await;
    support_claim(&db, "p1", &revisions[0], 0.9).await;

    let job = lease_page(&db, "p1", 1, "worker").await;
    let outcome = db.evaluate_page_support("p1", 1).await.unwrap();
    assert_eq!(outcome, SupportOutcome::Supported);
    assert!(db
        .finalize_page_support("p1", 1, &job, "worker", &outcome)
        .await
        .unwrap());
    assert!(db.finish_derivation_job(&job, "worker").await.unwrap());
    assert_eq!(truth_row(&db, "p1").await.unwrap().0, "supported");
    assert_eq!(job_for(&db, "p1").await.unwrap().1, "done");

    {
        let conn = db.conn.lock().await;
        conn.execute(
            "DELETE FROM memories WHERE source_id = ?1",
            libsql::params![format!("mem_{}", revisions[0])],
        )
        .await
        .unwrap();
    }

    let (status, evaluated, _) = truth_row(&db, "p1").await.unwrap();
    assert_eq!(
        status, "provisional",
        "evidence that is gone costs the page its verdict at once"
    );
    assert!(
        evaluated.is_none(),
        "and costs it as Unevaluated — we stopped asserting support, we did not \
         assert the prose is wrong"
    );
    assert_eq!(
        job_for(&db, "p1").await.unwrap().1,
        "pending",
        "with nothing back on the queue, nothing would ever re-judge it"
    );
}

/// Row 15. A page whose evidence no longer clears the LIVE bar keeps a current
/// marker and a `done` job, so a scan that looks only at markers walks straight
/// past it and the page stays `supported` forever. `SUPPORT_THRESHOLD` is a
/// compile-time constant, so this reaches the same state from the other side: to
/// the scan, an edge whose score falls below the bar and a bar that rises above
/// the score are the same row.
#[tokio::test]
async fn support_that_no_longer_clears_the_bar_is_re_enqueued() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 1).await;
    support_claim(&db, "p1", &revisions[0], SUPPORT_THRESHOLD + 0.02).await;

    let job = lease_page(&db, "p1", 1, "worker").await;
    let outcome = db.evaluate_page_support("p1", 1).await.unwrap();
    assert_eq!(outcome, SupportOutcome::Supported);
    assert!(db
        .finalize_page_support("p1", 1, &job, "worker", &outcome)
        .await
        .unwrap());
    assert!(db.finish_derivation_job(&job, "worker").await.unwrap());
    assert_eq!(job_for(&db, "p1").await.unwrap().1, "done");

    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE edges SET payload = ?1 WHERE edge_type = 'supports'",
            libsql::params![format!(
                "{{\"score\":{},\"threshold_at_write\":0.75}}",
                SUPPORT_THRESHOLD - 0.05
            )],
        )
        .await
        .unwrap();
    }

    assert_eq!(
        db.enqueue_stale_derivation_jobs(100).await.unwrap(),
        1,
        "the scan has to see edge validity and the live bar, not only the marker"
    );
    assert_eq!(job_for(&db, "p1").await.unwrap().1, "pending");
}

/// A signed SQL LIMIT is not a bound: SQLite treats a negative value as "no
/// limit", while zero can disguise a caller bug as a clean drain. Both are
/// rejected before the transaction or its temp batch can mutate anything.
#[tokio::test]
async fn non_positive_sweep_limits_reject_without_mutation() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 1).await;
    support_claim(&db, "p1", &revisions[0], SUPPORT_THRESHOLD + 0.02).await;
    publish_supported(&db, "p1").await;
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE edges SET payload = json_set(payload, '$.score', ?1)
              WHERE edge_type = 'supports'",
            libsql::params![SUPPORT_THRESHOLD - 0.05],
        )
        .await
        .unwrap();
    }

    let before_job = job_for(&db, "p1").await.unwrap();
    let before_truth = truth_row(&db, "p1").await.unwrap();
    for invalid in [-1, 0] {
        assert!(
            matches!(
                db.enqueue_stale_derivation_jobs(invalid).await,
                Err(crate::WenlanError::Validation(_))
            ),
            "limit {invalid} must be rejected"
        );
        assert_eq!(job_for(&db, "p1").await.unwrap(), before_job);
        assert_eq!(truth_row(&db, "p1").await.unwrap(), before_truth);
    }
}

/// Stale-marker and support-drift work share one overall budget. Capturing a
/// full batch from each class would advance four pages under a limit of two;
/// ordering the distinct union first advances exactly two and leaves both kinds
/// of tail byte-for-byte ready for the next turn.
#[tokio::test]
async fn mixed_disjoint_work_shares_one_bound_and_leaves_the_tail_unchanged() {
    let (db, _temp) = db_with_queue().await;
    for page_id in ["a_stale", "c_stale"] {
        add_page(&db, page_id).await;
        mark_derived(&db, page_id, 1, EXTRACTOR_VERSION - 1).await;
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE claim_derivation_jobs SET status = 'done' WHERE page_id = ?1",
            libsql::params![page_id],
        )
        .await
        .unwrap();
    }
    for page_id in ["b_drift", "d_drift"] {
        add_page(&db, page_id).await;
        let revisions = derive_page(&db, page_id, 1).await;
        support_claim(&db, page_id, &revisions[0], SUPPORT_THRESHOLD + 0.02).await;
        publish_supported(&db, page_id).await;
    }
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE edges SET payload = json_set(payload, '$.score', ?1)
              WHERE edge_type = 'supports'",
            libsql::params![SUPPORT_THRESHOLD - 0.05],
        )
        .await
        .unwrap();
    }

    assert_eq!(db.enqueue_stale_derivation_jobs(2).await.unwrap(), 2);
    assert_eq!(job_for(&db, "a_stale").await.unwrap().1, "pending");
    assert_eq!(job_for(&db, "b_drift").await.unwrap().1, "pending");
    assert_eq!(truth_row(&db, "b_drift").await.unwrap().0, "provisional");

    assert_eq!(job_for(&db, "c_stale").await.unwrap().1, "done");
    assert_eq!(job_for(&db, "d_drift").await.unwrap().1, "done");
    assert_eq!(truth_row(&db, "d_drift").await.unwrap().0, "supported");
}

/// One page may be stale by extractor marker and drifted by evidence at once.
/// The batch and its return value count pages, not candidate-class matches.
#[tokio::test]
async fn overlapping_stale_and_drift_work_is_counted_once() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "overlap").await;
    let revisions = derive_page(&db, "overlap", 1).await;
    support_claim(&db, "overlap", &revisions[0], SUPPORT_THRESHOLD + 0.02).await;
    publish_supported(&db, "overlap").await;
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE claim_derivation_markers SET extractor_version = ?1
              WHERE page_id = 'overlap'",
            libsql::params![EXTRACTOR_VERSION - 1],
        )
        .await
        .unwrap();
        conn.execute(
            "UPDATE edges SET payload = json_set(payload, '$.score', ?1)
              WHERE edge_type = 'supports'",
            libsql::params![SUPPORT_THRESHOLD - 0.05],
        )
        .await
        .unwrap();
    }

    assert_eq!(db.enqueue_stale_derivation_jobs(1).await.unwrap(), 1);
    assert_eq!(job_for(&db, "overlap").await.unwrap().1, "pending");
    assert_eq!(truth_row(&db, "overlap").await.unwrap().0, "provisional");
    assert_eq!(db.enqueue_stale_derivation_jobs(1).await.unwrap(), 0);
}

/// Row 15's runtime bound must constrain the SAME set for enqueue and
/// demotion. If the enqueue takes 25 rows but the demotion takes all 26, the
/// last page is no longer `supported`, so the next drift scan cannot find it;
/// it has no job either, and the continuation incorrectly reports completion.
#[tokio::test]
async fn drifted_support_converges_past_the_runtime_batch_without_losing_the_tail() {
    const RUNTIME_BATCH: i64 = 25;
    const PAGE_COUNT: usize = RUNTIME_BATCH as usize + 1;

    let (db, _temp) = db_with_queue().await;
    for i in 0..PAGE_COUNT {
        let page_id = format!("drift_tail_{i:02}");
        add_page(&db, &page_id).await;
        let revisions = derive_page(&db, &page_id, 1).await;
        support_claim(&db, &page_id, &revisions[0], SUPPORT_THRESHOLD + 0.02).await;
        publish_supported(&db, &page_id).await;
    }

    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE edges SET payload = json_set(payload, '$.score', ?1)
              WHERE edge_type = 'supports'",
            libsql::params![SUPPORT_THRESHOLD - 0.05],
        )
        .await
        .unwrap();
    }

    assert_eq!(
        db.enqueue_stale_derivation_jobs(RUNTIME_BATCH)
            .await
            .unwrap(),
        RUNTIME_BATCH as usize,
        "the first bounded turn may queue only its batch"
    );

    let untouched_tail = "drift_tail_25";
    assert_eq!(
        truth_row(&db, untouched_tail).await.unwrap().0,
        "supported",
        "the page outside the captured batch must keep its pre-turn truth state"
    );
    assert_eq!(
        job_for(&db, untouched_tail)
            .await
            .map(|(_, status, _)| status),
        Some("done".to_string()),
        "the page outside the captured batch must keep its done job until its turn"
    );

    {
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT COUNT(*)
                   FROM page_truth_state t
                  WHERE t.page_id LIKE 'drift_tail_%'
                    AND t.support_status = 'provisional'
                    AND NOT EXISTS (
                        SELECT 1 FROM claim_derivation_jobs j
                         WHERE j.page_id = t.page_id
                           AND j.page_version = t.page_version
                           AND j.status = 'pending'
                    )",
                (),
            )
            .await
            .unwrap();
        let demoted_without_pending: i64 = rows.next().await.unwrap().unwrap().get(0).unwrap();
        assert_eq!(
            demoted_without_pending, 0,
            "a bounded sweep must never demote a page it did not durably queue"
        );
    }

    assert_eq!(
        db.enqueue_stale_derivation_jobs(RUNTIME_BATCH)
            .await
            .unwrap(),
        1,
        "the tail must remain discoverable for the next runtime turn"
    );
    assert_eq!(
        db.enqueue_stale_derivation_jobs(RUNTIME_BATCH)
            .await
            .unwrap(),
        0,
        "only after the tail is queued may the continuation report completion"
    );

    for i in 0..PAGE_COUNT {
        let page_id = format!("drift_tail_{i:02}");
        assert_eq!(truth_row(&db, &page_id).await.unwrap().0, "provisional");
        assert_eq!(job_for(&db, &page_id).await.unwrap().1, "pending");
    }
}

/// F6. One bounded sweep is a truncation unless something continues it. The
/// migration queues a batch and stops; every page past it was reachable only by
/// being edited, so on a vault larger than the batch the "backlog scan"
/// converged on a subset and the queue was quietly partial on exactly the
/// installs it exists for.
#[tokio::test]
async fn the_backlog_drain_converges_past_a_single_batch() {
    let (db, _temp) = db_before_the_worker().await;
    for i in 0..7 {
        add_page(&db, &format!("p{i}")).await;
    }
    install_triggers(&db).await;

    assert_eq!(
        db.enqueue_stale_derivation_jobs(2).await.unwrap(),
        2,
        "one bounded pass reaches two of the seven"
    );
    assert_eq!(
        db.drain_stale_derivation_jobs(2).await.unwrap(),
        5,
        "the drain has to finish the other five"
    );
    for i in 0..7 {
        assert_eq!(job_for(&db, &format!("p{i}")).await.unwrap().1, "pending");
    }
}

// ---------------------------------------------------------------------------
// Row 8, N1, and row 6's judging half: evidence and text that move afterwards
// ---------------------------------------------------------------------------

/// Another chunk of the document the fixture's evidence came from, so a test can
/// delete or rewrite the CITED chunk while the document itself survives.
async fn add_sibling_chunk(db: &MemoryDB, revision_id: &str, chunk_index: i64) {
    let conn = db.conn.lock().await;
    let source_id = format!("mem_{revision_id}");
    conn.execute(
        "INSERT INTO memories (id, content, source, source_id, title, chunk_index,
                               last_modified, chunk_type, space)
         SELECT ?1, 'other prose', 'memory', ?2, 'evidence', ?3, 0, 'text', space
           FROM memories WHERE source_id = ?2 LIMIT 1",
        libsql::params![
            format!("m_{source_id}_c{chunk_index}"),
            source_id,
            chunk_index
        ],
    )
    .await
    .unwrap();
}

/// A real candidate chunk with no support edge. This is the positive control
/// for a completed judgement that genuinely fell short: the locator resolves,
/// but no active edge says the candidate supports the claim.
async fn add_live_candidate_chunk(db: &MemoryDB, revision_id: &str, chunk_index: i64) {
    let conn = db.conn.lock().await;
    let space: String = {
        let mut rows = conn
            .query(
                "SELECT p.space
                   FROM pages p
                   JOIN page_version_claims pvc ON pvc.page_id = p.id
                  WHERE pvc.claim_revision_id = ?1 LIMIT 1",
                libsql::params![revision_id],
            )
            .await
            .unwrap();
        rows.next().await.unwrap().unwrap().get(0).unwrap()
    };
    let source_id = format!("mem_{revision_id}");
    conn.execute(
        "INSERT OR IGNORE INTO memories
             (id, content, source, source_id, title, chunk_index,
              last_modified, chunk_type, space)
         VALUES (?1, ?2, 'memory', ?3, 'candidate', ?4, 0, 'text', ?5)",
        libsql::params![
            format!("candidate_{source_id}_{chunk_index}"),
            EVIDENCE_TEXT,
            source_id,
            chunk_index,
            space,
        ],
    )
    .await
    .unwrap();
}

/// Whether the read surface still tells callers this page is supported.
async fn exposed_support(db: &MemoryDB, page_id: &str) -> crate::truth_contract::Support {
    db.page_truth_states(&[page_id.to_string()])
        .await
        .unwrap()
        .get(page_id)
        .copied()
        .unwrap_or_default()
        .support
}

/// Drive one page from queued to published `supported`, the state every test
/// below starts from.
async fn publish_supported(db: &MemoryDB, page_id: &str) {
    let job = lease_page(db, page_id, 1, "worker").await;
    let outcome = db.evaluate_page_support(page_id, 1).await.unwrap();
    assert_eq!(outcome, SupportOutcome::Supported);
    assert!(db
        .finalize_page_support(page_id, 1, &job, "worker", &outcome)
        .await
        .unwrap());
    assert!(db.finish_derivation_job(&job, "worker").await.unwrap());
    assert_eq!(truth_row(db, page_id).await.unwrap().0, "supported");
    assert_eq!(
        exposed_support(db, page_id).await,
        crate::truth_contract::Support::Supported
    );
}

/// Row 8, the synchronous half. Rewriting a page's text WITHOUT bumping its
/// version leaves `page_truth_state.page_version` equal to `pages.version`, so
/// the version comparison at the read surface sees nothing wrong and the page
/// goes on being served as supported prose it no longer contains. A queued job
/// does not help: on a substrate whose producer does not exist yet, the page
/// waits forever in the meantime.
#[tokio::test]
async fn an_in_place_page_rewrite_stops_the_page_exposing_supported() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 1).await;
    support_claim(&db, "p1", &revisions[0], 0.9).await;
    publish_supported(&db, "p1").await;

    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE pages SET content = 'different prose entirely' WHERE id = 'p1'",
            (),
        )
        .await
        .unwrap();
    }

    let (status, evaluated, _) = truth_row(&db, "p1").await.unwrap();
    assert_eq!(
        status, "provisional",
        "a page whose text was replaced under its own version must lose its stored verdict"
    );
    assert!(
        evaluated.is_none(),
        "and lose it as Unevaluated: our bookkeeping moved, the prose was not refuted"
    );
    assert_ne!(
        exposed_support(&db, "p1").await,
        crate::truth_contract::Support::Supported,
        "the point is what the READ surface says, not what is queued"
    );
}

/// Row 15, the synchronous half. The backlog scan finds a page whose evidence no
/// longer clears the live bar and re-queues it — but a queued job is a promise
/// about the future, and until it is kept the page keeps telling every caller
/// its claims are supported.
#[tokio::test]
async fn drifted_support_stops_the_page_exposing_supported() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 1).await;
    support_claim(&db, "p1", &revisions[0], SUPPORT_THRESHOLD + 0.02).await;
    publish_supported(&db, "p1").await;

    {
        let conn = db.conn.lock().await;
        conn.execute(
            // `json_set`, so the edge stays otherwise well-formed: this test is
            // about a score that fell under the bar, not about a payload that
            // lost the fields revalidation reads.
            "UPDATE edges SET payload = json_set(payload, '$.score', ?1)
              WHERE edge_type = 'supports'",
            libsql::params![SUPPORT_THRESHOLD - 0.05],
        )
        .await
        .unwrap();
    }

    assert_eq!(db.enqueue_stale_derivation_jobs(100).await.unwrap(), 1);
    assert_eq!(job_for(&db, "p1").await.unwrap().1, "pending");
    let (status, evaluated, _) = truth_row(&db, "p1").await.unwrap();
    assert_eq!(
        status, "provisional",
        "the scan has to demote the page it just re-queued, not only queue it"
    );
    assert!(evaluated.is_none());
    assert_ne!(
        exposed_support(&db, "p1").await,
        crate::truth_contract::Support::Supported
    );
}

/// Row 15's order argument is only a safety argument if BOTH statements land.
///
/// The enqueue must run before the demotion — being `supported` is part of what
/// makes a page drifted, so demoting first empties the drift query and the
/// re-derivation is never queued. That ordering makes the failure window
/// specific: a failure after the enqueue and before the demotion leaves the page
/// queued and still asserting `supported` over evidence that stopped qualifying,
/// which is precisely the fail-open state row 15 exists to close. Worse, the
/// caller sees an error and cannot tell that half of it committed.
///
/// So the pair has to be atomic. The demotion is made to fail here, and the
/// assertion is that the enqueue did not survive it — the page is exactly as it
/// was, and the error is the whole truth. Run as two autocommitted statements
/// the job row would be sitting there, which is the state under test.
#[tokio::test]
async fn a_failed_demotion_takes_its_enqueue_down_with_it() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 1).await;
    support_claim(&db, "p1", &revisions[0], SUPPORT_THRESHOLD + 0.02).await;
    publish_supported(&db, "p1").await;

    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE edges SET payload = json_set(payload, '$.score', ?1)
              WHERE edge_type = 'supports'",
            libsql::params![SUPPORT_THRESHOLD - 0.05],
        )
        .await
        .unwrap();
        // Break the demotion and nothing else. A storage-level refusal is the
        // honest shape of the failure -- a disk error, a lock, a constraint --
        // and it lands on the second statement of the pair, which is the only
        // ordering where the enqueue could be left behind.
        conn.execute_batch(
            "CREATE TRIGGER red_proof_demotion_fails
             BEFORE UPDATE OF support_status ON page_truth_state
             WHEN NEW.support_status = 'provisional'
             BEGIN SELECT RAISE(ABORT, 'demotion refused'); END;",
        )
        .await
        .unwrap();
    }

    // The page's job is `done` from the publication above. The sweep's drift arm
    // deletes that row and re-inserts a `pending` one, so `pending` afterwards is
    // this sweep's enqueue and nothing else's.
    assert_eq!(job_for(&db, "p1").await.unwrap().1, "done");

    let failed = db.enqueue_stale_derivation_jobs(100).await;
    assert!(
        failed.is_err(),
        "a demotion that cannot land must be reported, not swallowed"
    );

    assert_eq!(
        job_for(&db, "p1").await.unwrap().1,
        "done",
        "the enqueue must not outlive the demotion it was paired with -- a queued \
         page still asserting `supported` is the fail-open state row 15 closes"
    );
    assert_eq!(
        truth_row(&db, "p1").await.unwrap().0,
        "supported",
        "and nothing else moved either: the error is the whole truth"
    );
}

/// N1, the invalidation half. A support edge cites a CHUNK, but the retraction
/// and demotion triggers that shipped first fire only when the LAST row for a
/// `source_id` disappears — correct for what they guard, and blind to the update
/// and merge paths that delete a non-head chunk and keep the source head. Delete
/// the cited chunk 1 with chunk 0 still present and nothing used to happen at
/// all: the edge kept its `valid_until IS NULL`, and the page kept its verdict.
#[tokio::test]
async fn deleting_only_the_cited_chunk_demotes_the_page() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 1).await;
    support_claim_at_chunk(&db, "p1", &revisions[0], 0.9, 1).await;
    add_sibling_chunk(&db, &revisions[0], 0).await;
    publish_supported(&db, "p1").await;

    let source_id = format!("mem_{}", revisions[0]);
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "DELETE FROM memories WHERE source_id = ?1 AND chunk_index = 1",
            libsql::params![source_id.clone()],
        )
        .await
        .unwrap();
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM memories WHERE source_id = ?1",
                libsql::params![source_id],
            )
            .await
            .unwrap();
        let survivors: i64 = rows.next().await.unwrap().unwrap().get(0).unwrap();
        assert_eq!(
            survivors, 1,
            "the shape under test is a chunk dying while its document lives — a \
             whole-source trigger cannot see this"
        );
    }

    let (status, evaluated, _) = truth_row(&db, "p1").await.unwrap();
    assert_eq!(status, "provisional");
    assert!(evaluated.is_none());
    assert_ne!(
        exposed_support(&db, "p1").await,
        crate::truth_contract::Support::Supported
    );
    assert_eq!(
        job_for(&db, "p1").await.unwrap().1,
        "pending",
        "and it has to be re-judged, not merely un-asserted"
    );
}

/// N1, the same shape reached by rewriting the chunk instead of deleting it.
/// Every row the edge names still exists, so nothing about the document's
/// structure changed — but the span offsets now index into a different string,
/// which makes the citation false in exactly the way a reader cannot see.
#[tokio::test]
async fn rewriting_only_the_cited_chunk_demotes_the_page() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 1).await;
    support_claim_at_chunk(&db, "p1", &revisions[0], 0.9, 1).await;
    add_sibling_chunk(&db, &revisions[0], 0).await;
    publish_supported(&db, "p1").await;

    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE memories SET content = 'rewritten evidence'
              WHERE source_id = ?1 AND chunk_index = 1",
            libsql::params![format!("mem_{}", revisions[0])],
        )
        .await
        .unwrap();
    }

    assert_eq!(truth_row(&db, "p1").await.unwrap().0, "provisional");
    assert_ne!(
        exposed_support(&db, "p1").await,
        crate::truth_contract::Support::Supported
    );
}

/// N1's other half, and the reason the recompute-in-transaction design is safe
/// at all: evaluation itself must revalidate the evidence it asserts, so a page
/// cannot be re-published on a dead citation with no trigger having fired.
/// `valid_until IS NULL` means nobody retracted this edge — it has never meant
/// the bytes it names are still there.
///
/// The triggers are dropped deliberately. They are the first line of defence and
/// the teeth above prove they work; this proves the second line holds on its own,
/// which is what makes the two independent rather than one defence counted twice.
///
/// Precisely: two of the three are dropped outright below
/// (`m5_demote_support_on_chunk_delete`, `m5_demote_support_on_memory_delete`).
/// `m5_retract_support_on_memory_delete` stays installed and is left in place on
/// purpose — its last-chunk `NOT EXISTS` guard is structurally false while a
/// sibling chunk survives, so it cannot fire in the delete-chunk-1-keep-chunk-0
/// shape this test uses. The assertions below do not take that on trust: the
/// edge is asserted still active and the stored verdict still `supported`
/// immediately before evaluation runs, so whatever the surviving trigger did or
/// did not do, the demotion that follows is revalidation's alone.
#[tokio::test]
async fn a_dead_chunk_edge_cannot_republish_support() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 1).await;
    support_claim_at_chunk(&db, "p1", &revisions[0], 0.9, 1).await;
    add_sibling_chunk(&db, &revisions[0], 0).await;
    publish_supported(&db, "p1").await;

    {
        let conn = db.conn.lock().await;
        conn.execute_batch(
            "DROP TRIGGER IF EXISTS m5_demote_support_on_chunk_delete;
             DROP TRIGGER IF EXISTS m5_demote_support_on_memory_delete;",
        )
        .await
        .unwrap();
        conn.execute(
            "DELETE FROM memories WHERE source_id = ?1 AND chunk_index = 1",
            libsql::params![format!("mem_{}", revisions[0])],
        )
        .await
        .unwrap();
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM edges
                  WHERE edge_type = 'supports' AND valid_until IS NULL",
                (),
            )
            .await
            .unwrap();
        let live: i64 = rows.next().await.unwrap().unwrap().get(0).unwrap();
        assert_eq!(
            live, 1,
            "with the triggers gone the edge is still active — that is the state \
             evaluation has to survive"
        );
    }
    assert_eq!(
        truth_row(&db, "p1").await.unwrap().0,
        "supported",
        "and the stored verdict is untouched, so republication is genuinely on the table"
    );

    let outcome = db.evaluate_page_support("p1", 1).await.unwrap();
    assert!(
        matches!(outcome, SupportOutcome::NoPublish { .. }),
        "an active, above-threshold edge whose chunk is gone is malformed live \
         evidence, not support or a refutation; got {outcome:?}"
    );
    db.begin_support_reconcile_pass().await.unwrap();
    assert_eq!(db.reconcile_supported_pages(1000).await.unwrap().demoted, 1);
    assert_eq!(
        truth_row(&db, "p1").await.unwrap().0,
        "provisional",
        "the fail-closed reconciler must take inherited malformed support out \
         of circulation without publishing the malformed run"
    );
}

/// Row 6, the judging half — the failure a cache-row discriminator gets wrong.
/// `entailment_cache` is keyed by content digests alone, so it is global and
/// timeless: one candidate concluding leaves a row, and the claim then reads as
/// judged no matter how many other required candidates never came back. Judged
/// with no support is `Refuted`, which stamps `evaluated_at` and costs the page
/// its file — for evaluation that never finished.
#[tokio::test]
async fn a_candidate_this_run_never_concluded_blocks_publication() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 1).await;

    add_live_candidate_chunk(&db, &revisions[0], 0).await;
    attempt_on(
        &db,
        &revisions[0],
        &Candidate::at_chunk(&revisions[0], 0),
        "concluded",
    )
    .await;
    assert!(
        matches!(
            db.evaluate_page_support("p1", 1).await.unwrap(),
            SupportOutcome::Refuted { .. }
        ),
        "a run that concluded on every candidate and found nothing IS a verdict"
    );

    add_live_candidate_chunk(&db, &revisions[0], 1).await;
    attempt_on(
        &db,
        &revisions[0],
        &Candidate::at_chunk(&revisions[0], 1),
        "deferred",
    )
    .await;
    let outcome = db.evaluate_page_support("p1", 1).await.unwrap();
    assert!(
        matches!(outcome, SupportOutcome::NoPublish { .. }),
        "one candidate short and another still out is an unfinished run, not a \
         refutation; got {outcome:?}"
    );
}

/// The same distinction from the other side: absence of attempt rows is
/// never-judged, and never-judged may not read as judged-and-found-wanting. This
/// is what keeps a substrate with no producer at all — every page, zero attempts
/// — on `Unevaluated` instead of refuting the entire vault.
#[tokio::test]
async fn a_claim_with_no_attempt_rows_is_never_judged() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    derive_page(&db, "p1", 1).await;

    assert!(
        matches!(
            db.evaluate_page_support("p1", 1).await.unwrap(),
            SupportOutcome::Unevaluated { .. }
        ),
        "nobody looked, so there is nothing to refute"
    );
}

// ---------------------------------------------------------------------------
// Condition 4's two halves: no candidate left hanging, and no candidate left
// unweighed. A run is complete or it publishes nothing.
// ---------------------------------------------------------------------------

/// The short-circuit. Condition 3 is satisfiable by ONE edge, so a loop that
/// stops at the first candidate that revalidates publishes `Supported` for a run
/// that is still out on another candidate — a real, above-threshold citation
/// standing in for evidence nobody finished weighing. The unfinished candidate
/// has to win over the finished one, or "every candidate was weighed" degrades
/// into "some candidate was weighed", which is not condition 4.
#[tokio::test]
async fn a_deferred_candidate_outranks_a_support_edge_that_revalidates() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 1).await;
    support_claim(&db, "p1", &revisions[0], 0.9).await;
    assert_eq!(
        db.evaluate_page_support("p1", 1).await.unwrap(),
        SupportOutcome::Supported,
        "the valid edge on its own is genuine support, so the refusal below is \
         about the deferral and nothing else"
    );

    attempt_on(
        &db,
        &revisions[0],
        &Candidate::named(&revisions[0], "a_second_passage"),
        "deferred",
    )
    .await;

    let outcome = db.evaluate_page_support("p1", 1).await.unwrap();
    let SupportOutcome::NoPublish { ref reason } = outcome else {
        panic!("a run still out on a candidate has not finished; got {outcome:?}");
    };
    assert!(
        reason.contains("incomplete") || reason.contains("did not finish"),
        "the stored reason must name the unfinished run: {reason}"
    );
}

/// Candidate identity is a LOCATION, not a digest. One document can hold the
/// same sentence twice, and the two occurrences are two separate obligations
/// even though their span digests are byte-identical.
///
/// Keyed on the digest alone, the run's conclusion about the first occurrence
/// silently discharges the second, and the page publishes `Supported` on
/// evidence nobody weighed. Keyed on the location, the second occurrence is
/// unaccounted-for until it is weighed on its own.
#[tokio::test]
async fn two_occurrences_of_one_sentence_are_two_obligations() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 1).await;
    // Edge + conclusion at offsets 0..14 of the evidence chunk.
    support_claim(&db, "p1", &revisions[0], 0.9).await;

    // The document actually says it twice. Both spans locate, and both hash to
    // the same digest — that is the whole point.
    let doubled = format!("{EVIDENCE_TEXT} {EVIDENCE_TEXT}");
    let second = Candidate::at_chunk(&revisions[0], 0)
        .at_offset(EVIDENCE_TEXT.len() as i64 + 1, doubled.len() as i64);
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE memories SET content = ?2 WHERE source_id = ?1",
            libsql::params![second.source_id.clone(), doubled.clone()],
        )
        .await
        .unwrap();
        // The provenance root is content-addressed over the WHOLE chunk, and
        // revalidation recomputes it, so the root has to be the root of the
        // document that holds the sentence twice. Leaving the single-occurrence
        // root behind would make both edges fail §5 and the test would pass for
        // the wrong reason — a broken provenance binding rather than an
        // unweighed candidate.
        conn.execute(
            "UPDATE provenance_roots SET identity_digest = ?2 WHERE root_id = ?1",
            libsql::params![
                "root_evidence_prose",
                crate::provenance::identity_digest("document_ingest", &doubled)
            ],
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO edges (edge_id, src_id, src_kind, dst_id, dst_kind, edge_type,
                                lineage, grounded, root_id, space, weight, payload,
                                provenance, operation_id, created_at, superseded_by,
                                valid_until)
             SELECT 'e_second_occurrence', src_id, src_kind, dst_id, dst_kind, edge_type,
                    lineage, grounded, root_id, space, weight,
                    json_set(json_set(payload, '$.span_start', ?2), '$.span_end', ?3),
                    provenance, operation_id, created_at, superseded_by, valid_until
               FROM edges WHERE edge_id = ?1",
            libsql::params![
                format!("e_{}", revisions[0]),
                second.span_start,
                second.span_end
            ],
        )
        .await
        .unwrap();
    }

    let outcome = db.evaluate_page_support("p1", 1).await.unwrap();
    let SupportOutcome::NoPublish { ref reason } = outcome else {
        panic!(
            "a conclusion about the first occurrence says nothing about the \
             second; got {outcome:?}"
        );
    };
    assert!(
        reason.contains("never weighed"),
        "the stored reason must name the unweighed evidence: {reason}"
    );

    // Weigh it, and the page is supported again — which is only reachable if the
    // two conclusions are two ROWS. Under digest-only identity the second write
    // would replace the first and the count would still be one.
    attempt_on(&db, &revisions[0], &second, "concluded").await;
    assert_eq!(
        db.evaluate_page_support("p1", 1).await.unwrap(),
        SupportOutcome::Supported
    );
    let recorded: i64 = {
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM claim_judgment_attempts WHERE claim_revision_id = ?1",
                libsql::params![revisions[0].clone()],
            )
            .await
            .unwrap();
        rows.next().await.unwrap().unwrap().get(0).unwrap()
    };
    assert_eq!(
        recorded, 2,
        "two occurrences must occupy two attempt rows, or one conclusion is \
         answering for both"
    );
}

/// The inventory half on its own: a live, above-threshold, fully revalidating
/// support edge that this run has no conclusion for at all. Nothing is wrong
/// with the edge — it is simply work the run never did, and publishing on it is
/// publishing on somebody else's homework.
#[tokio::test]
async fn a_support_edge_this_run_never_weighed_blocks_publication() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 1).await;
    support_claim(&db, "p1", &revisions[0], 0.9).await;
    {
        let conn = db.conn.lock().await;
        conn.execute("DELETE FROM claim_judgment_attempts", ())
            .await
            .unwrap();
    }

    let outcome = db.evaluate_page_support("p1", 1).await.unwrap();
    assert!(
        matches!(outcome, SupportOutcome::NoPublish { .. }),
        "an unaccounted-for candidate is an unfinished run, not support; got \
         {outcome:?}"
    );
}

/// Run identity, through the real lease. `(page_id, page_version)` names a JOB,
/// which outlives every attempt made on it: a reclaim or a retry produces a new
/// run against the same key. Without a generation the new run inherits the old
/// one's conclusions and can publish work it never did — the same defect as the
/// demotion path deleting attempt rows a leased worker then repopulates.
///
/// The old rows must SURVIVE the boundary. They are the only durable record of
/// what the previous attempt concluded; evaluation ignores them, which is a
/// different thing from destroying them while a stalled worker may still be
/// writing.
#[tokio::test]
async fn a_retried_job_does_not_inherit_the_previous_runs_conclusions() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 1).await;
    support_claim(&db, "p1", &revisions[0], 0.9).await;

    let first = db
        .lease_next_derivation_job("worker-a")
        .await
        .unwrap()
        .unwrap();
    carry_seeded_judgements(&db, "p1", 1).await;
    assert_eq!(
        db.evaluate_page_support("p1", 1).await.unwrap(),
        SupportOutcome::Supported
    );

    db.release_derivation_job(&first, "worker-a", "daemon restart")
        .await
        .unwrap();
    let retry = db
        .lease_next_derivation_job("worker-b")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(retry.job_id, first.job_id, "same job, second attempt");
    assert_ne!(
        retry.run_generation, first.run_generation,
        "a fresh attempt is a fresh run, or there is nothing to tell them apart"
    );

    let outcome = db.evaluate_page_support("p1", 1).await.unwrap();
    assert!(
        matches!(outcome, SupportOutcome::NoPublish { .. }),
        "the retry owes its own conclusion on evidence it has not weighed; got \
         {outcome:?}"
    );

    let survived: i64 = {
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM claim_judgment_attempts WHERE run_generation = ?1",
                libsql::params![first.run_generation],
            )
            .await
            .unwrap();
        rows.next().await.unwrap().unwrap().get(0).unwrap()
    };
    assert_eq!(
        survived, 1,
        "the previous run's record must outlive the run itself"
    );
}

// ---------------------------------------------------------------------------
// The startup reconciler, and the rename that no trigger can see
// ---------------------------------------------------------------------------

/// Blocker 3. A stored `supported` is a claim about the LIVE §1 conditions, and
/// the boot-time pass is what re-proves them. Migration 105's threshold scan can
/// only see scores; the conditions it cannot see are exactly the ones that go
/// stale across a restart — a marker that never landed, an extractor bump, prose
/// rewritten in place.
///
/// Here the marker is gone. `evaluate_page_support` calls that `Unevaluated` —
/// condition 1 unproven — so a reconciler that re-proves §1 demotes, and one
/// that defaults a missing marker to agreement serves the stale verdict forever,
/// because nothing else will ever come along to disturb it.
#[tokio::test]
async fn the_reconciler_demotes_a_page_whose_derivation_marker_is_gone() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 1).await;
    support_claim(&db, "p1", &revisions[0], 0.9).await;
    publish_supported(&db, "p1").await;

    {
        let conn = db.conn.lock().await;
        conn.execute(
            "DELETE FROM claim_derivation_markers WHERE page_id = 'p1'",
            (),
        )
        .await
        .unwrap();
    }
    assert_eq!(
        truth_row(&db, "p1").await.unwrap().0,
        "supported",
        "nothing has disturbed the stored verdict, so the demotion below is the \
         reconciler's doing and not a trigger's"
    );

    assert_eq!(db.reconcile_supported_pages(1000).await.unwrap().demoted, 1);
    assert_eq!(truth_row(&db, "p1").await.unwrap().0, "provisional");
    assert_eq!(
        exposed_support(&db, "p1").await,
        crate::truth_contract::Support::Unevaluated,
        "and the read surface stops asserting support"
    );
    assert_eq!(
        job_for(&db, "p1").await.unwrap().1,
        "pending",
        "the page is queued for the derivation that would settle it"
    );
}

/// The same pass against the condition-3 half — the one no SQL scan can reach.
/// The cited chunk is deleted, so the span the verdict named cannot be read back
/// out of live bytes. A reconciler that re-runs the real predicate sees this; a
/// SQL restatement of "score >= threshold" does not, because the score is still
/// 0.9 and the edge is still active.
#[tokio::test]
async fn the_reconciler_demotes_a_page_whose_cited_span_no_longer_exists() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 1).await;
    support_claim_at_chunk(&db, "p1", &revisions[0], 0.9, 3).await;
    add_sibling_chunk(&db, &revisions[0], 0).await;
    publish_supported(&db, "p1").await;

    {
        let conn = db.conn.lock().await;
        // Drop the cited chunk WITHOUT letting the invalidation trigger fire, so
        // the reconciler is the only thing that can notice. This is the state a
        // restart inherits when the edit happened under an older binary.
        conn.execute_batch("DROP TRIGGER IF EXISTS m5_demote_support_on_chunk_delete;")
            .await
            .unwrap();
        conn.execute(
            "DELETE FROM memories WHERE source_id = ?1 AND chunk_index = 3",
            libsql::params![format!("mem_{}", revisions[0])],
        )
        .await
        .unwrap();
    }
    assert_eq!(truth_row(&db, "p1").await.unwrap().0, "supported");

    assert_eq!(db.reconcile_supported_pages(1000).await.unwrap().demoted, 1);
    assert_eq!(truth_row(&db, "p1").await.unwrap().0, "provisional");
}

/// And the other direction, so the pass cannot degrade into "demote everything".
/// A page whose §1 conditions all still hold must come through untouched —
/// otherwise every boot costs every supported page its file and the reconciler
/// is an outage rather than a safety net.
#[tokio::test]
async fn the_reconciler_leaves_a_page_that_still_proves_out_alone() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 1).await;
    support_claim(&db, "p1", &revisions[0], 0.9).await;
    publish_supported(&db, "p1").await;

    assert_eq!(db.reconcile_supported_pages(1000).await.unwrap().demoted, 0);
    assert_eq!(truth_row(&db, "p1").await.unwrap().0, "supported");
    assert_eq!(
        exposed_support(&db, "p1").await,
        crate::truth_contract::Support::Supported
    );
}

/// The gap the attempt rows and the live edges cannot close between them.
///
/// Both are things a run PRODUCES. Candidate A concludes and leaves an attempt
/// row and a support edge; candidate B is one the worker died before reaching,
/// and nothing ever scored it, so it left neither. Evaluation reading only those
/// two sees one candidate, one conclusion, and no trace at all that a second
/// obligation existed — so a run that covered half its inventory publishes as
/// though it had covered all of it, which is exactly what row 6 forbids.
///
/// The existing teeth cover B-with-a-deferred-attempt-row and B-with-a-live-
/// edge. Neither covers B absent from both, because until the inventory table
/// there was nothing in the database that could tell anyone B was owed.
///
/// `inventory_count` is not this check. It counts CLAIMS, and B's claim is on
/// the membership roll and fully accounted for — with one of its two citations
/// silently dropped.
#[tokio::test]
async fn a_candidate_that_left_neither_an_attempt_nor_an_edge_blocks_publication() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 1).await;
    support_claim(&db, "p1", &revisions[0], 0.9).await;
    assert_eq!(
        db.evaluate_page_support("p1", 1).await.unwrap(),
        SupportOutcome::Supported,
        "candidate A alone proves out, which is what makes the next step a test"
    );

    add_live_candidate_chunk(&db, &revisions[0], 1).await;
    let unfinished = Candidate::at_chunk(&revisions[0], 1);
    owed_but_unjudged(&db, &revisions[0], &unfinished).await;

    let outcome = db.evaluate_page_support("p1", 1).await.unwrap();
    let SupportOutcome::NoPublish { ref reason } = outcome else {
        panic!(
            "a run that owed two conclusions and reached one has not finished, \
             whatever the one it reached said; got {outcome:?}"
        );
    };
    assert!(
        reason.contains("never concluded on"),
        "the stored reason must name the unfinished derivation: {reason}"
    );

    // Conclude on it and the page is supported again — so the refusal is about
    // the missing conclusion, not about the row's mere existence.
    attempt_on(&db, &revisions[0], &unfinished, "concluded").await;
    assert_eq!(
        db.evaluate_page_support("p1", 1).await.unwrap(),
        SupportOutcome::Supported
    );
}

/// A sealed inventory is authoritative in both directions. A conclusion that
/// names a candidate outside that set is not extra proof; it is evidence that
/// the run judged a different set from the one it declared complete.
///
/// Checking only `expected ⊆ concluded` misses this. With an empty sealed set
/// and one concluded attempt, the subset check is vacuously true and the page
/// becomes `Refuted` even though this run never declared the candidate as work
/// it owed. The two sets must agree exactly before either support or refutation
/// can publish.
#[tokio::test]
async fn a_conclusion_outside_the_sealed_inventory_blocks_publication() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 1).await;
    judged_but_short(&db, &revisions[0]).await;

    {
        let conn = db.conn.lock().await;
        conn.execute("DELETE FROM claim_run_candidates", ())
            .await
            .unwrap();
    }

    let outcome = db.evaluate_page_support("p1", 1).await.unwrap();
    let SupportOutcome::NoPublish { ref reason } = outcome else {
        panic!(
            "a run whose conclusions exceed its sealed inventory is internally incomplete; \
             got {outcome:?}"
        );
    };
    assert!(
        reason.contains("outside the sealed inventory"),
        "the stored reason must name the inventory mismatch: {reason}"
    );
}

/// A sealed inventory is a declaration about candidates that existed in the
/// evidence universe the run judged. Set equality alone is not enough: a
/// worker can seal a locator that resolves to no chunk (or no valid span),
/// write the matching concluded attempt, and otherwise look complete. Reading
/// that as `Refuted` turns malformed pipeline state into a verdict about the
/// page. Every sealed, concluded locator must still resolve to the live bytes
/// and digest it names before any verdict may publish.
#[tokio::test]
async fn a_concluded_locator_that_resolves_to_no_live_evidence_publishes_nothing() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 1).await;
    attempt_on(
        &db,
        &revisions[0],
        &Candidate::named(&revisions[0], "ghost_span"),
        "concluded",
    )
    .await;

    let outcome = db.evaluate_page_support("p1", 1).await.unwrap();
    let SupportOutcome::NoPublish { ref reason } = outcome else {
        panic!(
            "a matching seal and conclusion cannot turn an unresolvable locator into a verdict; \
             got {outcome:?}"
        );
    };
    assert!(
        reason.contains("locator") || reason.contains("evidence"),
        "the refusal must name the unresolvable candidate: {reason}"
    );
}

/// One level up: a run that died before recording ANY of its inventory.
///
/// Row counts cannot see this. A run whose claims genuinely drew no candidates
/// writes zero rows, and so does a run that leased and then crashed — and those
/// land on opposite verdicts. The seal is what separates them, so removing it
/// has to cost the publication.
///
/// Leased on purpose. Generation 0 is exempt from the seal because it is the
/// never-leased value and means no run happened at all, which the never-judged
/// path already answers; the requirement to have recorded an inventory attaches
/// to a run, and a run is what a lease creates.
#[tokio::test]
async fn a_run_that_never_recorded_its_inventory_publishes_nothing() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 1).await;
    support_claim(&db, "p1", &revisions[0], 0.9).await;
    publish_supported(&db, "p1").await;
    assert_eq!(
        db.evaluate_page_support("p1", 1).await.unwrap(),
        SupportOutcome::Supported
    );

    {
        let conn = db.conn.lock().await;
        conn.execute("DELETE FROM claim_run_inventories", ())
            .await
            .unwrap();
    }

    let outcome = db.evaluate_page_support("p1", 1).await.unwrap();
    let SupportOutcome::NoPublish { ref reason } = outcome else {
        panic!("a run that recorded no inventory has not finished; got {outcome:?}");
    };
    assert!(
        reason.contains("never sealed"),
        "the stored reason must name the unrecorded inventory: {reason}"
    );
}

/// The other half of bounding the startup pass.
///
/// The reconciler holds the single connection every foreground request needs and
/// spends a multi-query evaluator per page, per claim, per candidate, so it runs
/// in batches rather than scanning a whole vault before the daemon serves. That
/// is only safe if the part it has not reached stops being asserted — otherwise
/// truncating the scan serves exactly the stale `supported` the pass exists to
/// retract, and does it silently.
///
/// So the durable cursor is read by BOTH sides: the pass resumes from it, and
/// the read gate withholds everything above it.
#[tokio::test]
async fn a_bounded_pass_withholds_the_pages_it_has_not_reached() {
    let (db, _temp) = db_with_queue().await;
    for page_id in ["p1", "p2"] {
        add_page(&db, page_id).await;
        let revisions = derive_page(&db, page_id, 1).await;
        support_claim(&db, page_id, &revisions[0], 0.9).await;
        publish_supported(&db, page_id).await;
    }

    db.begin_support_reconcile_pass().await.unwrap();
    for page_id in ["p1", "p2"] {
        assert_eq!(
            exposed_support(&db, page_id).await,
            crate::truth_contract::Support::Unevaluated,
            "a pass that has re-proved nothing yet asserts nothing"
        );
        assert_eq!(
            truth_row(&db, page_id).await.unwrap().0,
            "supported",
            "and it withholds the verdict rather than destroying it -- the page \
             keeps its file and its stored state"
        );
    }

    let first = db.reconcile_supported_pages(1).await.unwrap();
    assert!(
        !first.complete,
        "a batch that filled its budget has not finished the vault"
    );
    assert_eq!(first.demoted, 0);
    assert_eq!(
        exposed_support(&db, "p1").await,
        crate::truth_contract::Support::Supported,
        "the page this batch re-proved is assertable again"
    );
    assert_eq!(
        exposed_support(&db, "p2").await,
        crate::truth_contract::Support::Unevaluated,
        "the page behind the cursor is not"
    );

    // Drain the rest, and the pass reports itself finished rather than being
    // assumed so.
    let mut passes = 1;
    loop {
        let pass = db.reconcile_supported_pages(1).await.unwrap();
        passes += 1;
        assert_eq!(pass.demoted, 0);
        if pass.complete {
            break;
        }
        assert!(passes < 10, "a bounded pass must terminate on its own data");
    }
    for page_id in ["p1", "p2"] {
        assert_eq!(
            exposed_support(&db, page_id).await,
            crate::truth_contract::Support::Supported
        );
    }

    // And a finished pass is idempotent rather than restarting: the frontier
    // says complete, so a further call does no work at all.
    let after = db.reconcile_supported_pages(1).await.unwrap();
    assert!(after.complete);
    assert_eq!(after.demoted, 0);
}

/// A daemon restart must resume an incomplete pass instead of erasing its
/// durable frontier. Resetting to the beginning on every boot can starve the
/// tail forever when the vault is larger than one boot budget: each process
/// re-proves the same prefix and never reaches the next page.
#[tokio::test]
async fn an_interrupted_reconcile_pass_resumes_from_its_durable_frontier() {
    let (db, _temp) = db_with_queue().await;
    for page_id in ["p1", "p2"] {
        add_page(&db, page_id).await;
        let revisions = derive_page(&db, page_id, 1).await;
        support_claim(&db, page_id, &revisions[0], 0.9).await;
        publish_supported(&db, page_id).await;
    }

    db.begin_support_reconcile_pass().await.unwrap();
    let first = db.reconcile_supported_pages(1).await.unwrap();
    assert!(!first.complete);
    assert_eq!(
        exposed_support(&db, "p1").await,
        crate::truth_contract::Support::Supported
    );
    assert_eq!(
        exposed_support(&db, "p2").await,
        crate::truth_contract::Support::Unevaluated
    );

    // The same startup entry point runs again after a process restart. It must
    // preserve an incomplete frontier under the same reconciliation rules.
    db.begin_support_reconcile_pass().await.unwrap();
    assert_eq!(
        exposed_support(&db, "p1").await,
        crate::truth_contract::Support::Supported,
        "restart must not discard already-proved progress"
    );

    let resumed = db.reconcile_supported_pages(1).await.unwrap();
    assert_eq!(resumed.demoted, 0);
    assert_eq!(
        exposed_support(&db, "p2").await,
        crate::truth_contract::Support::Supported,
        "the next batch must advance past the pre-restart frontier"
    );
}

/// Ruleset and frontier form one promise: the cursor was earned by exactly the
/// evaluator named by the ruleset. A failure between two autocommit writes can
/// leave the new ruleset next to the old cursor, and the next boot then trusts
/// an old prefix as though the new evaluator had proved it. Inject failure on
/// the frontier write and require the ruleset write to roll back with it.
#[tokio::test]
async fn a_reconcile_ruleset_reset_is_atomic_with_its_frontier_reset() {
    let (db, _temp) = db_with_queue().await;
    db.set_app_metadata("support_reconcile_ruleset", "old-ruleset")
        .await
        .unwrap();
    db.set_app_metadata("support_reconcile_frontier", "p1")
        .await
        .unwrap();
    {
        let conn = db.conn.lock().await;
        conn.execute_batch(
            "CREATE TRIGGER fail_reconcile_frontier_reset
             BEFORE INSERT ON app_metadata
             WHEN NEW.key = 'support_reconcile_frontier' AND NEW.value = ''
             BEGIN
               SELECT RAISE(ABORT, 'injected frontier reset failure');
             END;",
        )
        .await
        .unwrap();
    }

    assert!(db.begin_support_reconcile_pass().await.is_err());
    assert_eq!(
        db.get_app_metadata("support_reconcile_ruleset")
            .await
            .unwrap()
            .as_deref(),
        Some("old-ruleset"),
        "a failed frontier reset must not leave a new ruleset beside an old cursor"
    );
    assert_eq!(
        db.get_app_metadata("support_reconcile_frontier")
            .await
            .unwrap()
            .as_deref(),
        Some("p1")
    );
}

/// Blocker 2, shaped like the caller that actually does this. The folder sync in
/// `crates/wenlan-server/src/source_routes.rs` treats a file whose content hash
/// matches a vanished file as a RENAME and calls
/// `rebind_source_id_with_source_page` — the same call this test makes, with the
/// same arguments in the same order.
///
/// A support edge names its evidence by `source_id`. Rebinding moves every chunk
/// to the new id and leaves the edge pointing at an id no memory has, so the
/// page goes on asserting `supported` over a citation that resolves to nothing.
/// Evaluation-time revalidation is no answer: nothing in production calls the
/// predicate on a rename, so "a worker will notice eventually" means never.
///
/// Retract-and-demote rather than re-point, because `edge_id` is content-
/// addressed over the endpoint — re-pointing means re-minting every edge — and
/// `rebind_source_id` is a general primitive whose other callers promise nothing
/// about the new id holding the same bytes. Retraction lands the page on
/// `Unevaluated`, which keeps its file and costs only judge budget.
#[tokio::test]
async fn a_renamed_document_stops_its_pages_asserting_support() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 1).await;
    support_claim(&db, "p1", &revisions[0], 0.9).await;
    publish_supported(&db, "p1").await;

    let old_source_id = format!("mem_{}", revisions[0]);
    db.rebind_source_id_with_source_page(
        "memory",
        &old_source_id,
        "doc_renamed",
        "page_source_old",
        "page_source_new",
    )
    .await
    .unwrap();

    assert_eq!(
        truth_row(&db, "p1").await.unwrap().0,
        "provisional",
        "the rename has to cost the verdict in the SAME transaction that moves \
         the evidence"
    );
    assert_eq!(
        exposed_support(&db, "p1").await,
        crate::truth_contract::Support::Unevaluated,
        "and the page keeps its file while it waits to be re-derived"
    );
    assert_eq!(
        job_for(&db, "p1").await.unwrap().1,
        "pending",
        "re-derivation is queued rather than merely hoped for"
    );

    let live_edges: i64 = {
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM edges
                  WHERE edge_type = 'supports' AND dst_id = ?1 AND valid_until IS NULL",
                libsql::params![old_source_id],
            )
            .await
            .unwrap();
        rows.next().await.unwrap().unwrap().get(0).unwrap()
    };
    assert_eq!(
        live_edges, 0,
        "an edge citing an id no memory holds must not stay active"
    );
}

/// A same-ID rebind is observationally a no-op after proving the source exists.
/// Even `UPDATE memories SET source_id = old WHERE source_id = old` is not a
/// no-op in SQLite: the unconditional AFTER UPDATE parity trigger advances the
/// generation, invalidating proof that depends on that generation. The public
/// primitive must return before issuing any write at all.
#[tokio::test]
async fn a_same_id_rebind_has_no_database_side_effects() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 1).await;
    support_claim(&db, "p1", &revisions[0], 0.9).await;
    let source_id = format!("mem_{}", revisions[0]);
    let before = {
        let conn = db.conn.lock().await;
        conn.total_changes()
    };

    db.rebind_source_id("memory", &source_id, &source_id)
        .await
        .unwrap();

    let after = {
        let conn = db.conn.lock().await;
        conn.total_changes()
    };
    assert_eq!(
        after, before,
        "same-ID rebind must not execute UPDATEs or fire parity triggers"
    );

    let missing = db
        .rebind_source_id("memory", "missing", "missing")
        .await
        .unwrap_err();
    assert!(
        matches!(missing, crate::error::WenlanError::NotFound(_)),
        "the no-op fold must preserve the public primitive's existence check: {missing}"
    );
    let after_missing = {
        let conn = db.conn.lock().await;
        conn.total_changes()
    };
    assert_eq!(
        after_missing, after,
        "a missing same-ID lookup is read-only"
    );
}

// ---------------------------------------------------------------------------
// Production page-linked truth promoter
// ---------------------------------------------------------------------------

struct ProducerJudge {
    backend: LlmBackend,
    model_id: &'static str,
    model_version: &'static str,
    malformed: bool,
    calls: AtomicUsize,
    saw_unlocked_db: AtomicBool,
    conn: Arc<tokio::sync::Mutex<libsql::Connection>>,
}

impl ProducerJudge {
    fn pinned(db: &MemoryDB) -> Self {
        Self {
            backend: LlmBackend::OnDevice,
            model_id: M5_PRODUCTION_JUDGE_MODEL_ID,
            model_version: M5_PRODUCTION_JUDGE_MODEL_VERSION,
            malformed: false,
            calls: AtomicUsize::new(0),
            saw_unlocked_db: AtomicBool::new(true),
            conn: Arc::clone(&db.conn),
        }
    }

    fn malformed(db: &MemoryDB) -> Self {
        Self {
            malformed: true,
            ..Self::pinned(db)
        }
    }
}

#[async_trait]
impl LlmProvider for ProducerJudge {
    async fn generate(&self, request: LlmRequest) -> Result<String, LlmError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.conn.try_lock().is_err() {
            self.saw_unlocked_db.store(false, Ordering::SeqCst);
        }
        if self.malformed {
            return Ok("{\"scores\":[]}".to_string());
        }
        let payload = request
            .user_prompt
            .strip_prefix("UNTRUSTED_JSON_DATA:\n")
            .expect("producer always uses the fixed M5 prompt envelope");
        let input: serde_json::Value = serde_json::from_str(payload).unwrap();
        let scores: Vec<_> = input["items"]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| {
                serde_json::json!({
                    "item_id": item["item_id"].as_str().unwrap(),
                    "score": 0.95,
                })
            })
            .collect();
        Ok(serde_json::json!({ "scores": scores }).to_string())
    }

    fn is_available(&self) -> bool {
        true
    }

    fn name(&self) -> &str {
        "producer-test"
    }

    fn backend(&self) -> LlmBackend {
        self.backend
    }

    fn model_id(&self) -> String {
        self.model_id.to_string()
    }

    fn model_version(&self) -> Option<String> {
        Some(self.model_version.to_string())
    }
}

async fn authorize_producer_judge(db: &MemoryDB) {
    let judge = ProducerJudge::pinned(db);
    let snapshot = crate::claim_judge::snapshot_m5_claim_judge(&judge).unwrap();
    assert!(db
        .activate_approved_m5_claim_judge(&snapshot)
        .await
        .unwrap());
}

async fn authorize_producer_judge_version(db: &MemoryDB, model_version: &str, state: &str) {
    force_producer_judge_policy(db, model_version, state, M5_PRODUCTION_JUDGE_THRESHOLD).await;
}

async fn force_producer_judge_policy(
    db: &MemoryDB,
    model_version: &str,
    state: &str,
    threshold: f64,
) {
    let conn = db.conn.lock().await;
    let tx = conn
        .transaction_with_behavior(libsql::TransactionBehavior::Immediate)
        .await
        .unwrap();
    let generation: i64 = {
        let mut rows = tx
            .query(
                "UPDATE claim_judge_eligibility_generation
                    SET last_generation = last_generation + 1
                  WHERE id = 1 RETURNING last_generation",
                (),
            )
            .await
            .unwrap();
        rows.next().await.unwrap().unwrap().get(0).unwrap()
    };
    tx.execute(
        "INSERT INTO claim_judge_eligibility
             (model_id, model_version, prompt_version, state, threshold, generation)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)
         ON CONFLICT(model_id, model_version, prompt_version) DO UPDATE SET
             state = excluded.state, threshold = excluded.threshold,
             generation = excluded.generation",
        libsql::params![
            M5_PRODUCTION_JUDGE_MODEL_ID,
            model_version,
            crate::claim_judge::M5_CLAIM_ENTAILMENT_PROMPT_VERSION,
            state,
            threshold,
            generation,
        ],
    )
    .await
    .unwrap();
    tx.commit().await.unwrap();
}

#[tokio::test]
async fn production_judge_rejects_unapproved_snapshot_without_writes_or_lease() {
    let (db, _temp) = db_with_queue().await;
    add_producer_page(&db, "unapproved-page", "Alpha is true.", "space-a").await;
    let judge = ProducerJudge {
        model_version: "hf:unapproved/model@deadbeef/model.gguf",
        ..ProducerJudge::pinned(&db)
    };
    let before_generation = test_eligibility_generation(&db).await;
    let before_changes = {
        let conn = db.conn.lock().await;
        conn.total_changes()
    };

    assert_eq!(
        db.run_page_linked_truth_promotion_turn(&judge, "producer-worker")
            .await
            .unwrap(),
        PromotionTurn::RefusedProvider
    );
    assert_eq!(judge.calls.load(Ordering::SeqCst), 0);
    assert_eq!(test_eligibility_generation(&db).await, before_generation);
    let conn = db.conn.lock().await;
    assert_eq!(conn.total_changes(), before_changes);
    let mut rows = conn
        .query(
            "SELECT status, attempts FROM claim_derivation_jobs
              WHERE page_id = 'unapproved-page'",
            (),
        )
        .await
        .unwrap();
    let row = rows.next().await.unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "pending");
    assert_eq!(row.get::<i64>(1).unwrap(), 0);
}

#[tokio::test]
async fn public_judge_rollover_drains_old_prompts_and_requeues_supported_done_pages() {
    let (db, _temp) = db_with_queue().await;
    let page_id = "public-rollover-supported";
    let evidence = "The Alpha project launched successfully.";
    add_producer_page(&db, page_id, evidence, "space-a").await;
    add_producer_memory(
        &db,
        "public-rollover-evidence",
        evidence,
        "space-a",
        "folder",
    )
    .await;
    link_page_evidence(&db, page_id, "public-rollover-evidence").await;
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE claim_derivation_jobs SET status = 'done' WHERE page_id = ?1",
            libsql::params![page_id],
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO page_truth_state
                 (page_id, page_version, support_status, evaluated_at,
                  human_reviewed, updated_at)
             SELECT id, version, 'supported', 1, 0, 1 FROM pages WHERE id = ?1
             ON CONFLICT(page_id) DO UPDATE SET
                 page_version = excluded.page_version,
                 support_status = excluded.support_status,
                 evaluated_at = excluded.evaluated_at,
                 updated_at = excluded.updated_at",
            libsql::params![page_id],
        )
        .await
        .unwrap();
    }
    let before_generation = test_eligibility_generation(&db).await;

    let turn = db
        .run_page_linked_truth_promotion_turn(&ProducerJudge::malformed(&db), "producer-worker")
        .await
        .unwrap();
    assert!(
        matches!(
            &turn,
            PromotionTurn::Requeued {
                page_id: id, ..
            } if id == page_id
        ),
        "unexpected rollover turn: {turn:?}"
    );
    let generation = test_eligibility_generation(&db).await;
    assert_eq!(generation, before_generation + 1);
    assert_eq!(
        job_for(&db, page_id).await.unwrap().1,
        "pending",
        "public activation must reopen the completed supported page; malformed judge then releases it"
    );

    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT state FROM claim_judge_eligibility
              WHERE model_id = 'judge' AND model_version = 'v1' AND prompt_version = 'p1'",
            (),
        )
        .await
        .unwrap();
    assert_eq!(
        rows.next()
            .await
            .unwrap()
            .unwrap()
            .get::<String>(0)
            .unwrap(),
        "draining"
    );
    let mut metadata = conn
        .query(
            "SELECT value FROM app_metadata WHERE key = 'support_reconcile_frontier'",
            (),
        )
        .await
        .unwrap();
    assert_eq!(
        metadata
            .next()
            .await
            .unwrap()
            .unwrap()
            .get::<String>(0)
            .unwrap(),
        ""
    );
}

#[tokio::test]
async fn first_production_judge_activation_drains_old_and_resets_reconcile_atomically() {
    let (db, _temp) = db_with_queue().await;
    authorize_producer_judge_version(&db, "future-old-weights", "active").await;
    db.set_app_metadata("support_reconcile_ruleset", "old-ruleset")
        .await
        .unwrap();
    db.set_app_metadata("support_reconcile_frontier", "complete pass")
        .await
        .unwrap();
    let before_generation = test_eligibility_generation(&db).await;
    let judge = ProducerJudge::pinned(&db);
    let snapshot = crate::claim_judge::snapshot_m5_claim_judge(&judge).unwrap();

    assert!(db
        .activate_approved_m5_claim_judge(&snapshot)
        .await
        .unwrap());
    let generation = test_eligibility_generation(&db).await;
    assert_eq!(generation, before_generation + 1);

    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT model_version, state, threshold, generation
               FROM claim_judge_eligibility
              WHERE model_id = ?1 AND prompt_version = ?2
              ORDER BY CASE WHEN model_version = ?3 THEN 0 ELSE 1 END, model_version",
            libsql::params![
                M5_PRODUCTION_JUDGE_MODEL_ID,
                crate::claim_judge::M5_CLAIM_ENTAILMENT_PROMPT_VERSION,
                M5_PRODUCTION_JUDGE_MODEL_VERSION,
            ],
        )
        .await
        .unwrap();
    let mut policies = Vec::new();
    while let Some(row) = rows.next().await.unwrap() {
        policies.push((
            row.get::<String>(0).unwrap(),
            row.get::<String>(1).unwrap(),
            row.get::<f64>(2).unwrap(),
            row.get::<i64>(3).unwrap(),
        ));
    }
    assert_eq!(
        policies,
        vec![
            (
                M5_PRODUCTION_JUDGE_MODEL_VERSION.to_string(),
                "active".to_string(),
                M5_PRODUCTION_JUDGE_THRESHOLD,
                generation,
            ),
            (
                "future-old-weights".to_string(),
                "draining".to_string(),
                M5_PRODUCTION_JUDGE_THRESHOLD,
                generation,
            ),
        ]
    );
    drop(rows);
    let mut metadata = conn
        .query(
            "SELECT key, value FROM app_metadata
              WHERE key IN ('support_reconcile_ruleset', 'support_reconcile_frontier')
              ORDER BY key",
            (),
        )
        .await
        .unwrap();
    let frontier = metadata.next().await.unwrap().unwrap();
    assert_eq!(
        frontier.get::<String>(0).unwrap(),
        "support_reconcile_frontier"
    );
    assert_eq!(frontier.get::<String>(1).unwrap(), "");
    let ruleset = metadata.next().await.unwrap().unwrap();
    assert_eq!(
        ruleset.get::<String>(0).unwrap(),
        "support_reconcile_ruleset"
    );
    assert_eq!(
        ruleset.get::<String>(1).unwrap(),
        super::claim_derivation::support_reconcile_ruleset(generation)
    );
    let mut enforcement = conn
        .query(
            "SELECT value FROM app_metadata
              WHERE key = 'claim_promoter_enforcement'",
            (),
        )
        .await
        .unwrap();
    assert!(
        enforcement.next().await.unwrap().is_none(),
        "judge activation must not activate whole-page visibility enforcement"
    );
}

#[tokio::test]
async fn repeated_exact_production_judge_activation_is_a_write_free_noop() {
    let (db, _temp) = db_with_queue().await;
    authorize_producer_judge(&db).await;
    let judge = ProducerJudge::pinned(&db);
    let snapshot = crate::claim_judge::snapshot_m5_claim_judge(&judge).unwrap();
    let before_generation = test_eligibility_generation(&db).await;
    let before_changes = {
        let conn = db.conn.lock().await;
        conn.total_changes()
    };

    assert!(db
        .activate_approved_m5_claim_judge(&snapshot)
        .await
        .unwrap());
    assert_eq!(test_eligibility_generation(&db).await, before_generation);
    let conn = db.conn.lock().await;
    assert_eq!(conn.total_changes(), before_changes);
}

#[tokio::test]
async fn production_judge_never_reactivates_draining_or_retired_exact_row() {
    for state in ["draining", "retired"] {
        let (db, _temp) = db_with_queue().await;
        authorize_producer_judge_version(&db, M5_PRODUCTION_JUDGE_MODEL_VERSION, state).await;
        let judge = ProducerJudge::pinned(&db);
        let snapshot = crate::claim_judge::snapshot_m5_claim_judge(&judge).unwrap();
        let before_generation = test_eligibility_generation(&db).await;
        let before_changes = {
            let conn = db.conn.lock().await;
            conn.total_changes()
        };

        assert!(!db
            .activate_approved_m5_claim_judge(&snapshot)
            .await
            .unwrap());
        assert_eq!(test_eligibility_generation(&db).await, before_generation);
        let conn = db.conn.lock().await;
        assert_eq!(conn.total_changes(), before_changes);
        let mut rows = conn
            .query(
                "SELECT state FROM claim_judge_eligibility
                  WHERE model_id = ?1 AND model_version = ?2 AND prompt_version = ?3",
                libsql::params![
                    M5_PRODUCTION_JUDGE_MODEL_ID,
                    M5_PRODUCTION_JUDGE_MODEL_VERSION,
                    crate::claim_judge::M5_CLAIM_ENTAILMENT_PROMPT_VERSION,
                ],
            )
            .await
            .unwrap();
        assert_eq!(
            rows.next()
                .await
                .unwrap()
                .unwrap()
                .get::<String>(0)
                .unwrap(),
            state
        );
    }
}

#[tokio::test]
async fn production_judge_refuses_exact_active_row_with_wrong_threshold() {
    let (db, _temp) = db_with_queue().await;
    force_producer_judge_policy(&db, M5_PRODUCTION_JUDGE_MODEL_VERSION, "active", 0.74).await;
    let judge = ProducerJudge::pinned(&db);
    let snapshot = crate::claim_judge::snapshot_m5_claim_judge(&judge).unwrap();
    let before_generation = test_eligibility_generation(&db).await;
    let before_changes = {
        let conn = db.conn.lock().await;
        conn.total_changes()
    };

    assert!(!db
        .activate_approved_m5_claim_judge(&snapshot)
        .await
        .unwrap());
    assert_eq!(test_eligibility_generation(&db).await, before_generation);
    let conn = db.conn.lock().await;
    assert_eq!(conn.total_changes(), before_changes);
}

async fn add_producer_page(db: &MemoryDB, page_id: &str, content: &str, space: &str) {
    db.insert_page(
        page_id,
        page_id,
        None,
        content,
        None,
        Some(space),
        &[],
        "2026-08-02T00:00:00Z",
    )
    .await
    .unwrap();
}

async fn add_producer_memory(
    db: &MemoryDB,
    source_id: &str,
    content: &str,
    space: &str,
    source_agent: &str,
) {
    let conn = db.conn.lock().await;
    conn.execute(
        "INSERT INTO memories
             (id, content, source, source_id, title, chunk_index, last_modified,
              chunk_type, source_agent, space, pending_revision, version)
         VALUES (?1, ?2, 'memory', ?3, ?3, 0, 0, 'text', ?4, ?5, 0, 1)",
        libsql::params![
            format!("row_{source_id}"),
            content,
            source_id,
            source_agent,
            space,
        ],
    )
    .await
    .unwrap();
}

async fn link_page_evidence(db: &MemoryDB, page_id: &str, source_id: &str) {
    let conn = db.conn.lock().await;
    conn.execute(
        "INSERT INTO page_evidence
             (page_id, source_kind, locator, title, linked_at, link_reason)
         VALUES (?1, 'memory', ?2, NULL, 0, 'producer-test')",
        libsql::params![page_id, source_id],
    )
    .await
    .unwrap();
}

async fn link_active_cite(db: &MemoryDB, page_id: &str, source_id: &str, space: &str) {
    let conn = db.conn.lock().await;
    conn.execute(
        "INSERT INTO edges
             (edge_id, src_id, src_kind, dst_id, dst_kind, edge_type, lineage,
              grounded, root_id, space, created_at, superseded_by, valid_until)
         VALUES (?1, ?2, 'page', ?3, 'memory', 'cites', 'evidence',
                 0, NULL, ?4, 0, NULL, NULL)",
        libsql::params![
            format!("cite_{page_id}_{source_id}"),
            page_id,
            source_id,
            space,
        ],
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn producer_promotes_from_only_linked_same_space_evidence_and_reuses_exact_cache() {
    let (db, _temp) = db_with_queue().await;
    authorize_producer_judge(&db).await;
    add_producer_memory(
        &db,
        "good-evidence",
        "The Alpha project launched successfully.",
        "space-a",
        "folder",
    )
    .await;
    add_producer_memory(
        &db,
        "other-space",
        "The Alpha project launched successfully.",
        "space-b",
        "folder",
    )
    .await;
    add_producer_memory(
        &db,
        "unlinked",
        "The Alpha project launched successfully.",
        "space-a",
        "folder",
    )
    .await;
    let id_linked_content = "The Alpha project launched successfully with records.";
    add_producer_memory(&db, "id-linked", id_linked_content, "space-a", "folder").await;
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE memories SET memory_type = 'observation'
              WHERE source_id = 'id-linked'",
            (),
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO provenance_roots
                 (root_id, identity_version, identity_digest, root_kind,
                  independence_group_id, status, created_at)
             VALUES ('id-linked-human-root', ?1, ?2, 'human_capture',
                     'human:local', 'active', 0)",
            libsql::params![
                crate::provenance::IDENTITY_VERSION,
                crate::provenance::identity_digest("human_capture", id_linked_content),
            ],
        )
        .await
        .unwrap();
    }
    add_producer_page(
        &db,
        "producer-p1",
        "The Alpha project launched successfully.",
        "space-a",
    )
    .await;
    link_page_evidence(&db, "producer-p1", "good-evidence").await;
    link_page_evidence(&db, "producer-p1", "other-space").await;
    link_page_evidence(&db, "producer-p1", "row_id-linked").await;
    // The same exact locator through the second sanctioned link source must
    // dedupe rather than become two obligations.
    link_active_cite(&db, "producer-p1", "good-evidence", "space-a").await;

    let judge = ProducerJudge::pinned(&db);
    assert_eq!(
        db.run_page_linked_truth_promotion_turn(&judge, "producer-worker")
            .await
            .unwrap(),
        PromotionTurn::Completed {
            page_id: "producer-p1".to_string(),
            page_version: 1,
        }
    );
    assert_eq!(judge.calls.load(Ordering::SeqCst), 1);
    assert!(judge.saw_unlocked_db.load(Ordering::SeqCst));
    {
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT candidate_source_id, COUNT(*)
                   FROM claim_run_candidates
                  WHERE page_id = 'producer-p1'
                  GROUP BY candidate_source_id",
                (),
            )
            .await
            .unwrap();
        let mut sources = Vec::new();
        while let Some(row) = rows.next().await.unwrap() {
            sources.push((row.get::<String>(0).unwrap(), row.get::<i64>(1).unwrap()));
        }
        assert_eq!(
            sources,
            vec![
                ("good-evidence".to_string(), 1),
                ("id-linked".to_string(), 1)
            ]
        );
        let mut roots = conn
            .query(
                "SELECT root_id FROM edges
                  WHERE edge_type = 'supports' AND dst_id = 'id-linked'
                    AND valid_until IS NULL",
                (),
            )
            .await
            .unwrap();
        assert_eq!(
            roots
                .next()
                .await
                .unwrap()
                .unwrap()
                .get::<String>(0)
                .unwrap(),
            "id-linked-human-root"
        );
    }

    // Same canonical claim and exact evidence span on another page: no model
    // call, but a fresh attempt row for this run is still mandatory.
    add_producer_page(
        &db,
        "producer-p2",
        "The Alpha project launched successfully.",
        "space-a",
    )
    .await;
    link_page_evidence(&db, "producer-p2", "good-evidence").await;
    link_page_evidence(&db, "producer-p2", "row_id-linked").await;
    assert!(matches!(
        db.run_page_linked_truth_promotion_turn(&judge, "producer-worker")
            .await
            .unwrap(),
        PromotionTurn::Completed { ref page_id, .. } if page_id == "producer-p2"
    ));
    assert_eq!(
        judge.calls.load(Ordering::SeqCst),
        1,
        "an exact five-part hit must spend no model call"
    );
    let attempts: i64 = {
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM claim_judgment_attempts
                  WHERE page_id = 'producer-p2'",
                (),
            )
            .await
            .unwrap();
        rows.next().await.unwrap().unwrap().get(0).unwrap()
    };
    assert_eq!(attempts, 2, "cache hits still belong to this run");
}

#[tokio::test]
async fn producer_malformed_output_requeues_without_attempt_or_publication() {
    let (db, _temp) = db_with_queue().await;
    authorize_producer_judge(&db).await;
    add_producer_memory(
        &db,
        "malformed-evidence",
        "The launch happened in May.",
        "space-a",
        "folder",
    )
    .await;
    add_producer_page(
        &db,
        "malformed-page",
        "The launch happened in May.",
        "space-a",
    )
    .await;
    link_page_evidence(&db, "malformed-page", "malformed-evidence").await;

    let turn = db
        .run_page_linked_truth_promotion_turn(&ProducerJudge::malformed(&db), "producer-worker")
        .await
        .unwrap();
    assert!(matches!(turn, PromotionTurn::Requeued { .. }));
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT
                 (SELECT COUNT(*) FROM claim_judgment_attempts
                   WHERE page_id = 'malformed-page'),
                 (SELECT COUNT(*) FROM page_truth_state
                   WHERE page_id = 'malformed-page' AND evaluated_at IS NOT NULL),
                 (SELECT status FROM claim_derivation_jobs
                   WHERE page_id = 'malformed-page')",
            (),
        )
        .await
        .unwrap();
    let row = rows.next().await.unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 1);
    let deferred: String = {
        let mut attempts = conn
            .query(
                "SELECT outcome FROM claim_judgment_attempts
                  WHERE page_id = 'malformed-page'",
                (),
            )
            .await
            .unwrap();
        attempts.next().await.unwrap().unwrap().get(0).unwrap()
    };
    assert_eq!(deferred, "deferred");
    assert_eq!(row.get::<i64>(1).unwrap(), 0);
    assert_eq!(row.get::<String>(2).unwrap(), "pending");
}

#[tokio::test]
async fn producer_candidate_inventory_keeps_locations_excludes_empty_and_caps_by_rank() {
    let (db, _temp) = db_with_queue().await;
    authorize_producer_judge(&db).await;
    add_producer_page(&db, "rank-page", "Alpha evidence repeats.", "space-a").await;
    add_producer_memory(
        &db,
        "a-repeat",
        "Alpha evidence repeats. Alpha evidence repeats. ",
        "space-a",
        "folder",
    )
    .await;
    link_page_evidence(&db, "rank-page", "a-repeat").await;
    for index in 0..9 {
        let source_id = format!("a{index}");
        add_producer_memory(
            &db,
            &source_id,
            "Alpha evidence repeats.",
            "space-a",
            "folder",
        )
        .await;
        link_page_evidence(&db, "rank-page", &source_id).await;
    }
    add_producer_memory(&db, "z-empty", "It is what it is.", "space-a", "folder").await;
    link_page_evidence(&db, "rank-page", "z-empty").await;

    assert!(matches!(
        db.run_page_linked_truth_promotion_turn(&ProducerJudge::malformed(&db), "producer-worker",)
            .await
            .unwrap(),
        PromotionTurn::Requeued { .. }
    ));
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT candidate_source_id, candidate_span_start,
                    candidate_span_end, candidate_digest
               FROM claim_run_candidates
              WHERE page_id = 'rank-page'
              ORDER BY candidate_source_id, candidate_span_start",
            (),
        )
        .await
        .unwrap();
    let mut locators = Vec::new();
    while let Some(row) = rows.next().await.unwrap() {
        locators.push((
            row.get::<String>(0).unwrap(),
            row.get::<i64>(1).unwrap(),
            row.get::<i64>(2).unwrap(),
            row.get::<String>(3).unwrap(),
        ));
    }
    assert_eq!(
        locators.len(),
        super::claim_derivation::MAX_CANDIDATES_PER_CLAIM
    );
    let repeats: Vec<_> = locators
        .iter()
        .filter(|(source, ..)| source == "a-repeat")
        .collect();
    assert_eq!(repeats.len(), 2);
    assert_ne!((repeats[0].1, repeats[0].2), (repeats[1].1, repeats[1].2));
    assert_eq!(
        repeats[0].3, repeats[1].3,
        "digest is content, not location"
    );
    assert!(locators.iter().all(|(source, ..)| source != "z-empty"));
    assert!(locators.iter().all(|(source, ..)| source != "a6"));
    assert!(locators.iter().all(|(source, ..)| source != "a7"));
    assert!(locators.iter().all(|(source, ..)| source != "a8"));
}

#[tokio::test]
async fn producer_excludes_ambiguous_live_locators_before_sealing() {
    let (db, _temp) = db_with_queue().await;
    authorize_producer_judge(&db).await;
    add_producer_page(&db, "ambiguous-memory-page", "Alpha fact.", "space-a").await;
    add_producer_memory(&db, "ambiguous-source", "Alpha fact.", "space-a", "folder").await;
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "INSERT INTO memories
                 (id, content, source, source_id, title, chunk_index, last_modified,
                  chunk_type, source_agent, space, pending_revision, version)
             VALUES ('ambiguous-row-2', 'Different alpha fact.', 'memory',
                     'ambiguous-source', 'ambiguous', 0, 0, 'text', 'folder',
                     'space-a', 0, 1)",
            (),
        )
        .await
        .unwrap();
    }
    link_page_evidence(&db, "ambiguous-memory-page", "ambiguous-source").await;
    let judge = ProducerJudge::pinned(&db);
    assert!(matches!(
        db.run_page_linked_truth_promotion_turn(&judge, "producer-worker")
            .await
            .unwrap(),
        PromotionTurn::Completed { .. }
    ));
    assert_eq!(judge.calls.load(Ordering::SeqCst), 0);
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT COUNT(*) FROM claim_run_candidates
              WHERE page_id = 'ambiguous-memory-page'",
            (),
        )
        .await
        .unwrap();
    assert_eq!(
        rows.next().await.unwrap().unwrap().get::<i64>(0).unwrap(),
        0
    );
}

#[tokio::test]
async fn producer_does_not_launder_non_observation_human_evidence_through_another_root_kind() {
    let (db, _temp) = db_with_queue().await;
    authorize_producer_judge(&db).await;
    let evidence = "A human preference about Alpha.";
    add_producer_page(&db, "human-kind-page", evidence, "space-a").await;
    add_producer_memory(&db, "human-kind-memory", evidence, "space-a", "human").await;
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE memories SET memory_type = 'preference'
              WHERE source_id = 'human-kind-memory'",
            (),
        )
        .await
        .unwrap();
    }
    link_page_evidence(&db, "human-kind-page", "human-kind-memory").await;
    let judge = ProducerJudge::pinned(&db);
    assert!(matches!(
        db.run_page_linked_truth_promotion_turn(&judge, "producer-worker")
            .await
            .unwrap(),
        PromotionTurn::Completed { .. }
    ));
    assert_eq!(judge.calls.load(Ordering::SeqCst), 0);
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT COUNT(*) FROM claim_run_candidates
              WHERE page_id = 'human-kind-page'",
            (),
        )
        .await
        .unwrap();
    assert_eq!(
        rows.next().await.unwrap().unwrap().get::<i64>(0).unwrap(),
        0
    );
    let mut roots = conn
        .query(
            "SELECT COUNT(*) FROM provenance_roots
              WHERE identity_digest = ?1 AND root_kind = 'generated'",
            libsql::params![crate::provenance::identity_digest("generated", evidence)],
        )
        .await
        .unwrap();
    assert_eq!(
        roots.next().await.unwrap().unwrap().get::<i64>(0).unwrap(),
        0,
        "a human preference must not be relabelled as generated to gain a vote"
    );
}

#[tokio::test]
async fn producer_observation_human_evidence_without_human_root_does_not_mint_generated_root() {
    let (db, _temp) = db_with_queue().await;
    authorize_producer_judge(&db).await;
    let evidence = "A human observation about Alpha.";
    add_producer_page(&db, "human-observation-page", evidence, "space-a").await;
    add_producer_memory(
        &db,
        "human-observation-memory",
        evidence,
        "space-a",
        "human",
    )
    .await;
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE memories SET memory_type = 'observation'
              WHERE source_id = 'human-observation-memory'",
            (),
        )
        .await
        .unwrap();
    }
    link_page_evidence(&db, "human-observation-page", "human-observation-memory").await;

    let judge = ProducerJudge::pinned(&db);
    assert!(matches!(
        db.run_page_linked_truth_promotion_turn(&judge, "producer-worker")
            .await
            .unwrap(),
        PromotionTurn::Completed { .. }
    ));
    assert_eq!(judge.calls.load(Ordering::SeqCst), 0);

    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT
                 (SELECT COUNT(*) FROM claim_run_candidates
                   WHERE page_id = 'human-observation-page'),
                 (SELECT COUNT(*) FROM provenance_roots
                   WHERE root_kind = 'generated'
                     AND identity_digest = ?1),
                 (SELECT COUNT(*) FROM provenance_roots
                   WHERE root_kind IN ('human_capture', 'human_edit_delta')
                     AND identity_digest IN (?2, ?3))",
            libsql::params![
                crate::provenance::identity_digest("generated", evidence),
                crate::provenance::identity_digest("human_capture", evidence),
                crate::provenance::identity_digest("human_edit_delta", evidence),
            ],
        )
        .await
        .unwrap();
    let row = rows.next().await.unwrap().unwrap();
    assert_eq!(
        row.get::<i64>(0).unwrap(),
        0,
        "human observation is non-voting without a human root"
    );
    assert_eq!(
        row.get::<i64>(1).unwrap(),
        0,
        "promoter must not mint a generated root for human evidence"
    );
    assert_eq!(
        row.get::<i64>(2).unwrap(),
        0,
        "fixture must have no pre-existing human root"
    );
}

#[tokio::test]
async fn producer_observation_human_evidence_with_existing_generated_root_does_not_vote() {
    let (db, _temp) = db_with_queue().await;
    authorize_producer_judge(&db).await;
    let evidence = "A human observation about Alpha.";
    add_producer_page(&db, "human-observation-collision-page", evidence, "space-a").await;
    add_producer_memory(
        &db,
        "human-observation-collision-memory",
        evidence,
        "space-a",
        "human",
    )
    .await;
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE memories SET memory_type = 'observation'
              WHERE source_id = 'human-observation-collision-memory'",
            (),
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO provenance_roots
                 (root_id, identity_version, identity_digest, root_kind,
                  independence_group_id, status, created_at)
             VALUES ('human-observation-generated-root', ?1, ?2, 'generated',
                     'machine:fixture', 'active', 0)",
            libsql::params![
                crate::provenance::IDENTITY_VERSION,
                crate::provenance::identity_digest("generated", evidence),
            ],
        )
        .await
        .unwrap();
    }
    link_page_evidence(
        &db,
        "human-observation-collision-page",
        "human-observation-collision-memory",
    )
    .await;

    let judge = ProducerJudge::pinned(&db);
    assert!(matches!(
        db.run_page_linked_truth_promotion_turn(&judge, "producer-worker")
            .await
            .unwrap(),
        PromotionTurn::Completed { ref page_id, .. }
            if page_id == "human-observation-collision-page"
    ));
    assert_eq!(
        judge.calls.load(Ordering::SeqCst),
        0,
        "a generated root must not make human observation evidence vote"
    );
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT
                 (SELECT COUNT(*) FROM claim_run_candidates
                   WHERE page_id = 'human-observation-collision-page'),
                 (SELECT COUNT(*) FROM provenance_roots
                   WHERE root_kind = 'generated' AND identity_digest = ?1)",
            libsql::params![crate::provenance::identity_digest("generated", evidence)],
        )
        .await
        .unwrap();
    let row = rows.next().await.unwrap().unwrap();
    assert_eq!(
        row.get::<i64>(0).unwrap(),
        0,
        "human observation may vote only through an eligible human root"
    );
    assert_eq!(
        row.get::<i64>(1).unwrap(),
        1,
        "the pre-existing generated root must remain but never be reused"
    );
}

#[tokio::test]
async fn producer_root_acquisition_error_requeues_before_sealing_absence() {
    let (db, _temp) = db_with_queue().await;
    authorize_producer_judge(&db).await;
    add_producer_page(&db, "root-error-page", "Alpha is factual.", "space-a").await;
    add_producer_memory(
        &db,
        "root-error-memory",
        "Alpha is factual.",
        "space-a",
        "folder",
    )
    .await;
    link_page_evidence(&db, "root-error-page", "root-error-memory").await;
    {
        let conn = db.conn.lock().await;
        conn.execute_batch(
            "CREATE TRIGGER fail_promoter_root
             BEFORE INSERT ON provenance_roots
             WHEN NEW.root_kind = 'document_ingest'
             BEGIN SELECT RAISE(ABORT, 'injected root acquisition failure'); END;",
        )
        .await
        .unwrap();
    }

    let judge = ProducerJudge::pinned(&db);
    assert!(matches!(
        db.run_page_linked_truth_promotion_turn(&judge, "producer-worker")
            .await
            .unwrap(),
        PromotionTurn::Requeued { .. }
    ));
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT
                 (SELECT COUNT(*) FROM claim_run_inventories
                   WHERE page_id = 'root-error-page'),
                 (SELECT COUNT(*) FROM claim_run_candidates
                   WHERE page_id = 'root-error-page')",
            (),
        )
        .await
        .unwrap();
    let row = rows.next().await.unwrap().unwrap();
    assert_eq!(
        (row.get::<i64>(0).unwrap(), row.get::<i64>(1).unwrap()),
        (0, 0),
        "an operational root error is retryable, not a durable empty inventory"
    );
}

#[tokio::test]
async fn producer_refuses_non_on_device_before_leasing() {
    let (db, _temp) = db_with_queue().await;
    authorize_producer_judge(&db).await;
    add_producer_page(&db, "api-page", "A factual claim.", "space-a").await;
    let judge = ProducerJudge {
        backend: LlmBackend::Api,
        ..ProducerJudge::pinned(&db)
    };

    assert_eq!(
        db.run_page_linked_truth_promotion_turn(&judge, "producer-worker")
            .await
            .unwrap(),
        PromotionTurn::RefusedProvider
    );
    assert_eq!(job_for(&db, "api-page").await.unwrap().2, 0);
}

#[tokio::test]
async fn producer_rejudges_a_non_on_device_cache_row_before_supporting() {
    let (db, _temp) = db_with_queue().await;
    authorize_producer_judge(&db).await;
    let text = "Cached claim.";
    add_producer_page(&db, "cache-backend-page", text, "space-a").await;
    add_producer_memory(&db, "cache-backend-memory", text, "space-a", "folder").await;
    link_page_evidence(&db, "cache-backend-page", "cache-backend-memory").await;
    {
        let conn = db.conn.lock().await;
        let claim_digest = crate::provenance::revision_content_digest("Cached claim");
        let span_digest = crate::provenance::revision_content_digest(text);
        conn.execute(
            "INSERT INTO entailment_cache
                 (claim_text_digest, source_span_digest, model_id, model_version,
                  prompt_version, score, threshold_at_write, backend, scored_at)
             VALUES (?1, ?2, ?3, ?4, ?5, 0.99, 0.75, 'api', 0)",
            libsql::params![
                claim_digest,
                span_digest,
                M5_PRODUCTION_JUDGE_MODEL_ID,
                M5_PRODUCTION_JUDGE_MODEL_VERSION,
                crate::claim_judge::M5_CLAIM_ENTAILMENT_PROMPT_VERSION,
            ],
        )
        .await
        .unwrap();
    }

    let judge = ProducerJudge::pinned(&db);
    assert!(matches!(
        db.run_page_linked_truth_promotion_turn(&judge, "producer-worker")
            .await
            .unwrap(),
        PromotionTurn::Completed { .. }
    ));
    assert_eq!(
        judge.calls.load(Ordering::SeqCst),
        1,
        "a cache row from a non-on-device backend must not bypass live judging"
    );
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT backend FROM entailment_cache
              WHERE model_id = ?1 AND model_version = ?2 AND prompt_version = ?3",
            libsql::params![
                M5_PRODUCTION_JUDGE_MODEL_ID,
                M5_PRODUCTION_JUDGE_MODEL_VERSION,
                crate::claim_judge::M5_CLAIM_ENTAILMENT_PROMPT_VERSION,
            ],
        )
        .await
        .unwrap();
    assert_eq!(
        rows.next()
            .await
            .unwrap()
            .unwrap()
            .get::<String>(0)
            .unwrap(),
        "on_device"
    );
}

#[tokio::test]
async fn producer_stale_lease_derivation_writes_neither_rows_nor_marker() {
    let (db, _temp) = db_with_queue().await;
    authorize_producer_judge(&db).await;
    add_producer_page(&db, "stale-producer", "A factual sentence.", "space-a").await;
    let job = db
        .lease_next_derivation_job("stale-worker")
        .await
        .unwrap()
        .unwrap();
    assert!(db
        .release_derivation_job(&job, "stale-worker", "test reclaim")
        .await
        .unwrap());

    let (write, claims) = db
        .derive_leased_page_claims(&job, "stale-worker")
        .await
        .unwrap();
    assert_eq!(write, DerivationWrite::StaleLease);
    assert!(claims.is_empty());
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT
                 (SELECT COUNT(*) FROM claims WHERE page_id = 'stale-producer'),
                 (SELECT COUNT(*) FROM claim_derivation_markers
                   WHERE page_id = 'stale-producer')",
            (),
        )
        .await
        .unwrap();
    let row = rows.next().await.unwrap().unwrap();
    assert_eq!(
        (row.get::<i64>(0).unwrap(), row.get::<i64>(1).unwrap()),
        (0, 0)
    );
}

#[tokio::test]
async fn producer_derivation_is_atomic_and_oversize_pages_park_without_marker() {
    let (db, _temp) = db_with_queue().await;
    authorize_producer_judge(&db).await;
    add_producer_page(&db, "atomic-page", "One factual sentence.", "space-a").await;
    {
        let conn = db.conn.lock().await;
        conn.execute_batch(
            "CREATE TRIGGER fail_producer_marker
             BEFORE INSERT ON claim_derivation_markers
             WHEN NEW.page_id = 'atomic-page'
             BEGIN SELECT RAISE(ABORT, 'injected marker failure'); END;",
        )
        .await
        .unwrap();
    }
    let judge = ProducerJudge::pinned(&db);
    assert!(matches!(
        db.run_page_linked_truth_promotion_turn(&judge, "producer-worker")
            .await
            .unwrap(),
        PromotionTurn::Requeued { .. }
    ));
    {
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT
                     (SELECT COUNT(*) FROM claims WHERE page_id = 'atomic-page'),
                     (SELECT COUNT(*) FROM claim_derivation_markers
                       WHERE page_id = 'atomic-page')",
                (),
            )
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        assert_eq!(
            (row.get::<i64>(0).unwrap(), row.get::<i64>(1).unwrap()),
            (0, 0)
        );
        conn.execute_batch(
            "DROP TRIGGER fail_producer_marker;
             UPDATE claim_derivation_jobs SET status = 'parked'
              WHERE page_id = 'atomic-page';",
        )
        .await
        .unwrap();
    }

    let oversized = (0..=super::claim_derivation::MAX_CLAIMS_PER_PAGE)
        .map(|index| format!("Claim number {index} is independently stated."))
        .collect::<Vec<_>>()
        .join(" ");
    add_producer_page(&db, "oversized-page", &oversized, "space-a").await;
    assert!(matches!(
        db.run_page_linked_truth_promotion_turn(&judge, "producer-worker")
            .await
            .unwrap(),
        PromotionTurn::Parked { ref page_id, .. } if page_id == "oversized-page"
    ));
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT
                 (SELECT COUNT(*) FROM claims WHERE page_id = 'oversized-page'),
                 (SELECT COUNT(*) FROM claim_derivation_markers
                   WHERE page_id = 'oversized-page'),
                 (SELECT status FROM claim_derivation_jobs
                   WHERE page_id = 'oversized-page')",
            (),
        )
        .await
        .unwrap();
    let row = rows.next().await.unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 0);
    assert_eq!(row.get::<i64>(1).unwrap(), 0);
    assert_eq!(row.get::<String>(2).unwrap(), "parked");
}

#[tokio::test]
async fn producer_canonicalizes_conservatively_and_ambiguous_duplicates_mint_new_ids() {
    let (db, _temp) = db_with_queue().await;
    authorize_producer_judge(&db).await;
    add_producer_page(
        &db,
        "identity-page",
        "Cafe\u{301} is open.[1] Über is ready. It is what it is.",
        "space-a",
    )
    .await;
    let judge = ProducerJudge::pinned(&db);
    assert!(matches!(
        db.run_page_linked_truth_promotion_turn(&judge, "producer-worker")
            .await
            .unwrap(),
        PromotionTurn::Completed { .. }
    ));
    let first_ids: Vec<String> = {
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT cr.claim_id, cr.canonical_text, ca.span_start, ca.span_end,
                        ca.span_digest
                   FROM page_version_claims pvc
                   JOIN claim_revisions cr ON cr.claim_revision_id = pvc.claim_revision_id
                   JOIN claim_anchors ca ON ca.claim_revision_id = pvc.claim_revision_id
                    AND ca.source_doc_id = pvc.page_id AND ca.source_version = pvc.page_version
                  WHERE pvc.page_id = 'identity-page' AND pvc.page_version = 1
                  ORDER BY pvc.ordinal",
                (),
            )
            .await
            .unwrap();
        let first = rows.next().await.unwrap().unwrap();
        let first_id: String = first.get(0).unwrap();
        assert_eq!(first.get::<String>(1).unwrap(), "Café is open");
        let raw = "Cafe\u{301} is open.[1] Über is ready. It is what it is.";
        let start = first.get::<i64>(2).unwrap() as usize;
        let end = first.get::<i64>(3).unwrap() as usize;
        assert_eq!(&raw[start..end], "Cafe\u{301} is open");
        assert_eq!(end, raw.find(".[1]").unwrap());
        assert_eq!(
            crate::provenance::revision_content_digest(&raw[start..end]),
            first.get::<String>(4).unwrap(),
            "marker stripping must map the bare sentence back to exact page bytes"
        );
        let second = rows.next().await.unwrap().unwrap();
        let second_id: String = second.get(0).unwrap();
        assert_eq!(second.get::<String>(1).unwrap(), "Über is ready");
        let second_start = second.get::<i64>(2).unwrap() as usize;
        let second_end = second.get::<i64>(3).unwrap() as usize;
        assert_eq!(&raw[second_start..second_end], "Über is ready");
        assert!(second_start >= raw.find("] Über").unwrap() + 2);
        let third = rows.next().await.unwrap().unwrap();
        let third_id: String = third.get(0).unwrap();
        assert_eq!(third.get::<String>(1).unwrap(), "It is what it is");
        assert!(rows.next().await.unwrap().is_none());
        vec![first_id, second_id, third_id]
    };

    db.update_page_content(
        "identity-page",
        "Café is open. Café is open.",
        &[],
        "producer-test",
    )
    .await
    .unwrap();
    assert!(matches!(
        db.run_page_linked_truth_promotion_turn(&judge, "producer-worker")
            .await
            .unwrap(),
        PromotionTurn::Completed {
            page_version: 2,
            ..
        }
    ));
    let duplicate_ids: Vec<String> = {
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT cr.claim_id
                   FROM page_version_claims pvc
                   JOIN claim_revisions cr ON cr.claim_revision_id = pvc.claim_revision_id
                  WHERE pvc.page_id = 'identity-page' AND pvc.page_version = 2
                  ORDER BY pvc.ordinal",
                (),
            )
            .await
            .unwrap();
        let mut ids = Vec::new();
        while let Some(row) = rows.next().await.unwrap() {
            ids.push(row.get(0).unwrap());
        }
        ids
    };
    assert_eq!(duplicate_ids.len(), 2);
    assert_ne!(duplicate_ids[0], duplicate_ids[1]);
    assert!(duplicate_ids.iter().all(|id| !first_ids.contains(id)));
}

#[tokio::test]
async fn producer_same_version_digest_change_replaces_inventory_and_marker_atomically() {
    let (db, _temp) = db_with_queue().await;
    authorize_producer_judge(&db).await;
    add_producer_page(&db, "same-version-page", "Alpha was first.", "space-a").await;
    let judge = ProducerJudge::pinned(&db);
    assert!(matches!(
        db.run_page_linked_truth_promotion_turn(&judge, "producer-worker")
            .await
            .unwrap(),
        PromotionTurn::Completed { .. }
    ));

    let replacement = "Beta is now the only claim.";
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE page_history SET content = ?1
              WHERE page_id = 'same-version-page' AND version = 1",
            libsql::params![replacement],
        )
        .await
        .unwrap();
        conn.execute(
            "UPDATE pages SET content = ?1 WHERE id = 'same-version-page'",
            libsql::params![replacement],
        )
        .await
        .unwrap();
        conn.execute(
            "UPDATE claim_derivation_jobs
                SET status = 'pending', attempts = 0, lease_owner = NULL,
                    lease_expires_at = NULL, last_error = NULL
              WHERE page_id = 'same-version-page'",
            (),
        )
        .await
        .unwrap();
    }

    assert!(matches!(
        db.run_page_linked_truth_promotion_turn(&judge, "producer-worker")
            .await
            .unwrap(),
        PromotionTurn::Completed {
            page_version: 1,
            ..
        }
    ));
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT m.page_version_digest, m.inventory_count, cr.canonical_text
               FROM claim_derivation_markers m
               JOIN page_version_claims pvc
                 ON pvc.page_id = m.page_id AND pvc.page_version = m.page_version
               JOIN claim_revisions cr ON cr.claim_revision_id = pvc.claim_revision_id
              WHERE m.page_id = 'same-version-page'",
            (),
        )
        .await
        .unwrap();
    let row = rows.next().await.unwrap().unwrap();
    assert_eq!(
        row.get::<String>(0).unwrap(),
        crate::provenance::revision_content_digest(replacement)
    );
    assert_eq!(row.get::<i64>(1).unwrap(), 1);
    assert_eq!(row.get::<String>(2).unwrap(), "Beta is now the only claim");
    assert!(rows.next().await.unwrap().is_none());
}

#[tokio::test]
async fn producer_requeues_if_judge_generation_moves_between_policy_read_and_lease() {
    let (db, _temp) = db_with_queue().await;
    authorize_producer_judge(&db).await;
    add_producer_page(&db, "generation-race-page", "Alpha is true.", "space-a").await;
    let judge = ProducerJudge::pinned(&db);
    let snapshot = crate::claim_judge::snapshot_m5_claim_judge(&judge).unwrap();
    let (threshold, stale_generation) = db.active_judge_policy(&snapshot).await.unwrap().unwrap();
    authorize_producer_judge_version(&db, M5_PRODUCTION_JUDGE_MODEL_VERSION, "active").await;
    let job = db
        .lease_next_derivation_job("generation-worker")
        .await
        .unwrap()
        .unwrap();
    assert_ne!(job.eligibility_generation, stale_generation);

    let turn = db
        .run_leased_page_linked_truth_promotion(
            &judge,
            &snapshot,
            threshold,
            stale_generation,
            &job,
            "generation-worker",
        )
        .await
        .unwrap();
    assert!(matches!(turn, PromotionTurn::Requeued { .. }));
    assert_eq!(judge.calls.load(Ordering::SeqCst), 0);
    {
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT
                     (SELECT COUNT(*) FROM claims WHERE page_id = 'generation-race-page'),
                     (SELECT COUNT(*) FROM claim_derivation_markers
                       WHERE page_id = 'generation-race-page')",
                (),
            )
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        assert_eq!(
            (row.get::<i64>(0).unwrap(), row.get::<i64>(1).unwrap()),
            (0, 0)
        );
    }
    assert!(db
        .release_derivation_job(&job, "generation-worker", "test cleanup")
        .await
        .unwrap());
}

#[test]
fn producer_judge_budget_accepts_the_boundary_and_rejects_one_more_byte() {
    let fixed = crate::claim_judge::M5_CLAIM_ENTAILMENT_SYSTEM_PROMPT.len() + 2048;
    let remaining = super::claim_derivation::MAX_JUDGE_TOKENS_PER_PAGE - fixed;
    assert!(super::claim_derivation::m5_judge_budget_allows(&[
        remaining
    ]));
    assert!(!super::claim_derivation::m5_judge_budget_allows(&[
        remaining + 1
    ]));
}

#[tokio::test]
async fn producer_mints_new_identity_when_text_is_unique_but_anchor_is_ambiguous() {
    let (db, _temp) = db_with_queue().await;
    authorize_producer_judge(&db).await;
    let content = "Alpha is true. Beta is true.";
    add_producer_page(&db, "ambiguous-alignment-page", content, "space-a").await;
    let judge = ProducerJudge::pinned(&db);
    assert!(matches!(
        db.run_page_linked_truth_promotion_turn(&judge, "producer-worker")
            .await
            .unwrap(),
        PromotionTurn::Completed { .. }
    ));

    let (alpha_claim_id, beta_revision, alpha_end, alpha_digest) = {
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT cr.claim_id, cr.claim_revision_id, ca.span_start,
                        ca.span_end, ca.span_digest
                   FROM page_version_claims pvc
                   JOIN claim_revisions cr ON cr.claim_revision_id = pvc.claim_revision_id
                   JOIN claim_anchors ca ON ca.claim_revision_id = pvc.claim_revision_id
                    AND ca.source_doc_id = pvc.page_id AND ca.source_version = pvc.page_version
                  WHERE pvc.page_id = 'ambiguous-alignment-page' AND pvc.page_version = 1
                  ORDER BY pvc.ordinal",
                (),
            )
            .await
            .unwrap();
        let alpha = rows.next().await.unwrap().unwrap();
        let beta = rows.next().await.unwrap().unwrap();
        (
            alpha.get::<String>(0).unwrap(),
            beta.get::<String>(1).unwrap(),
            alpha.get::<i64>(3).unwrap(),
            alpha.get::<String>(4).unwrap(),
        )
    };
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "INSERT INTO claim_anchors
                 (claim_revision_id, source_doc_id, source_version,
                  span_start, span_end, span_digest, created_at)
             VALUES (?1, 'ambiguous-alignment-page', 1, 0, ?2, ?3, 0)",
            libsql::params![beta_revision, alpha_end, alpha_digest],
        )
        .await
        .unwrap();
    }
    db.update_page_content("ambiguous-alignment-page", content, &[], "producer-test")
        .await
        .unwrap();
    assert!(matches!(
        db.run_page_linked_truth_promotion_turn(&judge, "producer-worker")
            .await
            .unwrap(),
        PromotionTurn::Completed {
            page_version: 2,
            ..
        }
    ));
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT cr.claim_id
               FROM page_version_claims pvc
               JOIN claim_revisions cr ON cr.claim_revision_id = pvc.claim_revision_id
              WHERE pvc.page_id = 'ambiguous-alignment-page' AND pvc.page_version = 2
                AND pvc.ordinal = 0",
            (),
        )
        .await
        .unwrap();
    let new_alpha_claim_id: String = rows.next().await.unwrap().unwrap().get(0).unwrap();
    assert_ne!(new_alpha_claim_id, alpha_claim_id);
}

#[tokio::test]
async fn producer_rolling_judge_upgrade_replaces_edge_without_losing_supported_state() {
    let (db, _temp) = db_with_queue().await;
    authorize_producer_judge(&db).await;
    let evidence = "Alpha launched successfully.";
    add_producer_memory(&db, "rolling-source", evidence, "space-a", "folder").await;
    add_producer_page(&db, "rolling-page", evidence, "space-a").await;
    link_page_evidence(&db, "rolling-page", "rolling-source").await;
    let v1 = ProducerJudge::pinned(&db);
    assert!(matches!(
        db.run_page_linked_truth_promotion_turn(&v1, "producer-worker")
            .await
            .unwrap(),
        PromotionTurn::Completed { .. }
    ));
    assert_eq!(truth_row(&db, "rolling-page").await.unwrap().0, "supported");

    authorize_producer_judge_version(&db, M5_PRODUCTION_JUDGE_MODEL_VERSION, "draining").await;
    authorize_producer_judge_version(&db, "future-weights-v2", "active").await;
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE claim_derivation_jobs
                SET status = 'pending', attempts = 0, lease_owner = NULL,
                    lease_expires_at = NULL, last_error = NULL
              WHERE page_id = 'rolling-page'",
            (),
        )
        .await
        .unwrap();
    }
    let v2 = ProducerJudge {
        model_version: "future-weights-v2",
        ..ProducerJudge::pinned(&db)
    };
    let v2_snapshot = crate::claim_judge::snapshot_m5_claim_judge(&v2).unwrap();
    let (v2_threshold, v2_generation) =
        db.active_judge_policy(&v2_snapshot).await.unwrap().unwrap();
    let v2_job = db
        .lease_next_derivation_job("producer-worker")
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(
        db.run_leased_page_linked_truth_promotion(
            &v2,
            &v2_snapshot,
            v2_threshold,
            v2_generation,
            &v2_job,
            "producer-worker",
        )
        .await
        .unwrap(),
        PromotionTurn::Completed { .. }
    ));
    assert_eq!(truth_row(&db, "rolling-page").await.unwrap().0, "supported");

    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT json_extract(payload, '$.model_version'),
                    superseded_by IS NOT NULL, valid_until IS NOT NULL
              FROM edges
             WHERE edge_type = 'supports' AND dst_id = 'rolling-source'
              ORDER BY valid_until IS NULL",
            (),
        )
        .await
        .unwrap();
    let old = rows.next().await.unwrap().unwrap();
    assert_eq!(
        old.get::<String>(0).unwrap(),
        M5_PRODUCTION_JUDGE_MODEL_VERSION
    );
    assert_eq!(
        (old.get::<i64>(1).unwrap(), old.get::<i64>(2).unwrap()),
        (1, 1)
    );
    let new = rows.next().await.unwrap().unwrap();
    assert_eq!(new.get::<String>(0).unwrap(), "future-weights-v2");
    assert_eq!(
        (new.get::<i64>(1).unwrap(), new.get::<i64>(2).unwrap()),
        (0, 0)
    );
    assert!(rows.next().await.unwrap().is_none());
}
