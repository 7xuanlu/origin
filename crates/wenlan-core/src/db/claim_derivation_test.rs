// SPDX-License-Identifier: Apache-2.0
//! Teeth for the M5 claim-derivation queue. Each test names the weakening it
//! guards; deleting the rule in the doc comment must turn the test RED.

use super::claim_derivation::{SupportOutcome, EXTRACTOR_VERSION, MAX_ATTEMPTS, SUPPORT_THRESHOLD};
use super::tests::test_db;
use super::MemoryDB;

/// A fully migrated database with NO pages. `test_db` runs the real migration
/// chain, so migration 104 has already installed the triggers here — re-running
/// the installer is itself the idempotence check a resumed migration needs.
async fn db_with_queue() -> (MemoryDB, tempfile::TempDir) {
    let (db, temp) = test_db().await;
    {
        let conn = db.conn.lock().await;
        let tx = conn.transaction().await.unwrap();
        MemoryDB::ensure_claim_identity_tables(&tx).await.unwrap();
        MemoryDB::ensure_claim_derivation_triggers(&tx)
            .await
            .unwrap();
        tx.commit().await.unwrap();
    }
    (db, temp)
}

/// The state of a real vault the moment before migration 104: the substrate
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
    (db, temp)
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
    db.lease_next_derivation_job("rescuer").await.unwrap();

    assert!(
        !db.finish_derivation_job(&job.job_id, "zombie")
            .await
            .unwrap(),
        "the old owner's finish must be a no-op"
    );
    assert_eq!(job_for(&db, "p1").await.unwrap().1, "leased");
    assert!(db
        .finish_derivation_job(&job.job_id, "rescuer")
        .await
        .unwrap());
    assert_eq!(job_for(&db, "p1").await.unwrap().1, "done");
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
        db.release_derivation_job(&job.job_id, "worker", "boom")
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
    assert_eq!(db.park_exhausted_derivation_jobs().await.unwrap(), 1);
    assert_eq!(job_for(&db, "poison").await.unwrap().1, "parked");
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
            libsql::params![
                rev_id.clone(),
                claim_id,
                format!("claim {i}"),
                format!("d{i}"),
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

/// A `supports` edge at `score`, written straight to `edges` — these tests are
/// about what the predicate READS, not about the write path's own refusals.
async fn support_claim(db: &MemoryDB, page_id: &str, revision_id: &str, score: f64) {
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
         VALUES (?1, 'evidence prose', 'memory', ?2, 'evidence', 0, 0, 'text', ?3)",
        libsql::params![format!("m_{mem_id}"), mem_id.clone(), space.clone()],
    )
    .await
    .unwrap();
    // `edges` CHECKs that a supports edge carries a root — the schema enforcing
    // the same rule `write_support_edge` verifies. These tests are about the
    // predicate, so the root is seeded rather than earned.
    let root_id = format!("root_{revision_id}");
    conn.execute(
        "INSERT OR IGNORE INTO provenance_roots
             (root_id, identity_version, identity_digest, root_kind,
              independence_group_id, status, created_at)
         VALUES (?1, 1, ?2, 'document_ingest', ?3, 'active', 0)",
        libsql::params![
            root_id.clone(),
            format!("digest_{revision_id}"),
            format!("group_{revision_id}")
        ],
    )
    .await
    .unwrap();
    let payload = format!("{{\"score\":{score},\"threshold_at_write\":0.75}}");
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
    assert!(db
        .finalize_page_support("p1", 1, &SupportOutcome::Supported)
        .await
        .unwrap());
    let (status, evaluated, reviewed) = truth_row(&db, "p1").await.unwrap();
    assert_eq!(status, "supported");
    assert!(evaluated.is_some());
    assert_eq!(reviewed, 0, "machine support never manufactures review");
}

/// Condition 3, and the only failure that is a verdict: the claims are real,
/// the evidence was looked for, and it was not there.
#[tokio::test]
async fn a_claim_with_no_support_refutes_the_page() {
    let (db, _temp) = db_with_queue().await;
    add_page(&db, "p1").await;
    let revisions = derive_page(&db, "p1", 2).await;
    support_claim(&db, "p1", &revisions[0], 0.9).await;

    let outcome = db.evaluate_page_support("p1", 1).await.unwrap();
    assert!(
        matches!(outcome, SupportOutcome::Refuted { .. }),
        "got {outcome:?}"
    );
    db.finalize_page_support("p1", 1, &outcome).await.unwrap();
    let (status, evaluated, _) = truth_row(&db, "p1").await.unwrap();
    assert_eq!(status, "provisional");
    assert!(
        evaluated.is_some(),
        "a real verdict stamps evaluated_at — this is the one case that may cost a file"
    );
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
        db.finalize_page_support(id, 1, &outcome).await.unwrap();
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
    db.finalize_page_support("p1", 1, &outcome).await.unwrap();
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
    assert!(
        !db.finalize_page_support("p1", 1, &outcome).await.unwrap(),
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
    assert!(
        !db.finalize_page_support("p1", 1, &SupportOutcome::Supported)
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

    db.finalize_page_support("p1", 1, &SupportOutcome::Supported)
        .await
        .unwrap();
    let (status, _, reviewed) = truth_row(&db, "p1").await.unwrap();
    assert_eq!(status, "supported");
    assert_eq!(reviewed, 1, "a current approval survives a support write");
}
