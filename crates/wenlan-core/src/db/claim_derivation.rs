// SPDX-License-Identifier: Apache-2.0
//! M5 claim-derivation work queue (schema 105).
//!
//! `claim_derivation_jobs` shipped with PR-A's substrate but with no writer and
//! no reader, so on every real install the derivation backlog was permanently
//! empty and `support_status = 'supported'` matched zero rows. This module is
//! the queue half of the missing worker: what puts a page on the list, what
//! takes it off, and what happens when the worker holding it dies.
//!
//! Three entry points feed the queue, and the third is the one that matters on
//! an existing install:
//!
//! 1. **Insert trigger** — a new active page enqueues itself.
//! 2. **Update trigger** — a page whose version or text moved enqueues itself
//!    again, because the previous derivation described text that is gone.
//! 3. **Backlog scan** ([`MemoryDB::enqueue_stale_derivation_jobs`]) — every
//!    already-existing page whose current version carries no marker at the
//!    current extractor version.
//!
//! Without (3) the triggers alone would fire for nothing on a populated vault:
//! the pages are already there, so nothing inserts and nothing updates, the
//! queue stays empty, and the resulting zero reads as "no page needs
//! derivation" when the truth is "the substrate is dead". That is precisely the
//! misread this whole worker exists to end, so the backlog scan is not an
//! optimization — it is the difference between the queue being live and the
//! queue merely existing.
//!
//! Leases are the crash story. A worker takes a job with an expiry; if it dies,
//! the expiry passes and another worker reclaims it. Every terminal transition
//! is guarded on `lease_owner`, so a worker whose lease was reclaimed while it
//! was stalled cannot finish the job out from under its new owner — it fails
//! the guard and its write is a no-op.

use super::MemoryDB;
use crate::WenlanError;

/// Version of the claim extractor whose output a marker describes.
///
/// A marker is keyed by `(page_id, page_version)` but validated against this as
/// well: identical page text under a changed extractor yields a different claim
/// set, so a marker minted by an older extractor no longer describes the page.
/// Bumping this constant is therefore a re-derivation order for the whole
/// vault, which the backlog scan carries out.
pub const EXTRACTOR_VERSION: i64 = 1;

/// How long a leased job stays the leaseholder's before anyone may reclaim it.
///
/// Ten minutes is chosen against the slowest thing a derivation turn can do —
/// an on-device entailment pass — not against the fast path, because a lease
/// that expires under a worker that is merely slow causes two workers to derive
/// the same page concurrently.
pub const LEASE_SECS: i64 = 600;

/// How many leases a job may burn before it is parked for a human.
///
/// A job that keeps crashing its worker is a poison item; retrying it forever
/// starves the rest of the queue. `attempts` increments at lease time rather
/// than at failure time on purpose: a worker that dies mid-job never reaches
/// its failure handler, so counting failures would let a hard-crashing page
/// retry without limit.
pub const MAX_ATTEMPTS: i64 = 5;

/// A leased unit of derivation work.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivationJob {
    pub job_id: String,
    pub page_id: String,
    pub page_version: i64,
}

/// The score a stored entailment verdict must clear *now*.
///
/// Deliberately compared against the live constant rather than the edge's own
/// `threshold_at_write`. Row 15 of the truth-state matrix requires that raising
/// the bar demotes pages whose evidence no longer clears it; comparing each edge
/// to the threshold it was written under would make every stored verdict
/// permanently self-certifying.
pub const SUPPORT_THRESHOLD: f64 = 0.75;

/// The result of evaluating §1 of the truth-state matrix for one page version.
///
/// Three of the four variants all mean `support_status = 'provisional'`. They
/// are separate because `provisional` alone conflates never-looked-at with
/// looked-and-the-evidence-fell-short, and only the second may cost a page its
/// projected file. What distinguishes them on disk is `evaluated_at`.
#[derive(Debug, Clone, PartialEq)]
pub enum SupportOutcome {
    /// Every condition in §1 held.
    Supported,
    /// Derivation ran to completion over real inventory and the evidence fell
    /// short. The ONLY outcome besides `Supported` that stamps `evaluated_at`,
    /// and therefore the only one that can ever read as `Unsupported`.
    Refuted { reason: String },
    /// Nothing was judged — no marker, a marker describing text that is gone, or
    /// an inventory with nothing in it. Writes `provisional` but leaves
    /// `evaluated_at` alone, so the page stays `Unevaluated` and keeps its file.
    Unevaluated { reason: String },
    /// Row 6: a partial or malformed derivation publishes nothing at all. The
    /// page keeps whatever state it already had, so a run that completed some
    /// claims and lost others cannot flip a page on the strength of the ones
    /// that happened to finish first.
    NoPublish { reason: String },
}

/// The enqueue triggers, installed by migration 105.
///
/// `kind = 'entity'` pages are excluded: those are the M3 entity shadow pages,
/// projections of an `entities` row rather than distilled prose, so there is no
/// authored claim in them to derive or support.
///
/// The upsert reopens a `done` job rather than ignoring the conflict, because a
/// page whose text moved under a version that was already derived has a marker
/// describing text that no longer exists. Reopening is scoped to `done` so a
/// `leased` job is never stomped mid-flight; a leased job that is superseded
/// this way is caught instead at finalization, where the marker's
/// `page_version_digest` is re-checked against the live page.
pub(super) const ENQUEUE_TRIGGERS: &str = "
    CREATE TRIGGER IF NOT EXISTS m5_page_insert_enqueues_derivation
    AFTER INSERT ON pages
    WHEN NEW.status = 'active' AND NEW.kind <> 'entity'
    BEGIN
        INSERT INTO claim_derivation_jobs
            (job_id, page_id, page_version, status, attempts, created_at, updated_at)
        VALUES (NEW.id || ':' || NEW.version, NEW.id, NEW.version, 'pending', 0,
                CAST(strftime('%s','now') AS INTEGER),
                CAST(strftime('%s','now') AS INTEGER))
        ON CONFLICT(job_id) DO UPDATE SET
            status = 'pending',
            lease_owner = NULL,
            lease_expires_at = NULL,
            attempts = 0,
            last_error = NULL,
            updated_at = CAST(strftime('%s','now') AS INTEGER)
        WHERE claim_derivation_jobs.status = 'done';
    END;

    CREATE TRIGGER IF NOT EXISTS m5_page_update_enqueues_derivation
    AFTER UPDATE ON pages
    WHEN NEW.status = 'active' AND NEW.kind <> 'entity'
         AND (NEW.version <> OLD.version
              OR NEW.content <> OLD.content
              OR OLD.status <> 'active')
    BEGIN
        INSERT INTO claim_derivation_jobs
            (job_id, page_id, page_version, status, attempts, created_at, updated_at)
        VALUES (NEW.id || ':' || NEW.version, NEW.id, NEW.version, 'pending', 0,
                CAST(strftime('%s','now') AS INTEGER),
                CAST(strftime('%s','now') AS INTEGER))
        ON CONFLICT(job_id) DO UPDATE SET
            status = 'pending',
            lease_owner = NULL,
            lease_expires_at = NULL,
            attempts = 0,
            last_error = NULL,
            updated_at = CAST(strftime('%s','now') AS INTEGER)
        WHERE claim_derivation_jobs.status = 'done';
    END;
";

/// The pages whose claims are supported by one memory, as a subquery yielding
/// `page_id`.
///
/// Deliberately **not** scoped to `valid_until IS NULL`. Two triggers fire on
/// each of these events — the retraction in `claim_identity`, and the demotion
/// here — and SQLite does not define which runs first. A validity filter would
/// therefore make this find every affected page or none of them depending on an
/// ordering nobody controls. Matching regardless of validity is order-independent
/// and errs toward re-deriving a page whose edge was already withdrawn, which
/// costs one derivation and cannot cost correctness.
fn pages_supported_by_memory(memory_ref: &str) -> String {
    format!(
        "SELECT c.page_id
                       FROM edges e
                       JOIN claim_revisions cr ON cr.claim_revision_id = e.src_id
                       JOIN claims c ON c.claim_id = cr.claim_id
                      WHERE e.edge_type = 'supports'
                        AND e.src_kind = 'claim_revision'
                        AND e.dst_kind = 'memory'
                        AND e.dst_id = {memory_ref}"
    )
}

/// Demote the named pages out of `supported`, then queue them for re-derivation.
///
/// **Demotion is synchronous and enqueueing is not enough on its own.** Row 13
/// is "a supporting memory is deleted or moved": the edge is retracted, but
/// `page_truth_state` still says `supported` and every reader believes it until
/// some worker gets around to the job. On a branch whose producer does not exist
/// yet that is *forever*. Queueing the work and demoting the claim are answers
/// to different questions — when will we know again, and what do we say in the
/// meantime — so this does both.
///
/// `evaluated_at = NULL` is the whole safety argument for the demotion. It lands
/// the page on `Unevaluated`, not `Unsupported`: we have stopped asserting the
/// evidence backs the prose, without asserting that it doesn't. Only the second
/// costs a page its projected file, and losing a file because *our* evidence
/// bookkeeping changed is the mass-flip failure this rung exists to prevent.
fn support_demotion_body(affected_pages: &str) -> String {
    format!(
        "
                 UPDATE page_truth_state
                    SET support_status = 'provisional',
                        provisional_reason = 'supporting evidence was withdrawn; this page \
                                              needs re-derivation',
                        evaluated_at = NULL,
                        updated_at = CAST(strftime('%s','now') AS INTEGER)
                  WHERE support_status = 'supported'
                    AND page_id IN ({affected_pages});

                 INSERT INTO claim_derivation_jobs
                     (job_id, page_id, page_version, status, attempts, created_at, updated_at)
                 SELECT p.id || ':' || p.version, p.id, p.version, 'pending', 0,
                        CAST(strftime('%s','now') AS INTEGER),
                        CAST(strftime('%s','now') AS INTEGER)
                   FROM pages p
                  WHERE p.status = 'active'
                    AND p.kind <> 'entity'
                    AND p.id IN ({affected_pages})
                 ON CONFLICT(job_id) DO UPDATE SET
                     status = 'pending',
                     lease_owner = NULL,
                     lease_expires_at = NULL,
                     attempts = 0,
                     last_error = NULL,
                     updated_at = CAST(strftime('%s','now') AS INTEGER)
                  WHERE claim_derivation_jobs.status = 'done';
        "
    )
}

/// Pages whose published `supported` verdict no longer survives a fresh reading
/// of §1 condition 3, as a subquery yielding `(page_id, page_version)`.
///
/// This is the reconciliation half of rows 13 and 15. A marker check cannot see
/// either of them: the marker is about *extraction*, and both rows are about
/// evidence that stopped qualifying afterwards — an edge retracted when its
/// memory was deleted, or a threshold constant that rose under an edge nobody
/// touched. A page in that state has a current marker AND a `done` job, so the
/// marker-only scan skipped it and it stayed `supported` forever.
///
/// `?1` is the live threshold, deliberately, for the same reason
/// [`SUPPORT_THRESHOLD`] is compared live everywhere else: comparing each edge
/// to the bar it was written under makes every stored verdict self-certifying.
const DRIFTED_SUPPORTED_PAGES: &str = "
    SELECT p.id AS drifted_page_id, p.version AS drifted_page_version
      FROM pages p
      JOIN page_truth_state t
        ON t.page_id = p.id AND t.page_version = p.version
     WHERE p.status = 'active'
       AND p.kind <> 'entity'
       AND t.support_status = 'supported'
       AND EXISTS (
           SELECT 1 FROM page_version_claims pvc
            WHERE pvc.page_id = p.id AND pvc.page_version = p.version
              AND NOT EXISTS (
                  SELECT 1 FROM edges e
                   WHERE e.edge_type = 'supports'
                     AND e.src_kind = 'claim_revision'
                     AND e.src_id = pvc.claim_revision_id
                     AND e.valid_until IS NULL
                     AND e.superseded_by IS NULL
                     AND json_extract(e.payload, '$.score') >= ?1
              )
       )
";

impl MemoryDB {
    /// Install the enqueue triggers. Idempotent; safe on a resumed migration.
    pub(super) async fn ensure_claim_derivation_triggers(
        tx: &libsql::Transaction,
    ) -> Result<(), WenlanError> {
        tx.execute_batch(ENQUEUE_TRIGGERS).await.map_err(|error| {
            WenlanError::VectorDb(format!("m105 claim-derivation triggers: {error}"))
        })?;
        Ok(())
    }

    /// Install the support-invalidation demotion triggers (migration 105).
    ///
    /// Separate triggers with separate names rather than an edit to the
    /// retraction triggers `ensure_claim_edge_lifecycle_triggers` already
    /// installs: `CREATE TRIGGER IF NOT EXISTS` will not replace a trigger that
    /// exists, so amending those bodies would mean dropping and recreating
    /// proven objects on every install. These are additive and their effect does
    /// not depend on firing before or after the retraction — see
    /// [`pages_supported_by_memory`].
    ///
    /// There is no page-delete twin on purpose. `page_truth_state` and
    /// `claim_derivation_jobs` both cascade from `pages`, so a deleted page's
    /// truth row and queue entry are already gone; demoting a row on its way out
    /// and queueing derivation for a page that will not exist are both no-ops
    /// dressed as care.
    pub(super) async fn ensure_support_invalidation_triggers(
        tx: &libsql::Transaction,
    ) -> Result<(), WenlanError> {
        tx.execute_batch(&format!(
            "CREATE TRIGGER IF NOT EXISTS m5_demote_support_on_memory_delete
             BEFORE DELETE ON memories
             WHEN EXISTS (
                      SELECT 1 FROM edges
                       WHERE edge_type = 'supports' AND src_kind = 'claim_revision'
                         AND lineage = 'evidence'
                         AND dst_id = OLD.source_id
                  )
              AND NOT EXISTS (
                      SELECT 1 FROM memories
                       WHERE source_id = OLD.source_id AND id <> OLD.id
                  )
             BEGIN{delete_body}END;

             CREATE TRIGGER IF NOT EXISTS m5_demote_support_on_memory_space_move
             AFTER UPDATE OF space ON memories
             WHEN NEW.space IS NOT OLD.space
             BEGIN{memory_move_body}END;

             -- The page's own move retracts its claims' support edges, because
             -- they carry the space they were written under. Same loss of
             -- support, reached from the other end.
             CREATE TRIGGER IF NOT EXISTS m5_demote_support_on_page_space_move
             AFTER UPDATE OF space ON pages
             WHEN NEW.space IS NOT OLD.space
             BEGIN{page_move_body}END;",
            delete_body = support_demotion_body(&pages_supported_by_memory("OLD.source_id")),
            memory_move_body = support_demotion_body(&pages_supported_by_memory("NEW.source_id")),
            page_move_body = support_demotion_body("SELECT NEW.id"),
        ))
        .await
        .map_err(|error| {
            WenlanError::VectorDb(format!("m105 support-invalidation triggers: {error}"))
        })?;
        Ok(())
    }

    /// Enqueue up to `limit` already-existing pages that carry no valid marker.
    ///
    /// Returns how many jobs were created. A zero from this function means the
    /// vault is fully derived at the current extractor version — it is a real
    /// measurement, which is exactly what the empty queue before this worker
    /// was not.
    pub async fn enqueue_stale_derivation_jobs(&self, limit: i64) -> Result<usize, WenlanError> {
        let now = chrono::Utc::now().timestamp();
        let conn = self.conn.lock().await;

        // A 'done' job whose marker no longer satisfies the current extractor
        // is a stale claim of completion. Dropping it is what makes an
        // EXTRACTOR_VERSION bump re-derive the vault: the marker check below
        // already fails, but the job row would otherwise still say 'done' and
        // block the re-enqueue.
        conn.execute(
            "DELETE FROM claim_derivation_jobs
              WHERE status = 'done'
                AND NOT EXISTS (
                    SELECT 1 FROM claim_derivation_markers m
                     WHERE m.page_id = claim_derivation_jobs.page_id
                       AND m.page_version = claim_derivation_jobs.page_version
                       AND m.extractor_version = ?1
                )",
            libsql::params![EXTRACTOR_VERSION],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("derivation backlog sweep: {error}")))?;

        // The same stale-completion argument as above, for the drift rows: a
        // page whose support no longer clears the live bar has a `done` job
        // occupying its unique (page_id, page_version) slot, and that row would
        // block the re-enqueue below exactly the way a stale marker's does.
        conn.execute(
            &format!(
                "DELETE FROM claim_derivation_jobs
                  WHERE status = 'done'
                    AND (page_id, page_version) IN (
                        SELECT drifted_page_id, drifted_page_version
                          FROM ({DRIFTED_SUPPORTED_PAGES})
                    )"
            ),
            libsql::params![SUPPORT_THRESHOLD],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("derivation drift sweep: {error}")))?;

        let created = conn
            .execute(
                "INSERT INTO claim_derivation_jobs
                     (job_id, page_id, page_version, status, attempts, created_at, updated_at)
                 SELECT p.id || ':' || p.version, p.id, p.version, 'pending', 0, ?1, ?1
                   FROM pages p
                  WHERE p.status = 'active'
                    AND p.kind <> 'entity'
                    AND NOT EXISTS (
                        SELECT 1 FROM claim_derivation_markers m
                         WHERE m.page_id = p.id
                           AND m.page_version = p.version
                           AND m.extractor_version = ?2
                    )
                    AND NOT EXISTS (
                        SELECT 1 FROM claim_derivation_jobs j
                         WHERE j.page_id = p.id AND j.page_version = p.version
                    )
                  ORDER BY p.id
                  LIMIT ?3",
                libsql::params![now, EXTRACTOR_VERSION, limit],
            )
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!("derivation backlog enqueue: {error}"))
            })?;

        let re_derive = conn
            .execute(
                &format!(
                    "INSERT INTO claim_derivation_jobs
                         (job_id, page_id, page_version, status, attempts, created_at, updated_at)
                     SELECT d.drifted_page_id || ':' || d.drifted_page_version,
                            d.drifted_page_id, d.drifted_page_version, 'pending', 0, ?2, ?2
                       FROM ({DRIFTED_SUPPORTED_PAGES}) d
                      WHERE NOT EXISTS (
                          SELECT 1 FROM claim_derivation_jobs j
                           WHERE j.page_id = d.drifted_page_id
                             AND j.page_version = d.drifted_page_version
                      )
                      ORDER BY d.drifted_page_id
                      LIMIT ?3"
                ),
                libsql::params![SUPPORT_THRESHOLD, now, limit],
            )
            .await
            .map_err(|error| WenlanError::VectorDb(format!("derivation drift enqueue: {error}")))?;

        Ok((created + re_derive) as usize)
    }

    /// Run [`Self::enqueue_stale_derivation_jobs`] until the backlog is drained,
    /// in batches of `batch`. Returns the total enqueued.
    ///
    /// The bound on a single scan is a boot-latency bound, not a policy: the
    /// migration runs while the daemon is still coming up, and a full scan of a
    /// large vault would hold every foreground request behind it. Bounding it
    /// and then never continuing is a different thing entirely — a vault with
    /// 1,200 pages had 500 queued and 700 that no scan would ever reach, so the
    /// backlog sweep converged on a *subset* and the queue was quietly partial
    /// on exactly the installs it exists for. This is the continuation: the
    /// batch keeps each statement short, the loop keeps the sweep total.
    ///
    /// Termination is bounded by the data, not by trust. A pass that fills its
    /// batch may have more behind it; a pass that comes back short has reached
    /// the end, because every row a pass creates fails the next pass's
    /// `NOT EXISTS` guard. Worst case is `ceil(pages / batch)` passes.
    ///
    /// ponytail: each pass restarts the `pages` scan, so draining a vault of n
    /// pages costs O(n²/batch) index probes. Irrelevant at the thousands of
    /// pages a personal vault holds; if one ever outgrows that, the fix is a
    /// keyset cursor on `p.id`, not a bigger batch.
    pub async fn drain_stale_derivation_jobs(&self, batch: i64) -> Result<usize, WenlanError> {
        let batch = batch.max(1);
        let mut total = 0usize;
        loop {
            let created = self.enqueue_stale_derivation_jobs(batch).await?;
            total += created;
            if created < batch as usize {
                return Ok(total);
            }
        }
    }

    /// Take the oldest claimable job, or `None` when the queue is drained.
    ///
    /// Claimable means pending, or leased with an expired lease — the second
    /// half is the reclaim path the lease columns exist for. The select and the
    /// update are one statement so two workers racing cannot both win: SQLite
    /// serializes the write, and the loser's subquery re-evaluates against the
    /// already-leased row.
    pub async fn lease_next_derivation_job(
        &self,
        owner: &str,
    ) -> Result<Option<DerivationJob>, WenlanError> {
        let now = chrono::Utc::now().timestamp();
        let expires = now + LEASE_SECS;
        let conn = self.conn.lock().await;

        let mut rows = conn
            .query(
                "UPDATE claim_derivation_jobs
                    SET status = 'leased',
                        lease_owner = ?1,
                        lease_expires_at = ?2,
                        attempts = attempts + 1,
                        updated_at = ?3
                  WHERE job_id = (
                      SELECT job_id FROM claim_derivation_jobs
                       WHERE attempts < ?4
                         AND (status = 'pending'
                              OR (status = 'leased'
                                  AND lease_expires_at IS NOT NULL
                                  AND lease_expires_at <= ?3))
                       ORDER BY created_at, job_id
                       LIMIT 1
                  )
                  RETURNING job_id, page_id, page_version",
                libsql::params![owner, expires, now, MAX_ATTEMPTS],
            )
            .await
            .map_err(|error| WenlanError::VectorDb(format!("derivation lease: {error}")))?;

        let Some(row) = rows
            .next()
            .await
            .map_err(|error| WenlanError::VectorDb(format!("derivation lease row: {error}")))?
        else {
            return Ok(None);
        };

        Ok(Some(DerivationJob {
            job_id: row.get::<String>(0).map_err(|error| {
                WenlanError::VectorDb(format!("derivation lease job_id: {error}"))
            })?,
            page_id: row.get::<String>(1).map_err(|error| {
                WenlanError::VectorDb(format!("derivation lease page_id: {error}"))
            })?,
            page_version: row.get::<i64>(2).map_err(|error| {
                WenlanError::VectorDb(format!("derivation lease page_version: {error}"))
            })?,
        }))
    }

    /// Mark a job done. Returns false when the caller no longer holds the lease.
    ///
    /// The `lease_owner` guard is the whole point: a worker that stalled past
    /// its lease, had the job reclaimed, and then woke up must not be able to
    /// declare the job finished — the new owner is mid-derivation and the old
    /// worker's result describes a turn nobody is waiting for.
    pub async fn finish_derivation_job(
        &self,
        job_id: &str,
        owner: &str,
    ) -> Result<bool, WenlanError> {
        let now = chrono::Utc::now().timestamp();
        let conn = self.conn.lock().await;
        let changed = conn
            .execute(
                "UPDATE claim_derivation_jobs
                    SET status = 'done',
                        lease_owner = NULL,
                        lease_expires_at = NULL,
                        last_error = NULL,
                        updated_at = ?1
                  WHERE job_id = ?2 AND status = 'leased' AND lease_owner = ?3",
                libsql::params![now, job_id, owner],
            )
            .await
            .map_err(|error| WenlanError::VectorDb(format!("derivation finish: {error}")))?;
        Ok(changed > 0)
    }

    /// Hand a failed job back to the queue, recording why.
    ///
    /// The attempt is already spent — `attempts` incremented at lease time — so
    /// this only returns the job to `pending`; the park sweep is what retires it
    /// once the attempts are exhausted.
    pub async fn release_derivation_job(
        &self,
        job_id: &str,
        owner: &str,
        error_text: &str,
    ) -> Result<bool, WenlanError> {
        let now = chrono::Utc::now().timestamp();
        let conn = self.conn.lock().await;
        let changed = conn
            .execute(
                "UPDATE claim_derivation_jobs
                    SET status = 'pending',
                        lease_owner = NULL,
                        lease_expires_at = NULL,
                        last_error = ?1,
                        updated_at = ?2
                  WHERE job_id = ?3 AND status = 'leased' AND lease_owner = ?4",
                libsql::params![error_text, now, job_id, owner],
            )
            .await
            .map_err(|error| WenlanError::VectorDb(format!("derivation release: {error}")))?;
        Ok(changed > 0)
    }

    /// Evaluate §1 of the truth-state matrix for one page version.
    ///
    /// Read-only and free of model work: this is phase 2's *verdict*, computed
    /// from what phase 1's extraction and judging already wrote down. Keeping it
    /// out of a transaction is what lets phase 3 be short.
    ///
    /// The four conditions, in order, each with the outcome its failure yields:
    ///
    /// 1. An exact-page-version marker exists whose recorded digest equals the
    ///    current page-version digest AND whose `extractor_version` equals the
    ///    current one. Failure → `Unevaluated`: a missing or stale marker is not
    ///    a judgement about evidence, it is the absence of one.
    /// 2. The marker's membership inventory is nonempty. Failure →
    ///    `Unevaluated`. See `empty_inventory_is_unevaluated_not_refuted` for
    ///    why this is not `Refuted`.
    /// 3. Every active claim revision in the inventory has at least one
    ///    `supports` edge that is simultaneously active and above threshold.
    ///    Failure → `Refuted`. This is the one condition whose failure is a
    ///    verdict, because reaching it means the derivation ran, the claims are
    ///    real, and the evidence was looked for and was not there.
    /// 4. No revision is in a deferred, timed-out, or malformed support state.
    ///    Detected as a membership count that disagrees with the marker's own
    ///    `inventory_count` — the on-disk signature of a derivation that stopped
    ///    partway. Failure → `NoPublish` (row 6).
    ///
    /// **Known gap, condition 3.** The matrix also requires the supporting edge
    /// to come from a "currently-eligible model version" (row 14). No model
    /// eligibility registry exists anywhere in the tree yet, so that clause is
    /// unimplementable today and this function does not pretend otherwise: it
    /// accepts a qualifying score from any judge. The gap is not currently
    /// reachable — nothing writes support edges in production — but it must be
    /// closed before one does.
    pub async fn evaluate_page_support(
        &self,
        page_id: &str,
        page_version: i64,
    ) -> Result<SupportOutcome, WenlanError> {
        let conn = self.conn.lock().await;
        Self::evaluate_support_on(&conn, page_id, page_version).await
    }

    /// [`Self::evaluate_page_support`] over a connection the caller already
    /// holds.
    ///
    /// It exists so [`Self::finalize_page_support`] can recompute the whole
    /// verdict *inside its own transaction* rather than trust one handed to it.
    /// Everything the verdict rests on — the page digest, the marker, the
    /// inventory, edge validity, the live threshold — can move between phase 2
    /// and phase 3 without the page version changing, so re-reading the page
    /// version alone was never enough to make a publication safe.
    async fn evaluate_support_on(
        conn: &libsql::Connection,
        page_id: &str,
        page_version: i64,
    ) -> Result<SupportOutcome, WenlanError> {
        let (live_version, live_digest) = {
            let mut rows = conn
                .query(
                    "SELECT version, content FROM pages WHERE id = ?1 AND status = 'active'",
                    libsql::params![page_id],
                )
                .await
                .map_err(|error| WenlanError::VectorDb(format!("support page read: {error}")))?;
            let Some(row) = rows
                .next()
                .await
                .map_err(|error| WenlanError::VectorDb(format!("support page decode: {error}")))?
            else {
                return Ok(SupportOutcome::NoPublish {
                    reason: format!("page {page_id} is gone or no longer active"),
                });
            };
            let version: i64 = row
                .get(0)
                .map_err(|error| WenlanError::VectorDb(format!("support page decode: {error}")))?;
            let content: String = row
                .get(1)
                .map_err(|error| WenlanError::VectorDb(format!("support page decode: {error}")))?;
            (
                version,
                crate::provenance::revision_content_digest(&content),
            )
        };

        // The page moved while the job was in flight. Its own enqueue trigger
        // has already queued the new version; publishing against the old one
        // would attach a verdict to text nobody judged.
        if live_version != page_version {
            return Ok(SupportOutcome::NoPublish {
                reason: format!("page {page_id} moved to version {live_version} mid-derivation"),
            });
        }

        // Condition 1.
        let marker = {
            let mut rows = conn
                .query(
                    "SELECT page_version_digest, extractor_version, inventory_count
                       FROM claim_derivation_markers
                      WHERE page_id = ?1 AND page_version = ?2",
                    libsql::params![page_id, page_version],
                )
                .await
                .map_err(|error| WenlanError::VectorDb(format!("support marker read: {error}")))?;
            match rows
                .next()
                .await
                .map_err(|error| WenlanError::VectorDb(format!("support marker decode: {error}")))?
            {
                Some(row) => {
                    let digest: String = row.get(0).map_err(|error| {
                        WenlanError::VectorDb(format!("support marker decode: {error}"))
                    })?;
                    let extractor: i64 = row.get(1).map_err(|error| {
                        WenlanError::VectorDb(format!("support marker decode: {error}"))
                    })?;
                    let count: i64 = row.get(2).map_err(|error| {
                        WenlanError::VectorDb(format!("support marker decode: {error}"))
                    })?;
                    Some((digest, extractor, count))
                }
                None => None,
            }
        };
        let Some((marker_digest, marker_extractor, inventory_count)) = marker else {
            return Ok(SupportOutcome::Unevaluated {
                reason: "no derivation marker: this page version has never been derived".into(),
            });
        };
        if marker_extractor != EXTRACTOR_VERSION {
            return Ok(SupportOutcome::Unevaluated {
                reason: format!(
                    "marker was written by extractor {marker_extractor}, not {EXTRACTOR_VERSION}"
                ),
            });
        }
        if marker_digest != live_digest {
            return Ok(SupportOutcome::Unevaluated {
                reason: "marker describes different page text than the page now holds".into(),
            });
        }

        // Condition 4, checked before 2 and 3: a membership count that disagrees
        // with the marker is a derivation that stopped partway, and a partial
        // run must publish nothing rather than be read as an empty or a failing
        // one.
        let membership: i64 = {
            let mut rows = conn
                .query(
                    "SELECT COUNT(*) FROM page_version_claims
                      WHERE page_id = ?1 AND page_version = ?2",
                    libsql::params![page_id, page_version],
                )
                .await
                .map_err(|error| {
                    WenlanError::VectorDb(format!("support membership read: {error}"))
                })?;
            rows.next()
                .await
                .map_err(|error| {
                    WenlanError::VectorDb(format!("support membership decode: {error}"))
                })?
                .map(|row| row.get::<i64>(0))
                .transpose()
                .map_err(|error| {
                    WenlanError::VectorDb(format!("support membership decode: {error}"))
                })?
                .unwrap_or(0)
        };
        if membership != inventory_count {
            return Ok(SupportOutcome::NoPublish {
                reason: format!(
                    "partial derivation: marker claims {inventory_count} claims, \
                     {membership} are on record"
                ),
            });
        }

        // Condition 2, the vacuous-truth guard. `forall x in {}` is trivially
        // true in every natural implementation, so an empty inventory has to be
        // refused explicitly or an underived page reads as fully supported.
        if inventory_count == 0 {
            return Ok(SupportOutcome::Unevaluated {
                reason: "derivation produced no claims: there is nothing here to support".into(),
            });
        }

        // Condition 3, split by WHY a claim is unsupported. `provisional`
        // conflates two situations that call for opposite responses, and the
        // split has to reach the STORED reason or the distinction dies at the
        // point a human would use it:
        //
        //   never judged -- no judge has ever scored this claim's text. The
        //                   gathering or judging step did not happen, so we have
        //                   not weighed this page at all.
        //   judged short -- a judge scored this text and nothing cleared the
        //                   bar. That IS a verdict about the evidence.
        //
        // The discriminator is a row in `entailment_cache` for the claim's
        // canonical text digest, which exists if and only if some judge scored
        // that exact text under some model and prompt.
        let (never_judged, judged_short): (i64, i64) =
            {
                let mut rows = conn
                    .query(
                        "SELECT
                         SUM(CASE WHEN ec.hit IS NULL THEN 1 ELSE 0 END),
                         SUM(CASE WHEN ec.hit IS NULL THEN 0 ELSE 1 END)
                       FROM page_version_claims pvc
                       JOIN claim_revisions cr
                         ON cr.claim_revision_id = pvc.claim_revision_id
                       LEFT JOIN (
                           SELECT DISTINCT claim_text_digest AS digest, 1 AS hit
                             FROM entailment_cache
                       ) ec ON ec.digest = cr.canonical_text_digest
                      WHERE pvc.page_id = ?1 AND pvc.page_version = ?2
                        AND NOT EXISTS (
                            SELECT 1 FROM edges e
                             WHERE e.edge_type = 'supports'
                               AND e.src_kind = 'claim_revision'
                               AND e.src_id = pvc.claim_revision_id
                               AND e.valid_until IS NULL
                               AND e.superseded_by IS NULL
                               AND json_extract(e.payload, '$.score') >= ?3
                        )",
                        libsql::params![page_id, page_version, SUPPORT_THRESHOLD],
                    )
                    .await
                    .map_err(|error| WenlanError::VectorDb(format!("support scan: {error}")))?;
                match rows.next().await.map_err(|error| {
                    WenlanError::VectorDb(format!("support scan decode: {error}"))
                })? {
                    Some(row) => (
                        row.get::<Option<i64>>(0)
                            .map_err(|error| {
                                WenlanError::VectorDb(format!("support scan decode: {error}"))
                            })?
                            .unwrap_or(0),
                        row.get::<Option<i64>>(1)
                            .map_err(|error| {
                                WenlanError::VectorDb(format!("support scan decode: {error}"))
                            })?
                            .unwrap_or(0),
                    ),
                    None => (0, 0),
                }
            };

        // A claim nobody has judged means the derivation has not run to
        // completion over this inventory, whatever the marker says about
        // extraction. Calling that `Refuted` would stamp `evaluated_at` and cost
        // the page its file for a gap in OUR pipeline rather than a fact about
        // the page -- the mass-flip failure wearing a different hat. Fail closed
        // to `Unevaluated`, and name how many and why.
        if never_judged > 0 {
            return Ok(SupportOutcome::Unevaluated {
                reason: format!(
                    "no candidate evidence: {never_judged} of {inventory_count} claim(s) have \
                     never been scored by any judge{}",
                    if judged_short > 0 {
                        format!(" ({judged_short} more were judged and fell short)")
                    } else {
                        String::new()
                    }
                ),
            });
        }
        if judged_short > 0 {
            return Ok(SupportOutcome::Refuted {
                reason: format!(
                    "candidates judged and fell short: {judged_short} of {inventory_count} \
                     claim(s) have no active support edge at or above threshold \
                     {SUPPORT_THRESHOLD}"
                ),
            });
        }

        Ok(SupportOutcome::Supported)
    }

    /// Phase 3: publish an outcome. Returns whether anything was written.
    ///
    /// This is the ONLY place `evaluated_at` is stamped, and it is stamped
    /// inside the same short transaction that writes the status it belongs to.
    /// Stamping at lease time instead would mean that every page the worker
    /// merely *looked at* — including ones it crashed on, ones whose evidence it
    /// could not resolve, and ones it skipped — became `Unsupported` the moment
    /// the cutover generation advanced, which is the whole-vault archive this
    /// worker exists to make impossible.
    ///
    /// **Publication is bound to the lease, and to evidence that has not moved.**
    /// Two guards run inside the transaction, before anything is written:
    ///
    /// 1. The job named by `job_id` must still be `leased` by `owner`. The
    ///    `lease_owner` guard on [`Self::finish_derivation_job`] protected only
    ///    the *queue*, not the truth row: a worker that stalled past its lease
    ///    could be reclaimed, overtaken by a worker that reached the opposite
    ///    verdict, and still overwrite that published verdict on waking. Its
    ///    later `finish` failed, but the stale truth write had already landed.
    ///    A parked job's former worker had the same opening.
    /// 2. The verdict is **recomputed here** and must equal the one handed in.
    ///    Re-reading the page version cannot close this: a support edge can be
    ///    retracted, a marker can be superseded, and the threshold constant can
    ///    rise, all without the page version moving. Recomputing under the same
    ///    transaction that writes is the only comparison that covers every input
    ///    at once, and it costs one extra read of rows this function was already
    ///    going to touch.
    ///
    /// Both guards *refuse* rather than repair — a mismatch returns `false` and
    /// writes nothing, leaving the job for whoever legitimately holds it. This
    /// module refuses rather than repairs everywhere else for the same reason:
    /// a worker that publishes a verdict it did not compute is indistinguishable
    /// from one that did.
    pub async fn finalize_page_support(
        &self,
        page_id: &str,
        page_version: i64,
        job_id: &str,
        owner: &str,
        outcome: &SupportOutcome,
    ) -> Result<bool, WenlanError> {
        if let SupportOutcome::NoPublish { .. } = outcome {
            return Ok(false);
        }
        let now = chrono::Utc::now().timestamp();
        let conn = self.conn.lock().await;
        let tx = conn
            .transaction_with_behavior(libsql::TransactionBehavior::Immediate)
            .await
            .map_err(|error| WenlanError::VectorDb(format!("support finalize begin: {error}")))?;

        let published = async {
            // Guard 1: the lease. `page_id`/`page_version` are matched too, so a
            // job id cannot be spent against a different page than the one it
            // names.
            let holds_lease = {
                let mut rows = tx
                    .query(
                        "SELECT 1 FROM claim_derivation_jobs
                          WHERE job_id = ?1 AND page_id = ?2 AND page_version = ?3
                            AND status = 'leased' AND lease_owner = ?4",
                        libsql::params![job_id, page_id, page_version, owner],
                    )
                    .await
                    .map_err(|error| {
                        WenlanError::VectorDb(format!("support finalize lease: {error}"))
                    })?;
                rows.next()
                    .await
                    .map_err(|error| {
                        WenlanError::VectorDb(format!("support finalize lease decode: {error}"))
                    })?
                    .is_some()
            };
            if !holds_lease {
                return Ok(false);
            }

            // Guard 2: the verdict, recomputed against the state this
            // transaction is about to write into. A moved page reaches here as
            // `NoPublish`, which never equals a publishable outcome, so the
            // version check the old code did by hand is subsumed rather than
            // dropped.
            if Self::evaluate_support_on(&tx, page_id, page_version).await? != *outcome {
                return Ok(false);
            }

            let (status, reason, stamp) = match outcome {
                SupportOutcome::Supported => ("supported", None, Some(now)),
                SupportOutcome::Refuted { reason } => {
                    ("provisional", Some(reason.clone()), Some(now))
                }
                // No stamp, and the stored one is CLEARED rather than kept.
                // `evaluated_at` answers "has this page version been judged",
                // and `page_truth_state` holds one row per PAGE, so carrying the
                // old value forward answers it about a version that is gone: a
                // v1 `Refuted` stamp surviving a v2 `Unevaluated` publish makes
                // v2 read as Unsupported and costs it its file — the exact
                // inversion of the fail-safe this variant exists to give.
                SupportOutcome::Unevaluated { reason } => {
                    ("provisional", Some(reason.clone()), None)
                }
                SupportOutcome::NoPublish { .. } => return Ok(false),
            };

            // `human_reviewed` is never written here — the axes are independent,
            // and machine support may not manufacture human review. It is only
            // ever CLEARED, and only when the stored approval names a different
            // version than the one being published (matrix row 8). Losing trust
            // on a text change is safe; the reverse never happens on this path.
            tx.execute(
                "INSERT INTO page_truth_state
                     (page_id, page_version, support_status, provisional_reason,
                      evaluated_at, human_reviewed, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, 0, ?6)
                 ON CONFLICT(page_id) DO UPDATE SET
                     page_version = ?2,
                     support_status = ?3,
                     provisional_reason = ?4,
                     evaluated_at = ?5,
                     human_reviewed = CASE
                         WHEN page_truth_state.reviewed_page_version = ?2
                         THEN page_truth_state.human_reviewed ELSE 0 END,
                     updated_at = ?6",
                libsql::params![page_id, page_version, status, reason, stamp, now],
            )
            .await
            .map_err(|error| WenlanError::VectorDb(format!("support finalize write: {error}")))?;
            Ok::<bool, WenlanError>(true)
        }
        .await;

        match published {
            Ok(true) => {
                tx.commit().await.map_err(|error| {
                    WenlanError::VectorDb(format!("support finalize commit: {error}"))
                })?;
                Ok(true)
            }
            Ok(false) => {
                let _ = tx.rollback().await;
                Ok(false)
            }
            Err(error) => {
                let _ = tx.rollback().await;
                Err(error)
            }
        }
    }

    /// Retire jobs that have burned every attempt. Returns how many were parked.
    ///
    /// Parked is a terminal state on purpose: the page keeps whatever truth
    /// state it already had (provisional and unevaluated, for a page that never
    /// derived), which is the honest record of "we could not judge this" — a
    /// worker that cannot finish must never be the reason a page is called
    /// Unsupported.
    ///
    /// A `leased` job is only parked once its lease has **expired**. Attempts
    /// are counted at lease time (see [`MAX_ATTEMPTS`]), so the worker holding
    /// the last one is at `attempts == MAX_ATTEMPTS` from the moment it starts:
    /// parking on the count alone retires a run that is healthy and still going,
    /// and its finish then fails for a reason that has nothing to do with it.
    /// Four daemon restarts followed by one good run is an ordinary history, not
    /// a poison item. Expiry is what distinguishes the two, so expiry is what
    /// this waits for; a `pending` job has no run to interrupt and parks at once.
    pub async fn park_exhausted_derivation_jobs(&self) -> Result<usize, WenlanError> {
        let now = chrono::Utc::now().timestamp();
        let conn = self.conn.lock().await;
        let changed = conn
            .execute(
                "UPDATE claim_derivation_jobs
                    SET status = 'parked',
                        lease_owner = NULL,
                        lease_expires_at = NULL,
                        updated_at = ?1
                  WHERE attempts >= ?2
                    AND (status = 'pending'
                         OR (status = 'leased'
                             AND lease_expires_at IS NOT NULL
                             AND lease_expires_at <= ?1))",
                libsql::params![now, MAX_ATTEMPTS],
            )
            .await
            .map_err(|error| WenlanError::VectorDb(format!("derivation park: {error}")))?;
        Ok(changed as usize)
    }
}
