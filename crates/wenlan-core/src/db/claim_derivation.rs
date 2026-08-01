// SPDX-License-Identifier: Apache-2.0
//! M5 claim-derivation work queue (schema 104).
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

/// The enqueue triggers, installed by migration 104.
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

impl MemoryDB {
    /// Install the enqueue triggers. Idempotent; safe on a resumed migration.
    pub(super) async fn ensure_claim_derivation_triggers(
        tx: &libsql::Transaction,
    ) -> Result<(), WenlanError> {
        tx.execute_batch(ENQUEUE_TRIGGERS).await.map_err(|error| {
            WenlanError::VectorDb(format!("m104 claim-derivation triggers: {error}"))
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

        Ok(created as usize)
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

    /// Retire jobs that have burned every attempt. Returns how many were parked.
    ///
    /// Parked is a terminal state on purpose: the page keeps whatever truth
    /// state it already had (provisional and unevaluated, for a page that never
    /// derived), which is the honest record of "we could not judge this" — a
    /// worker that cannot finish must never be the reason a page is called
    /// Unsupported.
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
                  WHERE status IN ('pending','leased') AND attempts >= ?2",
                libsql::params![now, MAX_ATTEMPTS],
            )
            .await
            .map_err(|error| WenlanError::VectorDb(format!("derivation park: {error}")))?;
        Ok(changed as usize)
    }
}
