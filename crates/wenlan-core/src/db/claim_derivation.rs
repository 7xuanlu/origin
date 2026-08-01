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

        // Condition 3.
        let unsupported: i64 = {
            let mut rows = conn
                .query(
                    "SELECT COUNT(*) FROM page_version_claims pvc
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
            rows.next()
                .await
                .map_err(|error| WenlanError::VectorDb(format!("support scan decode: {error}")))?
                .map(|row| row.get::<i64>(0))
                .transpose()
                .map_err(|error| WenlanError::VectorDb(format!("support scan decode: {error}")))?
                .unwrap_or(0)
        };
        if unsupported > 0 {
            return Ok(SupportOutcome::Refuted {
                reason: format!(
                    "{unsupported} of {inventory_count} claim(s) have no active support edge \
                     above threshold {SUPPORT_THRESHOLD}"
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
    /// The page version is re-read under the transaction and the write is
    /// abandoned if it moved: model work happened outside any lock, so the text
    /// the verdict describes may no longer be the text on disk.
    pub async fn finalize_page_support(
        &self,
        page_id: &str,
        page_version: i64,
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
            let live_version: Option<i64> = {
                let mut rows = tx
                    .query(
                        "SELECT version FROM pages WHERE id = ?1 AND status = 'active'",
                        libsql::params![page_id],
                    )
                    .await
                    .map_err(|error| {
                        WenlanError::VectorDb(format!("support finalize page: {error}"))
                    })?;
                match rows.next().await.map_err(|error| {
                    WenlanError::VectorDb(format!("support finalize page decode: {error}"))
                })? {
                    Some(row) => Some(row.get(0).map_err(|error| {
                        WenlanError::VectorDb(format!("support finalize page decode: {error}"))
                    })?),
                    None => None,
                }
            };
            if live_version != Some(page_version) {
                return Ok(false);
            }

            let (status, reason, stamp) = match outcome {
                SupportOutcome::Supported => ("supported", None, Some(now)),
                SupportOutcome::Refuted { reason } => {
                    ("provisional", Some(reason.clone()), Some(now))
                }
                // No stamp: `evaluated_at` stays whatever it was, which for a
                // page that has never been judged is NULL.
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
                     evaluated_at = COALESCE(?5, page_truth_state.evaluated_at),
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
