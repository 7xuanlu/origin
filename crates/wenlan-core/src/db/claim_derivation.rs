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
//! is guarded on the exact leased [`DerivationJob`], including its monotone run
//! generation, so a worker whose lease was reclaimed while it was stalled
//! cannot finish the successor even when both runs use the same owner string.

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

/// Where a judged candidate lives, to the byte.
///
/// The exact tuple `write_support_edge` folds into a support edge's identity,
/// and the exact tuple `claim_judgment_attempts` records. A span digest alone
/// does not identify a candidate — one document can hold the same sentence
/// twice, and under a digest-only key concluding on the first occurrence
/// silently discharges the obligation to conclude on the second.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateLocator {
    pub source_id: String,
    pub chunk_index: i64,
    pub span_start: i64,
    pub span_end: i64,
    pub span_digest: String,
}

/// One live `supports` edge, with everything condition 3 and condition 4 need.
struct SupportCandidate {
    source_id: String,
    root_id: Option<String>,
    payload: String,
    /// `None` when the payload does not say which bytes were judged, which is
    /// unmatchable against any attempt row and therefore unregistered.
    locator: Option<CandidateLocator>,
}

/// A leased unit of derivation work.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivationJob {
    pub job_id: String,
    pub page_id: String,
    pub page_version: i64,
    /// The run this lease started. Every `claim_judgment_attempts` row the
    /// worker writes must carry it, and evaluation reads only rows that do —
    /// see [`MemoryDB::lease_next_derivation_job`].
    pub run_generation: i64,
}

/// The score a stored entailment verdict must clear *now*.
///
/// Deliberately compared against the live constant rather than the edge's own
/// `threshold_at_write`. Row 15 of the truth-state matrix requires that raising
/// the bar demotes pages whose evidence no longer clears it; comparing each edge
/// to the threshold it was written under would make every stored verdict
/// permanently self-certifying.
pub const SUPPORT_THRESHOLD: f64 = 0.75;

/// The one bounded claim-derivation write migration 105 may perform while
/// `MemoryDB::new` is still on the pre-serve startup path. Runtime continuation
/// drains everything beyond this seed after the listener can accept requests.
pub(super) const MIGRATION_105_BACKLOG_SEED_LIMIT: i64 = 500;

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

/// The pages whose claims are supported by one *chunk* of one memory.
///
/// N1's subject. A `source_id` names a document and a document is several
/// `memories` rows, so "this memory is gone" and "the chunk this verdict read is
/// gone" are different events — and the second is the one a support edge is
/// actually about. The existing whole-source triggers deliberately fire only
/// when the LAST row for a source disappears, which is correct for what they
/// guard and blind to a merge or update that drops chunk 1 and keeps chunk 0.
///
/// The chunk comes out of the edge payload rather than a column, because that is
/// where `write_support_edge` records the chunk it verified; the same value is
/// the leading field of the edge's span locator.
///
/// Not scoped to `valid_until IS NULL`, for the reason
/// [`pages_supported_by_memory`] gives at length: trigger order is undefined, so
/// a validity filter would make this find every affected page or none depending
/// on which trigger SQLite happened to run first.
fn pages_supported_by_memory_chunk(memory_ref: &str, chunk_ref: &str) -> String {
    format!(
        "SELECT c.page_id
                       FROM edges e
                       JOIN claim_revisions cr ON cr.claim_revision_id = e.src_id
                       JOIN claims c ON c.claim_id = cr.claim_id
                      WHERE e.edge_type = 'supports'
                        AND e.src_kind = 'claim_revision'
                        AND e.dst_kind = 'memory'
                        AND e.dst_id = {memory_ref}
                        AND json_extract(e.payload, '$.chunk_index') = {chunk_ref}"
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
/// Why a page was demoted when the evidence under it was withdrawn.
const REASON_EVIDENCE_WITHDRAWN: &str =
    "supporting evidence was withdrawn; this page needs re-derivation";

/// Why a page was demoted when the document its verdict cites was renamed.
const REASON_EVIDENCE_REBOUND: &str =
    "the cited document was renamed; this page needs re-derivation";

/// Why a page was demoted when startup could not re-prove its stored verdict.
const REASON_RECONCILE_UNPROVEN: &str =
    "the stored support verdict no longer survives re-evaluation; this page needs re-derivation";

/// `app_metadata` key holding how far the current reconciliation pass has got.
///
/// A page ID (the last one re-proved), the empty string (a pass has begun and
/// re-proved nothing yet), or [`SUPPORT_RECONCILE_COMPLETE`]. Absent means no
/// pass has ever run here.
const SUPPORT_RECONCILE_FRONTIER_KEY: &str = "support_reconcile_frontier";

/// Which implementation of the whole support predicate the durable frontier
/// was proved under. Bump the first component whenever reconciliation semantics
/// change without changing the extractor or threshold.
const SUPPORT_RECONCILE_RULESET_KEY: &str = "support_reconcile_ruleset";
const SUPPORT_RECONCILE_RULESET_VERSION: i64 = 2;

/// The frontier value meaning the pass finished. Not a page ID, and it cannot
/// collide with one: page IDs are `page_`-prefixed and never contain a space.
const SUPPORT_RECONCILE_COMPLETE: &str = "complete pass";

fn support_reconcile_ruleset() -> String {
    format!(
        "{SUPPORT_RECONCILE_RULESET_VERSION}:{EXTRACTOR_VERSION}:{:016x}",
        SUPPORT_THRESHOLD.to_bits()
    )
}

/// One batch of [`MemoryDB::reconcile_supported_pages`].
///
/// `complete` is the load-bearing half. The caller uses it to decide whether to
/// keep going, and the READ path uses the same durable cursor to decide whether
/// a stored `supported` may be asserted yet — so a caller that stops early
/// degrades the pages it never reached rather than serving them unchecked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SupportReconcilePass {
    /// Pages this batch demoted out of `supported`.
    pub demoted: usize,
    /// Whether the pass has now covered every supported page.
    pub complete: bool,
}

/// The two statements a demotion is: stop asserting, then queue the re-check.
///
/// One definition, two consumers — the trigger bodies below join them, and
/// [`MemoryDB::retract_support_for_rebound_source`] executes them with a bound
/// parameter. Writing them out twice is how a synchronous route path and a
/// trigger come to disagree about what a demotion means.
///
/// There is deliberately no third statement clearing `claim_judgment_attempts`.
/// Deleting them was unsound: a worker holding a live lease is neither revoked
/// nor fenced by a delete, so it repopulates the same key straight afterwards —
/// the delete removed the durable run-to-candidate record without removing the
/// stale conclusions it was aimed at. The run generation does that job properly.
/// The reset below returns the job to `pending`, the next lease stamps a fresh
/// generation, and every conclusion the superseded run reached goes inert
/// without a row being touched.
fn support_demotion_statements(affected_pages: &str, reason: &str) -> [String; 2] {
    [
        format!(
            "UPDATE page_truth_state
                    SET support_status = 'provisional',
                        provisional_reason = '{reason}',
                        evaluated_at = NULL,
                        updated_at = CAST(strftime('%s','now') AS INTEGER)
                  WHERE support_status = 'supported'
                    AND page_id IN ({affected_pages})"
        ),
        format!(
            "INSERT INTO claim_derivation_jobs
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
                  WHERE claim_derivation_jobs.status = 'done'"
        ),
    ]
}

fn support_demotion_body(affected_pages: &str) -> String {
    let [demote, requeue] = support_demotion_statements(affected_pages, REASON_EVIDENCE_WITHDRAWN);
    format!("\n{demote};\n{requeue};\n")
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
             BEGIN{page_move_body}END;

             -- Matrix row 8, the synchronous half. The enqueue trigger queues
             -- the re-derivation an edited page needs; it does not touch
             -- `page_truth_state`, so until a worker runs the page goes on
             -- exposing a verdict about text it no longer holds.
             --
             -- The WHEN mirrors `m5_page_update_enqueues_derivation` exactly, so
             -- the work and the demotion are triggered by the same event and
             -- cannot disagree about what counts as an edit. `NEW.content IS NOT
             -- OLD.content` is byte comparison rather than a digest comparison:
             -- SQLite cannot hash, and it does not need to -- comparing the
             -- bytes themselves is strictly stronger than comparing digests of
             -- them, and it catches the same-version content replacement that a
             -- version check alone cannot see.
             CREATE TRIGGER IF NOT EXISTS m5_demote_support_on_page_edit
             AFTER UPDATE ON pages
             WHEN NEW.status = 'active' AND NEW.kind <> 'entity'
                  AND (NEW.version IS NOT OLD.version
                       OR NEW.content IS NOT OLD.content
                       OR OLD.status <> 'active')
             BEGIN{page_edit_body}END;

             -- N1. A verdict cites a CHUNK, so losing that chunk is losing the
             -- evidence even when the document survives. The whole-source
             -- triggers above fire only when the last row for a source_id goes,
             -- which is right for what they guard and blind to an update or
             -- merge that deletes chunk 1 and keeps chunk 0.
             CREATE TRIGGER IF NOT EXISTS m5_demote_support_on_chunk_delete
             BEFORE DELETE ON memories
             WHEN EXISTS (
                      SELECT 1 FROM edges
                       WHERE edge_type = 'supports' AND src_kind = 'claim_revision'
                         AND lineage = 'evidence'
                         AND dst_id = OLD.source_id
                         AND json_extract(payload, '$.chunk_index') = OLD.chunk_index
                  )
             BEGIN{chunk_delete_body}END;

             -- And the same chunk surviving with different bytes in it. The
             -- span offsets a verdict recorded index into a string that is no
             -- longer there, so the citation is stale even though every row the
             -- edge names still exists.
             CREATE TRIGGER IF NOT EXISTS m5_demote_support_on_chunk_edit
             AFTER UPDATE OF content ON memories
             WHEN NEW.content IS NOT OLD.content
              AND EXISTS (
                      SELECT 1 FROM edges
                       WHERE edge_type = 'supports' AND src_kind = 'claim_revision'
                         AND lineage = 'evidence'
                         AND dst_id = NEW.source_id
                         AND json_extract(payload, '$.chunk_index') = NEW.chunk_index
                  )
             BEGIN{chunk_edit_body}END;",
            delete_body = support_demotion_body(&pages_supported_by_memory("OLD.source_id")),
            memory_move_body = support_demotion_body(&pages_supported_by_memory("NEW.source_id")),
            page_move_body = support_demotion_body("SELECT NEW.id"),
            page_edit_body = support_demotion_body("SELECT NEW.id"),
            chunk_delete_body = support_demotion_body(&pages_supported_by_memory_chunk(
                "OLD.source_id",
                "OLD.chunk_index"
            )),
            chunk_edit_body = support_demotion_body(&pages_supported_by_memory_chunk(
                "NEW.source_id",
                "NEW.chunk_index"
            )),
        ))
        .await
        .map_err(|error| {
            WenlanError::VectorDb(format!("m105 support-invalidation triggers: {error}"))
        })?;
        Ok(())
    }

    /// Advance bounded batches of stale extraction and drift work.
    ///
    /// Returns how many pages advanced into the durable queue/demotion batch. A
    /// zero from this function means there is no stale extraction or drift work
    /// left to advance — it is a real measurement, which is exactly what the
    /// empty queue before this worker was not.
    /// **The enqueue and the demotion are one transaction, not two statements
    /// that usually both run.** The order below is load-bearing — being
    /// `supported` is part of what makes a page drifted, so demoting first
    /// empties the drift query and the re-derivation is never queued — but an
    /// order argument is only a safety argument if both statements land. A
    /// failure after the enqueue and before the demotion leaves the page queued
    /// and still asserting `supported` on evidence that stopped qualifying,
    /// which is precisely the fail-open state row 15 exists to close, and the
    /// caller that sees the error has no way to tell that half of it committed.
    /// One transaction makes the pair atomic: either the page is demoted and
    /// queued, or nothing changed and the error is the whole truth.
    pub async fn enqueue_stale_derivation_jobs(&self, limit: i64) -> Result<usize, WenlanError> {
        if limit <= 0 {
            return Err(WenlanError::Validation(format!(
                "claim derivation sweep limit must be positive; got {limit}"
            )));
        }
        let now = chrono::Utc::now().timestamp();
        let conn = self.conn.lock().await;
        let tx = conn
            .transaction_with_behavior(libsql::TransactionBehavior::Immediate)
            .await
            .map_err(|error| WenlanError::VectorDb(format!("derivation sweep begin: {error}")))?;
        let swept = Self::sweep_stale_derivation_jobs(&tx, now, limit).await;
        match swept {
            Ok(total) => {
                tx.commit().await.map_err(|error| {
                    WenlanError::VectorDb(format!("derivation sweep commit: {error}"))
                })?;
                Ok(total)
            }
            Err(error) => {
                let _ = tx.rollback().await;
                Err(error)
            }
        }
    }

    /// [`Self::enqueue_stale_derivation_jobs`]'s body, over the transaction that
    /// makes its statement order mean something.
    async fn sweep_stale_derivation_jobs(
        conn: &libsql::Transaction,
        now: i64,
        limit: i64,
    ) -> Result<usize, WenlanError> {
        // Freeze ONE distinct union before any cleanup write. A `done` job is
        // what makes stale completion block a new pending row, but deleting
        // every stale `done` row before applying `limit` turns a nominal
        // 25-page turn into a corpus-sized write. Giving stale markers and
        // drift separate limited tables is not bounded either: disjoint sets
        // advance 2 * limit pages, while overlaps are counted twice. The one
        // connection-local batch is the overall budget and carries the bit
        // needed to demote only rows selected from the drift arm. It
        // participates in this transaction, so rollback loses the capture
        // together with every queue and truth-state change.
        conn.execute_batch(
            "CREATE TEMP TABLE IF NOT EXISTS claim_derivation_sweep_batch (
                 page_id TEXT NOT NULL,
                 page_version INTEGER NOT NULL,
                 needs_demotion INTEGER NOT NULL CHECK (needs_demotion IN (0, 1)),
                 PRIMARY KEY (page_id, page_version)
             ) WITHOUT ROWID;
             DELETE FROM claim_derivation_sweep_batch;",
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("derivation batch setup: {error}")))?;

        let captured = conn
            .execute(
                &format!(
                    "INSERT INTO claim_derivation_sweep_batch
                         (page_id, page_version, needs_demotion)
                     WITH stale_candidates(page_id, page_version) AS (
                         SELECT p.id, p.version
                           FROM pages p
                          WHERE p.status = 'active'
                            AND p.kind <> 'entity'
                            AND NOT EXISTS (
                                SELECT 1 FROM claim_derivation_markers m
                                 WHERE m.page_id = p.id
                                   AND m.page_version = p.version
                                   AND m.extractor_version = ?2
                            )
                            AND (
                                NOT EXISTS (
                                    SELECT 1 FROM claim_derivation_jobs j
                                     WHERE j.page_id = p.id
                                       AND j.page_version = p.version
                                )
                                OR EXISTS (
                                    SELECT 1 FROM claim_derivation_jobs j
                                     WHERE j.page_id = p.id
                                       AND j.page_version = p.version
                                       AND j.status = 'done'
                                )
                            )
                     ),
                     drift_candidates(page_id, page_version) AS (
                         SELECT d.drifted_page_id, d.drifted_page_version
                           FROM ({DRIFTED_SUPPORTED_PAGES}) d
                     ),
                     candidate_pages(page_id, page_version) AS (
                         SELECT page_id, page_version FROM stale_candidates
                         UNION
                         SELECT page_id, page_version FROM drift_candidates
                     )
                     SELECT c.page_id,
                            c.page_version,
                            EXISTS (
                                SELECT 1 FROM drift_candidates d
                                 WHERE d.page_id = c.page_id
                                   AND d.page_version = c.page_version
                            )
                       FROM candidate_pages c
                      ORDER BY c.page_id, c.page_version
                      LIMIT ?3"
                ),
                libsql::params![SUPPORT_THRESHOLD, EXTRACTOR_VERSION, limit],
            )
            .await
            .map_err(|error| WenlanError::VectorDb(format!("derivation batch capture: {error}")))?;

        // A selected `done` row is a stale claim of completion. Delete only
        // rows captured above; everything beyond the bound must remain byte-for-
        // byte discoverable for the next runtime turn.
        conn.execute(
            "DELETE FROM claim_derivation_jobs
              WHERE status = 'done'
                AND (page_id, page_version) IN (
                    SELECT page_id, page_version
                      FROM claim_derivation_sweep_batch
                )",
            (),
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("derivation backlog sweep: {error}")))?;

        conn.execute(
            "INSERT INTO claim_derivation_jobs
                 (job_id, page_id, page_version, status, attempts, created_at, updated_at)
             SELECT b.page_id || ':' || b.page_version,
                    b.page_id, b.page_version, 'pending', 0, ?1, ?1
               FROM claim_derivation_sweep_batch b
              WHERE NOT EXISTS (
                  SELECT 1 FROM claim_derivation_jobs j
                   WHERE j.page_id = b.page_id AND j.page_version = b.page_version
              )",
            libsql::params![now],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("derivation backlog enqueue: {error}")))?;

        // Demote NOW, not when a worker eventually reaches the job just queued.
        // Queueing answers "when will we know again"; it does not answer "what
        // do we say in the meantime", and until something answers the second the
        // page goes on asserting `supported` on evidence that stopped
        // qualifying. With no producer on this branch, "eventually" is never —
        // the same argument [`support_demotion_body`] makes for row 13, reached
        // here from the scan rather than from a trigger.
        //
        // Strictly after the enqueue above, and against the exact batch captured
        // before it. Being `supported` is part of what makes a page drifted:
        // demoting a fresh SELECT empties that predicate, while demoting more
        // rows than the bounded enqueue makes the tail disappear forever. The
        // temp batch binds both halves to one durable set.
        //
        // `evaluated_at = NULL` for the same reason it is NULL there: the page
        // lands on `Unevaluated`, so it keeps its projected file. We have stopped
        // asserting the evidence backs the prose without asserting that it does
        // not, which is the honest state when the bar moved under a stored
        // verdict.
        conn.execute(
            "UPDATE page_truth_state
                    SET support_status = 'provisional',
                        provisional_reason = 'supporting evidence no longer clears the \
                                              threshold; this page needs re-derivation',
                        evaluated_at = NULL,
                        updated_at = ?1
                  WHERE support_status = 'supported'
                    AND (page_id, page_version) IN (
                        SELECT page_id, page_version
                          FROM claim_derivation_sweep_batch
                         WHERE needs_demotion = 1
                    )",
            libsql::params![now],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("derivation drift demotion: {error}")))?;

        Ok(captured as usize)
    }

    /// Run [`Self::enqueue_stale_derivation_jobs`] until the backlog is drained,
    /// in batches of `batch`. Returns the total pages advanced.
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
    /// Each pass restarts the `pages` scan, so draining a vault of n
    /// pages costs O(n²/batch) index probes. Irrelevant at the thousands of
    /// pages a personal vault holds; if one ever outgrows that, the fix is a
    /// keyset cursor on `p.id`, not a bigger batch.
    pub async fn drain_stale_derivation_jobs(&self, batch: i64) -> Result<usize, WenlanError> {
        let mut total = 0usize;
        loop {
            let created = self.enqueue_stale_derivation_jobs(batch).await?;
            total += created;
            if created < batch as usize {
                return Ok(total);
            }
        }
    }

    /// Record, before judging, every candidate this run owes a conclusion on.
    ///
    /// The first thing a leased run does with its candidate set and the only
    /// thing that makes a partial run detectable. Attempt rows and support edges
    /// are both things a run PRODUCES, so the candidate a worker died before
    /// reaching leaves no trace in either, and evaluation reading only those two
    /// cannot tell a run that covered its inventory from one that covered half
    /// of it. See `claim_run_candidates`.
    ///
    /// Rows and seal go in under one Immediate transaction, because a partial
    /// inventory is the same defect one level up: an inventory that might be
    /// missing entries proves nothing about the entries it does have.
    ///
    /// Bound to the exact leased job and owner. Looking up the job's current
    /// generation here would let an expired worker wake after reclamation and
    /// stamp its old candidate set onto the new owner's run. A stale lease
    /// returns `false` and writes nothing, matching finalization's fence.
    ///
    /// There is no in-tree derivation worker yet; this is its required entry
    /// point when it lands. Same disclosed state as
    /// `claim_judgment_attempts`, whose rows only fixtures write today. The
    /// VERIFY half in [`Self::evaluate_support_on`] is live now and fails
    /// closed, so the ordering is safe: a real worker that never calls this
    /// publishes nothing at all.
    pub async fn record_run_candidate_inventory(
        &self,
        job: &DerivationJob,
        owner: &str,
        candidates: &[(String, CandidateLocator)],
    ) -> Result<bool, WenlanError> {
        let conn = self.conn.lock().await;
        let now = chrono::Utc::now().timestamp();
        let tx = conn
            .transaction_with_behavior(libsql::TransactionBehavior::Immediate)
            .await
            .map_err(|error| WenlanError::VectorDb(format!("inventory begin: {error}")))?;
        let written = async {
            let holds_lease = {
                let mut rows = tx
                    .query(
                        "SELECT 1 FROM claim_derivation_jobs
                          WHERE job_id = ?1 AND page_id = ?2 AND page_version = ?3
                            AND run_generation = ?4
                            AND status = 'leased' AND lease_owner = ?5",
                        libsql::params![
                            job.job_id.as_str(),
                            job.page_id.as_str(),
                            job.page_version,
                            job.run_generation,
                            owner,
                        ],
                    )
                    .await
                    .map_err(|error| WenlanError::VectorDb(format!("inventory lease: {error}")))?;
                rows.next()
                    .await
                    .map_err(|error| {
                        WenlanError::VectorDb(format!("inventory lease decode: {error}"))
                    })?
                    .is_some()
            };
            if !holds_lease {
                return Ok(false);
            }

            // A retry by the same lease replaces the authoritative set rather
            // than unioning it with an abandoned first enumeration. The whole
            // replacement and seal are one transaction, so no reader can see a
            // partial set.
            tx.execute(
                "DELETE FROM claim_run_candidates
                  WHERE page_id = ?1 AND page_version = ?2 AND run_generation = ?3",
                libsql::params![job.page_id.as_str(), job.page_version, job.run_generation,],
            )
            .await
            .map_err(|error| WenlanError::VectorDb(format!("inventory replace: {error}")))?;
            for (claim_revision_id, locator) in candidates {
                tx.execute(
                    "INSERT OR REPLACE INTO claim_run_candidates
                         (page_id, page_version, run_generation, claim_revision_id,
                          candidate_source_id, candidate_chunk_index,
                          candidate_span_start, candidate_span_end, candidate_digest)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                    libsql::params![
                        job.page_id.as_str(),
                        job.page_version,
                        job.run_generation,
                        claim_revision_id.as_str(),
                        locator.source_id.as_str(),
                        locator.chunk_index,
                        locator.span_start,
                        locator.span_end,
                        locator.span_digest.as_str(),
                    ],
                )
                .await
                .map_err(|error| WenlanError::VectorDb(format!("inventory row: {error}")))?;
            }
            tx.execute(
                "INSERT OR REPLACE INTO claim_run_inventories
                     (page_id, page_version, run_generation, sealed_at)
                 VALUES (?1, ?2, ?3, ?4)",
                libsql::params![
                    job.page_id.as_str(),
                    job.page_version,
                    job.run_generation,
                    now,
                ],
            )
            .await
            .map_err(|error| WenlanError::VectorDb(format!("inventory seal: {error}")))?;
            Ok::<_, WenlanError>(true)
        }
        .await;
        match written {
            Ok(recorded) => {
                tx.commit()
                    .await
                    .map_err(|error| WenlanError::VectorDb(format!("inventory commit: {error}")))?;
                Ok(recorded)
            }
            Err(error) => {
                let _ = tx.rollback().await;
                Err(error)
            }
        }
    }

    /// Re-prove every published `supported` verdict against the whole of §1, and
    /// demote the ones that no longer survive it. Returns how many were demoted.
    ///
    /// The startup reconciler, and the only pass that proves ALL FOUR
    /// conditions. Everything else that reconciles is a SQL scan, and SQL is
    /// where the gap was: [`DRIFTED_SUPPORTED_PAGES`] sees a score that fell
    /// below the live bar and an edge somebody retracted, and it is structurally
    /// blind to the rest of §1 —
    ///
    ///   * a marker that is **missing**, or written by an older
    ///     `EXTRACTOR_VERSION`, or describing text the page no longer holds. A
    ///     marker-only backlog scan re-queues such a page and stops there, so
    ///     the stored `supported` stayed readable until a worker arrived, which
    ///     on a branch with no producer is never.
    ///   * an edge whose citation no longer **revalidates** — the chunk deleted,
    ///     the span rewritten, the provenance root retired. Deciding that needs
    ///     to read a span out of live bytes and recompute an identity digest,
    ///     neither of which SQLite can do, so no scan can ever cover it.
    ///
    /// Rather than restate §1 in a second dialect, this runs the one evaluator:
    /// whatever [`Self::evaluate_page_support`] would say now is what the page
    /// is. Two implementations of a four-condition predicate is two chances to
    /// disagree about it, and the disagreement would surface as pages that are
    /// `supported` to the reconciler and `Refuted` to the worker.
    ///
    /// A page that fails lands on `Unevaluated`, never on `Unsupported`, even
    /// when the fresh verdict is `Refuted`. The reconciler is a CHECK, not a
    /// run: it establishes that the stored verdict is no longer earned, which is
    /// not the same as establishing the opposite. Stamping `evaluated_at` here
    /// would cost the page its projected file on the strength of a pass that
    /// spent no judge budget at all. The re-derivation it queues is what may
    /// legitimately reach `Refuted`.
    /// # Bounded, resumable, and fail-closed on the part it has not reached
    ///
    /// Each call examines at most `budget` pages and then returns, releasing the
    /// connection mutex. The pass itself is longer than one call: a durable
    /// keyset cursor in `app_metadata` names the last page checked, so the next
    /// call resumes rather than restarts, and a crash costs the current batch
    /// and nothing before it.
    ///
    /// The bound is not tidiness. This runs before the daemon serves, holding
    /// the single connection every foreground request needs, and it spends a
    /// multi-query evaluator per page, per claim, per candidate. Unbounded, a
    /// large supported vault occupies the port for as long as the scan takes —
    /// the exact failure [`Self::drain_stale_derivation_jobs`] bounds the
    /// migration's own scan to avoid, reintroduced by the pass that follows it.
    ///
    /// Bounding a pass that decides what may be served is only safe with the
    /// other half: until the cursor passes a page, that page's stored
    /// `supported` reads [`Support::Unevaluated`] rather than being asserted.
    /// Truncating the scan otherwise would serve precisely the stale
    /// `supported` the pass exists to retract, and quietly.
    ///
    /// [`Support::Unevaluated`]: crate::truth_contract::Support::Unevaluated
    pub async fn reconcile_supported_pages(
        &self,
        budget: usize,
    ) -> Result<SupportReconcilePass, WenlanError> {
        let budget = budget.max(1);
        let cursor = self
            .get_app_metadata(SUPPORT_RECONCILE_FRONTIER_KEY)
            .await?
            .unwrap_or_default();
        if cursor == SUPPORT_RECONCILE_COMPLETE {
            return Ok(SupportReconcilePass {
                demoted: 0,
                complete: true,
            });
        }

        let (failing, complete, reached) = {
            let conn = self.conn.lock().await;

            let supported: Vec<(String, i64)> = {
                let mut rows = conn
                    .query(
                        "SELECT t.page_id, t.page_version
                           FROM page_truth_state t
                           JOIN pages p ON p.id = t.page_id
                          WHERE t.support_status = 'supported'
                            AND p.status = 'active'
                            AND p.kind <> 'entity'
                            AND t.page_id > ?1
                          ORDER BY t.page_id
                          LIMIT ?2",
                        libsql::params![cursor.as_str(), budget as i64],
                    )
                    .await
                    .map_err(|error| {
                        WenlanError::VectorDb(format!("support reconcile scan: {error}"))
                    })?;
                let mut found = Vec::new();
                while let Some(row) = rows.next().await.map_err(|error| {
                    WenlanError::VectorDb(format!("support reconcile decode: {error}"))
                })? {
                    let decode =
                        |error| WenlanError::VectorDb(format!("support reconcile decode: {error}"));
                    found.push((
                        row.get::<String>(0).map_err(decode)?,
                        row.get::<i64>(1).map_err(decode)?,
                    ));
                }
                found
            };

            // A batch that comes back short has reached the end of the ordering,
            // the same data-bounded termination `drain_stale_derivation_jobs`
            // uses. A demoted page leaves the `supported` set, so it cannot
            // reappear behind the cursor either.
            let complete = supported.len() < budget;
            let reached = supported.last().map(|(id, _)| id.clone());

            let mut failing = Vec::new();
            for (page_id, page_version) in &supported {
                // `NoPublish` counts as a failure here, unlike at publication
                // time. A worker refusing to publish leaves a page alone because
                // some OTHER run owns it; the reconciler finding that the
                // current state cannot be re-derived is exactly the case where
                // continuing to assert `supported` is the unearned claim.
                if Self::evaluate_support_on(&conn, page_id, *page_version).await?
                    != SupportOutcome::Supported
                {
                    failing.push(page_id.clone());
                }
            }

            if !failing.is_empty() {
                // One transaction for the whole demotion, for the reason
                // [`Self::enqueue_stale_derivation_jobs`] gives: a page demoted
                // but not queued is recoverable, a page queued but not demoted
                // goes on serving a verdict we have just established it cannot
                // earn, and a partial failure must not be able to leave the
                // second.
                let tx = conn
                    .transaction_with_behavior(libsql::TransactionBehavior::Immediate)
                    .await
                    .map_err(|error| {
                        WenlanError::VectorDb(format!("support reconcile begin: {error}"))
                    })?;
                let demoted = async {
                    for page_id in &failing {
                        for statement in
                            support_demotion_statements("SELECT ?1", REASON_RECONCILE_UNPROVEN)
                        {
                            tx.execute(&statement, libsql::params![page_id.as_str()])
                                .await
                                .map_err(|error| {
                                    WenlanError::VectorDb(format!(
                                        "support reconcile demote: {error}"
                                    ))
                                })?;
                        }
                    }
                    Ok::<(), WenlanError>(())
                }
                .await;
                match demoted {
                    Ok(()) => tx.commit().await.map_err(|error| {
                        WenlanError::VectorDb(format!("support reconcile commit: {error}"))
                    })?,
                    Err(error) => {
                        let _ = tx.rollback().await;
                        return Err(error);
                    }
                }
            }
            (failing.len(), complete, reached)
        };

        // Cursor last, and only after the demotion has committed. The cursor is
        // a promise that everything at or below it has been re-proved, so
        // advancing it over a demotion that failed to land would mark a page
        // checked while it goes on asserting the verdict this pass refused it.
        // Advancing late can only re-check a page, which costs time and nothing
        // else.
        let advanced = if complete {
            SUPPORT_RECONCILE_COMPLETE.to_string()
        } else {
            reached.unwrap_or(cursor)
        };
        self.set_app_metadata(SUPPORT_RECONCILE_FRONTIER_KEY, &advanced)
            .await?;

        Ok(SupportReconcilePass {
            demoted: failing,
            complete,
        })
    }

    /// Start a reconciliation pass, or resume the durable frontier an
    /// interrupted process left under the same ruleset.
    ///
    /// Resetting an incomplete pass on every boot can starve a large vault's
    /// tail forever: every short-lived process proves the same prefix and never
    /// reaches the next page. The ruleset binds the cursor to the evaluator that
    /// earned it. A changed extractor, threshold, or reconciliation algorithm
    /// starts over; the same evaluator resumes.
    ///
    /// A completed pass starts over on the next boot. Until that pass completes,
    /// every stored `supported` above the cursor reads `Unevaluated`, so no old
    /// verdict is asserted before the current process re-earns it.
    pub async fn begin_support_reconcile_pass(&self) -> Result<(), WenlanError> {
        let ruleset = support_reconcile_ruleset();
        let conn = self.conn.lock().await;
        let tx = conn
            .transaction_with_behavior(libsql::TransactionBehavior::Immediate)
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!("support reconcile pass begin: {error}"))
            })?;
        let reset = async {
            let frontier = {
                let mut rows = tx
                    .query(
                        "SELECT value FROM app_metadata WHERE key = ?1",
                        libsql::params![SUPPORT_RECONCILE_FRONTIER_KEY],
                    )
                    .await
                    .map_err(|error| {
                        WenlanError::VectorDb(format!("support reconcile frontier read: {error}"))
                    })?;
                rows.next()
                    .await
                    .map_err(|error| {
                        WenlanError::VectorDb(format!("support reconcile frontier decode: {error}"))
                    })?
                    .map(|row| row.get::<String>(0))
                    .transpose()
                    .map_err(|error| {
                        WenlanError::VectorDb(format!("support reconcile frontier decode: {error}"))
                    })?
            };
            let recorded_ruleset = {
                let mut rows = tx
                    .query(
                        "SELECT value FROM app_metadata WHERE key = ?1",
                        libsql::params![SUPPORT_RECONCILE_RULESET_KEY],
                    )
                    .await
                    .map_err(|error| {
                        WenlanError::VectorDb(format!("support reconcile ruleset read: {error}"))
                    })?;
                rows.next()
                    .await
                    .map_err(|error| {
                        WenlanError::VectorDb(format!("support reconcile ruleset decode: {error}"))
                    })?
                    .map(|row| row.get::<String>(0))
                    .transpose()
                    .map_err(|error| {
                        WenlanError::VectorDb(format!("support reconcile ruleset decode: {error}"))
                    })?
            };
            if frontier
                .as_deref()
                .is_some_and(|value| value != SUPPORT_RECONCILE_COMPLETE)
                && recorded_ruleset.as_deref() == Some(ruleset.as_str())
            {
                return Ok::<_, WenlanError>(());
            }

            for (key, value) in [
                (SUPPORT_RECONCILE_RULESET_KEY, ruleset.as_str()),
                (SUPPORT_RECONCILE_FRONTIER_KEY, ""),
            ] {
                tx.execute(
                    "INSERT INTO app_metadata (key, value) VALUES (?1, ?2)
                     ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    libsql::params![key, value],
                )
                .await
                .map_err(|error| {
                    WenlanError::VectorDb(format!("support reconcile pass reset: {error}"))
                })?;
            }
            Ok(())
        }
        .await;
        match reset {
            Ok(()) => tx.commit().await.map_err(|error| {
                WenlanError::VectorDb(format!("support reconcile pass commit: {error}"))
            }),
            Err(error) => {
                let _ = tx.rollback().await;
                Err(error)
            }
        }
    }

    /// How far the current reconciliation pass has got, for the read gate.
    ///
    /// `None` means nothing is withheld — either the pass finished, or no pass
    /// has ever begun on this database. The second case is deliberately
    /// permissive: the gate exists to cover the window between a pass starting
    /// and finishing, and a database on which reconciliation has never run is
    /// not in that window. Startup opens it explicitly with
    /// [`Self::begin_support_reconcile_pass`].
    ///
    /// `Some(cursor)` means pages ordering after `cursor` have not been
    /// re-proved yet. An empty string is that value at the start of a pass, and
    /// no page ID is empty, so it withholds everything.
    pub async fn support_reconcile_frontier(&self) -> Result<Option<String>, WenlanError> {
        Ok(
            match self
                .get_app_metadata(SUPPORT_RECONCILE_FRONTIER_KEY)
                .await?
            {
                None => None,
                Some(value) if value == SUPPORT_RECONCILE_COMPLETE => None,
                Some(cursor) => Some(cursor),
            },
        )
    }

    /// Retract the support edges that cite one document, and demote the pages
    /// resting on them — inside the caller's transaction.
    ///
    /// The rename path is the one production mutation that moves evidence
    /// without touching a single row a trigger watches. `rebind_source_id`
    /// changes `memories.source_id`; it does not delete a memory, does not
    /// change a memory's content, and does not move a space, so
    /// `m5_demote_support_on_memory_delete`, `..._chunk_delete`, `..._chunk_edit`
    /// and the two space-move triggers all stay silent. The support edge keeps
    /// pointing at a `dst_id` that names nothing, `valid_until` is still NULL
    /// because nobody retracted it, and the page goes on exposing `supported`
    /// indefinitely — nothing later re-checks it, because no production caller
    /// re-evaluates a page whose text did not change.
    ///
    /// A catch-all trigger on `memories.source_id` is the wrong answer: it is
    /// the hottest table in the database and the rename is a known, bounded,
    /// already-transactional caller. So the caller does it explicitly, in the
    /// transaction that performs the rename, and the page cannot be observed
    /// renamed-but-still-supported at any point.
    ///
    /// **Retract rather than re-point.** The rename optimization fires on equal
    /// content hashes, so re-pointing each edge at the new `source_id` would
    /// usually be true — but `edge_id` is content-addressed over the endpoint,
    /// so every edge would have to be re-minted, and `rebind_source_id` is a
    /// general primitive whose other callers promise nothing about content.
    /// Retracting costs one re-derivation per affected page and lands them on
    /// `Unevaluated`, which keeps every file. Being wrong in that direction
    /// costs judge budget; being wrong in the other direction is a citation
    /// pointing at bytes nobody checked.
    pub(super) async fn retract_support_for_rebound_source(
        conn: &libsql::Connection,
        old_source_id: &str,
    ) -> Result<(), WenlanError> {
        // Demote and re-queue BEFORE retracting, because the edges are what
        // identify the affected pages. `pages_supported_by_memory` deliberately
        // ignores `valid_until` — see its doc comment — so this order is belt
        // and braces rather than the only thing holding it up.
        for statement in
            support_demotion_statements(&pages_supported_by_memory("?1"), REASON_EVIDENCE_REBOUND)
        {
            conn.execute(&statement, libsql::params![old_source_id])
                .await
                .map_err(|error| {
                    WenlanError::VectorDb(format!("rebind support demotion: {error}"))
                })?;
        }
        conn.execute(
            "UPDATE edges
                SET valid_until = CAST(strftime('%s','now') AS INTEGER),
                    superseded_by = NULL
              WHERE edge_type = 'supports'
                AND src_kind = 'claim_revision'
                AND dst_kind = 'memory'
                AND dst_id = ?1
                AND valid_until IS NULL",
            libsql::params![old_source_id],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("rebind support retraction: {error}")))?;
        Ok(())
    }

    /// Allocate the next run generation. Monotone, never 0.
    ///
    /// A generation is burned whether or not the lease that asked for it finds
    /// work. Gaps are free — the only property anything depends on is that a
    /// value is never reissued, so two runs can never be mistaken for one.
    pub(super) async fn allocate_run_generation(
        conn: &libsql::Connection,
    ) -> Result<i64, WenlanError> {
        let mut rows = conn
            .query(
                "UPDATE claim_run_generations
                    SET last_generation = last_generation + 1
                  WHERE id = 1
                  RETURNING last_generation",
                (),
            )
            .await
            .map_err(|error| WenlanError::VectorDb(format!("run generation allocate: {error}")))?;
        let Some(row) = rows.next().await.map_err(|error| {
            WenlanError::VectorDb(format!("run generation allocate row: {error}"))
        })?
        else {
            // The allocator row is seeded by the same statement batch that
            // creates the table, so its absence means the substrate is not
            // there. Refusing beats inventing a generation nothing else agrees
            // with: a wrong generation makes another run's rows look like this
            // one's, which is the exact confusion it exists to prevent.
            return Err(WenlanError::VectorDb(
                "run generation allocator row is missing; refusing to start a derivation run \
                 without a run identity"
                    .to_string(),
            ));
        };
        row.get::<i64>(0)
            .map_err(|error| WenlanError::VectorDb(format!("run generation decode: {error}")))
    }

    /// Take the oldest claimable job, or `None` when the queue is drained.
    ///
    /// Claimable means pending, or leased with an expired lease — the second
    /// half is the reclaim path the lease columns exist for. The select and the
    /// update are one statement so two workers racing cannot both win: SQLite
    /// serializes the write, and the loser's subquery re-evaluates against the
    /// already-leased row.
    ///
    /// **Every lease is a new run.** `(page_id, page_version)` names a job that
    /// outlives its attempts: a released or reclaimed job keeps the same row
    /// while `attempts` climbs, so judgment rows from an attempt that died
    /// halfway would otherwise sit beside rows from the attempt that replaced
    /// it and read as one complete run. A claim that reads as fully judged with
    /// no qualifying edge is `Refuted`, which costs the page its file — so the
    /// mix-up fails in the most expensive direction available. Stamping a fresh
    /// generation here makes the previous attempt's rows inert without deleting
    /// them, which is also the lease invalidation a reclaim needs: the stalled
    /// worker's rows land under a generation nobody reads, and its finalization
    /// is refused by the lease guard.
    pub async fn lease_next_derivation_job(
        &self,
        owner: &str,
    ) -> Result<Option<DerivationJob>, WenlanError> {
        let now = chrono::Utc::now().timestamp();
        let expires = now + LEASE_SECS;
        let conn = self.conn.lock().await;
        let generation = Self::allocate_run_generation(&conn).await?;

        let mut rows = conn
            .query(
                "UPDATE claim_derivation_jobs
                    SET status = 'leased',
                        lease_owner = ?1,
                        lease_expires_at = ?2,
                        attempts = attempts + 1,
                        run_generation = ?5,
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
                libsql::params![owner, expires, now, MAX_ATTEMPTS, generation],
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

        // Superseded runs' rows are left in place, deliberately. They are the
        // only durable record of what each attempt weighed and how far it got,
        // the generation filter already makes them unreadable as this run's
        // work, and deleting them is what the previous design got wrong: a
        // stalled worker can still be writing, so a delete removes live state
        // and cannot remove the state it was aiming at.
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
            run_generation: generation,
        }))
    }

    /// Mark one exact run done. Returns false when the caller no longer holds
    /// that run's lease.
    ///
    /// Owner alone is not an identity: a restarted worker commonly reuses its
    /// configured owner string. The whole [`DerivationJob`] tuple, including
    /// `run_generation`, is therefore the fence between a stale attempt and the
    /// successor that reclaimed the same durable job.
    pub async fn finish_derivation_job(
        &self,
        job: &DerivationJob,
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
                  WHERE job_id = ?2
                    AND page_id = ?3
                    AND page_version = ?4
                    AND run_generation = ?5
                    AND status = 'leased'
                    AND lease_owner = ?6",
                libsql::params![
                    now,
                    job.job_id.as_str(),
                    job.page_id.as_str(),
                    job.page_version,
                    job.run_generation,
                    owner
                ],
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
        job: &DerivationJob,
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
                  WHERE job_id = ?3
                    AND page_id = ?4
                    AND page_version = ?5
                    AND run_generation = ?6
                    AND status = 'leased'
                    AND lease_owner = ?7",
                libsql::params![
                    error_text,
                    now,
                    job.job_id.as_str(),
                    job.page_id.as_str(),
                    job.page_version,
                    job.run_generation,
                    owner
                ],
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
    ///    `supports` edge that is active, above threshold, and still
    ///    revalidates against the evidence it names — see
    ///    [`Self::revalidate_support_edge`]. Failure → `Refuted`. This is the
    ///    one condition whose failure is a verdict, because reaching it means
    ///    the derivation ran, the claims are real, and the evidence was looked
    ///    for and was not there.
    /// 4. No revision is in a deferred, timed-out, malformed, or internally
    ///    inconsistent support state. Detection covers every place a run can
    ///    stop or disagree with its declaration: claim membership must match
    ///    the marker count; the per-run candidate set must be sealed; sealed
    ///    candidates and attempt locators must match in both directions; every
    ///    concluded locator must still resolve to the live bytes and digest it
    ///    names; every attempt must have concluded; and every live candidate
    ///    edge must have been weighed by the run. Failure → `NoPublish` (row 6).
    ///
    ///    **Condition 4 is checked before condition 3, per claim, and a valid
    ///    support edge does not excuse it.** Reading condition 3 first and
    ///    short-circuiting on the first edge that revalidates publishes
    ///    `Supported` for a claim with one good edge and one candidate the run
    ///    timed out on — a run that did not finish, reported as one that
    ///    finished and succeeded. §1 condition 4 is a property of the RUN, not
    ///    of whether some candidate happened to work out.
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

    /// Re-run §4a's evidence checks against the bytes that are there NOW.
    ///
    /// [`Self::write_support_edge`] runs exactly these checks before it will
    /// create an edge, which establishes that the citation was true at write
    /// time and nothing beyond that. An edge is a durable record and evidence
    /// is mutable, so the interval between those two moments is where a
    /// support edge goes wrong: the chunk it names can be deleted while its
    /// document survives, be rewritten in place, or be renumbered, and none of
    /// that touches the edge's own `valid_until`. `valid_until IS NULL` means
    /// nobody retracted this edge — never that what it cites is still there.
    ///
    /// Publication reads a support edge as truth, so publication re-earns it.
    /// Everything unverifiable is a refusal: a payload missing the fields that
    /// say WHICH bytes were judged cannot be checked against those bytes, and
    /// unknown is not true.
    ///
    /// Returns the reason the edge no longer holds, or `None` when it does.
    async fn revalidate_support_edge(
        conn: &libsql::Connection,
        memory_source_id: &str,
        root_id: Option<&str>,
        payload: &str,
    ) -> Result<Option<String>, WenlanError> {
        let Some(root_id) = root_id else {
            return Ok(Some(format!(
                "support edge on {memory_source_id} names no provenance root"
            )));
        };
        let payload: serde_json::Value = match serde_json::from_str(payload) {
            Ok(value) => value,
            Err(error) => {
                return Ok(Some(format!(
                    "support edge on {memory_source_id} has an unreadable payload: {error}"
                )));
            }
        };
        let (Some(chunk_index), Some(source_version), Some(span_start), Some(span_end)) = (
            payload["chunk_index"].as_i64(),
            payload["source_version"].as_i64(),
            payload["span_start"].as_i64(),
            payload["span_end"].as_i64(),
        ) else {
            return Ok(Some(format!(
                "support edge on {memory_source_id} does not record which bytes it judged"
            )));
        };
        let Some(span_digest) = payload["span_digest"].as_str() else {
            return Ok(Some(format!(
                "support edge on {memory_source_id} records no span digest"
            )));
        };

        // The evidence row, addressed by CHUNK. Two rows is a refusal for the
        // reason `write_support_edge` gives at length: nothing makes
        // `(source_id, chunk_index)` unique, and choosing one of two candidates
        // would be the silent arbitrary pick the chunk index exists to remove.
        let (content, live_version) = {
            let mut rows = conn
                .query(
                    "SELECT content, COALESCE(version, 1) FROM memories
                      WHERE source_id = ?1 AND chunk_index = ?2 LIMIT 2",
                    libsql::params![memory_source_id, chunk_index],
                )
                .await
                .map_err(|error| {
                    WenlanError::VectorDb(format!("support revalidation memory: {error}"))
                })?;
            let Some(row) = rows.next().await.map_err(|error| {
                WenlanError::VectorDb(format!("support revalidation memory decode: {error}"))
            })?
            else {
                return Ok(Some(format!(
                    "the evidence is gone: {memory_source_id} has no chunk {chunk_index}"
                )));
            };
            let content: String = row.get(0).map_err(|error| {
                WenlanError::VectorDb(format!("support revalidation memory decode: {error}"))
            })?;
            let live_version: i64 = row.get(1).map_err(|error| {
                WenlanError::VectorDb(format!("support revalidation memory decode: {error}"))
            })?;
            if rows
                .next()
                .await
                .map_err(|error| {
                    WenlanError::VectorDb(format!("support revalidation memory decode: {error}"))
                })?
                .is_some()
            {
                return Ok(Some(format!(
                    "the evidence is ambiguous: {memory_source_id} now has more than one chunk \
                     {chunk_index}"
                )));
            }
            (content, live_version)
        };

        if live_version != source_version {
            return Ok(Some(format!(
                "the evidence moved on: the verdict read version {source_version} of \
                 {memory_source_id}, which is at version {live_version}"
            )));
        }

        // `get` rather than `[..]`: these are byte offsets into text that may
        // have changed under us, so a boundary that no longer lands on a char
        // is a refusal, never a panic.
        let start = usize::try_from(span_start).unwrap_or(usize::MAX);
        let end = usize::try_from(span_end).unwrap_or(usize::MAX);
        let Some(span) = content.get(start..end) else {
            return Ok(Some(format!(
                "the cited span [{start}, {end}) is no longer a valid span of {memory_source_id}"
            )));
        };
        if crate::provenance::revision_content_digest(span) != span_digest {
            return Ok(Some(format!(
                "the text changed: the bytes at [{start}, {end}) in {memory_source_id} are not \
                 the ones this verdict judged"
            )));
        }

        // §5. Roots are content-addressed, so this binding is recomputable from
        // the evidence's own bytes — which is what makes it a check here rather
        // than a fact taken on trust from write time.
        let mut rows = conn
            .query(
                "SELECT identity_version, identity_digest, root_kind, status
                   FROM provenance_roots WHERE root_id = ?1",
                libsql::params![root_id],
            )
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!("support revalidation root: {error}"))
            })?;
        let Some(row) = rows.next().await.map_err(|error| {
            WenlanError::VectorDb(format!("support revalidation root decode: {error}"))
        })?
        else {
            return Ok(Some(format!("provenance root {root_id} is gone")));
        };
        let identity_version: i64 = row.get(0).map_err(|error| {
            WenlanError::VectorDb(format!("support revalidation root decode: {error}"))
        })?;
        let identity_digest: String = row.get(1).map_err(|error| {
            WenlanError::VectorDb(format!("support revalidation root decode: {error}"))
        })?;
        let root_kind: String = row.get(2).map_err(|error| {
            WenlanError::VectorDb(format!("support revalidation root decode: {error}"))
        })?;
        let status: String = row.get(3).map_err(|error| {
            WenlanError::VectorDb(format!("support revalidation root decode: {error}"))
        })?;
        if status != "active" {
            return Ok(Some(format!("provenance root {root_id} is now '{status}'")));
        }
        if identity_version != crate::provenance::IDENTITY_VERSION {
            return Ok(Some(format!(
                "provenance root {root_id} was recorded under identity version \
                 {identity_version}, which this code cannot recompute"
            )));
        }
        if crate::provenance::identity_digest(&root_kind, &content) != identity_digest {
            return Ok(Some(format!(
                "provenance root {root_id} no longer identifies the evidence at {memory_source_id}"
            )));
        }

        Ok(None)
    }

    /// Resolve one run-inventory locator against the evidence bytes that are
    /// live now. A seal plus a matching attempt proves only that two tables
    /// agree with each other; it does not prove that either names a real
    /// candidate. Missing, ambiguous, out-of-range, and digest-mismatched
    /// locators are malformed run state, never a short judgement.
    async fn revalidate_candidate_locator(
        conn: &libsql::Connection,
        locator: &CandidateLocator,
    ) -> Result<Option<String>, WenlanError> {
        let content = {
            let mut rows = conn
                .query(
                    "SELECT content FROM memories
                      WHERE source_id = ?1 AND chunk_index = ?2 LIMIT 2",
                    libsql::params![locator.source_id.as_str(), locator.chunk_index],
                )
                .await
                .map_err(|error| {
                    WenlanError::VectorDb(format!("candidate locator memory: {error}"))
                })?;
            let Some(row) = rows.next().await.map_err(|error| {
                WenlanError::VectorDb(format!("candidate locator memory decode: {error}"))
            })?
            else {
                return Ok(Some(format!(
                    "candidate locator names no live chunk {} of {}",
                    locator.chunk_index, locator.source_id
                )));
            };
            let content: String = row.get(0).map_err(|error| {
                WenlanError::VectorDb(format!("candidate locator memory decode: {error}"))
            })?;
            if rows
                .next()
                .await
                .map_err(|error| {
                    WenlanError::VectorDb(format!("candidate locator memory decode: {error}"))
                })?
                .is_some()
            {
                return Ok(Some(format!(
                    "candidate locator is ambiguous: {} has more than one chunk {}",
                    locator.source_id, locator.chunk_index
                )));
            }
            content
        };

        let start = usize::try_from(locator.span_start).unwrap_or(usize::MAX);
        let end = usize::try_from(locator.span_end).unwrap_or(usize::MAX);
        let Some(span) = content.get(start..end) else {
            return Ok(Some(format!(
                "candidate locator span [{start}, {end}) is invalid for {} chunk {}",
                locator.source_id, locator.chunk_index
            )));
        };
        if crate::provenance::revision_content_digest(span) != locator.span_digest {
            return Ok(Some(format!(
                "candidate locator digest no longer matches {} chunk {} span [{start}, {end})",
                locator.source_id, locator.chunk_index
            )));
        }
        Ok(None)
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

        // Which RUN's conclusions count. `(page_id, page_version)` names a job
        // that outlives its attempts, so the generation stamped on the job row
        // is what says which rows belong to the attempt currently in force. A
        // page with no job row has no run at all, and 0 — the never-leased
        // value, never allocated to a real run — is the honest reading.
        let run_generation: i64 =
            {
                let mut rows = conn
                    .query(
                        "SELECT run_generation FROM claim_derivation_jobs
                      WHERE page_id = ?1 AND page_version = ?2",
                        libsql::params![page_id, page_version],
                    )
                    .await
                    .map_err(|error| WenlanError::VectorDb(format!("support run read: {error}")))?;
                match rows.next().await.map_err(|error| {
                    WenlanError::VectorDb(format!("support run decode: {error}"))
                })? {
                    Some(row) => row.get::<i64>(0).map_err(|error| {
                        WenlanError::VectorDb(format!("support run decode: {error}"))
                    })?,
                    None => 0,
                }
            };

        // Row 6's third face: a run that never wrote down what it owed.
        //
        // The two checks further down both read a run's OUTPUTS — the attempt
        // rows it left and the support edges it produced — and neither can see
        // a candidate that produced neither. A worker that concludes on A and
        // dies before touching B leaves exactly that state when B scored no
        // edge, and the claim then reads as fully judged on A alone.
        //
        // The seal is what makes the obligation legible. It is written with the
        // candidate rows under one transaction, so its presence means the run
        // recorded its whole inventory, and its absence means the run got no
        // further than leasing.
        //
        // Generation 0 is exempt because it is not a run. It is the never-leased
        // value, so there is no inventory that was owed and not recorded — the
        // honest reading is "nobody has looked", which the never-judged path
        // below already reaches as `Unevaluated`. Folding it in here would turn
        // every underived page into `NoPublish` and lose that distinction.
        if run_generation > 0 {
            let sealed: i64 = {
                let mut rows = conn
                    .query(
                        "SELECT COUNT(*) FROM claim_run_inventories
                          WHERE page_id = ?1 AND page_version = ?2 AND run_generation = ?3",
                        libsql::params![page_id, page_version, run_generation],
                    )
                    .await
                    .map_err(|error| {
                        WenlanError::VectorDb(format!("support inventory seal: {error}"))
                    })?;
                rows.next()
                    .await
                    .map_err(|error| {
                        WenlanError::VectorDb(format!("support inventory seal decode: {error}"))
                    })?
                    .map(|row| row.get::<i64>(0))
                    .transpose()
                    .map_err(|error| {
                        WenlanError::VectorDb(format!("support inventory seal decode: {error}"))
                    })?
                    .unwrap_or(0)
            };
            if sealed == 0 {
                return Ok(SupportOutcome::NoPublish {
                    reason: format!(
                        "unrecorded candidate inventory: run {run_generation} never sealed the \
                         candidate set it owed"
                    ),
                });
            }
        }

        // Conditions 3 and 4, one claim at a time. Deciding them needs two
        // things SQL cannot do — reading a span back out of live bytes, and
        // recomputing a provenance identity digest — so the scan gathers
        // candidates and the judgement happens here.
        //
        // Per claim, in the order the conditions are ranked:
        //
        //   outside set  -- this run concluded on a candidate outside the
        //                   inventory it declared complete.
        //   incomplete   -- this run left some candidate without a conclusion.
        //                   The run has not finished, whatever else is true.
        //   unregistered -- a live candidate edge this run never weighed at
        //                   all. The same failure from the other side: the run
        //                   did not cover the inventory it faced.
        //   supported    -- at least one active, above-threshold support edge
        //                   that this run concluded on and that STILL
        //                   revalidates against current evidence.
        //   never judged -- no such edge, and no candidate was ever weighed. We
        //                   have not looked at this claim at all.
        //   judged short -- no such edge, and every candidate concluded. That
        //                   IS a verdict about the evidence.
        //
        // The bottom three are what `provisional` alone conflates, and the
        // split has to reach the STORED reason or the distinction dies at the
        // point a human would use it.
        let claim_revision_ids: Vec<String> =
            {
                let mut rows = conn
                    .query(
                        "SELECT claim_revision_id FROM page_version_claims
                      WHERE page_id = ?1 AND page_version = ?2",
                        libsql::params![page_id, page_version],
                    )
                    .await
                    .map_err(|error| WenlanError::VectorDb(format!("support scan: {error}")))?;
                let mut ids = Vec::new();
                while let Some(row) = rows.next().await.map_err(|error| {
                    WenlanError::VectorDb(format!("support scan decode: {error}"))
                })? {
                    ids.push(row.get::<String>(0).map_err(|error| {
                        WenlanError::VectorDb(format!("support scan decode: {error}"))
                    })?);
                }
                ids
            };

        let (
            mut never_judged,
            mut judged_short,
            mut incomplete,
            mut outside_inventory,
            mut unregistered,
            mut invalid_locator,
        ) = (0i64, 0i64, 0i64, 0i64, 0i64, 0i64);
        let mut first_invalid_locator_reason = None;
        for claim_revision_id in &claim_revision_ids {
            // The candidate's LOCATION comes out of the payload alongside the
            // edge, because location is what identifies a candidate: the same
            // sentence can appear twice in one document, and two occurrences
            // share a span digest. A payload missing any part of the locator
            // matches no attempt row, which reads as unregistered — the same
            // refusal `revalidate_support_edge` reaches for the same reason.
            let candidates: Vec<SupportCandidate> = {
                let mut rows = conn
                    .query(
                        "SELECT e.dst_id, e.root_id, e.payload,
                                json_extract(e.payload, '$.chunk_index'),
                                json_extract(e.payload, '$.span_start'),
                                json_extract(e.payload, '$.span_end'),
                                json_extract(e.payload, '$.span_digest')
                           FROM edges e
                          WHERE e.edge_type = 'supports'
                            AND e.src_kind = 'claim_revision'
                            AND e.dst_kind = 'memory'
                            AND e.src_id = ?1
                            AND e.valid_until IS NULL
                            AND e.superseded_by IS NULL
                            AND json_extract(e.payload, '$.score') >= ?2",
                        libsql::params![claim_revision_id.clone(), SUPPORT_THRESHOLD],
                    )
                    .await
                    .map_err(|error| {
                        WenlanError::VectorDb(format!("support candidate scan: {error}"))
                    })?;
                let mut found = Vec::new();
                while let Some(row) = rows.next().await.map_err(|error| {
                    WenlanError::VectorDb(format!("support candidate decode: {error}"))
                })? {
                    let decode =
                        |error| WenlanError::VectorDb(format!("support candidate decode: {error}"));
                    let source_id = row.get::<String>(0).map_err(decode)?;
                    let locator = match (
                        row.get::<Option<i64>>(3).map_err(decode)?,
                        row.get::<Option<i64>>(4).map_err(decode)?,
                        row.get::<Option<i64>>(5).map_err(decode)?,
                        row.get::<Option<String>>(6).map_err(decode)?,
                    ) {
                        (Some(chunk), Some(start), Some(end), Some(digest)) => {
                            Some(CandidateLocator {
                                source_id: source_id.clone(),
                                chunk_index: chunk,
                                span_start: start,
                                span_end: end,
                                span_digest: digest,
                            })
                        }
                        _ => None,
                    };
                    found.push(SupportCandidate {
                        source_id,
                        root_id: row.get::<Option<String>>(1).map_err(decode)?,
                        payload: row.get::<String>(2).map_err(decode)?,
                        locator,
                    });
                }
                found
            };

            // What THIS run managed to do about this claim — which
            // `entailment_cache` cannot answer, being global and timeless, and
            // which rows from a superseded run must not answer either. See
            // `claim_judgment_attempts`.
            let attempts: Vec<(CandidateLocator, String)> = {
                let mut rows = conn
                    .query(
                        "SELECT candidate_source_id, candidate_chunk_index,
                                candidate_span_start, candidate_span_end,
                                candidate_digest, outcome
                           FROM claim_judgment_attempts
                          WHERE page_id = ?1 AND page_version = ?2
                            AND run_generation = ?3
                            AND claim_revision_id = ?4",
                        libsql::params![
                            page_id,
                            page_version,
                            run_generation,
                            claim_revision_id.clone()
                        ],
                    )
                    .await
                    .map_err(|error| WenlanError::VectorDb(format!("support attempts: {error}")))?;
                let mut found = Vec::new();
                while let Some(row) = rows.next().await.map_err(|error| {
                    WenlanError::VectorDb(format!("support attempts decode: {error}"))
                })? {
                    let decode =
                        |error| WenlanError::VectorDb(format!("support attempts decode: {error}"));
                    found.push((
                        CandidateLocator {
                            source_id: row.get::<String>(0).map_err(decode)?,
                            chunk_index: row.get::<i64>(1).map_err(decode)?,
                            span_start: row.get::<i64>(2).map_err(decode)?,
                            span_end: row.get::<i64>(3).map_err(decode)?,
                            span_digest: row.get::<String>(4).map_err(decode)?,
                        },
                        row.get::<String>(5).map_err(decode)?,
                    ));
                }
                found
            };

            // What this run RECORDED owing on this claim, before it judged
            // anything. Empty at generation 0, and empty-with-a-seal for a
            // claim whose candidate set really was empty; the seal check above
            // is what separates those from an inventory that was never written.
            let expected: Vec<CandidateLocator> = {
                let mut rows = conn
                    .query(
                        "SELECT candidate_source_id, candidate_chunk_index,
                                candidate_span_start, candidate_span_end,
                                candidate_digest
                           FROM claim_run_candidates
                          WHERE page_id = ?1 AND page_version = ?2
                            AND run_generation = ?3
                            AND claim_revision_id = ?4",
                        libsql::params![
                            page_id,
                            page_version,
                            run_generation,
                            claim_revision_id.clone()
                        ],
                    )
                    .await
                    .map_err(|error| WenlanError::VectorDb(format!("support expected: {error}")))?;
                let mut found = Vec::new();
                while let Some(row) = rows.next().await.map_err(|error| {
                    WenlanError::VectorDb(format!("support expected decode: {error}"))
                })? {
                    let decode =
                        |error| WenlanError::VectorDb(format!("support expected decode: {error}"));
                    found.push(CandidateLocator {
                        source_id: row.get::<String>(0).map_err(decode)?,
                        chunk_index: row.get::<i64>(1).map_err(decode)?,
                        span_start: row.get::<i64>(2).map_err(decode)?,
                        span_end: row.get::<i64>(3).map_err(decode)?,
                        span_digest: row.get::<String>(4).map_err(decode)?,
                    });
                }
                found
            };

            // The sealed inventory is authoritative in both directions. It is
            // not enough for every expected candidate to have a conclusion:
            // an attempt outside the sealed set means the worker judged a
            // different inventory from the one it declared complete. Without
            // this reverse check, an empty sealed set plus one extra attempt
            // vacuously passes `expected ⊆ attempts` and can publish either a
            // refutation or support.
            if attempts
                .iter()
                .any(|(observed, _)| !expected.iter().any(|declared| declared == observed))
            {
                outside_inventory += 1;
                continue;
            }

            // Condition 4 first, and ahead of condition 3 rather than after it.
            // A run that timed out on one candidate did not finish, and one
            // other candidate working out does not make it finished — reading
            // condition 3 first and stopping at the first edge that revalidates
            // publishes `Supported` for exactly that page.
            if attempts.iter().any(|(_, outcome)| outcome != "concluded") {
                incomplete += 1;
                continue;
            }
            // The same refusal, for the candidate that left no row at all. A
            // deferred attempt is a run saying "I could not finish this one";
            // a recorded candidate with NO attempt row is a run that stopped
            // before it could say anything, and the two mean the same thing
            // about whether the run finished. Compared against the recorded
            // inventory rather than against the run's own outputs, because
            // comparing outputs to themselves is what let this case through.
            if expected.iter().any(|owed| {
                !attempts
                    .iter()
                    .any(|(concluded, outcome)| concluded == owed && outcome == "concluded")
            }) {
                incomplete += 1;
                continue;
            }
            let mut locator_invalid = false;
            for locator in &expected {
                if let Some(reason) = Self::revalidate_candidate_locator(conn, locator).await? {
                    invalid_locator += 1;
                    locator_invalid = true;
                    first_invalid_locator_reason.get_or_insert(reason);
                    break;
                }
            }
            if locator_invalid {
                continue;
            }
            // The inventory half of condition 4. Every live candidate has to be
            // one this run actually weighed; an edge with no attempt row behind
            // it is evidence the run never accounted for, and publishing on it
            // is publishing on work nobody did.
            if candidates.iter().any(|candidate| {
                candidate.locator.as_ref().is_none_or(|locator| {
                    !attempts.iter().any(|(concluded, _)| concluded == locator)
                })
            }) {
                unregistered += 1;
                continue;
            }

            let mut supported = false;
            for candidate in &candidates {
                if Self::revalidate_support_edge(
                    conn,
                    &candidate.source_id,
                    candidate.root_id.as_deref(),
                    &candidate.payload,
                )
                .await?
                .is_none()
                {
                    supported = true;
                    break;
                }
            }
            if supported {
                continue;
            }

            if attempts.is_empty() {
                never_judged += 1;
            } else {
                judged_short += 1;
            }
        }

        // A sealed inventory and the run's outputs must name the same candidate
        // set. An output outside it is not extra evidence; it means the run's
        // completeness assertion and the work it performed disagree.
        if outside_inventory > 0 {
            return Ok(SupportOutcome::NoPublish {
                reason: format!(
                    "candidate inventory mismatch: {outside_inventory} of {inventory_count} \
                     claim(s) have a conclusion outside the sealed inventory"
                ),
            });
        }

        // Row 6, the no-publication-on-incomplete-derivation invariant. A
        // candidate with no conclusion means the run is still in flight, so
        // nothing about it may be published — not even `Unevaluated`, which is
        // itself a statement that we looked and came up empty. Ranked above
        // never-judged because a run that started and stalled is a stronger
        // reason to write nothing than one that never began.
        //
        // This is the case a cache-row discriminator gets wrong and gets wrong
        // in the worst direction: one candidate concludes short, another times
        // out, the cache has a row either way, and the claim reads as fully
        // judged with no support — which is `Refuted`, which costs the page its
        // file for evaluation that never actually finished.
        if incomplete > 0 {
            return Ok(SupportOutcome::NoPublish {
                reason: format!(
                    "incomplete derivation: {incomplete} of {inventory_count} claim(s) have a \
                     candidate this run never concluded on"
                ),
            });
        }

        // A concluded locator that cannot be resolved is malformed pipeline
        // state, not evidence that was judged and fell short. Set equality
        // between the seal and attempts cannot manufacture the missing bytes.
        if invalid_locator > 0 {
            return Ok(SupportOutcome::NoPublish {
                reason: format!(
                    "unresolvable candidate locator: {invalid_locator} of {inventory_count} \
                     claim(s) name evidence that cannot be verified{}",
                    first_invalid_locator_reason
                        .as_deref()
                        .map(|reason| format!(": {reason}"))
                        .unwrap_or_default()
                ),
            });
        }

        // The other half of row 6. A live candidate with no attempt row behind
        // it is not a claim we judged and lost — it is evidence this run never
        // opened, and a verdict that rests on it rests on nothing. Ranked with
        // `incomplete` rather than below it because they are the same failure:
        // the run did not cover its inventory.
        if unregistered > 0 {
            return Ok(SupportOutcome::NoPublish {
                reason: format!(
                    "unaccounted evidence: {unregistered} of {inventory_count} claim(s) have a \
                     live support candidate this run never weighed"
                ),
            });
        }

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
    /// 1. The exact job run must still be `leased` by `owner`. The
    ///    owner guard on [`Self::finish_derivation_job`] protected only
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
        job: &DerivationJob,
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
            // Guard 1: the exact lease. Every field in `DerivationJob` is
            // matched, especially the run generation: owner strings can be
            // reused by a restarted worker after reclaim.
            let holds_lease = {
                let mut rows = tx
                    .query(
                        "SELECT 1 FROM claim_derivation_jobs
                          WHERE job_id = ?1
                            AND page_id = ?2
                            AND page_version = ?3
                            AND run_generation = ?4
                            AND status = 'leased'
                            AND lease_owner = ?5",
                        libsql::params![
                            job.job_id.as_str(),
                            job.page_id.as_str(),
                            job.page_version,
                            job.run_generation,
                            owner
                        ],
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
            if job.page_id != page_id || job.page_version != page_version {
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
