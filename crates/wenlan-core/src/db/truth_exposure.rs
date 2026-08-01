// SPDX-License-Identifier: Apache-2.0
//! The durable half of the M5 exposure contract: the marker audit log, the
//! cutover generation that decides whether any of it is live yet, and the fence
//! that keeps page writers from crossing the ceremony.
//!
//! Three small things, all here because they are properties of the database
//! rather than of a request.
//!
//! **The audit log** is the *only* compensating control for the one attack D3
//! concedes. `Collection` and `NamedPage` compose: a caller willing to forge the
//! intent marker can list unsupported page IDs and then fetch them one at a
//! time. Nothing at cooperative tier prevents that -- the daemon is loopback and
//! unauthenticated, and cannot tell a forged gesture from a real one. What it
//! can do is leave a trail, so page-at-a-time extraction is a visible pattern
//! instead of an invisible one. An untested audit row is a sentence, not a
//! control, which is why `truth_guard` has a test that goes RED when the write
//! is removed.
//!
//! **The cutover generation** is why PR-B changes no behavior. Every adapter is
//! installed and mutation-tested against a generation the tests advance
//! themselves; in production it stays 0 and every adapter is pass-through. PR-C
//! advances it once, through the fenced ceremony -- the same shape as the M4
//! reader cutover, deliberately.
//!
//! **The fence** is what makes "advances it once" true rather than aspirational.
//! Pages are projected at runtime, so a page written while the ceremony is
//! mid-flight would slip past the pass that just ran and land in the vault
//! unexamined. [`MemoryDB::begin_cutover`] closes the window, and
//! [`crate::truth_adapter::page_write_permit`] is where every page writer feels
//! it.

use super::MemoryDB;
use crate::truth_contract::{visible_at, MarkerOutcome, TruthGrant, Visibility};
use crate::WenlanError;
use std::collections::HashMap;

/// `app_metadata` key holding the durable cutover generation. Absent or `0`
/// means every truth adapter is inert.
pub const TRUTH_CUTOVER_GENERATION_KEY: &str = "truth_cutover_generation";

/// `app_metadata` key holding the durable cutover fence, as `"<epoch>:<phase>"`.
/// Absent is the initial fence: epoch 0, phase off.
pub const TRUTH_CUTOVER_FENCE_KEY: &str = "truth_cutover_fence";

/// Where the cutover ceremony currently stands.
///
/// `Committed` never returns to `Off`. Rebuilding the full legacy directory is
/// exactly the flip the rollback contract forbids -- every provisional page's
/// prose would reappear at a path `wenlan pages` reads directly, with nothing
/// able to stop it. After commit the only moves are forward.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CutoverPhase {
    Off,
    Preparing,
    Committed,
}

impl CutoverPhase {
    fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Preparing => "preparing",
            Self::Committed => "committed",
        }
    }

    fn parse(raw: &str) -> Option<Self> {
        match raw {
            "off" => Some(Self::Off),
            "preparing" => Some(Self::Preparing),
            "committed" => Some(Self::Committed),
            _ => None,
        }
    }
}

/// The durable writer fence: an epoch paired with a phase.
///
/// # Why the phase alone is not enough
///
/// `off -> preparing -> off` on an abort would let a writer that captured `off`
/// before the ceremony compare-and-swap successfully afterwards -- a textbook
/// ABA. The phase enum cannot distinguish "the `off` I read" from "a later
/// `off`", and the writer that wins that CAS is exactly the one whose page the
/// ceremony never examined.
///
/// So every transition bumps the epoch, monotonically, and a writer swaps the
/// *pair*. A stale `off` capture then fails against a higher epoch even when the
/// phase matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CutoverFence {
    pub epoch: i64,
    pub phase: CutoverPhase,
}

impl CutoverFence {
    /// What a database with no fence row means: nothing has ever run.
    pub const INITIAL: Self = Self {
        epoch: 0,
        phase: CutoverPhase::Off,
    };

    fn encode(&self) -> String {
        format!("{}:{}", self.epoch, self.phase.as_str())
    }

    fn parse(raw: &str) -> Option<Self> {
        let (epoch, phase) = raw.split_once(':')?;
        Some(Self {
            epoch: epoch.trim().parse().ok()?,
            phase: CutoverPhase::parse(phase.trim())?,
        })
    }

    fn next(&self, phase: CutoverPhase) -> Self {
        Self {
            epoch: self.epoch + 1,
            phase,
        }
    }
}

/// Proof that this process holds the cutover lease.
///
/// The field is private, so a caller cannot fabricate one: the only way to hold
/// a lease is to have won [`MemoryDB::begin_cutover`], and the only way to spend
/// it is to hand it back to commit or abort. Same discipline as
/// [`crate::truth_adapter::PagePermit`], for the same reason -- the ceremony
/// runs outside the request path, where there is no guard to forget to call.
///
/// Deliberately **not** `Clone`, and commit/abort take it **by value**, so it is
/// linear: usable exactly once, checked by the compiler. Borrowing it instead
/// left a real hole, because the lease check and the generation write are
/// separate statements under a mutex that serializes statements, not protocols.
/// Two aliases could both pass the check, then one could write its generation
/// after the other had already committed and returned success -- leaving the
/// database at a generation nobody was told about. There is no way to express
/// that now: the second call does not compile.
#[derive(Debug, PartialEq, Eq)]
pub struct CutoverLease {
    fence: CutoverFence,
}

impl CutoverLease {
    /// The fence this lease was minted at. A commit that finds anything else in
    /// the database has lost the lease.
    pub fn fence(&self) -> CutoverFence {
        self.fence
    }
}

/// The two truth axes for one page.
///
/// Independent by construction: neither is inferred from the other. A page can
/// be machine-supported and unreviewed, or human-reviewed and unsupported, and
/// both facts have to reach the reader for a `collection` entry to be legal.
///
/// The machine axis is a three-state [`Support`] rather than a bool, so an
/// unjudged page cannot be mistaken for a judged-and-failed one. It was a bool
/// until the ceremony review: `support_status` has only ever held `supported`
/// or `provisional`, and reading `provisional` as "the evidence does not back
/// this" made every page that predates claim derivation an eviction target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageTruth {
    pub support: crate::truth_contract::Support,
    pub human_reviewed: bool,
}

/// A page nothing knows anything about: unjudged, unreviewed. The right answer
/// for a missing row, because absence of a judgement is not a judgement.
impl Default for PageTruth {
    fn default() -> Self {
        Self {
            support: crate::truth_contract::Support::Unevaluated,
            human_reviewed: false,
        }
    }
}

impl PageTruth {
    /// Whether the evidence has been checked and found to back the prose.
    ///
    /// Kept as a method rather than a field so `unevaluated` can never silently
    /// read as `supported` at a call site that only wanted the happy case.
    pub fn supported(&self) -> bool {
        self.support == crate::truth_contract::Support::Supported
    }
}

/// One recorded marked call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TruthMarkerAudit {
    pub marked_at: i64,
    /// Who called, from `x-agent-name`. `unknown` when the caller did not say --
    /// honest-unknown beats guessing, same as the rest of agent attribution.
    pub caller: String,
    pub method: String,
    /// The route *template*, not the concrete URI: `/api/pages/{id}`. The IDs
    /// live in `page_ids`, where they can be counted.
    pub path: String,
    pub outcome: String,
    /// The page IDs this call named, in path order.
    pub page_ids: Vec<String>,
}

impl MemoryDB {
    /// Migration 101 DDL. Additive and idempotent.
    pub(super) async fn ensure_truth_exposure_tables(
        tx: &libsql::Transaction,
    ) -> Result<(), WenlanError> {
        tx.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS truth_marker_audit (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                marked_at  INTEGER NOT NULL,
                caller     TEXT    NOT NULL,
                method     TEXT    NOT NULL,
                path       TEXT    NOT NULL,
                outcome    TEXT    NOT NULL CHECK (outcome IN (
                               'refused', 'automatic',
                               'granted_collection', 'granted_named_page')),
                -- JSON array, empty for a collection call. Stored as text
                -- because the interesting query is 'which IDs did this caller
                -- name over the last hour', which reads the whole row anyway.
                page_ids   TEXT    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_truth_marker_audit_recent
                ON truth_marker_audit(marked_at DESC);
            CREATE INDEX IF NOT EXISTS idx_truth_marker_audit_caller
                ON truth_marker_audit(caller, marked_at DESC);
            ",
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m101 truth exposure tables: {error}")))?;
        Ok(())
    }

    /// Record one marked call.
    ///
    /// Called for every request carrying the intent marker, granted or refused.
    /// A refusal is the signature of a misconfigured integration and costs one
    /// row to keep.
    pub async fn record_truth_marker(
        &self,
        caller: &str,
        method: &str,
        path: &str,
        outcome: MarkerOutcome,
        page_ids: &[String],
    ) -> Result<(), WenlanError> {
        let ids = serde_json::to_string(page_ids)
            .map_err(|e| WenlanError::VectorDb(format!("record_truth_marker page_ids: {e}")))?;
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO truth_marker_audit (marked_at, caller, method, path, outcome, page_ids)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            libsql::params![
                chrono::Utc::now().timestamp(),
                caller,
                method,
                path,
                outcome.as_str(),
                ids
            ],
        )
        .await
        .map_err(|e| WenlanError::VectorDb(format!("record_truth_marker: {e}")))?;
        Ok(())
    }

    /// The most recent marked calls, newest first.
    pub async fn recent_truth_markers(
        &self,
        limit: usize,
    ) -> Result<Vec<TruthMarkerAudit>, WenlanError> {
        let conn = self.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT marked_at, caller, method, path, outcome, page_ids
                   FROM truth_marker_audit
                  ORDER BY marked_at DESC, id DESC
                  LIMIT ?1",
                libsql::params![limit as i64],
            )
            .await
            .map_err(|e| WenlanError::VectorDb(format!("recent_truth_markers: {e}")))?;

        let mut out = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| WenlanError::VectorDb(format!("recent_truth_markers next: {e}")))?
        {
            let ids: String = row.get(5).unwrap_or_default();
            out.push(TruthMarkerAudit {
                marked_at: row.get(0).unwrap_or_default(),
                caller: row.get(1).unwrap_or_default(),
                method: row.get(2).unwrap_or_default(),
                path: row.get(3).unwrap_or_default(),
                outcome: row.get(4).unwrap_or_default(),
                page_ids: serde_json::from_str(&ids).unwrap_or_default(),
            });
        }
        Ok(out)
    }

    /// The durable cutover generation. `0` -- the production value throughout
    /// PR-B -- means every truth adapter is pass-through.
    ///
    /// ponytail: one indexed `app_metadata` read per request. Cache it on
    /// `MemoryDB` behind an atomic if it ever shows up in a profile; a
    /// single-user loopback daemon is nowhere near needing that today.
    pub async fn truth_cutover_generation(&self) -> Result<i64, WenlanError> {
        Ok(self
            .get_app_metadata(TRUTH_CUTOVER_GENERATION_KEY)
            .await?
            .and_then(|raw| raw.trim().parse::<i64>().ok())
            .unwrap_or(0))
    }

    /// Both truth axes for a batch of pages, in one query.
    ///
    /// A page with no `page_truth_state` row reads as [`Support::Unevaluated`],
    /// NOT as unsupported. Absence of a record is absence of a judgement, and a
    /// judgement that never ran is not a judgement that failed -- post-migration
    /// that absence is the normal case, so reading it as "unsupported" would
    /// condemn every page. `evaluated_at` is what separates the two, and it is
    /// NULL for every row migration 99's backfill wrote.
    ///
    /// `human_reviewed` is **computed, not read**. The stored bit is a
    /// historical receipt — "somebody approved this page, at this version, with
    /// this content digest" — and nothing clears it when the page is edited: the
    /// update bumps `version` and rewrites `content`, and the truth row is not in
    /// its reach. Left as-is, a page could be reviewed once, rewritten into
    /// something else entirely, and stay both "reviewed" and (post-cutover)
    /// fully visible on the strength of an approval of text nobody can see any
    /// more.
    ///
    /// So the review is in force only while `reviewed_page_digest` still matches
    /// the page's current content. An edited page falls back to unreviewed by
    /// itself, and back to whatever its machine verdict earns it — fail-closed,
    /// with no new write path and nothing to migrate. Restoring the exact text
    /// restores the review, which is right: the receipt is about content, not
    /// about a version counter that only ever climbs.
    ///
    /// Content is pulled only for rows that claim a review, so the ordinary page
    /// carries no extra cost.
    ///
    /// [`Support::Unevaluated`]: crate::truth_contract::Support::Unevaluated
    pub async fn page_truth_states(
        &self,
        page_ids: &[String],
    ) -> Result<HashMap<String, PageTruth>, WenlanError> {
        let mut out = HashMap::new();
        if page_ids.is_empty() {
            return Ok(out);
        }
        let conn = self.conn.lock().await;
        // Chunked because SQLite caps bound parameters, and a page list comes
        // from a response payload whose length nobody here controls.
        for chunk in page_ids.chunks(400) {
            let placeholders = (1..=chunk.len())
                .map(|i| format!("?{i}"))
                .collect::<Vec<_>>()
                .join(", ");
            let sql = format!(
                "SELECT t.page_id, t.support_status, t.human_reviewed, t.evaluated_at,
                        t.reviewed_page_digest,
                        CASE WHEN t.human_reviewed = 1 THEN p.content END
                   FROM page_truth_state t
                   LEFT JOIN pages p ON p.id = t.page_id
                  WHERE t.page_id IN ({placeholders})"
            );
            let params = chunk
                .iter()
                .map(|id| libsql::Value::from(id.as_str()))
                .collect::<Vec<_>>();
            let mut rows = conn
                .query(&sql, params)
                .await
                .map_err(|e| WenlanError::VectorDb(format!("page_truth_states: {e}")))?;
            while let Some(row) = rows
                .next()
                .await
                .map_err(|e| WenlanError::VectorDb(format!("page_truth_states next: {e}")))?
            {
                let page_id: String = row.get(0).unwrap_or_default();
                let status: String = row.get(1).unwrap_or_default();
                let reviewed: i64 = row.get(2).unwrap_or(0);
                // `support_status` cannot answer this alone. Its only
                // non-supported value is `provisional`, which migration 99
                // stamps on every pre-existing page with the reason "never
                // evaluated: predates claim derivation" -- so `provisional`
                // covers both "nobody looked" and "looked, and the evidence
                // fell short". `evaluated_at` is what separates them, and it is
                // a nullable column rather than a third `support_status` value
                // because SQLite cannot widen a CHECK without rebuilding the
                // table.
                //
                // `unwrap_or(None)` puts a malformed or absent column on the
                // unjudged side -- the side that keeps a page's file.
                let evaluated_at: Option<i64> = row.get(3).unwrap_or(None);
                let support = match (status.as_str(), evaluated_at) {
                    ("supported", _) => crate::truth_contract::Support::Supported,
                    (_, Some(_)) => crate::truth_contract::Support::Unsupported,
                    (_, None) => crate::truth_contract::Support::Unevaluated,
                };
                // Fail closed on both halves: a NULL digest and a page that is
                // no longer there both read as unreviewed. The schema's CHECK
                // already forbids a `human_reviewed = 1` row without a digest
                // (`claim_identity.rs`), so the NULL arm is unreachable through
                // any current write path -- it is here so that stays true even
                // if some future one forgets.
                let reviewed_digest: Option<String> = row.get(4).unwrap_or(None);
                let live_content: Option<String> = row.get(5).unwrap_or(None);
                let human_reviewed = reviewed == 1
                    && match (reviewed_digest, live_content) {
                        (Some(recorded), Some(content)) => {
                            recorded == crate::provenance::revision_content_digest(&content)
                        }
                        _ => false,
                    };
                out.insert(
                    page_id,
                    PageTruth {
                        support,
                        human_reviewed,
                    },
                );
            }
        }
        Ok(out)
    }

    /// One call's verdict for a batch of pages -- **the** entry point every
    /// adapter uses.
    ///
    /// At generation 0 this is a single `app_metadata` read and every page comes
    /// back `Full`, so an inert adapter costs one cheap lookup and never touches
    /// `page_truth_state` at all. That is what makes installing 79 of them
    /// before the cutover affordable.
    ///
    /// A page ID with no entry in the result was not asked about; adapters
    /// should treat a missing key the same as `Hidden` rather than as
    /// permission, but the map is total over `page_ids` by construction.
    pub async fn page_visibility(
        &self,
        grant: &TruthGrant,
        page_ids: &[String],
    ) -> Result<HashMap<String, Visibility>, WenlanError> {
        let generation = self.truth_cutover_generation().await?;
        if generation == 0 {
            return Ok(page_ids
                .iter()
                .map(|id| (id.clone(), Visibility::Full))
                .collect());
        }
        let states = self.page_truth_states(page_ids).await?;
        Ok(page_ids
            .iter()
            .map(|id| {
                // A page with no row at all has never been judged either, so it
                // gets the unjudged verdict rather than the failed one. Same
                // reasoning as the NULL `evaluated_at` above: absence of a
                // judgement is not a judgement.
                let truth = states.get(id).copied().unwrap_or(PageTruth {
                    support: crate::truth_contract::Support::Unevaluated,
                    human_reviewed: false,
                });
                (
                    id.clone(),
                    visible_at(generation, grant, id, truth.support, truth.human_reviewed),
                )
            })
            .collect())
    }

    /// Advance (or roll back) the cutover generation.
    ///
    /// PR-B calls this only from tests. PR-C owns the production advance, which
    /// is a two-phase fenced ceremony this setter is merely the last step of --
    /// so nothing here should be read as permission to flip it.
    ///
    /// # Advancing this alone is destructive, not protective
    ///
    /// As of PR-B the ONLY production consumer of [`Self::page_visibility`] is
    /// the projection-directory invariant in `export::knowledge`, which DELETES
    /// the `.md` file of every page the verdict hides. No HTTP adapter reads the
    /// grant the guard resolves -- `select_visible_pages` filters on scope,
    /// trust tier and `kind`, and never consults truth state at all.
    ///
    /// So the destructive half of this contract is wired and the protective half
    /// is not. Advancing the generation today would evict pages from the user's
    /// vault while every page route kept serving them. PR-C must land the
    /// adapters BEFORE the ceremony, not alongside it.
    pub async fn set_truth_cutover_generation(&self, generation: i64) -> Result<(), WenlanError> {
        self.set_app_metadata(TRUTH_CUTOVER_GENERATION_KEY, &generation.to_string())
            .await
    }

    // ==================== the writer fence ====================

    /// The durable cutover fence. Absent means [`CutoverFence::INITIAL`].
    ///
    /// A value that does not parse is an **error**, not a default. Every other
    /// gate in this module fails toward inert, but "inert" for a fence means
    /// letting writers through, and an indeterminate fence is precisely the
    /// state where that is unsafe -- recovery consults durable state, never a
    /// guess. The error propagates into
    /// [`crate::truth_adapter::page_write_permit`], which then refuses the write.
    pub async fn cutover_fence(&self) -> Result<CutoverFence, WenlanError> {
        let Some(raw) = self.get_app_metadata(TRUTH_CUTOVER_FENCE_KEY).await? else {
            return Ok(CutoverFence::INITIAL);
        };
        CutoverFence::parse(raw.trim()).ok_or_else(|| {
            WenlanError::VectorDb(format!(
                "cutover fence is unreadable ({raw:?}); refusing to guess whether a \
                 ceremony is in flight"
            ))
        })
    }

    /// Swap the fence from `observed` to `next`, atomically. `false` means
    /// somebody else moved it first.
    ///
    /// `set_app_metadata` cannot express this: it is an unconditional upsert,
    /// with nowhere to put the `WHERE value = ?3` predicate that makes the swap
    /// a compare-and-set rather than a clobber. The second statement exists only
    /// for the first ceremony a database ever runs, where the row is absent
    /// rather than holding the initial value; both run under the same connection
    /// guard, and `app_metadata.key` is a primary key, so exactly one of two
    /// racing callers can see `changed == 1`.
    async fn swap_cutover_fence(
        &self,
        observed: CutoverFence,
        next: CutoverFence,
    ) -> Result<bool, WenlanError> {
        let conn = self.conn.lock().await;
        let mut changed = conn
            .execute(
                "UPDATE app_metadata SET value = ?2 WHERE key = ?1 AND value = ?3",
                libsql::params![TRUTH_CUTOVER_FENCE_KEY, next.encode(), observed.encode()],
            )
            .await
            .map_err(|e| WenlanError::VectorDb(format!("swap_cutover_fence update: {e}")))?;
        if changed == 0 && observed == CutoverFence::INITIAL {
            changed = conn
                .execute(
                    "INSERT INTO app_metadata (key, value)
                     SELECT ?1, ?2
                      WHERE NOT EXISTS (SELECT 1 FROM app_metadata WHERE key = ?1)",
                    libsql::params![TRUTH_CUTOVER_FENCE_KEY, next.encode()],
                )
                .await
                .map_err(|e| WenlanError::VectorDb(format!("swap_cutover_fence insert: {e}")))?;
        }
        Ok(changed == 1)
    }

    /// Take the cutover lease: `off` -> `preparing`, at a new epoch.
    ///
    /// From here until commit or abort, [`crate::truth_adapter::page_write_permit`]
    /// refuses every page write, so nothing lands in the vault behind the pass
    /// the ceremony is about to run.
    pub async fn begin_cutover(&self) -> Result<CutoverLease, WenlanError> {
        let observed = self.cutover_fence().await?;
        if observed.phase != CutoverPhase::Off {
            return Err(WenlanError::VectorDb(format!(
                "cannot begin a cutover: the fence is at epoch {} phase {}",
                observed.epoch,
                observed.phase.as_str()
            )));
        }
        let next = observed.next(CutoverPhase::Preparing);
        if !self.swap_cutover_fence(observed, next).await? {
            return Err(WenlanError::VectorDb(
                "cannot begin a cutover: the fence moved underneath us".to_string(),
            ));
        }
        Ok(CutoverLease { fence: next })
    }

    /// Commit the cutover: write the generation, *then* release the fence.
    ///
    /// That order is the §7.5 ordering invariant. Releasing first would reopen
    /// page writes against a generation that has not been committed yet, which
    /// is the window the whole fence exists to close.
    ///
    /// Refuses unless the fence still reads exactly what the lease was minted
    /// at -- an epoch bump by anyone else means this lease is stale.
    pub async fn commit_cutover(
        &self,
        lease: CutoverLease,
        generation: i64,
    ) -> Result<(), WenlanError> {
        let observed = self.cutover_fence().await?;
        if observed != lease.fence {
            return Err(WenlanError::VectorDb(format!(
                "cannot commit the cutover: lease is epoch {} phase {}, fence is epoch {} phase {}",
                lease.fence.epoch,
                lease.fence.phase.as_str(),
                observed.epoch,
                observed.phase.as_str()
            )));
        }
        self.set_truth_cutover_generation(generation).await?;
        if !self
            .swap_cutover_fence(observed, observed.next(CutoverPhase::Committed))
            .await?
        {
            return Err(WenlanError::VectorDb(
                "the cutover generation is committed but the fence could not be released; \
                 the fence moved underneath us"
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// Give the lease back without committing: `preparing` -> `off`, at a new
    /// epoch.
    ///
    /// The new epoch is the point. Returning to the *same* `off` would let a
    /// writer that captured the pre-ceremony fence swap successfully, and that
    /// writer's page is one the aborted ceremony may already have examined.
    pub async fn abort_cutover(&self, lease: CutoverLease) -> Result<(), WenlanError> {
        let observed = self.cutover_fence().await?;
        if observed != lease.fence || observed.phase != CutoverPhase::Preparing {
            return Err(WenlanError::VectorDb(format!(
                "cannot abort the cutover: lease is epoch {} phase {}, fence is epoch {} phase {}",
                lease.fence.epoch,
                lease.fence.phase.as_str(),
                observed.epoch,
                observed.phase.as_str()
            )));
        }
        if !self
            .swap_cutover_fence(observed, observed.next(CutoverPhase::Off))
            .await?
        {
            return Err(WenlanError::VectorDb(
                "cannot abort the cutover: the fence moved underneath us".to_string(),
            ));
        }
        Ok(())
    }

    /// Release a fence stranded at `preparing`, at a new epoch. `true` means one
    /// was found and released.
    ///
    /// [`Self::abort_cutover`] cannot do this: it needs a [`CutoverLease`], and a
    /// lease dies with the process that minted it. A ceremony killed between
    /// `begin_cutover` and `commit_cutover` -- SIGINT during the eviction loop,
    /// a panic, a power cut -- therefore leaves `preparing` on disk with nothing
    /// alive that can take it back, and `preparing` refuses **every** page write.
    /// Fail-closed, but permanently, and the only remaining move would be editing
    /// `app_metadata` by hand against a live WAL.
    ///
    /// Only the daemon's startup calls this, and only because a ceremony and a
    /// running daemon are mutually exclusive by construction: `truth-cutover`
    /// takes the data-root lock, refuses while the port answers, and refuses
    /// while the service unit exists. So a fence still reading `preparing` when
    /// the daemon boots belongs to a ceremony that died -- there is no live lease
    /// to invalidate.
    ///
    /// A lease that somehow survived cannot commit onto the released fence: the
    /// phase alone already differs, so the tuple compare refuses. The epoch still
    /// bumps, to keep it monotone the way every other transition does -- not
    /// because the bump is what closes that case. And `committed` is untouched:
    /// only `preparing` is releasable, so the forward-only rule still holds.
    pub async fn release_stranded_cutover_fence(&self) -> Result<bool, WenlanError> {
        let observed = self.cutover_fence().await?;
        if observed.phase != CutoverPhase::Preparing {
            return Ok(false);
        }
        self.swap_cutover_fence(observed, observed.next(CutoverPhase::Off))
            .await
    }
}

#[cfg(test)]
#[path = "truth_exposure_test.rs"]
mod tests;
