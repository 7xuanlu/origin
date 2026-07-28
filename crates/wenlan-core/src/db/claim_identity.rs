// SPDX-License-Identifier: Apache-2.0
//! M5 claim-identity substrate (schema 97, additive half).
//!
//! Creates the tables PR-A adds: logical claims, immutable content-addressed
//! revisions, per-revision anchors, page-version membership, derivation
//! markers, the five-part entailment cache, page truth state, derivation
//! jobs/leases, and the one-shot presence nonce + receipt tables.
//!
//! The table creation is purely additive, so it is a plain `CREATE TABLE IF
//! NOT EXISTS` batch. The `edges` widening is NOT here: SQLite cannot alter a
//! `CHECK` constraint, so adding the `claim_revision`/`root` endpoint kinds and
//! the `attests` edge type needs the guarded row-for-row table rebuild of
//! `docs/plans/2026-07-27-m5-edge-rebuild-matrix.md` §7. Both run inside the
//! one migration 97 transaction, tables first, because the rebuilt space fence
//! resolves a `claim_revision` endpoint through `claim_revisions` — a
//! half-stamped 97 is exactly the state the ordering invariant
//! (`docs/plans/2026-07-27-m5-migration-state-machine.md` §7.1) forbids.
//!
//! This module also owns the **human-root minter**, the one path by which a
//! `human_edit_delta` provenance root and its span-addressable memory come into
//! existence.

use super::MemoryDB;
use crate::WenlanError;

/// Every human-authored root shares ONE independence group.
///
/// Independence groups exist to count *independent* corroboration (M6's support
/// floor). One person writing the same thing on two pages, or in two sessions,
/// is one source and not two — so collapsing all human authorship into a single
/// group is the honest reading, and it is also the conservative one: this can
/// only ever under-count independence, never inflate it, which is the failure
/// Q6 B.4 refuses to risk.
///
/// It is what artifact 6 §2a requires in the concrete: "because a human-authored
/// claim rests on exactly one independence group, it can never accumulate
/// independent corroboration from its own page ... must not be worked around by
/// counting the same delta twice."
///
/// The near-dup LSH overlay inside `acquire_provenance_root` still outranks this
/// key, and that is correct rather than a leak: a delta a human copied out of a
/// document adopts that document's group, so the copy does not become a second
/// independent voice for what the document already said.
const HUMAN_SOURCE_IDENTITY: &str = "human:local";

/// A minted human edit delta: the provenance root, and the memory that makes its
/// text span-addressable.
#[derive(Debug, Clone)]
pub struct HumanEditDelta {
    /// `provenance_roots.root_id`, kind `human_edit_delta`.
    pub root_id: String,
    /// The delta memory's `source_id` — the id an `edges` row addresses, since
    /// the space fence resolves a memory endpoint by `source_id`, not by `id`.
    pub memory_source_id: String,
    /// The added prose itself.
    pub delta_text: String,
}

/// The lines present in `new_content` that are absent from `base_content`, in
/// the order they appear in `new_content`.
///
/// ponytail: a line-set difference, not a positional diff — no diff crate, and
/// none needed. A moved line is not new evidence (its text was already in the
/// base), so treating a move as "unchanged" is what this path wants, not a
/// shortcut it tolerates. The ceiling is that a line edited in place shows up
/// whole rather than as a within-line delta; the upgrade path, if a within-line
/// span ever matters, is a real diff over the same two strings.
fn added_lines(base_content: &str, new_content: &str) -> String {
    let base: std::collections::HashSet<&str> = base_content.lines().collect();
    new_content
        .lines()
        .filter(|line| !line.trim().is_empty() && !base.contains(line))
        .collect::<Vec<_>>()
        .join("\n")
}

impl MemoryDB {
    /// Create the additive M5 claim-identity tables inside `tx`.
    ///
    /// Idempotent: every statement is `IF NOT EXISTS`, so a resumed or
    /// re-fired migration converges rather than failing.
    pub(super) async fn ensure_claim_identity_tables(
        tx: &libsql::Transaction,
    ) -> Result<(), WenlanError> {
        tx.execute_batch(
            // A logical claim. `claim_id` is durable and opaque: never reused,
            // never reassigned, never content-derived, so it survives edits
            // that preserve alignment (artifact 1 §1).
            "CREATE TABLE IF NOT EXISTS claims (
                claim_id TEXT PRIMARY KEY,
                page_id TEXT NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
                created_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_claims_page ON claims(page_id);

            -- An immutable revision. claim_revision_id is content-addressed:
            -- H(claim_id, predecessor_revision_id, canonical_text_digest,
            -- claim_kind). predecessor_revision_id is '' for a claim's first
            -- revision so the hash is total and the chain has ONE root.
            -- claim_id participates, so two identical sentences on one page
            -- get different revision ids and a revision can never be silently
            -- shared between logical claims (artifact 1 §1).
            CREATE TABLE IF NOT EXISTS claim_revisions (
                claim_revision_id TEXT PRIMARY KEY,
                claim_id TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
                predecessor_revision_id TEXT NOT NULL,
                canonical_text TEXT NOT NULL,
                canonical_text_digest TEXT NOT NULL,
                claim_kind TEXT NOT NULL,
                extractor_version INTEGER NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_claim_revisions_claim
                ON claim_revisions(claim_id);
            CREATE INDEX IF NOT EXISTS idx_claim_revisions_predecessor
                ON claim_revisions(predecessor_revision_id)
                WHERE predecessor_revision_id <> '';

            -- Immutability is what makes trust durable: the object a human or
            -- a model approved must not change underneath the approval
            -- (artifact 1 §1). Enforced in the schema rather than by
            -- convention, because a single stray UPDATE would silently move
            -- every support and attestation onto text nobody judged.
            CREATE TRIGGER IF NOT EXISTS claim_revisions_are_immutable
            BEFORE UPDATE ON claim_revisions
            BEGIN
                SELECT RAISE(ABORT, 'claim_revisions are immutable');
            END;

            -- An anchor binds a revision to the exact source bytes that
            -- produced it. Offsets alone are never trusted -- a document edit
            -- shifts them, and a stale offset pointing at plausible text is
            -- precisely how support gets attached to the wrong sentence. The
            -- digest is the check that makes the offsets safe (artifact 1 §3).
            CREATE TABLE IF NOT EXISTS claim_anchors (
                claim_revision_id TEXT NOT NULL
                    REFERENCES claim_revisions(claim_revision_id) ON DELETE CASCADE,
                source_doc_id TEXT NOT NULL,
                source_version INTEGER NOT NULL,
                span_start INTEGER NOT NULL,
                span_end INTEGER NOT NULL,
                span_digest TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                PRIMARY KEY (claim_revision_id, source_doc_id, source_version, span_start, span_end)
            );
            CREATE INDEX IF NOT EXISTS idx_claim_anchors_source
                ON claim_anchors(source_doc_id, source_version);

            CREATE TRIGGER IF NOT EXISTS claim_anchors_are_immutable
            BEFORE UPDATE ON claim_anchors
            BEGIN
                SELECT RAISE(ABORT, 'claim_anchors are immutable');
            END;

            -- Which revisions constitute a given page version.
            CREATE TABLE IF NOT EXISTS page_version_claims (
                page_id TEXT NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
                page_version INTEGER NOT NULL,
                claim_revision_id TEXT NOT NULL
                    REFERENCES claim_revisions(claim_revision_id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL,
                PRIMARY KEY (page_id, page_version, claim_revision_id)
            );
            CREATE INDEX IF NOT EXISTS idx_page_version_claims_revision
                ON page_version_claims(claim_revision_id);

            -- Derivation completeness. Validated by page-version digest AND
            -- extractor_version: identical page text under a changed extractor
            -- yields a different claim set, so a digest-only check would accept
            -- an inventory that no longer describes the page (artifact 2 §1
            -- condition 1, artifact 9 §4).
            --
            -- inventory_count is stored, and zero is a VALID value: a page that
            -- derives to zero claims still gets a marker, because skipping it
            -- would leave the page in the 'never derived' state -- an unknown,
            -- not an outcome -- and hold readiness under 100% forever
            -- (artifact 9 §5).
            CREATE TABLE IF NOT EXISTS claim_derivation_markers (
                page_id TEXT NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
                page_version INTEGER NOT NULL,
                page_version_digest TEXT NOT NULL,
                extractor_version INTEGER NOT NULL,
                inventory_count INTEGER NOT NULL CHECK(inventory_count >= 0),
                created_at INTEGER NOT NULL,
                PRIMARY KEY (page_id, page_version)
            );

            -- The five-part cache key. Changing ANY component misses, so
            -- scores from two weight-sets of one model_id are never compared
            -- under a single threshold, a prompt edit invalidates every cached
            -- score, and a cached score can never be attributed to text the
            -- judge never saw (artifact 6 §2). threshold_at_write is recorded
            -- so a later threshold change is detectable per row instead of
            -- silently reinterpreting stored scores.
            CREATE TABLE IF NOT EXISTS entailment_cache (
                claim_text_digest TEXT NOT NULL,
                source_span_digest TEXT NOT NULL,
                model_id TEXT NOT NULL,
                model_version TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                score REAL NOT NULL,
                threshold_at_write REAL NOT NULL,
                backend TEXT NOT NULL,
                scored_at INTEGER NOT NULL,
                PRIMARY KEY (claim_text_digest, source_span_digest, model_id, model_version, prompt_version)
            );
            CREATE INDEX IF NOT EXISTS idx_entailment_cache_model
                ON entailment_cache(model_id, model_version);
            CREATE INDEX IF NOT EXISTS idx_entailment_cache_prompt
                ON entailment_cache(prompt_version);
            CREATE INDEX IF NOT EXISTS idx_entailment_cache_scored_at
                ON entailment_cache(scored_at);

            -- The two truth axes, independent by construction. support_status
            -- is the machine axis and is a whitelist -- an unanticipated state
            -- is provisional by construction rather than by enumeration
            -- (artifact 2 §1). human_reviewed is the human axis and is bound to
            -- the exact reviewed version AND digest, so approval is of a
            -- specific text, never of a page in perpetuity (artifact 2 §2).
            CREATE TABLE IF NOT EXISTS page_truth_state (
                page_id TEXT PRIMARY KEY REFERENCES pages(id) ON DELETE CASCADE,
                page_version INTEGER NOT NULL,
                support_status TEXT NOT NULL
                    CHECK(support_status IN ('supported','provisional')),
                provisional_reason TEXT,
                human_reviewed INTEGER NOT NULL DEFAULT 0
                    CHECK(human_reviewed IN (0,1)),
                reviewed_page_version INTEGER,
                reviewed_page_digest TEXT,
                updated_at INTEGER NOT NULL,
                CHECK (human_reviewed = 0
                       OR (reviewed_page_version IS NOT NULL
                           AND reviewed_page_digest IS NOT NULL))
            );
            CREATE INDEX IF NOT EXISTS idx_page_truth_state_status
                ON page_truth_state(support_status);

            -- Durable derivation work. The lease columns make a crashed worker
            -- reclaimable instead of parking a page forever.
            CREATE TABLE IF NOT EXISTS claim_derivation_jobs (
                job_id TEXT PRIMARY KEY,
                page_id TEXT NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
                page_version INTEGER NOT NULL,
                status TEXT NOT NULL
                    CHECK(status IN ('pending','leased','done','parked')),
                lease_owner TEXT,
                lease_expires_at INTEGER,
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_claim_derivation_jobs_claimable
                ON claim_derivation_jobs(status, lease_expires_at);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_claim_derivation_jobs_page_version
                ON claim_derivation_jobs(page_id, page_version);

            -- One-shot presence nonces. The DIGEST is the key, never the nonce
            -- itself (artifact 5 §6). Consumption is an insert inside the
            -- mutation's own transaction, so there is no window where one
            -- succeeded and the other did not.
            CREATE TABLE IF NOT EXISTS presence_nonces (
                nonce_digest TEXT PRIMARY KEY,
                caller_id TEXT NOT NULL,
                operation_id TEXT NOT NULL,
                request_digest TEXT NOT NULL,
                consumed_at INTEGER NOT NULL,
                expires_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_presence_nonces_lookup
                ON presence_nonces(caller_id, operation_id);
            CREATE INDEX IF NOT EXISTS idx_presence_nonces_expiry
                ON presence_nonces(expires_at);

            -- Receipts serve the replay lookup that must run BEFORE capability
            -- validation. Validating first would burn the nonce on a retry of
            -- an already-applied mutation, turning an idempotent retry into a
            -- hard failure -- the client would conclude the write failed when
            -- it had actually succeeded (artifact 5 §4).
            CREATE TABLE IF NOT EXISTS presence_receipts (
                caller_id TEXT NOT NULL,
                operation_id TEXT NOT NULL,
                request_digest TEXT NOT NULL,
                response_json TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                PRIMARY KEY (caller_id, operation_id)
            );",
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m97 claim-identity DDL: {error}")))?;
        Ok(())
    }

    /// Give every existing page a truth row, `provisional` and unreviewed.
    ///
    /// **Nothing is inferred.** Not from `citations`, not from a page having
    /// been distilled, not from any legacy field that reads like approval. A
    /// page's support status means "the D8 finalizer evaluated this exact
    /// version and found supporting evidence", and that has never run for any
    /// row here, so the only honest value is `provisional`. `human_reviewed`
    /// is stricter still: it means a specific human approved a specific text,
    /// and no such record exists to migrate from. Inventing either would put
    /// the two axes this rung exists to separate back on the same footing —
    /// and it would do so silently, since a wrongly-`supported` page looks
    /// exactly like a correctly-`supported` one.
    ///
    /// **Resumable by construction, with no cursor.** `WHERE NOT EXISTS` makes
    /// the statement fill precisely the gap that remains, so an interrupted run
    /// resumes by being run again. A cursor would be strictly weaker: it can go
    /// stale or be written wrong, whereas "which pages lack a row" is derived
    /// from the data every time.
    ///
    /// It also means the backfill never touches a row that already exists.
    /// At migration time there can be none — `page_truth_state` was created one
    /// migration earlier — so that only matters on a re-run, where clobbering a
    /// real evaluation with `provisional` would be a regression, not a repair.
    ///
    /// The post-condition is coverage, checked rather than assumed: if any page
    /// still lacks a row when the statement finishes, the migration fails
    /// rather than reporting success over a partial backfill.
    pub(super) async fn backfill_page_truth_state(
        tx: &libsql::Transaction,
    ) -> Result<u64, WenlanError> {
        let now = chrono::Utc::now().timestamp();
        let filled = tx
            .execute(
                "INSERT INTO page_truth_state
                     (page_id, page_version, support_status, provisional_reason,
                      human_reviewed, updated_at)
                 SELECT p.id, p.version, 'provisional',
                        'never evaluated: predates claim derivation', 0, ?1
                   FROM pages p
                  WHERE NOT EXISTS (
                      SELECT 1 FROM page_truth_state t WHERE t.page_id = p.id
                  )",
                libsql::params![now],
            )
            .await
            .map_err(|error| WenlanError::VectorDb(format!("m98 backfill truth state: {error}")))?;

        let mut rows = tx
            .query(
                "SELECT count(*) FROM pages p
                  WHERE NOT EXISTS (
                      SELECT 1 FROM page_truth_state t WHERE t.page_id = p.id
                  )",
                (),
            )
            .await
            .map_err(|error| WenlanError::VectorDb(format!("m98 coverage check: {error}")))?;
        let uncovered: i64 = rows
            .next()
            .await
            .map_err(|error| WenlanError::VectorDb(format!("m98 coverage read: {error}")))?
            .ok_or_else(|| WenlanError::VectorDb("m98 coverage returned no row".into()))?
            .get(0)
            .map_err(|error| WenlanError::VectorDb(format!("m98 coverage decode: {error}")))?;
        if uncovered != 0 {
            return Err(WenlanError::VectorDb(format!(
                "m98 backfill left {uncovered} page(s) with no truth row"
            )));
        }
        Ok(filled)
    }

    /// Mint the `human_edit_delta` root for a page save that supplied a valid
    /// **exact** base, and store the delta as a span-addressable memory.
    ///
    /// This closes the gap both artifact 3 §5 and artifact 6 §2a name:
    /// `provenance_roots.root_kind` has permitted `human_capture` and
    /// `human_edit_delta` since migration 81, but nothing in the tree minted
    /// either — the only production minter is
    /// `acquire_provenance_root("document_ingest", …)`. Without this, every
    /// human-edited page is permanently unsupported (a human sentence with no
    /// evidence keeps its page `provisional` forever) and attestation has no
    /// valid source root, since §5 requires the attesting root to be one of the
    /// two human kinds.
    ///
    /// **Only the exact-base shape mints anything** (D4, restated in artifact 6
    /// §2a's table). The other two shapes are handled by not being handled here:
    ///
    /// | Save shape | Result |
    /// |---|---|
    /// | valid exact base | delta minted — this function |
    /// | stale base | `Err(Conflict)`, nothing written |
    /// | base omitted | caller never reaches this function; prose saves with no evidence |
    ///
    /// The stale case returns before any write, so "conflict, nothing written"
    /// is structural rather than a promise: the base check is the first thing
    /// that happens and there is nothing to roll back.
    ///
    /// **Order is mint-then-store, deliberately.** Roots are content-addressed
    /// and converge under `ON CONFLICT`, so a root left behind by a failed store
    /// is inert and a retry re-acquires the same one. A memory left behind by a
    /// failed mint would be a stray row in the user's corpus that search can
    /// surface. The cheap orphan is the one to risk.
    ///
    /// **The delta memory takes the page's space, not a default.** The rebuilt
    /// space fence resolves a `claim_revision → memory` support edge by
    /// comparing the claim's page space against the memory's, so a delta filed
    /// anywhere else would be silently unciteable — the edge would be refused at
    /// write time by the fence, not by anything in this module.
    ///
    /// `memory_type` is left NULL on purpose. D4 lets only an `observation`
    /// delta act as evidence, and classifying which kind this prose is belongs
    /// to the extractor, not to the store. NULL is D4's "legacy/missing kind —
    /// non-voting", so an unclassified delta is inert rather than assumed
    /// eligible.
    pub async fn mint_human_edit_delta(
        &self,
        page_id: &str,
        base_version: i64,
        base_content_digest: &str,
        new_content: &str,
    ) -> Result<Option<HumanEditDelta>, WenlanError> {
        // Scoped so the connection guard is dropped before
        // `acquire_provenance_root`, which locks the same mutex itself.
        let (base_content, space) = {
            let conn = self.conn.lock().await;
            let mut rows = conn
                .query(
                    "SELECT content FROM page_history WHERE page_id = ?1 AND version = ?2",
                    libsql::params![page_id, base_version],
                )
                .await
                .map_err(|error| {
                    WenlanError::VectorDb(format!("human delta base lookup: {error}"))
                })?;
            let base_content: String = rows
                .next()
                .await
                .map_err(|error| WenlanError::VectorDb(format!("human delta base row: {error}")))?
                .ok_or_else(|| {
                    WenlanError::Conflict(format!(
                        "human_delta_base_unknown: {page_id} has no version {base_version}"
                    ))
                })?
                .get(0)
                .map_err(|error| {
                    WenlanError::VectorDb(format!("human delta base decode: {error}"))
                })?;
            drop(rows);

            let mut rows = conn
                .query(
                    "SELECT space FROM pages WHERE id = ?1",
                    libsql::params![page_id],
                )
                .await
                .map_err(|error| {
                    WenlanError::VectorDb(format!("human delta page space: {error}"))
                })?;
            let space: Option<String> = rows
                .next()
                .await
                .map_err(|error| WenlanError::VectorDb(format!("human delta space row: {error}")))?
                .ok_or_else(|| {
                    WenlanError::Conflict(format!("human_delta_base_unknown: no page {page_id}"))
                })?
                .get(0)
                .map_err(|error| {
                    WenlanError::VectorDb(format!("human delta space decode: {error}"))
                })?;
            (base_content, space)
        };

        // T6: the human must have been looking at exactly this text. Compared
        // byte-exactly, never canonically -- see `revision_content_digest`.
        if crate::provenance::revision_content_digest(&base_content) != base_content_digest {
            return Err(WenlanError::Conflict(format!(
                "human_delta_base_stale: {page_id} version {base_version} is not the text this \
                 save was written against"
            )));
        }

        let delta_text = added_lines(&base_content, new_content);
        if delta_text.is_empty() {
            // A save that reorders or deletes adds no prose, so there is no new
            // evidence to ground. Not an error: refusing it would make a
            // legitimate edit look like a conflict.
            return Ok(None);
        }

        let signals = crate::provenance::IndependenceSignals {
            source_identity: Some(HUMAN_SOURCE_IDENTITY),
            agent_turn: None,
            import_batch: None,
        };
        let root_id = self
            .acquire_provenance_root("human_edit_delta", &delta_text, &signals)
            .await?;

        // Both ids derive from the root, which is itself content-addressed, so a
        // retry after a failed store lands on the same row rather than a second
        // copy of the same prose. `INSERT OR IGNORE` makes that convergence
        // explicit instead of relying on the retry never happening.
        let memory_source_id = format!("hed_{root_id}");
        let memory_id = format!("mem_{memory_source_id}");
        let now_ts = chrono::Utc::now().timestamp();
        let word_count = delta_text.split_whitespace().count() as i64;
        {
            let conn = self.conn.lock().await;
            conn.execute(
                "INSERT OR IGNORE INTO memories
                     (id, content, source, source_id, title, chunk_index, last_modified,
                      chunk_type, word_count, created_at, source_agent, space)
                 VALUES (?1, ?2, 'memory', ?3, ?4, 0, ?5, 'text', ?6, ?5, 'human', ?7)",
                libsql::params![
                    memory_id,
                    delta_text.clone(),
                    memory_source_id.clone(),
                    format!("Edit to {page_id}"),
                    now_ts,
                    word_count,
                    space,
                ],
            )
            .await
            .map_err(|error| WenlanError::VectorDb(format!("human delta memory store: {error}")))?;
        }

        Ok(Some(HumanEditDelta {
            root_id,
            memory_source_id,
            delta_text,
        }))
    }
}
