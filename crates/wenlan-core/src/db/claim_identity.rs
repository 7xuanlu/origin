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
//! one migration 98 transaction, tables first, because the rebuilt space fence
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

/// The span binding and the verdict a `supports` edge freezes into its
/// immutable payload (artifact 3 §4a's table).
///
/// One struct rather than nine arguments because these nine fields are a single
/// indivisible fact — "this exact text, judged this way, by this judge". Split
/// across a signature they can be assembled from two different judgments
/// without anything noticing, which is the failure §4a is written to prevent.
#[derive(Debug, Clone)]
pub struct SupportVerdict {
    /// Which chunk of the cited source the span was read from.
    ///
    /// A `source_id` names a *document*, and one document is several `memories`
    /// rows sharing that id — `store_raw_import_memories_batch` writes exactly
    /// that shape. Without this, resolving evidence by `source_id` alone picks
    /// whichever row the query happens to return: verification recomputes the
    /// span digest against the wrong chunk and refuses support that is perfectly
    /// valid, and the stored edge cannot say which chunk was ever verified.
    /// Fail-closed, so the visible symptom is unsupportable pages rather than
    /// wrong support — the same dead-substrate reading this rung exists to end.
    pub chunk_index: i64,
    /// The exact memory version the span was read from.
    pub source_version: i64,
    /// Byte offsets into that version, per `faithfulness::sentence_spans`.
    pub span_start: i64,
    pub span_end: i64,
    /// SHA-256 of the exact span bytes.
    pub span_digest: String,
    /// Which judge produced this.
    pub model_id: String,
    pub model_version: String,
    pub prompt_version: String,
    /// The verdict, and the bar it cleared.
    pub score: f64,
    pub threshold_at_write: f64,
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

/// The retraction body shared by the two page-side lifecycle triggers.
///
/// `anchor` is the row alias naming the page — `NEW` for the space move, `OLD`
/// for the delete. One string for both, the same reason the space fence has one
/// `FENCE_BODY`: two hand-written copies of "which edges depend on this page"
/// are two chances to answer it differently.
///
/// The `CASE` picks the endpoint that carries the claim side of the edge —
/// `supports` asserts *from* a revision, `attests` asserts *into* one — and
/// yields NULL for every other edge type, so `NULL IN (...)` leaves those rows
/// alone. Writing it as one comparison rather than two `OR`ed subqueries keeps
/// the dependent set defined in exactly one place.
fn page_retraction_body(anchor: &str) -> String {
    format!(
        "
                 UPDATE edges
                    SET valid_until = CAST(strftime('%s','now') AS INTEGER),
                        superseded_by = NULL
                  WHERE valid_until IS NULL
                    AND CASE
                          WHEN edge_type = 'supports' AND src_kind = 'claim_revision'
                            THEN src_id
                          WHEN edge_type = 'attests' AND dst_kind = 'claim_revision'
                            THEN dst_id
                        END IN (
                          SELECT cr.claim_revision_id
                            FROM claim_revisions cr
                            JOIN claims c ON c.claim_id = cr.claim_id
                           WHERE c.page_id = {anchor}.id
                        );
             "
    )
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

            -- Which candidates THIS run had to weigh, and how far each got.
            --
            -- `entailment_cache` cannot answer that question and was never
            -- meant to. It is keyed by content digests alone, so it is global
            -- and timeless: a row means some judge once scored this text
            -- against this span, with no way back to which run required it or
            -- whether the run finished. Reading a cache hit as this-claim-has-
            -- been-judged therefore lets a run that concluded on one candidate
            -- and timed out on another read as fully judged -- and a claim that
            -- is fully judged with no qualifying edge is Refuted, which costs
            -- the page its file for evaluation that never finished.
            --
            -- The run is (page_id, page_version, run_generation), and the
            -- generation is the load-bearing third of that. `(page_id,
            -- page_version)` names a JOB, not a run attempt: an expired or
            -- released lease reuses the same row, so rows written by an attempt
            -- that crashed halfway sit beside rows written by the attempt that
            -- replaced it, and the two read as one complete run. That is the
            -- worst direction to be wrong in -- a claim that looks fully judged
            -- with no qualifying edge is Refuted, which costs the page its
            -- file. A generation is allocated per LEASE (see
            -- `claim_run_generations`), so every attempt is its own run and
            -- rows from two of them can never be combined.
            --
            -- The candidate is identified by its exact LOCATION, not by its
            -- span digest. A digest alone is span CONTENT, and a document may
            -- contain the same sentence twice: two genuinely distinct citations
            -- then collide on one row, and concluding on one silently discharges
            -- the obligation to conclude on the other. The columns mirror
            -- `write_support_edge`'s span locator exactly -- source, chunk,
            -- offsets, digest -- because matching an attempt row to the support
            -- edge it produced is the whole point of recording it.
            --
            -- `outcome` is a whitelist of two, and 'deferred' deliberately
            -- covers timed-out, errored, and malformed alike: they differ in
            -- how they happened and not at all in what they mean here, which
            -- is that this candidate has no conclusion. A third value would
            -- add a distinction no reader acts on.
            CREATE TABLE IF NOT EXISTS claim_judgment_attempts (
                page_id TEXT NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
                page_version INTEGER NOT NULL,
                run_generation INTEGER NOT NULL,
                claim_revision_id TEXT NOT NULL
                    REFERENCES claim_revisions(claim_revision_id) ON DELETE CASCADE,
                candidate_source_id TEXT NOT NULL,
                candidate_chunk_index INTEGER NOT NULL,
                candidate_span_start INTEGER NOT NULL,
                candidate_span_end INTEGER NOT NULL,
                candidate_digest TEXT NOT NULL,
                outcome TEXT NOT NULL CHECK(outcome IN ('concluded','deferred')),
                updated_at INTEGER NOT NULL,
                PRIMARY KEY (page_id, page_version, run_generation, claim_revision_id,
                             candidate_source_id, candidate_chunk_index,
                             candidate_span_start, candidate_span_end, candidate_digest)
            );

            -- What this run OWED a conclusion on, recorded before it judged.
            --
            -- `claim_judgment_attempts` and the live support edges are both
            -- OUTPUTS of a run, and a run that dies mid-flight leaves neither
            -- for the candidate it never reached. That is the hole they cannot
            -- close between them: candidate A concludes, the worker dies before
            -- writing candidate B's attempt row, and B produced no edge because
            -- nothing ever scored it. Evaluation then sees one candidate, one
            -- conclusion, and no evidence at all that a second obligation
            -- existed -- so a run that covered half its inventory publishes as
            -- if it had covered all of it, which is exactly what row 6 forbids.
            --
            -- The marker's `inventory_count` does not cover this. It counts
            -- CLAIMS, and the missing thing is a CANDIDATE: B's claim is on the
            -- membership roll and accounted for, with one of its two citations
            -- silently dropped.
            --
            -- So the obligation is written down first and independently, and
            -- evaluation compares outputs against it rather than against
            -- themselves. Columns mirror `claim_judgment_attempts` exactly --
            -- same locator, same run identity -- because the whole operation is
            -- a set difference between the two.
            CREATE TABLE IF NOT EXISTS claim_run_candidates (
                page_id TEXT NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
                page_version INTEGER NOT NULL,
                run_generation INTEGER NOT NULL,
                claim_revision_id TEXT NOT NULL
                    REFERENCES claim_revisions(claim_revision_id) ON DELETE CASCADE,
                candidate_source_id TEXT NOT NULL,
                candidate_chunk_index INTEGER NOT NULL,
                candidate_span_start INTEGER NOT NULL,
                candidate_span_end INTEGER NOT NULL,
                candidate_digest TEXT NOT NULL,
                PRIMARY KEY (page_id, page_version, run_generation, claim_revision_id,
                             candidate_source_id, candidate_chunk_index,
                             candidate_span_start, candidate_span_end, candidate_digest)
            );

            -- The seal that says the row set above is COMPLETE for this run.
            --
            -- Without it the inventory has the defect it was built to fix, one
            -- level up: a run that wrote A's inventory row and died before B's
            -- leaves an inventory of one, and an inventory nobody can tell is
            -- partial proves nothing. The seal and the rows go in under one
            -- transaction, so the seal's presence IS the completeness claim.
            --
            -- It also separates the two absences that matter. A run with an
            -- empty candidate set writes zero rows and a seal; a run that died
            -- before recording anything writes zero rows and no seal. Row
            -- counts cannot tell those apart and they land on opposite
            -- verdicts.
            CREATE TABLE IF NOT EXISTS claim_run_inventories (
                page_id TEXT NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
                page_version INTEGER NOT NULL,
                run_generation INTEGER NOT NULL,
                sealed_at INTEGER NOT NULL,
                PRIMARY KEY (page_id, page_version, run_generation)
            );

            -- The run-generation allocator: one row, monotonically increasing.
            --
            -- Deliberately global rather than a counter on the job row. A job
            -- row is DELETED and recreated by the backlog sweep whenever its
            -- marker or its support goes stale, and a per-job counter would
            -- restart at 0 there -- handing the next run a generation that
            -- orphaned attempt rows from an earlier run already carry. A single
            -- monotone source has no reset to reason about, and gaps (a lease
            -- that allocated and then found nothing claimable) cost nothing.
            CREATE TABLE IF NOT EXISTS claim_run_generations (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                last_generation INTEGER NOT NULL
            );
            INSERT OR IGNORE INTO claim_run_generations (id, last_generation)
                 VALUES (1, 0);

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
                -- When a judgement last ran, or NULL if none ever has.
                -- provisional alone conflates never-looked-at with
                -- looked-and-the-evidence-fell-short, and only the second is a
                -- verdict that may cost a page its projected file. Migration 99
                -- leaves this NULL for every backfilled page, which is what
                -- keeps a cutover from archiving a whole vault on day one.
                evaluated_at INTEGER,
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
            --
            -- `run_generation` is the run this job is currently on, stamped
            -- from `claim_run_generations` at every lease. It is what makes a
            -- reclaimed or retried job a NEW run rather than a continuation of
            -- the one that died: judgment rows carry the generation they were
            -- written under, and evaluation reads only the generation the job
            -- row names now. 0 is the never-leased value -- no run has happened
            -- yet, and no allocated generation is ever 0.
            CREATE TABLE IF NOT EXISTS claim_derivation_jobs (
                job_id TEXT PRIMARY KEY,
                page_id TEXT NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
                page_version INTEGER NOT NULL,
                status TEXT NOT NULL
                    CHECK(status IN ('pending','leased','done','parked')),
                lease_owner TEXT,
                lease_expires_at INTEGER,
                attempts INTEGER NOT NULL DEFAULT 0,
                run_generation INTEGER NOT NULL DEFAULT 0,
                eligibility_generation INTEGER NOT NULL DEFAULT 0,
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
        .map_err(|error| WenlanError::VectorDb(format!("m98 claim-identity DDL: {error}")))?;
        Ok(())
    }

    /// Keep the active support/attestation set honest when the rows it depends
    /// on move or disappear (F5).
    ///
    /// An `edges` row is a claim about the world at a point in time, and
    /// `valid_until IS NULL` is the assertion that it is still true. Three
    /// ordinary mutations falsify one without touching it: moving a page to
    /// another space leaves its claims' support edges stamped with the old
    /// space, moving an evidence memory leaves an edge that is now cross-space
    /// with the fence never having fired, and deleting a page cascades the
    /// claims and revisions away while the edges — whose endpoints are
    /// polymorphic and therefore carry no foreign key — survive pointing at
    /// nothing. Each leaves a row that still reads as a live, fenced,
    /// space-consistent support.
    ///
    /// **Triggers rather than the call sites.** Nine production statements
    /// write `memories.space` or `pages.space`, and more will exist; a
    /// convention that every one of them must remember to retract is a
    /// convention with nine chances to be forgotten and no way to notice. The
    /// invariant belongs to storage, where it holds for writers that do not
    /// exist yet — the same reasoning the space fence and the human-root rule
    /// are already built on.
    ///
    /// Retraction is soft (`valid_until` stamped), never a delete: the support
    /// *was* true, and the history of when it stopped being true is the point.
    /// `superseded_by` stays NULL because nothing replaced these — they were
    /// invalidated, not re-asserted.
    ///
    /// The fence's UPDATE twin is scoped `NEW.valid_until IS NULL`, so
    /// retracting does not trip it. That matters for the space-move triggers,
    /// which necessarily fire *after* the endpoint has already moved: a fence
    /// that ran here would abort the very write that repairs the inconsistency.
    ///
    /// Idempotent (`IF NOT EXISTS`) like the rest of the M5 DDL.
    pub(super) async fn ensure_claim_edge_lifecycle_triggers(
        tx: &libsql::Transaction,
    ) -> Result<(), WenlanError> {
        tx.execute_batch(&format!(
            "CREATE TRIGGER IF NOT EXISTS m5_retract_claim_edges_on_page_space_move
             AFTER UPDATE OF space ON pages
             WHEN NEW.space IS NOT OLD.space
             BEGIN{move_body}END;

             CREATE TRIGGER IF NOT EXISTS m5_retract_claim_edges_on_page_delete
             BEFORE DELETE ON pages
             BEGIN{delete_body}END;

             CREATE TRIGGER IF NOT EXISTS m5_retract_support_on_memory_space_move
             AFTER UPDATE OF space ON memories
             WHEN NEW.space IS NOT OLD.space
             BEGIN
                 UPDATE edges
                    SET valid_until = CAST(strftime('%s','now') AS INTEGER),
                        superseded_by = NULL
                  WHERE valid_until IS NULL
                    AND edge_type = 'supports'
                    AND dst_kind = 'memory'
                    AND dst_id = NEW.source_id;
             END;

             -- Deleting the evidence itself is the page-delete hazard from the
             -- other end: `memories` has no foreign key pointing at it either,
             -- so the support survives naming a memory that is gone. Guarded
             -- twice because `memories` is chunked -- one logical memory is
             -- several rows sharing a `source_id`, and deleting chunk 3 of 5
             -- retracts nothing. The edge probe comes first so the common case
             -- (a delete with no support edge anywhere near it, including every
             -- bulk delete) costs one lookup on the partial support index and
             -- stops.
             CREATE TRIGGER IF NOT EXISTS m5_retract_support_on_memory_delete
             BEFORE DELETE ON memories
             WHEN EXISTS (
                      SELECT 1 FROM edges
                       WHERE edge_type = 'supports' AND src_kind = 'claim_revision'
                         AND lineage = 'evidence' AND valid_until IS NULL
                         AND dst_id = OLD.source_id
                  )
              AND NOT EXISTS (
                      SELECT 1 FROM memories
                       WHERE source_id = OLD.source_id AND id <> OLD.id
                  )
             BEGIN
                 UPDATE edges
                    SET valid_until = CAST(strftime('%s','now') AS INTEGER),
                        superseded_by = NULL
                  WHERE valid_until IS NULL
                    AND edge_type = 'supports'
                    AND dst_kind = 'memory'
                    AND dst_id = OLD.source_id;
             END;

             -- The fourth F5 path, closed by prevention rather than repair.
             -- Flipping an attesting root from human to `generated` would leave
             -- an active human-style attestation over a machine root, and no
             -- retraction could fire afterwards anyway: the human-root trigger
             -- guards every UPDATE naming `attests`, so the repair would be
             -- refused by the rule it exists to restore. Nothing writes this
             -- column today and nothing coherently could -- `root_kind` is an
             -- input to `identity_digest`, so a root that changed kind would no
             -- longer be the row its own content address names.
             CREATE TRIGGER IF NOT EXISTS provenance_roots_kind_is_immutable
             BEFORE UPDATE OF root_kind ON provenance_roots
             WHEN NEW.root_kind IS NOT OLD.root_kind
             BEGIN
                 SELECT RAISE(ABORT, 'provenance_roots.root_kind is immutable');
             END;",
            move_body = page_retraction_body("NEW"),
            delete_body = page_retraction_body("OLD"),
        ))
        .await
        .map_err(|error| {
            WenlanError::VectorDb(format!("m100 claim-edge lifecycle triggers: {error}"))
        })?;
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
            .map_err(|error| WenlanError::VectorDb(format!("m99 backfill truth state: {error}")))?;

        let mut rows = tx
            .query(
                "SELECT count(*) FROM pages p
                  WHERE NOT EXISTS (
                      SELECT 1 FROM page_truth_state t WHERE t.page_id = p.id
                  )",
                (),
            )
            .await
            .map_err(|error| WenlanError::VectorDb(format!("m99 coverage check: {error}")))?;
        let uncovered: i64 = rows
            .next()
            .await
            .map_err(|error| WenlanError::VectorDb(format!("m99 coverage read: {error}")))?
            .ok_or_else(|| WenlanError::VectorDb("m99 coverage returned no row".into()))?
            .get(0)
            .map_err(|error| WenlanError::VectorDb(format!("m99 coverage decode: {error}")))?;
        if uncovered != 0 {
            return Err(WenlanError::VectorDb(format!(
                "m99 backfill left {uncovered} page(s) with no truth row"
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
        let (base_content, space, live_version) = {
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
                    "SELECT space, version FROM pages WHERE id = ?1",
                    libsql::params![page_id],
                )
                .await
                .map_err(|error| {
                    WenlanError::VectorDb(format!("human delta page space: {error}"))
                })?;
            let row = rows
                .next()
                .await
                .map_err(|error| WenlanError::VectorDb(format!("human delta space row: {error}")))?
                .ok_or_else(|| {
                    WenlanError::Conflict(format!("human_delta_base_unknown: no page {page_id}"))
                })?;
            let space: Option<String> = row.get(0).map_err(|error| {
                WenlanError::VectorDb(format!("human delta space decode: {error}"))
            })?;
            let live_version: i64 = row.get(1).map_err(|error| {
                WenlanError::VectorDb(format!("human delta version decode: {error}"))
            })?;
            (base_content, space, live_version)
        };

        // The base has to be the CURRENT text, not merely a real one. The digest
        // check below asks "did version N say what you think it said" and every
        // superseded version answers yes forever, because `page_history` is
        // immutable -- so on its own it accepts a save written against version 3
        // of a page that is now at version 9, mints the delta, and loses the
        // intervening edits with no conflict reported. That is the "stale base"
        // row of D4's table, and until this check it was the one row the code did
        // not implement.
        //
        // `append_page_history` writes the history row for the version a write
        // just produced, inside that write's own transaction, so the current
        // version is always present in `page_history` -- an equal `base_version`
        // can always be read back above, and this guard rejects only genuinely
        // superseded saves.
        if live_version != base_version {
            return Err(WenlanError::Conflict(format!(
                "human_delta_base_stale: {page_id} has moved to version {live_version} since this \
                 save was written against version {base_version}"
            )));
        }

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
                    space.clone(),
                ],
            )
            .await
            .map_err(|error| WenlanError::VectorDb(format!("human delta memory store: {error}")))?;

            // `INSERT OR IGNORE` converges a retry onto the same row, which is
            // the point -- but the row it converges onto carries whichever SPACE
            // won the race, and roots are content-addressed over the prose
            // ALONE. Two pages in different spaces gaining the identical line
            // therefore resolve to one memory filed in the first page's space.
            //
            // Left unchecked, this call returns Ok with evidence the space fence
            // will refuse to cite, and the refusal surfaces later at an unrelated
            // support write that has no way to explain itself. Read the stored
            // space back and refuse HERE, where the cause is.
            //
            // The identity question underneath -- whether one sentence written
            // twice in two spaces is one piece of evidence or two -- is deferred
            // to M5's identity axis (docs/plans/2026-07-28-m5-pr-a-review-followups.md).
            // This closes the silent half only.
            let mut rows = conn
                .query(
                    "SELECT space FROM memories WHERE source_id = ?1",
                    libsql::params![memory_source_id.clone()],
                )
                .await
                .map_err(|error| {
                    WenlanError::VectorDb(format!("human delta space readback: {error}"))
                })?;
            let stored_space: Option<String> = rows
                .next()
                .await
                .map_err(|error| {
                    WenlanError::VectorDb(format!("human delta space readback row: {error}"))
                })?
                .ok_or_else(|| {
                    WenlanError::VectorDb(
                        "human delta space readback: the row just written is missing".to_string(),
                    )
                })?
                .get(0)
                .map_err(|error| {
                    WenlanError::VectorDb(format!("human delta space readback decode: {error}"))
                })?;
            if stored_space != space {
                return Err(WenlanError::Conflict(format!(
                    "human_delta_space_conflict: this prose already exists as evidence in another \
                     space, so the delta minted for {page_id} would be uncitable there"
                )));
            }
        }

        Ok(Some(HumanEditDelta {
            root_id,
            memory_source_id,
            delta_text,
        }))
    }

    /// Write one `supports` edge, binding it to the exact span it was judged on
    /// and the exact verdict that judged it (artifact 3 §4a).
    ///
    /// §4a exists because `edges` has no span columns, so a
    /// `claim_revision → memory` edge names a *memory*, not the span inside it
    /// that was actually read. Three independent records of "the evidence" —
    /// the anchor, the entailment-cache key, and the edge — with nothing forcing
    /// them to agree is the drift pattern this program keeps hitting. This
    /// function is the one place that forces them to agree, and it refuses
    /// rather than repairs.
    ///
    /// Both §4a invariants are checked before anything is written:
    ///
    /// 1. **Same evidence.** `span_digest` must equal the digest of the live
    ///    bytes at `[span_start, span_end)` in the destination memory AND the
    ///    `source_span_digest` component of the cache key that produced the
    ///    verdict. Offsets alone are never trusted (artifact 1 §3): a span whose
    ///    digest no longer matches invalidates the edge rather than silently
    ///    re-pointing it at whatever text now occupies those offsets.
    /// 2. **Same verdict.** The `model_id`/`model_version`/`prompt_version`/
    ///    `score` must be the ones the cache recorded. A support edge may never
    ///    be written from a verdict it cannot name.
    ///
    /// **`grounded` is 0, and that is the rule rather than a stub.** §3 says
    /// grounding is inherited and never asserted — `supports` may be `grounded=1`
    /// *only if* the destination memory span is itself grounded. No memory→root
    /// link exists in the schema to inherit that from: the sole production
    /// minter of a grounded root is M3g's promotion sweep, which grounds
    /// `relates` edges, not memory spans. With nothing to inherit, writing 1
    /// would be manufacturing grounding for evidence that has none, which is the
    /// one thing §3 names outright. 0 satisfies the "only if" honestly.
    ///
    /// **The evidence root is resolved by the caller and verified here.** §5
    /// defines `root_id` as the provenance root of the cited evidence, and the
    /// `supports` CHECK requires it to be non-NULL. PR-A derived it by parsing an
    /// `hed_` prefix off the memory id, because that naming convention was the
    /// only memory→root link in the schema. The cost of that shortcut was the
    /// whole M5 blocker: only prose a human typed into the page itself was
    /// citable, so a distilled page could never be supported and
    /// `support_status = 'supported'` matched zero rows on every install.
    ///
    /// The convention is replaced by a check that is strictly stronger, not
    /// weaker. Roots are content-addressed —
    /// `identity_digest(root_kind, canonical_content)` — so the root a caller
    /// names is confirmed against the cited memory's OWN bytes. A root minted
    /// over any other content is refused. Where the old prefix was a promise
    /// about a *name*, this is a proof about *content*, and it generalizes to
    /// ingested evidence instead of excluding it.
    ///
    /// Still refuses rather than repairs: an unknown root, an inactive one, or a
    /// root recorded under a different `identity_version` (whose digest this
    /// code cannot recompute, so it cannot be checked) are all named refusals
    /// rather than an invented or assumed link.
    pub async fn write_support_edge(
        &self,
        claim_revision_id: &str,
        memory_source_id: &str,
        root_id: &str,
        verdict: &SupportVerdict,
    ) -> Result<String, WenlanError> {
        // A `supports` edge is what M5 reads as truth, so the verdict must
        // actually clear the bar it recorded. Everything below this line checks
        // that the three records AGREE; none of it checks that what they agree
        // on means "supported". Without this, an honestly-cached failure --
        // score 0.55 against a threshold of 0.7, all three records in perfect
        // agreement that the claim was NOT entailed -- writes an edge that
        // reads as support. Comparing against the cache cannot catch that,
        // because the cache faithfully recorded the failure.
        //
        // Written `!(score >= bar)` rather than `score < bar` so a NaN score is
        // a refusal too: every float comparison against NaN is false, so the
        // `<` form would wave it through.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(verdict.score >= verdict.threshold_at_write) {
            return Err(WenlanError::Conflict(format!(
                "support_verdict_below_threshold: {claim_revision_id} scored {} against a bar of \
                 {}; a verdict that failed its own threshold is not support",
                verdict.score, verdict.threshold_at_write
            )));
        }

        let conn = self.conn.lock().await;

        let claim_text_digest: String = {
            let mut rows = conn
                .query(
                    "SELECT canonical_text_digest FROM claim_revisions WHERE claim_revision_id = ?1",
                    libsql::params![claim_revision_id],
                )
                .await
                .map_err(|error| WenlanError::VectorDb(format!("support claim lookup: {error}")))?;
            rows.next()
                .await
                .map_err(|error| WenlanError::VectorDb(format!("support claim decode: {error}")))?
                .ok_or_else(|| {
                    WenlanError::Conflict(format!(
                        "support_claim_unknown: no claim revision {claim_revision_id}"
                    ))
                })?
                .get(0)
                .map_err(|error| WenlanError::VectorDb(format!("support claim decode: {error}")))?
        };

        // The evidence row, addressed by CHUNK and not by document. `source_id`
        // names a document; one document is many `memories` rows sharing that
        // id, so a lookup on `source_id` alone resolves to an arbitrary chunk
        // and the digest check below then answers about text the verdict never
        // read.
        //
        // Nothing in the schema makes `(source_id, chunk_index)` unique -- only
        // a plain index on `source_id` exists -- so the pair is a convention the
        // rest of the tree keeps rather than a guarantee this code may lean on.
        // A second row is therefore read for and REFUSED by name: an ambiguous
        // corpus is a state this function cannot verify against, and picking one
        // of two candidates would be exactly the silent arbitrary choice the
        // chunk index was added to remove.
        let (content, space, live_version): (String, String, i64) = {
            let mut rows = conn
                .query(
                    // The fallback is the reserved UUID sentinel, never the word
                    // "unfiled" -- that word is a legal user space name, and the
                    // sentinel is a UUID precisely so the two cannot collide. A
                    // literal here would produce a space the fence then rejects
                    // with a misleading cross-space error. Unreachable while
                    // `memories.space` stays NOT NULL (migration 91), which is
                    // exactly why it has to be right rather than merely untested.
                    "SELECT content, COALESCE(space, ?2), COALESCE(version, 1)
                     FROM memories WHERE source_id = ?1 AND chunk_index = ?3
                     LIMIT 2",
                    libsql::params![
                        memory_source_id,
                        super::UNFILED_SPACE_ID,
                        verdict.chunk_index
                    ],
                )
                .await
                .map_err(|error| {
                    WenlanError::VectorDb(format!("support memory lookup: {error}"))
                })?;
            let row = rows
                .next()
                .await
                .map_err(|error| WenlanError::VectorDb(format!("support memory decode: {error}")))?
                .ok_or_else(|| {
                    WenlanError::Conflict(format!(
                        "support_evidence_unknown: no memory {memory_source_id} chunk {}",
                        verdict.chunk_index
                    ))
                })?;
            let resolved = (
                row.get(0).map_err(|error| {
                    WenlanError::VectorDb(format!("support memory decode: {error}"))
                })?,
                row.get(1).map_err(|error| {
                    WenlanError::VectorDb(format!("support memory decode: {error}"))
                })?,
                row.get(2).map_err(|error| {
                    WenlanError::VectorDb(format!("support memory decode: {error}"))
                })?,
            );
            if rows
                .next()
                .await
                .map_err(|error| WenlanError::VectorDb(format!("support memory decode: {error}")))?
                .is_some()
            {
                return Err(WenlanError::Conflict(format!(
                    "support_evidence_ambiguous: {memory_source_id} has more than one chunk {}; \
                     a support edge may not name evidence that cannot be resolved to one row",
                    verdict.chunk_index
                )));
            }
            resolved
        };

        // §5, and the replacement for PR-A's `hed_` prefix: prove the root the
        // caller named actually identifies THIS evidence. The schema stores no
        // memory->root column, but it does not have to -- roots are
        // content-addressed, so the binding is RECOMPUTABLE from the evidence's
        // own bytes even though it is not stored. That makes this a check rather
        // than the trust the prefix convention asked for.
        //
        // The root must be minted over the cited memory row's content, which is
        // what `acquire_provenance_root` converges on for identical bytes. A root
        // minted over some larger document that merely CONTAINS this memory does
        // not verify, and that refusal is correct: §4a's whole subject is the
        // exact text a verdict read, and a whole-document root cannot answer for
        // a chunk of it.
        {
            let mut rows = conn
                .query(
                    "SELECT identity_version, identity_digest, root_kind, status
                       FROM provenance_roots WHERE root_id = ?1",
                    libsql::params![root_id],
                )
                .await
                .map_err(|error| WenlanError::VectorDb(format!("support root lookup: {error}")))?;
            let row = rows
                .next()
                .await
                .map_err(|error| WenlanError::VectorDb(format!("support root decode: {error}")))?
                .ok_or_else(|| {
                    WenlanError::Conflict(format!(
                        "support_root_unknown: no provenance root {root_id}; a support edge may \
                         not cite a root that does not exist"
                    ))
                })?;
            let identity_version: i64 = row
                .get(0)
                .map_err(|error| WenlanError::VectorDb(format!("support root decode: {error}")))?;
            let identity_digest: String = row
                .get(1)
                .map_err(|error| WenlanError::VectorDb(format!("support root decode: {error}")))?;
            let root_kind: String = row
                .get(2)
                .map_err(|error| WenlanError::VectorDb(format!("support root decode: {error}")))?;
            let status: String = row
                .get(3)
                .map_err(|error| WenlanError::VectorDb(format!("support root decode: {error}")))?;

            if status != "active" {
                return Err(WenlanError::Conflict(format!(
                    "support_root_inactive: provenance root {root_id} is '{status}'; evidence \
                     whose root is not active may not back a claim"
                )));
            }
            // A digest written under a different identity scheme cannot be
            // recomputed by this code, so the binding cannot be checked at all.
            // Unknown is not true: refuse rather than accept it unverified.
            if identity_version != crate::provenance::IDENTITY_VERSION {
                return Err(WenlanError::Conflict(format!(
                    "support_root_identity_version: root {root_id} was recorded under identity \
                     version {identity_version}, which this code cannot recompute (it verifies \
                     version {})",
                    crate::provenance::IDENTITY_VERSION
                )));
            }
            if crate::provenance::identity_digest(&root_kind, &content) != identity_digest {
                return Err(WenlanError::Conflict(format!(
                    "support_root_not_evidence_root: root {root_id} was minted over different \
                     content than {memory_source_id}; a support edge may not cite a root that \
                     does not identify the evidence it names"
                )));
            }
        }

        // The payload records which VERSION the span was read from, and until
        // this check nothing made that number true -- the caller supplied it and
        // it was copied into the edge verbatim. The digest check below proves the
        // BYTES are right; it says nothing about the version they are labelled
        // with. A verdict claiming version 1 of a memory now at version 7 stores
        // a provenance record that is false in the one field a reader would use
        // to fetch the text back, and false in a way no later reader can detect,
        // because the digest agrees.
        //
        // Fail-closed on disagreement rather than overwriting with the live
        // number: a caller that judged an older version and mislabelled it, and a
        // caller that judged the live version and mistyped the label, are
        // indistinguishable here, and silently rewriting the field would launder
        // the first case into a record that looks correct.
        if live_version != verdict.source_version {
            return Err(WenlanError::Conflict(format!(
                "support_source_version_stale: verdict names version {} of {memory_source_id}, \
                 which is at version {live_version}",
                verdict.source_version
            )));
        }

        // Invariant 1, live-bytes half. `get` rather than `[..]`: these are byte
        // offsets into a version that may have changed under us, so a boundary
        // that no longer lands on a char is a refusal, never a panic.
        let start = usize::try_from(verdict.span_start).unwrap_or(usize::MAX);
        let end = usize::try_from(verdict.span_end).unwrap_or(usize::MAX);
        let span = content.get(start..end).ok_or_else(|| {
            WenlanError::Conflict(format!(
                "support_span_unreadable: [{start}, {end}) is not a valid span of \
                 {memory_source_id}"
            ))
        })?;
        if crate::provenance::revision_content_digest(span) != verdict.span_digest {
            return Err(WenlanError::Conflict(format!(
                "support_span_moved: the bytes at [{start}, {end}) in {memory_source_id} are not \
                 the ones this verdict judged"
            )));
        }

        // Invariant 2: the verdict must be one the cache actually recorded,
        // keyed by all five parts (artifact 6 §2).
        let (cached_score, cached_threshold): (f64, f64) = {
            let mut rows = conn
                .query(
                    "SELECT score, threshold_at_write FROM entailment_cache
                     WHERE claim_text_digest = ?1 AND source_span_digest = ?2
                       AND model_id = ?3 AND model_version = ?4 AND prompt_version = ?5
                       AND backend = ?6",
                    libsql::params![
                        claim_text_digest,
                        verdict.span_digest.clone(),
                        verdict.model_id.clone(),
                        verdict.model_version.clone(),
                        verdict.prompt_version.clone(),
                        crate::claim_judge::M5_CLAIM_ENTAILMENT_CACHE_BACKEND,
                    ],
                )
                .await
                .map_err(|error| WenlanError::VectorDb(format!("support cache lookup: {error}")))?;
            let row = rows
                .next()
                .await
                .map_err(|error| WenlanError::VectorDb(format!("support cache decode: {error}")))?
                .ok_or_else(|| {
                    WenlanError::Conflict(
                        "support_verdict_uncached: this edge cannot name the verdict that \
                         produced it"
                            .to_string(),
                    )
                })?;
            (
                row.get(0).map_err(|error| {
                    WenlanError::VectorDb(format!("support cache decode: {error}"))
                })?,
                row.get(1).map_err(|error| {
                    WenlanError::VectorDb(format!("support cache decode: {error}"))
                })?,
            )
        };
        if cached_score != verdict.score || cached_threshold != verdict.threshold_at_write {
            return Err(WenlanError::Conflict(format!(
                "support_verdict_altered: {claim_revision_id} cites a score the cache does not \
                 record"
            )));
        }

        // The LOCATION discriminates -- offsets and digest, not the digest
        // alone. Cross-model review caught the difference: a memory may contain
        // the same sentence twice, and two occurrences share a digest. Under a
        // digest-only discriminator those two genuinely distinct citations
        // collide on one edge id, and the second is either silently dropped or
        // (once the conflict below started refusing) rejected with a
        // "supersedes" error that misdescribes what happened -- nothing is
        // superseding anything, they are two different places in the text.
        //
        // Offsets in, and the two axes separate cleanly: a different PLACE is a
        // different edge, while a re-judgment of the SAME place keeps the same
        // id and is caught by the conflict check as the verdict change it is.
        //
        // The chunk index leads, because "a place" in a multi-chunk document is
        // a chunk AND an offset: offsets restart at 0 in every chunk, so two
        // genuinely distinct citations in chunks 0 and 1 would otherwise collide
        // on one edge id and the second would be refused as a superseding
        // re-judgment of the first.
        let span_locator = format!(
            "{}:{}:{}:{}",
            verdict.chunk_index, verdict.span_start, verdict.span_end, verdict.span_digest
        );
        let judgment_locator = serde_json::json!([
            span_locator,
            verdict.model_id,
            verdict.model_version,
            verdict.prompt_version,
        ])
        .to_string();
        let edge_id = crate::provenance::compute_edge_id(
            "supports",
            "claim_revision",
            claim_revision_id,
            "memory",
            memory_source_id,
            &judgment_locator,
        );
        let payload = serde_json::json!({
            "chunk_index": verdict.chunk_index,
            "source_version": verdict.source_version,
            "span_start": verdict.span_start,
            "span_end": verdict.span_end,
            "span_digest": verdict.span_digest,
            "model_id": verdict.model_id,
            "model_version": verdict.model_version,
            "prompt_version": verdict.prompt_version,
            "score": verdict.score,
            "threshold_at_write": verdict.threshold_at_write,
        })
        .to_string();

        // Judge identity participates in the edge id so a rolling upgrade can
        // replace a draining judgment without overwriting its audit record.
        // Score deliberately does not: changing a score under the SAME exact
        // judge triple is still a conflicting rewrite and must be explicit.
        //
        // No TOCTOU: `conn` is the single writer's one connection and this
        // guard has been held unbroken since the top of the function.
        let existing: Option<(String, Option<String>, Option<i64>)> = {
            let mut rows = conn
                .query(
                    "SELECT payload, superseded_by, valid_until
                       FROM edges WHERE edge_id = ?1",
                    libsql::params![edge_id.clone()],
                )
                .await
                .map_err(|error| {
                    WenlanError::VectorDb(format!("support edge conflict read: {error}"))
                })?;
            match rows.next().await.map_err(|error| {
                WenlanError::VectorDb(format!("support edge conflict decode: {error}"))
            })? {
                Some(row) => Some((
                    row.get(0).map_err(|error| {
                        WenlanError::VectorDb(format!("support edge conflict decode: {error}"))
                    })?,
                    row.get(1).map_err(|error| {
                        WenlanError::VectorDb(format!("support edge conflict decode: {error}"))
                    })?,
                    row.get(2).map_err(|error| {
                        WenlanError::VectorDb(format!("support edge conflict decode: {error}"))
                    })?,
                )),
                None => None,
            }
        };
        if let Some((stored, superseded_by, valid_until)) = existing {
            if stored == payload {
                // A lifecycle retraction leaves the exact edge id and payload
                // but no superseding edge. Reassert that same verified verdict
                // when the endpoint is back in scope. Judge replacement sets
                // `superseded_by`, and that historical row must remain dead.
                if valid_until.is_some() && superseded_by.is_none() {
                    conn.execute(
                        "UPDATE edges
                            SET valid_until = NULL, superseded_by = NULL
                          WHERE edge_id = ?1 AND payload = ?2
                            AND valid_until IS NOT NULL AND superseded_by IS NULL",
                        libsql::params![edge_id.clone(), payload.clone()],
                    )
                    .await
                    .map_err(|error| {
                        WenlanError::VectorDb(format!("support edge reactivation: {error}"))
                    })?;
                }
                return Ok(edge_id);
            }
            return Err(WenlanError::Conflict(format!(
                "support_verdict_supersedes_existing: {claim_revision_id} already has a support \
                 edge for this span naming a different verdict; supersede it explicitly rather \
                 than overwriting it"
            )));
        }

        let now = chrono::Utc::now().timestamp();
        let tx = conn
            .transaction_with_behavior(libsql::TransactionBehavior::Immediate)
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!("support edge transaction begin: {error}"))
            })?;
        let written = async {
            tx.execute(
            "INSERT INTO edges (edge_id, src_id, src_kind, dst_id, dst_kind, edge_type, lineage,
                                grounded, root_id, space, weight, payload, provenance,
                                operation_id, created_at, superseded_by, valid_until)
             VALUES (?1, ?2, 'claim_revision', ?3, 'memory', 'supports', 'evidence',
                     0, ?4, ?5, NULL, ?6, NULL, NULL, ?7, NULL, NULL)",
            libsql::params![
                edge_id.clone(),
                claim_revision_id,
                memory_source_id,
                root_id,
                space,
                payload.clone(),
                now,
            ],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("support edge write: {error}")))?;

            // The new row must exist before old rows point at it. Both writes
            // commit together, so readers see either the prior active verdict
            // or the replacement, never a gap between them.
            tx.execute(
                "UPDATE edges
                    SET superseded_by = ?1, valid_until = ?2
                  WHERE edge_type = 'supports'
                    AND src_kind = 'claim_revision' AND src_id = ?3
                    AND dst_kind = 'memory' AND dst_id = ?4
                    AND edge_id <> ?1 AND valid_until IS NULL
                    AND superseded_by IS NULL
                    AND json_extract(payload, '$.chunk_index') = ?5
                    AND json_extract(payload, '$.span_start') = ?6
                    AND json_extract(payload, '$.span_end') = ?7
                    AND json_extract(payload, '$.span_digest') = ?8",
                libsql::params![
                    edge_id.clone(),
                    now,
                    claim_revision_id,
                    memory_source_id,
                    verdict.chunk_index,
                    verdict.span_start,
                    verdict.span_end,
                    verdict.span_digest.clone(),
                ],
            )
            .await
            .map_err(|error| WenlanError::VectorDb(format!("support edge supersede: {error}")))?;
            Ok::<_, WenlanError>(())
        }
        .await;
        match written {
            Ok(()) => tx.commit().await.map_err(|error| {
                WenlanError::VectorDb(format!("support edge transaction commit: {error}"))
            })?,
            Err(error) => {
                let _ = tx.rollback().await;
                return Err(error);
            }
        }

        Ok(edge_id)
    }
}
