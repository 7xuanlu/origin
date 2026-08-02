// SPDX-License-Identifier: Apache-2.0
//! Migration 108 DDL: the M6 genesis substrate.
//!
//! Five tables and the two partial unique indexes that are D4's entire
//! exclusion mechanism. Every table starts empty and **nothing in this build
//! writes to any of them** — the writers are later rungs. The tables exist
//! first for the same reason migration 101's audit log did: a control that
//! appears at the same moment as the thing it controls has a window where it
//! does not.
//!
//! Contract sources, all on this branch:
//! - `docs/plans/2026-08-01-m6-state-machines.md` §4 (machine A, the candidate
//!   column list), §5 (machine B, the claim row and both partial unique
//!   indexes, quoted verbatim below), §7 (machine D, `genesis_coverage_state`),
//!   §9 (machine F, `genesis_frontier` + `genesis_group_coverage`).
//!
//! What the artifacts fix, and what this file decided:
//! - Column *names* are the artifacts'. Column *types*, primary keys, and the
//!   CHECK constraints are this file's, because §4/§9 name columns without
//!   typing them. Each choice is commented where it is not mechanical.
//! - The two partial unique indexes are byte-for-byte the artifact's SQL with
//!   `IF NOT EXISTS` added, which the idempotent-`ensure_` pattern requires.
//! - `genesis_frontier`'s key is D7's ordering triple; §9 names the ordering
//!   (`next_scan_at, first_seen_at, group_id`) and the key (space, group,
//!   epoch) but not an index, so the covering index here is derived from the
//!   stated ordering rather than quoted.
//!
//! Deliberately NOT created here: the suppression and quarantine homes for
//! machine F's `suppressed`/`quarantined` states. The artifacts contradict
//! themselves on whether those are keyed rows or repeatable entries, and a
//! table shape guessed under a contradiction is worse than a later additive
//! migration. Same for every `m6_*` table of the relevance, overview, and
//! refresh contracts. See the rung-2 open-question list in the PR-A report.

use super::MemoryDB;
use crate::WenlanError;

impl MemoryDB {
    /// Migration 108 DDL. Additive and idempotent.
    pub(super) async fn ensure_genesis_tables(tx: &libsql::Transaction) -> Result<(), WenlanError> {
        tx.execute_batch(
            "
            -- Machine A (§4). One attempt at one slot. The row is STABLE
            -- across retries (I-8): `attempt`, `state`, and `next_attempt_at`
            -- move, identity does not.
            CREATE TABLE IF NOT EXISTS genesis_candidates (
                -- m6_digest('m6-candidate-v1', [slot_id, coverage_epoch]) —
                -- S0-40. Derived, so it is the natural key rather than a
                -- surrogate: a retry that minted a second row would be I-8's
                -- failure made storable.
                candidate_id          TEXT    PRIMARY KEY,
                slot_id               TEXT    NOT NULL,
                page_id               TEXT    NOT NULL,
                space                 TEXT    NOT NULL,
                -- The four D5 signals, spelled exactly as `m6::constants`
                -- spells them. A fifth signal is a contract change, not a
                -- value change, so the constraint is here rather than in a
                -- Rust match a caller can forget to run.
                signal_kind           TEXT    NOT NULL CHECK (signal_kind IN (
                                          'evidence-cluster', 'orphan-wikilink',
                                          'community-overview', 'space-overview')),
                coverage_epoch        INTEGER NOT NULL,
                input_generation      INTEGER NOT NULL,
                active_root_digest    TEXT    NOT NULL,
                -- D3 fixes the state set at ten, and S0-12 says exhausted
                -- retry is NOT an eleventh state — it is `stale` carrying a
                -- reason. The CHECK is the mechanical form of that sentence.
                state                 TEXT    NOT NULL CHECK (state IN (
                                          'observed', 'prepared', 'inferencing',
                                          'validating', 'published', 'retry_wait',
                                          'stale', 'review_required', 'suppressed',
                                          'superseded')),
                -- NULL until a transition carries one. Stored as NULL rather
                -- than '' so `reason IS NULL` distinguishes 'no reason yet'
                -- from a reason that failed to write.
                reason                TEXT,
                attempt               INTEGER NOT NULL DEFAULT 0,
                next_attempt_at       INTEGER,
                lease_token           TEXT,
                receipt_id            TEXT,
                -- The candidate's OUTPUT. D7 compaction at 90 days nulls this
                -- and stamps `payload_compacted_at` (S0-10); the row itself is
                -- never deleted.
                payload               TEXT,
                payload_compacted_at  INTEGER,
                created_at            INTEGER NOT NULL,
                updated_at            INTEGER NOT NULL
            );
            -- The recovery scan (§11 step 2) reads in-flight candidates by
            -- state; the frontier reads them by slot.
            CREATE INDEX IF NOT EXISTS idx_genesis_candidates_space_state
                ON genesis_candidates(space, state);
            CREATE INDEX IF NOT EXISTS idx_genesis_candidates_slot
                ON genesis_candidates(slot_id, coverage_epoch);

            -- Machine B (§5). One (candidate, root) pair.
            CREATE TABLE IF NOT EXISTS genesis_candidate_roots (
                candidate_id           TEXT    NOT NULL
                                           REFERENCES genesis_candidates(candidate_id),
                root_id                TEXT    NOT NULL,
                independence_group_id  TEXT    NOT NULL,
                coverage_epoch         INTEGER NOT NULL,
                -- I-3: a witness row must never be readable as coverage. Both
                -- partial indexes below exclude witnesses by predicate, so the
                -- only way a third role could break that is by existing.
                claim_role             TEXT    NOT NULL
                                           CHECK (claim_role IN ('concept', 'witness')),
                -- S0-6: 'active' is a STORED bit, not a join to the lease,
                -- because a SQLite partial index may only reference columns of
                -- its own table. The recovery scan is the sole reconciler.
                released_at            INTEGER,
                released_reason        TEXT,
                consumed               INTEGER NOT NULL DEFAULT 0,
                retained               INTEGER NOT NULL DEFAULT 0,
                created_at             INTEGER NOT NULL,
                PRIMARY KEY(candidate_id, root_id)
            );
            -- Verbatim from §5, plus IF NOT EXISTS. These two indexes are the
            -- ENTIRE exclusion mechanism (I-2); a code-level check would be a
            -- second, weaker one.
            CREATE UNIQUE INDEX IF NOT EXISTS idx_genesis_root_claim
                ON genesis_candidate_roots(root_id, coverage_epoch)
                WHERE claim_role = 'concept' AND released_at IS NULL;
            CREATE UNIQUE INDEX IF NOT EXISTS idx_genesis_group_claim
                ON genesis_candidate_roots(independence_group_id, coverage_epoch)
                WHERE claim_role = 'concept' AND released_at IS NULL;

            -- Machine D (§7). One space's identity-contract era. The column
            -- list is quoted from §7; the defaults are D1's insert values, so
            -- a row created by naming only its space is already correct.
            CREATE TABLE IF NOT EXISTS genesis_coverage_state (
                space             TEXT    PRIMARY KEY,
                coverage_epoch    INTEGER NOT NULL DEFAULT 1,
                epoch_state       TEXT    NOT NULL DEFAULT 'active'
                                      CHECK (epoch_state IN ('active', 'migrating')),
                opened_at         INTEGER NOT NULL,
                -- The resumable position of D3's forward mapping. TEXT because
                -- what it resumes over is group identity, which is text.
                migration_cursor  TEXT,
                -- D1 inserts 0, and a space with NO ROW is genesis-disabled —
                -- so an empty table means nothing runs.
                genesis_enabled   INTEGER NOT NULL DEFAULT 0
            );

            -- Machine F (§9), permanent half. I-7: no transition deletes a
            -- coverage row, and the epoch never decreases. Keyed by epoch so
            -- D3's forward mapping writes new rows rather than rewriting these.
            CREATE TABLE IF NOT EXISTS genesis_group_coverage (
                space                  TEXT    NOT NULL,
                independence_group_id  TEXT    NOT NULL,
                coverage_epoch         INTEGER NOT NULL,
                page_id                TEXT    NOT NULL,
                candidate_id           TEXT    NOT NULL,
                covered_at             INTEGER NOT NULL,
                PRIMARY KEY(space, independence_group_id, coverage_epoch)
            );

            -- Machine F (§9), re-derivable half. F1 says frontier rows are
            -- re-derivable from the differential query, which is why they need
            -- atomicity with nothing.
            CREATE TABLE IF NOT EXISTS genesis_frontier (
                space                  TEXT    NOT NULL,
                independence_group_id  TEXT    NOT NULL,
                coverage_epoch         INTEGER NOT NULL,
                first_seen_at          INTEGER NOT NULL,
                next_scan_at           INTEGER NOT NULL,
                PRIMARY KEY(space, independence_group_id, coverage_epoch)
            );
            -- D7's ordering, made an index so the scan does not sort.
            CREATE INDEX IF NOT EXISTS idx_genesis_frontier_scan
                ON genesis_frontier(space, next_scan_at, first_seen_at,
                                    independence_group_id);
            ",
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m108 genesis tables: {error}")))?;
        Ok(())
    }
}
