// SPDX-License-Identifier: Apache-2.0
//! Shadow evidence — per-space readiness (§7.1) and shadow statistics (§7.2).
//!
//! Two jobs, both write-only-into-M6's-own-tables:
//!
//! 1. **Readiness.** PR-B's target is stage `B_genesis_shadow`, whose artifact-11
//!    entry condition is literally "PR-B deployed, jobs dry-run". This module
//!    drives at most `off → preparing` and **never** reaches `committed`,
//!    because committed is a cutover fact and PR-B cuts nothing over (§7.1).
//! 2. **Statistics.** The numbers §7.2 names, recorded as durable monotone
//!    counters in `m6_counters` rather than as log lines, plus a live read of
//!    the state a turn can describe.
//!
//! **The zero-mutation proof lives here, and it is a proof rather than an
//! assertion.** `genesis_coverage_state.m6_mutation_count` is guarded by four
//! migration-109 triggers: it cannot decrease (`m6_genesis_mutation_count_monotone`),
//! it cannot be reset by an insert (`..._replace_monotone`), its space identity
//! cannot be re-pointed (`..._identity_insert` / `..._identity_update`), and its
//! row cannot be deleted (`..._no_delete`). A reader that observes the same
//! value before and after a shadow soak has therefore observed that **no write
//! path incremented it at any point in between** — not that none was observed
//! doing so. That is the difference between a monotone counter and a sampled
//! gauge, and it is why the counter, not the sample, is the evidence.
//!
//! Nothing in this module writes `genesis_enabled`. PR-B creates coverage rows
//! and leaves the flag at its `DEFAULT 0` (Q11 adjudication); a shadow that
//! could enable itself would not be a shadow.

use std::collections::BTreeMap;

use super::candidates::{CandidateState, CANDIDATE_STATES};
use super::finalize::CasGate;
use super::refresh_readiness::{
    initialize_readiness, readiness_fence, transition_is_legal, transition_readiness, ReadinessKey,
    ReadinessPhase,
};
use crate::WenlanError;

/// Artifact 11's stage for the PR-B shadow. The `m6_readiness` CHECK constraint
/// pairs stages `A`–`D` with the `-` signal, so the two constants travel
/// together and neither is independently choosable.
pub const READINESS_STAGE: &str = "B";
/// The signal component for a non-`E` stage — see [`READINESS_STAGE`].
pub const READINESS_SIGNAL: &str = "-";

// ---------------------------------------------------------------------------
// Counters (§7.2)
// ---------------------------------------------------------------------------

/// Shadow turns that reached this space with work to do.
pub const COUNTER_TURNS: &str = "shadow_turns";
/// Frontier rows written by F1 repair — the count *is* the signal (a healthy
/// space repairs nothing).
pub const COUNTER_FRONTIER_REPAIRS: &str = "frontier_repairs";
/// Candidates that reached `prepared`.
pub const COUNTER_PREPARED: &str = "candidates_prepared";
/// Dry runs where all eight gates held.
pub const COUNTER_DRY_RUN_PASSED: &str = "dry_run_passed";
/// Oracle divergences observed by the sampled full-vs-incremental check (§6.2).
/// Expected to stay at zero forever; a non-zero value is never auto-repaired,
/// because auto-repair would hide the bug the oracle exists to find.
pub const COUNTER_ORACLE_DIVERGENCE: &str = "oracle_divergence";

/// The counter name for a refusal at `gate` — §7.2's "dry-run outcomes by
/// first-failing CAS gate". One counter per gate rather than one counter with a
/// gate column, because `m6_counters` is keyed `(space_id, name)` and its
/// monotone trigger is per row: separate names give each gate its own
/// independently monotone series.
pub fn refusal_counter(gate: CasGate) -> String {
    format!("dry_run_refused_{}", gate.as_str())
}

/// Add `by` to a space's counter, creating it at zero on first use.
///
/// Read-then-insert-then-update rather than an UPSERT, for a mechanical reason:
/// migration 109's `m6_counters_replace_monotone` is a `BEFORE INSERT` trigger
/// that raises `ABORT` when the incoming value is below the stored one, and
/// SQLite evaluates `BEFORE INSERT` triggers before upsert conflict resolution.
/// An `ON CONFLICT DO UPDATE SET value = value + ?` would therefore abort as
/// soon as the stored value exceeded the delta. `INSERT OR IGNORE` fares no
/// better — `OR IGNORE` does not swallow a trigger's `RAISE(ABORT)`, the same
/// trap `candidates::open_coverage_epoch` documents.
pub async fn bump(
    tx: &libsql::Transaction,
    space_id: &str,
    space: &str,
    name: &str,
    by: i64,
) -> Result<(), WenlanError> {
    if by <= 0 {
        return Ok(());
    }
    if read_counter(tx, space_id, name).await?.is_none() {
        tx.execute(
            "INSERT INTO m6_counters (space_id, space, name, value) VALUES (?1, ?2, ?3, 0)",
            libsql::params![space_id, space, name],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 counter create: {error}")))?;
    }
    tx.execute(
        "UPDATE m6_counters SET value = value + ?3 WHERE space_id = ?1 AND name = ?2",
        libsql::params![space_id, name, by],
    )
    .await
    .map_err(|error| WenlanError::VectorDb(format!("m6 counter bump: {error}")))?;
    Ok(())
}

/// One counter's value, or `None` when it has never been touched.
pub async fn read_counter(
    tx: &libsql::Transaction,
    space_id: &str,
    name: &str,
) -> Result<Option<i64>, WenlanError> {
    let mut rows = tx
        .query(
            "SELECT value FROM m6_counters WHERE space_id = ?1 AND name = ?2",
            libsql::params![space_id, name],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 counter read: {error}")))?;
    rows.next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 counter read row: {error}")))?
        .map(|row| row.get::<i64>(0))
        .transpose()
        .map_err(|error| WenlanError::VectorDb(format!("m6 counter decode: {error}")))
}

// ---------------------------------------------------------------------------
// Readiness (§7.1)
// ---------------------------------------------------------------------------

/// Bring a space's `B_genesis_shadow` readiness row into being and move it to
/// `preparing` once, idempotently.
///
/// Every one of §7.1's four wiring points is exercised in order:
/// `initialize_readiness` on first observation, `readiness_fence` before any
/// transition, [`transition_is_legal`] as an explicit pre-check, and
/// `transition_readiness` for the CAS'd move (which increments the epoch).
///
/// The legality pre-check is not redundant with `transition_readiness`'s own
/// internal check. `transition_readiness` returns `Err` on an illegal edge;
/// here an already-`preparing` or already-`committed` space is a **normal
/// steady state**, not a fault, so asking first is what keeps a healthy second
/// turn from raising. PR-B never passes anything but `Preparing` as the target,
/// so `committed` is unreachable from this lane by construction rather than by
/// a guard someone could relax.
pub async fn note_shadow_readiness(
    tx: &libsql::Transaction,
    space: &str,
) -> Result<ReadinessPhase, WenlanError> {
    let key = ReadinessKey {
        space,
        stage: READINESS_STAGE,
        signal: READINESS_SIGNAL,
    };
    let now = sqlite_now(tx).await?;
    initialize_readiness(tx, key, now)
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 readiness init: {error}")))?;
    let fence = readiness_fence(tx, key)
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 readiness fence: {error}")))?
        .ok_or_else(|| {
            WenlanError::VectorDb(format!("m6 readiness row vanished for space {space:?}"))
        })?;
    if !transition_is_legal(fence.phase, ReadinessPhase::Preparing) {
        return Ok(fence.phase);
    }
    let moved = transition_readiness(tx, key, fence, ReadinessPhase::Preparing, now)
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 readiness transition: {error}")))?;
    // A lost CAS means another worker moved the same row in this instant. The
    // observed phase is then whatever it moved to, and reporting the pre-move
    // phase would be a lie about durable state, so re-read rather than assume.
    match moved {
        Some(fence) => Ok(fence.phase),
        None => Ok(readiness_fence(tx, key)
            .await
            .map_err(|error| WenlanError::VectorDb(format!("m6 readiness re-read: {error}")))?
            .map(|fence| fence.phase)
            .unwrap_or(ReadinessPhase::Off)),
    }
}

// ---------------------------------------------------------------------------
// Per-space statistics (§7.2)
// ---------------------------------------------------------------------------

/// Everything §7.2 asks a turn to record about one space.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ShadowStats {
    /// Live `genesis_frontier` rows for the epoch.
    pub frontier_size: i64,
    /// Groups with a durable coverage row — the admitted set.
    pub admitted: i64,
    /// Frontier rows whose `next_scan_at` is still in the future.
    pub deferred: i64,
    /// Unlapsed `genesis_suppression` rows.
    pub suppressed: i64,
    /// Unlifted `genesis_quarantine` rows.
    pub quarantined: i64,
    /// Live candidate counts, one entry per state that has any.
    pub candidates_by_state: BTreeMap<CandidateState, i64>,
    /// Cumulative refusals per first-failing gate, from the durable counters.
    pub refusals_by_gate: BTreeMap<CasGate, i64>,
    /// Cumulative passing dry runs.
    pub dry_runs_passed: i64,
    /// Cumulative oracle divergences. Zero, forever.
    pub oracle_divergences: i64,
    /// The observed `m6_mutation_count`. **Zero, forever** — see the module
    /// doc for why an observation of this particular value is a proof.
    pub m6_mutation_count: i64,
}

/// Read one space's shadow statistics. Pure reads; records nothing.
pub async fn observe_space(
    tx: &libsql::Transaction,
    space_id: &str,
    space: &str,
    coverage_epoch: i64,
) -> Result<ShadowStats, WenlanError> {
    let mut stats = ShadowStats {
        frontier_size: count(
            tx,
            "SELECT COUNT(*) FROM genesis_frontier WHERE space = ?1 AND coverage_epoch = ?2",
            libsql::params![space, coverage_epoch],
        )
        .await?,
        admitted: count(
            tx,
            "SELECT COUNT(*) FROM genesis_group_coverage WHERE space = ?1 AND coverage_epoch = ?2",
            libsql::params![space, coverage_epoch],
        )
        .await?,
        deferred: count(
            tx,
            "SELECT COUNT(*) FROM genesis_frontier
              WHERE space = ?1 AND coverage_epoch = ?2 AND next_scan_at > unixepoch()",
            libsql::params![space, coverage_epoch],
        )
        .await?,
        suppressed: count(
            tx,
            "SELECT COUNT(*) FROM genesis_suppression
              WHERE space = ?1 AND coverage_epoch = ?2 AND lapsed_at IS NULL",
            libsql::params![space, coverage_epoch],
        )
        .await?,
        quarantined: count(
            tx,
            "SELECT COUNT(*) FROM genesis_quarantine
              WHERE space = ?1 AND coverage_epoch = ?2 AND lifted_at IS NULL",
            libsql::params![space, coverage_epoch],
        )
        .await?,
        m6_mutation_count: count(
            tx,
            "SELECT COALESCE(MAX(m6_mutation_count), 0) FROM genesis_coverage_state
              WHERE space_id = ?1",
            libsql::params![space_id],
        )
        .await?,
        ..ShadowStats::default()
    };

    for state in CANDIDATE_STATES {
        let live = count(
            tx,
            "SELECT COUNT(*) FROM genesis_candidates WHERE space = ?1 AND state = ?2",
            libsql::params![space, state.as_str()],
        )
        .await?;
        if live > 0 {
            stats.candidates_by_state.insert(state, live);
        }
    }
    for gate in CasGate::ALL {
        if let Some(value) = read_counter(tx, space_id, &refusal_counter(gate)).await? {
            if value > 0 {
                stats.refusals_by_gate.insert(gate, value);
            }
        }
    }
    stats.dry_runs_passed = read_counter(tx, space_id, COUNTER_DRY_RUN_PASSED)
        .await?
        .unwrap_or_default();
    stats.oracle_divergences = read_counter(tx, space_id, COUNTER_ORACLE_DIVERGENCE)
        .await?
        .unwrap_or_default();
    Ok(stats)
}

/// SQLite's clock, read in-statement (S0-11).
///
/// `refresh_readiness` takes `now` as a parameter — a PR-A signature this slice
/// does not get to change — so the value it is handed comes from the database
/// rather than from Rust. The clock stays single-sourced either way.
pub async fn sqlite_now(tx: &libsql::Transaction) -> Result<i64, WenlanError> {
    count(tx, "SELECT unixepoch()", ()).await
}

async fn count(
    tx: &libsql::Transaction,
    sql: &str,
    params: impl libsql::params::IntoParams,
) -> Result<i64, WenlanError> {
    let mut rows = tx
        .query(sql, params)
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 evidence count: {error}")))?;
    Ok(rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 evidence count row: {error}")))?
        .map(|row| row.get::<i64>(0))
        .transpose()
        .map_err(|error| WenlanError::VectorDb(format!("m6 evidence count decode: {error}")))?
        .unwrap_or_default())
}
