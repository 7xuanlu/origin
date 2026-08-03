// SPDX-License-Identifier: Apache-2.0
//! The startup recovery scan (artifact 2 §11; spec §4.4, §4.5).
//!
//! One eager pass at daemon start, before the first refinery turn (S0-5), in
//! three ordered steps. Eager rather than lazy for a reason worth stating: M6
//! reads *reservations* to decide whether a group has a durable reason, so a
//! reservation orphaned by a killed process hides its group from the frontier
//! for as long as it survives. M4's lazy reap-at-acquire would only clear that
//! when someone next tried to work the same space — which, for a space whose
//! only pending work is the hidden group, is never.
//!
//! This module takes a `&libsql::Connection`, not a transaction, because
//! artifact 2 §11 fixes the granularity at **one transaction per candidate**:
//! a scan that recovered a thousand candidates in one transaction would make
//! partial progress impossible, so a crash during recovery would restart it
//! from zero every time.
//!
//! PR-B2 wires no caller. `wenlan-server` starting this scan is PR-B3's.

use super::candidates::{self, CandidateState, StaleReason};
use super::leases;
use crate::WenlanError;

/// What one recovery pass did. Every field is a count a startup log line or a
/// restart-resumability test can assert on.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RecoveryReport {
    /// Step 1 — expired lease rows deleted across all phases (C8).
    pub leases_reaped: u64,
    /// Step 2 — candidates moved to `stale` with `reason = 'lease_lost'` (A7).
    pub candidates_staled: Vec<String>,
    /// Step 2 — candidates left alone because their lease is still live. The
    /// honest weakness of the design (artifact 2 §11's last row): a killed
    /// daemon's unexpired lease blocks its own space for the remainder of the
    /// TTL after restart. Counted rather than engineered around, because the
    /// daemon has no durable process identity that could tell "my previous
    /// life" from "a concurrent instance".
    pub candidates_lease_live: usize,
    /// Step 3 — `page_projection_outbox` rows in `handed_off`. **Observed, not
    /// requeued.** See [`scan`].
    pub projections_handed_off: u64,
}

/// Run the three-step scan.
///
/// **Step 3 is a probe, not a requeue, and that is a deliberate scope call.**
/// Artifact 2 §11 step 3 requeues `page_projection_outbox` rows from
/// `handed_off` to `pending` (E8). `page_projection_outbox` matches the
/// `page_*` prefix this slice is forbidden from writing to, and the projection
/// hand-off it drives has no other half in PR-B2 — machine E lands in PR-B3.
/// Counting the rows proves the table is visible to recovery and lets a test
/// assert the shadow leaves none behind; the `UPDATE ... SET state = 'pending'`
/// belongs with the writer that can also complete it.
pub async fn scan(conn: &libsql::Connection) -> Result<RecoveryReport, WenlanError> {
    let mut report = RecoveryReport::default();

    // Step 1 — reap every expired lease, all phases, in its own transaction so
    // step 2 reads a settled registry rather than one it is still changing.
    let tx = begin(conn).await?;
    report.leases_reaped = leases::reap_expired_all_phases(&tx).await?;
    commit(tx).await?;

    // Step 2 — one transaction per candidate.
    for candidate_id in orphaned_candidates(conn).await? {
        let tx = begin(conn).await?;
        let Some(candidate) = candidates::read_candidate(&tx, &candidate_id).await? else {
            // Recovery is the only writer at this point, so a vanished row is
            // not a race; it is a candidate whose state moved between the two
            // reads and no longer needs recovering.
            commit(tx).await?;
            continue;
        };
        if !RECOVERABLE_STATES.contains(&candidate.state) {
            commit(tx).await?;
            continue;
        }
        if lease_is_live(&tx, &candidate).await? {
            report.candidates_lease_live += 1;
            commit(tx).await?;
            continue;
        }
        // One transaction: release the reservation (B6), return each group to
        // `waiting_frontier` (F5), and move the candidate to `stale` (A7).
        // Splitting them would leave a window where the group has zero durable
        // reasons, which is precisely what I-1 forbids and what a recovery scan
        // must not itself create.
        candidates::mark_stale(&tx, &candidate_id, StaleReason::LeaseLost).await?;
        commit(tx).await?;
        report.candidates_staled.push(candidate_id);
    }

    // Step 3 — probe only; see the doc comment.
    let tx = begin(conn).await?;
    report.projections_handed_off = handed_off_projections(&tx).await?;
    commit(tx).await?;

    Ok(report)
}

/// The four states step 2 recovers from. `retry_wait` is included on purpose
/// and is stricter than the backoff alone: a crash converts a pending retry
/// into a `stale` exit and returns the group to the frontier, from where the
/// *same* candidate row re-prepares via A19 once its S0-151 delay elapses.
const RECOVERABLE_STATES: [CandidateState; 4] = [
    CandidateState::Prepared,
    CandidateState::Inferencing,
    CandidateState::Validating,
    CandidateState::RetryWait,
];

/// Candidates holding an unreleased reservation while in a recoverable state.
///
/// Read in its own transaction and collected before any write: holding a cursor
/// open across the per-candidate transactions would make the scan's granularity
/// a fiction.
async fn orphaned_candidates(conn: &libsql::Connection) -> Result<Vec<String>, WenlanError> {
    let tx = begin(conn).await?;
    let mut rows = tx
        .query(
            "SELECT DISTINCT c.candidate_id
               FROM genesis_candidates c
               JOIN genesis_candidate_roots gr ON gr.candidate_id = c.candidate_id
              WHERE gr.released_at IS NULL
                AND c.state IN ('prepared', 'inferencing', 'validating', 'retry_wait')
              ORDER BY c.candidate_id",
            (),
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 recovery scan: {error}")))?;
    let mut candidate_ids = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 recovery scan row: {error}")))?
    {
        candidate_ids.push(
            row.get::<String>(0)
                .map_err(|error| WenlanError::VectorDb(format!("m6 recovery decode: {error}")))?,
        );
    }
    drop(rows);
    commit(tx).await?;
    Ok(candidate_ids)
}

/// Whether the candidate's stored token still names a live genesis lease.
///
/// Token-scoped, not space-scoped: a candidate whose lease expired and whose
/// space was then re-leased by another operation must still recover, and a
/// space-scoped check would read that new lease as its own.
async fn lease_is_live(
    tx: &libsql::Transaction,
    candidate: &candidates::CandidateRow,
) -> Result<bool, WenlanError> {
    let Some(token) = candidate.lease_token.as_deref() else {
        return Ok(false);
    };
    leases::owns(
        tx,
        leases::LeasePhase::Genesis,
        &candidate.space,
        candidate.input_generation,
        token,
    )
    .await
}

async fn handed_off_projections(tx: &libsql::Transaction) -> Result<u64, WenlanError> {
    let mut rows = tx
        .query(
            "SELECT COUNT(*) FROM page_projection_outbox WHERE state = 'handed_off'",
            (),
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 recovery outbox probe: {error}")))?;
    Ok(rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 recovery outbox row: {error}")))?
        .map(|row| row.get::<i64>(0))
        .transpose()
        .map_err(|error| WenlanError::VectorDb(format!("m6 recovery outbox decode: {error}")))?
        .unwrap_or_default()
        .max(0) as u64)
}

async fn begin(conn: &libsql::Connection) -> Result<libsql::Transaction, WenlanError> {
    conn.transaction()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 recovery begin: {error}")))
}

async fn commit(tx: libsql::Transaction) -> Result<(), WenlanError> {
    tx.commit()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 recovery commit: {error}")))
}
