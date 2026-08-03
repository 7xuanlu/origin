// SPDX-License-Identifier: Apache-2.0
//! The shadow turn driver (spec §4.1–§4.5).
//!
//! `wenlan-server` owns the loop; this owns **one turn of it**. The split is
//! the same one the M5 promoter uses: the daemon decides when to tick and when
//! to stop, and `wenlan-core` decides what a tick means. It also keeps every
//! statement in this file inside `wenlan-core`, where the crate-boundary rule
//! puts business logic.
//!
//! **What a turn may do (§4.2): at most one unit of work, in priority order.**
//!
//! | Priority | Step | Bound |
//! |---|---|---|
//! | 1 | startup recovery scan | once per process, one transaction per candidate |
//! | 2 | frontier reconciliation | one space, cursor-resumed, ≤ 512 rows |
//! | 3 | candidate prepare | one space, ≤ 16 considered, ≤ 64 roots each |
//! | 4 | dry-run finalization | exactly one candidate |
//!
//! Steps 2–4 are tried against **one space per turn**, chosen round-robin, so a
//! turn's cost is bounded by one space's differential query rather than by the
//! store's space count. A turn that finds no work in its space advances the
//! rotation and reports [`GenesisTurn::Idle`].
//!
//! **Nothing runs while `genesis_enabled = 0`, and nothing here can change
//! that.** Not one statement in this module writes `genesis_enabled`; the flag
//! stays at migration 108's `DEFAULT 0` for every space (Q11). What the shadow
//! does under that flag is exactly what it does under an absent row — build
//! `genesis_*` state and refuse to publish — because the flag gates
//! *publication*, which PR-B does not implement at all. There is no code path
//! from a turn to `pages`, `page_*`, `entities`, `relations`, `observations`,
//! `edges`, `memories`, or `chunks`, and none to `page_projection_outbox`.
//! `shadow_test` proves it by counting, not by asserting.
//!
//! **Strict page suppression is not reintroduced here, directly or
//! indirectly.** No turn writes `genesis_suppression`, and no verdict this
//! module produces reaches page visibility. One unsupported claim still hides
//! nothing.

use super::candidates::{
    self, CandidateState, ClaimReservation, EpochState, ObserveOutcome, ObservedCandidate,
    PrepareOutcome, StaleReason,
};
use super::community_gate;
use super::constants::CANDIDATE_PREPARES_PER_CYCLE_CAP;
use super::evidence;
use super::finalize::{self, CasEnvelope, CasGate, Finalization};
use super::frontier;
use super::identity;
use super::leases::{self, LeasePhase};
use super::oracle;
use super::recovery::{self, RecoveryReport};
use super::signals::{self, CandidateProposal};
use crate::WenlanError;

/// §4.1's turn report, verbatim. `RefusedBudget` is distinct from `Requeued`
/// for a contract reason: **S0-9 — a budget refusal does not increment
/// `attempt`.** Folding it in would burn retries on backpressure and eventually
/// park healthy work.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenesisTurn {
    Idle,
    Frontier {
        space: String,
        admitted: usize,
        deferred: usize,
    },
    Prepared {
        candidate_id: String,
    },
    DryRunPassed {
        candidate_id: String,
    },
    DryRunRefused {
        candidate_id: String,
        gate: CasGate,
    },
    Requeued {
        candidate_id: String,
        reason: String,
    },
    Parked {
        candidate_id: String,
        reason: String,
    },
    RefusedBudget,
}

impl GenesisTurn {
    /// Did this turn do durable work? The loop's adaptive delay reads this:
    /// 100ms when true, 1s when false (§4.1 point 5).
    pub fn did_work(&self) -> bool {
        !matches!(self, GenesisTurn::Idle)
    }
}

/// Turn-to-turn state the driver owns in memory.
///
/// Deliberately tiny and deliberately **not** a work queue (§4.5): everything
/// that matters is durable before a turn returns — the frontier cursor is an
/// `app_metadata` key, candidate state is a row, reservations are rows, and
/// leases expire on their own. A `kill -9` between two turns costs one turn's
/// work and one lease TTL of latency, and losing this struct costs a rotation
/// position and nothing else.
#[derive(Debug, Default)]
pub struct ShadowState {
    /// Set once the S0-5 startup scan has run, with its report kept so the
    /// daemon can log it. `None` means the scan is still owed.
    recovery: Option<RecoveryReport>,
    /// Round-robin position across spaces. An index rather than a name so a
    /// deleted space cannot wedge the rotation on a row that no longer exists.
    rotation: usize,
    /// Turns since process start, for the §6.2 oracle sampling cadence.
    turns: u64,
}

impl ShadowState {
    /// The startup scan's report, once it has run.
    pub fn recovery_report(&self) -> Option<&RecoveryReport> {
        self.recovery.as_ref()
    }
}

/// §6.2 — how many turns between full-vs-incremental oracle samples.
///
/// Large enough to be negligible (the oracle recomputes a space from primary
/// evidence, which is the expensive read in the whole subsystem) and small
/// enough that a divergence surfaces within an hour of ordinary ticking. The
/// sample is read-only and **never auto-repairs**: auto-repair would hide the
/// bug the oracle exists to find.
const ORACLE_SAMPLE_TURNS: u64 = 512;

/// Run one shadow turn.
///
/// Takes a `&libsql::Connection` rather than a transaction because §4.4 fixes
/// the recovery scan's granularity at one transaction per candidate, and
/// because a turn that spans several independent units must be able to commit
/// each one. Every transaction opened below is short and none of them spans a
/// long-running call — there is no long-running call to span (§5.1).
pub async fn run_genesis_shadow_turn(
    conn: &libsql::Connection,
    state: &mut ShadowState,
) -> Result<GenesisTurn, WenlanError> {
    state.turns = state.turns.saturating_add(1);

    // Priority 1 — the eager startup scan (S0-5), once per process, before the
    // first normal turn. Reported as `Idle` because §4.1's enum has no recovery
    // variant and inventing one would be a spec change made in an
    // implementation; the report itself is on `ShadowState` for the caller.
    if state.recovery.is_none() {
        state.recovery = Some(recovery::scan(conn).await?);
        return Ok(GenesisTurn::Idle);
    }

    let spaces = spaces(conn).await?;
    if spaces.is_empty() {
        return Ok(GenesisTurn::Idle);
    }
    let (space_id, space) = spaces[state.rotation % spaces.len()].clone();

    // D1 — the coverage epoch the rest of the turn keys on, plus this space's
    // stage-B readiness. Both are idempotent and neither writes `genesis_enabled`.
    let tx = begin(conn).await?;
    let coverage = candidates::open_coverage_epoch(&tx, &space_id, &space).await?;
    evidence::note_shadow_readiness(&tx, &space).await?;
    commit(tx).await?;

    if state.turns.is_multiple_of(ORACLE_SAMPLE_TURNS) {
        sample_oracle(conn, &space_id, &space, coverage.coverage_epoch).await?;
    }

    // Priority 2 — frontier reconciliation for this space.
    if let Some(turn) = reconcile(conn, &space_id, &space, coverage.coverage_epoch).await? {
        return Ok(turn);
    }
    // Priority 3 — one candidate prepare for this space.
    if let Some(turn) = prepare(conn, &space_id, &space, coverage.coverage_epoch).await? {
        return Ok(turn);
    }
    // Priority 4 — one dry-run finalization.
    if let Some(turn) = finalize_one(conn, &space_id, &space).await? {
        return Ok(turn);
    }

    state.rotation = state.rotation.wrapping_add(1);
    Ok(GenesisTurn::Idle)
}

// ---------------------------------------------------------------------------
// Priority 2 — frontier reconciliation
// ---------------------------------------------------------------------------

/// One space's differential reconciliation plus one cursor-resumed scan slice.
///
/// Returns `None` when the pass repaired nothing — i.e. the space is quiescent
/// and the turn should fall through to a lower priority. Returning `Some` for a
/// no-op pass would make the loop spin at the did-work delay forever on an idle
/// store.
async fn reconcile(
    conn: &libsql::Connection,
    space_id: &str,
    space: &str,
    coverage_epoch: i64,
) -> Result<Option<GenesisTurn>, WenlanError> {
    let tx = begin(conn).await?;
    let input_generation = finalize::grouping_generation(&tx, space).await?;
    let Some(token) =
        leases::acquire(&tx, LeasePhase::Frontier, space, input_generation, 0).await?
    else {
        // C2 — another holder owns the scan. Not work, not an error.
        commit(tx).await?;
        return Ok(None);
    };

    let outcome = frontier::reconcile_space(&tx, space, coverage_epoch).await?;
    let cursor = frontier::read_cursor(&tx, space).await?;
    let slice = frontier::scan_slice(&tx, space, coverage_epoch, cursor.as_ref()).await?;
    frontier::write_cursor(&tx, space, slice.next_cursor.as_ref()).await?;

    let repaired = outcome.repaired.len();
    if repaired > 0 {
        evidence::bump(
            &tx,
            space_id,
            space,
            evidence::COUNTER_FRONTIER_REPAIRS,
            repaired as i64,
        )
        .await?;
    }
    // **Only a repair counts as work; advancing the scan cursor does not.**
    // The §7.3 benchmark is what settled this. With `cursor.is_some()` in the
    // disjunction, any space holding more than `FRONTIER_SCAN_SLICE_ROWS` (512)
    // frontier rows pinned the loop at the 100ms did-work cadence forever: the
    // sweep wraps, the cursor clears, the next visit starts a fresh sweep, and
    // three of every four turns report work again. Measured at 2000 groups, the
    // loop was still sweeping after 134s and 170 counted turns, each ~700ms
    // because `reconcile_space` re-runs its whole differential every turn.
    // A slice that repaired nothing found nothing; it is bookkeeping, and the
    // sweep still advances one slice per idle-cadence tick.
    let did_work = repaired > 0 || outcome.lapsed_suppressions > 0;
    if did_work {
        evidence::bump(&tx, space_id, space, evidence::COUNTER_TURNS, 1).await?;
    }
    leases::release(&tx, LeasePhase::Frontier, space, input_generation, &token).await?;
    commit(tx).await?;

    if !did_work {
        return Ok(None);
    }
    let stats = {
        let tx = begin(conn).await?;
        let stats = evidence::observe_space(&tx, space_id, space, coverage_epoch).await?;
        commit(tx).await?;
        stats
    };
    Ok(Some(GenesisTurn::Frontier {
        space: space.to_string(),
        admitted: stats.admitted.max(0) as usize,
        deferred: stats.deferred.max(0) as usize,
    }))
}

// ---------------------------------------------------------------------------
// Priority 3 — candidate prepare
// ---------------------------------------------------------------------------

/// Observe this space's admitted signals and prepare **one** candidate.
///
/// **§4.2's "≤ 16 prepares" and the shipped lease semantics disagree, and this
/// takes the stricter reading.** `candidates::prepare_candidate` acquires the
/// `genesis` lease for `(phase, space, input_generation)`, and D6 makes that
/// primary key the operation identity — so a second prepare in the same
/// transaction gets `LeaseHeld` by construction. The per-turn bound is
/// therefore **one prepare**, not sixteen, which satisfies "≤ 16" and also
/// satisfies §4.2's own "one turn, one unit of work" headline. The cap still
/// bounds how many proposals are *considered*, which is the read the cap was
/// written against.
///
/// The lease is **not** released here: §4.3 scopes one `genesis` lease across
/// "candidate prepare + dry-run finalize", so the same token that authorised
/// the prepare is what E-1 verifies at finalization. Releasing it between turns
/// would make E-1 unfalsifiable — it would refuse every candidate, and a gate
/// that always refuses tests nothing.
async fn prepare(
    conn: &libsql::Connection,
    space_id: &str,
    space: &str,
    coverage_epoch: i64,
) -> Result<Option<GenesisTurn>, WenlanError> {
    let proposals = {
        let tx = begin(conn).await?;
        let proposals = admitted_proposals(&tx, space_id).await?;
        commit(tx).await?;
        proposals
    };

    for proposal in proposals.iter().take(CANDIDATE_PREPARES_PER_CYCLE_CAP) {
        let candidate_id = identity::candidate_id(&proposal.slot_id, coverage_epoch);
        // One transaction per proposal: `PrepareOutcome::ClaimRefused` obliges
        // the caller to roll back (B1, A2 is atomic), and a shared transaction
        // would drag the previous proposals' work down with it.
        let tx = begin(conn).await?;
        match prepare_one(
            &tx,
            space_id,
            space,
            coverage_epoch,
            proposal,
            &candidate_id,
        )
        .await?
        {
            Some(turn) => {
                commit(tx).await?;
                return Ok(Some(turn));
            }
            None => {
                // Nothing durable happened for this proposal. Commit rather
                // than roll back so any *other* statement the helper ran (a
                // stale-marking, a counter) survives; the helper only returns
                // `None` on paths that wrote nothing or wrote a completed unit.
                commit(tx).await?;
            }
        }
    }
    Ok(None)
}

/// One proposal's observe-and-prepare, or `None` when it had nothing to do.
async fn prepare_one(
    tx: &libsql::Transaction,
    space_id: &str,
    space: &str,
    coverage_epoch: i64,
    proposal: &CandidateProposal,
    candidate_id: &str,
) -> Result<Option<GenesisTurn>, WenlanError> {
    let roots = finalize::root_status(tx, &proposal.root_ids).await?;
    let digest = identity::active_root_digest(&roots);
    let input_generation = finalize::grouping_generation(tx, space).await?;

    match candidates::observe_candidate(
        tx,
        &ObservedCandidate {
            candidate_id,
            slot_id: &proposal.slot_id,
            page_id: &proposal.page_id,
            space,
            signal_kind: proposal.signal_kind,
            coverage_epoch,
            input_generation,
            active_root_digest: &digest,
        },
    )
    .await?
    {
        // S0-54/S0-9: the cap wrote nothing, so the group keeps its frontier
        // row and `attempt` is untouched. Reported as backpressure, never as a
        // retry.
        ObserveOutcome::RefusedPendingCap => return Ok(Some(GenesisTurn::RefusedBudget)),
        ObserveOutcome::Observed | ObserveOutcome::AlreadyPresent => {}
    }

    let Some(candidate) = candidates::read_candidate(tx, candidate_id).await? else {
        return Ok(None);
    };
    // A14 — five model attempts were already charged. Park it (still `stale`,
    // never an eleventh state, S0-12) rather than letting `prepare_candidate`
    // hand out a sixth lease.
    if candidate.state == CandidateState::RetryWait
        && candidates::retry_exhausted(candidate.attempt)
    {
        candidates::mark_stale(tx, candidate_id, StaleReason::RetryExhausted).await?;
        return Ok(Some(GenesisTurn::Parked {
            candidate_id: candidate_id.to_string(),
            reason: StaleReason::RetryExhausted.as_str().to_string(),
        }));
    }
    if !candidates::transition_is_legal(candidate.state, CandidateState::Prepared) {
        return Ok(None);
    }

    let reservations: Vec<ClaimReservation> = proposal
        .root_ids
        .iter()
        .map(|root_id| ClaimReservation {
            root_id: root_id.clone(),
            independence_group_id: String::new(),
        })
        .collect();
    let reservations = fill_groups(tx, reservations).await?;

    match candidates::prepare_candidate(tx, candidate_id, &reservations).await? {
        PrepareOutcome::Prepared { .. } => {
            // Stamp the CAS envelope in the *same* transaction as the prepare.
            // A prepare whose envelope landed separately could be finalized
            // against a generation nobody recorded — which is the whole failure
            // the stored-input rule exists to prevent.
            finalize::record_envelope(
                tx,
                candidate_id,
                &CasEnvelope {
                    receipt_id: finalize::receipt_id(candidate_id),
                    epoch_state: EpochState::Active,
                    community_generation: finalize::published_generation(tx, space).await?,
                    verdict: None,
                },
            )
            .await?;
            evidence::bump(tx, space_id, space, evidence::COUNTER_PREPARED, 1).await?;
            evidence::bump(tx, space_id, space, evidence::COUNTER_TURNS, 1).await?;
            Ok(Some(GenesisTurn::Prepared {
                candidate_id: candidate_id.to_string(),
            }))
        }
        // B1 — a partial unique index refused an exclusive claim. The caller's
        // transaction carries a partial prepare, so nothing may be reported as
        // done; the outer loop's commit is of a transaction whose only
        // statements were reads plus an insert the index rejected.
        PrepareOutcome::ClaimRefused { root_id } => Ok(Some(GenesisTurn::Requeued {
            candidate_id: candidate_id.to_string(),
            reason: format!("claim refused on root {root_id}"),
        })),
        // C2 / A13-not-due: legal no-ops, and the reason the loop falls through
        // to finalization on the very next step.
        PrepareOutcome::LeaseHeld | PrepareOutcome::NotDue => Ok(None),
    }
}

/// Fill each reservation's independence group from primary evidence.
///
/// The proposal carries root IDs only; the group is what D4's second partial
/// unique index excludes on, so it has to come from `provenance_roots` rather
/// than be inferred. A root with no row is dropped rather than reserved under a
/// blank group, because a blank group would collide with every other blank.
async fn fill_groups(
    tx: &libsql::Transaction,
    reservations: Vec<ClaimReservation>,
) -> Result<Vec<ClaimReservation>, WenlanError> {
    let mut filled = Vec::with_capacity(reservations.len());
    for reservation in reservations {
        let mut rows = tx
            .query(
                "SELECT independence_group_id FROM provenance_roots WHERE root_id = ?1",
                libsql::params![reservation.root_id.as_str()],
            )
            .await
            .map_err(|error| WenlanError::VectorDb(format!("m6 reservation group: {error}")))?;
        let group = rows
            .next()
            .await
            .map_err(|error| WenlanError::VectorDb(format!("m6 reservation group row: {error}")))?
            .map(|row| row.get::<String>(0))
            .transpose()
            .map_err(|error| {
                WenlanError::VectorDb(format!("m6 reservation group decode: {error}"))
            })?;
        if let Some(independence_group_id) = group {
            filled.push(ClaimReservation {
                independence_group_id,
                ..reservation
            });
        }
    }
    Ok(filled)
}

/// Every proposal the four D2 signals admit for a space, in slot order.
///
/// `community_partition_durable` comes from the **real** M4 gate
/// (`community_gate::partition_is_durable`), never a hardcoded bool and never a
/// copy of the gate SQL: two of the four signals read the community partition,
/// and a shadow that assumed the partition was durable would admit candidates
/// production would refuse.
async fn admitted_proposals(
    tx: &libsql::Transaction,
    space_id: &str,
) -> Result<Vec<CandidateProposal>, WenlanError> {
    let durable = community_gate::partition_is_durable(tx).await;
    let mut proposals = Vec::new();
    if let Some(proposal) = signals::space_overview(tx, space_id).await? {
        proposals.push(proposal);
    }
    proposals.extend(signals::evidence_cluster(tx, space_id, durable).await?);
    proposals.extend(signals::community_overview(tx, space_id, durable).await?);
    proposals.extend(signals::orphan_wikilink(tx, space_id).await?);
    proposals.sort_by(|a, b| a.slot_id.cmp(&b.slot_id));
    Ok(proposals)
}

// ---------------------------------------------------------------------------
// Priority 4 — dry-run finalization
// ---------------------------------------------------------------------------

/// Finalize exactly one candidate, or `None` when none is due.
///
/// "Due" is `prepared` with a stamped envelope carrying **no verdict**. That
/// predicate is what makes the shadow settle: a candidate whose dry run has
/// been recorded is never re-verified until something re-prepares it (which
/// re-stamps the envelope and clears the verdict), so a quiescent store ticks
/// idle instead of re-running the same eight gates ten times a second.
async fn finalize_one(
    conn: &libsql::Connection,
    space_id: &str,
    space: &str,
) -> Result<Option<GenesisTurn>, WenlanError> {
    let tx = begin(conn).await?;
    let due = due_candidate(&tx, space).await?;
    let Some(candidate_id) = due else {
        commit(tx).await?;
        return Ok(None);
    };
    let outcome = finalize::dry_run_finalize(&tx, &candidate_id).await?;
    let turn = match &outcome {
        Finalization::DryRunPassed { candidate_id } => {
            evidence::bump(&tx, space_id, space, evidence::COUNTER_DRY_RUN_PASSED, 1).await?;
            evidence::bump(&tx, space_id, space, evidence::COUNTER_TURNS, 1).await?;
            Some(GenesisTurn::DryRunPassed {
                candidate_id: candidate_id.clone(),
            })
        }
        Finalization::Refused { candidate_id, gate } => {
            evidence::bump(&tx, space_id, space, &evidence::refusal_counter(*gate), 1).await?;
            evidence::bump(&tx, space_id, space, evidence::COUNTER_TURNS, 1).await?;
            Some(GenesisTurn::DryRunRefused {
                candidate_id: candidate_id.clone(),
                gate: *gate,
            })
        }
        Finalization::NotEligible { .. } => None,
    };
    commit(tx).await?;
    Ok(turn)
}

/// The first `prepared` candidate in the space whose envelope has no verdict.
///
/// Filtered in Rust rather than with `json_extract` in SQL: the pending-candidate
/// cap bounds this list at 128 rows per space, and a JSON1 dependency in a
/// hot-path predicate would be a portability bet bought for nothing.
async fn due_candidate(
    tx: &libsql::Transaction,
    space: &str,
) -> Result<Option<String>, WenlanError> {
    let mut rows = tx
        .query(
            "SELECT candidate_id FROM genesis_candidates
              WHERE space = ?1 AND state = 'prepared' AND payload IS NOT NULL
              ORDER BY candidate_id",
            libsql::params![space],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 due candidates: {error}")))?;
    let mut candidate_ids = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 due candidates row: {error}")))?
    {
        candidate_ids.push(
            row.get::<String>(0)
                .map_err(|error| WenlanError::VectorDb(format!("m6 due decode: {error}")))?,
        );
    }
    drop(rows);
    for candidate_id in candidate_ids {
        if finalize::read_envelope(tx, &candidate_id)
            .await?
            .is_some_and(|envelope| envelope.verdict.is_none())
        {
            return Ok(Some(candidate_id));
        }
    }
    Ok(None)
}

// ---------------------------------------------------------------------------
// §6.2 — the sampled oracle
// ---------------------------------------------------------------------------

/// One read-only full-vs-incremental comparison, recording the divergence
/// count. Never repairs anything (§6.2).
async fn sample_oracle(
    conn: &libsql::Connection,
    space_id: &str,
    space: &str,
    coverage_epoch: i64,
) -> Result<(), WenlanError> {
    let tx = begin(conn).await?;
    let durable = community_gate::partition_is_durable(&tx).await;
    let full = oracle::recompute_full(&tx, space_id, coverage_epoch, durable).await?;
    let incremental = oracle::snapshot_incremental(&tx, space_id, coverage_epoch).await?;
    let divergences = oracle::diff(&full, &incremental).len();
    if divergences > 0 {
        evidence::bump(
            &tx,
            space_id,
            space,
            evidence::COUNTER_ORACLE_DIVERGENCE,
            divergences as i64,
        )
        .await?;
    }
    commit(tx).await
}

// ---------------------------------------------------------------------------
// Plumbing
// ---------------------------------------------------------------------------

/// `(space_id, name)` for every space, in a stable order so the rotation is
/// deterministic across restarts.
async fn spaces(conn: &libsql::Connection) -> Result<Vec<(String, String)>, WenlanError> {
    let tx = begin(conn).await?;
    let mut rows = tx
        .query("SELECT id, name FROM spaces ORDER BY name", ())
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 shadow spaces: {error}")))?;
    let mut spaces = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 shadow spaces row: {error}")))?
    {
        let decode = |error: libsql::Error| {
            WenlanError::VectorDb(format!("m6 shadow spaces decode: {error}"))
        };
        spaces.push((row.get(0).map_err(decode)?, row.get(1).map_err(decode)?));
    }
    drop(rows);
    commit(tx).await?;
    Ok(spaces)
}

async fn begin(conn: &libsql::Connection) -> Result<libsql::Transaction, WenlanError> {
    conn.transaction_with_behavior(libsql::TransactionBehavior::Immediate)
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 shadow begin: {error}")))
}

async fn commit(tx: libsql::Transaction) -> Result<(), WenlanError> {
    tx.commit()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 shadow commit: {error}")))
}
