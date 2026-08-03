// SPDX-License-Identifier: Apache-2.0
//! Machines A (candidate), B (claim/reservation) and D (coverage epoch) —
//! artifact 2 §4, §5, §7; spec §2.2, §3.3.
//!
//! Every entry point takes a **caller-owned** `&libsql::Transaction`, matching
//! `frontier_policy.rs`: the turn driver owns the transaction and the writers
//! are transaction-scoped, so a release and the group's next state can never be
//! observed separately (I-1). PR-B2 has no turn driver — nothing here is called
//! from `scheduler.rs`, `refinery/`, or any HTTP route.
//!
//! **What D4 exclusion is, and what it is not.** The two partial unique indexes
//! of migration 108 are the *entire* mechanism. [`prepare_candidate`] inserts
//! and lets SQLite refuse; it never pre-checks with a `SELECT` and skips the
//! insert, because that converts a database guarantee into a TOCTOU race
//! (spec §3.3). The one place this module does read-then-write is
//! [`open_coverage_epoch`], and the reason is mechanical rather than a
//! preference: migration 109's `m6_genesis_mutation_count_identity_insert`
//! trigger raises `ABORT`, which `INSERT OR IGNORE` does **not** swallow, so
//! the idempotent-insert idiom is unavailable on that one table.
//!
//! Every durable timestamp is SQLite `unixepoch()` evaluated in-statement
//! (S0-11). Backoff *durations* are computed in Rust because they are pure
//! functions of `attempt`, never of a clock.

use super::constants::{
    PENDING_CANDIDATES_PER_SPACE_CAP, RETRY_ATTEMPT_LIMIT, RETRY_BASE_DELAY_SECONDS,
    RETRY_MAX_DELAY_SECONDS, STALE_CARD_EXPIRED_DELAY_SECONDS, STALE_RETRY_EXHAUSTED_DELAY_SECONDS,
};
use super::frontier;
use super::frontier_policy::{bind_frontier_groups_to_card, CardGroup};
use super::identity::SignalKind;
use super::leases::{self, LeasePhase};
use crate::WenlanError;

// ---------------------------------------------------------------------------
// Machine A — states and the 19 transitions
// ---------------------------------------------------------------------------

/// D3 fixes the candidate state set at ten. S0-12 is why exhausted retry is not
/// an eleventh: it is `stale` carrying `reason = 'retry_exhausted'`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CandidateState {
    Observed,
    Prepared,
    Inferencing,
    Validating,
    Published,
    RetryWait,
    Stale,
    ReviewRequired,
    Suppressed,
    Superseded,
}

/// Every state, in the artifact's order — the enumeration the exit matrix and
/// the transition census walk.
pub const CANDIDATE_STATES: [CandidateState; 10] = [
    CandidateState::Observed,
    CandidateState::Prepared,
    CandidateState::Inferencing,
    CandidateState::Validating,
    CandidateState::Published,
    CandidateState::RetryWait,
    CandidateState::Stale,
    CandidateState::ReviewRequired,
    CandidateState::Suppressed,
    CandidateState::Superseded,
];

/// Terminal-final states (artifact 4 S0-41). `review_required` is
/// terminal-*pending* — it leaves by human action or expiry — and `stale` is
/// re-enterable via A19, so neither belongs here.
pub const TERMINAL_STATES: [CandidateState; 3] = [
    CandidateState::Published,
    CandidateState::Suppressed,
    CandidateState::Superseded,
];

/// States that count against the D8 pending-candidates cap: the ones with work
/// in the pipeline. `review_required` waits on a human and `stale` on a
/// multi-hour backoff, so neither is pending work and neither may starve a
/// space out of preparing new candidates.
pub const PENDING_STATES: [CandidateState; 5] = [
    CandidateState::Observed,
    CandidateState::Prepared,
    CandidateState::Inferencing,
    CandidateState::Validating,
    CandidateState::RetryWait,
];

impl CandidateState {
    /// The stored `genesis_candidates.state` value; the table's CHECK
    /// constraint is the mechanical form of D3's ten-state contract.
    pub fn as_str(self) -> &'static str {
        match self {
            CandidateState::Observed => "observed",
            CandidateState::Prepared => "prepared",
            CandidateState::Inferencing => "inferencing",
            CandidateState::Validating => "validating",
            CandidateState::Published => "published",
            CandidateState::RetryWait => "retry_wait",
            CandidateState::Stale => "stale",
            CandidateState::ReviewRequired => "review_required",
            CandidateState::Suppressed => "suppressed",
            CandidateState::Superseded => "superseded",
        }
    }

    /// Inverse of [`CandidateState::as_str`]; `None` for a value the CHECK
    /// constraint should have refused.
    pub fn parse(value: &str) -> Option<Self> {
        CANDIDATE_STATES
            .into_iter()
            .find(|state| state.as_str() == value)
    }

    /// Terminal-final (S0-41), not merely "stopped".
    pub fn is_terminal(self) -> bool {
        TERMINAL_STATES.contains(&self)
    }
}

/// One machine-A edge. `id` is the artifact's transition number, so a test can
/// name the transition it drives rather than describing it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CandidateTransition {
    pub id: &'static str,
    pub from: CandidateState,
    pub to: CandidateState,
}

/// Machine A's 19 transitions (artifact 2 §4). A7 is one transition over three
/// source states, so it is three rows sharing an ID — the recovery scan treats
/// `prepared`, `inferencing`, and `validating` identically.
pub const CANDIDATE_TRANSITIONS: [CandidateTransition; 21] = [
    edge("A2", CandidateState::Observed, CandidateState::Prepared),
    edge("A3", CandidateState::Observed, CandidateState::Stale),
    edge("A4", CandidateState::Observed, CandidateState::Superseded),
    edge("A5", CandidateState::Prepared, CandidateState::Inferencing),
    edge("A6", CandidateState::Prepared, CandidateState::RetryWait),
    edge("A7", CandidateState::Prepared, CandidateState::Stale),
    edge("A7", CandidateState::Inferencing, CandidateState::Stale),
    edge("A7", CandidateState::Validating, CandidateState::Stale),
    edge(
        "A8",
        CandidateState::Inferencing,
        CandidateState::Validating,
    ),
    edge("A9", CandidateState::Inferencing, CandidateState::RetryWait),
    edge("A10", CandidateState::Validating, CandidateState::Published),
    edge("A11", CandidateState::Validating, CandidateState::RetryWait),
    edge(
        "A12",
        CandidateState::Validating,
        CandidateState::ReviewRequired,
    ),
    edge("A13", CandidateState::RetryWait, CandidateState::Prepared),
    edge("A14", CandidateState::RetryWait, CandidateState::Stale),
    edge("A15", CandidateState::RetryWait, CandidateState::Stale),
    edge("A16", CandidateState::RetryWait, CandidateState::Superseded),
    edge(
        "A17",
        CandidateState::ReviewRequired,
        CandidateState::Suppressed,
    ),
    edge("A18", CandidateState::ReviewRequired, CandidateState::Stale),
    edge("A19", CandidateState::Stale, CandidateState::Prepared),
    // A1 (∅ → observed) has no source state; it is the insert in
    // `observe_candidate` and is listed here as the self-edge that makes the
    // census total rather than being silently absent.
    edge("A1", CandidateState::Observed, CandidateState::Observed),
];

const fn edge(id: &'static str, from: CandidateState, to: CandidateState) -> CandidateTransition {
    CandidateTransition { id, from, to }
}

/// Whether machine A permits `from → to`. The table is the contract; there is
/// no second predicate anywhere that could disagree with it.
pub fn transition_is_legal(from: CandidateState, to: CandidateState) -> bool {
    CANDIDATE_TRANSITIONS
        .iter()
        .any(|edge| edge.from == from && edge.to == to)
}

// ---------------------------------------------------------------------------
// Machine B — claim roles
// ---------------------------------------------------------------------------

/// D4's two roles. A third would break I-3 by existing: both partial unique
/// indexes exclude witnesses *by predicate*, so a role outside this pair would
/// fall outside the exclusion mechanism entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ClaimRole {
    Concept,
    Witness,
}

impl ClaimRole {
    pub fn as_str(self) -> &'static str {
        match self {
            ClaimRole::Concept => "concept",
            ClaimRole::Witness => "witness",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "concept" => Some(ClaimRole::Concept),
            "witness" => Some(ClaimRole::Witness),
            _ => None,
        }
    }
}

/// Artifact 2 §5: evidence-cluster and orphan-wikilink candidates insert
/// `'concept'`; the two overview signals insert `'witness'`, which is what
/// lets witness-only overview candidates coexist without stealing evidence.
pub fn claim_role_for(signal: SignalKind) -> ClaimRole {
    match signal {
        SignalKind::EvidenceCluster | SignalKind::OrphanWikilink => ClaimRole::Concept,
        SignalKind::CommunityOverview | SignalKind::SpaceOverview => ClaimRole::Witness,
    }
}

// ---------------------------------------------------------------------------
// Backoff (S0-2) and the stale re-entry delays (S0-151)
// ---------------------------------------------------------------------------

/// `60s * 4^(attempt-1)`, capped at 4h, **no jitter** (S0-2). `attempt = 0`
/// means nothing has been charged yet and the retry is immediate.
pub fn retry_backoff_seconds(attempt: i64) -> i64 {
    if attempt <= 0 {
        return 0;
    }
    let mut delay = RETRY_BASE_DELAY_SECONDS;
    for _ in 1..attempt {
        delay = delay.saturating_mul(4);
        if delay >= RETRY_MAX_DELAY_SECONDS {
            return RETRY_MAX_DELAY_SECONDS;
        }
    }
    delay.min(RETRY_MAX_DELAY_SECONDS)
}

/// S0-2/S0-12: `attempt` strictly greater than the limit is exhaustion.
pub fn retry_exhausted(attempt: i64) -> bool {
    attempt > RETRY_ATTEMPT_LIMIT
}

/// Why a candidate went `stale`. The reason is durable and distinguishable
/// (S0-12) and it is what sets the A19 re-entry delay (S0-151).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StaleReason {
    /// A3/A15 — `input_generation` or `active_root_digest` moved.
    InputMoved,
    /// A7 — the recovery scan found no live lease.
    LeaseLost,
    /// A14 — `attempt > 5` (S0-12).
    RetryExhausted,
    /// A18 — a surfaced review card expired.
    CardExpired,
}

impl StaleReason {
    pub fn as_str(self) -> &'static str {
        match self {
            StaleReason::InputMoved => "input_moved",
            StaleReason::LeaseLost => "lease_lost",
            StaleReason::RetryExhausted => "retry_exhausted",
            StaleReason::CardExpired => "card_expired",
        }
    }

    /// S0-151's per-reason delay, in seconds from now. `input_moved` wasted
    /// nothing and its fingerprint already differs, so it re-enters at once.
    pub fn re_entry_delay_seconds(self, attempt: i64) -> i64 {
        match self {
            StaleReason::InputMoved => 0,
            StaleReason::LeaseLost => retry_backoff_seconds(attempt),
            StaleReason::RetryExhausted => STALE_RETRY_EXHAUSTED_DELAY_SECONDS,
            StaleReason::CardExpired => STALE_CARD_EXPIRED_DELAY_SECONDS,
        }
    }
}

/// Whether a `retry_wait` entry charges an attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryCharge {
    /// A6 — the turn's LLM budget was spent. **`attempt` is unchanged**
    /// (S0-9): a busy space would otherwise burn its five-attempt budget on
    /// turns where it never reached the model, converting a scheduling artefact
    /// into a permanent `stale`.
    BudgetRefusal,
    /// A9/A11 — a model error, a timeout, or a finalize CAS miss.
    Charged,
}

// ---------------------------------------------------------------------------
// Machine D — coverage epoch
// ---------------------------------------------------------------------------

/// Whether this space's coverage rows are being mapped to a new identity
/// contract. Orthogonal to `genesis_enabled`, which PR-B never writes (Q11).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpochState {
    Active,
    Migrating,
}

impl EpochState {
    pub fn as_str(self) -> &'static str {
        match self {
            EpochState::Active => "active",
            EpochState::Migrating => "migrating",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "active" => Some(EpochState::Active),
            "migrating" => Some(EpochState::Migrating),
            _ => None,
        }
    }
}

/// Machine D's epoch-state edges (D2, D3, D4). D1 is the insert and D5/D6 are
/// the orthogonal `genesis_enabled` gate, which PR-B does not drive.
pub const EPOCH_TRANSITIONS: [(&str, EpochState, EpochState); 3] = [
    ("D2", EpochState::Active, EpochState::Migrating),
    ("D3", EpochState::Migrating, EpochState::Active),
    ("D4", EpochState::Migrating, EpochState::Active),
];

/// Whether machine D permits `from → to`.
pub fn epoch_transition_is_legal(from: EpochState, to: EpochState) -> bool {
    EPOCH_TRANSITIONS
        .iter()
        .any(|(_, edge_from, edge_to)| *edge_from == from && *edge_to == to)
}

/// One space's identity-contract era.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoverageState {
    pub space: String,
    pub space_id: String,
    pub coverage_epoch: i64,
    pub epoch_state: EpochState,
    /// Read-only here. PR-B creates rows and never writes the flag (Q11): both
    /// the `DEFAULT 0` and the absent-row rule independently mean "disabled",
    /// so a row with the flag at 0 is disabled by the same contract an absent
    /// row is.
    pub genesis_enabled: bool,
}

/// D1 — open this space's coverage epoch, idempotently.
///
/// Legal while `genesis_enabled = 0` (Q11 adjudication): the row is what gives
/// candidates a `coverage_epoch` to key on, and creating it enables nothing.
///
/// Read-then-insert rather than `INSERT OR IGNORE`, because migration 109's
/// identity trigger raises `ABORT` on a duplicate and `INSERT OR IGNORE` does
/// not swallow a trigger `RAISE(ABORT)`. The trigger — not this read — is still
/// the exclusion mechanism: a lost race fails loudly on the insert instead of
/// silently creating a second era.
pub async fn open_coverage_epoch(
    tx: &libsql::Transaction,
    space_id: &str,
    space: &str,
) -> Result<CoverageState, WenlanError> {
    if let Some(state) = read_coverage_state(tx, space_id).await? {
        return Ok(state);
    }
    tx.execute(
        "INSERT INTO genesis_coverage_state
             (space, space_id, coverage_epoch, epoch_state, opened_at,
              migration_cursor, genesis_enabled)
         VALUES (?1, ?2, 1, 'active', unixepoch(), NULL, 0)",
        libsql::params![space, space_id],
    )
    .await
    .map_err(|error| WenlanError::VectorDb(format!("m6 open coverage epoch: {error}")))?;
    read_coverage_state(tx, space_id)
        .await?
        .ok_or_else(|| WenlanError::VectorDb("m6 coverage epoch vanished after insert".into()))
}

/// The current coverage state, or `None` for a space with no row — which the
/// contract treats as genesis-disabled.
pub async fn read_coverage_state(
    tx: &libsql::Transaction,
    space_id: &str,
) -> Result<Option<CoverageState>, WenlanError> {
    let mut rows = tx
        .query(
            "SELECT space, space_id, coverage_epoch, epoch_state, genesis_enabled
               FROM genesis_coverage_state
              WHERE space_id = ?1",
            libsql::params![space_id],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 read coverage state: {error}")))?;
    let Some(row) = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 read coverage state row: {error}")))?
    else {
        return Ok(None);
    };
    let decode = |error: libsql::Error| {
        WenlanError::VectorDb(format!("m6 read coverage state decode: {error}"))
    };
    let epoch_state: String = row.get(3).map_err(decode)?;
    Ok(Some(CoverageState {
        space: row.get(0).map_err(decode)?,
        space_id: row.get(1).map_err(decode)?,
        coverage_epoch: row.get(2).map_err(decode)?,
        epoch_state: EpochState::parse(&epoch_state).ok_or_else(|| {
            WenlanError::VectorDb(format!("m6 unknown epoch_state {epoch_state:?}"))
        })?,
        genesis_enabled: row.get::<i64>(4).map_err(decode)? != 0,
    }))
}

/// D2 — the global identity-contract version outran this space's.
pub async fn begin_epoch_migration(
    tx: &libsql::Transaction,
    space_id: &str,
) -> Result<(), WenlanError> {
    let affected = tx
        .execute(
            "UPDATE genesis_coverage_state
                SET epoch_state = 'migrating', migration_cursor = NULL
              WHERE space_id = ?1 AND epoch_state = 'active'",
            libsql::params![space_id],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 begin epoch migration: {error}")))?;
    require_one(affected, "D2 begin epoch migration")
}

/// D3 — the forward mapping completed; the epoch increments and never
/// decreases (S0-8/I-7).
pub async fn complete_epoch_migration(
    tx: &libsql::Transaction,
    space_id: &str,
) -> Result<i64, WenlanError> {
    let affected = tx
        .execute(
            "UPDATE genesis_coverage_state
                SET coverage_epoch = coverage_epoch + 1,
                    epoch_state = 'active',
                    migration_cursor = NULL
              WHERE space_id = ?1 AND epoch_state = 'migrating'",
            libsql::params![space_id],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 complete epoch migration: {error}")))?;
    require_one(affected, "D3 complete epoch migration")?;
    Ok(read_coverage_state(tx, space_id)
        .await?
        .map(|state| state.coverage_epoch)
        .unwrap_or_default())
}

/// D4 — the mapping was abandoned; the epoch is **unchanged**, which is why
/// abandonment is free (S0-8).
pub async fn abandon_epoch_migration(
    tx: &libsql::Transaction,
    space_id: &str,
) -> Result<(), WenlanError> {
    let affected = tx
        .execute(
            "UPDATE genesis_coverage_state
                SET epoch_state = 'active', migration_cursor = NULL
              WHERE space_id = ?1 AND epoch_state = 'migrating'",
            libsql::params![space_id],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 abandon epoch migration: {error}")))?;
    require_one(affected, "D4 abandon epoch migration")
}

// ---------------------------------------------------------------------------
// Machine A — the candidate row
// ---------------------------------------------------------------------------

/// The identity a signal admission hands machine A (A1).
#[derive(Debug, Clone)]
pub struct ObservedCandidate<'a> {
    pub candidate_id: &'a str,
    pub slot_id: &'a str,
    pub page_id: &'a str,
    pub space: &'a str,
    pub signal_kind: SignalKind,
    pub coverage_epoch: i64,
    pub input_generation: i64,
    pub active_root_digest: &'a str,
}

/// What A1 did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObserveOutcome {
    /// A new candidate row exists.
    Observed,
    /// The derived `candidate_id` already had a row — the A13/A19 re-entry
    /// path, not an error (I-8: a retry reuses the candidate, it never mints a
    /// second one).
    AlreadyPresent,
    /// The D8 pending cap is full. **Nothing was written**: S0-54 forbids a cap
    /// from terminalizing a candidate or retiring a frontier row, so the group
    /// keeps its frontier row and is re-observed next cycle.
    RefusedPendingCap,
}

/// A1 — record an admitted signal as a candidate.
pub async fn observe_candidate(
    tx: &libsql::Transaction,
    candidate: &ObservedCandidate<'_>,
) -> Result<ObserveOutcome, WenlanError> {
    if read_candidate(tx, candidate.candidate_id).await?.is_some() {
        return Ok(ObserveOutcome::AlreadyPresent);
    }
    if pending_candidate_count(tx, candidate.space).await? >= PENDING_CANDIDATES_PER_SPACE_CAP {
        return Ok(ObserveOutcome::RefusedPendingCap);
    }
    tx.execute(
        "INSERT INTO genesis_candidates (
             candidate_id, slot_id, page_id, space, signal_kind,
             coverage_epoch, input_generation, active_root_digest,
             state, reason, attempt, next_attempt_at, lease_token,
             receipt_id, payload, payload_compacted_at, created_at, updated_at
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8,
                   'observed', NULL, 0, NULL, NULL,
                   NULL, NULL, NULL, unixepoch(), unixepoch())",
        libsql::params![
            candidate.candidate_id,
            candidate.slot_id,
            candidate.page_id,
            candidate.space,
            candidate.signal_kind.tag(),
            candidate.coverage_epoch,
            candidate.input_generation,
            candidate.active_root_digest
        ],
    )
    .await
    .map_err(|error| WenlanError::VectorDb(format!("m6 observe candidate: {error}")))?;
    Ok(ObserveOutcome::Observed)
}

/// The durable candidate row, as the machines read it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateRow {
    pub candidate_id: String,
    pub slot_id: String,
    pub page_id: String,
    pub space: String,
    pub signal_kind: SignalKind,
    pub coverage_epoch: i64,
    pub input_generation: i64,
    pub active_root_digest: String,
    pub state: CandidateState,
    pub reason: Option<String>,
    pub attempt: i64,
    pub next_attempt_at: Option<i64>,
    pub lease_token: Option<String>,
}

/// Read one candidate, or `None` when the ID has no row.
pub async fn read_candidate(
    tx: &libsql::Transaction,
    candidate_id: &str,
) -> Result<Option<CandidateRow>, WenlanError> {
    let mut rows = tx
        .query(
            "SELECT candidate_id, slot_id, page_id, space, signal_kind,
                    coverage_epoch, input_generation, active_root_digest,
                    state, reason, attempt, next_attempt_at, lease_token
               FROM genesis_candidates
              WHERE candidate_id = ?1",
            libsql::params![candidate_id],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 read candidate: {error}")))?;
    let Some(row) = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 read candidate row: {error}")))?
    else {
        return Ok(None);
    };
    let decode =
        |error: libsql::Error| WenlanError::VectorDb(format!("m6 read candidate decode: {error}"));
    let signal_tag: String = row.get(4).map_err(decode)?;
    let state: String = row.get(8).map_err(decode)?;
    Ok(Some(CandidateRow {
        candidate_id: row.get(0).map_err(decode)?,
        slot_id: row.get(1).map_err(decode)?,
        page_id: row.get(2).map_err(decode)?,
        space: row.get(3).map_err(decode)?,
        signal_kind: SignalKind::from_tag(&signal_tag).ok_or_else(|| {
            WenlanError::VectorDb(format!("m6 unknown signal_kind {signal_tag:?}"))
        })?,
        coverage_epoch: row.get(5).map_err(decode)?,
        input_generation: row.get(6).map_err(decode)?,
        active_root_digest: row.get(7).map_err(decode)?,
        state: CandidateState::parse(&state)
            .ok_or_else(|| WenlanError::VectorDb(format!("m6 unknown state {state:?}")))?,
        reason: row.get(9).map_err(decode)?,
        attempt: row.get(10).map_err(decode)?,
        next_attempt_at: row.get(11).map_err(decode)?,
        lease_token: row.get(12).map_err(decode)?,
    }))
}

/// One root a prepare wants to claim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimReservation {
    pub root_id: String,
    pub independence_group_id: String,
}

/// What a prepare attempt did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrepareOutcome {
    /// A2/A13/A19 committed: the lease row, every claim row, and
    /// `state = 'prepared'` are all in the caller's transaction.
    Prepared { lease_token: String },
    /// C2 — another holder owns `(phase, space, input_generation)`. The caller
    /// observes the winner and does no work.
    LeaseHeld,
    /// A13/A19 arrived before `next_attempt_at`. A legal no-op, not an error.
    NotDue,
    /// B1 — a partial unique index refused an exclusive claim. **The caller
    /// must roll back**: A2 is atomic, and an index rejection fails the whole
    /// prepare rather than leaving a partially claimed candidate.
    ClaimRefused { root_id: String },
}

/// A2 / A13 / A19 — the one atomic prepare.
///
/// The three transitions are one code path because the artifact makes them one:
/// A13 and A19 "reuse candidate, slot, page ID, lease operation, and receipt"
/// (I-8), so the only difference is the source state and whether `attempt`
/// resets. A19 resets it to 0; A13 keeps the charge its backoff already paid.
pub async fn prepare_candidate(
    tx: &libsql::Transaction,
    candidate_id: &str,
    claims: &[ClaimReservation],
) -> Result<PrepareOutcome, WenlanError> {
    let candidate = read_candidate(tx, candidate_id)
        .await?
        .ok_or_else(|| WenlanError::NotFound(format!("m6 prepare: no candidate {candidate_id}")))?;
    if !transition_is_legal(candidate.state, CandidateState::Prepared) {
        return Err(WenlanError::Conflict(format!(
            "m6 prepare: {} → prepared is not a machine-A transition",
            candidate.state.as_str()
        )));
    }
    let re_entry = matches!(
        candidate.state,
        CandidateState::RetryWait | CandidateState::Stale
    );
    if re_entry && !backoff_elapsed(tx, candidate_id).await? {
        return Ok(PrepareOutcome::NotDue);
    }

    let Some(lease_token) = leases::acquire(
        tx,
        LeasePhase::Genesis,
        &candidate.space,
        candidate.input_generation,
        candidate.attempt,
    )
    .await?
    else {
        return Ok(PrepareOutcome::LeaseHeld);
    };

    let role = claim_role_for(candidate.signal_kind);
    for claim in claims {
        match insert_claim(tx, candidate_id, claim, candidate.coverage_epoch, role).await? {
            ClaimInsert::Accepted => {}
            ClaimInsert::Refused => {
                return Ok(PrepareOutcome::ClaimRefused {
                    root_id: claim.root_id.clone(),
                })
            }
        }
    }
    // F2: a concept claim is the group's durable reason from here, so its
    // frontier row retires in the same transaction. A witness claim never gave
    // its group a reason (B2/B8) and retires nothing.
    if role == ClaimRole::Concept {
        for group in distinct_groups(claims) {
            frontier::retire_frontier_row(tx, &candidate.space, &group, candidate.coverage_epoch)
                .await?;
        }
    }

    let attempt = if candidate.state == CandidateState::Stale {
        0
    } else {
        candidate.attempt
    };
    let affected = tx
        .execute(
            "UPDATE genesis_candidates
                SET state = 'prepared', lease_token = ?2, attempt = ?3,
                    reason = NULL, next_attempt_at = NULL, updated_at = unixepoch()
              WHERE candidate_id = ?1
                AND state IN ('observed', 'retry_wait', 'stale')",
            libsql::params![candidate_id, lease_token.clone(), attempt],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 prepare candidate: {error}")))?;
    require_one(affected, "A2/A13/A19 prepare")?;
    Ok(PrepareOutcome::Prepared { lease_token })
}

/// A5 — the model call begins. No transaction, mutex, lock, or guard crosses
/// this edge (D3, I-5); in the shadow it is trivially satisfied, and the edge
/// exists so PR-E cannot quietly acquire one.
pub async fn mark_inferencing(
    tx: &libsql::Transaction,
    candidate_id: &str,
) -> Result<(), WenlanError> {
    move_state(
        tx,
        candidate_id,
        &[CandidateState::Prepared],
        CandidateState::Inferencing,
        None,
    )
    .await
}

/// A8 — the model returned and its output parsed.
pub async fn mark_validating(
    tx: &libsql::Transaction,
    candidate_id: &str,
) -> Result<(), WenlanError> {
    move_state(
        tx,
        candidate_id,
        &[CandidateState::Inferencing],
        CandidateState::Validating,
        None,
    )
    .await
}

/// A6 / A9 / A11 — enter `retry_wait`.
///
/// [`RetryCharge::BudgetRefusal`] leaves `attempt` alone (S0-9) and re-enters
/// immediately; [`RetryCharge::Charged`] increments it and applies the S0-2
/// backoff. Reservations are retained across this edge (B3): what bounds them
/// is the recovery scan plus A14's exhaustion, not a timer on the claim row.
pub async fn mark_retry_wait(
    tx: &libsql::Transaction,
    candidate_id: &str,
    charge: RetryCharge,
) -> Result<(), WenlanError> {
    let candidate = read_candidate(tx, candidate_id).await?.ok_or_else(|| {
        WenlanError::NotFound(format!("m6 retry_wait: no candidate {candidate_id}"))
    })?;
    guard_transition(candidate.state, CandidateState::RetryWait)?;
    let (attempt, delay) = match charge {
        RetryCharge::BudgetRefusal => (candidate.attempt, 0),
        RetryCharge::Charged => {
            let attempt = candidate.attempt + 1;
            (attempt, retry_backoff_seconds(attempt))
        }
    };
    let affected = tx
        .execute(
            "UPDATE genesis_candidates
                SET state = 'retry_wait', attempt = ?2,
                    next_attempt_at = unixepoch() + ?3, updated_at = unixepoch()
              WHERE candidate_id = ?1
                AND state IN ('prepared', 'inferencing', 'validating')",
            libsql::params![candidate_id, attempt, delay],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 retry_wait: {error}")))?;
    require_one(affected, "A6/A9/A11 retry_wait")
}

/// A3 / A7 / A14 / A15 / A18 — enter `stale`.
///
/// Atomic with the group's re-destination (D4): every live claim is released
/// and every group this candidate held returns to `waiting_frontier` (F5), and
/// an expired review card's binding is closed on the way (F8). A crash between
/// the release and the frontier row would leave a group with zero durable
/// reasons, which is exactly what I-1 forbids — so it is one transaction, and
/// the transaction is the caller's.
pub async fn mark_stale(
    tx: &libsql::Transaction,
    candidate_id: &str,
    reason: StaleReason,
) -> Result<(), WenlanError> {
    let candidate = read_candidate(tx, candidate_id)
        .await?
        .ok_or_else(|| WenlanError::NotFound(format!("m6 stale: no candidate {candidate_id}")))?;
    guard_transition(candidate.state, CandidateState::Stale)?;
    settle_groups_to_frontier(tx, &candidate, reason.as_str()).await?;
    let delay = reason.re_entry_delay_seconds(candidate.attempt);
    let affected = tx
        .execute(
            "UPDATE genesis_candidates
                SET state = 'stale', reason = ?2,
                    lease_token = NULL,
                    next_attempt_at = unixepoch() + ?3, updated_at = unixepoch()
              WHERE candidate_id = ?1
                AND state IN ('observed', 'prepared', 'inferencing',
                              'validating', 'retry_wait', 'review_required')",
            libsql::params![candidate_id, reason.as_str(), delay],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 stale: {error}")))?;
    require_one(affected, "A3/A7/A14/A15/A18 stale")
}

/// A4 / A16 — another candidate published this slot. Terminal-final, and the
/// published page is the durable witness.
pub async fn mark_superseded(
    tx: &libsql::Transaction,
    candidate_id: &str,
) -> Result<(), WenlanError> {
    let candidate = read_candidate(tx, candidate_id).await?.ok_or_else(|| {
        WenlanError::NotFound(format!("m6 superseded: no candidate {candidate_id}"))
    })?;
    guard_transition(candidate.state, CandidateState::Superseded)?;
    settle_groups_to_frontier(tx, &candidate, "slot_published").await?;
    let affected = tx
        .execute(
            "UPDATE genesis_candidates
                SET state = 'superseded', reason = 'slot_published',
                    lease_token = NULL, next_attempt_at = NULL,
                    updated_at = unixepoch()
              WHERE candidate_id = ?1 AND state IN ('observed', 'retry_wait')",
            libsql::params![candidate_id],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 superseded: {error}")))?;
    require_one(affected, "A4/A16 superseded")
}

/// A12 — entailment refusal or a policy call for human judgment.
///
/// D4 requires the reservation release and the card binding to be atomic; a
/// crash between them would leave a group with two durable reasons. The release
/// restores each group's frontier row and `bind_frontier_groups_to_card`
/// retires it again inside the same transaction, so the intermediate state is
/// never observable and `first_seen_at` travels through the binding rather than
/// being re-minted.
pub async fn mark_review_required(
    tx: &libsql::Transaction,
    candidate_id: &str,
    card_id: &str,
) -> Result<(), WenlanError> {
    let candidate = read_candidate(tx, candidate_id).await?.ok_or_else(|| {
        WenlanError::NotFound(format!("m6 review_required: no candidate {candidate_id}"))
    })?;
    guard_transition(candidate.state, CandidateState::ReviewRequired)?;
    let groups = release_claims_to_frontier(tx, &candidate, "review_required").await?;
    let card_groups: Vec<CardGroup<'_>> = groups
        .iter()
        .map(|group| CardGroup::new(group.as_str(), candidate.page_id.as_str()))
        .collect();
    bind_frontier_groups_to_card(
        tx,
        &candidate.space,
        candidate.coverage_epoch,
        card_id,
        &card_groups,
    )
    .await
    .map_err(|error| WenlanError::VectorDb(format!("m6 review card binding: {error}")))?;
    let affected = tx
        .execute(
            "UPDATE genesis_candidates
                SET state = 'review_required', reason = 'review_required',
                    lease_token = NULL, next_attempt_at = NULL,
                    updated_at = unixepoch()
              WHERE candidate_id = ?1 AND state = 'validating'",
            libsql::params![candidate_id],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 review_required: {error}")))?;
    require_one(affected, "A12 review_required")
}

/// A17 — a human dismissed the card. The card is coalesced per
/// `(space, coverage_epoch)`, so dismissal closes **every** live binding for
/// `card_id` and writes one suppression per group, in one transaction
/// (artifact 5 §3.6). Human decisions are never discarded (D14), which is why
/// the closed bindings and the suppression identities are retained history.
pub async fn mark_suppressed(
    tx: &libsql::Transaction,
    candidate_id: &str,
    card_id: &str,
    reason: &str,
) -> Result<(), WenlanError> {
    let candidate = read_candidate(tx, candidate_id).await?.ok_or_else(|| {
        WenlanError::NotFound(format!("m6 suppressed: no candidate {candidate_id}"))
    })?;
    guard_transition(candidate.state, CandidateState::Suppressed)?;
    super::frontier_policy::dismiss_card_to_suppression(tx, card_id, reason)
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 card dismissal: {error}")))?;
    let affected = tx
        .execute(
            "UPDATE genesis_candidates
                SET state = 'suppressed', reason = ?2, lease_token = NULL,
                    next_attempt_at = NULL, updated_at = unixepoch()
              WHERE candidate_id = ?1 AND state = 'review_required'",
            libsql::params![candidate_id, reason],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 suppressed: {error}")))?;
    require_one(affected, "A17 suppressed")
}

/// S0-10/S0-53 compaction: null the payload of terminal candidates older than
/// the retention window and stamp `payload_compacted_at`.
///
/// Not a transition — a compacted candidate keeps its state — and never a
/// delete. Deleting a terminal candidate row would drop its group's accounting
/// to `reason_count = 0` and the reconciler would faithfully re-discover a
/// group that was already handled: an infinite rediscovery loop caused by a
/// retention policy (S0-53).
pub async fn compact_terminal_payloads(
    tx: &libsql::Transaction,
    space: &str,
    retention_seconds: i64,
) -> Result<u64, WenlanError> {
    tx.execute(
        "UPDATE genesis_candidates
            SET payload = NULL, payload_compacted_at = unixepoch()
          WHERE space = ?1
            AND state IN ('published', 'stale', 'suppressed', 'superseded')
            AND updated_at <= unixepoch() - ?2
            AND payload IS NOT NULL",
        libsql::params![space, retention_seconds],
    )
    .await
    .map_err(|error| WenlanError::VectorDb(format!("m6 compact payloads: {error}")))
}

// ---------------------------------------------------------------------------
// Machine B — claim rows
// ---------------------------------------------------------------------------

/// One live claim, as the frontier and the oracle read it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimRow {
    pub candidate_id: String,
    pub root_id: String,
    pub independence_group_id: String,
    pub coverage_epoch: i64,
    pub claim_role: ClaimRole,
}

/// Every unreleased claim in a space, ordered for determinism.
///
/// `released_at IS NULL` is the liveness definition (S0-6/Q8) because it is the
/// one the two partial unique indexes enforce; the candidate-state definition
/// diverges from it exactly in the window before the recovery scan runs, and
/// that divergence is observable rather than papered over.
pub async fn live_claims(
    tx: &libsql::Transaction,
    space: &str,
    coverage_epoch: i64,
) -> Result<Vec<ClaimRow>, WenlanError> {
    let mut rows = tx
        .query(
            "SELECT gr.candidate_id, gr.root_id, gr.independence_group_id,
                    gr.coverage_epoch, gr.claim_role
               FROM genesis_candidate_roots gr
               JOIN genesis_candidates c ON c.candidate_id = gr.candidate_id
              WHERE c.space = ?1
                AND gr.coverage_epoch = ?2
                AND gr.released_at IS NULL
              ORDER BY gr.candidate_id, gr.root_id",
            libsql::params![space, coverage_epoch],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 live claims: {error}")))?;
    let mut claims = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 live claims row: {error}")))?
    {
        let decode =
            |error: libsql::Error| WenlanError::VectorDb(format!("m6 live claims decode: {error}"));
        let role: String = row.get(4).map_err(decode)?;
        claims.push(ClaimRow {
            candidate_id: row.get(0).map_err(decode)?,
            root_id: row.get(1).map_err(decode)?,
            independence_group_id: row.get(2).map_err(decode)?,
            coverage_epoch: row.get(3).map_err(decode)?,
            claim_role: ClaimRole::parse(&role)
                .ok_or_else(|| WenlanError::VectorDb(format!("m6 unknown claim_role {role:?}")))?,
        });
    }
    Ok(claims)
}

enum ClaimInsert {
    Accepted,
    Refused,
}

/// B1/B2 — insert and let SQLite refuse.
///
/// A unique-index violation is the *answer*, not an error: it means another
/// candidate already holds this root or group at this epoch. SQLite's default
/// `ABORT` resolution reverts the statement and leaves the transaction active,
/// so the caller can turn the refusal into `PrepareOutcome::ClaimRefused` and
/// roll back deliberately.
///
/// **The upsert is what makes re-entry possible at all.** `genesis_candidate_roots`
/// is keyed `PRIMARY KEY(candidate_id, root_id)`, so a candidate holds each root
/// at most one row *ever* — while A13 and A19 both re-present the same claim set
/// (I-8: a retry reuses the candidate rather than minting a second one). A plain
/// insert would therefore make every re-prepare collide with the candidate's own
/// history: still-live under A13 (reservations are retained across `retry_wait`,
/// B3), released under A19. `ON CONFLICT(candidate_id, root_id) DO UPDATE`
/// revives the candidate's **own** row and leaves `created_at` standing, so the
/// group's first-claim time survives churn.
///
/// This is not a second exclusion check (I-2). The conflict target is the
/// candidate's own primary key; the two partial unique indexes still decide
/// whether another candidate holds the root or group, and a revival that would
/// cross one of them still fails with `SQLITE_CONSTRAINT` → `Refused`.
async fn insert_claim(
    tx: &libsql::Transaction,
    candidate_id: &str,
    claim: &ClaimReservation,
    coverage_epoch: i64,
    role: ClaimRole,
) -> Result<ClaimInsert, WenlanError> {
    let result = tx
        .execute(
            "INSERT INTO genesis_candidate_roots (
                 candidate_id, root_id, independence_group_id, coverage_epoch,
                 claim_role, released_at, released_reason, consumed, retained,
                 created_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, NULL, NULL, 0, 0, unixepoch())
             ON CONFLICT(candidate_id, root_id) DO UPDATE
                SET independence_group_id = excluded.independence_group_id,
                    coverage_epoch        = excluded.coverage_epoch,
                    claim_role            = excluded.claim_role,
                    released_at           = NULL,
                    released_reason       = NULL",
            libsql::params![
                candidate_id,
                claim.root_id.as_str(),
                claim.independence_group_id.as_str(),
                coverage_epoch,
                role.as_str()
            ],
        )
        .await;
    match result {
        Ok(_) => Ok(ClaimInsert::Accepted),
        Err(error) if is_constraint_violation(&error) => Ok(ClaimInsert::Refused),
        Err(error) => Err(WenlanError::VectorDb(format!("m6 insert claim: {error}"))),
    }
}

/// libSQL surfaces a constraint failure as a `SqliteFailure` carrying SQLite's
/// `SQLITE_CONSTRAINT` (19) primary result code; the extended code identifies
/// which constraint. Matching on the code rather than on message text is what
/// keeps this from silently swallowing an unrelated error.
fn is_constraint_violation(error: &libsql::Error) -> bool {
    match error {
        libsql::Error::SqliteFailure(code, _) => code & 0xff == 19,
        _ => false,
    }
}

/// B4/B6/B8 plus F5 — release every live claim and return each concept group
/// to `waiting_frontier`, in the caller's transaction.
async fn release_claims_to_frontier(
    tx: &libsql::Transaction,
    candidate: &CandidateRow,
    reason: &str,
) -> Result<Vec<String>, WenlanError> {
    let groups = candidate_concept_groups(tx, &candidate.candidate_id, true).await?;
    tx.execute(
        "UPDATE genesis_candidate_roots
            SET released_at = unixepoch(), released_reason = ?2
          WHERE candidate_id = ?1 AND released_at IS NULL",
        libsql::params![candidate.candidate_id.as_str(), reason],
    )
    .await
    .map_err(|error| WenlanError::VectorDb(format!("m6 release claims: {error}")))?;
    for group in &groups {
        frontier::restore_frontier_row(tx, &candidate.space, group, candidate.coverage_epoch)
            .await?;
    }
    Ok(groups)
}

/// Release live claims **and** close any open review-card binding this
/// candidate's groups still hold, restoring each group's frontier row. The two
/// paths are disjoint per group — a group with a live claim has no open binding
/// and vice versa — so running both is how one exit writer covers A7/A14/A15
/// (from a reservation) and A18 (from a card) without a second code path.
async fn settle_groups_to_frontier(
    tx: &libsql::Transaction,
    candidate: &CandidateRow,
    reason: &str,
) -> Result<(), WenlanError> {
    release_claims_to_frontier(tx, candidate, reason).await?;
    for group in candidate_concept_groups(tx, &candidate.candidate_id, false).await? {
        frontier::expire_card_binding_to_frontier(
            tx,
            &candidate.space,
            &group,
            candidate.coverage_epoch,
        )
        .await?;
    }
    Ok(())
}

/// The concept groups this candidate claims. `live_only` selects the
/// unreleased ones (F5's inputs); the full set is what F8 needs, because an
/// expired card's claims were already released at A12.
async fn candidate_concept_groups(
    tx: &libsql::Transaction,
    candidate_id: &str,
    live_only: bool,
) -> Result<Vec<String>, WenlanError> {
    let sql = if live_only {
        "SELECT DISTINCT independence_group_id
           FROM genesis_candidate_roots
          WHERE candidate_id = ?1 AND claim_role = 'concept'
            AND released_at IS NULL
          ORDER BY independence_group_id"
    } else {
        "SELECT DISTINCT independence_group_id
           FROM genesis_candidate_roots
          WHERE candidate_id = ?1 AND claim_role = 'concept'
          ORDER BY independence_group_id"
    };
    let mut rows = tx
        .query(sql, libsql::params![candidate_id])
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 candidate groups: {error}")))?;
    let mut groups = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 candidate groups row: {error}")))?
    {
        groups.push(row.get::<String>(0).map_err(|error| {
            WenlanError::VectorDb(format!("m6 candidate groups decode: {error}"))
        })?);
    }
    Ok(groups)
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn distinct_groups(claims: &[ClaimReservation]) -> Vec<String> {
    let mut groups: Vec<String> = claims
        .iter()
        .map(|claim| claim.independence_group_id.clone())
        .collect();
    groups.sort();
    groups.dedup();
    groups
}

fn guard_transition(from: CandidateState, to: CandidateState) -> Result<(), WenlanError> {
    if transition_is_legal(from, to) {
        Ok(())
    } else {
        Err(WenlanError::Conflict(format!(
            "m6 candidate: {} → {} is not a machine-A transition",
            from.as_str(),
            to.as_str()
        )))
    }
}

async fn move_state(
    tx: &libsql::Transaction,
    candidate_id: &str,
    from: &[CandidateState],
    to: CandidateState,
    reason: Option<&str>,
) -> Result<(), WenlanError> {
    let candidate = read_candidate(tx, candidate_id).await?.ok_or_else(|| {
        WenlanError::NotFound(format!("m6 transition: no candidate {candidate_id}"))
    })?;
    guard_transition(candidate.state, to)?;
    if !from.contains(&candidate.state) {
        return Err(WenlanError::Conflict(format!(
            "m6 candidate {candidate_id} is {} , not one of {from:?}",
            candidate.state.as_str()
        )));
    }
    let affected = tx
        .execute(
            "UPDATE genesis_candidates
                SET state = ?2, reason = COALESCE(?3, reason), updated_at = unixepoch()
              WHERE candidate_id = ?1 AND state = ?4",
            libsql::params![candidate_id, to.as_str(), reason, candidate.state.as_str()],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 candidate transition: {error}")))?;
    require_one(affected, "machine-A transition")
}

/// S0-11: the backoff comparison is SQLite's, not Rust's, so a test can move it
/// only by moving the stored `next_attempt_at`.
async fn backoff_elapsed(
    tx: &libsql::Transaction,
    candidate_id: &str,
) -> Result<bool, WenlanError> {
    let mut rows = tx
        .query(
            "SELECT unixepoch() >= COALESCE(next_attempt_at, 0)
               FROM genesis_candidates WHERE candidate_id = ?1",
            libsql::params![candidate_id],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 backoff check: {error}")))?;
    Ok(rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 backoff row: {error}")))?
        .map(|row| row.get::<i64>(0))
        .transpose()
        .map_err(|error| WenlanError::VectorDb(format!("m6 backoff decode: {error}")))?
        .map(|due| due != 0)
        .unwrap_or(false))
}

async fn pending_candidate_count(
    tx: &libsql::Transaction,
    space: &str,
) -> Result<usize, WenlanError> {
    let mut rows = tx
        .query(
            "SELECT COUNT(*) FROM genesis_candidates
              WHERE space = ?1
                AND state IN ('observed', 'prepared', 'inferencing',
                              'validating', 'retry_wait')",
            libsql::params![space],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 pending count: {error}")))?;
    let count: i64 = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 pending count row: {error}")))?
        .map(|row| row.get::<i64>(0))
        .transpose()
        .map_err(|error| WenlanError::VectorDb(format!("m6 pending count decode: {error}")))?
        .unwrap_or_default();
    Ok(count.max(0) as usize)
}

fn require_one(affected: u64, what: &str) -> Result<(), WenlanError> {
    if affected == 1 {
        Ok(())
    } else {
        Err(WenlanError::Conflict(format!(
            "m6 {what} affected {affected} rows, expected exactly 1"
        )))
    }
}
