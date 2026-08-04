// SPDX-License-Identifier: Apache-2.0
//! Machine E — **dry-run** finalization (spec §2.4, §5).
//!
//! The real path would, at this point, call the model, synthesize a page, and
//! publish it. PR-B does the first half only: it opens the caller's
//! transaction, evaluates the **eight CAS gates of §5.2 in order**, records the
//! verdict as shadow evidence, and stops. Nothing outside `genesis_*`, the
//! shared `grouping_leases` registry, and M6's own counters is written, in
//! either outcome.
//!
//! **No LLM provider is consulted, and that is checkable by signature rather
//! than by reading the body** (§5.1): this module does not import, name, or
//! receive `LlmProvider`, `LlmEngine`, or any provider type. A runtime `if` can
//! be flipped; a missing parameter cannot. `finalize_test::the_module_names_no_llm_provider`
//! asserts it against this file's own source text (G4.5).
//!
//! **Why the gates run even though nothing publishes.** A dry run that skipped
//! the gates would prove nothing about the real path — the whole value of the
//! shadow is that the CAS discipline is exercised against live, concurrently
//! moving state before a writer depends on it. So every gate reads live state
//! and compares it against a **stored** CAS input, never against a fresh
//! recomputation of the same thing (§5.2). M5 learned this the expensive way in
//! `finalize_page_support` (`crates/wenlan-core/src/db/claim_derivation.rs`):
//! `active → draining` leaves the recomputed verdict identical while
//! invalidating the regime the worker leased under, so the regime needs its own
//! stored input.
//!
//! **Where the stored inputs live.** Three are columns machine A already
//! writes — `input_generation`, `active_root_digest`, `coverage_epoch`. Two
//! have no column in migration 108/109: the epoch *regime* (`epoch_state`) and
//! M4's published community generation. Rather than recompute them at
//! finalization — the exact mistake above — PR-B3 stamps them into
//! `genesis_candidates.payload` at prepare time as a [`CasEnvelope`], which is
//! machine E's own column (artifact 12 defers "receipts are machine E's" to
//! this PR). The envelope also carries the deterministic `receipt_id` that I-8
//! requires a retry to reuse. **No migration was needed and none was added**;
//! `PRAGMA user_version` stays at 110.
//!
//! Every clock is SQLite `unixepoch()` evaluated in-statement (S0-11).

use serde_json::{json, Value};

use super::candidates::{
    self, claim_role_for, CandidateRow, CandidateState, ClaimRole, EpochState, RetryCharge,
};
use super::community_gate;
use super::identity::{self, SignalKind};
use super::leases::{self, LeasePhase};
use crate::WenlanError;

// ---------------------------------------------------------------------------
// The eight gates
// ---------------------------------------------------------------------------

/// §5.2's eight CAS gates, **in evaluation order**.
///
/// The discriminant order is the evaluation order and [`CasGate::ALL`] is the
/// same sequence, so "the gates ran in order" is a property a test can assert
/// against a single list rather than by reading the body. Order is load-bearing
/// (§5.2): a run that checked evidence liveness before lease ownership would be
/// acting on state it does not own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CasGate {
    /// E-1 — the lease token still belongs to this worker.
    LeaseOwnership,
    /// E-2 — `input_generation` unchanged.
    InputGeneration,
    /// E-3 — the active-root-set digest unchanged.
    RootDigest,
    /// E-4 — `coverage_epoch` **and** `epoch_state` unchanged.
    CoverageEpoch,
    /// E-5 — claims still unreleased; witness rows still present.
    ClaimLiveness,
    /// E-6 — every root still active, every edge still grounded and live.
    EvidenceLiveness,
    /// E-7 — M4's community generation still current and durable.
    CommunityGeneration,
    /// E-8 — M5's page/dependency truth preconditions still hold.
    PageTruth,
}

impl CasGate {
    /// Every gate, in §5.2's order. The list a caller walks and a test asserts.
    pub const ALL: [CasGate; 8] = [
        CasGate::LeaseOwnership,
        CasGate::InputGeneration,
        CasGate::RootDigest,
        CasGate::CoverageEpoch,
        CasGate::ClaimLiveness,
        CasGate::EvidenceLiveness,
        CasGate::CommunityGeneration,
        CasGate::PageTruth,
    ];

    /// The artifact's gate label. Used in counter names and stored verdicts, so
    /// it is a wire value: changing one renames a durable counter.
    pub fn as_str(self) -> &'static str {
        match self {
            CasGate::LeaseOwnership => "e1_lease_ownership",
            CasGate::InputGeneration => "e2_input_generation",
            CasGate::RootDigest => "e3_root_digest",
            CasGate::CoverageEpoch => "e4_coverage_epoch",
            CasGate::ClaimLiveness => "e5_claim_liveness",
            CasGate::EvidenceLiveness => "e6_evidence_liveness",
            CasGate::CommunityGeneration => "e7_community_generation",
            CasGate::PageTruth => "e8_page_truth",
        }
    }

    /// Inverse of [`CasGate::as_str`], for reading a stored verdict back.
    pub fn parse(value: &str) -> Option<Self> {
        CasGate::ALL.into_iter().find(|gate| gate.as_str() == value)
    }
}

// ---------------------------------------------------------------------------
// The stored CAS envelope
// ---------------------------------------------------------------------------

/// The two CAS inputs with no column of their own, plus the I-8 receipt.
///
/// Stamped into `genesis_candidates.payload` inside the **prepare**
/// transaction, read back inside the **finalize** transaction. Prepare and
/// finalize are separate turns (§4.2 priorities 3 and 4), so an in-memory carry
/// would not survive the gap — and a value that does not survive a restart is
/// not a CAS input, it is a guess.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CasEnvelope {
    /// I-8: a retry reuses the same receipt rather than minting new identity.
    /// Derived from `candidate_id`, so reuse is structural rather than a rule
    /// someone has to remember.
    pub receipt_id: String,
    /// The epoch regime the prepare ran under. E-4's second half.
    pub epoch_state: EpochState,
    /// M4's `space_graph_state.published_generation` at prepare time, or `None`
    /// when the space had no M4 row. E-7's stored input.
    pub community_generation: Option<i64>,
    /// The dry-run verdict, once one has been recorded. `None` means this
    /// prepare has not been finalized yet — which is exactly the driver's
    /// selection predicate, so a verified candidate is never re-verified until
    /// something re-prepares it.
    pub verdict: Option<Verdict>,
}

/// What one dry-run finalization concluded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Every gate passed. In the real path this is where a page would be
    /// published; here it is where the run stops (§2.4).
    Passed,
    /// The first gate that refused. "First" matters: §7.2 records outcomes *by
    /// first-failing gate*, which is only meaningful because evaluation
    /// short-circuits in §5.2's order.
    Refused(CasGate),
}

/// The deterministic receipt for a candidate (I-8).
pub fn receipt_id(candidate_id: &str) -> String {
    format!("m6r_{candidate_id}")
}

impl CasEnvelope {
    fn to_json(&self) -> String {
        json!({
            "receipt_id": self.receipt_id,
            "epoch_state": self.epoch_state.as_str(),
            "community_generation": self.community_generation,
            "verdict": match self.verdict {
                None => Value::Null,
                Some(Verdict::Passed) => Value::from("passed"),
                Some(Verdict::Refused(gate)) => Value::from(gate.as_str()),
            },
        })
        .to_string()
    }

    fn from_json(raw: &str) -> Option<Self> {
        let value: Value = serde_json::from_str(raw).ok()?;
        let verdict = match value.get("verdict") {
            None | Some(Value::Null) => None,
            Some(Value::String(text)) if text == "passed" => Some(Verdict::Passed),
            Some(Value::String(text)) => Some(Verdict::Refused(CasGate::parse(text)?)),
            Some(_) => return None,
        };
        Some(CasEnvelope {
            receipt_id: value.get("receipt_id")?.as_str()?.to_string(),
            epoch_state: EpochState::parse(value.get("epoch_state")?.as_str()?)?,
            community_generation: match value.get("community_generation") {
                None | Some(Value::Null) => None,
                Some(other) => Some(other.as_i64()?),
            },
            verdict,
        })
    }
}

/// Stamp the prepare-time CAS envelope onto the candidate (called inside the
/// prepare transaction).
///
/// Unconditional overwrite, and deliberately so: a re-prepare (A13/A19) is a
/// new lease under possibly-new M4 state, so the envelope it verifies against
/// must be the new one. Clearing the verdict is what makes the candidate
/// eligible for finalization again — the single mechanism that stops a passing
/// dry run from re-running every turn forever.
pub async fn record_envelope(
    tx: &libsql::Transaction,
    candidate_id: &str,
    envelope: &CasEnvelope,
) -> Result<(), WenlanError> {
    let affected = tx
        .execute(
            "UPDATE genesis_candidates
                SET payload = ?2, receipt_id = ?3, updated_at = unixepoch()
              WHERE candidate_id = ?1",
            libsql::params![
                candidate_id,
                envelope.to_json(),
                envelope.receipt_id.as_str()
            ],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 record cas envelope: {error}")))?;
    if affected != 1 {
        return Err(WenlanError::NotFound(format!(
            "m6 finalize: no candidate {candidate_id} to stamp"
        )));
    }
    Ok(())
}

/// Read the stored envelope back. `None` covers both "never stamped" and "the
/// payload was compacted" (S0-53 nulls `payload` on terminal rows); the caller
/// treats both as "not finalizable", never as an error, because a compacted
/// terminal candidate is a normal state rather than a fault.
pub async fn read_envelope(
    tx: &libsql::Transaction,
    candidate_id: &str,
) -> Result<Option<CasEnvelope>, WenlanError> {
    let mut rows = tx
        .query(
            "SELECT payload FROM genesis_candidates WHERE candidate_id = ?1",
            libsql::params![candidate_id],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 read cas envelope: {error}")))?;
    let Some(row) = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 read cas envelope row: {error}")))?
    else {
        return Ok(None);
    };
    let raw: Option<String> = row
        .get(0)
        .map_err(|error| WenlanError::VectorDb(format!("m6 decode cas envelope: {error}")))?;
    Ok(raw.as_deref().and_then(CasEnvelope::from_json))
}

// ---------------------------------------------------------------------------
// The dry run
// ---------------------------------------------------------------------------

/// What one dry-run finalization did, as the turn driver reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Finalization {
    /// Every gate passed. Nothing was published — that is the point.
    DryRunPassed { candidate_id: String },
    /// A gate refused. The candidate was requeued via A11 (§5.2: a CAS miss
    /// means the world moved, which is normal — not a park, not a terminal).
    Refused { candidate_id: String, gate: CasGate },
    /// The candidate is not in a finalizable shape: no envelope, already
    /// verified for this prepare, or not in `prepared`. A legal no-op.
    NotEligible { candidate_id: String },
}

/// Run the eight gates against live state and record the verdict.
///
/// **The dry run is dry, structurally.** Between entry and the first gate's
/// verdict this function issues only `SELECT`s. The only statements it can ever
/// execute are: the payload stamp that records the verdict, the machine-A
/// `retry_wait` move on a refusal, and the lease release. Every one of them
/// targets `genesis_candidates`, `genesis_candidate_roots`, or
/// `grouping_leases`. There is no code path from here to `pages`, `page_*`,
/// `entities`, `relations`, `observations`, `edges`, `memories`, or `chunks`,
/// and none to `page_projection_outbox`.
///
/// **E3 rollback** (§5.2) is realised as "evaluate everything before writing
/// anything": a refusal has, at the moment it is known, made zero writes, so
/// there is no partial state to undo. The verdict and the A11 requeue that
/// follow are the *response* to the refusal, not part of the attempted publish.
pub async fn dry_run_finalize(
    tx: &libsql::Transaction,
    candidate_id: &str,
) -> Result<Finalization, WenlanError> {
    let Some(candidate) = candidates::read_candidate(tx, candidate_id).await? else {
        return Ok(Finalization::NotEligible {
            candidate_id: candidate_id.to_string(),
        });
    };
    if candidate.state != CandidateState::Prepared {
        return Ok(Finalization::NotEligible {
            candidate_id: candidate_id.to_string(),
        });
    }
    let Some(envelope) = read_envelope(tx, candidate_id).await? else {
        return Ok(Finalization::NotEligible {
            candidate_id: candidate_id.to_string(),
        });
    };
    if envelope.verdict.is_some() {
        return Ok(Finalization::NotEligible {
            candidate_id: candidate_id.to_string(),
        });
    }

    let verdict = match evaluate_gates(tx, &candidate, &envelope).await? {
        Some(gate) => Verdict::Refused(gate),
        None => Verdict::Passed,
    };

    // Everything below is the *response*. No gate is consulted again, and the
    // response for a refusal writes strictly less than the response for a pass.
    record_envelope(
        tx,
        candidate_id,
        &CasEnvelope {
            verdict: Some(verdict),
            ..envelope
        },
    )
    .await?;

    // The lease is released in both outcomes (C4/C5, token-scoped). A dry run
    // that held its lease to the TTL would block the space for 15 minutes per
    // candidate and turn the shadow into the contention the whole design avoids.
    if let Some(token) = candidate.lease_token.as_deref() {
        leases::release(
            tx,
            LeasePhase::Genesis,
            &candidate.space,
            candidate.input_generation,
            token,
        )
        .await?;
    }

    match verdict {
        Verdict::Passed => Ok(Finalization::DryRunPassed {
            candidate_id: candidate_id.to_string(),
        }),
        Verdict::Refused(gate) => {
            // A11 — requeue with the S0-2 backoff and a charged attempt. Not a
            // park and not terminal: a CAS miss means the world moved.
            candidates::mark_retry_wait(tx, candidate_id, RetryCharge::Charged).await?;
            Ok(Finalization::Refused {
                candidate_id: candidate_id.to_string(),
                gate,
            })
        }
    }
}

/// The eight gates, in order, short-circuiting on the first refusal.
///
/// Returns the first failing gate, or `None` when all eight passed. Read-only:
/// this is the half of finalization that may not write, and keeping it a
/// separate function with no `&mut` anything is what makes that reviewable.
async fn evaluate_gates(
    tx: &libsql::Transaction,
    candidate: &CandidateRow,
    envelope: &CasEnvelope,
) -> Result<Option<CasGate>, WenlanError> {
    for gate in CasGate::ALL {
        let held = match gate {
            CasGate::LeaseOwnership => e1_lease_ownership(tx, candidate).await?,
            CasGate::InputGeneration => e2_input_generation(tx, candidate).await?,
            CasGate::RootDigest => e3_root_digest(tx, candidate).await?,
            CasGate::CoverageEpoch => e4_coverage_epoch(tx, candidate, envelope).await?,
            CasGate::ClaimLiveness => e5_claim_liveness(tx, candidate).await?,
            CasGate::EvidenceLiveness => e6_evidence_liveness(tx, candidate).await?,
            CasGate::CommunityGeneration => {
                e7_community_generation(tx, candidate, envelope).await?
            }
            CasGate::PageTruth => e8_page_truth(tx, candidate).await?,
        };
        if !held {
            return Ok(Some(gate));
        }
    }
    Ok(None)
}

/// **E-1** — the lease token still belongs to this worker.
///
/// Delegated to `leases::owns`, which is also the acquire path's counterpart,
/// so an expired-but-present row reads as *not* owned. A candidate with no
/// stored token cannot own anything and refuses here rather than falling
/// through to gates that would then act on state it does not hold.
async fn e1_lease_ownership(
    tx: &libsql::Transaction,
    candidate: &CandidateRow,
) -> Result<bool, WenlanError> {
    let Some(token) = candidate.lease_token.as_deref() else {
        return Ok(false);
    };
    leases::owns(
        tx,
        LeasePhase::Genesis,
        &candidate.space,
        candidate.input_generation,
        token,
    )
    .await
}

/// **E-2** — `input_generation` unchanged.
///
/// The stored input is the candidate column; the live value is M4's
/// `space_graph_state.grouping_generation`, which is the generation the prepare
/// leased under (fingerprint field 7). A space with no M4 row reads as
/// generation 0, matching what the prepare would have read.
async fn e2_input_generation(
    tx: &libsql::Transaction,
    candidate: &CandidateRow,
) -> Result<bool, WenlanError> {
    Ok(grouping_generation(tx, &candidate.space).await? == candidate.input_generation)
}

/// **E-3** — the active-root-set digest unchanged.
///
/// Recomputes `identity::active_root_digest` over the candidate's *claimed*
/// roots paired with their **current** `provenance_roots.status`, and compares
/// against the stored `active_root_digest`. A root that went inactive, or one
/// that vanished, moves the digest; the stored column is what it is compared
/// to, never a second recomputation.
async fn e3_root_digest(
    tx: &libsql::Transaction,
    candidate: &CandidateRow,
) -> Result<bool, WenlanError> {
    let roots = claimed_root_status(tx, &candidate.candidate_id).await?;
    Ok(identity::active_root_digest(&roots) == candidate.active_root_digest)
}

/// `(root_id, status)` for a proposal's roots, in the shape
/// `identity::active_root_digest` hashes.
///
/// **The prepare and E-3 call this same function**, which is the only reason
/// the two digests are comparable at all: a second spelling of "read the roots
/// and their statuses" would let the observe side and the verify side order,
/// coalesce, or filter differently and silently never agree. A root that has
/// disappeared reads as `missing` rather than vanishing from the vector, so a
/// deletion moves the digest instead of shortening it.
pub async fn root_status(
    tx: &libsql::Transaction,
    root_ids: &[String],
) -> Result<Vec<(String, String)>, WenlanError> {
    let mut pairs = Vec::with_capacity(root_ids.len());
    for root_id in root_ids {
        let mut rows = tx
            .query(
                "SELECT status FROM provenance_roots WHERE root_id = ?1",
                libsql::params![root_id.as_str()],
            )
            .await
            .map_err(|error| WenlanError::VectorDb(format!("m6 root status: {error}")))?;
        let status = rows
            .next()
            .await
            .map_err(|error| WenlanError::VectorDb(format!("m6 root status row: {error}")))?
            .map(|row| row.get::<String>(0))
            .transpose()
            .map_err(|error| WenlanError::VectorDb(format!("m6 root status decode: {error}")))?
            .unwrap_or_else(|| "missing".to_string());
        pairs.push((root_id.clone(), status));
    }
    pairs.sort();
    Ok(pairs)
}

/// [`root_status`] for the roots a candidate actually claimed.
async fn claimed_root_status(
    tx: &libsql::Transaction,
    candidate_id: &str,
) -> Result<Vec<(String, String)>, WenlanError> {
    let mut rows = tx
        .query(
            "SELECT root_id FROM genesis_candidate_roots
              WHERE candidate_id = ?1 ORDER BY root_id",
            libsql::params![candidate_id],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 claimed roots: {error}")))?;
    let mut root_ids = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 claimed roots row: {error}")))?
    {
        root_ids.push(
            row.get::<String>(0).map_err(|error| {
                WenlanError::VectorDb(format!("m6 claimed roots decode: {error}"))
            })?,
        );
    }
    root_status(tx, &root_ids).await
}

/// **E-4** — `coverage_epoch` *and* `epoch_state` unchanged.
///
/// Both halves, because they fail differently. The epoch **number** moving
/// means the identity contract was re-cut under the candidate; the epoch
/// **regime** moving (`active → migrating`) leaves every recomputed value
/// identical while invalidating the regime the worker leased under — the exact
/// `active → draining` shape M5's `finalize_page_support` documents. The regime
/// is compared against the [`CasEnvelope`] stamped at prepare, not against a
/// constant, so a prepare that legitimately ran under `migrating` would be
/// checked against `migrating`.
///
/// A space whose coverage row has disappeared refuses: there is no epoch left
/// to be unchanged.
async fn e4_coverage_epoch(
    tx: &libsql::Transaction,
    candidate: &CandidateRow,
    envelope: &CasEnvelope,
) -> Result<bool, WenlanError> {
    let Some(space_id) = space_id_for(tx, &candidate.space).await? else {
        return Ok(false);
    };
    let Some(state) = candidates::read_coverage_state(tx, &space_id).await? else {
        return Ok(false);
    };
    Ok(state.coverage_epoch == candidate.coverage_epoch
        && state.epoch_state == envelope.epoch_state)
}

/// **E-5** — claims still unreleased, and the witness rows still present.
///
/// Three conditions, all on `genesis_candidate_roots`: the candidate still has
/// at least one claim row, every one of them is unreleased (`released_at IS
/// NULL`, the S0-6/Q8 liveness definition the two partial unique indexes
/// enforce), and every one carries the role `claim_role_for(signal_kind)`
/// dictates. The last is what makes "witness rows still present" a real check
/// for the two overview signals rather than a restatement of the first: a
/// witness-role candidate whose rows had been rewritten to `concept` would have
/// silently acquired an exclusive claim it was never granted.
async fn e5_claim_liveness(
    tx: &libsql::Transaction,
    candidate: &CandidateRow,
) -> Result<bool, WenlanError> {
    let expected = claim_role_for(candidate.signal_kind);
    let mut rows = tx
        .query(
            "SELECT claim_role, released_at
               FROM genesis_candidate_roots
              WHERE candidate_id = ?1",
            libsql::params![candidate.candidate_id.as_str()],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 e5 claim liveness: {error}")))?;
    let mut seen = 0usize;
    while let Some(row) = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 e5 claim liveness row: {error}")))?
    {
        let decode = |error: libsql::Error| WenlanError::VectorDb(format!("m6 e5 decode: {error}"));
        let role: String = row.get(0).map_err(decode)?;
        let released: Option<i64> = row.get(1).map_err(decode)?;
        if released.is_some() || ClaimRole::parse(&role) != Some(expected) {
            return Ok(false);
        }
        seen += 1;
    }
    Ok(seen > 0)
}

/// **E-6** — every root still active, every edge still grounded and live.
///
/// Distinct from E-3 rather than a duplicate of it. E-3 sees a root's *status*
/// move because status is inside the digest; it is blind to an edge that was
/// retracted (`valid_until` set) or un-grounded, because neither touches
/// `provenance_roots`. This gate closes that: each claimed root must still be
/// `active`, non-`generated` (R4 — no self-bootstrapping), and still carry at
/// least one grounded, unretracted edge.
async fn e6_evidence_liveness(
    tx: &libsql::Transaction,
    candidate: &CandidateRow,
) -> Result<bool, WenlanError> {
    let mut rows = tx
        .query(
            "SELECT COUNT(*)
               FROM genesis_candidate_roots gr
              WHERE gr.candidate_id = ?1
                AND (NOT EXISTS (
                         SELECT 1 FROM provenance_roots r
                          WHERE r.root_id   = gr.root_id
                            AND r.status    = 'active'
                            AND r.root_kind <> 'generated'
                     )
                     OR NOT EXISTS (
                         SELECT 1 FROM edges e
                          WHERE e.root_id     = gr.root_id
                            AND e.grounded    = 1
                            AND e.valid_until IS NULL
                     ))",
            libsql::params![candidate.candidate_id.as_str()],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 e6 evidence liveness: {error}")))?;
    Ok(scalar(&mut rows, "m6 e6 evidence liveness").await? == 0)
}

/// **E-7** — M4's community generation still current and durable.
///
/// **Signal-scoped, and the scope is the contract** (S0-137). Only
/// `evidence_cluster` and `community_overview` read the community partition, so
/// only they can be invalidated by it moving; for `space_overview` and
/// `orphan_wikilink` this gate is vacuously true, and saying so explicitly is
/// better than an unconditional check that would refuse every candidate on any
/// install where M4 was never proven. `finalize_test` carries the non-vacuous
/// positive control.
///
/// "Current and durable" is two facts, and E-2 covers neither: the partition
/// must be **durable** per the real M4 gate (`community_gate::partition_is_durable`,
/// never a hardcoded bool and never a copy of the gate SQL), and M4's
/// `published_generation` must still equal the value stamped at prepare with
/// the space not marked `dirty`. E-2's `grouping_generation` is the *lease*
/// generation; this is the *published* one.
async fn e7_community_generation(
    tx: &libsql::Transaction,
    candidate: &CandidateRow,
    envelope: &CasEnvelope,
) -> Result<bool, WenlanError> {
    if !matches!(
        candidate.signal_kind,
        SignalKind::EvidenceCluster | SignalKind::CommunityOverview
    ) {
        return Ok(true);
    }
    if !community_gate::partition_is_durable(tx).await {
        return Ok(false);
    }
    let mut rows = tx
        .query(
            "SELECT published_generation, dirty
               FROM space_graph_state
              WHERE space = ?1",
            libsql::params![candidate.space.as_str()],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 e7 community generation: {error}")))?;
    let Some(row) = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 e7 row: {error}")))?
    else {
        // No M4 row at all: the stored input must have said the same, and a
        // dirty-free comparison against nothing is not "current".
        return Ok(false);
    };
    let decode = |error: libsql::Error| WenlanError::VectorDb(format!("m6 e7 decode: {error}"));
    let published: Option<i64> = row.get(0).map_err(decode)?;
    let dirty: i64 = row.get(1).map_err(decode)?;
    Ok(dirty == 0 && published.is_some() && published == envelope.community_generation)
}

/// **E-8** — M5's page/dependency truth preconditions still hold.
///
/// A claimed root is *page-mediated* when a live `supports` edge on a claim
/// revision that some page version contains names it — the R8 shape
/// `signals::orphan_wikilink` admits under. For every such page, M5's
/// precondition must still hold: the page is `active`, its truth state is
/// `supported`, and that truth state is for the page's **current** version
/// (`page_truth_state.page_version = pages.version` — a page whose derivation
/// lagged its body has stale anchors, and trusting them is the bug M5's own
/// reader closes).
///
/// **Vacuous for a candidate with no page-mediated root** (S0-137): a
/// space-overview candidate built from raw evidence has no M5 dependency, and
/// this gate says so rather than inventing one. The non-vacuous control is an
/// orphan-wikilink candidate whose page loses `supported`.
///
/// Formulated as "count the violations" rather than "count the survivors" so
/// the empty input set produces `0` and passes, instead of producing `0` and
/// failing a `> 0` test — the direction a vacuity bug hides in.
async fn e8_page_truth(
    tx: &libsql::Transaction,
    candidate: &CandidateRow,
) -> Result<bool, WenlanError> {
    let mut rows = tx
        .query(
            "SELECT COUNT(*)
               FROM genesis_candidate_roots gr
               JOIN edges s
                      ON s.edge_type    = 'supports'
                     AND s.src_kind     = 'claim_revision'
                     AND s.root_id      = gr.root_id
                     AND s.valid_until IS NULL
               JOIN page_version_claims pvc ON pvc.claim_revision_id = s.src_id
               JOIN pages p                 ON p.id = pvc.page_id
               LEFT JOIN page_truth_state pts
                      ON pts.page_id      = p.id
                     AND pts.page_version = p.version
              WHERE gr.candidate_id = ?1
                AND (p.status <> 'active'
                     OR pvc.page_version <> p.version
                     OR pts.support_status IS NULL
                     OR pts.support_status <> 'supported')",
            libsql::params![candidate.candidate_id.as_str()],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 e8 page truth: {error}")))?;
    Ok(scalar(&mut rows, "m6 e8 page truth").await? == 0)
}

// ---------------------------------------------------------------------------
// Shared reads
// ---------------------------------------------------------------------------

/// M4's lease/CAS generation for a space — the value a prepare stores as
/// `input_generation`. Absent row reads as 0, matching the prepare's own read.
pub async fn grouping_generation(
    tx: &libsql::Transaction,
    space: &str,
) -> Result<i64, WenlanError> {
    let mut rows = tx
        .query(
            "SELECT grouping_generation FROM space_graph_state WHERE space = ?1",
            libsql::params![space],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 grouping generation: {error}")))?;
    Ok(rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 grouping generation row: {error}")))?
        .map(|row| row.get::<i64>(0))
        .transpose()
        .map_err(|error| WenlanError::VectorDb(format!("m6 grouping generation decode: {error}")))?
        .unwrap_or(0))
}

/// M4's published community generation for a space, for the prepare-time stamp.
pub async fn published_generation(
    tx: &libsql::Transaction,
    space: &str,
) -> Result<Option<i64>, WenlanError> {
    let mut rows = tx
        .query(
            "SELECT published_generation FROM space_graph_state WHERE space = ?1",
            libsql::params![space],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 published generation: {error}")))?;
    Ok(rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 published generation row: {error}")))?
        .map(|row| row.get::<Option<i64>>(0))
        .transpose()
        .map_err(|error| WenlanError::VectorDb(format!("m6 published generation decode: {error}")))?
        .flatten())
}

/// The immutable `spaces.id` behind a renameable space name (Q6). `None` for a
/// space that no longer exists, which every caller treats as a refusal rather
/// than as an error.
pub async fn space_id_for(
    tx: &libsql::Transaction,
    space: &str,
) -> Result<Option<String>, WenlanError> {
    let mut rows = tx
        .query(
            "SELECT id FROM spaces WHERE name = ?1",
            libsql::params![space],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 space id: {error}")))?;
    rows.next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 space id row: {error}")))?
        .map(|row| row.get::<String>(0))
        .transpose()
        .map_err(|error| WenlanError::VectorDb(format!("m6 space id decode: {error}")))
}

async fn scalar(rows: &mut libsql::Rows, label: &str) -> Result<i64, WenlanError> {
    Ok(rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("{label} row: {error}")))?
        .map(|row| row.get::<i64>(0))
        .transpose()
        .map_err(|error| WenlanError::VectorDb(format!("{label} decode: {error}")))?
        .unwrap_or_default())
}
