// SPDX-License-Identifier: Apache-2.0
//! Machines A, B and D, the G3 gate family, and the S0-139 exit matrix.
//!
//! Spec §9.1: "machine A's 19 transitions, machine B's 8, machine D's 6, each
//! as a legal/illegal pair."

use super::candidates::{
    self, claim_role_for, retry_backoff_seconds, retry_exhausted, CandidateState, ClaimRole,
    EpochState, ObserveOutcome, PrepareOutcome, RetryCharge, StaleReason, CANDIDATE_STATES,
    CANDIDATE_TRANSITIONS, EPOCH_TRANSITIONS,
};
use super::constants::{PENDING_CANDIDATES_PER_SPACE_CAP, RETRY_ATTEMPT_LIMIT};
use super::genesis_test_support::{claims, observe, GenesisDb};
use super::identity::SignalKind;
use crate::WenlanError;

const SPACE: &str = "space-a";
const SPACE_ID: &str = "space-id-a";

async fn seeded() -> (GenesisDb, i64) {
    let db = GenesisDb::new().await;
    db.seed_space(SPACE_ID, SPACE).await;
    let epoch = db.open_epoch(SPACE_ID, SPACE).await;
    (db, epoch)
}

/// Drive one candidate from `observed` to `prepared` over the given claims.
async fn prepared(
    db: &GenesisDb,
    candidate_id: &str,
    epoch: i64,
    signal: SignalKind,
    reservations: &[(&str, &str)],
) -> PrepareOutcome {
    observe(db, candidate_id, SPACE, signal, epoch, 1).await;
    let tx = db.tx().await;
    let outcome = candidates::prepare_candidate(&tx, candidate_id, &claims(reservations))
        .await
        .expect("prepare");
    if matches!(outcome, PrepareOutcome::Prepared { .. }) {
        tx.commit().await.expect("commit prepare");
    } else {
        // A refused prepare is atomic: the caller rolls back rather than
        // leaving a partially-claimed candidate behind.
        tx.rollback().await.expect("roll back refused prepare");
    }
    outcome
}

// ---------------------------------------------------------------------------
// Transition tables (machine A's 19, machine D's)
// ---------------------------------------------------------------------------

/// The table is the contract, so the census is over the table: all 19 machine-A
/// transition IDs are present, and A7 is the one that legally spans three source
/// states.
#[tokio::test]
async fn machine_a_declares_every_numbered_transition() {
    let mut ids: Vec<&str> = CANDIDATE_TRANSITIONS.iter().map(|edge| edge.id).collect();
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len(), 19, "machine A has 19 transitions: {ids:?}");
    for number in 1..=19 {
        let id = format!("A{number}");
        assert!(ids.contains(&id.as_str()), "{id} is missing from the table");
    }
    let a7: Vec<_> = CANDIDATE_TRANSITIONS
        .iter()
        .filter(|edge| edge.id == "A7")
        .collect();
    assert_eq!(a7.len(), 3, "A7 spans prepared|inferencing|validating");
    assert!(a7.iter().all(|edge| edge.to == CandidateState::Stale));
}

/// Legal/illegal pairs. Every edge in the table is legal; a representative
/// non-edge from each state is not.
#[tokio::test]
async fn machine_a_refuses_transitions_outside_the_table() {
    for edge in CANDIDATE_TRANSITIONS {
        assert!(
            candidates::transition_is_legal(edge.from, edge.to),
            "{} must be legal",
            edge.id
        );
    }
    for (from, to) in [
        (CandidateState::Observed, CandidateState::Published),
        (CandidateState::Observed, CandidateState::Inferencing),
        (CandidateState::Prepared, CandidateState::Published),
        (CandidateState::Published, CandidateState::Prepared),
        (CandidateState::Suppressed, CandidateState::Prepared),
        (CandidateState::Superseded, CandidateState::Stale),
        (CandidateState::Stale, CandidateState::Published),
    ] {
        assert!(
            !candidates::transition_is_legal(from, to),
            "{} -> {} is not a machine-A transition",
            from.as_str(),
            to.as_str()
        );
    }
}

/// D3's ten states, and S0-12's rule that exhausted retry is not an eleventh.
#[tokio::test]
async fn machine_a_has_exactly_ten_states() {
    assert_eq!(CANDIDATE_STATES.len(), 10);
    let mut names: Vec<&str> = CANDIDATE_STATES.iter().map(|s| s.as_str()).collect();
    names.sort();
    names.dedup();
    assert_eq!(names.len(), 10);
    assert!(
        !names.contains(&"retry_exhausted"),
        "S0-12: exhausted retry is `stale` carrying a reason, never a state"
    );
    for state in CANDIDATE_STATES {
        assert_eq!(CandidateState::parse(state.as_str()), Some(state));
    }
    assert_eq!(CandidateState::parse("nonsense"), None);
}

#[tokio::test]
async fn machine_d_declares_its_epoch_edges() {
    assert_eq!(EPOCH_TRANSITIONS.len(), 3);
    assert!(candidates::epoch_transition_is_legal(
        EpochState::Active,
        EpochState::Migrating
    ));
    assert!(candidates::epoch_transition_is_legal(
        EpochState::Migrating,
        EpochState::Active
    ));
    assert!(
        !candidates::epoch_transition_is_legal(EpochState::Active, EpochState::Active),
        "an epoch does not re-enter `active` from `active`"
    );
}

// ---------------------------------------------------------------------------
// Machine D — the coverage epoch
// ---------------------------------------------------------------------------

/// D1 is idempotent, and (Q11) legal while `genesis_enabled = 0`.
#[tokio::test]
async fn opening_a_coverage_epoch_is_idempotent_and_leaves_the_flag_alone() {
    let db = GenesisDb::new().await;
    db.seed_space(SPACE_ID, SPACE).await;
    let first = db.open_epoch(SPACE_ID, SPACE).await;
    let second = db.open_epoch(SPACE_ID, SPACE).await;
    assert_eq!(first, 1);
    assert_eq!(second, 1);
    assert_eq!(
        db.scalar("SELECT COUNT(*) FROM genesis_coverage_state", ())
            .await,
        1
    );
    assert_eq!(
        db.scalar("SELECT genesis_enabled FROM genesis_coverage_state", ())
            .await,
        0,
        "PR-B creates rows and never writes the flag"
    );
}

/// D2 → D3: the epoch increments and never decreases (I-7).
#[tokio::test]
async fn a_completed_migration_advances_the_epoch() {
    let (db, epoch) = seeded().await;
    let tx = db.tx().await;
    candidates::begin_epoch_migration(&tx, SPACE_ID)
        .await
        .unwrap();
    let state = candidates::read_coverage_state(&tx, SPACE_ID)
        .await
        .unwrap()
        .expect("coverage row");
    assert_eq!(state.epoch_state, EpochState::Migrating);
    let advanced = candidates::complete_epoch_migration(&tx, SPACE_ID)
        .await
        .unwrap();
    tx.commit().await.unwrap();
    assert_eq!(advanced, epoch + 1);
}

/// D4 — abandonment is free: the epoch is unchanged.
#[tokio::test]
async fn an_abandoned_migration_leaves_the_epoch_where_it_was() {
    let (db, epoch) = seeded().await;
    let tx = db.tx().await;
    candidates::begin_epoch_migration(&tx, SPACE_ID)
        .await
        .unwrap();
    candidates::abandon_epoch_migration(&tx, SPACE_ID)
        .await
        .unwrap();
    let state = candidates::read_coverage_state(&tx, SPACE_ID)
        .await
        .unwrap()
        .expect("coverage row");
    tx.commit().await.unwrap();
    assert_eq!(state.epoch_state, EpochState::Active);
    assert_eq!(state.coverage_epoch, epoch);
}

/// Illegal machine-D pair: completing a migration that never began.
#[tokio::test]
async fn completing_a_migration_that_never_began_refuses() {
    let (db, _) = seeded().await;
    let tx = db.tx().await;
    let error = candidates::complete_epoch_migration(&tx, SPACE_ID)
        .await
        .unwrap_err();
    assert!(matches!(error, WenlanError::Conflict(_)), "{error:?}");
}

/// A space with no coverage row is genesis-disabled by the same contract the
/// `DEFAULT 0` flag is (Q11). Vacuous case for machine D.
#[tokio::test]
async fn a_space_with_no_row_reads_as_absent() {
    let db = GenesisDb::new().await;
    let tx = db.tx().await;
    assert_eq!(
        candidates::read_coverage_state(&tx, "space-id-unknown")
            .await
            .unwrap(),
        None
    );
}

// ---------------------------------------------------------------------------
// Machine A — observation and the D8 pending cap
// ---------------------------------------------------------------------------

#[tokio::test]
async fn observing_the_same_candidate_twice_reuses_the_row() {
    let (db, epoch) = seeded().await;
    observe(&db, "cand-a", SPACE, SignalKind::EvidenceCluster, epoch, 1).await;
    let tx = db.tx().await;
    let outcome = candidates::observe_candidate(
        &tx,
        &candidates::ObservedCandidate {
            candidate_id: "cand-a",
            slot_id: "slot-cand-a",
            page_id: "m6p_cand-a",
            space: SPACE,
            signal_kind: SignalKind::EvidenceCluster,
            coverage_epoch: epoch,
            input_generation: 1,
            active_root_digest: "digest-0",
        },
    )
    .await
    .unwrap();
    tx.commit().await.unwrap();
    assert_eq!(outcome, ObserveOutcome::AlreadyPresent);
    assert_eq!(
        db.scalar("SELECT COUNT(*) FROM genesis_candidates", ())
            .await,
        1,
        "I-8: a retry reuses the candidate, it never mints a second one"
    );
}

/// S0-54 — a cap refuses to start work and **writes nothing**. It may not
/// terminalize a candidate or retire a frontier row.
#[tokio::test]
async fn the_pending_cap_refuses_without_writing() {
    let (db, epoch) = seeded().await;
    for index in 0..PENDING_CANDIDATES_PER_SPACE_CAP {
        observe(
            &db,
            &format!("cand-{index}"),
            SPACE,
            SignalKind::EvidenceCluster,
            epoch,
            1,
        )
        .await;
    }
    let before = db
        .scalar("SELECT COUNT(*) FROM genesis_candidates", ())
        .await;

    let tx = db.tx().await;
    let outcome = candidates::observe_candidate(
        &tx,
        &candidates::ObservedCandidate {
            candidate_id: "cand-overflow",
            slot_id: "slot-overflow",
            page_id: "m6p_overflow",
            space: SPACE,
            signal_kind: SignalKind::EvidenceCluster,
            coverage_epoch: epoch,
            input_generation: 1,
            active_root_digest: "digest-0",
        },
    )
    .await
    .unwrap();
    tx.commit().await.unwrap();

    assert_eq!(outcome, ObserveOutcome::RefusedPendingCap);
    assert_eq!(
        db.scalar("SELECT COUNT(*) FROM genesis_candidates", ())
            .await,
        before,
        "the refusal wrote nothing"
    );
}

// ---------------------------------------------------------------------------
// Machine B — G3.1 / G3.2 / G3.3 / G3.4
// ---------------------------------------------------------------------------

/// G3.1 — `idx_genesis_root_claim`. Two candidates cannot both claim one root.
/// This exercises the **shipped** index, installed by the real migration.
#[tokio::test]
async fn two_candidates_cannot_both_claim_one_root() {
    let (db, epoch) = seeded().await;
    let first = prepared(
        &db,
        "cand-a",
        epoch,
        SignalKind::EvidenceCluster,
        &[("root-1", "group-1")],
    )
    .await;
    assert!(matches!(first, PrepareOutcome::Prepared { .. }));
    db.release_leases().await;

    // A different group, so only the root-claim index can refuse this.
    let second = prepared(
        &db,
        "cand-b",
        epoch,
        SignalKind::EvidenceCluster,
        &[("root-1", "group-2")],
    )
    .await;
    assert_eq!(
        second,
        PrepareOutcome::ClaimRefused {
            root_id: "root-1".to_string()
        }
    );
    assert_eq!(
        db.scalar(
            "SELECT COUNT(*) FROM genesis_candidate_roots WHERE released_at IS NULL",
            ()
        )
        .await,
        1
    );
}

/// G3.2 — `idx_genesis_group_claim`. Two candidates cannot both claim one
/// group, even over different roots. Overlapping candidates publish once.
#[tokio::test]
async fn two_candidates_cannot_both_claim_one_group() {
    let (db, epoch) = seeded().await;
    let first = prepared(
        &db,
        "cand-a",
        epoch,
        SignalKind::EvidenceCluster,
        &[("root-1", "group-1")],
    )
    .await;
    assert!(matches!(first, PrepareOutcome::Prepared { .. }));
    db.release_leases().await;

    let second = prepared(
        &db,
        "cand-b",
        epoch,
        SignalKind::EvidenceCluster,
        &[("root-2", "group-1")],
    )
    .await;
    assert_eq!(
        second,
        PrepareOutcome::ClaimRefused {
            root_id: "root-2".to_string()
        }
    );
}

/// G3.3 — a witness row is never readable as coverage (I-3). Both partial
/// indexes exclude witnesses by predicate, so two overview candidates may share
/// a group where two concept candidates may not.
#[tokio::test]
async fn witness_rows_do_not_exclude_each_other() {
    let (db, epoch) = seeded().await;
    assert_eq!(
        claim_role_for(SignalKind::CommunityOverview),
        ClaimRole::Witness
    );
    assert_eq!(
        claim_role_for(SignalKind::EvidenceCluster),
        ClaimRole::Concept
    );

    let first = prepared(
        &db,
        "cand-a",
        epoch,
        SignalKind::CommunityOverview,
        &[("root-1", "group-1")],
    )
    .await;
    db.release_leases().await;
    let second = prepared(
        &db,
        "cand-b",
        epoch,
        SignalKind::SpaceOverview,
        &[("root-2", "group-1")],
    )
    .await;
    assert!(matches!(first, PrepareOutcome::Prepared { .. }));
    assert!(
        matches!(second, PrepareOutcome::Prepared { .. }),
        "witness claims over one group coexist: {second:?}"
    );
    assert_eq!(
        db.scalar(
            "SELECT COUNT(*) FROM genesis_candidate_roots
              WHERE claim_role = 'witness' AND released_at IS NULL",
            ()
        )
        .await,
        2
    );
}

/// G3.4 — insert-and-refuse, not pre-check-then-insert. The refusal comes from
/// the database, so a concurrent reservation still yields exactly one winner
/// even though no code inspected the table first.
#[tokio::test]
async fn a_released_claim_frees_the_group_for_a_new_winner() {
    let (db, epoch) = seeded().await;
    prepared(
        &db,
        "cand-a",
        epoch,
        SignalKind::EvidenceCluster,
        &[("root-1", "group-1")],
    )
    .await;

    let tx = db.tx().await;
    candidates::mark_stale(&tx, "cand-a", StaleReason::InputMoved)
        .await
        .unwrap();
    tx.commit().await.unwrap();
    db.release_leases().await;

    let second = prepared(
        &db,
        "cand-b",
        epoch,
        SignalKind::EvidenceCluster,
        &[("root-1", "group-1")],
    )
    .await;
    assert!(
        matches!(second, PrepareOutcome::Prepared { .. }),
        "a released claim no longer excludes: {second:?}"
    );
    assert_eq!(
        db.scalar(
            "SELECT COUNT(*) FROM genesis_candidate_roots WHERE released_at IS NOT NULL",
            ()
        )
        .await,
        1,
        "G3.5: the released row is retained, not deleted"
    );
}

/// C2 through the prepare path: a candidate whose space and generation are
/// already leased observes the holder and does no work.
#[tokio::test]
async fn a_held_lease_stops_a_prepare() {
    let (db, epoch) = seeded().await;
    prepared(
        &db,
        "cand-a",
        epoch,
        SignalKind::EvidenceCluster,
        &[("root-1", "group-1")],
    )
    .await;
    let outcome = prepared(
        &db,
        "cand-b",
        epoch,
        SignalKind::EvidenceCluster,
        &[("root-2", "group-2")],
    )
    .await;
    assert_eq!(outcome, PrepareOutcome::LeaseHeld);
    assert_eq!(db.candidate_state("cand-b").await, "observed");
}

/// F2 — a concept claim retires the group's frontier row in the same
/// transaction, so the group never holds two live reasons.
#[tokio::test]
async fn a_concept_claim_retires_the_groups_frontier_row() {
    let (db, epoch) = seeded().await;
    db.exec(
        "INSERT INTO genesis_frontier
             (space, independence_group_id, coverage_epoch, first_seen_at, next_scan_at)
         VALUES (?1, 'group-1', ?2, unixepoch(), unixepoch())",
        libsql::params![SPACE, epoch],
    )
    .await;
    prepared(
        &db,
        "cand-a",
        epoch,
        SignalKind::EvidenceCluster,
        &[("root-1", "group-1")],
    )
    .await;
    assert_eq!(
        db.scalar("SELECT COUNT(*) FROM genesis_frontier", ()).await,
        0
    );
}

// ---------------------------------------------------------------------------
// Retry, backoff and staleness
// ---------------------------------------------------------------------------

/// S0-2's sequence exactly: 60s, 4m, 16m, 64m, then the 4h ceiling. No jitter.
#[tokio::test]
async fn the_backoff_sequence_is_the_stage_zero_one() {
    assert_eq!(retry_backoff_seconds(0), 0);
    assert_eq!(retry_backoff_seconds(1), 60);
    assert_eq!(retry_backoff_seconds(2), 240);
    assert_eq!(retry_backoff_seconds(3), 960);
    assert_eq!(retry_backoff_seconds(4), 3_840);
    assert_eq!(retry_backoff_seconds(5), 14_400);
    assert_eq!(retry_backoff_seconds(6), 14_400, "capped at 4h");
    assert_eq!(retry_backoff_seconds(60), 14_400, "and stays capped");
    assert!(!retry_exhausted(RETRY_ATTEMPT_LIMIT));
    assert!(retry_exhausted(RETRY_ATTEMPT_LIMIT + 1));
}

/// S0-9 — a budget refusal does not charge `attempt`. A busy space would
/// otherwise burn its five-attempt budget on turns where it never reached the
/// model, converting a scheduling artefact into a permanent `stale`.
#[tokio::test]
async fn a_budget_refusal_does_not_charge_an_attempt() {
    let (db, epoch) = seeded().await;
    prepared(
        &db,
        "cand-a",
        epoch,
        SignalKind::EvidenceCluster,
        &[("root-1", "group-1")],
    )
    .await;

    let tx = db.tx().await;
    candidates::mark_retry_wait(&tx, "cand-a", RetryCharge::BudgetRefusal)
        .await
        .unwrap();
    tx.commit().await.unwrap();
    assert_eq!(
        db.scalar(
            "SELECT attempt FROM genesis_candidates WHERE candidate_id = 'cand-a'",
            ()
        )
        .await,
        0
    );

    let tx = db.tx().await;
    candidates::mark_retry_wait(&tx, "cand-a", RetryCharge::Charged)
        .await
        .unwrap_err();
    drop(tx);
}

/// A9/A11 — a model failure charges the attempt and applies the backoff.
#[tokio::test]
async fn a_model_failure_charges_the_attempt_and_defers() {
    let (db, epoch) = seeded().await;
    prepared(
        &db,
        "cand-a",
        epoch,
        SignalKind::EvidenceCluster,
        &[("root-1", "group-1")],
    )
    .await;
    let tx = db.tx().await;
    candidates::mark_inferencing(&tx, "cand-a").await.unwrap();
    candidates::mark_retry_wait(&tx, "cand-a", RetryCharge::Charged)
        .await
        .unwrap();
    tx.commit().await.unwrap();

    assert_eq!(db.candidate_state("cand-a").await, "retry_wait");
    assert_eq!(
        db.scalar(
            "SELECT attempt FROM genesis_candidates WHERE candidate_id = 'cand-a'",
            ()
        )
        .await,
        1
    );
    assert_eq!(
        db.scalar(
            "SELECT next_attempt_at - unixepoch() FROM genesis_candidates
              WHERE candidate_id = 'cand-a'",
            ()
        )
        .await,
        60
    );
}

/// A13 — the retry is not due until its stored backoff elapses, and "elapses"
/// means the stored value moved (S0-11).
#[tokio::test]
async fn a_retry_before_its_backoff_is_not_due() {
    let (db, epoch) = seeded().await;
    prepared(
        &db,
        "cand-a",
        epoch,
        SignalKind::EvidenceCluster,
        &[("root-1", "group-1")],
    )
    .await;
    let tx = db.tx().await;
    candidates::mark_retry_wait(&tx, "cand-a", RetryCharge::Charged)
        .await
        .unwrap();
    tx.commit().await.unwrap();
    db.release_leases().await;

    let tx = db.tx().await;
    let early = candidates::prepare_candidate(&tx, "cand-a", &claims(&[("root-1", "group-1")]))
        .await
        .unwrap();
    tx.commit().await.unwrap();
    assert_eq!(early, PrepareOutcome::NotDue);

    db.age_candidate("cand-a", 120).await;
    let tx = db.tx().await;
    let due = candidates::prepare_candidate(&tx, "cand-a", &claims(&[("root-1", "group-1")]))
        .await
        .unwrap();
    tx.commit().await.unwrap();
    assert!(matches!(due, PrepareOutcome::Prepared { .. }), "{due:?}");
}

/// A19 + S0-151 — `stale` is re-enterable, and the re-entry delay depends on
/// the reason. `input_moved` wasted nothing, so it re-enters at once.
#[tokio::test]
async fn stale_re_entry_delay_depends_on_the_reason() {
    assert_eq!(StaleReason::InputMoved.re_entry_delay_seconds(3), 0);
    assert_eq!(StaleReason::LeaseLost.re_entry_delay_seconds(2), 240);
    assert_eq!(
        StaleReason::RetryExhausted.re_entry_delay_seconds(6),
        86_400
    );
    assert_eq!(
        StaleReason::CardExpired.re_entry_delay_seconds(0),
        180 * 86_400
    );
}

/// A19 — re-preparing from `stale` resets `attempt`, because the reason the
/// candidate went stale is not a model failure to charge for.
#[tokio::test]
async fn re_preparing_from_stale_resets_the_attempt() {
    let (db, epoch) = seeded().await;
    prepared(
        &db,
        "cand-a",
        epoch,
        SignalKind::EvidenceCluster,
        &[("root-1", "group-1")],
    )
    .await;
    let tx = db.tx().await;
    candidates::mark_retry_wait(&tx, "cand-a", RetryCharge::Charged)
        .await
        .unwrap();
    candidates::mark_stale(&tx, "cand-a", StaleReason::InputMoved)
        .await
        .unwrap();
    tx.commit().await.unwrap();
    db.release_leases().await;

    let tx = db.tx().await;
    let outcome = candidates::prepare_candidate(&tx, "cand-a", &claims(&[("root-1", "group-1")]))
        .await
        .unwrap();
    tx.commit().await.unwrap();
    assert!(matches!(outcome, PrepareOutcome::Prepared { .. }));
    assert_eq!(
        db.scalar(
            "SELECT attempt FROM genesis_candidates WHERE candidate_id = 'cand-a'",
            ()
        )
        .await,
        0
    );
}

/// S0-12 — exhaustion is `stale` plus a distinguishable reason.
#[tokio::test]
async fn exhausted_retry_is_stale_with_a_reason() {
    let (db, epoch) = seeded().await;
    prepared(
        &db,
        "cand-a",
        epoch,
        SignalKind::EvidenceCluster,
        &[("root-1", "group-1")],
    )
    .await;
    let tx = db.tx().await;
    candidates::mark_retry_wait(&tx, "cand-a", RetryCharge::Charged)
        .await
        .unwrap();
    tx.commit().await.unwrap();
    db.exec(
        "UPDATE genesis_candidates SET attempt = ?1 WHERE candidate_id = 'cand-a'",
        libsql::params![RETRY_ATTEMPT_LIMIT + 1],
    )
    .await;

    let tx = db.tx().await;
    candidates::mark_stale(&tx, "cand-a", StaleReason::RetryExhausted)
        .await
        .unwrap();
    tx.commit().await.unwrap();

    assert_eq!(db.candidate_state("cand-a").await, "stale");
    assert_eq!(
        db.candidate_reason("cand-a").await.as_deref(),
        Some("retry_exhausted")
    );
}

/// D4 atomicity — going stale releases the reservation *and* returns the group
/// to the frontier in one transaction. A split would leave the group with zero
/// durable reasons, which is exactly what I-1 forbids.
#[tokio::test]
async fn going_stale_releases_the_claim_and_restores_the_frontier() {
    let (db, epoch) = seeded().await;
    prepared(
        &db,
        "cand-a",
        epoch,
        SignalKind::EvidenceCluster,
        &[("root-1", "group-1")],
    )
    .await;
    let tx = db.tx().await;
    candidates::mark_stale(&tx, "cand-a", StaleReason::LeaseLost)
        .await
        .unwrap();
    tx.commit().await.unwrap();

    assert_eq!(
        db.scalar(
            "SELECT COUNT(*) FROM genesis_candidate_roots WHERE released_at IS NULL",
            ()
        )
        .await,
        0
    );
    assert_eq!(
        db.scalar(
            "SELECT COUNT(*) FROM genesis_frontier WHERE independence_group_id = 'group-1'",
            ()
        )
        .await,
        1
    );
}

// ---------------------------------------------------------------------------
// Compaction — G5.7 / S0-53
// ---------------------------------------------------------------------------

/// Compaction nulls the payload and stamps the marker. It never deletes the
/// row: deleting it would drop the group's accounting to `reason_count = 0`
/// and the reconciler would faithfully re-discover a group already handled — an
/// infinite rediscovery loop caused by a retention policy.
#[tokio::test]
async fn compaction_nulls_the_payload_and_keeps_the_row() {
    let (db, epoch) = seeded().await;
    prepared(
        &db,
        "cand-a",
        epoch,
        SignalKind::EvidenceCluster,
        &[("root-1", "group-1")],
    )
    .await;
    let tx = db.tx().await;
    candidates::mark_stale(&tx, "cand-a", StaleReason::InputMoved)
        .await
        .unwrap();
    tx.commit().await.unwrap();
    db.exec(
        "UPDATE genesis_candidates SET payload = 'inference blob' WHERE candidate_id = 'cand-a'",
        (),
    )
    .await;
    db.age_candidate("cand-a", 91 * 86_400).await;

    let tx = db.tx().await;
    let compacted = candidates::compact_terminal_payloads(&tx, SPACE, 90 * 86_400)
        .await
        .unwrap();
    tx.commit().await.unwrap();

    assert_eq!(compacted, 1);
    assert_eq!(
        db.scalar("SELECT COUNT(*) FROM genesis_candidates", ())
            .await,
        1,
        "S0-53: the row survives"
    );
    assert_eq!(
        db.scalar(
            "SELECT payload IS NULL AND payload_compacted_at IS NOT NULL
               FROM genesis_candidates WHERE candidate_id = 'cand-a'",
            ()
        )
        .await,
        1,
        "compacted and never-had-a-payload stay distinguishable"
    );
}

/// Vacuous case: a space with no terminal candidates compacts nothing.
#[tokio::test]
async fn compaction_is_zero_on_an_empty_space() {
    let (db, _) = seeded().await;
    let tx = db.tx().await;
    assert_eq!(
        candidates::compact_terminal_payloads(&tx, SPACE, 90 * 86_400)
            .await
            .unwrap(),
        0
    );
}

// ---------------------------------------------------------------------------
// The reservation exit matrix (S0-139)
// ---------------------------------------------------------------------------

/// The six legal resting states, in artifact 5 §1.1's order.
const RESTING_STATES: [&str; 6] = [
    "coverage",
    "reservation",
    "frontier",
    "surfaced_card",
    "suppression",
    "quarantine",
];

/// The eight terminal transitions of the reservation exit matrix, each mapped
/// to the one resting state it must leave the group in.
///
/// `published` is driven by a direct coverage-row write rather than by machine
/// E: `finalize.rs` is PR-B3's, and asserting an incomplete matrix now would be
/// worse than standing the one cell up explicitly. PR-B3 replaces the stand-in
/// with the real finalize; the expected cell does not change.
const EXIT_MATRIX: [(&str, &str); 8] = [
    ("published", "coverage"),
    ("bounded_retry", "reservation"),
    ("review_required", "surfaced_card"),
    ("stale", "frontier"),
    ("suppressed", "suppression"),
    ("superseded", "frontier"),
    ("exhausted_retry", "frontier"),
    ("compaction", "coverage"),
];

/// Which of the six homes hold a live row for this group.
async fn live_homes(db: &GenesisDb, epoch: i64) -> Vec<&'static str> {
    let group = "group-1";
    let mut homes = Vec::new();
    let probes: [(&'static str, &str); 6] = [
        (
            "coverage",
            "SELECT COUNT(*) FROM genesis_group_coverage
              WHERE space = ?1 AND independence_group_id = ?2 AND coverage_epoch = ?3",
        ),
        (
            "reservation",
            "SELECT COUNT(*) FROM genesis_candidate_roots gr
               JOIN genesis_candidates c ON c.candidate_id = gr.candidate_id
              WHERE c.space = ?1 AND gr.independence_group_id = ?2
                AND gr.coverage_epoch = ?3 AND gr.released_at IS NULL
                AND gr.claim_role = 'concept'",
        ),
        (
            "frontier",
            "SELECT COUNT(*) FROM genesis_frontier
              WHERE space = ?1 AND independence_group_id = ?2 AND coverage_epoch = ?3",
        ),
        (
            "surfaced_card",
            "SELECT COUNT(*) FROM genesis_card_binding
              WHERE space = ?1 AND independence_group_id = ?2 AND coverage_epoch = ?3
                AND closed_at IS NULL",
        ),
        (
            "suppression",
            "SELECT COUNT(*) FROM genesis_suppression
              WHERE space = ?1 AND independence_group_id = ?2 AND coverage_epoch = ?3
                AND lapsed_at IS NULL",
        ),
        (
            "quarantine",
            "SELECT COUNT(*) FROM genesis_quarantine
              WHERE space = ?1 AND independence_group_id = ?2 AND coverage_epoch = ?3
                AND lifted_at IS NULL",
        ),
    ];
    for (name, sql) in probes {
        if db.scalar(sql, libsql::params![SPACE, group, epoch]).await > 0 {
            homes.push(name);
        }
    }
    homes
}

/// Drive one exit and return the group's live homes afterwards.
async fn drive_exit(exit: &str) -> Vec<&'static str> {
    let (db, epoch) = seeded().await;
    let reservations = [("root-1", "group-1")];
    let signal = SignalKind::EvidenceCluster;

    match exit {
        "published" | "compaction" => {
            prepared(&db, "cand-a", epoch, signal, &reservations).await;
            // Stand-in for machine E: coverage is permanent within an epoch
            // (I-7), and the claim is consumed by the publication.
            db.exec(
                "UPDATE genesis_candidate_roots
                    SET released_at = unixepoch(), released_reason = 'published', consumed = 1",
                (),
            )
            .await;
            db.exec(
                "INSERT INTO genesis_group_coverage
                     (space, independence_group_id, coverage_epoch, page_id,
                      candidate_id, covered_at)
                 VALUES (?1, 'group-1', ?2, 'm6p_cand-a', 'cand-a', unixepoch())",
                libsql::params![SPACE, epoch],
            )
            .await;
            db.exec(
                "UPDATE genesis_candidates SET state = 'published', payload = 'blob'",
                (),
            )
            .await;
            if exit == "compaction" {
                db.age_candidate("cand-a", 91 * 86_400).await;
                let tx = db.tx().await;
                candidates::compact_terminal_payloads(&tx, SPACE, 90 * 86_400)
                    .await
                    .unwrap();
                tx.commit().await.unwrap();
            }
        }
        "bounded_retry" => {
            prepared(&db, "cand-a", epoch, signal, &reservations).await;
            let tx = db.tx().await;
            candidates::mark_retry_wait(&tx, "cand-a", RetryCharge::Charged)
                .await
                .unwrap();
            tx.commit().await.unwrap();
        }
        "review_required" => {
            prepared(&db, "cand-a", epoch, signal, &reservations).await;
            let tx = db.tx().await;
            candidates::mark_inferencing(&tx, "cand-a").await.unwrap();
            candidates::mark_validating(&tx, "cand-a").await.unwrap();
            candidates::mark_review_required(&tx, "cand-a", "card-1")
                .await
                .unwrap();
            tx.commit().await.unwrap();
        }
        "stale" => {
            prepared(&db, "cand-a", epoch, signal, &reservations).await;
            let tx = db.tx().await;
            candidates::mark_stale(&tx, "cand-a", StaleReason::LeaseLost)
                .await
                .unwrap();
            tx.commit().await.unwrap();
        }
        "suppressed" => {
            prepared(&db, "cand-a", epoch, signal, &reservations).await;
            let tx = db.tx().await;
            candidates::mark_inferencing(&tx, "cand-a").await.unwrap();
            candidates::mark_validating(&tx, "cand-a").await.unwrap();
            candidates::mark_review_required(&tx, "cand-a", "card-1")
                .await
                .unwrap();
            candidates::mark_suppressed(&tx, "cand-a", "card-1", "user_dismissed")
                .await
                .unwrap();
            tx.commit().await.unwrap();
        }
        "superseded" => {
            prepared(&db, "cand-a", epoch, signal, &reservations).await;
            let tx = db.tx().await;
            candidates::mark_retry_wait(&tx, "cand-a", RetryCharge::Charged)
                .await
                .unwrap();
            candidates::mark_superseded(&tx, "cand-a").await.unwrap();
            tx.commit().await.unwrap();
        }
        "exhausted_retry" => {
            prepared(&db, "cand-a", epoch, signal, &reservations).await;
            let tx = db.tx().await;
            candidates::mark_retry_wait(&tx, "cand-a", RetryCharge::Charged)
                .await
                .unwrap();
            tx.commit().await.unwrap();
            db.exec(
                "UPDATE genesis_candidates SET attempt = ?1",
                libsql::params![RETRY_ATTEMPT_LIMIT + 1],
            )
            .await;
            let tx = db.tx().await;
            candidates::mark_stale(&tx, "cand-a", StaleReason::RetryExhausted)
                .await
                .unwrap();
            tx.commit().await.unwrap();
        }
        other => panic!("unknown exit {other}"),
    }
    live_homes(&db, epoch).await
}

/// S0-139 — 8 exits × 6 resting states as a **total function**. All 48 cells
/// are enumerated: a missing cell fails, an extra cell fails, and every exit
/// leaves the group in exactly one home. Asserting per-exit "something
/// reasonable happened" would pass while some *other* exit dropped a group out
/// of all six.
#[tokio::test]
async fn the_reservation_exit_matrix_is_a_total_function() {
    let mut cells = 0usize;
    for (exit, expected) in EXIT_MATRIX {
        assert!(
            RESTING_STATES.contains(&expected),
            "{exit} maps to {expected}, which is not one of the six"
        );
        let homes = drive_exit(exit).await;
        assert_eq!(
            homes.len(),
            1,
            "exit {exit} left the group in {homes:?}, not in exactly one home"
        );
        for state in RESTING_STATES {
            let occupied = homes.contains(&state);
            assert_eq!(
                occupied,
                state == expected,
                "exit {exit}, resting state {state}: expected {}, observed {occupied}",
                state == expected
            );
            cells += 1;
        }
    }
    assert_eq!(cells, 48, "8 exits x 6 resting states");
}

/// The matrix's own shape: eight distinct exits, no duplicates, no extras.
#[tokio::test]
async fn the_exit_matrix_names_eight_distinct_exits() {
    let mut exits: Vec<&str> = EXIT_MATRIX.iter().map(|(exit, _)| *exit).collect();
    exits.sort();
    exits.dedup();
    assert_eq!(exits.len(), 8);
    assert_eq!(RESTING_STATES.len(), 6);
}
