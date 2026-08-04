// SPDX-License-Identifier: Apache-2.0
//! The startup recovery scan (spec §4.4) and restart-resumability (§4.5).
//!
//! Every case here kills the process by *not* running it: the fixture writes the
//! durable state a crashed turn would have left behind, then calls `scan` as a
//! restarted daemon would. That is the only crash model a hermetic test can
//! honestly claim — and it is the one that matters, because durable state is all
//! a restart ever sees.

use super::candidates::{self, CandidateState, PrepareOutcome, StaleReason};
use super::frontier;
use super::genesis_test_support::{claims, observe, GenesisDb};
use super::identity::SignalKind;
use super::leases::{self, LeasePhase};
use super::recovery;

const SPACE: &str = "space-a";
const SPACE_ID: &str = "space-id-a";

/// One space, one eligible group reconciled into the frontier, and a candidate
/// left mid-flight in `state` holding its reservation.
async fn crashed_mid_flight(state: CandidateState) -> (GenesisDb, i64) {
    let db = GenesisDb::new().await;
    db.seed_space(SPACE_ID, SPACE).await;
    let epoch = db.open_epoch(SPACE_ID, SPACE).await;
    db.seed_evidence("root-0", SPACE, "group-1", "entity-0")
        .await;
    let tx = db.tx().await;
    frontier::reconcile_space(&tx, SPACE, epoch).await.unwrap();
    tx.commit().await.unwrap();

    observe(&db, "cand-a", SPACE, SignalKind::EvidenceCluster, epoch, 1).await;
    let tx = db.tx().await;
    let outcome = candidates::prepare_candidate(&tx, "cand-a", &claims(&[("root-0", "group-1")]))
        .await
        .unwrap();
    assert!(matches!(outcome, PrepareOutcome::Prepared { .. }));
    match state {
        CandidateState::Prepared => {}
        CandidateState::Inferencing => {
            candidates::mark_inferencing(&tx, "cand-a").await.unwrap();
        }
        CandidateState::Validating => {
            candidates::mark_inferencing(&tx, "cand-a").await.unwrap();
            candidates::mark_validating(&tx, "cand-a").await.unwrap();
        }
        CandidateState::RetryWait => {
            candidates::mark_inferencing(&tx, "cand-a").await.unwrap();
            candidates::mark_retry_wait(&tx, "cand-a", candidates::RetryCharge::Charged)
                .await
                .unwrap();
        }
        other => panic!("not a mid-flight state: {other:?}"),
    }
    tx.commit().await.unwrap();
    assert_eq!(
        db.scalar("SELECT COUNT(*) FROM genesis_frontier", ()).await,
        0,
        "the claim retired the frontier row, which is what recovery must undo"
    );
    (db, epoch)
}

/// The crash model itself: an expired lease plus an unreleased reservation is a
/// candidate no live process owns. Recovery reaps the lease, releases the
/// reservation, returns the group to the frontier, and lands the candidate in
/// `stale(lease_lost)` — one transaction, so the group is never reasonless.
#[tokio::test]
async fn a_crashed_candidate_recovers_to_stale_with_its_group_back() {
    for state in [
        CandidateState::Prepared,
        CandidateState::Inferencing,
        CandidateState::Validating,
        CandidateState::RetryWait,
    ] {
        let (db, epoch) = crashed_mid_flight(state).await;
        db.exec(
            "UPDATE grouping_leases SET expires_at = unixepoch() - 1",
            (),
        )
        .await;

        let report = recovery::scan(&db.connection).await.unwrap();

        assert_eq!(report.leases_reaped, 1, "from {state:?}");
        assert_eq!(report.candidates_staled, vec!["cand-a".to_string()]);
        assert_eq!(report.candidates_lease_live, 0);
        assert_eq!(db.candidate_state("cand-a").await, "stale");
        assert_eq!(
            db.candidate_reason("cand-a").await.as_deref(),
            Some("lease_lost")
        );
        assert_eq!(
            db.scalar("SELECT COUNT(*) FROM genesis_frontier", ()).await,
            1,
            "the group is accounted for again after recovering from {state:?}"
        );
        let tx = db.tx().await;
        let counts = frontier::reason_counts(&tx, SPACE, epoch).await.unwrap();
        assert_eq!(
            counts[0].reason_count, 1,
            "exactly one reason, from {state:?}"
        );
    }
}

/// The honest weakness (artifact 2 §11's last row), asserted rather than
/// papered over: a killed daemon's *unexpired* lease blocks its own space until
/// the TTL runs out, because the daemon has no durable process identity that
/// could distinguish "my previous life" from "a concurrent instance". Recovery
/// counts that case and leaves the candidate alone.
#[tokio::test]
async fn a_live_lease_is_counted_and_left_alone() {
    let (db, _) = crashed_mid_flight(CandidateState::Inferencing).await;

    let report = recovery::scan(&db.connection).await.unwrap();

    assert_eq!(report.leases_reaped, 0);
    assert_eq!(report.candidates_lease_live, 1);
    assert!(report.candidates_staled.is_empty());
    assert_eq!(db.candidate_state("cand-a").await, "inferencing");
    assert_eq!(
        db.scalar("SELECT COUNT(*) FROM genesis_frontier", ()).await,
        0,
        "and the group stays hidden for the remainder of the TTL — the known cost"
    );
}

/// The lease check is token-scoped. A candidate whose lease expired and whose
/// space was then legitimately re-leased by another operation must still
/// recover; a space-scoped check would read the new holder's lease as its own
/// and strand the candidate forever.
#[tokio::test]
async fn a_re_leased_space_does_not_shield_the_old_candidate() {
    let (db, _) = crashed_mid_flight(CandidateState::Inferencing).await;
    db.exec(
        "UPDATE grouping_leases SET expires_at = unixepoch() - 1",
        (),
    )
    .await;
    let tx = db.tx().await;
    let fresh = leases::acquire(&tx, LeasePhase::Genesis, SPACE, 1, 0)
        .await
        .unwrap()
        .expect("a new operation takes the expired lease");
    tx.commit().await.unwrap();

    let report = recovery::scan(&db.connection).await.unwrap();

    assert_eq!(report.candidates_staled, vec!["cand-a".to_string()]);
    assert_eq!(report.candidates_lease_live, 0);
    let tx = db.tx().await;
    assert!(
        leases::owns(&tx, LeasePhase::Genesis, SPACE, 1, &fresh)
            .await
            .unwrap(),
        "and the new holder's lease survived recovery untouched"
    );
}

/// `observed` is not a recoverable state: nothing was claimed, so there is
/// nothing to release. Recovering it would convert a candidate that never
/// started into a `stale` exit and burn its A19 re-entry delay for free.
#[tokio::test]
async fn an_observed_candidate_is_not_touched() {
    let db = GenesisDb::new().await;
    db.seed_space(SPACE_ID, SPACE).await;
    let epoch = db.open_epoch(SPACE_ID, SPACE).await;
    db.seed_evidence("root-0", SPACE, "group-1", "entity-0")
        .await;
    observe(&db, "cand-a", SPACE, SignalKind::EvidenceCluster, epoch, 1).await;

    let report = recovery::scan(&db.connection).await.unwrap();

    assert_eq!(report, recovery::RecoveryReport::default());
    assert_eq!(db.candidate_state("cand-a").await, "observed");
}

/// A terminal candidate holding a released claim is likewise not recovered —
/// `released_at IS NULL` is the whole filter, and a released reservation is
/// already accounted for elsewhere.
#[tokio::test]
async fn a_released_reservation_is_not_recovered() {
    let (db, _) = crashed_mid_flight(CandidateState::Prepared).await;
    let tx = db.tx().await;
    candidates::mark_stale(&tx, "cand-a", StaleReason::InputMoved)
        .await
        .unwrap();
    tx.commit().await.unwrap();
    db.exec(
        "UPDATE grouping_leases SET expires_at = unixepoch() - 1",
        (),
    )
    .await;

    let report = recovery::scan(&db.connection).await.unwrap();

    assert_eq!(report.leases_reaped, 1, "the lease is still reaped");
    assert!(report.candidates_staled.is_empty());
    assert_eq!(
        db.candidate_reason("cand-a").await.as_deref(),
        Some("input_moved"),
        "recovery did not overwrite the real reason with lease_lost"
    );
}

/// §4.5 — recovery is resumable. Running it twice must reach the same state as
/// running it once, because a crash *during* recovery is just a shorter first
/// run.
#[tokio::test]
async fn recovery_is_idempotent_across_a_restart() {
    let (db, _) = crashed_mid_flight(CandidateState::Validating).await;
    db.exec(
        "UPDATE grouping_leases SET expires_at = unixepoch() - 1",
        (),
    )
    .await;

    let first = recovery::scan(&db.connection).await.unwrap();
    let second = recovery::scan(&db.connection).await.unwrap();

    assert_eq!(first.candidates_staled, vec!["cand-a".to_string()]);
    assert_eq!(second, recovery::RecoveryReport::default());
    assert_eq!(db.candidate_state("cand-a").await, "stale");
    assert_eq!(
        db.scalar("SELECT COUNT(*) FROM genesis_frontier", ()).await,
        1,
        "the second pass did not duplicate the frontier row"
    );
}

/// Step 3 is a probe: it observes `page_projection_outbox` and writes nothing.
/// The requeue belongs to PR-B3's machine E, which is also the only thing that
/// could complete a hand-off.
#[tokio::test]
async fn the_outbox_is_observed_and_never_written() {
    let db = GenesisDb::new().await;
    db.exec(
        "INSERT INTO page_projection_outbox (page_id, page_version, state, content_digest)
         VALUES ('m6p_a', 1, 'handed_off', 'digest-a'),
                ('m6p_b', 1, 'pending', 'digest-b'),
                ('m6p_c', 1, 'complete', 'digest-c')",
        (),
    )
    .await;

    let report = recovery::scan(&db.connection).await.unwrap();

    assert_eq!(report.projections_handed_off, 1);
    assert_eq!(
        db.scalar(
            "SELECT COUNT(*) FROM page_projection_outbox WHERE state = 'handed_off'",
            ()
        )
        .await,
        1,
        "the probe requeued nothing"
    );
}

/// Vacuous case (S0-137): recovery over an empty database reports zeros rather
/// than erroring — a fresh install runs this on every start.
#[tokio::test]
async fn recovery_over_an_empty_database_is_all_zeros() {
    let db = GenesisDb::new().await;
    assert_eq!(
        recovery::scan(&db.connection).await.unwrap(),
        recovery::RecoveryReport::default()
    );
}
