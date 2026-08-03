// SPDX-License-Identifier: Apache-2.0
//! D6 — one durable lease registry (spec §9.1: "acquire/CAS/release/reap
//! against the real `grouping_leases`; the manual-joins-scheduler case").
//!
//! Every test here runs against the shipped `grouping_leases` DDL, installed by
//! `MemoryDB::ensure_community_substrate_tables`. That is the point of the
//! module: I-6 forbids a second lease system, and a fixture with its own copy of
//! the table would be structurally unable to notice one.

use super::genesis_test_support::GenesisDb;
use super::leases::{self, LeasePhase};

async fn db() -> GenesisDb {
    GenesisDb::new().await
}

async fn lease_rows(db: &GenesisDb) -> i64 {
    db.scalar("SELECT COUNT(*) FROM grouping_leases", ()).await
}

/// C1 — the first acquire wins and stores a token.
#[tokio::test]
async fn acquire_takes_the_operation() {
    let db = db().await;
    let tx = db.tx().await;
    let token = leases::acquire(&tx, LeasePhase::Genesis, "space-a", 7, 0)
        .await
        .unwrap();
    tx.commit().await.unwrap();
    assert!(token.is_some());
    assert_eq!(lease_rows(&db).await, 1);
}

/// C2 — a second acquire for the same `(phase, space, input_generation)`
/// returns `None`. Zero affected rows *is* the "someone else holds it" signal;
/// there is no separate check that could disagree with the insert.
#[tokio::test]
async fn second_acquire_observes_the_holder() {
    let db = db().await;
    let tx = db.tx().await;
    let first = leases::acquire(&tx, LeasePhase::Genesis, "space-a", 7, 0)
        .await
        .unwrap();
    let second = leases::acquire(&tx, LeasePhase::Genesis, "space-a", 7, 0)
        .await
        .unwrap();
    tx.commit().await.unwrap();
    assert!(first.is_some());
    assert_eq!(second, None);
    assert_eq!(lease_rows(&db).await, 1, "a refused acquire writes nothing");
}

/// The manual-joins-scheduler case (spec §9.1). Two callers that reach the same
/// operation identity join it structurally: the primary key is the operation,
/// so exactly one proceeds and no extra coordination exists to be forgotten.
#[tokio::test]
async fn a_manual_run_and_the_scheduler_join_one_operation() {
    let db = db().await;
    let scheduler = db.tx().await;
    let scheduler_token = leases::acquire(&scheduler, LeasePhase::Genesis, "space-a", 3, 0)
        .await
        .unwrap();
    scheduler.commit().await.unwrap();

    let manual = db.tx().await;
    let manual_token = leases::acquire(&manual, LeasePhase::Genesis, "space-a", 3, 0)
        .await
        .unwrap();
    manual.commit().await.unwrap();

    assert!(scheduler_token.is_some());
    assert_eq!(manual_token, None);
    assert_eq!(lease_rows(&db).await, 1);
}

/// The load-bearing property: the reap arm binds `phase` as well as the insert.
/// M4's reap spells the literal `'community'`; a copy that forgot to change it
/// would delete a live lease belonging to another phase.
#[tokio::test]
async fn acquiring_one_phase_never_reaps_another_phases_live_lease() {
    let db = db().await;
    let tx = db.tx().await;
    let frontier = leases::acquire(&tx, LeasePhase::Frontier, "space-a", 7, 0)
        .await
        .unwrap()
        .expect("frontier lease acquired");
    let genesis = leases::acquire(&tx, LeasePhase::Genesis, "space-a", 7, 0)
        .await
        .unwrap()
        .expect("genesis lease acquired on the same space");
    assert!(
        leases::owns(&tx, LeasePhase::Frontier, "space-a", 7, &frontier)
            .await
            .unwrap(),
        "the genesis acquire must not have reaped the live frontier lease"
    );
    assert!(
        leases::owns(&tx, LeasePhase::Genesis, "space-a", 7, &genesis)
            .await
            .unwrap()
    );
    tx.commit().await.unwrap();
    assert_eq!(lease_rows(&db).await, 2);
}

/// C9 — a superseded `input_generation` is reaped at acquire time, so the new
/// generation's worker is not blocked by the old one's row.
#[tokio::test]
async fn a_superseded_generation_is_reaped_at_acquire() {
    let db = db().await;
    let tx = db.tx().await;
    let stale = leases::acquire(&tx, LeasePhase::Genesis, "space-a", 7, 0)
        .await
        .unwrap()
        .expect("generation 7 lease");
    let fresh = leases::acquire(&tx, LeasePhase::Genesis, "space-a", 8, 0)
        .await
        .unwrap();
    tx.commit().await.unwrap();
    assert!(fresh.is_some());
    assert_eq!(lease_rows(&db).await, 1, "the old generation's row is gone");
    let tx = db.tx().await;
    assert!(
        !leases::owns(&tx, LeasePhase::Genesis, "space-a", 7, &stale)
            .await
            .unwrap()
    );
}

/// C7 — an expired row is reaped at acquire, and the clock moves only by moving
/// the stored `expires_at` (S0-11).
#[tokio::test]
async fn an_expired_lease_is_reaped_at_acquire() {
    let db = db().await;
    let tx = db.tx().await;
    leases::acquire(&tx, LeasePhase::Genesis, "space-a", 7, 0)
        .await
        .unwrap()
        .expect("first lease");
    tx.commit().await.unwrap();
    db.exec(
        "UPDATE grouping_leases SET expires_at = unixepoch() - 1",
        (),
    )
    .await;

    let tx = db.tx().await;
    let taken_over = leases::acquire(&tx, LeasePhase::Genesis, "space-a", 7, 0)
        .await
        .unwrap();
    tx.commit().await.unwrap();
    assert!(taken_over.is_some(), "an expired lease is takeable");
    assert_eq!(lease_rows(&db).await, 1);
}

/// An expired-but-present row is **not** ownership. The row is still physically
/// there and still visible to any query that forgets the time, which is machine
/// C's `expired` state — the one a finalize must never mistake for its lease.
#[tokio::test]
async fn ownership_requires_an_unexpired_row() {
    let db = db().await;
    let tx = db.tx().await;
    let token = leases::acquire(&tx, LeasePhase::Genesis, "space-a", 7, 0)
        .await
        .unwrap()
        .expect("lease");
    assert!(leases::owns(&tx, LeasePhase::Genesis, "space-a", 7, &token)
        .await
        .unwrap());
    tx.commit().await.unwrap();

    db.exec(
        "UPDATE grouping_leases SET expires_at = unixepoch() - 1",
        (),
    )
    .await;
    let tx = db.tx().await;
    assert!(
        !leases::owns(&tx, LeasePhase::Genesis, "space-a", 7, &token)
            .await
            .unwrap(),
        "an expired row must not read as ownership"
    );
    assert_eq!(lease_rows(&db).await, 1, "and `owns` deletes nothing");
}

/// C4/C5 — release is token-scoped, so a release arriving after a takeover
/// cannot delete the new holder's row.
#[tokio::test]
async fn release_is_token_scoped() {
    let db = db().await;
    let tx = db.tx().await;
    let token = leases::acquire(&tx, LeasePhase::Genesis, "space-a", 7, 0)
        .await
        .unwrap()
        .expect("lease");
    let wrong = leases::release(&tx, LeasePhase::Genesis, "space-a", 7, "not-the-token")
        .await
        .unwrap();
    let right = leases::release(&tx, LeasePhase::Genesis, "space-a", 7, &token)
        .await
        .unwrap();
    tx.commit().await.unwrap();
    assert_eq!(wrong, 0);
    assert_eq!(right, 1);
    assert_eq!(lease_rows(&db).await, 0);
}

/// C8 — the startup reap is phase-agnostic, and it is the only statement in the
/// module that is.
#[tokio::test]
async fn the_startup_reap_crosses_phases_and_spares_live_rows() {
    let db = db().await;
    let tx = db.tx().await;
    leases::acquire(&tx, LeasePhase::Genesis, "space-a", 7, 0)
        .await
        .unwrap()
        .expect("genesis lease");
    leases::acquire(&tx, LeasePhase::Frontier, "space-b", 7, 0)
        .await
        .unwrap()
        .expect("frontier lease");
    tx.commit().await.unwrap();
    db.exec(
        "UPDATE grouping_leases SET expires_at = unixepoch() - 1 WHERE space = 'space-a'",
        (),
    )
    .await;

    let tx = db.tx().await;
    let reaped = leases::reap_expired_all_phases(&tx).await.unwrap();
    tx.commit().await.unwrap();
    assert_eq!(reaped, 1);
    assert_eq!(
        db.scalar(
            "SELECT COUNT(*) FROM grouping_leases WHERE phase = 'frontier'",
            ()
        )
        .await,
        1,
        "a live lease of another phase survives the startup reap"
    );
}

/// Vacuous case (S0-137): a startup reap over an empty registry deletes nothing
/// and reports zero, rather than erroring or reporting work it did not do.
#[tokio::test]
async fn the_startup_reap_is_zero_on_an_empty_registry() {
    let db = db().await;
    let tx = db.tx().await;
    assert_eq!(leases::reap_expired_all_phases(&tx).await.unwrap(), 0);
    tx.commit().await.unwrap();
    assert_eq!(lease_rows(&db).await, 0);
}

/// S0-3's two TTLs are distinguishable in the stored row, not just in Rust.
#[tokio::test]
async fn each_phase_stores_its_own_ttl() {
    let db = db().await;
    let tx = db.tx().await;
    leases::acquire(&tx, LeasePhase::Genesis, "space-a", 7, 0)
        .await
        .unwrap()
        .expect("genesis lease");
    leases::acquire(&tx, LeasePhase::Frontier, "space-b", 7, 0)
        .await
        .unwrap()
        .expect("frontier lease");
    tx.commit().await.unwrap();

    assert_eq!(
        db.scalar(
            "SELECT expires_at - unixepoch() FROM grouping_leases WHERE phase = 'genesis'",
            ()
        )
        .await,
        900
    );
    assert_eq!(
        db.scalar(
            "SELECT expires_at - unixepoch() FROM grouping_leases WHERE phase = 'frontier'",
            ()
        )
        .await,
        120
    );
}
