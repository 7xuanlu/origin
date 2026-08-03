// SPDX-License-Identifier: Apache-2.0
//! Teeth for the C1 decay reference.
//!
//! These live under `src/` so the workspace library-test floor runs them; an
//! integration test would not be guaranteed in L3/L4.

use super::relevance_sweep::{
    advance_decay_reference, decay_reference, decayed_contribution,
    COUNTER_RELEVANCE_DECAY_REFERENCE, DECAY_HALF_LIFE_DAYS,
};
use crate::m6::genesis_test_support::GenesisDb;

const SECONDS_PER_DAY: i64 = 86_400;

#[tokio::test]
async fn the_reference_is_absent_until_the_first_pass_advances_it() {
    let db = GenesisDb::new().await;
    db.seed_space("space-id-a", "space-a").await;

    let tx = db.tx().await;
    assert_eq!(
        decay_reference(&tx, "space-id-a")
            .await
            .expect("read absent reference"),
        None,
        "a space with no re-reference pass must not report a reference"
    );

    let first = advance_decay_reference(&tx, "space-id-a", "space-a")
        .await
        .expect("first advance");
    assert!(first > 0, "the reference is a unixepoch, not a counter");
    assert_eq!(
        decay_reference(&tx, "space-id-a")
            .await
            .expect("read after advance"),
        Some(first)
    );
    tx.commit().await.expect("commit");
}

#[tokio::test]
async fn the_reference_never_moves_backwards() {
    let db = GenesisDb::new().await;
    db.seed_space("space-id-a", "space-a").await;

    // A reference already far ahead of now — the shape a clock step backwards
    // would produce. Lowering it would make every pair's decayed weight
    // *increase*, which no retraction could later undo.
    let far_future = 4_102_444_800_i64; // 2100-01-01
    db.exec(
        "INSERT INTO m6_counters (space_id, space, name, value) VALUES (?1, ?2, ?3, ?4)",
        libsql::params![
            "space-id-a",
            "space-a",
            COUNTER_RELEVANCE_DECAY_REFERENCE,
            far_future
        ],
    )
    .await;

    let tx = db.tx().await;
    let after = advance_decay_reference(&tx, "space-id-a", "space-a")
        .await
        .expect("advance against a future reference");
    assert_eq!(
        after, far_future,
        "advancing must never lower the stored reference"
    );
    tx.commit().await.expect("commit");
}

#[tokio::test]
async fn two_spaces_keep_independent_references() {
    let db = GenesisDb::new().await;
    db.seed_space("space-id-a", "space-a").await;
    db.seed_space("space-id-b", "space-b").await;

    let tx = db.tx().await;
    let a = advance_decay_reference(&tx, "space-id-a", "space-a")
        .await
        .expect("advance a");
    assert_eq!(
        decay_reference(&tx, "space-id-b").await.expect("read b"),
        None,
        "the reference is keyed per space, not global"
    );
    tx.commit().await.expect("commit a");
    assert!(a > 0);
}

#[tokio::test]
async fn a_contribution_decays_to_the_stored_reference_not_to_now() {
    // The whole point of Q1. Two evaluations of the same root at the same
    // stored reference agree exactly, whatever the wall clock says between
    // them — which is what makes the incremental-equals-full oracle exact by
    // construction rather than exact within a tolerance.
    let reference = 1_800_000_000_i64;
    let created_at = reference - 90 * SECONDS_PER_DAY;

    let first = decayed_contribution(1.0, created_at, reference);
    let second = decayed_contribution(1.0, created_at, reference);
    assert_eq!(
        first.to_bits(),
        second.to_bits(),
        "the same root at the same reference must decay bit-identically"
    );

    // Decaying the same root to a *later* instant, as an unpinned
    // implementation would, gives a different cell — the failure the pin
    // removes.
    let unpinned = decayed_contribution(1.0, created_at, reference + SECONDS_PER_DAY);
    assert_ne!(first.to_bits(), unpinned.to_bits());
}

#[tokio::test]
async fn a_root_halves_its_weight_at_the_half_life() {
    let reference = 1_800_000_000_i64;
    let one_half_life = reference - (DECAY_HALF_LIFE_DAYS as i64) * SECONDS_PER_DAY;
    let two_half_lives = reference - 2 * (DECAY_HALF_LIFE_DAYS as i64) * SECONDS_PER_DAY;

    assert!((decayed_contribution(1.0, reference, reference) - 1.0).abs() < 1e-12);
    assert!((decayed_contribution(1.0, one_half_life, reference) - 0.5).abs() < 1e-12);
    assert!((decayed_contribution(1.0, two_half_lives, reference) - 0.25).abs() < 1e-12);
    // Hub weight scales linearly.
    assert!((decayed_contribution(4.0, one_half_life, reference) - 2.0).abs() < 1e-12);
}

#[tokio::test]
async fn a_root_newer_than_the_reference_is_clamped_rather_than_amplified() {
    let reference = 1_800_000_000_i64;
    let newer = reference + 30 * SECONDS_PER_DAY;

    let contribution = decayed_contribution(1.0, newer, reference);
    assert!(
        (contribution - 1.0).abs() < 1e-12,
        "a root newer than the reference must contribute at full weight, got {contribution}"
    );
    assert!(
        contribution <= 1.0,
        "a negative age must never amplify a cell above its own hub weight; \
         apply_group_eligibility_change refuses to drive a cell negative, so an \
         inflated cell could never be retracted back"
    );
}

#[tokio::test]
async fn a_nonfinite_or_negative_hub_weight_contributes_nothing() {
    let reference = 1_800_000_000_i64;
    for weight in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
        assert_eq!(
            decayed_contribution(weight, reference, reference),
            0.0,
            "hub weight {weight} must contribute zero, never poison a cell"
        );
    }
}
