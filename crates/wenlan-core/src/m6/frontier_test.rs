// SPDX-License-Identifier: Apache-2.0
//! D7 and the G5 gate family (spec §9.1: "the differential query's
//! `reason_count` arms (0 / 1 / >1); cursor encode/decode round-trip; 24h
//! clamp").

use super::candidates::{self, PrepareOutcome, StaleReason};
use super::constants::{
    BELOW_FLOOR_SURFACING_SECONDS, NEXT_SCAN_MAX_AHEAD_SECONDS, PENDING_CANDIDATES_PER_SPACE_CAP,
};
use super::frontier::{self, ScanCursor};
use super::genesis_test_support::{claims, observe, GenesisDb};
use super::identity::SignalKind;
use crate::WenlanError;

const SPACE: &str = "space-a";
const SPACE_ID: &str = "space-id-a";

async fn seeded(groups: &[&str]) -> (GenesisDb, i64) {
    let db = GenesisDb::new().await;
    db.seed_space(SPACE_ID, SPACE).await;
    let epoch = db.open_epoch(SPACE_ID, SPACE).await;
    for (index, group) in groups.iter().enumerate() {
        db.seed_evidence(
            &format!("root-{index}"),
            SPACE,
            group,
            &format!("entity-{index}"),
        )
        .await;
    }
    (db, epoch)
}

async fn frontier_rows(db: &GenesisDb) -> i64 {
    db.scalar("SELECT COUNT(*) FROM genesis_frontier", ()).await
}

// ---------------------------------------------------------------------------
// The differential query — reason_count 0 / 1 / >1
// ---------------------------------------------------------------------------

/// G5.1 — an eligible group with no durable reason reads `0`, and the
/// reconciler's repair is a frontier insert (F1). This is what makes silent
/// parking detectable rather than invisible.
#[tokio::test]
async fn an_unaccounted_group_reads_zero_and_is_repaired() {
    let (db, epoch) = seeded(&["group-1", "group-2"]).await;
    let tx = db.tx().await;
    let before = frontier::reason_counts(&tx, SPACE, epoch).await.unwrap();
    assert_eq!(before.len(), 2);
    assert!(before.iter().all(|group| group.reason_count == 0));

    let outcome = frontier::reconcile_space(&tx, SPACE, epoch).await.unwrap();
    let after = frontier::reason_counts(&tx, SPACE, epoch).await.unwrap();
    tx.commit().await.unwrap();

    assert_eq!(outcome.repaired.len(), 2);
    assert_eq!(outcome.examined, 2);
    assert!(after.iter().all(|group| group.reason_count == 1));
    assert_eq!(frontier_rows(&db).await, 2);
}

/// Reconciling twice repairs nothing the second time. The repair count is the
/// signal, so a reconciler that kept "repairing" an already-accounted group
/// would drown the one number that matters.
#[tokio::test]
async fn a_second_reconciliation_repairs_nothing() {
    let (db, epoch) = seeded(&["group-1"]).await;
    let tx = db.tx().await;
    frontier::reconcile_space(&tx, SPACE, epoch).await.unwrap();
    let second = frontier::reconcile_space(&tx, SPACE, epoch).await.unwrap();
    tx.commit().await.unwrap();
    assert!(second.repaired.is_empty());
    assert_eq!(frontier_rows(&db).await, 1);
}

/// G5.5 — a group that becomes eligible *after* a pass must be surfaced by the
/// next one. The differential query is re-evaluated from the evidence every
/// pass, so there is no "already reconciled" memo that a newly-eligible group
/// could fall behind.
#[tokio::test]
async fn a_newly_eligible_group_is_repaired_on_the_next_pass() {
    let (db, epoch) = seeded(&["group-1"]).await;
    let tx = db.tx().await;
    frontier::reconcile_space(&tx, SPACE, epoch).await.unwrap();
    tx.commit().await.unwrap();

    db.seed_evidence("root-late", SPACE, "group-late", "entity-late")
        .await;
    let tx = db.tx().await;
    let outcome = frontier::reconcile_space(&tx, SPACE, epoch).await.unwrap();
    tx.commit().await.unwrap();

    assert_eq!(outcome.repaired, vec!["group-late".to_string()]);
    assert_eq!(outcome.examined, 2);
    assert_eq!(frontier_rows(&db).await, 2);
}

/// A live concept claim is the group's one reason; the group is not in the
/// frontier and is not repaired back into it.
#[tokio::test]
async fn a_live_claim_is_the_groups_one_reason() {
    let (db, epoch) = seeded(&["group-1"]).await;
    observe(&db, "cand-a", SPACE, SignalKind::EvidenceCluster, epoch, 1).await;
    let tx = db.tx().await;
    let outcome = candidates::prepare_candidate(&tx, "cand-a", &claims(&[("root-0", "group-1")]))
        .await
        .unwrap();
    assert!(matches!(outcome, PrepareOutcome::Prepared { .. }));
    let counts = frontier::reason_counts(&tx, SPACE, epoch).await.unwrap();
    let repaired = frontier::reconcile_space(&tx, SPACE, epoch).await.unwrap();
    tx.commit().await.unwrap();

    assert_eq!(counts.len(), 1);
    assert_eq!(counts[0].reason_count, 1);
    assert!(repaired.repaired.is_empty());
    assert_eq!(frontier_rows(&db).await, 0);
}

/// I-3 — a witness claim is not coverage, so a group only a witness names still
/// reads `0` and is repaired into the frontier. A witness row that counted
/// would hide the group forever.
#[tokio::test]
async fn a_witness_claim_does_not_account_for_a_group() {
    let (db, epoch) = seeded(&["group-1"]).await;
    observe(&db, "cand-a", SPACE, SignalKind::SpaceOverview, epoch, 1).await;
    let tx = db.tx().await;
    candidates::prepare_candidate(&tx, "cand-a", &claims(&[("root-0", "group-1")]))
        .await
        .unwrap();
    let counts = frontier::reason_counts(&tx, SPACE, epoch).await.unwrap();
    tx.commit().await.unwrap();
    assert_eq!(counts[0].reason_count, 0);
}

/// G5.2 — `reason_count > 1` is refused, never repaired (S0-43). Two reasons
/// means two transitions committed that should have been mutually exclusive,
/// and auto-picking a winner would destroy the only trace of which was wrong.
#[tokio::test]
async fn a_double_booked_group_refuses_and_writes_nothing() {
    let (db, epoch) = seeded(&["group-1", "group-2"]).await;
    db.exec(
        "INSERT INTO genesis_frontier
             (space, independence_group_id, coverage_epoch, first_seen_at, next_scan_at)
         VALUES (?1, 'group-1', ?2, unixepoch(), unixepoch())",
        libsql::params![SPACE, epoch],
    )
    .await;
    db.exec(
        "INSERT INTO genesis_quarantine
             (space, independence_group_id, coverage_epoch, reason,
              first_seen_at, quarantined_at)
         VALUES (?1, 'group-1', ?2, 'operator_hold', unixepoch(), unixepoch())",
        libsql::params![SPACE, epoch],
    )
    .await;

    let tx = db.tx().await;
    let counts = frontier::reason_counts(&tx, SPACE, epoch).await.unwrap();
    let double = counts
        .iter()
        .find(|group| group.independence_group_id == "group-1")
        .expect("group-1 present");
    assert_eq!(double.reason_count, 2);

    let error = frontier::reconcile_space(&tx, SPACE, epoch)
        .await
        .unwrap_err();
    tx.rollback().await.unwrap();
    assert!(matches!(error, WenlanError::Conflict(_)), "{error:?}");
    assert_eq!(
        frontier_rows(&db).await,
        1,
        "the refusal repaired nothing, including the innocent group-2"
    );
}

/// Vacuous case (S0-137): a space with no grounded edges has no eligible
/// groups, reconciles nothing, and writes nothing — asserted, not assumed.
#[tokio::test]
async fn an_empty_space_reconciles_nothing() {
    let (db, epoch) = seeded(&[]).await;
    let tx = db.tx().await;
    let outcome = frontier::reconcile_space(&tx, SPACE, epoch).await.unwrap();
    tx.commit().await.unwrap();
    assert_eq!(outcome.examined, 0);
    assert!(outcome.repaired.is_empty());
    assert_eq!(frontier_rows(&db).await, 0);
}

// ---------------------------------------------------------------------------
// G5.3 — first_seen_at survives re-entry
// ---------------------------------------------------------------------------

/// S0-50 under the F5 closure `frontier.rs` documents: repeated claim/release
/// churn — the normal shape of retry-then-fail — must not slide the seven-day
/// clock forward on each round, or the group is parked while every individual
/// transition looks correct. The restore is `MIN` over **every** claim the group
/// ever held in the epoch, so the oldest claim keeps winning.
///
/// This test also pins the **disclosed deviation**: the pre-claim frontier age
/// is not carried across F2's delete, because `genesis_candidate_roots` has no
/// `first_seen_at` column to have carried it. Closing that needs a migration,
/// which PR-B2 is forbidden to add.
#[tokio::test]
async fn churn_does_not_slide_the_seven_day_clock_forward() {
    let (db, epoch) = seeded(&["group-1"]).await;
    let tx = db.tx().await;
    frontier::reconcile_space(&tx, SPACE, epoch).await.unwrap();
    tx.commit().await.unwrap();
    db.age_frontier("group-1", 6 * 86_400).await;
    let before_first_claim = db
        .scalar("SELECT first_seen_at FROM genesis_frontier", ())
        .await;

    async fn churn(db: &GenesisDb, epoch: i64, round: usize) -> i64 {
        let candidate = format!("cand-{round}");
        observe(db, &candidate, SPACE, SignalKind::EvidenceCluster, epoch, 1).await;
        let tx = db.tx().await;
        candidates::prepare_candidate(&tx, &candidate, &claims(&[("root-0", "group-1")]))
            .await
            .unwrap();
        candidates::mark_stale(&tx, &candidate, StaleReason::LeaseLost)
            .await
            .unwrap();
        tx.commit().await.unwrap();
        db.release_leases().await;
        db.scalar("SELECT first_seen_at FROM genesis_frontier", ())
            .await
    }

    let after_first = churn(&db, epoch, 0).await;
    assert!(
        after_first > before_first_claim,
        "the disclosed F5 deviation: the pre-claim frontier age is not carried"
    );
    // Age the first claim's root so later rounds have something older to lose to.
    db.exec(
        "UPDATE genesis_candidate_roots SET created_at = created_at - 5 * 86400",
        (),
    )
    .await;
    let aged_claim = db
        .scalar("SELECT MIN(created_at) FROM genesis_candidate_roots", ())
        .await;

    let after_second = churn(&db, epoch, 1).await;
    let after_third = churn(&db, epoch, 2).await;
    assert_eq!(frontier_rows(&db).await, 1);
    assert_eq!(
        (after_second, after_third),
        (aged_claim, aged_claim),
        "each round restored to `now` instead of the oldest claim — the clock slid"
    );
}

/// An F1 insert against an existing row leaves the older `first_seen_at`
/// standing — the `INSERT OR REPLACE` that G5.3's RED mutation would introduce
/// is exactly the clock reset S0-50 forbids.
#[tokio::test]
async fn a_repeated_frontier_insert_never_resets_the_clock() {
    let (db, epoch) = seeded(&["group-1"]).await;
    let tx = db.tx().await;
    frontier::insert_frontier_row(&tx, SPACE, "group-1", epoch)
        .await
        .unwrap();
    tx.commit().await.unwrap();
    db.age_frontier("group-1", 5 * 86_400).await;
    let aged = db
        .scalar("SELECT first_seen_at FROM genesis_frontier", ())
        .await;

    let tx = db.tx().await;
    frontier::insert_frontier_row(&tx, SPACE, "group-1", epoch)
        .await
        .unwrap();
    tx.commit().await.unwrap();

    assert_eq!(frontier_rows(&db).await, 1);
    assert_eq!(
        db.scalar("SELECT first_seen_at FROM genesis_frontier", ())
            .await,
        aged
    );
}

// ---------------------------------------------------------------------------
// G5.4 — caps defer, never terminalize
// ---------------------------------------------------------------------------

/// S0-54 — hitting a cap leaves the remainder frontier-visible. A cap is a rate
/// limit; converting one into a terminal outcome is the most natural way to
/// violate D7 while looking like good hygiene.
#[tokio::test]
async fn a_cap_hit_leaves_the_remainder_frontier_visible() {
    let (db, epoch) = seeded(&["group-1"]).await;
    let tx = db.tx().await;
    frontier::reconcile_space(&tx, SPACE, epoch).await.unwrap();
    tx.commit().await.unwrap();
    for index in 0..PENDING_CANDIDATES_PER_SPACE_CAP {
        observe(
            &db,
            &format!("filler-{index}"),
            SPACE,
            SignalKind::EvidenceCluster,
            epoch,
            1,
        )
        .await;
    }

    let tx = db.tx().await;
    let refused = candidates::observe_candidate(
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

    assert_eq!(refused, candidates::ObserveOutcome::RefusedPendingCap);
    assert_eq!(
        frontier_rows(&db).await,
        1,
        "the capped group keeps its frontier row"
    );
    let tx = db.tx().await;
    let counts = frontier::reason_counts(&tx, SPACE, epoch).await.unwrap();
    assert_eq!(counts[0].reason_count, 1, "still exactly one reason");
}

// ---------------------------------------------------------------------------
// G5.5 — the cursor
// ---------------------------------------------------------------------------

#[tokio::test]
async fn the_cursor_round_trips_and_clears_on_wrap() {
    let (db, _) = seeded(&[]).await;
    let cursor = ScanCursor {
        next_scan_at: 1_700_000_000,
        first_seen_at: 1_600_000_000,
        independence_group_id: "group-7".to_string(),
    };
    let tx = db.tx().await;
    frontier::write_cursor(&tx, SPACE, Some(&cursor))
        .await
        .unwrap();
    let read_back = frontier::read_cursor(&tx, SPACE).await.unwrap();
    frontier::write_cursor(&tx, SPACE, None).await.unwrap();
    let wrapped = frontier::read_cursor(&tx, SPACE).await.unwrap();
    tx.commit().await.unwrap();

    assert_eq!(read_back, Some(cursor));
    assert_eq!(wrapped, None, "the cleared cursor means start over");
}

/// S0-44 — the cursor is a pure optimization, so a corrupt value costs a
/// rescan, never a failed scan.
#[tokio::test]
async fn a_corrupt_cursor_starts_the_pass_over() {
    let (db, _) = seeded(&[]).await;
    db.exec(
        "INSERT INTO app_metadata (key, value) VALUES (?1, 'not-a-cursor')",
        libsql::params![format!("{}{SPACE}", frontier::FRONTIER_CURSOR_KEY_PREFIX)],
    )
    .await;
    let tx = db.tx().await;
    assert_eq!(frontier::read_cursor(&tx, SPACE).await.unwrap(), None);
}

/// The cursor is per-space (Q7), so one space's position cannot move another's.
#[tokio::test]
async fn cursors_are_per_space() {
    let (db, _) = seeded(&[]).await;
    let cursor = ScanCursor {
        next_scan_at: 10,
        first_seen_at: 5,
        independence_group_id: "group-1".to_string(),
    };
    let tx = db.tx().await;
    frontier::write_cursor(&tx, SPACE, Some(&cursor))
        .await
        .unwrap();
    assert_eq!(frontier::read_cursor(&tx, "space-b").await.unwrap(), None);
    tx.commit().await.unwrap();
}

/// A scan resumes from a stored position and does not revisit what it passed;
/// losing the position costs a rescan and loses nothing (P1, P2).
#[tokio::test]
async fn a_scan_resumes_from_its_stored_position() {
    let (db, epoch) = seeded(&["group-a", "group-b", "group-c"]).await;
    let tx = db.tx().await;
    frontier::reconcile_space(&tx, SPACE, epoch).await.unwrap();
    tx.commit().await.unwrap();
    // Distinct next_scan_at values so the ordering is total and observable.
    for (index, group) in ["group-a", "group-b", "group-c"].into_iter().enumerate() {
        db.exec(
            "UPDATE genesis_frontier SET next_scan_at = ?2 WHERE independence_group_id = ?1",
            libsql::params![group, index as i64],
        )
        .await;
    }

    let tx = db.tx().await;
    let slice = frontier::scan_slice(&tx, SPACE, epoch, None).await.unwrap();
    assert_eq!(slice.rows.len(), 3);
    assert_eq!(slice.rows[0].independence_group_id, "group-a");
    assert_eq!(
        slice.next_cursor, None,
        "a short slice wrapped rather than parking a position"
    );

    let resumed = frontier::scan_slice(
        &tx,
        SPACE,
        epoch,
        Some(&ScanCursor {
            next_scan_at: slice.rows[0].next_scan_at,
            first_seen_at: slice.rows[0].first_seen_at,
            independence_group_id: slice.rows[0].independence_group_id.clone(),
        }),
    )
    .await
    .unwrap();
    assert_eq!(resumed.rows.len(), 2);
    assert_eq!(resumed.rows[0].independence_group_id, "group-b");
}

// ---------------------------------------------------------------------------
// G5.6 — the 24h ceiling
// ---------------------------------------------------------------------------

/// S0-46 — `next_scan_at` may never be more than 24h ahead, so the seven-day
/// timer is evaluated at least seven times before it can fire and no path can
/// defer a row past its own deadline (P3).
#[tokio::test]
async fn deferral_is_clamped_to_twenty_four_hours() {
    let (db, epoch) = seeded(&["group-1"]).await;
    let tx = db.tx().await;
    frontier::reconcile_space(&tx, SPACE, epoch).await.unwrap();
    frontier::defer_scan(&tx, SPACE, "group-1", epoch, 400 * 86_400)
        .await
        .unwrap();
    let lead = frontier::max_next_scan_lead_seconds(&tx, SPACE)
        .await
        .unwrap();
    tx.commit().await.unwrap();
    assert_eq!(lead, NEXT_SCAN_MAX_AHEAD_SECONDS);
    assert!(lead <= 86_400);
}

/// A negative delay cannot pull a row into the past either — the clamp is
/// two-sided, in the statement.
#[tokio::test]
async fn deferral_never_moves_a_row_backwards() {
    let (db, epoch) = seeded(&["group-1"]).await;
    let tx = db.tx().await;
    frontier::reconcile_space(&tx, SPACE, epoch).await.unwrap();
    frontier::defer_scan(&tx, SPACE, "group-1", epoch, -9_999)
        .await
        .unwrap();
    let lead = frontier::max_next_scan_lead_seconds(&tx, SPACE)
        .await
        .unwrap();
    tx.commit().await.unwrap();
    assert_eq!(lead, 0);
}

// ---------------------------------------------------------------------------
// F3 — the coalesced below-floor card
// ---------------------------------------------------------------------------

/// The card ID is deterministic and carries no space name (artifact 5 §3.2).
#[tokio::test]
async fn the_card_id_is_deterministic_and_nameless() {
    let first = frontier::card_id(SPACE_ID, 1);
    assert_eq!(first, frontier::card_id(SPACE_ID, 1));
    assert_ne!(
        first,
        frontier::card_id(SPACE_ID, 2),
        "the epoch is in the ID"
    );
    assert_ne!(first, frontier::card_id("space-id-b", 1));
    assert!(first.starts_with("m6_unformed_topic_"));
    assert!(!first.contains(SPACE), "a space name never enters an ID");
}

/// P10 — a permanently small space surfaces **once**: not zero times, and not
/// repeatedly. The card fires on the seven-day frontier age regardless of how
/// far below the floor the group is.
#[tokio::test]
async fn a_permanently_small_space_surfaces_once() {
    let (db, epoch) = seeded(&["group-1"]).await;
    let tx = db.tx().await;
    frontier::reconcile_space(&tx, SPACE, epoch).await.unwrap();
    tx.commit().await.unwrap();

    let tx = db.tx().await;
    assert_eq!(
        frontier::surface_below_floor_groups(&tx, SPACE, SPACE_ID, epoch)
            .await
            .unwrap(),
        None,
        "nothing surfaces before the seven-day timer"
    );
    tx.commit().await.unwrap();

    db.age_frontier("group-1", BELOW_FLOOR_SURFACING_SECONDS + 1)
        .await;
    let tx = db.tx().await;
    let surfacing = frontier::surface_below_floor_groups(&tx, SPACE, SPACE_ID, epoch)
        .await
        .unwrap()
        .expect("the seven-day card fires");
    tx.commit().await.unwrap();
    assert_eq!(surfacing.groups, vec!["group-1".to_string()]);

    let tx = db.tx().await;
    assert_eq!(
        frontier::surface_below_floor_groups(&tx, SPACE, SPACE_ID, epoch)
            .await
            .unwrap(),
        None,
        "the group left the frontier for the card, so it does not surface again"
    );
    let counts = frontier::reason_counts(&tx, SPACE, epoch).await.unwrap();
    assert_eq!(
        counts[0].reason_count, 1,
        "the binding is now the one reason"
    );
}

/// The shadow binds groups to a card; it never writes the user-visible card row
/// (Q11's positive control). Nothing in this slice touches `refinement_queue`.
#[tokio::test]
async fn surfacing_writes_only_genesis_rows() {
    let (db, epoch) = seeded(&["group-1"]).await;
    let tx = db.tx().await;
    frontier::reconcile_space(&tx, SPACE, epoch).await.unwrap();
    tx.commit().await.unwrap();
    db.age_frontier("group-1", BELOW_FLOOR_SURFACING_SECONDS + 1)
        .await;
    let tx = db.tx().await;
    frontier::surface_below_floor_groups(&tx, SPACE, SPACE_ID, epoch)
        .await
        .unwrap();
    tx.commit().await.unwrap();

    assert_eq!(
        db.scalar(
            "SELECT COUNT(*) FROM sqlite_master WHERE name = 'refinement_queue'",
            ()
        )
        .await,
        0,
        "the fixture has no card table at all, and the writer never needed one"
    );
    assert_eq!(
        db.scalar(
            "SELECT COUNT(*) FROM genesis_card_binding WHERE closed_at IS NULL",
            ()
        )
        .await,
        1
    );
}

/// F8 — an expired card returns the group to the frontier with the binding's
/// own `first_seen_at`, so this re-entry path preserves the clock exactly.
#[tokio::test]
async fn an_expired_card_returns_the_group_with_its_clock_intact() {
    let (db, epoch) = seeded(&["group-1"]).await;
    let tx = db.tx().await;
    frontier::reconcile_space(&tx, SPACE, epoch).await.unwrap();
    tx.commit().await.unwrap();
    db.age_frontier("group-1", BELOW_FLOOR_SURFACING_SECONDS + 1)
        .await;
    let bound_clock = db
        .scalar("SELECT first_seen_at FROM genesis_frontier", ())
        .await;

    let tx = db.tx().await;
    frontier::surface_below_floor_groups(&tx, SPACE, SPACE_ID, epoch)
        .await
        .unwrap();
    let expired = frontier::expire_card_binding_to_frontier(&tx, SPACE, "group-1", epoch)
        .await
        .unwrap();
    tx.commit().await.unwrap();

    assert!(expired);
    assert_eq!(
        db.scalar("SELECT first_seen_at FROM genesis_frontier", ())
            .await,
        bound_clock
    );
    assert_eq!(
        db.scalar(
            "SELECT COUNT(*) FROM genesis_card_binding WHERE closed_at IS NULL",
            ()
        )
        .await,
        0,
        "the binding is closed, not deleted"
    );
    assert_eq!(
        db.scalar("SELECT COUNT(*) FROM genesis_card_binding", ())
            .await,
        1
    );
}

/// Vacuous: a group with no open binding is not an error, it is the ordinary
/// case for a candidate that never reached review.
#[tokio::test]
async fn expiring_a_card_that_does_not_exist_is_a_no_op() {
    let (db, epoch) = seeded(&["group-1"]).await;
    let tx = db.tx().await;
    assert!(
        !frontier::expire_card_binding_to_frontier(&tx, SPACE, "group-1", epoch)
            .await
            .unwrap()
    );
}
