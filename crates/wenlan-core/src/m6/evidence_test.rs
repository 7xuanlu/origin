// SPDX-License-Identifier: Apache-2.0
//! §7.1 readiness and §7.2 shadow statistics.
//!
//! Two things these gates are careful about. First, **`committed` must be
//! unreachable from this lane and reachable from the substrate** — a test that
//! only showed the phase staying at `preparing` would pass just as well against
//! a substrate that forbade `committed` outright, and would therefore say
//! nothing about the caller. Second, **every statistic gets a non-zero**: a
//! reader that returned `ShadowStats::default()` would satisfy every "is zero"
//! assertion in the file, so each field is driven off its default at least once.

use super::candidates::CandidateState;
use super::evidence::{
    bump, note_shadow_readiness, observe_space, read_counter, refusal_counter, sqlite_now,
    ShadowStats, COUNTER_DRY_RUN_PASSED, COUNTER_ORACLE_DIVERGENCE, COUNTER_TURNS,
    READINESS_SIGNAL, READINESS_STAGE,
};
use super::finalize::CasGate;
use super::genesis_test_support::GenesisDb;
use super::refresh_readiness::{
    readiness_fence, transition_is_legal, ReadinessKey, ReadinessPhase,
};

const SPACE: &str = "space-a";
const SPACE_ID: &str = "space-id-a";
const EPOCH: i64 = 1;

fn key() -> ReadinessKey<'static> {
    ReadinessKey {
        space: SPACE,
        stage: READINESS_STAGE,
        signal: READINESS_SIGNAL,
    }
}

// ---------------------------------------------------------------------------
// §7.1 readiness
// ---------------------------------------------------------------------------

/// First observation initialises the row and moves it `off → preparing`, and
/// the epoch increments exactly once for that move.
#[tokio::test]
async fn the_first_observation_opens_the_stage_b_fence() {
    let db = GenesisDb::new().await;

    let tx = db.tx().await;
    let phase = note_shadow_readiness(&tx, SPACE).await.expect("readiness");
    let fence = readiness_fence(&tx, key())
        .await
        .expect("fence")
        .expect("row");
    tx.commit().await.unwrap();

    assert_eq!(phase, ReadinessPhase::Preparing);
    assert_eq!(fence.phase, ReadinessPhase::Preparing);
    assert_eq!(fence.epoch, 1);
}

/// A second observation is a normal steady state, not a fault: it returns the
/// same phase and does **not** increment the epoch again. `transition_readiness`
/// alone would have raised here — the legality pre-check is what makes the
/// repeat quiet, so this is the gate that would fail if the pre-check were
/// dropped as redundant.
#[tokio::test]
async fn a_repeat_observation_is_quiet_and_moves_no_epoch() {
    let db = GenesisDb::new().await;

    let tx = db.tx().await;
    note_shadow_readiness(&tx, SPACE).await.expect("first");
    for _ in 0..4 {
        assert_eq!(
            note_shadow_readiness(&tx, SPACE).await.expect("repeat"),
            ReadinessPhase::Preparing
        );
    }
    let fence = readiness_fence(&tx, key())
        .await
        .expect("fence")
        .expect("row");
    tx.commit().await.unwrap();

    assert_eq!(fence.epoch, 1, "five observations, one epoch move");
}

/// **The discriminating control.** `preparing → committed` is a legal edge in
/// the substrate, so `committed` is withheld by this lane's choice of target,
/// not by anything that would stop a future caller. The phase never reaching
/// `committed` is therefore a statement about PR-B, which is what §7.1 asks
/// for: committed is a cutover fact and PR-B cuts nothing over.
#[tokio::test]
async fn committed_is_reachable_in_the_substrate_and_never_from_this_lane() {
    assert!(
        transition_is_legal(ReadinessPhase::Preparing, ReadinessPhase::Committed),
        "the substrate must allow the edge this lane declines to take"
    );

    let db = GenesisDb::new().await;
    let tx = db.tx().await;
    for _ in 0..8 {
        note_shadow_readiness(&tx, SPACE).await.expect("readiness");
    }
    let fence = readiness_fence(&tx, key())
        .await
        .expect("fence")
        .expect("row");
    tx.commit().await.unwrap();

    assert_eq!(fence.phase, ReadinessPhase::Preparing);
}

/// Readiness is per space: one space's fence says nothing about another's.
#[tokio::test]
async fn readiness_is_keyed_per_space() {
    let db = GenesisDb::new().await;

    let tx = db.tx().await;
    note_shadow_readiness(&tx, SPACE).await.expect("readiness");
    let other = readiness_fence(
        &tx,
        ReadinessKey {
            space: "space-b",
            stage: READINESS_STAGE,
            signal: READINESS_SIGNAL,
        },
    )
    .await
    .expect("fence");
    tx.commit().await.unwrap();

    assert_eq!(other, None);
}

// ---------------------------------------------------------------------------
// §7.2 counters
// ---------------------------------------------------------------------------

/// An untouched counter reads `None`, not `Some(0)` — the caller needs to tell
/// "never happened" from "happened zero times" when it decides whether to
/// create the row.
#[tokio::test]
async fn an_untouched_counter_is_absent_rather_than_zero() {
    let db = GenesisDb::new().await;
    let tx = db.tx().await;
    let value = read_counter(&tx, SPACE_ID, COUNTER_TURNS)
        .await
        .expect("read");
    tx.commit().await.unwrap();
    assert_eq!(value, None);
}

/// A counter is created on first bump and accumulates after that. The
/// `BEFORE INSERT` monotone trigger is why this is read-then-insert-then-update
/// rather than an upsert, so a second bump over a stored value is precisely the
/// case that would abort under the obvious implementation.
#[tokio::test]
async fn a_counter_is_created_on_first_bump_and_accumulates() {
    let db = GenesisDb::new().await;
    db.seed_space(SPACE_ID, SPACE).await;

    let tx = db.tx().await;
    bump(&tx, SPACE_ID, SPACE, COUNTER_TURNS, 1)
        .await
        .expect("first");
    assert_eq!(
        read_counter(&tx, SPACE_ID, COUNTER_TURNS)
            .await
            .expect("read"),
        Some(1)
    );
    bump(&tx, SPACE_ID, SPACE, COUNTER_TURNS, 5)
        .await
        .expect("second");
    bump(&tx, SPACE_ID, SPACE, COUNTER_TURNS, 3)
        .await
        .expect("third");
    let value = read_counter(&tx, SPACE_ID, COUNTER_TURNS)
        .await
        .expect("read");
    tx.commit().await.unwrap();

    assert_eq!(value, Some(9));
}

/// A non-positive delta is a no-op, and specifically does **not** create the
/// row. The counter series is monotone by trigger; refusing the write here is
/// what keeps a caller from discovering that the hard way.
#[tokio::test]
async fn a_non_positive_bump_writes_nothing() {
    let db = GenesisDb::new().await;

    let tx = db.tx().await;
    bump(&tx, SPACE_ID, SPACE, COUNTER_TURNS, 0)
        .await
        .expect("zero");
    bump(&tx, SPACE_ID, SPACE, COUNTER_TURNS, -4)
        .await
        .expect("negative");
    let value = read_counter(&tx, SPACE_ID, COUNTER_TURNS)
        .await
        .expect("read");
    tx.commit().await.unwrap();

    assert_eq!(value, None);
    assert_eq!(db.scalar("SELECT COUNT(*) FROM m6_counters", ()).await, 0);
}

/// Each gate gets its own independently monotone series — eight distinct names,
/// and bumping one moves exactly one.
#[tokio::test]
async fn every_gate_has_its_own_counter_series() {
    let names: Vec<String> = CasGate::ALL
        .iter()
        .map(|gate| refusal_counter(*gate))
        .collect();
    let mut unique = names.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(unique.len(), 8, "eight gates, eight counter names");

    let db = GenesisDb::new().await;
    db.seed_space(SPACE_ID, SPACE).await;
    let tx = db.tx().await;
    bump(
        &tx,
        SPACE_ID,
        SPACE,
        &refusal_counter(CasGate::RootDigest),
        1,
    )
    .await
    .expect("bump");
    let mut moved = Vec::new();
    for gate in CasGate::ALL {
        if read_counter(&tx, SPACE_ID, &refusal_counter(gate))
            .await
            .expect("read")
            .is_some()
        {
            moved.push(gate);
        }
    }
    tx.commit().await.unwrap();

    assert_eq!(moved, vec![CasGate::RootDigest]);
}

/// Counters are per space as well as per name.
#[tokio::test]
async fn counters_are_keyed_per_space() {
    let db = GenesisDb::new().await;
    db.seed_space(SPACE_ID, SPACE).await;

    let tx = db.tx().await;
    bump(&tx, SPACE_ID, SPACE, COUNTER_TURNS, 7)
        .await
        .expect("bump");
    let other = read_counter(&tx, "space-id-b", COUNTER_TURNS)
        .await
        .expect("read");
    tx.commit().await.unwrap();

    assert_eq!(other, None);
}

// ---------------------------------------------------------------------------
// §7.2 statistics
// ---------------------------------------------------------------------------

/// **S0-137's vacuous case.** A space nobody has touched reads as all zeros and
/// empty maps rather than as an error — the shape the loop sees on every turn
/// against a quiet install.
#[tokio::test]
async fn an_untouched_space_observes_the_default() {
    let db = GenesisDb::new().await;

    let tx = db.tx().await;
    let stats = observe_space(&tx, SPACE_ID, SPACE, EPOCH)
        .await
        .expect("stats");
    tx.commit().await.unwrap();

    assert_eq!(stats, ShadowStats::default());
}

/// **Every field driven off its default.** Without this, a reader that returned
/// `ShadowStats::default()` would satisfy every other assertion in this module.
#[tokio::test]
async fn observe_space_reads_every_statistic() {
    let db = GenesisDb::new().await;
    db.seed_space(SPACE_ID, SPACE).await;
    db.open_epoch(SPACE_ID, SPACE).await;

    // One frontier row due now, one deferred into the future.
    db.exec(
        "INSERT INTO genesis_frontier
             (space, independence_group_id, coverage_epoch, first_seen_at, next_scan_at)
         VALUES (?1, 'group-due', ?2, unixepoch(), unixepoch()),
                (?1, 'group-later', ?2, unixepoch(), unixepoch() + 3600)",
        libsql::params![SPACE, EPOCH],
    )
    .await;
    db.exec(
        "INSERT INTO genesis_group_coverage
             (space, independence_group_id, coverage_epoch, page_id, candidate_id, covered_at)
         VALUES (?1, 'group-covered', ?2, 'page-1', 'cand-1', unixepoch())",
        libsql::params![SPACE, EPOCH],
    )
    .await;
    db.exec(
        "INSERT INTO genesis_suppression
             (space, independence_group_id, coverage_epoch, page_id, reason,
              first_seen_at, suppressed_at, expires_at, lapsed_at)
         VALUES (?1, 'group-sup', ?2, 'page-1', 'churn',
                 unixepoch(), unixepoch(), unixepoch() + 600, NULL)",
        libsql::params![SPACE, EPOCH],
    )
    .await;
    db.exec(
        "INSERT INTO genesis_quarantine
             (space, independence_group_id, coverage_epoch, reason,
              first_seen_at, quarantined_at, lifted_at)
         VALUES (?1, 'group-q', ?2, 'flapping', unixepoch(), unixepoch(), NULL)",
        libsql::params![SPACE, EPOCH],
    )
    .await;
    db.exec(
        "INSERT INTO genesis_candidates
             (candidate_id, slot_id, page_id, space, signal_kind, coverage_epoch,
              input_generation, active_root_digest, state, created_at, updated_at)
         VALUES ('cand-1', 'slot-1', 'page-1', ?1, 'space-overview', ?2, 1, 'd', 'prepared',
                 unixepoch(), unixepoch()),
                ('cand-2', 'slot-2', 'page-2', ?1, 'space-overview', ?2, 1, 'd', 'stale',
                 unixepoch(), unixepoch())",
        libsql::params![SPACE, EPOCH],
    )
    .await;

    let tx = db.tx().await;
    bump(&tx, SPACE_ID, SPACE, COUNTER_DRY_RUN_PASSED, 2)
        .await
        .expect("bump passes");
    bump(&tx, SPACE_ID, SPACE, COUNTER_ORACLE_DIVERGENCE, 1)
        .await
        .expect("bump divergence");
    bump(
        &tx,
        SPACE_ID,
        SPACE,
        &refusal_counter(CasGate::CoverageEpoch),
        3,
    )
    .await
    .expect("bump refusal");
    let stats = observe_space(&tx, SPACE_ID, SPACE, EPOCH)
        .await
        .expect("stats");
    tx.commit().await.unwrap();

    assert_eq!(stats.frontier_size, 2);
    assert_eq!(stats.deferred, 1);
    assert_eq!(stats.admitted, 1);
    assert_eq!(stats.suppressed, 1);
    assert_eq!(stats.quarantined, 1);
    assert_eq!(
        stats.candidates_by_state.get(&CandidateState::Prepared),
        Some(&1)
    );
    assert_eq!(
        stats.candidates_by_state.get(&CandidateState::Stale),
        Some(&1)
    );
    assert_eq!(stats.candidates_by_state.len(), 2, "only states with rows");
    assert_eq!(
        stats.refusals_by_gate.get(&CasGate::CoverageEpoch),
        Some(&3)
    );
    assert_eq!(stats.refusals_by_gate.len(), 1);
    assert_eq!(stats.dry_runs_passed, 2);
    assert_eq!(stats.oracle_divergences, 1);
    // The one field that stays at its default, read through the same reader as
    // the rest — so a broken read would show up as a zero here too, and the
    // twelve non-zeros above are what rule that out.
    assert_eq!(stats.m6_mutation_count, 0);
}

/// A lapsed suppression and a lifted quarantine are history, not live state.
#[tokio::test]
async fn lapsed_and_lifted_rows_leave_the_live_counts() {
    let db = GenesisDb::new().await;
    db.exec(
        "INSERT INTO genesis_suppression
             (space, independence_group_id, coverage_epoch, page_id, reason,
              first_seen_at, suppressed_at, expires_at, lapsed_at)
         VALUES (?1, 'group-sup', ?2, 'page-1', 'churn',
                 unixepoch(), unixepoch(), unixepoch() + 600, unixepoch() + 900)",
        libsql::params![SPACE, EPOCH],
    )
    .await;
    db.exec(
        "INSERT INTO genesis_quarantine
             (space, independence_group_id, coverage_epoch, reason,
              first_seen_at, quarantined_at, lifted_at)
         VALUES (?1, 'group-q', ?2, 'flapping', unixepoch(), unixepoch(), unixepoch())",
        libsql::params![SPACE, EPOCH],
    )
    .await;

    let tx = db.tx().await;
    let stats = observe_space(&tx, SPACE_ID, SPACE, EPOCH)
        .await
        .expect("stats");
    tx.commit().await.unwrap();

    assert_eq!(stats.suppressed, 0);
    assert_eq!(stats.quarantined, 0);
}

/// The statistics reader is a pure read — it must not create the counter rows
/// it fails to find, or observing a space would change it.
#[tokio::test]
async fn observing_a_space_writes_nothing() {
    let db = GenesisDb::new().await;
    db.seed_space(SPACE_ID, SPACE).await;
    db.open_epoch(SPACE_ID, SPACE).await;

    let tx = db.tx().await;
    observe_space(&tx, SPACE_ID, SPACE, EPOCH)
        .await
        .expect("stats");
    tx.commit().await.unwrap();

    assert_eq!(db.scalar("SELECT COUNT(*) FROM m6_counters", ()).await, 0);
    assert_eq!(
        db.scalar("SELECT COUNT(*) FROM m6_readiness", ()).await,
        0,
        "observing is not noting"
    );
}

/// S0-11 — the clock is SQLite's, read in-statement.
#[tokio::test]
async fn the_clock_comes_from_the_database() {
    let db = GenesisDb::new().await;
    let tx = db.tx().await;
    let observed = sqlite_now(&tx).await.expect("now");
    tx.commit().await.unwrap();

    let direct = db.scalar("SELECT unixepoch()", ()).await;
    assert!(
        (direct - observed).abs() <= 2,
        "{observed} is not the database's clock ({direct})"
    );
}
