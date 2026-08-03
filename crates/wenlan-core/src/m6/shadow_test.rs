// SPDX-License-Identifier: Apache-2.0
//! The shadow loop, driven end to end — **G4.4** (the dry run writes no page)
//! and the §10.2 zero-mutation proof.
//!
//! These run the real `run_genesis_shadow_turn` against the shipped substrate,
//! not a hand-assembled sequence of its parts, because the property under test
//! is about what the *driver* can reach. A test that called the pieces
//! individually would prove the pieces are dry and say nothing about the loop
//! that composes them.
//!
//! **The zero-mutation claim is proved, not asserted** (§10.2). Two independent
//! mechanisms carry it:
//!
//! 1. A **fingerprint** of every primary-evidence table — the contents, not
//!    just the row counts — taken before the run and compared after. A count
//!    comparison would miss an in-place update, which is the mutation class a
//!    shadow is most likely to commit by accident.
//! 2. `genesis_coverage_state.m6_mutation_count`, which four migration-109
//!    triggers make monotone, non-resettable, identity-pinned, and
//!    undeletable. An unchanged value there is a statement about **every**
//!    instant between the two reads, not about the two reads.
//!
//! The fingerprint alone could be defeated by a write-then-revert; the counter
//! alone cannot see a mutation nobody instrumented. Together they close both.

use super::evidence;
use super::genesis_test_support::GenesisDb;
use super::refresh_readiness::{readiness_fence, ReadinessKey, ReadinessPhase};
use super::shadow::{run_genesis_shadow_turn, GenesisTurn, ShadowState};

const SPACE: &str = "space-a";
const SPACE_ID: &str = "space-id-a";

/// Every primary-evidence table the shadow could conceivably reach, plus the
/// two projection tables §10.2 names. Fingerprinted whole.
///
/// `pages`, `page_*`, `edges`, and `provenance_roots` are the reader-visible
/// substrate; `page_projection_outbox` and `m6_genesis_provenance` are the
/// publication path. `entities` / `relations` / `observations` / `memories` /
/// `chunks` are absent from this fixture by construction — the M6 readers do
/// not name them, so the fixture never created them, and a statement against
/// one would fail as "no such table" rather than pass silently.
const GUARDED_TABLES: [&str; 8] = [
    "pages",
    "page_links",
    "page_truth_state",
    "page_version_claims",
    "claim_anchors",
    "edges",
    "provenance_roots",
    "page_projection_outbox",
];

/// A content fingerprint of the guarded tables: every row of every table,
/// ordered, concatenated. Equality across a shadow run is the mutation proof.
async fn fingerprint(db: &GenesisDb) -> String {
    let mut out = String::new();
    for table in GUARDED_TABLES {
        let mut rows = db
            .connection
            .query(&format!("SELECT * FROM {table}"), ())
            .await
            .unwrap_or_else(|error| panic!("fingerprint {table}: {error}"));
        // Every column as a typed `Value`, not a string: `NULL` vs `''` and
        // `1` vs `'1'` are different mutations and must fingerprint differently.
        let columns = rows.column_count();
        let mut lines = Vec::new();
        while let Some(row) = rows.next().await.expect("fingerprint row") {
            let mut cells = Vec::new();
            for column in 0..columns {
                cells.push(format!(
                    "{:?}",
                    row.get_value(column).expect("column value")
                ));
            }
            lines.push(cells.join(","));
        }
        lines.sort();
        out.push_str(table);
        out.push('=');
        out.push_str(&lines.join("|"));
        out.push('\n');
    }
    out
}

/// The single text cell a one-row, one-column query returns.
async fn one_string(db: &GenesisDb, sql: &str) -> String {
    let mut rows = db
        .connection
        .query(sql, ())
        .await
        .unwrap_or_else(|error| panic!("{sql}: {error}"));
    rows.next()
        .await
        .expect("read row")
        .expect("row present")
        .get(0)
        .expect("decode text cell")
}

/// Three groups, two roots each — clears D1's three-group floor, so the
/// space-overview signal admits and the loop has real work to reach quiescence
/// on. A store with nothing to do would make every "wrote nothing" assertion
/// vacuous, which is the trap S0-137 names.
async fn scene() -> GenesisDb {
    let db = GenesisDb::new().await;
    db.seed_space(SPACE_ID, SPACE).await;
    for group in 0..3 {
        for slot in 0..2 {
            db.seed_evidence(
                &format!("root-{group}-{slot}"),
                SPACE,
                &format!("group-{group}"),
                &format!("entity-{group}-{slot}"),
            )
            .await;
        }
    }
    db
}

/// Tick until the loop reports `Idle` twice running, or give up.
///
/// Two consecutive idles rather than one: a single `Idle` is also what a turn
/// returns while it advances the space rotation, so stopping at the first would
/// call a store quiescent before the rotation had visited it.
async fn drive(db: &GenesisDb, max_turns: usize) -> Vec<GenesisTurn> {
    let mut state = ShadowState::default();
    let mut turns = Vec::new();
    let mut idle_run = 0;
    for _ in 0..max_turns {
        let turn = run_genesis_shadow_turn(&db.connection, &mut state)
            .await
            .expect("shadow turn");
        idle_run = if turn.did_work() { 0 } else { idle_run + 1 };
        turns.push(turn);
        if idle_run >= 2 && turns.len() > 2 {
            return turns;
        }
    }
    panic!("the shadow loop did not reach quiescence in {max_turns} turns: {turns:?}");
}

// ---------------------------------------------------------------------------
// G4.4 — the dry run writes no page
// ---------------------------------------------------------------------------

/// **G4.4** — a full shadow run to quiescence mutates nothing outside M6.
///
/// The positive control is inside the same test on purpose: the run must have
/// *done something* (a prepare and a passing dry run both appear in the turn
/// log) before "nothing changed" means anything. A shadow that crashed on turn
/// one would also leave the fingerprint intact.
#[tokio::test]
async fn a_full_shadow_run_mutates_nothing_outside_m6() {
    let db = scene().await;
    let before = fingerprint(&db).await;

    let turns = drive(&db, 64).await;

    assert!(
        turns
            .iter()
            .any(|turn| matches!(turn, GenesisTurn::Prepared { .. })),
        "the run must have prepared a candidate: {turns:?}"
    );
    assert!(
        turns
            .iter()
            .any(|turn| matches!(turn, GenesisTurn::DryRunPassed { .. })),
        "the run must have reached a passing dry run: {turns:?}"
    );
    assert_eq!(
        before,
        fingerprint(&db).await,
        "the shadow mutated primary evidence"
    );
    assert_eq!(
        db.scalar("SELECT COUNT(*) FROM page_projection_outbox", ())
            .await,
        0,
        "zero projection hand-offs, asserted against the real table"
    );
    assert_eq!(
        db.scalar("SELECT COUNT(*) FROM m6_genesis_provenance", ())
            .await,
        0
    );
}

/// §10.2 proof 2 — the monotone counter never moved, and neither did the flag.
///
/// Split from the fingerprint test because the two prove different things: the
/// fingerprint is a before/after comparison a write-then-revert could defeat,
/// while the trigger-guarded counter cannot be decreased, reset, re-pointed, or
/// deleted, so an unchanged value covers the whole interval.
#[tokio::test]
async fn the_mutation_counter_and_the_enable_flag_never_move() {
    let db = scene().await;
    drive(&db, 64).await;

    assert_eq!(
        db.scalar(
            "SELECT COUNT(*) FROM genesis_coverage_state WHERE m6_mutation_count <> 0",
            (),
        )
        .await,
        0,
        "m6_mutation_count moved on some space"
    );
    assert_eq!(
        db.scalar(
            "SELECT COUNT(*) FROM genesis_coverage_state WHERE genesis_enabled <> 0",
            (),
        )
        .await,
        0,
        "the shadow enabled genesis on some space"
    );
    // The coverage row must exist, or the two zeros above are vacuous.
    assert_eq!(
        db.scalar("SELECT COUNT(*) FROM genesis_coverage_state", ())
            .await,
        1
    );
}

/// **Strict whole-page suppression is not reintroduced through this lane.** The
/// shadow writes no suppression row and binds no card, so no verdict it reaches
/// can hide a page. One unsupported claim still hides nothing.
#[tokio::test]
async fn the_shadow_writes_no_suppression_and_binds_no_card() {
    let db = scene().await;
    drive(&db, 64).await;

    for table in ["genesis_suppression", "genesis_card_binding"] {
        assert_eq!(
            db.scalar(&format!("SELECT COUNT(*) FROM {table}"), ())
                .await,
            0,
            "{table} must stay empty"
        );
    }
}

// ---------------------------------------------------------------------------
// Vacuous cases (S0-137)
// ---------------------------------------------------------------------------

/// §8's vacuous case for G4: **a candidate set of size zero produces zero
/// verdicts and zero `page_projection_outbox` rows.** An install with no spaces
/// is the state every daemon starts in, so this is the shape the loop spends
/// most of its life in.
#[tokio::test]
async fn an_empty_store_is_idle_and_writes_nothing() {
    let db = GenesisDb::new().await;
    let before = fingerprint(&db).await;
    let mut state = ShadowState::default();

    for _ in 0..8 {
        let turn = run_genesis_shadow_turn(&db.connection, &mut state)
            .await
            .expect("shadow turn");
        assert_eq!(turn, GenesisTurn::Idle, "an empty store has nothing to do");
    }

    assert_eq!(before, fingerprint(&db).await);
    assert_eq!(
        db.scalar("SELECT COUNT(*) FROM genesis_candidates", ())
            .await,
        0
    );
    assert_eq!(
        db.scalar("SELECT COUNT(*) FROM page_projection_outbox", ())
            .await,
        0
    );
}

/// A space with too little evidence to clear D1's floor admits nothing and
/// still writes no candidate — the "zero grounded edges admit zero candidates"
/// shape one rung up.
#[tokio::test]
async fn a_space_below_the_independence_floor_prepares_nothing() {
    let db = GenesisDb::new().await;
    db.seed_space(SPACE_ID, SPACE).await;
    db.seed_evidence("root-only", SPACE, "group-0", "entity-0")
        .await;

    let turns = drive(&db, 32).await;

    assert!(
        !turns
            .iter()
            .any(|turn| matches!(turn, GenesisTurn::Prepared { .. })),
        "a one-group space must not clear the floor: {turns:?}"
    );
    assert_eq!(
        db.scalar("SELECT COUNT(*) FROM genesis_candidates", ())
            .await,
        0
    );
}

// ---------------------------------------------------------------------------
// Loop bounds and restart-resumability (§4.2, §4.4, §4.5)
// ---------------------------------------------------------------------------

/// §4.2 priority 1 — the startup recovery scan runs **once per process**,
/// eagerly, before any other work.
///
/// Observable rather than inferred: the very first turn is the scan (it reports
/// `Idle` and leaves a report behind), and no later turn produces a second one,
/// which a re-running scan would.
#[tokio::test]
async fn the_startup_recovery_scan_runs_once_before_anything_else() {
    let db = scene().await;
    let mut state = ShadowState::default();
    assert!(state.recovery_report().is_none());

    let first = run_genesis_shadow_turn(&db.connection, &mut state)
        .await
        .expect("first turn");
    assert_eq!(first, GenesisTurn::Idle);
    let report = state
        .recovery_report()
        .cloned()
        .expect("the first turn is the recovery scan");
    assert_eq!(report.projections_handed_off, 0);
    assert_eq!(
        db.scalar("SELECT COUNT(*) FROM genesis_candidates", ())
            .await,
        0,
        "the scan turn does no candidate work"
    );

    for _ in 0..8 {
        run_genesis_shadow_turn(&db.connection, &mut state)
            .await
            .expect("later turn");
    }
    assert_eq!(
        state.recovery_report(),
        Some(&report),
        "the scan must not run again"
    );
}

/// §4.2 — a turn does at most one unit of work, so the turn log is a sequence
/// of single steps rather than one turn that did everything.
#[tokio::test]
async fn a_turn_reports_exactly_one_unit_of_work() {
    let db = scene().await;
    let turns = drive(&db, 64).await;

    let prepared = turns
        .iter()
        .filter(|turn| matches!(turn, GenesisTurn::Prepared { .. }))
        .count();
    let finalized = turns
        .iter()
        .filter(|turn| matches!(turn, GenesisTurn::DryRunPassed { .. }))
        .count();
    assert_eq!(prepared, 1, "one prepare, on its own turn: {turns:?}");
    assert_eq!(finalized, 1, "one dry run, on its own turn: {turns:?}");
}

/// §4.3 — the `genesis` lease spans prepare *and* finalize, and is released
/// when finalization completes. A quiescent shadow holds nothing, so a restart
/// waits on no TTL.
#[tokio::test]
async fn quiescence_leaves_no_lease_held() {
    let db = scene().await;
    drive(&db, 64).await;

    assert_eq!(
        db.scalar("SELECT COUNT(*) FROM grouping_leases", ()).await,
        0
    );
}

/// §4.5 — a restart costs at most one turn's work, and repeating that turn is
/// free of consequence.
///
/// **It does repeat the dry run, and that is the shipped machine, not a test
/// concession.** §4.3 releases the `genesis` lease when finalization completes,
/// so a dry-run-passed candidate sits in `prepared` with no live lease — which
/// is precisely the shape recovery's A7 rule reads as an interrupted worker. It
/// stales the candidate (`lease_lost`), `settle_groups_to_frontier` hands the
/// claims back, and the next turn re-derives the same candidate. Machine E has
/// no post-finalization state to move to, because the state that would follow
/// is `published` and PR-B publishes nothing.
///
/// What must hold, and is asserted here, is that the repeat is a **no-op in
/// identity terms**: the same deterministic `candidate_id`, so no second row;
/// the same receipt; `attempt` never charged (S0-9), so no number of restarts
/// can ratchet a candidate into `retry_exhausted`; and nothing written outside
/// M6.
#[tokio::test]
async fn restarts_repeat_a_turn_without_ratcheting_anything() {
    let db = scene().await;
    drive(&db, 64).await;
    let after_first = fingerprint(&db).await;
    let candidate_id = one_string(&db, "SELECT candidate_id FROM genesis_candidates").await;
    let receipt = one_string(&db, "SELECT receipt_id FROM genesis_candidates").await;

    // A brand new `ShadowState` is exactly what a process restart produces.
    for _ in 0..3 {
        let turns = drive(&db, 64).await;
        assert!(
            !turns.iter().any(|turn| matches!(
                turn,
                GenesisTurn::Parked { .. } | GenesisTurn::RefusedBudget
            )),
            "a restart must not park or budget-refuse anything: {turns:?}"
        );
    }

    assert_eq!(
        db.scalar("SELECT COUNT(*) FROM genesis_candidates", ())
            .await,
        1,
        "three restarts, one candidate row"
    );
    assert_eq!(
        one_string(&db, "SELECT candidate_id FROM genesis_candidates").await,
        candidate_id
    );
    assert_eq!(
        one_string(&db, "SELECT receipt_id FROM genesis_candidates").await,
        receipt,
        "the receipt is derived from the identity, so it survives every restart"
    );
    assert_eq!(
        db.scalar("SELECT attempt FROM genesis_candidates", ())
            .await,
        0,
        "a restart is not a retry (S0-9)"
    );
    assert_eq!(
        after_first,
        fingerprint(&db).await,
        "the restarts mutated primary evidence"
    );
}

// ---------------------------------------------------------------------------
// §7 — the evidence the loop records
// ---------------------------------------------------------------------------

/// §7.1 — the loop drives stage `B_genesis_shadow` to `preparing` and stops
/// there. `committed` is a cutover fact and PR-B cuts nothing over.
#[tokio::test]
async fn the_loop_reaches_preparing_and_never_committed() {
    let db = scene().await;
    drive(&db, 64).await;

    let tx = db.tx().await;
    let fence = readiness_fence(
        &tx,
        ReadinessKey {
            space: SPACE,
            stage: evidence::READINESS_STAGE,
            signal: evidence::READINESS_SIGNAL,
        },
    )
    .await
    .expect("fence")
    .expect("the loop initialises the readiness row");
    tx.commit().await.unwrap();

    assert_eq!(fence.phase, ReadinessPhase::Preparing);
    assert_eq!(fence.epoch, 1, "off → preparing increments the epoch once");
}

/// §7.2 — the statistics a turn records, read back through the same reader the
/// daemon would use. The zeros that matter (divergence, mutation count) are
/// asserted alongside a non-zero the run genuinely produced, so the reader is
/// demonstrably wired rather than returning a default struct.
#[tokio::test]
async fn the_loop_records_the_shadow_statistics() {
    let db = scene().await;
    drive(&db, 64).await;

    let tx = db.tx().await;
    let epoch = super::candidates::read_coverage_state(&tx, SPACE_ID)
        .await
        .expect("coverage")
        .expect("coverage row")
        .coverage_epoch;
    let stats = evidence::observe_space(&tx, SPACE_ID, SPACE, epoch)
        .await
        .expect("stats");
    tx.commit().await.unwrap();

    assert_eq!(stats.dry_runs_passed, 1);
    assert!(stats.refusals_by_gate.is_empty(), "{stats:?}");
    assert_eq!(stats.oracle_divergences, 0);
    assert_eq!(stats.m6_mutation_count, 0);
    assert_eq!(
        stats.candidates_by_state.values().sum::<i64>(),
        1,
        "one candidate, in exactly one state: {stats:?}"
    );
    assert_eq!(stats.suppressed, 0);
    assert_eq!(stats.quarantined, 0);
}
