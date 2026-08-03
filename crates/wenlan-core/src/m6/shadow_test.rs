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

/// **The §7.3 benchmark's regression gate.** A space whose frontier exceeds one
/// scan slice must still reach quiescence.
///
/// This is the shape that made the loop sweep forever: `scan_slice` returns a
/// cursor whenever it fills, the sweep wraps, and the next visit starts over —
/// so counting a cursor advance as work pinned a large space at the 100ms
/// cadence permanently. `drive` panics if quiescence is not reached, so the
/// assertion is the call itself; 520 groups is one row past
/// `FRONTIER_SCAN_SLICE_ROWS`, which is the smallest corpus that reproduces it.
#[tokio::test]
async fn a_frontier_larger_than_one_scan_slice_reaches_quiescence() {
    let db = GenesisDb::new().await;
    db.seed_space(SPACE_ID, SPACE).await;
    // Bulk-generated rather than 520 calls to `seed_evidence`: the shape under
    // test is the row count, and a thousand round trips would make the gate
    // slow enough that someone would delete it.
    db.exec(
        "WITH RECURSIVE n(i) AS (SELECT 0 UNION ALL SELECT i + 1 FROM n WHERE i < 519)
         INSERT INTO provenance_roots (root_id, root_kind, independence_group_id, status)
         SELECT 'root-' || i, 'document_ingest', 'group-' || i, 'active' FROM n",
        (),
    )
    .await;
    db.exec(
        "WITH RECURSIVE n(i) AS (SELECT 0 UNION ALL SELECT i + 1 FROM n WHERE i < 519)
         INSERT INTO edges (edge_id, src_kind, src_id, dst_kind, dst_id,
                            grounded, root_id, space, valid_until)
         SELECT 'root-' || i, 'entity', 'entity-' || i, 'entity', 'peer',
                1, 'root-' || i, ?1, NULL FROM n",
        libsql::params![SPACE],
    )
    .await;

    let turns = drive(&db, 64).await;

    assert_eq!(
        db.scalar("SELECT COUNT(*) FROM genesis_frontier", ()).await,
        520,
        "the corpus must exceed one 512-row slice, or the gate is vacuous"
    );
    assert!(
        turns
            .iter()
            .any(|turn| matches!(turn, GenesisTurn::Frontier { .. })),
        "the sweep must have reported repair work at least once: {turns:?}"
    );
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

// ---------------------------------------------------------------------------
// G4.9 — liveness: backpressure must never preempt draining
// ---------------------------------------------------------------------------
//
// **Everything above this line is a safety gate, and safety gates pass
// trivially on a wedged loop.** A shadow that spins forever without progressing
// writes nothing outside M6, never moves `m6_mutation_count`, binds no card,
// writes no suppression, and reports exactly one unit of work per turn — so all
// twelve of the tests above stay green while the lane makes no progress at all
// and holds the shared connection mutex ten times a second. The gates below are
// the other half: the loop must *get somewhere*, and no space may starve
// another.
//
// The defect they were written against: `prepare` refusing on the D8 pending
// cap used to return from the whole turn, which skipped priority 4 — the only
// step that moves a candidate out of a pending state. The cap blocked the
// prepare, the prepare blocked the drain, and nothing cleared the cap. The
// refusal also counted as work (100ms cadence) and left the rotation pinned to
// the wedged space. Three individually-correct pieces, composed into a
// livelock; only a driver-level test can see it.

/// Fill `space` with pending filler candidates until the D8 cap would refuse.
///
/// Written directly rather than through `observe_candidate` because the point is
/// to reach the cap, not to re-test the cap — `candidates_test.rs:269` already
/// proves `observe_candidate` refuses at 128 and `frontier_test.rs:300` proves
/// the refused group keeps its frontier row. `observed` is the cheapest of the
/// five `PENDING_STATES` to fabricate and counts identically.
async fn fill_pending_to_cap(db: &GenesisDb, space: &str) {
    let live: i64 = db
        .scalar(
            "SELECT COUNT(*) FROM genesis_candidates
              WHERE space = ?1
                AND state IN ('observed','prepared','inferencing','validating','retry_wait')",
            libsql::params![space],
        )
        .await;
    for filler in live..super::constants::PENDING_CANDIDATES_PER_SPACE_CAP as i64 {
        db.exec(
            "INSERT INTO genesis_candidates (
                 candidate_id, slot_id, page_id, space, signal_kind,
                 coverage_epoch, input_generation, active_root_digest,
                 state, attempt, created_at, updated_at
             ) VALUES (?1, ?2, ?3, ?4, 'space-overview',
                       1, 0, 'digest', 'observed', 0, unixepoch(), unixepoch())",
            libsql::params![
                format!("filler-cand-{filler}"),
                format!("filler-slot-{filler}"),
                // Distinct per filler: D4's exclusive concept claims are partial
                // unique indexes, and 127 rows sharing one page id would be
                // refused by an invariant this fixture is not trying to test.
                format!("filler-page-{filler}"),
                space
            ],
        )
        .await;
    }
}

/// Tick until a candidate is prepared, leaving `state` mid-rotation so the
/// caller keeps ticking the same loop rather than a fresh one.
async fn drive_to_prepared(db: &GenesisDb, state: &mut ShadowState, max_turns: usize) -> String {
    for _ in 0..max_turns {
        let turn = run_genesis_shadow_turn(&db.connection, state)
            .await
            .expect("shadow turn");
        if let GenesisTurn::Prepared { candidate_id } = turn {
            return candidate_id;
        }
    }
    panic!("no candidate was prepared in {max_turns} turns");
}

/// Advance the space's coverage epoch — the shipped way a slot re-proposes.
///
/// `candidate_id` is `identity::candidate_id(slot_id, coverage_epoch)`, so a new
/// epoch mints a *different* id for the same slot. This matters because
/// `observe_candidate` checks "does this id already exist" **before** it checks
/// the cap: re-proposing an unchanged slot returns `AlreadyPresent` and never
/// reaches the cap at all. Without an epoch advance, a "capped space" test is
/// vacuous — the prepare returns `None` for want of new work rather than
/// refusing, and finalization is reached whether or not the livelock is fixed.
async fn advance_coverage_epoch(db: &GenesisDb, space_id: &str) {
    db.exec(
        "UPDATE genesis_coverage_state SET coverage_epoch = coverage_epoch + 1
          WHERE space_id = ?1",
        libsql::params![space_id],
    )
    .await;
}

/// Candidate rows the space has minted at or past `epoch` — the observable
/// witness for whether a prepare inserted anything.
async fn candidates_at_epoch(db: &GenesisDb, space: &str, epoch: i64) -> i64 {
    db.scalar(
        "SELECT COUNT(*) FROM genesis_candidates WHERE space = ?1 AND coverage_epoch >= ?2",
        libsql::params![space, epoch],
    )
    .await
}

/// **G4.9.1 — a space at the pending cap still reaches finalization.**
///
/// The scene is the one the M5 shadow measured, not a contrivance: 13 of 98
/// evaluated pages were `supported`, so ~87% of page-mediated candidates refuse
/// at E-8 and requeue into `retry_wait` — a pending state. The cap fills with
/// the refusals of the work it is refusing to replace.
///
/// The three conditions the livelock needs must hold *simultaneously*, and each
/// is asserted rather than assumed: the space is at the cap, a **fresh** proposal
/// is arriving (so the cap genuinely refuses instead of returning
/// `AlreadyPresent`), and a prepared candidate is due. The refusal is observed
/// through its effect — no candidate row appears at the new epoch — and
/// [`a_fresh_proposal_below_the_cap_does_mint_a_row`] is the paired control that
/// makes that zero mean *refused* rather than *never proposed*.
///
/// RED control (source mutation, run by hand — a control-flow defect has no data
/// mutation that reproduces it): restore `Some(GenesisTurn::RefusedBudget)` as
/// an early `return` from `run_genesis_shadow_turn` and this test fails, because
/// the turn never reaches priority 4 and the envelope keeps its `None` verdict.
#[tokio::test]
async fn a_capped_space_still_finalizes_the_work_it_holds() {
    let db = scene().await;
    let mut state = ShadowState::default();
    let candidate_id = drive_to_prepared(&db, &mut state, 64).await;

    // Precondition 1: the candidate is finalizable — stamped, no verdict.
    let tx = db.tx().await;
    let envelope = super::finalize::read_envelope(&tx, &candidate_id)
        .await
        .expect("read envelope")
        .expect("the prepare stamped an envelope");
    tx.commit().await.unwrap();
    assert!(
        envelope.verdict.is_none(),
        "precondition: the candidate has not been finalized yet"
    );

    // Precondition 2: the space is at the D8 cap.
    fill_pending_to_cap(&db, SPACE).await;
    assert_eq!(
        db.scalar(
            "SELECT COUNT(*) FROM genesis_candidates
              WHERE space = ?1
                AND state IN ('observed','prepared','inferencing','validating','retry_wait')",
            libsql::params![SPACE],
        )
        .await,
        super::constants::PENDING_CANDIDATES_PER_SPACE_CAP as i64,
        "precondition: the space is at the D8 cap"
    );

    // Precondition 3: new work is arriving, so the cap has something to refuse.
    advance_coverage_epoch(&db, SPACE_ID).await;
    assert_eq!(
        candidates_at_epoch(&db, SPACE, 2).await,
        0,
        "precondition: the re-proposal has not been minted yet"
    );

    // Drive past the recovery scan and the frontier slices the epoch advance
    // dirtied. The loop does one unit of work per turn by design (§4.2), so the
    // property under test is that finalization is *reached*, not that it lands
    // on any particular turn. With the livelock present this loop breaks on the
    // first `RefusedBudget` and the assertions below fail — which is the point.
    let mut turn = GenesisTurn::Idle;
    for _ in 0..64 {
        turn = run_genesis_shadow_turn(&db.connection, &mut state)
            .await
            .expect("shadow turn");
        if !matches!(turn, GenesisTurn::Idle | GenesisTurn::Frontier { .. }) {
            break;
        }
    }

    assert_eq!(
        candidates_at_epoch(&db, SPACE, 2).await,
        0,
        "the cap must still have refused the arriving proposal"
    );
    assert!(
        matches!(
            &turn,
            GenesisTurn::DryRunPassed { candidate_id: id } | GenesisTurn::DryRunRefused { candidate_id: id, .. }
            if id == &candidate_id
        ),
        "a full cap must not preempt finalization; got {turn:?}"
    );
    let tx = db.tx().await;
    let after = super::finalize::read_envelope(&tx, &candidate_id)
        .await
        .expect("read envelope")
        .expect("envelope still present");
    tx.commit().await.unwrap();
    assert!(
        after.verdict.is_some(),
        "the turn must have recorded a verdict — durable progress, not a retry"
    );
}

/// The paired control for G4.9.1's "no row at the new epoch" assertion.
///
/// Identical scene and identical epoch advance, minus the cap fill. A row *does*
/// appear, which is what proves the arriving proposal is real and admitted — so
/// G4.9.1's zero is the cap refusing new work, not a space with nothing to say.
#[tokio::test]
async fn a_fresh_proposal_below_the_cap_does_mint_a_row() {
    let db = scene().await;
    let mut state = ShadowState::default();
    drive_to_prepared(&db, &mut state, 64).await;
    advance_coverage_epoch(&db, SPACE_ID).await;
    assert_eq!(candidates_at_epoch(&db, SPACE, 2).await, 0);

    for _ in 0..8 {
        run_genesis_shadow_turn(&db.connection, &mut state)
            .await
            .expect("shadow turn");
    }

    assert!(
        candidates_at_epoch(&db, SPACE, 2).await > 0,
        "a fresh proposal below the cap must mint a candidate row"
    );
}

/// **G4.9.2 — a budget refusal is not durable work.**
///
/// Mechanical control for the second fix. `did_work` drives the loop's cadence,
/// so a `RefusedBudget` counted as work is a 10 Hz poll on the shared connection
/// mutex for as long as the cap stays full. S0-9 already says a budget refusal
/// is not a retry; it is not work either.
///
/// RED control: restore `!matches!(self, GenesisTurn::Idle)` and this fails.
#[tokio::test]
async fn a_budget_refusal_is_not_durable_work() {
    assert!(
        !GenesisTurn::RefusedBudget.did_work(),
        "a turn that wrote nothing must take the idle delay"
    );
    assert!(!GenesisTurn::Idle.did_work());
    // Positive control, or the two above pass on a `did_work` that is always
    // false — which would peg the loop at 1s and look like a fix.
    assert!(GenesisTurn::Prepared {
        candidate_id: "c".into()
    }
    .did_work());
    assert!(GenesisTurn::DryRunPassed {
        candidate_id: "c".into()
    }
    .did_work());
}

/// **G4.9.3 — a real backpressure turn reports no durable work.**
///
/// The cap is filled *before* the proposal's candidate is ever observed, which
/// is the only shape that produces a genuine refusal: `observe_candidate` checks
/// "does this `candidate_id` already exist" before it checks the cap, so a
/// candidate the loop already minted comes back `AlreadyPresent` and never
/// reaches the cap at all. Filling first is therefore not a contrivance — it is
/// the arrival of new work at a full space, which is exactly the M5-measured
/// case where refusals accumulate as `retry_wait` faster than they drain.
///
/// This also proves `RefusedBudget` is **reachable from the driver**, without
/// which G4.9.2 would be asserting a property of a value nothing produces.
#[tokio::test]
async fn a_capped_space_refusing_new_work_reports_no_durable_work() {
    let db = scene().await;
    fill_pending_to_cap(&db, SPACE).await;

    // Drive past the recovery scan and the frontier repairs; the first turn that
    // is neither is the prepare, and it must refuse on the cap.
    let mut state = ShadowState::default();
    let mut turn = GenesisTurn::Idle;
    for _ in 0..64 {
        turn = run_genesis_shadow_turn(&db.connection, &mut state)
            .await
            .expect("shadow turn");
        if !matches!(turn, GenesisTurn::Idle | GenesisTurn::Frontier { .. }) {
            break;
        }
    }

    assert_eq!(
        turn,
        GenesisTurn::RefusedBudget,
        "new work arriving at a full cap is backpressure"
    );
    assert!(
        !turn.did_work(),
        "backpressure must take the 1s delay, not the 100ms one"
    );
}

/// **G4.9.4 — a space below the cap still prepares.**
///
/// The discriminating positive control (S0-155). Without it, "the capped space
/// reaches finalization" is also satisfied by a fix that simply stopped
/// enforcing the cap, which would be a worse defect than the one being fixed.
#[tokio::test]
async fn a_space_below_the_cap_still_prepares_and_the_cap_still_refuses() {
    let db = scene().await;
    let turns = drive(&db, 64).await;
    assert!(
        turns
            .iter()
            .any(|turn| matches!(turn, GenesisTurn::Prepared { .. })),
        "below the cap, a prepare must still happen: {turns:?}"
    );
    // And the cap is still a cap: fill it and the next observe refuses.
    fill_pending_to_cap(&db, SPACE).await;
    let tx = db.tx().await;
    let outcome = super::candidates::observe_candidate(
        &tx,
        &super::candidates::ObservedCandidate {
            candidate_id: "past-the-cap",
            slot_id: "past-the-cap-slot",
            page_id: "",
            space: SPACE,
            signal_kind: super::identity::SignalKind::SpaceOverview,
            coverage_epoch: 1,
            input_generation: 0,
            active_root_digest: "digest",
        },
    )
    .await
    .expect("observe");
    tx.commit().await.unwrap();
    assert_eq!(
        outcome,
        super::candidates::ObserveOutcome::RefusedPendingCap,
        "the fix must not have disabled the cap"
    );
}

/// **G4.9.5 — vacuous case (S0-137): a full cap with no admitted proposal.**
///
/// Nothing to refuse, nothing to finalize. The turn must report idle and the
/// rotation must advance, rather than the cap manufacturing a `RefusedBudget`
/// out of a space that never proposed anything.
#[tokio::test]
async fn a_capped_space_with_no_proposal_is_idle_and_rotates() {
    let db = GenesisDb::new().await;
    db.seed_space(SPACE_ID, SPACE).await;
    // No evidence at all: below D1's three-group independence floor, so no
    // signal admits and `prepare` never reaches `observe_candidate`.
    fill_pending_to_cap(&db, SPACE).await;

    let mut state = ShadowState::default();
    let mut turns = Vec::new();
    for _ in 0..4 {
        turns.push(
            run_genesis_shadow_turn(&db.connection, &mut state)
                .await
                .expect("shadow turn"),
        );
    }

    assert!(
        turns[1..].iter().all(|turn| turn == &GenesisTurn::Idle),
        "a capped space with nothing to propose is idle, not refusing: {turns:?}"
    );
}

/// **G4.9.6 — one busy space cannot starve the rotation.**
///
/// The property `state.rotation` advancing only on `Idle` silently broke. Three
/// spaces, the first permanently at the cap with a proposal it cannot observe;
/// every other space must still get a turn. Asserted by observable effect —
/// each space's coverage row is opened by the turn that visits it — rather than
/// by reading the private rotation counter.
#[tokio::test]
async fn a_busy_space_cannot_starve_the_rotation() {
    let db = GenesisDb::new().await;
    // `spaces()` orders by name, so "space-a" is visited first and is the one
    // held at the cap.
    for (id, name) in [
        (SPACE_ID, SPACE),
        ("space-id-b", "space-b"),
        ("space-id-c", "space-c"),
    ] {
        db.seed_space(id, name).await;
        for group in 0..3 {
            for slot in 0..2 {
                db.seed_evidence(
                    &format!("{name}-root-{group}-{slot}"),
                    name,
                    &format!("{name}-group-{group}"),
                    &format!("{name}-entity-{group}-{slot}"),
                )
                .await;
            }
        }
    }
    fill_pending_to_cap(&db, SPACE).await;

    let mut state = ShadowState::default();
    for _ in 0..32 {
        run_genesis_shadow_turn(&db.connection, &mut state)
            .await
            .expect("shadow turn");
    }

    assert_eq!(
        db.scalar("SELECT COUNT(*) FROM genesis_coverage_state", ())
            .await,
        3,
        "every space must have been visited; a wedged first space starves the rest"
    );
}
