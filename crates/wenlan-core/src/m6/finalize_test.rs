// SPDX-License-Identifier: Apache-2.0
//! The **G4** gate family (spec §8) — finalization is all-or-nothing, and the
//! dry run really is dry.
//!
//! Every row ships as a pair (S0-135/S0-155): a scripted concurrent mutation
//! that must go RED at a **named** gate, plus the discriminating positive
//! control that the *unmutated* scene passes all eight. The control is what
//! makes the refusals mean something — a gate that refused unconditionally
//! would satisfy every RED row here and prove nothing, and
//! [`the_unmutated_scene_passes_every_gate`] is the single assertion that rules
//! that out for the whole file.
//!
//! **Why "refuses at gate X" rather than "refuses":** §5.2 makes the *order*
//! load-bearing, and §7.2 records outcomes by first-failing gate. Asserting the
//! identity of the refusing gate is what turns each row into a test of one
//! condition (S0-154) instead of a test of the disjunction of eight.
//!
//! **Vacuous cases are asserted, not assumed** (S0-137). E-7 is vacuous for the
//! two signals that never read the community partition and E-8 is vacuous for a
//! candidate with no page-mediated root; both have an explicit vacuity test
//! *and* a non-vacuous positive control, because a gate that only ever passes
//! vacuously is not a gate.

use super::candidates::{
    self, CandidateState, ClaimReservation, EpochState, ObservedCandidate, PrepareOutcome,
};
use super::finalize::{self, dry_run_finalize, CasEnvelope, CasGate, Finalization, Verdict};
use super::genesis_test_support::GenesisDb;
use super::identity::{self, SignalKind};

const SPACE: &str = "space-a";
const SPACE_ID: &str = "space-id-a";
const SLOT: &str = "slot-a";
/// M4's lease/CAS generation in the base scene — E-2's stored input.
const GROUPING_GENERATION: i64 = 7;
/// M4's published community generation in the base scene — E-7's stored input.
const PUBLISHED_GENERATION: i64 = 3;

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

/// The candidate's claimed roots, in digest order.
///
/// A **concept**-role signal takes one root per group and a **witness**-role
/// one takes both. That is not a test convenience: `idx_genesis_group_claim` is
/// D4's per-group exclusion for concept claims, so a concept candidate holding
/// two roots of one group is refused by the index itself (B1). Witness claims
/// are excluded from both partial indexes by predicate, which is exactly what
/// lets overview candidates coexist without stealing evidence.
fn root_ids(signal: SignalKind) -> Vec<String> {
    let per_group = match candidates::claim_role_for(signal) {
        candidates::ClaimRole::Concept => 1,
        candidates::ClaimRole::Witness => 2,
    };
    (0..3)
        .flat_map(|group| (0..per_group).map(move |slot| format!("root-{group}-{slot}")))
        .collect()
}

/// Three groups, two roots each, all active and grounded; one M4
/// `space_graph_state` row; one `prepared` candidate holding a live genesis
/// lease and a stamped CAS envelope.
///
/// The candidate is a **space-overview** signal by default, which makes E-7
/// vacuous — `evidence_cluster_reads_the_real_community_gate` supplies the
/// non-vacuous half rather than this scene carrying a community dependency
/// every other row would then have to work around.
async fn scene(signal: SignalKind) -> (GenesisDb, i64, String) {
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
    db.exec(
        "INSERT INTO space_graph_state
             (space, graph_generation, grouping_generation, published_generation, dirty)
         VALUES (?1, 1, ?2, ?3, 0)",
        libsql::params![SPACE, GROUPING_GENERATION, PUBLISHED_GENERATION],
    )
    .await;
    let epoch = db.open_epoch(SPACE_ID, SPACE).await;
    let candidate_id = identity::candidate_id(SLOT, epoch);
    let roots = root_ids(signal);

    let tx = db.tx().await;
    let digest =
        identity::active_root_digest(&finalize::root_status(&tx, &roots).await.expect("status"));
    candidates::observe_candidate(
        &tx,
        &ObservedCandidate {
            candidate_id: &candidate_id,
            slot_id: SLOT,
            page_id: &identity::page_id(SLOT),
            space: SPACE,
            signal_kind: signal,
            coverage_epoch: epoch,
            input_generation: GROUPING_GENERATION,
            active_root_digest: &digest,
        },
    )
    .await
    .expect("observe");
    let reservations: Vec<ClaimReservation> = roots
        .iter()
        .map(|root_id| ClaimReservation {
            root_id: root_id.clone(),
            independence_group_id: format!("group-{}", &root_id[5..6]),
        })
        .collect();
    let outcome = candidates::prepare_candidate(&tx, &candidate_id, &reservations)
        .await
        .expect("prepare");
    assert!(
        matches!(outcome, PrepareOutcome::Prepared { .. }),
        "{outcome:?}"
    );
    finalize::record_envelope(
        &tx,
        &candidate_id,
        &CasEnvelope {
            receipt_id: finalize::receipt_id(&candidate_id),
            epoch_state: EpochState::Active,
            community_generation: Some(PUBLISHED_GENERATION),
            verdict: None,
        },
    )
    .await
    .expect("stamp envelope");
    tx.commit().await.unwrap();

    (db, epoch, candidate_id)
}

/// Run the dry run and return the gate that refused, or `None` on a pass.
async fn refusing_gate(db: &GenesisDb, candidate_id: &str) -> Option<CasGate> {
    let tx = db.tx().await;
    let outcome = dry_run_finalize(&tx, candidate_id).await.expect("dry run");
    tx.commit().await.unwrap();
    match outcome {
        Finalization::DryRunPassed { .. } => None,
        Finalization::Refused { gate, .. } => Some(gate),
        Finalization::NotEligible { candidate_id } => {
            panic!("candidate {candidate_id} was not eligible for finalization")
        }
    }
}

/// Attach one of the candidate's roots to a page through the R8 shape E-8
/// reads: a live `supports` edge from a claim revision that the page's
/// **current** version contains.
async fn attach_page(db: &GenesisDb, root_id: &str, page_id: &str, support_status: &str) {
    db.exec(
        "INSERT INTO pages (id, title, content, version, space, status)
         VALUES (?1, 'Anchored', 'body', 4, ?2, 'active')",
        libsql::params![page_id, SPACE],
    )
    .await;
    db.exec(
        "INSERT INTO page_truth_state (page_id, page_version, support_status)
         VALUES (?1, 4, ?2)",
        libsql::params![page_id, support_status],
    )
    .await;
    db.exec(
        "INSERT INTO page_version_claims (page_id, page_version, claim_revision_id, ordinal)
         VALUES (?1, 4, 'rev-1', 0)",
        libsql::params![page_id],
    )
    .await;
    db.exec(
        "INSERT INTO edges (edge_id, src_kind, src_id, dst_kind, dst_id, edge_type,
                            grounded, root_id, space, valid_until)
         VALUES ('supports-1', 'claim_revision', 'rev-1', 'entity', 'peer', 'supports',
                 1, ?1, ?2, NULL)",
        libsql::params![root_id, SPACE],
    )
    .await;
}

// ---------------------------------------------------------------------------
// The suite-level positive control
// ---------------------------------------------------------------------------

/// **The discriminating control for every RED row below.** An untouched scene
/// passes all eight gates, so each refusal that follows is attributable to the
/// one mutation that test scripted rather than to a gate that always refuses.
///
/// It also pins the two dryness properties a pass must have: the verdict is
/// recorded, and the candidate is still `prepared` — a passing dry run does
/// **not** advance to `published`, because publishing is precisely what PR-B
/// does not do.
#[tokio::test]
async fn the_unmutated_scene_passes_every_gate() {
    let (db, _epoch, candidate_id) = scene(SignalKind::SpaceOverview).await;

    assert_eq!(refusing_gate(&db, &candidate_id).await, None);
    assert_eq!(db.candidate_state(&candidate_id).await, "prepared");

    let tx = db.tx().await;
    let envelope = finalize::read_envelope(&tx, &candidate_id)
        .await
        .expect("envelope")
        .expect("envelope present");
    tx.commit().await.unwrap();
    assert_eq!(envelope.verdict, Some(Verdict::Passed));
}

// ---------------------------------------------------------------------------
// G4.1a–h — one row per CAS gate
// ---------------------------------------------------------------------------

/// G4.1a / E-1 — the lease was taken over under the worker.
#[tokio::test]
async fn a_lost_lease_refuses_at_e1() {
    let (db, _epoch, candidate_id) = scene(SignalKind::SpaceOverview).await;
    db.exec("DELETE FROM grouping_leases", ()).await;

    assert_eq!(
        refusing_gate(&db, &candidate_id).await,
        Some(CasGate::LeaseOwnership)
    );
}

/// G4.1a' / E-1 — an *expired but still present* row is not ownership. The
/// distinction matters because the row is physically there and visible to any
/// query that forgets the clock (machine C's `expired` state).
#[tokio::test]
async fn an_expired_lease_row_is_not_ownership() {
    let (db, _epoch, candidate_id) = scene(SignalKind::SpaceOverview).await;
    db.exec(
        "UPDATE grouping_leases SET expires_at = unixepoch() - 1 WHERE phase = 'genesis'",
        (),
    )
    .await;

    assert_eq!(
        refusing_gate(&db, &candidate_id).await,
        Some(CasGate::LeaseOwnership)
    );
}

/// G4.1b / E-2 — M4 re-grouped the space under the prepare.
#[tokio::test]
async fn a_moved_input_generation_refuses_at_e2() {
    let (db, _epoch, candidate_id) = scene(SignalKind::SpaceOverview).await;
    db.exec(
        "UPDATE space_graph_state SET grouping_generation = grouping_generation + 1",
        (),
    )
    .await;

    assert_eq!(
        refusing_gate(&db, &candidate_id).await,
        Some(CasGate::InputGeneration)
    );
}

/// G4.1c / E-3 — a claimed root was retracted, moving the active-root digest.
#[tokio::test]
async fn a_retracted_root_refuses_at_e3() {
    let (db, _epoch, candidate_id) = scene(SignalKind::SpaceOverview).await;
    db.exec(
        "UPDATE provenance_roots SET status = 'retracted' WHERE root_id = 'root-1-0'",
        (),
    )
    .await;

    assert_eq!(
        refusing_gate(&db, &candidate_id).await,
        Some(CasGate::RootDigest)
    );
}

/// G4.1c' / E-3 — a claimed root that has **disappeared** must move the digest
/// too. It reads as `missing` rather than dropping out of the hashed vector,
/// because a shorter vector of the surviving roots could collide with a
/// legitimately smaller claim set.
#[tokio::test]
async fn a_deleted_root_refuses_at_e3() {
    let (db, _epoch, candidate_id) = scene(SignalKind::SpaceOverview).await;
    db.exec(
        "DELETE FROM provenance_roots WHERE root_id = 'root-2-1'",
        (),
    )
    .await;

    assert_eq!(
        refusing_gate(&db, &candidate_id).await,
        Some(CasGate::RootDigest)
    );
}

/// G4.1d / E-4 — the identity contract was re-cut: a new coverage epoch.
#[tokio::test]
async fn a_moved_coverage_epoch_refuses_at_e4() {
    let (db, _epoch, candidate_id) = scene(SignalKind::SpaceOverview).await;
    db.exec(
        "UPDATE genesis_coverage_state SET coverage_epoch = coverage_epoch + 1",
        (),
    )
    .await;

    assert_eq!(
        refusing_gate(&db, &candidate_id).await,
        Some(CasGate::CoverageEpoch)
    );
}

/// G4.1e / E-5 — a claim was released under the candidate.
#[tokio::test]
async fn a_released_claim_refuses_at_e5() {
    let (db, _epoch, candidate_id) = scene(SignalKind::SpaceOverview).await;
    db.exec(
        "UPDATE genesis_candidate_roots SET released_at = unixepoch()
          WHERE root_id = 'root-0-0'",
        (),
    )
    .await;

    assert_eq!(
        refusing_gate(&db, &candidate_id).await,
        Some(CasGate::ClaimLiveness)
    );
}

/// G4.1e' / E-5 — "witness rows still present" has teeth of its own. A
/// witness-role candidate whose rows were rewritten to `concept` would have
/// silently acquired an exclusive claim it was never granted, and a row count
/// alone cannot see that.
#[tokio::test]
async fn a_witness_row_rewritten_to_concept_refuses_at_e5() {
    let (db, _epoch, candidate_id) = scene(SignalKind::SpaceOverview).await;
    db.exec(
        "UPDATE genesis_candidate_roots SET claim_role = 'concept' WHERE root_id = 'root-0-1'",
        (),
    )
    .await;

    assert_eq!(
        refusing_gate(&db, &candidate_id).await,
        Some(CasGate::ClaimLiveness)
    );
}

/// G4.1f / E-6 — an edge was retracted. The root's *status* is untouched, so
/// E-3's digest is unchanged and only E-6 can see this.
#[tokio::test]
async fn a_retracted_edge_refuses_at_e6() {
    let (db, _epoch, candidate_id) = scene(SignalKind::SpaceOverview).await;
    db.exec(
        "UPDATE edges SET valid_until = unixepoch() WHERE root_id = 'root-1-1'",
        (),
    )
    .await;

    assert_eq!(
        refusing_gate(&db, &candidate_id).await,
        Some(CasGate::EvidenceLiveness)
    );
}

/// G4.1f' / E-6 — an un-grounded edge is the same loss by another route (R2).
#[tokio::test]
async fn an_ungrounded_edge_refuses_at_e6() {
    let (db, _epoch, candidate_id) = scene(SignalKind::SpaceOverview).await;
    db.exec(
        "UPDATE edges SET grounded = 0 WHERE root_id = 'root-0-1'",
        (),
    )
    .await;

    assert_eq!(
        refusing_gate(&db, &candidate_id).await,
        Some(CasGate::EvidenceLiveness)
    );
}

/// G4.1g / E-7, **vacuity half** (S0-137). A space-overview candidate never
/// reads the community partition, so E-7 must not refuse it — and the fixture
/// installs none of M4's gate control-plane tables, which means the real gate
/// reports "not durable" here. If E-7 were unconditional, every test above
/// would be refusing at E-7 instead of where it says it does.
#[tokio::test]
async fn e7_is_vacuous_for_a_signal_that_never_reads_the_partition() {
    let (db, _epoch, candidate_id) = scene(SignalKind::SpaceOverview).await;

    assert_eq!(refusing_gate(&db, &candidate_id).await, None);
}

/// G4.1g / E-7, **non-vacuous half**. An evidence-cluster candidate *does*
/// depend on M4's partition, and the fixture's gate is unreadable — which the
/// real gate reports as not durable, fail-closed. This is the discriminating
/// control for the vacuity test above: the same scene, one signal kind apart,
/// with opposite outcomes.
#[tokio::test]
async fn evidence_cluster_reads_the_real_community_gate() {
    let (db, _epoch, candidate_id) = scene(SignalKind::EvidenceCluster).await;

    assert_eq!(
        refusing_gate(&db, &candidate_id).await,
        Some(CasGate::CommunityGeneration)
    );
}

/// G4.1h / E-8, **vacuity half**. A candidate whose roots no page anchors has
/// no M5 dependency, and E-8 says so rather than inventing one.
#[tokio::test]
async fn e8_is_vacuous_without_a_page_mediated_root() {
    let (db, _epoch, candidate_id) = scene(SignalKind::SpaceOverview).await;
    assert_eq!(db.scalar("SELECT COUNT(*) FROM pages", ()).await, 0);

    assert_eq!(refusing_gate(&db, &candidate_id).await, None);
}

/// G4.1h' / E-8, **non-vacuous half**: a page that anchors a claimed root loses
/// `supported`, and finalization must refuse.
#[tokio::test]
async fn a_page_losing_support_refuses_at_e8() {
    let (db, _epoch, candidate_id) = scene(SignalKind::SpaceOverview).await;
    attach_page(&db, "root-0-0", "page-anchor", "unsupported").await;

    assert_eq!(
        refusing_gate(&db, &candidate_id).await,
        Some(CasGate::PageTruth)
    );
}

/// G4.1h'' / E-8 — the **version** half. A page whose truth state describes an
/// older version has stale anchors; trusting them is the exact bug M5's own
/// reader closes with `page_truth_state.page_version = pages.version`.
#[tokio::test]
async fn a_page_whose_truth_state_lags_its_version_refuses_at_e8() {
    let (db, _epoch, candidate_id) = scene(SignalKind::SpaceOverview).await;
    attach_page(&db, "root-0-0", "page-anchor", "supported").await;
    db.exec("UPDATE pages SET version = 5 WHERE id = 'page-anchor'", ())
        .await;

    assert_eq!(
        refusing_gate(&db, &candidate_id).await,
        Some(CasGate::PageTruth)
    );
}

/// G4.1h''' / E-8 — the discriminating control for the two rows above: the same
/// page, still `supported` at its current version, passes.
#[tokio::test]
async fn a_supported_current_page_passes_e8() {
    let (db, _epoch, candidate_id) = scene(SignalKind::SpaceOverview).await;
    attach_page(&db, "root-0-0", "page-anchor", "supported").await;

    assert_eq!(refusing_gate(&db, &candidate_id).await, None);
}

// ---------------------------------------------------------------------------
// G4.1i–k — the three ordering permutations
// ---------------------------------------------------------------------------

/// G4.1i — E-1 before E-6. Both conditions are violated at once; the reported
/// gate must be the lease, because a run that checked evidence liveness first
/// would be acting on state it does not own.
#[tokio::test]
async fn a_lost_lease_outranks_a_retracted_edge() {
    let (db, _epoch, candidate_id) = scene(SignalKind::SpaceOverview).await;
    db.exec("DELETE FROM grouping_leases", ()).await;
    db.exec(
        "UPDATE edges SET valid_until = unixepoch() WHERE root_id = 'root-1-1'",
        (),
    )
    .await;

    assert_eq!(
        refusing_gate(&db, &candidate_id).await,
        Some(CasGate::LeaseOwnership)
    );
}

/// G4.1j — E-2 before E-8.
#[tokio::test]
async fn a_moved_generation_outranks_a_lost_page_support() {
    let (db, _epoch, candidate_id) = scene(SignalKind::SpaceOverview).await;
    attach_page(&db, "root-0-0", "page-anchor", "unsupported").await;
    db.exec(
        "UPDATE space_graph_state SET grouping_generation = grouping_generation + 1",
        (),
    )
    .await;

    assert_eq!(
        refusing_gate(&db, &candidate_id).await,
        Some(CasGate::InputGeneration)
    );
}

/// G4.1k — E-3 before E-5.
#[tokio::test]
async fn a_moved_digest_outranks_a_released_claim() {
    let (db, _epoch, candidate_id) = scene(SignalKind::SpaceOverview).await;
    db.exec(
        "UPDATE provenance_roots SET status = 'retracted' WHERE root_id = 'root-1-0'",
        (),
    )
    .await;
    db.exec(
        "UPDATE genesis_candidate_roots SET released_at = unixepoch()
          WHERE root_id = 'root-0-0'",
        (),
    )
    .await;

    assert_eq!(
        refusing_gate(&db, &candidate_id).await,
        Some(CasGate::RootDigest)
    );
}

/// The order the implementation walks is the order the artifact fixes. A
/// permutation test can only ever sample pairs; this pins the whole sequence.
#[tokio::test]
async fn the_gate_list_is_in_artifact_order() {
    assert_eq!(
        CasGate::ALL.map(CasGate::as_str),
        [
            "e1_lease_ownership",
            "e2_input_generation",
            "e3_root_digest",
            "e4_coverage_epoch",
            "e5_claim_liveness",
            "e6_evidence_liveness",
            "e7_community_generation",
            "e8_page_truth",
        ]
    );
    for gate in CasGate::ALL {
        assert_eq!(CasGate::parse(gate.as_str()), Some(gate));
    }
}

// ---------------------------------------------------------------------------
// G4.2a–g — a stored CAS input is not replaceable by a recomputation
// ---------------------------------------------------------------------------

/// G4.2a — **the `active → draining` analogue**, and the row this whole family
/// exists for.
///
/// The epoch regime moves `active → migrating` while *every other observable is
/// byte-identical*: same epoch number, same roots, same statuses, same digest,
/// same claims, same generation. A finalize that recomputed its verdict instead
/// of comparing the stored regime would find nothing wrong and publish under a
/// regime it never leased. E-4's second half is what refuses.
#[tokio::test]
async fn a_migrating_epoch_refuses_even_though_nothing_recomputable_moved() {
    let (db, _epoch, candidate_id) = scene(SignalKind::SpaceOverview).await;

    // Prove the "nothing recomputable moved" half rather than asserting it: the
    // digest over the live world still equals the stored one.
    let tx = db.tx().await;
    let live = identity::active_root_digest(
        &finalize::root_status(&tx, &root_ids(SignalKind::SpaceOverview))
            .await
            .expect("status"),
    );
    let stored = candidates::read_candidate(&tx, &candidate_id)
        .await
        .expect("read")
        .expect("present")
        .active_root_digest;
    tx.commit().await.unwrap();
    assert_eq!(live, stored, "the recomputable half must be unchanged");

    db.exec(
        "UPDATE genesis_coverage_state SET epoch_state = 'migrating'",
        (),
    )
    .await;

    assert_eq!(
        refusing_gate(&db, &candidate_id).await,
        Some(CasGate::CoverageEpoch)
    );
}

/// G4.2b — the community generation is a stored input, not a recomputation.
/// M4 republishes; the partition is still perfectly *derivable*, and still
/// wrong for a worker that leased under the previous publication.
#[tokio::test]
async fn a_republished_community_generation_moves_the_stored_input() {
    let (db, _epoch, candidate_id) = scene(SignalKind::EvidenceCluster).await;
    let tx = db.tx().await;
    let envelope = finalize::read_envelope(&tx, &candidate_id)
        .await
        .expect("envelope")
        .expect("present");
    tx.commit().await.unwrap();
    assert_eq!(envelope.community_generation, Some(PUBLISHED_GENERATION));

    db.exec(
        "UPDATE space_graph_state SET published_generation = published_generation + 1",
        (),
    )
    .await;
    let tx = db.tx().await;
    let live = finalize::published_generation(&tx, SPACE)
        .await
        .expect("published generation");
    tx.commit().await.unwrap();
    assert_ne!(
        live, envelope.community_generation,
        "the live value must have moved away from the stored one"
    );

    assert_eq!(
        refusing_gate(&db, &candidate_id).await,
        Some(CasGate::CommunityGeneration)
    );
}

/// G4.2c — the stored envelope survives a restart, because it is a row. An
/// in-memory carry between the prepare turn and the finalize turn would be
/// indistinguishable from this on a happy path and absent after a `kill -9`.
#[tokio::test]
async fn the_stored_envelope_is_durable_and_reused_on_retry() {
    let (db, _epoch, candidate_id) = scene(SignalKind::SpaceOverview).await;
    let expected = finalize::receipt_id(&candidate_id);

    let tx = db.tx().await;
    let first = finalize::read_envelope(&tx, &candidate_id)
        .await
        .expect("envelope")
        .expect("present");
    tx.commit().await.unwrap();
    assert_eq!(first.receipt_id, expected);
    assert_eq!(first.epoch_state, EpochState::Active);
    assert_eq!(first.verdict, None);

    // I-8: the receipt is a function of the candidate, so a re-stamp under a
    // new lease cannot mint a second identity.
    let tx = db.tx().await;
    finalize::record_envelope(
        &tx,
        &candidate_id,
        &CasEnvelope {
            verdict: None,
            ..first.clone()
        },
    )
    .await
    .expect("re-stamp");
    let second = finalize::read_envelope(&tx, &candidate_id)
        .await
        .expect("envelope")
        .expect("present");
    tx.commit().await.unwrap();
    assert_eq!(second.receipt_id, first.receipt_id);
    assert_eq!(
        db.scalar(
            "SELECT COUNT(DISTINCT receipt_id) FROM genesis_candidates",
            ()
        )
        .await,
        1
    );
}

/// G4.2d — catalog row G3.3, now executable. A retry reuses the same receipt
/// rather than writing a second one.
#[tokio::test]
async fn a_retry_reuses_the_same_receipt() {
    let (db, _epoch, candidate_id) = scene(SignalKind::SpaceOverview).await;
    assert_eq!(
        db.scalar(
            "SELECT COUNT(*) FROM genesis_candidates
              WHERE candidate_id = ?1 AND receipt_id = ?2",
            libsql::params![candidate_id.as_str(), finalize::receipt_id(&candidate_id)],
        )
        .await,
        1
    );

    // Force a refusal, which requeues via A11. The receipt a second attempt
    // would stamp is the same string, so the distinct count cannot grow.
    db.exec(
        "UPDATE space_graph_state SET grouping_generation = grouping_generation + 1",
        (),
    )
    .await;
    assert_eq!(
        refusing_gate(&db, &candidate_id).await,
        Some(CasGate::InputGeneration)
    );
    assert_eq!(db.candidate_state(&candidate_id).await, "retry_wait");
    assert_eq!(
        db.scalar(
            "SELECT COUNT(DISTINCT receipt_id) FROM genesis_candidates",
            ()
        )
        .await,
        1
    );
}

/// G4.2e — a candidate that has already been verified for this prepare is not
/// re-verified. Without it the shadow would re-run eight gates every tick
/// forever on a quiescent store, which is a load profile, not a proof.
#[tokio::test]
async fn a_recorded_verdict_makes_the_candidate_ineligible() {
    let (db, _epoch, candidate_id) = scene(SignalKind::SpaceOverview).await;
    assert_eq!(refusing_gate(&db, &candidate_id).await, None);

    let tx = db.tx().await;
    let again = dry_run_finalize(&tx, &candidate_id).await.expect("dry run");
    tx.commit().await.unwrap();
    assert!(
        matches!(again, Finalization::NotEligible { .. }),
        "{again:?}"
    );
}

/// G4.2f — a candidate with no stamped envelope is not finalizable at all.
/// The envelope *is* the stored CAS input; finalizing without one would be
/// finalizing against recomputations only, which is what this family forbids.
#[tokio::test]
async fn a_candidate_without_an_envelope_is_not_finalizable() {
    let (db, _epoch, candidate_id) = scene(SignalKind::SpaceOverview).await;
    db.exec(
        "UPDATE genesis_candidates SET payload = NULL WHERE candidate_id = ?1",
        libsql::params![candidate_id.as_str()],
    )
    .await;

    let tx = db.tx().await;
    let outcome = dry_run_finalize(&tx, &candidate_id).await.expect("dry run");
    tx.commit().await.unwrap();
    assert!(
        matches!(outcome, Finalization::NotEligible { .. }),
        "{outcome:?}"
    );
}

/// G4.2g — only a `prepared` candidate is finalizable. A waiting or terminal
/// candidate reaching machine E would be finalizing work no lease covers.
#[tokio::test]
async fn a_candidate_outside_prepared_is_not_finalizable() {
    let (db, _epoch, candidate_id) = scene(SignalKind::SpaceOverview).await;
    let tx = db.tx().await;
    candidates::mark_retry_wait(&tx, &candidate_id, candidates::RetryCharge::Charged)
        .await
        .expect("retry_wait");
    tx.commit().await.unwrap();
    assert_eq!(
        CandidateState::parse(&db.candidate_state(&candidate_id).await),
        Some(CandidateState::RetryWait)
    );

    let tx = db.tx().await;
    let outcome = dry_run_finalize(&tx, &candidate_id).await.expect("dry run");
    tx.commit().await.unwrap();
    assert!(
        matches!(outcome, Finalization::NotEligible { .. }),
        "{outcome:?}"
    );
}

// ---------------------------------------------------------------------------
// G4.3 — E3 rollback: a failed gate leaves no partial write
// ---------------------------------------------------------------------------

/// G4.3 — a refusal publishes nothing and damages nothing.
///
/// The assertion is deliberately over the tables a *publish* would have
/// touched, not over "some count is zero": a refused finalization must leave
/// the claim rows exactly as it found them (B3 retains reservations across
/// A11), and must have written no page, no projection hand-off, no provenance
/// mint, and no coverage row. The requeue is the only state change, and it is
/// the response to the refusal rather than a partial publish.
#[tokio::test]
async fn a_refused_finalization_leaves_no_partial_write() {
    let (db, _epoch, candidate_id) = scene(SignalKind::SpaceOverview).await;
    let claims_before = db
        .scalar(
            "SELECT COUNT(*) FROM genesis_candidate_roots WHERE released_at IS NULL",
            (),
        )
        .await;
    assert!(claims_before > 0, "the scene must hold live claims");
    db.exec(
        "UPDATE provenance_roots SET status = 'retracted' WHERE root_id = 'root-1-0'",
        (),
    )
    .await;

    assert_eq!(
        refusing_gate(&db, &candidate_id).await,
        Some(CasGate::RootDigest)
    );

    for sql in [
        "SELECT COUNT(*) FROM pages",
        "SELECT COUNT(*) FROM page_projection_outbox",
        "SELECT COUNT(*) FROM m6_genesis_provenance",
        "SELECT COUNT(*) FROM genesis_group_coverage",
    ] {
        assert_eq!(db.scalar(sql, ()).await, 0, "{sql} must be untouched");
    }
    assert_eq!(
        db.scalar(
            "SELECT COUNT(*) FROM genesis_candidate_roots WHERE released_at IS NULL",
            (),
        )
        .await,
        claims_before,
        "B3 retains reservations across the A11 requeue"
    );
    assert_eq!(db.candidate_state(&candidate_id).await, "retry_wait");
    // The lease is released in both outcomes — a dry run that held it to the
    // 900s TTL would block its space for fifteen minutes per candidate.
    assert_eq!(
        db.scalar(
            "SELECT COUNT(*) FROM grouping_leases WHERE phase = 'genesis'",
            (),
        )
        .await,
        0
    );
}

/// G4.3' — and the same is true of a *passing* dry run. This is the row that
/// says "dry" out loud: all eight gates held, the real path would publish here,
/// and every publication table is still empty.
#[tokio::test]
async fn a_passing_dry_run_publishes_nothing() {
    let (db, _epoch, candidate_id) = scene(SignalKind::SpaceOverview).await;

    assert_eq!(refusing_gate(&db, &candidate_id).await, None);

    for sql in [
        "SELECT COUNT(*) FROM pages",
        "SELECT COUNT(*) FROM page_projection_outbox",
        "SELECT COUNT(*) FROM m6_genesis_provenance",
        "SELECT COUNT(*) FROM genesis_group_coverage",
    ] {
        assert_eq!(db.scalar(sql, ()).await, 0, "{sql}");
    }
    assert_ne!(db.candidate_state(&candidate_id).await, "published");
    assert_eq!(
        db.scalar(
            "SELECT COALESCE(MAX(m6_mutation_count), -1) FROM genesis_coverage_state",
            (),
        )
        .await,
        0,
        "the monotone mutation counter never moved"
    );
    assert_eq!(
        db.scalar(
            "SELECT MAX(genesis_enabled) FROM genesis_coverage_state",
            ()
        )
        .await,
        0,
        "the shadow never writes genesis_enabled"
    );
}

// ---------------------------------------------------------------------------
// G4.5a–d — no LLM call anywhere, checked by signature
// ---------------------------------------------------------------------------

/// G4.5a–d — **`finalize.rs` names no provider type.**
///
/// A source-level assertion, not a runtime one (§10.2). A runtime check can
/// only observe that no call happened on the paths the test drove; reading the
/// source proves no such call can be written without this failing. The names
/// are the holders §8's row enumerates — the provider trait, the engine, the
/// module that owns them, and the readiness hook that hands one out.
#[tokio::test]
async fn the_module_names_no_llm_provider() {
    for (file, forbidden) in [
        (
            "src/m6/finalize.rs",
            &[
                "LlmProvider",
                "LlmEngine",
                "llm_provider",
                "LLM_READINESS_HOOK",
            ][..],
        ),
        (
            "src/m6/shadow.rs",
            &["LlmProvider", "LlmEngine", "llm_provider"][..],
        ),
    ] {
        let source =
            std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(file))
                .unwrap_or_else(|error| panic!("read {file}: {error}"));
        // The module docs explain *why* the rule exists and necessarily say the
        // words; strip comment lines so the assertion is about code.
        let code: String = source
            .lines()
            .filter(|line| !line.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        for name in forbidden {
            assert!(!code.contains(name), "{file} must not name {name}");
        }
    }
}
