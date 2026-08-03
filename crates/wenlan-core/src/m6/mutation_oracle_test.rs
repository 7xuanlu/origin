// SPDX-License-Identifier: Apache-2.0
//! The mutation oracles (spec §6.3) and the full-vs-incremental agreement the
//! §6.1 oracle exists for.
//!
//! §6.3's table is a contract about *deltas*: each mutation class names what the
//! genesis state must do in response. The tests below script one mutation each
//! and assert the delta against a from-primary-evidence recomputation, because
//! that side is derived from the evidence the mutation actually changed — an
//! assertion against the durable side would only be checking that the fixture
//! wrote what the fixture wrote.

use super::candidates::{self, ClaimReservation, ObservedCandidate, PrepareOutcome};
use super::frontier;
use super::genesis_test_support::GenesisDb;
use super::identity;
use super::oracle::{self, GenesisSnapshot};
use super::signals::CandidateProposal;

const SPACE: &str = "space-a";
const SPACE_ID: &str = "space-id-a";

/// Communities are inert in every test here except the durable-gate one, so the
/// flag is `false` by default and the one test that cares says so.
const NO_DURABLE_PARTITION: bool = false;

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

/// Three groups, two distinct entities each: clears D1's 3-group floor and
/// signal 4's 5-grounded-node floor, so the base scene carries exactly one
/// space-overview proposal and three eligible groups.
async fn base_scene() -> (GenesisDb, i64) {
    let db = GenesisDb::new().await;
    db.seed_space(SPACE_ID, SPACE).await;
    let epoch = db.open_epoch(SPACE_ID, SPACE).await;
    for (index, group) in ["group-a", "group-b", "group-c"].into_iter().enumerate() {
        for slot in 0..2 {
            db.seed_evidence(
                &format!("root-{index}-{slot}"),
                SPACE,
                group,
                &format!("entity-{index}-{slot}"),
            )
            .await;
        }
    }
    (db, epoch)
}

async fn full(db: &GenesisDb, epoch: i64, durable_partition: bool) -> GenesisSnapshot {
    let tx = db.tx().await;
    let snapshot = oracle::recompute_full(&tx, SPACE_ID, epoch, durable_partition)
        .await
        .expect("recompute_full");
    tx.commit().await.unwrap();
    snapshot
}

async fn incremental(db: &GenesisDb, epoch: i64) -> GenesisSnapshot {
    let tx = db.tx().await;
    let snapshot = oracle::snapshot_incremental(&tx, SPACE_ID, epoch)
        .await
        .expect("snapshot_incremental");
    tx.commit().await.unwrap();
    snapshot
}

/// `(root_id, independence_group_id)` for a proposal's roots, read from primary
/// evidence — the claim set a driver would reserve.
async fn claims_for(db: &GenesisDb, proposal: &CandidateProposal) -> Vec<ClaimReservation> {
    let mut reservations = Vec::new();
    for root_id in &proposal.root_ids {
        let mut rows = db
            .connection
            .query(
                "SELECT independence_group_id FROM provenance_roots WHERE root_id = ?1",
                libsql::params![root_id.as_str()],
            )
            .await
            .expect("query root group");
        let group: String = rows
            .next()
            .await
            .expect("read root group row")
            .expect("root present")
            .get(0)
            .expect("decode root group");
        reservations.push(ClaimReservation {
            root_id: root_id.clone(),
            independence_group_id: group,
        });
    }
    reservations
}

/// Stand in for PR-B3's shadow loop: observe every proposal, reserve its
/// claims, then reconcile the leftovers into the frontier. This is the
/// **quiescent** state `recompute_full` describes, so `diff` must be empty
/// after it — and that is the only claim it makes.
async fn drive_to_quiescence(db: &GenesisDb, epoch: i64, durable_partition: bool) {
    let snapshot = full(db, epoch, durable_partition).await;
    for proposal in &snapshot.proposals {
        let candidate_id = identity::candidate_id(&proposal.slot_id, epoch);
        let digest = identity::active_root_digest(
            &proposal
                .root_ids
                .iter()
                .map(|root_id| (root_id.clone(), "active".to_string()))
                .collect::<Vec<_>>(),
        );
        let reservations = claims_for(db, proposal).await;
        let tx = db.tx().await;
        candidates::observe_candidate(
            &tx,
            &ObservedCandidate {
                candidate_id: &candidate_id,
                slot_id: &proposal.slot_id,
                page_id: &proposal.page_id,
                space: SPACE,
                signal_kind: proposal.signal_kind,
                coverage_epoch: epoch,
                input_generation: 1,
                active_root_digest: &digest,
            },
        )
        .await
        .expect("observe");
        let outcome = candidates::prepare_candidate(&tx, &candidate_id, &reservations)
            .await
            .expect("prepare");
        assert!(
            matches!(outcome, PrepareOutcome::Prepared { .. }),
            "{outcome:?}"
        );
        tx.commit().await.unwrap();
        db.release_leases().await;
    }
    let tx = db.tx().await;
    frontier::reconcile_space(&tx, SPACE, epoch).await.unwrap();
    tx.commit().await.unwrap();
}

// ---------------------------------------------------------------------------
// §6.1 — full vs incremental
// ---------------------------------------------------------------------------

/// The vacuous agreement (S0-137). It is worth asserting because it is the
/// state every install starts in, and because an oracle that only agrees when
/// both sides are empty is the failure mode the next test rules out.
#[tokio::test]
async fn an_empty_space_agrees() {
    let db = GenesisDb::new().await;
    db.seed_space(SPACE_ID, SPACE).await;
    let epoch = db.open_epoch(SPACE_ID, SPACE).await;

    let full = full(&db, epoch, NO_DURABLE_PARTITION).await;
    let incremental = incremental(&db, epoch).await;

    assert_eq!(full, GenesisSnapshot::default());
    assert!(oracle::diff(&full, &incremental).is_empty());
}

/// The load-bearing agreement: a driven space with real proposals, claims and
/// frontier rows diffs clean against a recomputation that never read a
/// `genesis_*` row.
#[tokio::test]
async fn a_driven_space_agrees_with_its_recomputation() {
    let (db, epoch) = base_scene().await;
    drive_to_quiescence(&db, epoch, NO_DURABLE_PARTITION).await;

    let full = full(&db, epoch, NO_DURABLE_PARTITION).await;
    let incremental = incremental(&db, epoch).await;

    assert_eq!(full.proposals.len(), 1, "the space-overview proposal");
    assert_eq!(full.claims.len(), 6, "one witness claim per root");
    assert_eq!(full.frontier.len(), 3, "witness claims are not coverage");
    assert_eq!(
        oracle::diff(&full, &incremental),
        Vec::new(),
        "full and incremental disagreed on a quiescent space"
    );
}

/// The oracle must **find** a divergence, not just fail to find one. Dropping a
/// durable candidate is the cheapest injected drift, and it must surface in both
/// the candidates and proposals sections rather than being absorbed.
#[tokio::test]
async fn a_dropped_durable_candidate_is_reported() {
    let (db, epoch) = base_scene().await;
    drive_to_quiescence(&db, epoch, NO_DURABLE_PARTITION).await;
    db.exec("DELETE FROM genesis_candidate_roots", ()).await;
    db.exec("DELETE FROM genesis_candidates", ()).await;

    let divergences = oracle::diff(
        &full(&db, epoch, NO_DURABLE_PARTITION).await,
        &incremental(&db, epoch).await,
    );

    let sections: Vec<&str> = divergences.iter().map(|d| d.section).collect();
    assert!(sections.contains(&"candidates"), "{divergences:?}");
    assert!(sections.contains(&"proposals"), "{divergences:?}");
}

/// The `active_root_digest` field census earns its keep here: evidence moving
/// under a candidate changes nothing about its identity, so without this field
/// compared the two snapshots would agree while the durable row described roots
/// that no longer exist. That is A3's `input_moved` condition.
#[tokio::test]
async fn evidence_moving_under_a_candidate_is_reported() {
    let (db, epoch) = base_scene().await;
    drive_to_quiescence(&db, epoch, NO_DURABLE_PARTITION).await;
    db.exec(
        "UPDATE provenance_roots SET status = 'retracted' WHERE root_id = 'root-0-0'",
        (),
    )
    .await;

    let divergences = oracle::diff(
        &full(&db, epoch, NO_DURABLE_PARTITION).await,
        &incremental(&db, epoch).await,
    );

    assert!(
        divergences
            .iter()
            .any(|d| d.section == "candidates" && d.detail.contains("active_root_digest")),
        "{divergences:?}"
    );
}

/// §6.1's disclosed deviation, asserted rather than described: a candidate
/// reaching a terminal state leaves the incremental snapshot, so `state` shows
/// up as a membership divergence.
#[tokio::test]
async fn a_terminal_candidate_leaves_the_incremental_snapshot() {
    let (db, epoch) = base_scene().await;
    drive_to_quiescence(&db, epoch, NO_DURABLE_PARTITION).await;
    let candidate_id: String = {
        let full = full(&db, epoch, NO_DURABLE_PARTITION).await;
        full.candidates[0].candidate_id.clone()
    };
    // `stale` is re-enterable and therefore *not* terminal; only A4/A16's
    // `superseded` is, which is reachable from `retry_wait`.
    let tx = db.tx().await;
    candidates::mark_retry_wait(&tx, &candidate_id, candidates::RetryCharge::Charged)
        .await
        .unwrap();
    candidates::mark_superseded(&tx, &candidate_id)
        .await
        .unwrap();
    tx.commit().await.unwrap();

    let incremental = incremental(&db, epoch).await;
    assert!(incremental.candidates.is_empty());
    let divergences = oracle::diff(&full(&db, epoch, NO_DURABLE_PARTITION).await, &incremental);
    assert!(
        divergences.iter().any(|d| d.section == "candidates"),
        "{divergences:?}"
    );
}

// ---------------------------------------------------------------------------
// §6.3 — the mutation table, one test per row
// ---------------------------------------------------------------------------

/// Row 1 — a grounded edge on a new root in a **new** group adds exactly one
/// group to the state.
#[tokio::test]
async fn a_new_group_adds_exactly_one() {
    let (db, epoch) = base_scene().await;
    let before = full(&db, epoch, NO_DURABLE_PARTITION).await;
    db.seed_evidence("root-3-0", SPACE, "group-d", "entity-3-0")
        .await;
    let after = full(&db, epoch, NO_DURABLE_PARTITION).await;

    assert_eq!(before.frontier.len(), 3);
    assert_eq!(after.frontier.len(), 4);
    assert!(after.frontier.contains(&"group-d".to_string()));
    assert_eq!(
        after.proposals[0].group_count, 4,
        "and the signal saw the new group too"
    );
}

/// Row 2 — R1: a second root in an **existing** group changes nothing about the
/// group count. R1 is the whole reason independence is counted over groups
/// rather than roots; a fixture that counted roots would read 4 here.
#[tokio::test]
async fn a_second_root_in_an_existing_group_changes_nothing() {
    let (db, epoch) = base_scene().await;
    let before = full(&db, epoch, NO_DURABLE_PARTITION).await;
    db.seed_evidence("root-0-2", SPACE, "group-a", "entity-0-2")
        .await;
    let after = full(&db, epoch, NO_DURABLE_PARTITION).await;

    assert_eq!(after.frontier, before.frontier);
    assert_eq!(
        after.proposals[0].group_count,
        before.proposals[0].group_count
    );
    assert_eq!(
        after.proposals[0].root_ids.len(),
        before.proposals[0].root_ids.len() + 1,
        "the root set still grew — only the count is group-keyed"
    );
}

/// Row 3 — R2: a retraction drops the group **iff** it was that group's last
/// active root. Both arms, because only asserting the second would pass against
/// an implementation that dropped the group on the first retraction.
#[tokio::test]
async fn a_retracted_root_drops_the_group_only_when_it_was_the_last() {
    let (db, epoch) = base_scene().await;
    db.exec(
        "UPDATE provenance_roots SET status = 'retracted' WHERE root_id = 'root-0-0'",
        (),
    )
    .await;
    let partial = full(&db, epoch, NO_DURABLE_PARTITION).await;
    assert_eq!(partial.frontier.len(), 3, "group-a still has root-0-1");

    db.exec(
        "UPDATE provenance_roots SET status = 'retracted' WHERE root_id = 'root-0-1'",
        (),
    )
    .await;
    let dropped = full(&db, epoch, NO_DURABLE_PARTITION).await;
    assert_eq!(dropped.frontier.len(), 2);
    assert!(!dropped.frontier.contains(&"group-a".to_string()));
    assert!(
        dropped.proposals.is_empty(),
        "and the space fell below the D1 floor, so signal 4 stops admitting"
    );
}

/// Row 4 — R3: an edge retracted by `valid_until` behaves exactly like a
/// retracted root. Two ways to withdraw evidence, one observable outcome.
#[tokio::test]
async fn a_retracted_edge_drops_the_group_the_same_way() {
    let (db, epoch) = base_scene().await;
    db.exec(
        "UPDATE edges SET valid_until = unixepoch()
          WHERE edge_id IN ('root-1-0', 'root-1-1')",
        (),
    )
    .await;
    let after = full(&db, epoch, NO_DURABLE_PARTITION).await;

    assert_eq!(after.frontier.len(), 2);
    assert!(!after.frontier.contains(&"group-b".to_string()));
}

/// Row 5 — R4: a generated root is not independent evidence, so a whole new
/// group carried only by one adds nothing. This is the rule that stops M6 from
/// counting its own output as support for its next candidate.
#[tokio::test]
async fn a_generated_root_adds_nothing() {
    let (db, epoch) = base_scene().await;
    let before = full(&db, epoch, NO_DURABLE_PARTITION).await;
    db.exec(
        "INSERT INTO provenance_roots (root_id, root_kind, independence_group_id, status)
         VALUES ('root-gen', 'generated', 'group-gen', 'active')",
        (),
    )
    .await;
    db.exec(
        "INSERT INTO edges (edge_id, src_kind, src_id, dst_kind, dst_id,
                            grounded, root_id, space, valid_until)
         VALUES ('root-gen', 'entity', 'entity-gen', 'entity', 'peer', 1,
                 'root-gen', ?1, NULL)",
        libsql::params![SPACE],
    )
    .await;
    let after = full(&db, epoch, NO_DURABLE_PARTITION).await;

    assert_eq!(after.frontier, before.frontier);
    assert_eq!(after.proposals, before.proposals);
}

/// Row 6 — R5/S0-164: renaming a page to `Overview` withdraws its edges from the
/// independence count, and renaming it back restores them. The match is on
/// `lower(title)`, so the case of the rename must not matter.
#[tokio::test]
async fn renaming_a_page_to_overview_flips_signal_eligibility() {
    let db = GenesisDb::new().await;
    db.seed_space(SPACE_ID, SPACE).await;
    let epoch = db.open_epoch(SPACE_ID, SPACE).await;
    // Two ordinary groups plus one carried by a page-endpoint edge: exactly at
    // the D1 floor, so withdrawing the page's group is observable as admission
    // flipping rather than as a count that merely shrinks.
    for (index, group) in ["group-a", "group-b"].into_iter().enumerate() {
        for slot in 0..2 {
            db.seed_evidence(
                &format!("root-{index}-{slot}"),
                SPACE,
                group,
                &format!("entity-{index}-{slot}"),
            )
            .await;
        }
    }
    db.exec(
        "INSERT INTO pages (id, title, content, version, space, status)
         VALUES ('page-x', 'Rust', '', 1, ?1, 'active')",
        libsql::params![SPACE],
    )
    .await;
    db.exec(
        "INSERT INTO provenance_roots (root_id, root_kind, independence_group_id, status)
         VALUES ('root-page', 'document_ingest', 'group-page', 'active')",
        (),
    )
    .await;
    db.exec(
        "INSERT INTO edges (edge_id, src_kind, src_id, dst_kind, dst_id,
                            grounded, root_id, space, valid_until)
         VALUES ('root-page', 'page', 'page-x', 'entity', 'entity-page', 1,
                 'root-page', ?1, NULL)",
        libsql::params![SPACE],
    )
    .await;
    let admitting = full(&db, epoch, NO_DURABLE_PARTITION).await;
    assert_eq!(admitting.proposals.len(), 1);
    assert_eq!(admitting.proposals[0].group_count, 3);

    db.exec(
        "UPDATE pages SET title = 'OVERVIEW' WHERE id = 'page-x'",
        (),
    )
    .await;
    let refused = full(&db, epoch, NO_DURABLE_PARTITION).await;
    assert!(
        refused.proposals.is_empty(),
        "an Overview page's evidence must not count toward D1, whatever its case"
    );
    assert!(
        refused.frontier.contains(&"group-page".to_string()),
        "but the group stays eligible and therefore stays accounted for"
    );

    db.exec("UPDATE pages SET title = 'Rust' WHERE id = 'page-x'", ())
        .await;
    assert_eq!(
        full(&db, epoch, NO_DURABLE_PARTITION).await.proposals,
        admitting.proposals,
        "the flip is symmetric — renaming back restores the identical proposal"
    );
}

/// Row 7 — the community durable gate. When the M4 partition is not durable,
/// signals 1 and 3 admit nothing at all, no matter how much evidence a
/// community carries. The gate is caller-supplied (it stays `db.rs`'s to
/// register), which is exactly why a test must pin both arms.
#[tokio::test]
async fn a_non_durable_partition_silences_signals_one_and_three() {
    let (db, epoch) = base_scene().await;
    db.exec(
        "INSERT INTO communities (community_id, space, display_name, algo_version,
                                  projection_version, created_at, updated_at, retired_at)
         VALUES ('community-1', ?1, NULL, 'leiden-v1', 'proj-v1', 0, 0, NULL)",
        libsql::params![SPACE],
    )
    .await;
    for index in 0..3 {
        for slot in 0..2 {
            db.exec(
                "INSERT INTO community_members (space, node_id, node_kind, community_id,
                                                published_generation, attachment)
                 VALUES (?1, ?2, 'entity', 'community-1', 1, 'core')",
                libsql::params![SPACE, format!("entity-{index}-{slot}")],
            )
            .await;
        }
    }

    let gated = full(&db, epoch, false).await;
    let durable = full(&db, epoch, true).await;

    assert_eq!(gated.proposals.len(), 1, "space-overview only");
    assert_eq!(
        durable.proposals.len(),
        3,
        "evidence-cluster and community-overview join once the partition is durable"
    );
    assert_eq!(
        gated.frontier.len(),
        3,
        "gated, the three groups have no concept claim and wait in the frontier"
    );
    assert!(
        durable.frontier.is_empty(),
        "and the evidence-cluster candidate the gate was suppressing is a \
         concept claimant, so opening the gate is what moves them out"
    );
}

/// Row 8 — R8: an orphan-wikilink candidate's groups are mediated by M5 support.
/// A referring page dropping out of `supported` shrinks the underlying set, and
/// here it takes the label below the two-referring-page precondition.
#[tokio::test]
async fn losing_m5_support_shrinks_the_orphan_signals_evidence() {
    let db = GenesisDb::new().await;
    db.seed_space(SPACE_ID, SPACE).await;
    let epoch = db.open_epoch(SPACE_ID, SPACE).await;
    seed_orphan_baseline(&db, "Rust").await;

    let admitting = full(&db, epoch, NO_DURABLE_PARTITION).await;
    let orphan_before = admitting
        .proposals
        .iter()
        .filter(|proposal| proposal.signal_kind == super::identity::SignalKind::OrphanWikilink)
        .count();
    assert_eq!(orphan_before, 1, "the baseline admits one orphan label");

    db.exec(
        "UPDATE page_truth_state SET support_status = 'unsupported' WHERE page_id = 'page-b'",
        (),
    )
    .await;
    let after = full(&db, epoch, NO_DURABLE_PARTITION).await;

    assert!(
        after
            .proposals
            .iter()
            .all(|proposal| proposal.signal_kind != super::identity::SignalKind::OrphanWikilink),
        "one referring page is below the >= 2 precondition"
    );
    assert!(
        admitting.frontier.is_empty(),
        "while the orphan candidate stood, its concept claims accounted for all three groups"
    );
    assert_eq!(
        after.frontier,
        vec![
            "group-a".to_string(),
            "group-b".to_string(),
            "group-c".to_string()
        ],
        "losing support withdraws a signal's evidence, never a group's eligibility — \
         the groups return to the frontier rather than vanishing"
    );
}

/// Row 9 — a space rename must move every `space`-keyed M6 row and change no
/// proof value. `space_id` is the identity; the name is a label, and every
/// digest is keyed on the ID precisely so this holds.
#[tokio::test]
async fn a_space_rename_moves_rows_and_changes_no_proof_value() {
    let (db, epoch) = base_scene().await;
    drive_to_quiescence(&db, epoch, NO_DURABLE_PARTITION).await;
    let before = incremental(&db, epoch).await;

    // The rename, and the row moves a real migration would perform with it.
    db.exec(
        "UPDATE spaces SET name = 'space-renamed' WHERE id = ?1",
        libsql::params![SPACE_ID],
    )
    .await;
    for table in [
        "edges",
        "genesis_candidates",
        "genesis_frontier",
        "genesis_coverage_state",
    ] {
        db.exec(
            &format!("UPDATE {table} SET space = 'space-renamed' WHERE space = ?1"),
            libsql::params![SPACE],
        )
        .await;
    }

    let after = incremental(&db, epoch).await;
    assert_eq!(
        after, before,
        "a rename changed a slot_id, page_id, candidate_id, or digest"
    );
    let full_after = full(&db, epoch, NO_DURABLE_PARTITION).await;
    assert_eq!(oracle::diff(&full_after, &after), Vec::new());
}

// ---------------------------------------------------------------------------
// Orphan-wikilink fixture (signal 2 needs a page/anchor/supports shape)
// ---------------------------------------------------------------------------

fn orphan_body(prefix: &str, label: &str, suffix: &str) -> (String, std::ops::Range<usize>) {
    let content = format!("{prefix}[[{label}]]{suffix}");
    let target_start = prefix.len() + 2; // past "[["
    (content, target_start..target_start + label.len())
}

/// Two referring pages for `label`, three independence groups across their
/// containing revisions — the shape `signals_test.rs` uses for signal 2's
/// positive control, minus the variations that module already covers.
async fn seed_orphan_baseline(db: &GenesisDb, label: &str) {
    let pages = [
        ("page-a", "Intro sentence has ", " in it.", "rev-a1"),
        ("page-b", "Another mention of ", " over here.", "rev-b1"),
    ];
    for (page_id, prefix, suffix, revision_id) in pages {
        let (content, span) = orphan_body(prefix, label, suffix);
        db.exec(
            "INSERT INTO pages (id, title, content, version, space, status)
             VALUES (?1, 'Note', ?2, 1, ?3, 'active')",
            libsql::params![page_id, content.as_str(), SPACE],
        )
        .await;
        db.exec(
            "INSERT INTO page_truth_state (page_id, page_version, support_status)
             VALUES (?1, 1, 'supported')",
            libsql::params![page_id],
        )
        .await;
        let label_key =
            super::label_key::normalize_label_key(label).expect("the fixture label normalizes");
        db.exec(
            "INSERT INTO page_links (source_page_id, target_page_id, label_key, label)
             VALUES (?1, NULL, ?2, ?3)",
            libsql::params![page_id, label_key, label],
        )
        .await;
        db.exec(
            "INSERT INTO page_version_claims (page_id, page_version, claim_revision_id, ordinal)
             VALUES (?1, 1, ?2, 0)",
            libsql::params![page_id, revision_id],
        )
        .await;
        db.exec(
            "INSERT INTO claim_anchors (claim_revision_id, source_doc_id, source_version,
                                        span_start, span_end, span_digest, created_at)
             VALUES (?1, ?2, 1, ?3, ?4, ?5, 0)",
            libsql::params![
                revision_id,
                page_id,
                span.start as i64,
                span.end as i64,
                crate::provenance::revision_content_digest(&content[span.clone()]),
            ],
        )
        .await;
    }

    // page-a's revision carries one group; page-b's carries two, for three.
    let supports = [
        ("supp-a1", "rev-a1", "root-a1", "group-a", "entity-a1"),
        ("supp-b1", "rev-b1", "root-b1", "group-b", "entity-b1"),
        ("supp-b2", "rev-b1", "root-b2", "group-c", "entity-b2"),
    ];
    for (edge_id, revision_id, root_id, group, entity_id) in supports {
        db.seed_evidence(root_id, SPACE, group, entity_id).await;
        db.exec(
            "INSERT INTO edges (edge_id, src_kind, src_id, dst_kind, dst_id, edge_type,
                                grounded, root_id, space, valid_until)
             VALUES (?1, 'claim_revision', ?2, 'memory', 'mem-x', 'supports', 0, ?3, ?4, NULL)",
            libsql::params![edge_id, revision_id, root_id, SPACE],
        )
        .await;
    }
}
