// SPDX-License-Identifier: Apache-2.0
//! Teeth for the C1 decay reference.
//!
//! These live under `src/` so the workspace library-test floor runs them; an
//! integration test would not be guaranteed in L3/L4.

use super::relevance_sweep::{
    advance_decay_reference, apply_group_mutation, candidate_endpoints, decay_reference,
    decayed_contribution, eligible_groups, group_support, hub_weight, pack_within_budget,
    pair_stats_snapshot, recompute_pair_stats, run_relevance_sweep, CandidateEndpoint,
    GroupSupport, QueryBudget, RelevanceRereference, RelevanceTurn, SpaceMass,
    COUNTER_RELEVANCE_DECAY_REFERENCE, DECAY_HALF_LIFE_DAYS, MAX_QUERIES_PER_EVALUATION,
    MAX_ROWS_PER_EVALUATION, SWEEP_CANDIDATE_WINDOW,
};
use crate::m6::genesis_test_support::GenesisDb;
use crate::m6::relevance::{
    normalized_pair_stats_digest, qualified_co_citation, PairStatsSnapshotRow, PairStatsValues,
    MAX_NEIGHBORS_PER_ENDPOINT,
};

const SECONDS_PER_DAY: i64 = 86_400;

/// One eligible page-supporting edge: an active, non-generated root in `group`
/// grounding an edge that touches `page`.
async fn seed_page_support(
    db: &GenesisDb,
    root_id: &str,
    space: &str,
    group: &str,
    page: &str,
    created_at: i64,
) {
    db.exec(
        "INSERT OR IGNORE INTO provenance_roots
             (root_id, root_kind, independence_group_id, status, created_at)
         VALUES (?1, 'document_ingest', ?2, 'active', ?3)",
        libsql::params![root_id, group, created_at],
    )
    .await;
    db.exec(
        "INSERT INTO pages (id, title, space) VALUES (?1, ?1, ?2)
         ON CONFLICT(id) DO NOTHING",
        libsql::params![page, space],
    )
    .await;
    db.exec(
        "INSERT INTO edges (edge_id, src_kind, src_id, dst_kind, dst_id,
                            grounded, root_id, space, valid_until)
         VALUES (?1, 'page', ?2, 'entity', 'peer', 1, ?3, ?4, NULL)",
        libsql::params![format!("{root_id}:{page}"), page, root_id, space],
    )
    .await;
}

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
async fn a_hub_that_touches_everything_contributes_almost_nothing() {
    // The contract's §3 reference table, verbatim. The d=5000 row is G6.10's:
    // an implementation that returns 1.0 from hub_weight passes every other
    // row here, so the hub row is the one that has to be present.
    assert!((hub_weight(32) - 1.0).abs() < 1e-12);
    assert!((hub_weight(64) - 1.0).abs() < 1e-12);
    assert!((hub_weight(128) - 0.5).abs() < 1e-12);
    assert!((hub_weight(5_000) - 0.0128).abs() < 1e-12);
    // A group touching nothing forms no pairs; zero, so a stray multiply adds
    // nothing rather than a phantom unit of support.
    assert_eq!(hub_weight(0), 0.0);
    assert_eq!(hub_weight(-1), 0.0);
}

#[tokio::test]
async fn a_group_reports_its_true_degree_while_selecting_only_sixty_four_pages() {
    // G6.7 and G6.10 are two caps, not one. Selecting 64 pages while reporting
    // degree=70 is what keeps them separable: an implementation that derives
    // the degree from the returned rows reads every hub as d=64, weight 1.0,
    // and passes a selection-count assertion while dropping the weighting.
    let db = GenesisDb::new().await;
    for index in 0..70 {
        seed_page_support(
            &db,
            &format!("root-{index:03}"),
            "space-a",
            "group-hub",
            &format!("page-{index:03}"),
            1_700_000_000 + index,
        )
        .await;
    }

    let tx = db.tx().await;
    let support = group_support(&tx, "space-a", "group-hub")
        .await
        .expect("read hub support")
        .expect("hub group has support");
    tx.commit().await.expect("commit");

    assert_eq!(support.pages.len(), 64, "selection is capped at the top 64");
    assert_eq!(support.degree, 70, "the reported degree is the true one");
    assert!(support.truncated(), "a 70-page group is a truncated sample");
    assert!(
        (hub_weight(support.degree) - 64.0 / 70.0).abs() < 1e-12,
        "the weight must come from the true degree, not the selected count"
    );
}

#[tokio::test]
async fn the_top_sixty_four_is_ordered_by_recency_then_page_id() {
    // S0-89. Recency alone is not a total order — two pages supported in the
    // same second tie — and a nondeterministic tiebreak makes the same group
    // produce different pair sets on two runs, breaking the oracle silently.
    let db = GenesisDb::new().await;
    // Three pages at one instant (so only page_id can order them) and one
    // strictly newer, which must sort ahead of all three.
    for page in ["page-c", "page-a", "page-b"] {
        seed_page_support(&db, &format!("root-{page}"), "space-a", "g", page, 1_000).await;
    }
    seed_page_support(&db, "root-newest", "space-a", "g", "page-z", 2_000).await;

    let tx = db.tx().await;
    let support = group_support(&tx, "space-a", "g")
        .await
        .expect("read support")
        .expect("group has support");
    tx.commit().await.expect("commit");

    assert_eq!(
        support.pages,
        vec!["page-z", "page-a", "page-b", "page-c"],
        "recency DESC first, then page_id ASC among the tied instant"
    );
    assert_eq!(support.newest_root_created_at, 2_000);
}

#[tokio::test]
async fn a_page_on_either_end_of_an_edge_counts_as_supported() {
    // The page column is a UNION ALL over both endpoint positions. A `CASE
    // WHEN src_kind='page' THEN src_id ELSE dst_id END` keeps only the source
    // side, so a page reached through an edge's destination silently loses all
    // its support — and every page-to-page edge loses one of its two pages.
    let db = GenesisDb::new().await;
    db.exec(
        "INSERT INTO provenance_roots (root_id, root_kind, independence_group_id, status, created_at)
         VALUES ('root-1', 'document_ingest', 'g', 'active', 1000)",
        (),
    )
    .await;
    for page in ["page-src", "page-dst"] {
        db.exec(
            "INSERT INTO pages (id, title, space) VALUES (?1, ?1, 'space-a')",
            libsql::params![page],
        )
        .await;
    }
    db.exec(
        "INSERT INTO edges (edge_id, src_kind, src_id, dst_kind, dst_id,
                            grounded, root_id, space, valid_until)
         VALUES ('edge-1', 'page', 'page-src', 'page', 'page-dst', 1, 'root-1', 'space-a', NULL)",
        (),
    )
    .await;

    let tx = db.tx().await;
    let support = group_support(&tx, "space-a", "g")
        .await
        .expect("read support")
        .expect("group has support");
    tx.commit().await.expect("commit");

    assert_eq!(
        support.pages,
        vec!["page-dst", "page-src"],
        "one page-to-page edge supports both of its pages"
    );
    assert_eq!(support.degree, 2);
}

#[tokio::test]
async fn support_selects_over_exactly_d1s_eligible_edges() {
    // The sweep shares ELIGIBLE_EVIDENCE_PREDICATE with D1's count. Each row
    // below is one clause of R1-R5; if the sweep grew its own spelling, a
    // group could clear D1's independence floor and contribute nothing to a
    // pair cell, or the reverse.
    let db = GenesisDb::new().await;
    seed_page_support(&db, "root-ok", "space-a", "g", "page-ok", 1_000).await;

    // R1: ungrounded.
    seed_page_support(&db, "root-ung", "space-a", "g", "page-ung", 1_000).await;
    db.exec(
        "UPDATE edges SET grounded = 0 WHERE edge_id = 'root-ung:page-ung'",
        (),
    )
    .await;
    // R2: retracted.
    seed_page_support(&db, "root-ret", "space-a", "g", "page-ret", 1_000).await;
    db.exec(
        "UPDATE edges SET valid_until = 1 WHERE edge_id = 'root-ret:page-ret'",
        (),
    )
    .await;
    // R3: root not active.
    seed_page_support(&db, "root-fail", "space-a", "g", "page-fail", 1_000).await;
    db.exec(
        "UPDATE provenance_roots SET status = 'failed' WHERE root_id = 'root-fail'",
        (),
    )
    .await;
    // R4: generated root.
    seed_page_support(&db, "root-gen", "space-a", "g", "page-gen", 1_000).await;
    db.exec(
        "UPDATE provenance_roots SET root_kind = 'generated' WHERE root_id = 'root-gen'",
        (),
    )
    .await;
    // R5: overview page.
    seed_page_support(&db, "root-ov", "space-a", "g", "page-ov", 1_000).await;
    db.exec(
        "UPDATE pages SET title = 'Overview' WHERE id = 'page-ov'",
        (),
    )
    .await;
    // Another space's edge is not this space's support.
    seed_page_support(&db, "root-other", "space-b", "g", "page-other", 1_000).await;

    let tx = db.tx().await;
    let support = group_support(&tx, "space-a", "g")
        .await
        .expect("read support")
        .expect("group has support");
    tx.commit().await.expect("commit");

    assert_eq!(
        support.pages,
        vec!["page-ok"],
        "only the fully eligible edge supports; got {:?}",
        support.pages
    );
    assert_eq!(support.degree, 1);
}

/// A group whose pages all share one instant, at full hub weight.
fn group(id: &str, pages: &[&str], created_at: i64) -> GroupSupport {
    GroupSupport {
        group_id: id.to_string(),
        degree: pages.len() as i64,
        newest_root_created_at: created_at,
        pages: pages.iter().map(|page| page.to_string()).collect(),
    }
}

#[tokio::test]
async fn the_four_cells_always_sum_to_the_spaces_total_mass() {
    // This identity is why n00 can be derived instead of accumulated, and
    // deriving it is what keeps the incremental path bounded — a stored n00
    // would have to be rewritten for every pair in the space each time one
    // group's eligibility changed.
    let reference = 1_800_000_000_i64;
    let groups = [
        group("g1", &["page-a", "page-b"], reference),
        group(
            "g2",
            &["page-a", "page-c"],
            reference - 90 * SECONDS_PER_DAY,
        ),
        group("g3", &["page-d"], reference - 360 * SECONDS_PER_DAY),
        group("g4", &["page-b", "page-c", "page-a"], reference),
    ];
    // N accumulated independently of the function under test.
    let total: f64 = groups
        .iter()
        .map(|g| decayed_contribution(hub_weight(g.degree), g.newest_root_created_at, reference))
        .sum();

    let stats = recompute_pair_stats(
        &groups,
        reference,
        &SpaceMass::from_groups(&groups, reference),
    );
    assert!(!stats.is_empty(), "these groups form pairs");
    for (pair, values) in &stats {
        let sum = values.n11 + values.n10 + values.n01 + values.n00;
        assert!(
            (sum - total).abs() < 1e-9,
            "{pair:?} cells sum to {sum}, not the space total {total}"
        );
    }
}

#[tokio::test]
async fn a_pair_counts_each_group_once_however_many_pages_it_shares() {
    // D1's collapse rule at the pair level: `distinct_group_count` counts
    // groups, so one document split across many roots of one group cannot
    // manufacture a floor-clearing pair.
    let reference = 1_800_000_000_i64;
    let groups = [
        group("g1", &["page-a", "page-b"], reference),
        group("g2", &["page-a", "page-b"], reference),
        group("g3", &["page-a", "page-b"], reference),
    ];
    let stats = recompute_pair_stats(
        &groups,
        reference,
        &SpaceMass::from_groups(&groups, reference),
    );
    let pair = stats
        .get(&("page-a".to_string(), "page-b".to_string()))
        .expect("the pair exists");

    assert_eq!(
        pair.distinct_group_count, 3,
        "three groups, counted once each"
    );
    assert!(
        qualified_co_citation(pair),
        "three distinct groups clears D1's floor"
    );
    // Nothing supports only one side, so n10 and n01 are empty.
    assert!(pair.n10.abs() < 1e-12 && pair.n01.abs() < 1e-12);
    assert!(pair.n00.abs() < 1e-12, "every group supports both");
}

#[tokio::test]
async fn the_floor_is_derived_at_read_time_not_stored() {
    // G6's D1 row: a retraction below three groups must read `false`
    // immediately. `qualified_co_citation` recomputes from the stored count,
    // so there is no cached bit that could stay true.
    let reference = 1_800_000_000_i64;
    let three = [
        group("g1", &["page-a", "page-b"], reference),
        group("g2", &["page-a", "page-b"], reference),
        group("g3", &["page-a", "page-b"], reference),
    ];
    let key = ("page-a".to_string(), "page-b".to_string());
    assert!(qualified_co_citation(
        recompute_pair_stats(
            &three,
            reference,
            &SpaceMass::from_groups(&three, reference)
        )
        .get(&key)
        .unwrap()
    ));

    // Drop one group — the same pair, now below the floor.
    let two = &three[..2];
    let below = recompute_pair_stats(two, reference, &SpaceMass::from_groups(two, reference));
    let pair = below.get(&key).unwrap();
    assert_eq!(pair.distinct_group_count, 2);
    assert!(
        !qualified_co_citation(pair),
        "two groups must read as not qualified the moment the third goes"
    );
}

#[tokio::test]
async fn a_hub_group_forms_at_most_two_thousand_and_sixteen_pairs() {
    // G6.7. C(64,2) = 2016, and it must follow from capping the *selection*:
    // an implementation that caps only the weight still forms C(5000,2) pairs.
    let reference = 1_800_000_000_i64;
    let pages: Vec<String> = (0..MAX_NEIGHBORS_PER_ENDPOINT)
        .map(|index| format!("page-{index:04}"))
        .collect();
    let hub = GroupSupport {
        group_id: "hub".to_string(),
        // A real 5000-page hub, cut to its top 64.
        degree: 5_000,
        newest_root_created_at: reference,
        pages,
    };

    let stats = recompute_pair_stats(
        std::slice::from_ref(&hub),
        reference,
        &SpaceMass::from_groups(std::slice::from_ref(&hub), reference),
    );
    assert_eq!(stats.len(), 2_016, "C(64,2)");
    let contribution = decayed_contribution(hub_weight(5_000), reference, reference);
    assert!(
        (contribution - 0.0128).abs() < 1e-12,
        "and each of those pairs carries the hub-damped weight, not 1.0"
    );
    for values in stats.values() {
        assert!((values.n11 - 0.0128).abs() < 1e-12);
    }
}

#[tokio::test]
async fn an_older_group_contributes_less_to_the_same_pair() {
    let reference = 1_800_000_000_i64;
    let fresh_groups = [group("g", &["page-a", "page-b"], reference)];
    let fresh = recompute_pair_stats(
        &fresh_groups,
        reference,
        &SpaceMass::from_groups(&fresh_groups, reference),
    );
    let aged_groups = [group(
        "g",
        &["page-a", "page-b"],
        reference - (DECAY_HALF_LIFE_DAYS as i64) * SECONDS_PER_DAY,
    )];
    let aged = recompute_pair_stats(
        &aged_groups,
        reference,
        &SpaceMass::from_groups(&aged_groups, reference),
    );
    let key = ("page-a".to_string(), "page-b".to_string());
    assert!((fresh.get(&key).unwrap().n11 - 1.0).abs() < 1e-12);
    assert!((aged.get(&key).unwrap().n11 - 0.5).abs() < 1e-12);
    assert_eq!(
        aged.get(&key).unwrap().distinct_group_count,
        1,
        "decay never touches the undecayed floor count"
    );
}

#[tokio::test]
async fn the_full_recompute_reads_groups_in_independence_group_order() {
    // S0-92. The order comes from the query, so this asserts the query's
    // ORDER BY rather than a Rust-side sort — there is one place it can be
    // wrong and this is it.
    let db = GenesisDb::new().await;
    // The recency of each group runs *opposite* to its id, so ordering by
    // anything time-shaped produces a different sequence than ordering by id.
    // With all three at one instant the two orderings coincide and the test
    // passes under a recency-ordered implementation — verified: that fixture
    // survived the RED mutation, which is what a fixture must never do.
    for (root, group_id, page, created_at) in [
        ("root-3", "g-c", "page-x", 3_000),
        ("root-1", "g-a", "page-y", 1_000),
        ("root-2", "g-b", "page-z", 2_000),
    ] {
        seed_page_support(&db, root, "space-a", group_id, page, created_at).await;
    }

    let tx = db.tx().await;
    let groups = eligible_groups(&tx, "space-a").await.expect("read groups");
    tx.commit().await.expect("commit");

    let ids: Vec<&str> = groups.iter().map(|g| g.group_id.as_str()).collect();
    assert_eq!(
        ids,
        vec!["g-a", "g-b", "g-c"],
        "groups arrive in independence_group_id ASC, whatever order they were written"
    );
}

#[tokio::test]
async fn a_group_with_no_eligible_support_reports_none() {
    let db = GenesisDb::new().await;
    let tx = db.tx().await;
    assert_eq!(
        group_support(&tx, "space-a", "absent")
            .await
            .expect("read absent group"),
        None,
        "an empty group is None, never a zero-degree row that would divide"
    );
    tx.commit().await.expect("commit");
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

/// Two pages co-supported by `groups` distinct groups, so the pair clears D1.
async fn seed_pair(db: &GenesisDb, space: &str, page_a: &str, page_b: &str, groups: usize) {
    for index in 0..groups {
        let group = format!("g-{page_a}-{page_b}-{index}");
        for page in [page_a, page_b] {
            seed_page_support(
                db,
                &format!("root-{group}-{page}"),
                space,
                &group,
                page,
                1_700_000_000 + index as i64,
            )
            .await;
        }
    }
}

/// Run the full re-reference pass that establishes the space's pin, returning
/// what it did. Every bounded turn measures against this, so a test that wants
/// to exercise the incremental path runs this first and then moves the
/// generation on — which is the only thing that makes a page stale again.
async fn rereference(
    db: &GenesisDb,
    space_id: &str,
    space: &str,
    generation: i64,
) -> RelevanceRereference {
    let tx = db.tx().await;
    let turn = run_relevance_sweep(&tx, space_id, space, generation, 1)
        .await
        .expect("re-reference pass");
    tx.commit().await.expect("commit re-reference");
    db.exec("DELETE FROM grouping_leases", ()).await;
    match turn {
        RelevanceTurn::Rereferenced(pass) => pass,
        other => panic!("expected the full re-reference pass, got {other:?}"),
    }
}

#[tokio::test]
async fn a_sweep_writes_both_tables_and_releases_its_lease() {
    let db = GenesisDb::new().await;
    db.seed_space("space-id-a", "space-a").await;
    seed_pair(&db, "space-a", "page-a", "page-b", 3).await;

    // Establish the pin, then move the generation on so the same pages are
    // stale again and the bounded path is what runs.
    rereference(&db, "space-id-a", "space-a", 7).await;
    let tx = db.tx().await;
    let turn = run_relevance_sweep(&tx, "space-id-a", "space-a", 8, 1)
        .await
        .expect("sweep");
    tx.commit().await.expect("commit");

    let sweep = match turn {
        RelevanceTurn::Swept(sweep) => sweep,
        other => panic!("expected a swept slice, got {other:?}"),
    };
    assert_eq!(sweep.endpoints_considered, 2);
    assert!(!sweep.endpoints_truncated);
    assert_eq!(sweep.pairs_written, 1, "one pair from two pages");
    assert_eq!(sweep.adjacency_rows_written, 2, "each page names the other");

    assert_eq!(
        db.scalar(
            "SELECT COUNT(*) FROM m6_pair_stats WHERE space = 'space-a'",
            ()
        )
        .await,
        1
    );
    assert_eq!(
        db.scalar(
            "SELECT distinct_group_count FROM m6_pair_stats
              WHERE page_a = 'page-a' AND page_b = 'page-b'",
            ()
        )
        .await,
        3,
        "three independent groups co-support the pair"
    );
    assert_eq!(
        db.scalar(
            "SELECT COUNT(*) FROM grouping_leases WHERE phase = 'relevance'",
            ()
        )
        .await,
        0,
        "the lease is released in the same transaction as the last write"
    );
}

#[tokio::test]
async fn a_sweep_whose_lease_is_held_reports_lease_held_and_writes_nothing() {
    let db = GenesisDb::new().await;
    db.seed_space("space-id-a", "space-a").await;
    seed_pair(&db, "space-a", "page-a", "page-b", 3).await;

    // A live lease for the same (phase, space, generation), as a concurrent
    // holder would leave it.
    db.exec(
        "INSERT INTO grouping_leases
             (phase, space, input_generation, token, expires_at, attempt)
         VALUES ('relevance', 'space-a', 7, 'other-worker', unixepoch() + 300, 1)",
        (),
    )
    .await;

    let tx = db.tx().await;
    let turn = run_relevance_sweep(&tx, "space-id-a", "space-a", 7, 1)
        .await
        .expect("sweep");
    tx.commit().await.expect("commit");

    assert_eq!(turn, RelevanceTurn::LeaseHeld);
    assert_eq!(
        db.scalar("SELECT COUNT(*) FROM m6_pair_stats", ()).await,
        0,
        "a refused turn writes nothing"
    );
    assert_eq!(
        db.scalar(
            "SELECT COUNT(*) FROM grouping_leases WHERE token = 'other-worker'",
            ()
        )
        .await,
        1,
        "and never releases the other holder's lease"
    );
}

#[tokio::test]
async fn a_space_with_nothing_stale_sweeps_idle() {
    let db = GenesisDb::new().await;
    db.seed_space("space-id-a", "space-a").await;

    let tx = db.tx().await;
    let turn = run_relevance_sweep(&tx, "space-id-a", "space-a", 7, 1)
        .await
        .expect("sweep");
    tx.commit().await.expect("commit");

    assert_eq!(turn, RelevanceTurn::Idle);
    assert_eq!(
        db.scalar("SELECT COUNT(*) FROM grouping_leases", ()).await,
        0,
        "an idle turn still releases, or the next turn is locked out for a TTL"
    );
}

#[tokio::test]
async fn the_sweep_drains_instead_of_re_sweeping_one_prefix() {
    // Liveness. `updated_generation` is what makes a swept page stop being
    // stale; without it every turn re-sweeps the same lexicographic prefix and
    // the tail is never reached, which every safety gate would still pass.
    let db = GenesisDb::new().await;
    db.seed_space("space-id-a", "space-a").await;
    // 40 pages > the 32 cap, so it takes more than one turn.
    for index in 0..20 {
        seed_pair(
            &db,
            "space-a",
            &format!("page-{:03}", index * 2),
            &format!("page-{:03}", index * 2 + 1),
            3,
        )
        .await;
    }

    let mut turns = 0;
    loop {
        let tx = db.tx().await;
        let turn = run_relevance_sweep(&tx, "space-id-a", "space-a", 7, 1)
            .await
            .expect("sweep");
        tx.commit().await.expect("commit");
        turns += 1;
        if turn == RelevanceTurn::Idle {
            break;
        }
        assert!(
            turns < 10,
            "the sweep is not draining; it re-swept a prefix"
        );
    }

    // Every page ended up covered by at least one pair row.
    let covered = db
        .scalar(
            "SELECT COUNT(DISTINCT id) FROM pages p
              WHERE p.space = 'space-a'
                AND EXISTS (SELECT 1 FROM m6_pair_stats s
                             WHERE s.space = 'space-a'
                               AND (s.page_a = p.id OR s.page_b = p.id))",
            (),
        )
        .await;
    assert_eq!(covered, 40, "every page was reached, not just the first 32");
}

#[tokio::test]
async fn a_candidate_set_that_hits_its_cap_is_recorded_as_truncated() {
    // S0-90: an attachment decision made over a silently truncated candidate
    // set is a decision whose inputs cannot be reconstructed.
    let db = GenesisDb::new().await;
    db.seed_space("space-id-a", "space-a").await;
    for index in 0..20 {
        seed_pair(
            &db,
            "space-a",
            &format!("page-{:03}", index * 2),
            &format!("page-{:03}", index * 2 + 1),
            3,
        )
        .await;
    }

    rereference(&db, "space-id-a", "space-a", 7).await;
    let tx = db.tx().await;
    let turn = run_relevance_sweep(&tx, "space-id-a", "space-a", 8, 1)
        .await
        .expect("sweep");
    tx.commit().await.expect("commit");

    let sweep = match turn {
        RelevanceTurn::Swept(sweep) => sweep,
        other => panic!("expected a swept slice, got {other:?}"),
    };
    assert_eq!(
        sweep.endpoints_considered, SWEEP_CANDIDATE_WINDOW,
        "the slice is applied at the query"
    );
    assert!(
        sweep.endpoints_truncated,
        "40 stale pages against a {SWEEP_CANDIDATE_WINDOW} slice is a \
         truncated set and must say so"
    );
}

#[tokio::test]
async fn the_candidate_cap_is_enforced_by_the_query_not_by_a_later_truncate() {
    // G6.8's break is a post-hoc `Vec::truncate`: the returned length and the
    // truncated flag both still pass while the query reads the whole table, so
    // neither of those is evidence about the cap. The decoded-row count is —
    // it is the one observation that distinguishes "the database stopped at
    // 33" from "the database returned everything and Rust threw the rest
    // away". Exactly `SWEEP_CANDIDATE_WINDOW + 1`: the slice, plus the single
    // extra row that makes truncation observable without a second query.
    let db = GenesisDb::new().await;
    db.seed_space("space-id-a", "space-a").await;
    for index in 0..20 {
        seed_pair(
            &db,
            "space-a",
            &format!("page-{:03}", index * 2),
            &format!("page-{:03}", index * 2 + 1),
            3,
        )
        .await;
    }

    // Staleness is measured against existing rows, so the pin has to exist
    // before any page can be behind it.
    rereference(&db, "space-id-a", "space-a", 7).await;
    let tx = db.tx().await;
    let mut budget = QueryBudget::default();
    let (endpoints, truncated) = candidate_endpoints(&tx, "space-a", 8, &mut budget)
        .await
        .expect("candidate endpoints");
    tx.commit().await.expect("commit");

    assert_eq!(endpoints.len(), SWEEP_CANDIDATE_WINDOW);
    assert!(truncated);
    assert_eq!(
        budget.rows as usize,
        SWEEP_CANDIDATE_WINDOW + 1,
        "the candidate query decoded {} rows over a 40-page space; the slice \
         must be the LIMIT, not a truncate afterwards",
        budget.rows
    );
}

#[tokio::test]
async fn a_sweep_stays_inside_the_query_and_row_budget() {
    // G6.11b and G6.11c. Both counters are runtime counts, not SQL text: a
    // `LIMIT 512` on an unindexed predicate visits the whole table and still
    // reports 512 rows, which is why S0-95 makes the counter normative.
    let db = GenesisDb::new().await;
    db.seed_space("space-id-a", "space-a").await;
    for index in 0..20 {
        seed_pair(
            &db,
            "space-a",
            &format!("page-{:03}", index * 2),
            &format!("page-{:03}", index * 2 + 1),
            3,
        )
        .await;
    }

    rereference(&db, "space-id-a", "space-a", 7).await;
    let tx = db.tx().await;
    let turn = run_relevance_sweep(&tx, "space-id-a", "space-a", 8, 1)
        .await
        .expect("sweep");
    tx.commit().await.expect("commit");

    let sweep = match turn {
        RelevanceTurn::Swept(sweep) => sweep,
        other => panic!("expected a swept slice, got {other:?}"),
    };
    assert!(
        sweep.budget.queries <= MAX_QUERIES_PER_EVALUATION,
        "{} queries exceeds the cap of {MAX_QUERIES_PER_EVALUATION}",
        sweep.budget.queries
    );
    assert!(
        sweep.budget.rows <= MAX_ROWS_PER_EVALUATION,
        "{} rows exceeds the cap of {MAX_ROWS_PER_EVALUATION}",
        sweep.budget.rows
    );
    assert!(
        sweep.budget.queries > 0 && sweep.budget.rows > 0,
        "a budget of zero would pass every bound without reading anything"
    );
}

#[tokio::test]
async fn adjacency_ranks_are_contiguous_from_one_and_ordered_by_recency() {
    // S0-89 plus the table's `rank BETWEEN 1 AND 64` CHECK. A rewrite that
    // upserted instead of deleting first would leave a neighbour at its old
    // rank and the endpoint would carry two rows for one neighbour.
    let db = GenesisDb::new().await;
    db.seed_space("space-id-a", "space-a").await;
    // One group over four pages, so page-hub gets three ranked neighbours.
    for (page, created_at) in [
        ("page-hub", 1_000),
        ("page-old", 1_000),
        ("page-mid", 1_000),
        ("page-new", 1_000),
    ] {
        seed_page_support(
            &db,
            &format!("root-{page}"),
            "space-a",
            "g-one",
            page,
            created_at,
        )
        .await;
    }

    let tx = db.tx().await;
    run_relevance_sweep(&tx, "space-id-a", "space-a", 7, 1)
        .await
        .expect("sweep");
    tx.commit().await.expect("commit");

    let rows = db
        .scalar(
            "SELECT COUNT(*) FROM m6_adjacency
              WHERE space = 'space-a' AND endpoint_id = 'page-hub'",
            (),
        )
        .await;
    assert_eq!(rows, 3, "three neighbours");
    assert_eq!(
        db.scalar(
            "SELECT MIN(rank) FROM m6_adjacency WHERE endpoint_id = 'page-hub'",
            ()
        )
        .await,
        1,
        "ranks start at 1, which the CHECK requires"
    );
    assert_eq!(
        db.scalar(
            "SELECT COUNT(DISTINCT rank) FROM m6_adjacency WHERE endpoint_id = 'page-hub'",
            ()
        )
        .await,
        3,
        "and are contiguous, never duplicated"
    );

    // Re-sweeping the same endpoint must not accumulate rows.
    db.exec("UPDATE m6_pair_stats SET updated_generation = 0", ())
        .await;
    let tx = db.tx().await;
    run_relevance_sweep(&tx, "space-id-a", "space-a", 7, 2)
        .await
        .expect("second sweep");
    tx.commit().await.expect("commit");
    assert_eq!(
        db.scalar(
            "SELECT COUNT(*) FROM m6_adjacency WHERE endpoint_id = 'page-hub'",
            ()
        )
        .await,
        3,
        "a rewrite replaces the endpoint's rows rather than adding to them"
    );
}

// ---------------------------------------------------------------------------
// S0-91 — the incremental-equals-full oracle, and its negative control.
// ---------------------------------------------------------------------------

/// The space's stored pair table, in the digest's input shape.
async fn stored_pair_rows(db: &GenesisDb, space: &str) -> Vec<(String, String, PairStatsValues)> {
    // Deliberately the production snapshot reader, not a direct SELECT of the
    // stored cells. `n10`/`n01`/`n00` are derived from the pinned marginals;
    // a test that read the columns would be asserting against a cache the
    // bounded mutation path does not refresh, and would pass while the
    // estimator the readers actually see was wrong.
    let tx = db.tx().await;
    let rows = pair_stats_snapshot(&tx, space)
        .await
        .expect("pair stats snapshot");
    tx.commit().await.expect("commit snapshot read");
    rows
}

fn digest_of(rows: &[(String, String, PairStatsValues)]) -> String {
    let snapshot: Vec<PairStatsSnapshotRow<'_>> = rows
        .iter()
        .map(|(page_a, page_b, values)| PairStatsSnapshotRow {
            page_a,
            page_b,
            values: *values,
        })
        .collect();
    normalized_pair_stats_digest(&snapshot).expect("digest the snapshot")
}

/// The oracle's right-hand side: what a full recomputation of the whole space
/// produces at `reference`. Deliberately the unbounded read — every eligible
/// group, which is exactly what a bounded turn may not do.
async fn full_recompute_rows(
    db: &GenesisDb,
    space: &str,
    reference: i64,
) -> Vec<(String, String, PairStatsValues)> {
    let tx = db.tx().await;
    let groups = eligible_groups(&tx, space).await.expect("eligible groups");
    tx.commit().await.expect("commit full read");
    let mass = SpaceMass::from_groups(&groups, reference);
    recompute_pair_stats(&groups, reference, &mass)
        .into_iter()
        .map(|((page_a, page_b), values)| (page_a, page_b, values))
        .collect()
}

/// Run **bounded** turns until the sweep reports `Idle`, returning the turn
/// count and the largest per-turn decoded-row count. Reaching the full
/// re-reference pass here is a failure: the caller establishes the pin first,
/// so a second pass would mean the incremental path silently re-referenced.
async fn drain_bounded(
    db: &GenesisDb,
    space_id: &str,
    space: &str,
    generation: i64,
) -> (usize, u32) {
    let mut turns = 0;
    let mut peak_rows = 0;
    loop {
        let tx = db.tx().await;
        let turn = run_relevance_sweep(&tx, space_id, space, generation, 1)
            .await
            .expect("sweep turn");
        tx.commit().await.expect("commit sweep turn");
        db.exec("DELETE FROM grouping_leases", ()).await;
        match turn {
            RelevanceTurn::Idle => return (turns, peak_rows),
            RelevanceTurn::LeaseHeld => panic!("nothing else holds the lease in this test"),
            RelevanceTurn::Rereferenced(pass) => {
                panic!("the incremental path re-referenced the whole space: {pass:?}")
            }
            RelevanceTurn::Swept(sweep) => {
                peak_rows = peak_rows.max(sweep.budget.rows);
                turns += 1;
                assert!(turns < 64, "the sweep must drain, not spin");
            }
        }
    }
}

/// 60 pages over 30 overlapping groups. The page count is deliberately past
/// `SWEEP_CANDIDATE_WINDOW`, so the bounded path needs several turns and no
/// single turn sees every group — the only configuration in which a marginal
/// summed over one turn's own subset can diverge from the space-wide one. A
/// fixture that fits in a single turn passes this vacuously.
async fn seed_overlapping_space(db: &GenesisDb) {
    db.seed_space("space-id-a", "space-a").await;
    for index in 0..30 {
        let group = format!("g-{index:02}");
        for offset in 0..4 {
            let page = format!("page-{:02}", (index * 2 + offset) % 60);
            seed_page_support(
                db,
                &format!("root-{group}-{page}"),
                "space-a",
                &group,
                &page,
                1_700_000_000 + index as i64 * 1_000,
            )
            .await;
        }
    }
}

#[tokio::test]
async fn the_bounded_sweep_reproduces_a_full_recomputation() {
    // S0-91's oracle. Byte equality of the digest, not a per-cell float
    // tolerance: the reference and `N` are both pinned, so the two paths agree
    // exactly or the design is wrong.
    let db = GenesisDb::new().await;
    seed_overlapping_space(&db).await;

    let pass = rereference(&db, "space-id-a", "space-a", 7).await;
    assert_eq!(
        pass.pages_referenced, 60,
        "the pass references the whole space"
    );

    // Moving the generation on makes every page stale again, so the bounded
    // path rewrites from scratch what the full pass wrote — the comparison is
    // only meaningful because the incremental path really did the work.
    let (turns, peak_rows) = drain_bounded(&db, "space-id-a", "space-a", 8).await;
    assert!(
        turns >= 2,
        "60 pages against a 32 cap must take more than one bounded turn, took {turns}"
    );

    // The drain is only finished when *every* pair is at the new generation.
    // Values alone cannot show this: the re-reference pass already wrote them
    // correctly, so a page retired before its own sweep leaves rows that are
    // right but stamped stale — and they stay wrong the moment the underlying
    // groups next change. The generation is the only observation that
    // separates "re-derived" from "left behind".
    let behind = db
        .scalar(
            "SELECT COUNT(*) FROM m6_pair_stats
              WHERE space = 'space-a' AND updated_generation < 8",
            (),
        )
        .await;
    assert_eq!(
        behind, 0,
        "the drain reported Idle with {behind} pairs still at an older \
         generation; a page was retired before its own pairs were re-derived"
    );

    let reference = {
        let tx = db.tx().await;
        let reference = decay_reference(&tx, "space-id-a")
            .await
            .expect("read reference")
            .expect("the pass established a reference");
        tx.commit().await.expect("commit");
        reference
    };

    let stored = stored_pair_rows(&db, "space-a").await;
    let full = full_recompute_rows(&db, "space-a", reference).await;
    assert_eq!(
        stored.len(),
        full.len(),
        "the two paths wrote different pairs"
    );
    // Compared through the digest, not cell-by-cell: the two paths reach the
    // same cells by different association orders (the bounded path's marginals
    // come back through storage), so they can differ in the last bit. S0-92's
    // 9-dp rounding is the contract's own answer to that, and asserting raw f64
    // equality would be a stricter gate than the artifact specifies.
    assert_eq!(
        digest_of(&stored),
        digest_of(&full),
        "the bounded sweep's table must be byte-identical to a full \
         recomputation at the same pin"
    );

    // The negative control. An implementation that reached the same digest by
    // re-referencing the whole space on every turn fails here rather than
    // passing as a correct incremental one.
    assert!(
        peak_rows <= MAX_ROWS_PER_EVALUATION,
        "a bounded turn decoded {peak_rows} rows; the incremental bound is \
         {MAX_ROWS_PER_EVALUATION}"
    );
}

fn candidate(page: &str, projected_rows: i64) -> CandidateEndpoint {
    CandidateEndpoint {
        page_id: page.to_string(),
        projected_rows,
    }
}

#[test]
fn packing_stops_before_it_would_spend_the_row_budget() {
    // Each endpoint reserves twice its projection (the `groups_touching` row
    // plus the `pinned_space_mass` row its page can add), so 150 projected
    // rows reserves 300 and two of them cannot both fit under 512.
    let budget = QueryBudget {
        queries: 1,
        rows: 9,
    };
    let chosen = pack_within_budget(
        &[candidate("page-a", 150), candidate("page-b", 150)],
        &budget,
    );
    assert_eq!(
        chosen,
        vec!["page-a".to_string()],
        "two 300-row reservations do not fit in what is left of 512; the \
         second endpoint must wait for the next turn"
    );
}

#[test]
fn packing_always_takes_one_endpoint_even_when_it_does_not_fit() {
    // The progress guarantee. A turn that reserves its way to zero endpoints
    // writes nothing, so the page stays stale, so the next turn sees exactly
    // the same candidate and also writes nothing — the sweep would never
    // converge and the staleness would be permanent. Taking the endpoint and
    // letting the row counter report the overrun is the honest failure: it is
    // loud, and it names the one case `pack_within_budget` cannot avoid.
    let budget = QueryBudget {
        queries: 1,
        rows: 9,
    };
    let chosen = pack_within_budget(&[candidate("page-huge", 100_000)], &budget);
    assert_eq!(
        chosen,
        vec!["page-huge".to_string()],
        "an endpoint too big for the whole budget must still be attempted, or \
         the sweep stalls on it forever"
    );
}

#[test]
fn packing_takes_every_endpoint_that_fits() {
    let budget = QueryBudget {
        queries: 1,
        rows: 9,
    };
    let chosen = pack_within_budget(
        &[
            candidate("page-a", 10),
            candidate("page-b", 10),
            candidate("page-c", 10),
        ],
        &budget,
    );
    assert_eq!(chosen.len(), 3, "60 reserved rows fit inside 512 easily");
}

/// Four groups over three pages, so every pair keeps support after one group
/// leaves and the comparison is about *values*, not about rows vanishing.
async fn seed_mutable_space(db: &GenesisDb) {
    let base = 1_800_000_000;
    for (group, pages) in [
        ("g0", vec!["p1", "p2", "p3"]),
        ("g1", vec!["p1", "p2"]),
        ("g2", vec!["p2", "p3"]),
        ("g3", vec!["p1", "p3"]),
    ] {
        for page in pages {
            seed_page_support(
                db,
                &format!("r-{group}-{page}"),
                "space-a",
                group,
                page,
                base,
            )
            .await;
        }
    }
}

/// A space whose groups do **not** all touch all pages.
///
/// [`seed_mutable_space`] is degenerate for two cells: every group touches
/// every page, so `n00` is identically zero (each pair's `total - mass_a -
/// mass_b + n11` cancels) and no pair can lose its last supporting group.
/// Two RED controls -- dropping `+ n11` from the `n00` derivation, and never
/// deleting a pair whose count reaches zero -- both stayed green against it.
/// Here `g3` touches neither `p1` nor `p2`, so that pair carries real `n00`
/// mass, and `g2` is the sole supporter of `(p3, p4)`.
async fn seed_sparse_space(db: &GenesisDb) {
    let base = 1_800_000_000;
    for (group, pages) in [
        ("g0", vec!["p1", "p2", "p3"]),
        ("g1", vec!["p1", "p2"]),
        ("g2", vec!["p3", "p4"]),
        ("g3", vec!["p5", "p6"]),
    ] {
        for page in pages {
            seed_page_support(
                db,
                &format!("r-{group}-{page}"),
                "space-a",
                group,
                page,
                base,
            )
            .await;
        }
    }
}

/// Assert the space's whole pair table equals a full recomputation at the pin,
/// and return it so a caller can make its own non-vacuity claims.
async fn assert_equals_full_recomputation(
    db: &GenesisDb,
    space: &str,
    reference: i64,
    what: &str,
) -> Vec<(String, String, PairStatsValues)> {
    let stored = stored_pair_rows(db, space).await;
    let full = full_recompute_rows(db, space, reference).await;
    assert_eq!(
        digest_of(&stored),
        digest_of(&full),
        "the stored table and a full recomputation disagree after {what}\n\
         stored: {stored:#?}\nfull: {full:#?}"
    );
    stored
}

/// The claim `run_relevance_sweep` relies on when it refreshes a marginal but
/// leaves `N` pinned: a support change moves the space total only above the
/// hub cap.
///
/// `hub_weight(d) = 1` for every `d <= 64`, so a group under the cut
/// contributes the same whether it touches two pages or sixty. Above it the
/// weight is `64/d` and the contribution really does move, which is the one
/// case where a bounded turn leaves `N` stale until the next full
/// re-reference -- the bounded, queryable staleness Q1's pin sanctions.
#[test]
fn a_support_change_moves_the_space_total_only_above_the_hub_cap() {
    let created_at = 1_800_000_000;
    let group = |degree: usize| GroupSupport {
        group_id: "g".to_string(),
        degree: degree as i64,
        newest_root_created_at: created_at,
        pages: (0..degree.min(MAX_NEIGHBORS_PER_ENDPOINT))
            .map(|index| format!("p{index}"))
            .collect(),
    };
    let total = |degree: usize| SpaceMass::from_groups(&[group(degree)], created_at).total;

    assert_eq!(
        total(2),
        total(MAX_NEIGHBORS_PER_ENDPOINT),
        "under the cut the weight is 1 regardless of degree, so a support \
         change cannot move N"
    );
    assert!(
        total(MAX_NEIGHBORS_PER_ENDPOINT + 1) < total(MAX_NEIGHBORS_PER_ENDPOINT),
        "crossing the cut must lower the contribution, which is what leaves N \
         stale until the next full re-reference"
    );
}

/// S0-91 mutation: root activation.
///
/// A group that was not in the eligible universe joins it. That is the
/// applier's case rather than the sweep's -- the space total counts each
/// eligible group once, so only a path that owns `N` may add one.
#[tokio::test]
async fn a_root_activation_leaves_the_table_equal_to_a_full_recomputation() {
    let db = GenesisDb::new().await;
    db.seed_space("space-id-a", "space-a").await;
    seed_sparse_space(&db).await;

    let pass = rereference(&db, "space-id-a", "space-a", 7).await;
    let reference = pass.decay_reference;
    let before = stored_pair_rows(&db, "space-a").await;

    // g4 arrives already active over two pages that already have other support.
    for page in ["p1", "p2"] {
        seed_page_support(
            &db,
            &format!("r-g4-{page}"),
            "space-a",
            "g4",
            page,
            1_800_000_000,
        )
        .await;
    }

    let tx = db.tx().await;
    let group = group_support(&tx, "space-a", "g4")
        .await
        .expect("group support")
        .expect("g4 is eligible once its roots exist");
    tx.commit().await.expect("commit read");

    let tx = db.tx().await;
    apply_group_mutation(
        &tx,
        "space-a",
        &group,
        reference,
        crate::m6::relevance::EligibilityChange::Reactivated,
    )
    .await
    .expect("apply activation");
    tx.commit().await.expect("commit mutation");

    let stored =
        assert_equals_full_recomputation(&db, "space-a", reference, "a root activation").await;
    let group_count = |rows: &[(String, String, PairStatsValues)]| {
        rows.iter()
            .find(|(page_a, page_b, _)| page_a == "p1" && page_b == "p2")
            .map(|(_, _, values)| values.distinct_group_count)
            .expect("(p1, p2) exists")
    };
    assert_eq!(
        group_count(&stored),
        group_count(&before) + 1,
        "activating a group must raise the pair's distinct-group count"
    );
}

/// S0-91 mutation: page support gain.
///
/// `g1` gains `p3`, which it did not support before. Reached through the
/// sweep: the group was already in the universe, so `N` does not move, and
/// `g1` stays under the top-64 cut so its own contribution does not either.
#[tokio::test]
async fn a_support_gain_leaves_the_table_equal_to_a_full_recomputation() {
    let db = GenesisDb::new().await;
    db.seed_space("space-id-a", "space-a").await;
    seed_sparse_space(&db).await;

    let pass = rereference(&db, "space-id-a", "space-a", 7).await;
    let reference = pass.decay_reference;
    let before = stored_pair_rows(&db, "space-a").await;

    seed_page_support(&db, "r-g1-p3", "space-a", "g1", "p3", 1_800_000_000).await;
    drain_bounded(&db, "space-id-a", "space-a", 8).await;

    let stored =
        assert_equals_full_recomputation(&db, "space-a", reference, "a support gain").await;
    let count_of = |rows: &[(String, String, PairStatsValues)], a: &str, b: &str| {
        rows.iter()
            .find(|(page_a, page_b, _)| page_a == a && page_b == b)
            .map(|(_, _, values)| values.distinct_group_count)
            .unwrap_or(0)
    };
    assert_eq!(
        count_of(&stored, "p1", "p3"),
        count_of(&before, "p1", "p3") + 1,
        "the newly co-supported pair must gain a group"
    );
}

/// S0-91 mutation: community rebinding.
///
/// Rebinding advances the grouping generation without touching any evidence.
/// The pair table is keyed on pages and independence groups, so its *values*
/// must come back identical -- only the generation stamp may move. A sweep
/// that let community membership leak into a cell would show up here.
#[tokio::test]
async fn a_community_rebinding_changes_no_pair_value() {
    let db = GenesisDb::new().await;
    db.seed_space("space-id-a", "space-a").await;
    seed_sparse_space(&db).await;

    let pass = rereference(&db, "space-id-a", "space-a", 7).await;
    let reference = pass.decay_reference;
    let before = digest_of(&stored_pair_rows(&db, "space-a").await);

    let (turns, _) = drain_bounded(&db, "space-id-a", "space-a", 8).await;
    assert!(turns > 0, "the rebinding must actually re-derive something");

    let stored =
        assert_equals_full_recomputation(&db, "space-a", reference, "a community rebinding").await;
    assert_eq!(
        digest_of(&stored),
        before,
        "a rebinding carries no evidence change, so no cell may move"
    );
    let behind = db
        .scalar(
            "SELECT COUNT(*) FROM m6_pair_stats
              WHERE space = 'space-a' AND updated_generation < 8",
            (),
        )
        .await;
    assert_eq!(
        behind, 0,
        "every pair must be restamped at the new generation"
    );
}

/// S0-91 mutation: page deletion.
#[tokio::test]
async fn a_page_deletion_leaves_the_table_equal_to_a_full_recomputation() {
    let db = GenesisDb::new().await;
    db.seed_space("space-id-a", "space-a").await;
    seed_sparse_space(&db).await;

    let pass = rereference(&db, "space-id-a", "space-a", 7).await;
    let reference = pass.decay_reference;
    assert!(
        stored_pair_rows(&db, "space-a")
            .await
            .iter()
            .any(|(page_a, page_b, _)| page_a == "p3" || page_b == "p3"),
        "the page about to be deleted must carry pairs first"
    );

    db.exec("DELETE FROM edges WHERE src_id = 'p3'", ()).await;
    db.exec("DELETE FROM pages WHERE id = 'p3'", ()).await;
    drain_bounded(&db, "space-id-a", "space-a", 8).await;

    let stored =
        assert_equals_full_recomputation(&db, "space-a", reference, "a page deletion").await;
    assert!(
        !stored
            .iter()
            .any(|(page_a, page_b, _)| page_a == "p3" || page_b == "p3"),
        "no pair may reference a deleted page"
    );
}

/// S0-91 mutation: space move.
///
/// From the source space this looks like a deletion, and it must not leave the
/// moved page's pairs behind -- a stale pair keyed on a page that now lives
/// elsewhere is a cross-space leak into whatever reads the table next.
#[tokio::test]
async fn a_space_move_leaves_the_source_space_equal_to_a_full_recomputation() {
    let db = GenesisDb::new().await;
    db.seed_space("space-id-a", "space-a").await;
    db.seed_space("space-id-b", "space-b").await;
    seed_sparse_space(&db).await;

    let pass = rereference(&db, "space-id-a", "space-a", 7).await;
    let reference = pass.decay_reference;

    db.exec("UPDATE pages SET space = 'space-b' WHERE id = 'p3'", ())
        .await;
    db.exec("UPDATE edges SET space = 'space-b' WHERE src_id = 'p3'", ())
        .await;
    drain_bounded(&db, "space-id-a", "space-a", 8).await;

    let stored = assert_equals_full_recomputation(&db, "space-a", reference, "a space move").await;
    assert!(
        !stored
            .iter()
            .any(|(page_a, page_b, _)| page_a == "p3" || page_b == "p3"),
        "the source space must not keep pairs for a page that moved away"
    );
}

/// S0-91's closing requirement: the eight mutations interleaved.
///
/// Fixed order rather than a seeded shuffle -- the point is that the paths
/// compose, and a randomized order that only ever runs once is a fixed order
/// that nobody can reproduce from the failure message.
#[tokio::test]
async fn interleaved_mutations_leave_the_table_equal_to_a_full_recomputation() {
    let db = GenesisDb::new().await;
    db.seed_space("space-id-a", "space-a").await;
    db.seed_space("space-id-b", "space-b").await;
    seed_sparse_space(&db).await;

    let pass = rereference(&db, "space-id-a", "space-a", 7).await;
    let reference = pass.decay_reference;
    let mut generation = 8;

    // support gain, then a retraction through the applier, then a deletion,
    // then a reactivation, then a move -- alternating the two mechanisms so a
    // path that corrupted the other's state would not survive the sequence.
    seed_page_support(&db, "r-g1-p3", "space-a", "g1", "p3", 1_800_000_000).await;
    drain_bounded(&db, "space-id-a", "space-a", generation).await;
    generation += 1;

    let tx = db.tx().await;
    let g3 = group_support(&tx, "space-a", "g3")
        .await
        .expect("group support")
        .expect("g3 eligible");
    tx.commit().await.expect("commit read");
    for change in [
        crate::m6::relevance::EligibilityChange::Retracted,
        crate::m6::relevance::EligibilityChange::Reactivated,
    ] {
        let tx = db.tx().await;
        apply_group_mutation(&tx, "space-a", &g3, reference, change)
            .await
            .expect("apply mutation");
        tx.commit().await.expect("commit mutation");
    }

    db.exec("DELETE FROM edges WHERE src_id = 'p4'", ()).await;
    db.exec("DELETE FROM pages WHERE id = 'p4'", ()).await;
    drain_bounded(&db, "space-id-a", "space-a", generation).await;
    generation += 1;

    db.exec("UPDATE pages SET space = 'space-b' WHERE id = 'p5'", ())
        .await;
    db.exec("UPDATE edges SET space = 'space-b' WHERE src_id = 'p5'", ())
        .await;
    drain_bounded(&db, "space-id-a", "space-a", generation).await;

    let stored =
        assert_equals_full_recomputation(&db, "space-a", reference, "an interleaved sequence")
            .await;
    assert!(
        !stored.is_empty(),
        "the sequence must not empty the table, or the equality is vacuous"
    );
    assert!(
        !stored.iter().any(
            |(page_a, page_b, _)| ["p4", "p5"].contains(&page_a.as_str())
                || ["p4", "p5"].contains(&page_b.as_str())
        ),
        "neither the deleted page nor the moved one may survive"
    );
}

/// S0-91 mutation: page support loss.
///
/// `g0` is the only group co-supporting `(p1, p3)`, so retiring its `p3` edge
/// must remove that pair. The sweep reaches it through generation staleness
/// rather than through [`apply_group_mutation`] -- losing an edge is a
/// substrate change, not a group-eligibility transition.
#[tokio::test]
async fn a_support_loss_that_empties_a_pair_still_equals_a_full_recomputation() {
    let db = GenesisDb::new().await;
    db.seed_space("space-id-a", "space-a").await;
    seed_sparse_space(&db).await;

    let pass = rereference(&db, "space-id-a", "space-a", 7).await;
    let reference = pass.decay_reference;

    let before = stored_pair_rows(&db, "space-a").await;
    assert!(
        before
            .iter()
            .any(|(page_a, page_b, _)| page_a == "p1" && page_b == "p3"),
        "the pair about to lose its only supporting group must exist first"
    );

    db.exec(
        "UPDATE edges SET valid_until = 1 WHERE root_id = 'r-g0-p3'",
        (),
    )
    .await;

    drain_bounded(&db, "space-id-a", "space-a", 8).await;

    let stored = stored_pair_rows(&db, "space-a").await;
    let full = full_recompute_rows(&db, "space-a", reference).await;
    assert!(
        !stored
            .iter()
            .any(|(page_a, page_b, _)| page_a == "p1" && page_b == "p3"),
        "a pair whose last supporting edge retired must not survive the sweep"
    );
    assert_eq!(
        digest_of(&stored),
        digest_of(&full),
        "the swept table and a full recomputation disagree after a support loss\n\
         stored: {stored:#?}\nfull: {full:#?}"
    );
}

/// S0-91's oracle over the cells [`seed_mutable_space`] cannot reach.
///
/// Retracting `g2` takes the last group off `(p3, p4)`, so the incremental
/// path must *delete* that row rather than zero it -- a full recomputation
/// emits no row at all. The surviving pairs keep non-zero `n00`, so the
/// derivation is compared against a real quantity instead of a clamped zero.
#[tokio::test]
async fn a_retraction_that_empties_a_pair_still_equals_a_full_recomputation() {
    let db = GenesisDb::new().await;
    db.seed_space("space-id-a", "space-a").await;
    seed_sparse_space(&db).await;

    let pass = rereference(&db, "space-id-a", "space-a", 7).await;
    let reference = pass.decay_reference;

    let before = stored_pair_rows(&db, "space-a").await;
    assert!(
        before
            .iter()
            .any(|(page_a, page_b, _)| page_a == "p3" && page_b == "p4"),
        "the pair about to lose its last group must exist first"
    );

    let tx = db.tx().await;
    let group = group_support(&tx, "space-a", "g2")
        .await
        .expect("group support")
        .expect("g2 is eligible before the mutation");
    tx.commit().await.expect("commit read");

    db.exec(
        "UPDATE provenance_roots SET status = 'retracted'
          WHERE independence_group_id = 'g2'",
        (),
    )
    .await;

    let tx = db.tx().await;
    let receipt = apply_group_mutation(
        &tx,
        "space-a",
        &group,
        reference,
        crate::m6::relevance::EligibilityChange::Retracted,
    )
    .await
    .expect("apply retraction");
    tx.commit().await.expect("commit mutation");

    let incremental = stored_pair_rows(&db, "space-a").await;
    let full = full_recompute_rows(&db, "space-a", reference).await;

    // Non-vacuity, both directions: the emptied pair is gone, and what remains
    // exercises the derivation rather than a clamped zero. Without the second
    // assert a future fixture edit could re-degenerate `n00` and the digest
    // comparison would pass for the wrong reason.
    assert!(
        !incremental
            .iter()
            .any(|(page_a, page_b, _)| page_a == "p3" && page_b == "p4"),
        "the pair whose last group retracted must be deleted, not zeroed"
    );
    assert!(
        incremental.iter().any(|(_, _, values)| values.n00 > 0.0),
        "a sparse space must leave groups supporting neither page of some pair"
    );
    assert_eq!(
        digest_of(&incremental),
        digest_of(&full),
        "the incremental retraction and a full recomputation disagree\n\
         incremental: {incremental:#?}\nfull: {full:#?}"
    );

    assert_eq!(receipt.pages_touched, 2);
    assert_eq!(receipt.pairs_touched, 1);
}

#[tokio::test]
async fn a_retraction_leaves_the_table_equal_to_a_full_recomputation() {
    // S0-91's oracle for the retraction mutation: the incremental path and a
    // full recomputation must agree byte-for-byte on the normalized digest at
    // a fixed relevance generation.
    let db = GenesisDb::new().await;
    db.seed_space("space-id-a", "space-a").await;
    seed_mutable_space(&db).await;

    let pass = rereference(&db, "space-id-a", "space-a", 7).await;
    let reference = pass.decay_reference;

    let tx = db.tx().await;
    let group = group_support(&tx, "space-a", "g0")
        .await
        .expect("group support")
        .expect("g0 is eligible before the mutation");
    tx.commit().await.expect("commit read");

    // The mutation itself: g0's roots stop being eligible. The incremental
    // path is told about it separately, which is exactly the seam under test.
    db.exec(
        "UPDATE provenance_roots SET status = 'retracted'
          WHERE independence_group_id = 'g0'",
        (),
    )
    .await;

    let tx = db.tx().await;
    let receipt = apply_group_mutation(
        &tx,
        "space-a",
        &group,
        reference,
        crate::m6::relevance::EligibilityChange::Retracted,
    )
    .await
    .expect("apply retraction");
    tx.commit().await.expect("commit mutation");

    let incremental = stored_pair_rows(&db, "space-a").await;
    let full = full_recompute_rows(&db, "space-a", reference).await;
    // An empty-vs-empty digest comparison passes for the wrong reason.
    assert_eq!(
        incremental.len(),
        3,
        "three pages leave three pairs standing"
    );
    assert_eq!(full.len(), 3);
    assert!(
        incremental.iter().all(|(_, _, values)| values.n11 > 0.0),
        "every surviving pair must still carry support from g1/g2/g3"
    );
    assert_eq!(
        digest_of(&incremental),
        digest_of(&full),
        "the incremental retraction and a full recomputation disagree\n\
         incremental: {incremental:#?}\nfull: {full:#?}"
    );

    // S0-91's negative control: the mutation touched its own group's pairs,
    // not the table. Three pages form three pairs, well inside R-HUB-MAXPAIRS.
    assert_eq!(receipt.pages_touched, 3);
    assert_eq!(receipt.pairs_touched, 3);
    assert!(
        receipt.pairs_touched <= 2016,
        "R-HUB-MAXPAIRS: a mutation may not exceed one group's pair count"
    );
}

#[tokio::test]
async fn a_reactivation_restores_the_pre_retraction_digest() {
    let db = GenesisDb::new().await;
    db.seed_space("space-id-a", "space-a").await;
    seed_mutable_space(&db).await;

    let pass = rereference(&db, "space-id-a", "space-a", 7).await;
    let reference = pass.decay_reference;
    let before = digest_of(&stored_pair_rows(&db, "space-a").await);

    let tx = db.tx().await;
    let group = group_support(&tx, "space-a", "g0")
        .await
        .expect("group support")
        .expect("g0 eligible");
    tx.commit().await.expect("commit read");

    for change in [
        crate::m6::relevance::EligibilityChange::Retracted,
        crate::m6::relevance::EligibilityChange::Reactivated,
    ] {
        let tx = db.tx().await;
        apply_group_mutation(&tx, "space-a", &group, reference, change)
            .await
            .expect("apply mutation");
        tx.commit().await.expect("commit mutation");
    }

    assert_eq!(
        digest_of(&stored_pair_rows(&db, "space-a").await),
        before,
        "retract-then-reactivate is not the identity on the pair table"
    );
}
