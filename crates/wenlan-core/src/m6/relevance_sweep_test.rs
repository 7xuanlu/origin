// SPDX-License-Identifier: Apache-2.0
//! Teeth for the C1 decay reference.
//!
//! These live under `src/` so the workspace library-test floor runs them; an
//! integration test would not be guaranteed in L3/L4.

use super::relevance_sweep::{
    advance_decay_reference, decay_reference, decayed_contribution, eligible_groups, group_support,
    hub_weight, recompute_pair_stats, GroupSupport, COUNTER_RELEVANCE_DECAY_REFERENCE,
    DECAY_HALF_LIFE_DAYS,
};
use crate::m6::genesis_test_support::GenesisDb;
use crate::m6::relevance::{qualified_co_citation, MAX_NEIGHBORS_PER_ENDPOINT};

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

    let stats = recompute_pair_stats(&groups, reference);
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
    let stats = recompute_pair_stats(&groups, reference);
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
        recompute_pair_stats(&three, reference).get(&key).unwrap()
    ));

    // Drop one group — the same pair, now below the floor.
    let two = &three[..2];
    let below = recompute_pair_stats(two, reference);
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

    let stats = recompute_pair_stats(std::slice::from_ref(&hub), reference);
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
    let fresh = recompute_pair_stats(&[group("g", &["page-a", "page-b"], reference)], reference);
    let aged = recompute_pair_stats(
        &[group(
            "g",
            &["page-a", "page-b"],
            reference - (DECAY_HALF_LIFE_DAYS as i64) * SECONDS_PER_DAY,
        )],
        reference,
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
