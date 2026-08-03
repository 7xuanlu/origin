// SPDX-License-Identifier: Apache-2.0
//! Amendment teeth for M6 relevance D1 and D6.
//!
//! These controls live under `src/` so the workspace library-test floor runs
//! them; an arbitrary integration test would not be guaranteed in L3/L4.

use super::relevance::{
    apply_group_eligibility_change, common_neighbor_counts, ensure_relevance_tables,
    normalized_pair_stats_digest, qualified_co_citation, EligibilityChange, PairStatsSnapshotRow,
    PairStatsValues,
};

async fn relevance_db() -> (libsql::Database, libsql::Connection) {
    let database = libsql::Builder::new_local(":memory:")
        .build()
        .await
        .expect("build in-memory relevance database");
    let connection = database.connect().expect("connect relevance database");
    let transaction = connection
        .transaction_with_behavior(libsql::TransactionBehavior::Immediate)
        .await
        .expect("begin relevance schema transaction");
    ensure_relevance_tables(&transaction)
        .await
        .expect("install relevance schema");
    transaction.commit().await.expect("commit relevance schema");
    (database, connection)
}

async fn table_columns(connection: &libsql::Connection, table: &str) -> Vec<(String, i64)> {
    let mut rows = connection
        .query(&format!("PRAGMA table_info({table})"), ())
        .await
        .expect("inspect table columns");
    let mut columns = Vec::new();
    while let Some(row) = rows.next().await.expect("read table column") {
        columns.push((
            row.get::<String>(1).expect("column name"),
            row.get::<i64>(3).expect("not-null bit"),
        ));
    }
    columns
}

#[tokio::test]
async fn d1_schema_stores_the_raw_current_group_count_and_not_floor_cleared() {
    let (_database, connection) = relevance_db().await;
    let columns = table_columns(&connection, "m6_pair_stats").await;

    assert!(
        columns
            .iter()
            .any(|(name, not_null)| name == "distinct_group_count" && *not_null == 1),
        "D1 requires a NOT NULL raw count of currently eligible distinct groups"
    );
    assert!(
        !columns.iter().any(|(name, _)| name == "floor_cleared"),
        "floor-cleared is derived at read time, never stored"
    );
}

#[test]
fn d1_three_unequal_weighted_groups_qualify_then_retract_and_reactivate() {
    // Mutation control: a route that guesses the raw floor from decayed n11
    // rejects these three groups because their unequal aged/weighted total is
    // below 3. The raw count is the only field that can answer correctly.
    let mut stats = PairStatsValues {
        n11: 0.5 + 0.25 + 0.125,
        n10: 0.0,
        n01: 0.0,
        n00: 0.0,
        distinct_group_count: 3,
    };
    let mutant_uses_decayed_n11 = stats.n11 >= 3.0;
    assert!(
        !mutant_uses_decayed_n11,
        "control failed: decayed n11 unexpectedly revealed the raw group count"
    );
    assert!(
        qualified_co_citation(&stats),
        "three currently eligible groups must clear the raw floor"
    );

    apply_group_eligibility_change(&mut stats, 0.25, EligibilityChange::Retracted)
        .expect("retract one currently eligible group");
    assert_eq!(stats.distinct_group_count, 2);
    assert_eq!(stats.n11, 0.625);
    assert!(
        !qualified_co_citation(&stats),
        "retraction must immediately remove the group from floor eligibility"
    );

    apply_group_eligibility_change(&mut stats, 0.25, EligibilityChange::Reactivated)
        .expect("reactivate the same group");
    assert_eq!(stats.distinct_group_count, 3);
    assert_eq!(stats.n11, 0.875);
    assert!(
        qualified_co_citation(&stats),
        "reactivation must restore the group to floor eligibility"
    );
}

#[test]
fn d1_normalized_digest_includes_distinct_group_count() {
    let two_groups = [PairStatsSnapshotRow {
        page_a: "page-a",
        page_b: "page-b",
        values: PairStatsValues {
            n11: 0.875,
            n10: 1.25,
            n01: 2.5,
            n00: 10.0,
            distinct_group_count: 2,
        },
    }];
    let three_groups = [PairStatsSnapshotRow {
        values: PairStatsValues {
            distinct_group_count: 3,
            ..two_groups[0].values
        },
        ..two_groups[0]
    }];

    assert_ne!(
        normalized_pair_stats_digest(&two_groups).expect("digest two-group snapshot"),
        normalized_pair_stats_digest(&three_groups).expect("digest three-group snapshot"),
        "removing the D1 raw count from normalization must be observable"
    );
}

/// The marginals are derived from mass, so the pair-level applier must not
/// touch them.
///
/// This replaces an earlier test that retracted and reactivated the
/// `LeftOnly`/`RightOnly`/`Neither` cells and asserted the raw count did not
/// move. Those cells no longer exist — the applier cannot reach a marginal, so
/// that assertion is now structural. The hazard that remains is a future edit
/// re-introducing a marginal write here while `pair_stats_snapshot` keeps
/// deriving the same quantity from `m6_page_mass`, which would count one
/// transition twice. That is what this asserts.
#[test]
fn d1_the_pair_applier_leaves_the_derived_marginals_alone() {
    let mut stats = PairStatsValues {
        n11: 1.0,
        n10: 2.0,
        n01: 3.0,
        n00: 4.0,
        distinct_group_count: 1,
    };

    apply_group_eligibility_change(&mut stats, 0.5, EligibilityChange::Reactivated)
        .expect("reactivate a co-supporting group");
    apply_group_eligibility_change(&mut stats, 0.5, EligibilityChange::Retracted)
        .expect("retract it again");

    assert_eq!(stats.distinct_group_count, 1);
    assert_eq!(stats.n11, 1.0);
    assert_eq!(
        (stats.n10, stats.n01, stats.n00),
        (2.0, 3.0, 4.0),
        "n10/n01/n00 are derived from m6_page_mass; the pair applier must not \
         write them, or the transition is counted twice"
    );
}

#[tokio::test]
async fn d6_database_rejects_duplicate_neighbors_at_different_ranks() {
    let (_database, connection) = relevance_db().await;
    connection
        .execute(
            "INSERT INTO m6_adjacency
                 (space, endpoint_kind, endpoint_id, neighbor_id, rank)
             VALUES ('space-1', 'page', 'page-a', 'neighbor-1', 1)",
            (),
        )
        .await
        .expect("insert first adjacency slot");

    let duplicate_neighbor = connection
        .execute(
            "INSERT INTO m6_adjacency
                 (space, endpoint_kind, endpoint_id, neighbor_id, rank)
             VALUES ('space-1', 'page', 'page-a', 'neighbor-1', 2)",
            (),
        )
        .await;
    assert!(
        duplicate_neighbor.is_err(),
        "D6 requires database-enforced neighbor uniqueness, not a producer convention"
    );

    let duplicate_rank = connection
        .execute(
            "INSERT INTO m6_adjacency
                 (space, endpoint_kind, endpoint_id, neighbor_id, rank)
             VALUES ('space-1', 'page', 'page-a', 'neighbor-2', 1)",
            (),
        )
        .await;
    assert!(
        duplicate_rank.is_err(),
        "D6 retains rank as the deterministic primary-key slot"
    );

    let out_of_bounds_rank = connection
        .execute(
            "INSERT INTO m6_adjacency
                 (space, endpoint_kind, endpoint_id, neighbor_id, rank)
             VALUES ('space-1', 'page', 'page-b', 'neighbor-3', 65)",
            (),
        )
        .await;
    assert!(
        out_of_bounds_rank.is_err(),
        "the retained rank slot must enforce the frozen top-64 bound"
    );
}

#[tokio::test]
async fn d6_query_c_filters_and_groups_the_intended_endpoint_kind() {
    let (_database, connection) = relevance_db().await;
    for (kind, endpoint, neighbor, rank) in [
        ("page", "shared-id", "neighbor-page", 1_i64),
        ("entity", "shared-id", "neighbor-entity", 1_i64),
        ("page", "page-only", "neighbor-page", 1_i64),
    ] {
        connection
            .execute(
                "INSERT INTO m6_adjacency
                     (space, endpoint_kind, endpoint_id, neighbor_id, rank)
                 VALUES ('space-1', ?1, ?2, ?3, ?4)",
                libsql::params![kind, endpoint, neighbor, rank],
            )
            .await
            .expect("seed cross-kind adjacency");
    }

    let candidates = ["shared-id", "page-only"];
    let source_neighbors = ["neighbor-page", "neighbor-entity"];
    let page_counts = common_neighbor_counts(
        &connection,
        "space-1",
        "page",
        &candidates,
        &source_neighbors,
    )
    .await
    .expect("aggregate page common-neighbor counts");

    assert_eq!(
        page_counts,
        vec![("page-only".to_string(), 1), ("shared-id".to_string(), 1)],
        "the entity row sharing endpoint_id must not inflate the page count"
    );

    let entity_counts = common_neighbor_counts(
        &connection,
        "space-1",
        "entity",
        &["shared-id"],
        &source_neighbors,
    )
    .await
    .expect("aggregate entity common-neighbor counts");
    assert_eq!(entity_counts, vec![("shared-id".to_string(), 1)]);
}
