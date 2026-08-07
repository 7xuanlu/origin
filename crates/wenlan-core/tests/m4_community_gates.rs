// SPDX-License-Identifier: Apache-2.0

#[cfg(target_os = "macos")]
use std::process::Command;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::{Duration, Instant},
};
#[cfg(not(target_os = "macos"))]
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Barrier,
    },
    thread,
};

use wenlan_core::community_grouping::{
    compute_community_grouping, run_community_grouping_cycle, CommunityGroupingComputeMode,
    CommunityGroupingError, CommunityGroupingFullReason, CommunityGroupingOutcome,
    CommunityGroupingRuntime,
};
use wenlan_core::community_partition::{
    disconnected_community_count, full_partition, incremental_partition,
    label_propagation_partition, modularity, project_grounded_relates, rebind_durable_ids,
    rebind_durable_ids_weighted, IncrementalConfig, IncrementalPartitionError,
    IncrementalPartitionState, PartitionConfig, ProjectedEdge, ProjectedGraph, ProjectionConfig,
    ProjectionInputEdge,
};
use wenlan_core::db::MemoryDB;
use wenlan_core::edge_grounding::EdgePromotion;
use wenlan_core::events::NoopEmitter;
use wenlan_core::provenance::{compute_edge_id, IndependenceSignals};
use wenlan_core::sources::RawDocument;

#[test]
fn m4_projection_is_sorted_folded_and_parallel_capped() {
    let edges = vec![
        ProjectionInputEdge::new("edge-3", "entity-b", "entity-a"),
        ProjectionInputEdge::new("edge-1", "entity-a", "entity-b"),
        ProjectionInputEdge::new("edge-4", "entity-b", "entity-a"),
        ProjectionInputEdge::new("edge-2", "entity-a", "entity-c"),
        ProjectionInputEdge::new("edge-5", "entity-a", "entity-b"),
    ];

    let graph = project_grounded_relates(&edges, ProjectionConfig::default());

    assert_eq!(
        graph.node_ids(),
        &["entity-a", "entity-b", "entity-c"],
        "node indices must be stable across identical runs"
    );
    assert_eq!(
        graph.edges(),
        &[ProjectedEdge::new(0, 1, 3.0), ProjectedEdge::new(0, 2, 1.0),],
        "direction-folded parallel weight must stop at 3x W_relates"
    );
}

#[test]
fn m4_projection_soft_normalizes_hubs_without_dropping_edges() {
    let edges = vec![
        ProjectionInputEdge::new("edge-1", "hub", "entity-a"),
        ProjectionInputEdge::new("edge-2", "hub", "entity-b"),
        ProjectionInputEdge::new("edge-3", "hub", "entity-c"),
    ];
    let config = ProjectionConfig {
        hub_degree_cap: 2,
        ..ProjectionConfig::default()
    };

    let graph = project_grounded_relates(&edges, config);

    assert_eq!(graph.edges().len(), 3, "hub edges stay present");
    for edge in graph.edges() {
        assert!(
            (edge.weight - (2.0 / 3.0)).abs() < f64::EPSILON,
            "each incident edge is softly down-weighted by cap/degree: {edge:?}"
        );
    }
}

#[test]
fn m4_leiden_full_partition_is_seeded_and_byte_deterministic() {
    let graph = project_grounded_relates(&large_space_edges(0), ProjectionConfig::default());

    let first = full_partition(&graph, PartitionConfig::default()).expect("first Leiden run");
    for run in 1..10 {
        let repeated =
            full_partition(&graph, PartitionConfig::default()).expect("identical Leiden rerun");
        assert_eq!(
            first.membership(),
            repeated.membership(),
            "membership differs at seeded rerun {run}"
        );
        assert_eq!(
            first.modularity().to_bits(),
            repeated.modularity().to_bits(),
            "modularity differs at seeded rerun {run}"
        );
    }
}

#[test]
fn m4_leiden_quality_meets_label_prop_and_every_community_is_connected() {
    let edges = vec![
        ProjectionInputEdge::new("edge-01", "a", "b"),
        ProjectionInputEdge::new("edge-02", "a", "c"),
        ProjectionInputEdge::new("edge-03", "b", "c"),
        ProjectionInputEdge::new("edge-04", "c", "d"),
        ProjectionInputEdge::new("edge-05", "d", "e"),
        ProjectionInputEdge::new("edge-06", "d", "f"),
        ProjectionInputEdge::new("edge-07", "e", "f"),
    ];
    let graph = project_grounded_relates(&edges, ProjectionConfig::default());

    let leiden = full_partition(&graph, PartitionConfig::default()).expect("Leiden partition");
    let label_prop = label_propagation_partition(&graph);

    assert!(
        (leiden.modularity() - modularity(&graph, leiden.membership())).abs() <= 1e-12,
        "crate-reported modularity must match the local differential oracle"
    );
    assert!(
        leiden.modularity() + 1e-12 >= label_prop.modularity(),
        "Leiden {:.6} must meet label-prop {:.6}",
        leiden.modularity(),
        label_prop.modularity()
    );
    assert_eq!(
        disconnected_community_count(&graph, leiden.membership()),
        0,
        "Leiden's published communities must be connected"
    );
}

#[test]
fn m4_incremental_partition_is_frontier_bounded_and_never_degrades_below_full_modularity() {
    let mut edges = planted_cluster_edges(4, 8);
    let base = project_grounded_relates(&edges, ProjectionConfig::default());
    let previous =
        full_partition(&base, PartitionConfig::default()).expect("baseline Leiden partition");

    edges.push(ProjectionInputEdge::new(
        "edge-extra",
        "cluster-0-node-0",
        "cluster-0-node-1",
    ));
    let changed = project_grounded_relates(&edges, ProjectionConfig::default());
    assert_eq!(base.node_ids(), changed.node_ids(), "stable node index");
    let dirty = [0usize, 1usize];
    let carry_forward_q = modularity(&changed, previous.membership());
    let state =
        IncrementalPartitionState::new(&base, previous.membership()).expect("incremental state");

    let incremental = incremental_partition(
        &changed,
        state,
        &dirty,
        IncrementalConfig {
            max_frontier_fraction: 0.50,
            ..IncrementalConfig::default()
        },
    )
    .expect("frontier optimization");
    let fresh = full_partition(&changed, PartitionConfig::default()).expect("fresh full partition");
    let incremental_q = modularity(&changed, incremental.partition().membership());
    let fresh_q = modularity(&changed, fresh.membership());
    assert!(
        (fresh.modularity() - fresh_q).abs() <= 1e-12,
        "fresh crate-reported modularity must match the local differential oracle"
    );

    for node in 0..changed.node_ids().len() {
        if !incremental.optimized_nodes().contains(&node) {
            assert_eq!(
                incremental.partition().membership()[node],
                previous.membership()[node],
                "outside-frontier node {node} must be carried forward unchanged"
            );
        }
    }
    let signed_delta = incremental_q - carry_forward_q;
    assert!(
        signed_delta >= -1e-9,
        "incremental {incremental_q:.12} must not degrade below carried prior \
         {carry_forward_q:.12} (signed delta {signed_delta:.3e}, floor -1e-9)"
    );
    assert_eq!(
        disconnected_community_count(&changed, incremental.partition().membership()),
        0,
        "incremental communities must stay connected"
    );
}

#[test]
fn m4_incremental_rejects_an_expanded_frontier_above_the_full_repartition_threshold() {
    let mut edges = (1..=100)
        .map(|node| {
            ProjectionInputEdge::new(format!("edge-{node:03}"), "hub", format!("leaf-{node:03}"))
        })
        .collect::<Vec<_>>();
    let base = project_grounded_relates(&edges, ProjectionConfig::default());
    let previous = vec![0; base.node_ids().len()];
    edges.push(ProjectionInputEdge::new("edge-new", "hub", "leaf-001"));
    let changed = project_grounded_relates(&edges, ProjectionConfig::default());
    let dirty_hub = changed
        .node_ids()
        .binary_search(&"hub".to_owned())
        .expect("hub index");
    let config = IncrementalConfig {
        max_dirty_fraction: 0.01,
        ..IncrementalConfig::default()
    };

    let state = IncrementalPartitionState::new(&base, &previous).expect("incremental state");
    let result = incremental_partition(&changed, state, &[dirty_hub], config);

    assert!(
        result.is_err(),
        "a 1/101 dirty set expands to the whole graph and must route to full repartition"
    );
}

#[test]
fn m4_stateful_incremental_statistics_match_full_oracle_across_add_and_retract() {
    let base_edges = planted_cluster_edges(4, 8);
    let base = project_grounded_relates(&base_edges, ProjectionConfig::default());
    let initial =
        full_partition(&base, PartitionConfig::default()).expect("baseline Leiden partition");
    let mut state =
        IncrementalPartitionState::new(&base, initial.membership()).expect("incremental state");

    let mut changed_edges = base_edges.clone();
    changed_edges.push(ProjectionInputEdge::new(
        "cross-cluster-change",
        "cluster-0-node-1",
        "cluster-1-node-1",
    ));
    let changed = project_grounded_relates(&changed_edges, ProjectionConfig::default());
    let dirty = [
        changed
            .node_ids()
            .binary_search(&"cluster-0-node-1".to_owned())
            .expect("src"),
        changed
            .node_ids()
            .binary_search(&"cluster-1-node-1".to_owned())
            .expect("dst"),
    ];
    let added = incremental_partition(
        &changed,
        state,
        &dirty,
        IncrementalConfig {
            max_frontier_fraction: 0.75,
            ..IncrementalConfig::default()
        },
    )
    .expect("edge addition");
    assert!(
        (added.partition().modularity() - modularity(&changed, added.partition().membership()))
            .abs()
            <= 1e-12,
        "edge-add delta statistics must match the full oracle"
    );
    state = added.into_state();

    let retracted = incremental_partition(
        &base,
        state,
        &dirty,
        IncrementalConfig {
            max_frontier_fraction: 0.75,
            ..IncrementalConfig::default()
        },
    )
    .expect("edge retraction");
    assert!(
        (retracted.partition().modularity()
            - modularity(&base, retracted.partition().membership()))
        .abs()
            <= 1e-12,
        "edge-retraction delta statistics must match the full oracle"
    );
}

#[test]
fn m4_bridge_retraction_cannot_publish_a_disconnected_prior_community() {
    let base_edges = vec![
        ProjectionInputEdge::new("left-1", "a", "b"),
        ProjectionInputEdge::new("left-2", "b", "c"),
        ProjectionInputEdge::new("left-3", "a", "c"),
        ProjectionInputEdge::new("bridge", "c", "d"),
        ProjectionInputEdge::new("right-1", "d", "e"),
        ProjectionInputEdge::new("right-2", "e", "f"),
        ProjectionInputEdge::new("right-3", "d", "f"),
    ];
    let base = project_grounded_relates(&base_edges, ProjectionConfig::default());
    let membership = vec![0; base.node_ids().len()];
    assert_eq!(
        disconnected_community_count(&base, &membership),
        0,
        "precondition: the bridge makes the prior community connected"
    );
    let state = IncrementalPartitionState::new(&base, &membership).expect("incremental state");
    let changed = project_grounded_relates(
        &base_edges
            .into_iter()
            .filter(|edge| edge.edge_id != "bridge")
            .collect::<Vec<_>>(),
        ProjectionConfig::default(),
    );
    let dirty = [
        changed
            .node_ids()
            .binary_search(&"c".to_owned())
            .expect("c"),
        changed
            .node_ids()
            .binary_search(&"d".to_owned())
            .expect("d"),
    ];

    let result = incremental_partition(
        &changed,
        state,
        &dirty,
        IncrementalConfig {
            max_dirty_fraction: 0.75,
            max_frontier_fraction: 1.0,
            ..IncrementalConfig::default()
        },
    );

    assert_eq!(
        result,
        Err(IncrementalPartitionError::DisconnectedCommunity),
        "a bridge retraction must route to full repartition, never publish a disconnected community"
    );
}

#[test]
fn m4_rebinding_uses_member_overlap_not_partitioner_labels() {
    let previous = vec![
        "community-a".to_owned(),
        "community-a".to_owned(),
        "community-a".to_owned(),
        "community-b".to_owned(),
        "community-b".to_owned(),
        "community-b".to_owned(),
    ];

    let label_only_change = rebind_durable_ids(&previous, &[99, 99, 99, 7, 7, 7]);
    assert_eq!(label_only_change, previous);

    let one_node_moves = rebind_durable_ids(&previous, &[99, 99, 7, 7, 7, 7]);
    assert_eq!(
        one_node_moves,
        vec![
            "community-a",
            "community-a",
            "community-b",
            "community-b",
            "community-b",
            "community-b",
        ]
    );
}

#[test]
fn m4_rebinding_fractionally_weights_boundary_nodes_by_grounded_adjacency() {
    let graph = project_grounded_relates(
        &[
            ProjectionInputEdge::new("a-c-1", "a", "c"),
            ProjectionInputEdge::new("a-c-2", "a", "c"),
            ProjectionInputEdge::new("a-c-3", "a", "c"),
            ProjectionInputEdge::new("b-c-1", "b", "c"),
            ProjectionInputEdge::new("b-c-2", "b", "c"),
            ProjectionInputEdge::new("b-c-3", "b", "c"),
        ],
        ProjectionConfig::default(),
    );
    assert_eq!(graph.node_ids(), &["a", "b", "c"]);

    // A raw member count would choose old-a (two nodes versus one). Weighted
    // multi-membership instead lets a and b each contribute through their
    // grounded boundary to old-b, while c contributes through its two
    // old-a neighbors. That makes old-b the stable max-overlap identity.
    let previous = vec!["old-a".to_owned(), "old-a".to_owned(), "old-b".to_owned()];
    let rebound = rebind_durable_ids_weighted(&previous, &[7, 7, 7], &graph);

    assert_eq!(
        rebound,
        vec!["old-b".to_owned(), "old-b".to_owned(), "old-b".to_owned()]
    );
}

#[test]
fn m4_rebinding_never_lets_new_node_sentinels_steal_a_durable_identity() {
    let mut edges = vec![ProjectionInputEdge::new("old-hub", "old", "hub")];
    for leaf in 0..8 {
        edges.push(ProjectionInputEdge::new(
            format!("hub-leaf-{leaf}"),
            "hub",
            format!("leaf-{leaf}"),
        ));
    }
    let graph = project_grounded_relates(&edges, ProjectionConfig::default());
    let previous = graph
        .node_ids()
        .iter()
        .map(|node_id| {
            if node_id == "old" {
                "durable-old".to_owned()
            } else {
                format!("__m4-new-node-{node_id}")
            }
        })
        .collect::<Vec<_>>();
    let rebound = rebind_durable_ids_weighted(&previous, &vec![7; previous.len()], &graph);

    assert!(
        rebound.iter().all(|community_id| community_id == "durable-old"),
        "synthetic new-node markers are not prior communities and must never outvote a real durable id: {rebound:?}"
    );
}

#[test]
fn m4_rebinding_new_only_split_cannot_claim_an_old_groups_durable_id() {
    let mut edges = Vec::new();
    for leaf in 0..8 {
        edges.push(ProjectionInputEdge::new(
            format!("old-leaf-{leaf}"),
            "old",
            format!("new-{leaf}"),
        ));
    }
    let graph = project_grounded_relates(&edges, ProjectionConfig::default());
    let previous = graph
        .node_ids()
        .iter()
        .map(|node_id| {
            if node_id == "old" {
                "durable-old".to_owned()
            } else {
                format!("__m4-new-node-{node_id}")
            }
        })
        .collect::<Vec<_>>();
    let membership = graph
        .node_ids()
        .iter()
        .map(|node_id| usize::from(node_id != "old"))
        .collect::<Vec<_>>();
    let rebound = rebind_durable_ids_weighted(&previous, &membership, &graph);
    let old_index = graph
        .node_ids()
        .iter()
        .position(|node_id| node_id == "old")
        .expect("old member index");

    assert_eq!(
        rebound[old_index], "durable-old",
        "only the group containing an old member may inherit that member's durable identity"
    );
    assert!(
        rebound
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != old_index)
            .all(|(_, community_id)| community_id != "durable-old"),
        "a new-only split must mint instead of stealing the old group's identity: {rebound:?}"
    );
}

#[test]
#[ignore = "manual M4 Gate 1.2 multi-size locality receipt"]
fn m4_incremental_cost_tracks_frontier_size_not_total_graph_size() {
    const RUNS: usize = 21;
    let small = project_grounded_relates(&ring_edges(2_048), ProjectionConfig::default());
    let large = project_grounded_relates(&ring_edges(32_768), ProjectionConfig::default());
    let small_membership = vec![0; small.node_ids().len()];
    let large_membership = vec![0; large.node_ids().len()];

    let small_p95 = incremental_p95(
        &small,
        &small_membership,
        &[0],
        IncrementalConfig::default(),
        RUNS,
    );
    let large_p95 = incremental_p95(
        &large,
        &large_membership,
        &[0],
        IncrementalConfig::default(),
        RUNS,
    );
    assert!(
        large_p95.as_nanos() <= small_p95.as_nanos() * 3,
        "fixed one-node frontier grew with total graph size: small={small_p95:?}, large={large_p95:?}"
    );

    let narrow = (0..8).collect::<Vec<_>>();
    let wide = (0..64).collect::<Vec<_>>();
    let narrow_p95 = incremental_p95(
        &large,
        &large_membership,
        &narrow,
        IncrementalConfig::default(),
        RUNS,
    );
    let wide_p95 = incremental_p95(
        &large,
        &large_membership,
        &wide,
        IncrementalConfig::default(),
        RUNS,
    );
    assert!(
        wide_p95.as_nanos() <= narrow_p95.as_nanos() * 12,
        "8x wider frontier scaled superlinearly: narrow={narrow_p95:?}, wide={wide_p95:?}"
    );
    println!(
        "[m4_incremental_scaling] fixed_frontier_small_p95_us={:.3} \
         fixed_frontier_large_p95_us={:.3} narrow_frontier_p95_us={:.3} \
         wide_frontier_p95_us={:.3}",
        small_p95.as_secs_f64() * 1_000_000.0,
        large_p95.as_secs_f64() * 1_000_000.0,
        narrow_p95.as_secs_f64() * 1_000_000.0,
        wide_p95.as_secs_f64() * 1_000_000.0,
    );
}

#[test]
#[ignore = "fresh child used only by the M4 release receipt"]
fn m4_partition_rss_baseline_child() {
    assert_eq!(
        std::env::var("WENLAN_M4_RSS_CHILD").as_deref(),
        Ok("baseline"),
        "run through the parent receipt"
    );
    let graph = project_grounded_relates(&large_space_edges(0), ProjectionConfig::default());
    std::hint::black_box(graph);
}

#[test]
#[ignore = "fresh child used only by the M4 release receipt"]
fn m4_partition_rss_partition_child() {
    assert_eq!(
        std::env::var("WENLAN_M4_RSS_CHILD").as_deref(),
        Ok("partition"),
        "run through the parent receipt"
    );
    let graph = project_grounded_relates(&large_space_edges(0), ProjectionConfig::default());
    for _ in 0..5 {
        std::hint::black_box(
            full_partition(&graph, PartitionConfig::default()).expect("RSS child partition"),
        );
    }
}

#[test]
#[ignore = "manual M4 Gate 1 + Gate 2.1-2.3 release-mode receipt"]
fn m4_partition_scale_churn_and_poison_receipt() {
    const SPACE_COUNT: usize = 3;
    const RUNS_PER_SPACE: usize = 5;
    const CHURN_CYCLES: usize = 50;

    let spaces = (0..SPACE_COUNT).map(large_space_edges).collect::<Vec<_>>();
    let total_edges = spaces.iter().map(Vec::len).sum::<usize>();
    let total_participants = spaces
        .iter()
        .map(|edges| {
            project_grounded_relates(edges, ProjectionConfig::default())
                .node_ids()
                .len()
        })
        .sum::<usize>();
    assert!(total_edges >= 5_000, "Gate 1.0 edge floor");
    assert!(total_participants >= 2_000, "Gate 1.0 participant floor");
    assert!(spaces.len() >= 3, "Gate 1.0 space floor");

    #[cfg(target_os = "macos")]
    let peak_rss_delta = fresh_child_partition_peak_rss_delta();
    #[cfg(not(target_os = "macos"))]
    let peak_rss_delta = {
        let graph = project_grounded_relates(&spaces[0], ProjectionConfig::default());
        peak_rss_delta_for_full_partition(&graph, RUNS_PER_SPACE)
    };
    assert!(
        peak_rss_delta <= 256 * 1024 * 1024,
        "Gate 1.3 peak additional RSS {peak_rss_delta} exceeds 256 MiB"
    );
    assert!(
        peak_rss_delta <= 1024 * 1024 * 1024,
        "Gate 1.3 hard fail: peak additional RSS {peak_rss_delta} exceeds 1 GiB"
    );

    let mut full_p95 = Vec::new();
    let mut clean_community_count = 0usize;
    for (space, edges) in spaces.iter().enumerate() {
        let graph = project_grounded_relates(edges, ProjectionConfig::default());
        let label_prop = label_propagation_partition(&graph);
        let mut elapsed = Vec::with_capacity(RUNS_PER_SPACE);
        let mut reference_membership = None;
        let mut reference_durable = None::<Vec<String>>;
        for run in 0..RUNS_PER_SPACE {
            let started = Instant::now();
            let partition =
                full_partition(&graph, PartitionConfig::default()).expect("full Leiden partition");
            elapsed.push(started.elapsed());
            assert_eq!(
                disconnected_community_count(&graph, partition.membership()),
                0,
                "Gate 1.5 connectedness"
            );
            assert!(
                (partition.modularity() - modularity(&graph, partition.membership())).abs() <= 1e-9,
                "Gate 1.5 local modularity differential oracle"
            );
            assert!(
                partition.modularity() + 1e-9 >= label_prop.modularity(),
                "Gate 1.5 Leiden modularity {:.9} must meet label-prop {:.9}",
                partition.modularity(),
                label_prop.modularity()
            );
            if let Some(reference) = &reference_membership {
                assert_eq!(
                    partition.membership(),
                    reference,
                    "Gate 2.1 byte determinism, space {space}, run {run}"
                );
            } else {
                reference_membership = Some(partition.membership().to_vec());
            }
            if let Some(reference) = &reference_durable {
                let rebound = rebind_durable_ids(reference, partition.membership());
                assert_eq!(
                    &rebound, reference,
                    "Gate 2.1 durable community_members differ, space {space}, run {run}"
                );
            } else {
                reference_durable = Some(
                    partition
                        .membership()
                        .iter()
                        .map(|community| format!("durable-space-{space}-{community}"))
                        .collect::<Vec<_>>(),
                );
            }
            if space == 0 {
                clean_community_count = community_count(partition.membership());
            }
        }
        let p95 = duration_p95(&elapsed);
        assert!(
            p95 <= Duration::from_secs(10),
            "Gate 1.1 p95 {:?} exceeds 10s in space {space}",
            p95
        );
        assert!(
            elapsed
                .iter()
                .all(|duration| *duration <= Duration::from_secs(30)),
            "Gate 1.1 hard-fail band in space {space}: {elapsed:?}"
        );
        full_p95.push(p95);
    }

    let clean = incremental_churn_series(spaces[0].clone(), "clean", CHURN_CYCLES);
    assert!(
        clean.max_frontier_fraction <= 0.01,
        "Gate 1.2 clean optimized frontier exceeds 1%: {:.4}%",
        clean.max_frontier_fraction * 100.0
    );
    let clean_warm_p95 = duration_p95(&clean.durations);
    let full_reference = full_p95[0];
    assert!(
        clean_warm_p95.as_nanos() * 10 <= full_reference.as_nanos(),
        "Gate 1.2 warm p95 {:?} is {:.2}% of full p95 {:?}",
        clean_warm_p95,
        duration_ratio_percent(clean_warm_p95, full_reference),
        full_reference
    );
    assert!(
        clean_warm_p95.as_nanos() * 2 < full_reference.as_nanos(),
        "Gate 1.2 hard fail: warm p95 {:?} is not below 50% of full p95 {:?}",
        clean_warm_p95,
        full_reference
    );
    let clean_mean_churn = mean(&clean.churn);
    let clean_p95_churn = f64_p95(&clean.churn);
    assert!(
        clean_mean_churn <= 0.02,
        "Gate 2.2 mean churn {:.4}% exceeds 2%",
        clean_mean_churn * 100.0
    );
    assert!(
        clean_p95_churn <= 0.10,
        "Gate 2.2 p95 churn {:.4}% exceeds 10%",
        clean_p95_churn * 100.0
    );

    let poisoned_edges = add_five_percent_poison(spaces[0].clone());
    let poison_graph = project_grounded_relates(&poisoned_edges, ProjectionConfig::default());
    let poison_partition =
        full_partition(&poison_graph, PartitionConfig::default()).expect("poison partition");
    let poison_community_count = community_count(poison_partition.membership());
    assert!(
        poison_community_count * 100 >= clean_community_count * 70,
        "Gate 2.3 community count collapsed: clean={clean_community_count}, poison={poison_community_count}"
    );
    let poison = incremental_churn_series(poisoned_edges, "poison", CHURN_CYCLES);
    let poison_mean_churn = mean(&poison.churn);
    assert!(
        poison_mean_churn <= 0.04,
        "Gate 2.3 poison mean churn {:.4}% exceeds 4%",
        poison_mean_churn * 100.0
    );

    println!(
        "[m4_gate_receipt] edges={total_edges} participants={total_participants} spaces={} \
         full_p95_ms={:?} warm_p95_ms={:.3} warm_ratio_pct={:.3} peak_rss_delta_bytes={} \
         clean_churn_mean_pct={:.4} clean_churn_p95_pct={:.4} clean_min_modularity_delta={:.3e} \
         clean_communities={} poison_fraction_pct={:.3} poison_churn_mean_pct={:.4} \
         clean_max_frontier_pct={:.3} poison_min_modularity_delta={:.3e} \
         poison_max_frontier_pct={:.3} poison_communities={}",
        spaces.len(),
        full_p95
            .iter()
            .map(|duration| duration.as_secs_f64() * 1_000.0)
            .collect::<Vec<_>>(),
        clean_warm_p95.as_secs_f64() * 1_000.0,
        duration_ratio_percent(clean_warm_p95, full_reference),
        peak_rss_delta,
        clean_mean_churn * 100.0,
        clean_p95_churn * 100.0,
        clean.min_modularity_delta,
        clean_community_count,
        poison.poison_fraction * 100.0,
        poison_mean_churn * 100.0,
        clean.max_frontier_fraction * 100.0,
        poison.min_modularity_delta,
        poison.max_frontier_fraction * 100.0,
        poison_community_count,
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn m4_real_job_mutex_and_correction_latency_gate() {
    const SPACE: &str = "space-0";
    const SECOND_SPACE: &str = "space-1";
    const SOURCE_ID: &str = "m4-real-job-source";
    const SOURCE_CONTENT: &str =
        "The M4 acceptance graph records grounded relationships for the community job.";
    const BASE_EDGES_PER_SPACE: usize = 6_151;
    const BASE_GRAPH_GENERATION: i64 = BASE_EDGES_PER_SPACE as i64;
    const PARTICIPANTS_PER_SPACE: usize = 2_048;

    let dir = tempfile::tempdir().expect("M4 real-job tempdir");
    let db = Arc::new(
        MemoryDB::new(dir.path(), Arc::new(NoopEmitter))
            .await
            .expect("open M4 real-job database"),
    );
    db.upsert_documents(vec![RawDocument {
        source: "memory".to_owned(),
        source_id: SOURCE_ID.to_owned(),
        title: "M4 real grouping source".to_owned(),
        content: SOURCE_CONTENT.to_owned(),
        last_modified: 1_722_000_000,
        memory_type: Some("fact".to_owned()),
        space: Some(SPACE.to_owned()),
        source_agent: Some("folder".to_owned()),
        confirmed: Some(true),
        ..Default::default()
    }])
    .await
    .expect("seed canonical folder memory");

    let observer_db = libsql::Builder::new_local(dir.path().join("origin_memory.db"))
        .build()
        .await
        .expect("open independent M4 observer");
    let observer = observer_db.connect().expect("connect M4 observer");
    assert_m4_schema(&observer).await;
    seed_m4_graded_corpus(&observer).await;

    let root_id = db
        .acquire_provenance_root(
            "document_ingest",
            SOURCE_CONTENT,
            &IndependenceSignals {
                source_identity: Some("file:///m4-real-job.md"),
                agent_turn: None,
                import_batch: None,
            },
        )
        .await
        .expect("acquire M4 provenance root");

    let chords = m4_scale_chord_pairs(23);
    let initial_promotion = stage_m4_promotion(
        &db,
        &chords[0].0,
        &chords[0].1,
        SOURCE_ID,
        SOURCE_CONTENT,
        &root_id,
    )
    .await;
    assert_eq!(
        db.promote_edges_grounded(&[initial_promotion])
            .await
            .expect("promote initial M4 correction"),
        1,
        "the real M3g promotion path must bump the main-space generation"
    );

    let _ungrounded_negative_control = stage_m4_promotion(
        &db,
        &chords[1].0,
        &chords[1].1,
        SOURCE_ID,
        SOURCE_CONTENT,
        &root_id,
    )
    .await;
    let second_space_promotion = stage_m4_promotion(
        &db,
        &m4_scale_node(1, 0),
        &m4_scale_node(1, 17),
        SOURCE_ID,
        SOURCE_CONTENT,
        &root_id,
    )
    .await;
    assert_eq!(
        db.promote_edges_grounded(&[second_space_promotion])
            .await
            .expect("promote second-space negative control"),
        1
    );
    let second_space_state = read_m4_persisted_snapshot(&observer, SECOND_SPACE).await;
    assert_eq!(
        second_space_state.graph_generation,
        BASE_GRAPH_GENERATION + 1
    );
    assert!(second_space_state.dirty);
    assert!(second_space_state.members.is_empty());

    // G6 Stage 2 PR 2a: `detect_communities` now reads `edges` (where
    // `seed_m4_graded_corpus` seeds this test's corpus, via raw SQL, not
    // through `relations`) unconditionally -- the `set_reader_cutover`
    // flip this test used to need to route the legacy detector onto that
    // corpus is gone with the cutover lever it drove.
    db.detect_communities()
        .await
        .expect("seed the legacy label-prop shadow");
    let legacy_before = read_legacy_community_shadow(&observer, SPACE).await;
    assert!(
        legacy_before
            .iter()
            .all(|(_, community)| community.is_some()),
        "the non-interference oracle needs a populated legacy shadow"
    );

    let initial_state = read_m4_persisted_snapshot(&observer, SPACE).await;
    assert_eq!(initial_state.graph_generation, BASE_GRAPH_GENERATION + 1);
    assert_eq!(initial_state.published_generation, None);
    assert!(initial_state.dirty);
    assert!(initial_state.members.is_empty());
    let first_generation = read_m4_grouping_generation(&observer, SPACE).await;

    let mut runtime = CommunityGroupingRuntime::default();
    let mut mutex_holds = Vec::new();
    let mut foreground_latencies = Vec::new();
    let mut composer_latencies = Vec::new();
    let (first_result, first_foreground_latency) =
        run_m4_cycle_with_foreground_probe(&db, &mut runtime, SPACE).await;
    foreground_latencies.push(first_foreground_latency);
    let first = match first_result.expect("first production grouping cycle") {
        CommunityGroupingOutcome::Published(receipt) => receipt,
        other => panic!("first dirty cycle must publish, got {other:?}"),
    };
    assert_eq!(first.input_generation, first_generation);
    assert_eq!(first.published_generation, first_generation);
    assert_eq!(
        first.compute_mode,
        CommunityGroupingComputeMode::Full {
            reason: CommunityGroupingFullReason::FirstPublication,
        },
        "the first production publication is the positive full-mode control"
    );
    assert_eq!(
        first.projected_edge_count,
        BASE_EDGES_PER_SPACE + 1,
        "the real SELECT must exclude the grounded=0 relation and the promoted second-space edge"
    );
    assert_eq!(first.member_count, PARTICIPANTS_PER_SPACE);
    mutex_holds.push(first.db_mutex_hold);
    composer_latencies.push(first.composer_elapsed);
    let graded_input_rows = first.input_rows_loaded;
    let graded_member_rows = first.member_rows_written;

    let first_snapshot = read_m4_persisted_snapshot(&observer, SPACE).await;
    assert_eq!(first_snapshot.graph_generation, BASE_GRAPH_GENERATION + 1);
    assert_eq!(first_snapshot.published_generation, Some(first_generation));
    assert!(!first_snapshot.dirty);
    assert_eq!(first_snapshot.members.len(), PARTICIPANTS_PER_SPACE);
    assert!(first_snapshot
        .members
        .iter()
        .all(|member| member.published_generation == first_generation));

    let second = match run_community_grouping_cycle(&db, &mut runtime, SECOND_SPACE)
        .await
        .expect("publish second-space runtime control")
    {
        CommunityGroupingOutcome::Published(receipt) => receipt,
        other => panic!("second-space control must publish, got {other:?}"),
    };
    assert_eq!(
        second.compute_mode,
        CommunityGroupingComputeMode::Full {
            reason: CommunityGroupingFullReason::FirstPublication,
        }
    );
    let second_runtime_generation = second.published_generation;

    for (cycle, (from, to)) in chords.iter().skip(2).take(18).enumerate() {
        let promotion =
            stage_m4_promotion(&db, from, to, SOURCE_ID, SOURCE_CONTENT, &root_id).await;
        assert_eq!(
            db.promote_edges_grounded(&[promotion])
                .await
                .expect("promote common-case correction"),
            1
        );
        let (published_result, foreground_latency) =
            run_m4_cycle_with_foreground_probe(&db, &mut runtime, SPACE).await;
        foreground_latencies.push(foreground_latency);
        let published = match published_result.expect("common-case production grouping cycle") {
            CommunityGroupingOutcome::Published(receipt) => receipt,
            other => panic!("common correction cycle {cycle} must publish, got {other:?}"),
        };
        let expected_generation = read_m4_grouping_generation(&observer, SPACE).await;
        assert_eq!(published.input_generation, expected_generation);
        assert_eq!(published.published_generation, expected_generation);
        assert!(
            matches!(
                published.compute_mode,
                CommunityGroupingComputeMode::Incremental { optimized_nodes }
                    if optimized_nodes > 0 && optimized_nodes < PARTICIPANTS_PER_SPACE
            ),
            "one grounded correction must optimize only its bounded frontier: {:?}",
            published.compute_mode
        );
        assert_eq!(
            published.projected_edge_count,
            BASE_EDGES_PER_SPACE + cycle + 2,
            "the published receipt must include that cycle's corrected edge"
        );
        mutex_holds.push(published.db_mutex_hold);
        composer_latencies.push(published.composer_elapsed);

        let persisted = read_m4_persisted_snapshot(&observer, SPACE).await;
        assert_eq!(
            persisted.published_generation,
            Some(expected_generation),
            "common correction must publish within one grouping cycle"
        );
        assert!(!persisted.dirty);
        assert!(persisted
            .members
            .iter()
            .all(|member| member.published_generation == expected_generation));
    }

    let (stale_from, stale_to) = &chords[20];
    let stale_input_promotion = stage_m4_promotion(
        &db,
        stale_from,
        stale_to,
        SOURCE_ID,
        SOURCE_CONTENT,
        &root_id,
    )
    .await;
    assert_eq!(
        db.promote_edges_grounded(&[stale_input_promotion])
            .await
            .expect("promote correction before stale attempt"),
        1
    );
    let before_stale = read_m4_persisted_snapshot(&observer, SPACE).await;
    assert_eq!(before_stale.graph_generation, BASE_GRAPH_GENERATION + 20);
    let stale_input_generation = read_m4_grouping_generation(&observer, SPACE).await;
    let live_generation = before_stale
        .published_generation
        .expect("the preceding correction cycle is published");
    assert!(before_stale.dirty);

    seed_obsolete_m4_leases(&observer, SPACE, stale_input_generation).await;
    let attempt = db
        .prepare_community_grouping(SPACE)
        .await
        .expect("prepare real lease + grounded SELECT");
    assert_eq!(attempt.input_generation(), stale_input_generation);
    assert_eq!(
        read_m4_lease_generations(&observer, SPACE).await,
        vec![stale_input_generation],
        "prepare must reap expired and old-generation lease garbage"
    );
    let held = db
        .prepare_community_grouping(SPACE)
        .await
        .expect_err("a live same-generation lease must exclude a second attempt");
    assert!(
        matches!(
            held,
            CommunityGroupingError::LeaseHeld {
                ref space,
                input_generation,
            } if space == SPACE && input_generation == stale_input_generation
        ),
        "same-key exclusion must be typed LeaseHeld, got {held:?}"
    );
    assert!(
        db.run_next_community_grouping_cycle()
            .await
            .expect("phase-slot contention is a clean skip")
            .is_none(),
        "an already-held durable lease must not fail the whole refinery phase"
    );

    let work = compute_community_grouping(&attempt, &mut runtime)
        .expect("pure compute between DB phases, with no DB handle");
    assert_eq!(work.projected_edge_count(), BASE_EDGES_PER_SPACE + 20);

    let (midrun_from, midrun_to) = &chords[21];
    let midrun_promotion = stage_m4_promotion(
        &db,
        midrun_from,
        midrun_to,
        SOURCE_ID,
        SOURCE_CONTENT,
        &root_id,
    )
    .await;
    assert_eq!(
        db.promote_edges_grounded(&[midrun_promotion])
            .await
            .expect("promote correction between compute and finalize"),
        1
    );

    let stale = match work
        .finalize(&db, attempt, &mut runtime)
        .await
        .expect("generation-CAS stale finalize")
    {
        CommunityGroupingOutcome::Stale(receipt) => receipt,
        other => panic!("mid-run promotion must reject stale publication, got {other:?}"),
    };
    assert_eq!(stale.input_generation, stale_input_generation);
    assert_eq!(
        stale.published_generation, live_generation,
        "stale telemetry must report the still-live generation, not the rejected input"
    );
    mutex_holds.push(stale.db_mutex_hold);
    composer_latencies.push(stale.composer_elapsed);
    assert_eq!(
        runtime.published_generation(SPACE),
        Some(live_generation),
        "a stale finalize must not install its computed in-memory state"
    );

    let after_stale = read_m4_persisted_snapshot(&observer, SPACE).await;
    assert_eq!(after_stale.graph_generation, BASE_GRAPH_GENERATION + 21);
    assert_eq!(after_stale.published_generation, Some(live_generation));
    assert!(after_stale.dirty, "stale CAS must leave the space queued");
    assert_eq!(
        after_stale.members, before_stale.members,
        "stale finalize must publish no member row, not even a new generation stamp"
    );
    assert!(
        read_m4_lease_generations(&observer, SPACE).await.is_empty(),
        "stale finalize must consume its durable phase lease"
    );

    let (recovered_result, recovered_foreground_latency) =
        run_m4_cycle_with_foreground_probe(&db, &mut runtime, SPACE).await;
    foreground_latencies.push(recovered_foreground_latency);
    let recovered =
        match recovered_result.expect("production composer must recover the stale cycle") {
            CommunityGroupingOutcome::Published(receipt) => receipt,
            other => panic!("second cycle must publish corrected input, got {other:?}"),
        };
    let recovered_generation = read_m4_grouping_generation(&observer, SPACE).await;
    assert_eq!(recovered.input_generation, recovered_generation);
    assert_eq!(recovered.published_generation, recovered_generation);
    assert!(
        matches!(
            recovered.compute_mode,
            CommunityGroupingComputeMode::Incremental { optimized_nodes }
                if optimized_nodes > 0 && optimized_nodes < PARTICIPANTS_PER_SPACE
        ),
        "stale recovery must retain the prior runtime and stay incremental: {:?}",
        recovered.compute_mode
    );
    assert_eq!(recovered.projected_edge_count, BASE_EDGES_PER_SPACE + 21);
    mutex_holds.push(recovered.db_mutex_hold);
    composer_latencies.push(recovered.composer_elapsed);
    assert_eq!(
        runtime.published_generation(SPACE),
        Some(recovered_generation)
    );

    let recovered_snapshot = read_m4_persisted_snapshot(&observer, SPACE).await;
    assert_eq!(
        recovered_snapshot.graph_generation,
        BASE_GRAPH_GENERATION + 21
    );
    assert_eq!(
        recovered_snapshot.published_generation,
        Some(recovered_generation)
    );
    assert!(!recovered_snapshot.dirty);
    assert!(recovered_snapshot
        .members
        .iter()
        .all(|member| member.published_generation == recovered_generation));

    let (error_from, error_to) = &chords[22];
    let error_promotion = stage_m4_promotion(
        &db,
        error_from,
        error_to,
        SOURCE_ID,
        SOURCE_CONTENT,
        &root_id,
    )
    .await;
    assert_eq!(
        db.promote_edges_grounded(&[error_promotion])
            .await
            .expect("promote correction before finalize-error control"),
        1
    );
    let error_attempt = db
        .prepare_community_grouping(SPACE)
        .await
        .expect("prepare finalize-error control");
    let error_work = compute_community_grouping(&error_attempt, &mut runtime)
        .expect("compute finalize-error control");
    observer
        .execute(
            "DELETE FROM grouping_leases WHERE phase='community' AND space=?1",
            libsql::params![SPACE],
        )
        .await
        .expect("remove lease before finalize-error control");
    error_work
        .finalize(&db, error_attempt, &mut runtime)
        .await
        .expect_err("missing lease must fail finalization");
    assert_eq!(
        runtime.published_generation(SPACE),
        Some(recovered_generation),
        "finalize errors must roll the frontier state back to the live publication"
    );
    let error_retry = match run_community_grouping_cycle(&db, &mut runtime, SPACE)
        .await
        .expect("retry finalize-error generation")
    {
        CommunityGroupingOutcome::Published(receipt) => receipt,
        other => panic!("finalize-error retry must publish, got {other:?}"),
    };
    assert!(
        matches!(
            error_retry.compute_mode,
            CommunityGroupingComputeMode::Incremental { optimized_nodes }
                if optimized_nodes > 0 && optimized_nodes < PARTICIPANTS_PER_SPACE
        ),
        "error recovery must retain the prior runtime and stay incremental"
    );
    mutex_holds.push(error_retry.db_mutex_hold);
    composer_latencies.push(error_retry.composer_elapsed);
    assert_eq!(
        runtime.published_generation(SECOND_SPACE),
        Some(second_runtime_generation),
        "updating one space must neither copy nor alter another runtime partition"
    );
    assert_eq!(
        read_legacy_community_shadow(&observer, SPACE).await,
        legacy_before,
        "PR-1 is write-only shadow: entities.community_id must be byte-unchanged"
    );
    assert!(
        read_m4_lease_generations(&observer, SPACE).await.is_empty(),
        "successful recovery publication must consume its durable phase lease"
    );

    assert!(
        mutex_holds.len() >= 20,
        "Gate 1.4 needs at least 20 real-path firings at graded scale"
    );
    let mutex_p95 = duration_p95(&mutex_holds);
    let mutex_max = mutex_holds.iter().copied().max().expect("mutex receipts");
    assert!(
        mutex_p95 <= Duration::from_millis(500),
        "Gate 1.4 real-path DB-mutex p95 {mutex_p95:?} exceeds 500 ms"
    );
    assert!(
        mutex_max < Duration::from_secs(2),
        "Gate 1.4 real-path DB-mutex max {mutex_max:?} reaches the 2 s hard fail"
    );

    let foreground_p95 = duration_p95(&foreground_latencies);
    let foreground_max = foreground_latencies
        .iter()
        .copied()
        .max()
        .expect("foreground latency receipts");
    assert!(
        foreground_p95 <= Duration::from_millis(500),
        "independent foreground-query p95 {foreground_p95:?} exceeds the mutex bar"
    );
    assert!(
        foreground_max < Duration::from_secs(2),
        "independent foreground-query max {foreground_max:?} reaches the hard fail"
    );
    println!(
        "[m4_production_composer] composer_p95_ms={:.3} composer_max_ms={:.3} \
         input_rows_loaded={} member_rows_written={} mutex_p95_ms={:.3} \
         foreground_p95_ms={:.3}",
        duration_p95(&composer_latencies).as_secs_f64() * 1_000.0,
        composer_latencies
            .iter()
            .copied()
            .max()
            .expect("composer latency receipts")
            .as_secs_f64()
            * 1_000.0,
        graded_input_rows,
        graded_member_rows,
        mutex_p95.as_secs_f64() * 1_000.0,
        foreground_p95.as_secs_f64() * 1_000.0,
    );
    assert_m4_job_phase_structure();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "M4 PR-2 graded 100k-memory / 5k-page routing and consistency receipt"]
async fn m4_pr2_routing_consistency_and_index_scale_receipt() {
    let dir = tempfile::tempdir().expect("M4 PR-2 scale tempdir");
    let db = Arc::new(
        MemoryDB::new(dir.path(), Arc::new(NoopEmitter))
            .await
            .expect("open M4 PR-2 scale database"),
    );
    let observer_db = libsql::Builder::new_local(dir.path().join("origin_memory.db"))
        .build()
        .await
        .expect("open independent M4 PR-2 observer");
    let observer = observer_db.connect().expect("connect M4 PR-2 observer");
    seed_m4_graded_corpus(&observer).await;

    let mut runtime = CommunityGroupingRuntime::default();
    let mut routed_pages = 0usize;
    let mut stale_writes = 0usize;
    for index in 0..3 {
        let space = format!("space-{index}");
        let outcome = run_community_grouping_cycle(&db, &mut runtime, &space)
            .await
            .expect("publish M4 PR-2 community snapshot");
        let generation = match outcome {
            CommunityGroupingOutcome::Published(receipt) => receipt.published_generation,
            CommunityGroupingOutcome::Held(_) => panic!("graded snapshot unexpectedly held"),
            CommunityGroupingOutcome::Stale(_) => panic!("graded snapshot unexpectedly stale"),
        };
        let routing = db
            .refresh_page_community_routes(&space, generation)
            .await
            .expect("route M4 PR-2 pages");
        routed_pages += routing.assignments_published;
        stale_writes += routing.stale_writes;
    }

    let consistency = db
        .reconcile_community_consistency()
        .await
        .expect("M4 PR-2 consistency sweep");
    assert_eq!(
        consistency.violation_count(),
        0,
        "Gate 4.9 requires exact-zero consistency violations: {consistency:?}"
    );
    assert_eq!(routed_pages, 5_000);
    assert_eq!(stale_writes, 0);

    let mut assignment_rows = observer
        .query("SELECT COUNT(*) FROM page_community_assignments", ())
        .await
        .expect("count M4 PR-2 assignments");
    let assignment_count = assignment_rows
        .next()
        .await
        .expect("read M4 PR-2 assignment count")
        .expect("M4 PR-2 assignment count row")
        .get::<i64>(0)
        .expect("decode M4 PR-2 assignment count");
    assert_eq!(assignment_count, 5_000);

    let mut assignment_state_rows = observer
        .query(
            "SELECT state, COUNT(*)
               FROM page_community_assignments
              GROUP BY state ORDER BY state",
            (),
        )
        .await
        .expect("count M4 PR-2 assignment states");
    let mut assignment_states = BTreeMap::new();
    while let Some(row) = assignment_state_rows
        .next()
        .await
        .expect("read M4 PR-2 assignment state")
    {
        assignment_states.insert(
            row.get::<String>(0)
                .expect("decode M4 PR-2 assignment state"),
            row.get::<i64>(1)
                .expect("decode M4 PR-2 assignment state count"),
        );
    }
    assert_eq!(
        assignment_states,
        BTreeMap::from([("assigned".to_owned(), 5_000)]),
        "the graded routing receipt must exercise native-edge assignments, not trivial drops"
    );

    for (sql, expected_index) in [
        (
            "EXPLAIN QUERY PLAN SELECT edge_id FROM edges
              WHERE space='space-0' AND edge_type='relates'
                AND grounded=1 AND valid_until IS NULL
                AND src_kind='entity' AND dst_kind='entity'
                AND lineage='assertion'",
            "idx_edges_active_grounded_space_type",
        ),
        (
            "EXPLAIN QUERY PLAN SELECT node_id FROM community_members
              WHERE community_id='missing'",
            "idx_community_members_community",
        ),
    ] {
        let mut rows = observer.query(sql, ()).await.expect("explain M4 hot read");
        let mut details = Vec::new();
        while let Some(row) = rows.next().await.expect("read M4 explain row") {
            details.push(row.get::<String>(3).expect("decode M4 explain detail"));
        }
        assert!(
            details.iter().any(|detail| detail.contains(expected_index)),
            "Gate 4.10 expected {expected_index}, got {details:?}"
        );
    }

    println!(
        "[m4_pr2_scale] memories=100000 pages=5000 routed_pages={routed_pages} \
         stale_writes={stale_writes} consistency_violations={} indexes=accepted",
        consistency.violation_count()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "M4 PR-2 graded 100k-memory / 5k-page global parity reconciliation receipt"]
async fn m4_pr2_global_parity_reconcile_scale_receipt() {
    const RUNS: usize = 20;
    const CONSUMER: &str = "summary_buckets";

    let dir = tempfile::tempdir().expect("M4 parity scale tempdir");
    let db = Arc::new(
        MemoryDB::new(dir.path(), Arc::new(NoopEmitter))
            .await
            .expect("open M4 parity scale database"),
    );
    let observer_db = libsql::Builder::new_local(dir.path().join("origin_memory.db"))
        .build()
        .await
        .expect("open independent M4 parity scale observer");
    let observer = observer_db
        .connect()
        .expect("connect M4 parity scale observer");
    seed_m4_graded_corpus(&observer).await;

    let mut runtime = CommunityGroupingRuntime::default();
    for index in 0..3 {
        let space = format!("space-{index}");
        assert!(
            matches!(
                run_community_grouping_cycle(&db, &mut runtime, &space)
                    .await
                    .expect("publish M4 parity scale community snapshot"),
                CommunityGroupingOutcome::Published(_)
            ),
            "graded parity corpus must publish every space"
        );
    }

    observer
        .execute(
            "UPDATE entities
                SET community_id=1
              WHERE id IN (SELECT node_id FROM community_members)",
            (),
        )
        .await
        .expect("seed scale legacy community coverage");
    observer
        .execute(
            "UPDATE memories
                SET embedding=vector32(?1)
              WHERE entity_id IS NOT NULL",
            libsql::params![m4_unit_embedding(0)],
        )
        .await
        .expect("seed scale parity candidate embeddings");

    let mut memory_rows = observer
        .query("SELECT COUNT(*) FROM memories", ())
        .await
        .expect("count parity scale memories");
    let memory_count = memory_rows
        .next()
        .await
        .expect("read parity scale memory count")
        .expect("parity scale memory count row")
        .get::<i64>(0)
        .expect("decode parity scale memory count");
    drop(memory_rows);
    let mut page_rows = observer
        .query("SELECT COUNT(*) FROM pages", ())
        .await
        .expect("count parity scale pages");
    let page_count = page_rows
        .next()
        .await
        .expect("read parity scale page count")
        .expect("parity scale page count row")
        .get::<i64>(0)
        .expect("decode parity scale page count");
    drop(page_rows);
    assert_eq!((memory_count, page_count), (100_000, 5_000));

    let mut mutex_holds = Vec::with_capacity(RUNS);
    let mut reconcile_elapsed = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let receipt = db
            .reconcile_community_reader_parity(CONSUMER)
            .await
            .expect("reconcile global community-reader parity at scale");
        assert!(
            receipt.ready,
            "graded parity corpus must retain equal global source coverage: {receipt:?}"
        );
        assert_eq!(receipt.spaces_checked, 3);
        mutex_holds.push(receipt.canonical_mutex_hold);
        reconcile_elapsed.push(receipt.reconcile_elapsed);
    }
    let mutex_p95 = duration_p95(&mutex_holds);
    let mutex_max = mutex_holds.iter().copied().max().expect("mutex receipts");
    assert!(
        mutex_p95 <= Duration::from_millis(500),
        "global parity canonical-mutex p95 {mutex_p95:?} exceeds 500 ms"
    );
    assert!(
        mutex_max < Duration::from_secs(2),
        "global parity canonical-mutex max {mutex_max:?} reaches the 2 s hard fail"
    );

    let total_p50 = duration_percentile(&reconcile_elapsed, 50);
    let total_p95 = duration_p95(&reconcile_elapsed);
    let total_max = reconcile_elapsed
        .iter()
        .copied()
        .max()
        .expect("reconcile receipts");
    let mut checkpoint_rows = observer
        .query("PRAGMA wal_checkpoint(PASSIVE)", ())
        .await
        .expect("checkpoint parity scale WAL");
    let checkpoint = checkpoint_rows
        .next()
        .await
        .expect("read parity scale checkpoint")
        .expect("parity scale checkpoint row");
    let wal_busy = checkpoint
        .get::<i64>(0)
        .expect("decode parity scale WAL busy");
    let wal_frames = checkpoint
        .get::<i64>(1)
        .expect("decode parity scale WAL frames");
    let checkpointed_frames = checkpoint
        .get::<i64>(2)
        .expect("decode parity scale checkpointed frames");

    println!(
        "[m4_pr2_global_parity_scale] memories={memory_count} pages={page_count} runs={RUNS} \
         canonical_mutex_p95_ms={:.3} canonical_mutex_max_ms={:.3} \
         reconcile_p50_ms={:.3} reconcile_p95_ms={:.3} reconcile_max_ms={:.3} \
         wal_busy={wal_busy} wal_frames={wal_frames} \
         checkpointed_frames={checkpointed_frames}",
        mutex_p95.as_secs_f64() * 1_000.0,
        mutex_max.as_secs_f64() * 1_000.0,
        total_p50.as_secs_f64() * 1_000.0,
        total_p95.as_secs_f64() * 1_000.0,
        total_max.as_secs_f64() * 1_000.0,
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn m4_concurrent_lease_is_exact_once_and_isolated_nodes_attach_posthoc() {
    const SPACE: &str = "m4-isolated-space";
    const ROOT: &str = "m4-isolated-root";
    const SOURCE_ID: &str = "m4-isolated-source";
    const SOURCE_CONTENT: &str = "The isolated M4 fixture supports bounded core corrections.";

    let dir = tempfile::tempdir().expect("M4 isolated tempdir");
    let db = Arc::new(
        MemoryDB::new(dir.path(), Arc::new(NoopEmitter))
            .await
            .expect("open M4 isolated database"),
    );
    db.upsert_documents(vec![RawDocument {
        source: "memory".to_owned(),
        source_id: SOURCE_ID.to_owned(),
        title: "M4 isolated source".to_owned(),
        content: SOURCE_CONTENT.to_owned(),
        last_modified: 1_722_000_000,
        memory_type: Some("fact".to_owned()),
        space: Some(SPACE.to_owned()),
        source_agent: Some("folder".to_owned()),
        confirmed: Some(true),
        ..Default::default()
    }])
    .await
    .expect("seed isolated promotion source");
    let observer_db = libsql::Builder::new_local(dir.path().join("origin_memory.db"))
        .build()
        .await
        .expect("open M4 isolated observer");
    let observer = observer_db.connect().expect("connect M4 isolated observer");
    let unit_embedding = m4_unit_embedding(0);

    observer
        .execute("BEGIN", ())
        .await
        .expect("begin M4 isolated seed");
    observer
        .execute(
            "INSERT INTO provenance_roots \
                (root_id, identity_version, identity_digest, root_kind, \
                 independence_group_id, status, created_at) \
             VALUES (?1, 1, 'm4-isolated-digest', 'generated', \
                     'm4-isolated-group', 'active', 1712707200)",
            libsql::params![ROOT],
        )
        .await
        .expect("seed M4 isolated provenance root");
    for node in 0..10 {
        let entity_id = format!("core-{node:02}");
        observer
            .execute(
                "INSERT INTO entities \
                    (id, name, entity_type, space, created_at, updated_at, embedding) \
                 VALUES (?1, ?1, 'concept', ?2, 1712707200, 1712707200, vector32(?3))",
                libsql::params![entity_id, SPACE, unit_embedding.as_str()],
            )
            .await
            .expect("seed core entity");
    }
    observer
        .execute(
            "INSERT INTO entities \
                (id, name, entity_type, space, created_at, updated_at) VALUES \
                ('isolated-ungrounded', 'isolated-ungrounded', 'concept', ?1, 1712707200, 1712707200), \
                ('isolated-none', 'isolated-none', 'concept', ?1, 1712707200, 1712707200), \
                ('z-chain-a', 'z-chain-a', 'concept', ?1, 1712707200, 1712707200), \
                ('z-chain-b', 'z-chain-b', 'concept', ?1, 1712707200, 1712707200), \
                ('z-cycle-a', 'z-cycle-a', 'concept', ?1, 1712707200, 1712707200), \
                ('z-cycle-b', 'z-cycle-b', 'concept', ?1, 1712707200, 1712707200)",
            libsql::params![SPACE],
        )
        .await
        .expect("seed non-embedded isolated entities");
    observer
        .execute(
            "INSERT INTO entities \
                (id, name, entity_type, space, created_at, updated_at, embedding) \
             VALUES ('isolated-embedding', 'isolated-embedding', 'concept', ?1, \
                     1712707200, 1712707200, vector32(?2))",
            libsql::params![SPACE, unit_embedding],
        )
        .await
        .expect("seed embedded isolated entity");
    for node in 0..10 {
        observer
            .execute(
                "INSERT INTO edges \
                    (edge_id, src_id, src_kind, dst_id, dst_kind, edge_type, lineage, \
                     grounded, root_id, space, created_at) \
                 VALUES (?1, ?2, 'entity', ?3, 'entity', 'relates', 'assertion', \
                         1, ?4, ?5, 1712707200)",
                libsql::params![
                    format!("core-edge-{node:02}"),
                    format!("core-{node:02}"),
                    format!("core-{:02}", (node + 1) % 10),
                    ROOT,
                    SPACE
                ],
            )
            .await
            .expect("seed grounded core ring");
    }
    observer
        .execute(
            "INSERT INTO edges \
                (edge_id, src_id, src_kind, dst_id, dst_kind, edge_type, lineage, \
                 grounded, root_id, space, created_at) \
             VALUES ('isolated-ungrounded-edge', 'isolated-ungrounded', 'entity', \
                     'core-00', 'entity', 'relates', 'assertion', 0, NULL, ?1, 1712707200)",
            libsql::params![SPACE],
        )
        .await
        .expect("seed ungrounded isolated attachment");
    observer
        .execute(
            "INSERT INTO edges \
                (edge_id, src_id, src_kind, dst_id, dst_kind, edge_type, lineage, \
                 grounded, root_id, space, created_at) VALUES \
                ('chain-a-b', 'z-chain-a', 'entity', 'z-chain-b', 'entity', \
                 'relates', 'assertion', 0, NULL, ?1, 1712707200), \
                ('chain-b-core-1', 'z-chain-b', 'entity', 'core-00', 'entity', \
                 'relates', 'assertion', 0, NULL, ?1, 1712707200), \
                ('chain-b-core-2', 'z-chain-b', 'entity', 'core-00', 'entity', \
                 'relates', 'assertion', 0, NULL, ?1, 1712707200), \
                ('cycle-a-b', 'z-cycle-a', 'entity', 'z-cycle-b', 'entity', \
                 'relates', 'assertion', 0, NULL, ?1, 1712707200)",
            libsql::params![SPACE],
        )
        .await
        .expect("seed multi-hop and cyclic ungrounded attachments");
    observer
        .execute("COMMIT", ())
        .await
        .expect("commit M4 isolated seed");

    let (left, right) = tokio::join!(
        db.prepare_community_grouping(SPACE),
        db.prepare_community_grouping(SPACE)
    );
    let attempt = match (left, right) {
        (Ok(attempt), Err(CommunityGroupingError::LeaseHeld { .. }))
        | (Err(CommunityGroupingError::LeaseHeld { .. }), Ok(attempt)) => attempt,
        outcome => panic!("exactly one concurrent same-generation lease must win: {outcome:?}"),
    };
    let leased_generation = attempt.input_generation();

    let mut runtime = CommunityGroupingRuntime::default();
    let work =
        compute_community_grouping(&attempt, &mut runtime).expect("compute isolated grouping");
    let late_relation = db
        .create_relation(
            "isolated-none",
            "core-00",
            "related_to",
            Some("m4-review"),
            Some(1.0),
            Some("attachment input changes after prepare"),
            None,
        )
        .await
        .expect("write an ungrounded attachment after prepare");
    let stale = match work
        .finalize(&db, attempt, &mut runtime)
        .await
        .expect("finalize isolated grouping")
    {
        CommunityGroupingOutcome::Stale(receipt) => receipt,
        other => panic!("an attachment-input write must stale the leased snapshot: {other:?}"),
    };
    assert_eq!(stale.input_generation, leased_generation);

    let stale_snapshot = read_m4_persisted_snapshot(&observer, SPACE).await;
    assert_eq!(stale_snapshot.graph_generation, 10);
    assert_eq!(
        read_m4_grouping_generation(&observer, SPACE).await,
        leased_generation + 1
    );
    assert_eq!(stale_snapshot.published_generation, None);
    assert!(stale_snapshot.dirty);
    assert!(stale_snapshot.members.is_empty());

    let receipt = match db
        .run_next_community_grouping_cycle()
        .await
        .expect("rerun phase slot after attachment-input stale")
        .expect("dirty attachment generation must be selected")
    {
        CommunityGroupingOutcome::Published(receipt) => receipt,
        other => panic!("the current attachment generation must publish: {other:?}"),
    };
    assert_eq!(receipt.published_generation, leased_generation + 1);
    assert_eq!(receipt.member_count, 15);
    assert_eq!(
        receipt.compute_mode,
        CommunityGroupingComputeMode::Full {
            reason: CommunityGroupingFullReason::FirstPublication,
        }
    );

    let snapshot = read_m4_persisted_snapshot(&observer, SPACE).await;
    assert_eq!(snapshot.published_generation, Some(leased_generation + 1));
    assert!(!snapshot.dirty);
    let core = snapshot
        .members
        .iter()
        .find(|member| member.node_id == "core-00")
        .expect("core member");
    assert_eq!(core.attachment, "core");
    let ungrounded = snapshot
        .members
        .iter()
        .find(|member| member.node_id == "isolated-ungrounded")
        .expect("ungrounded isolated node must attach");
    assert_eq!(ungrounded.attachment, "isolated_ungrounded");
    assert_eq!(
        ungrounded.community_id, core.community_id,
        "strongest ungrounded neighbor must supply the attachment community"
    );
    assert_eq!(
        snapshot
            .members
            .iter()
            .find(|member| member.node_id == "isolated-embedding")
            .expect("embedded isolated node must attach")
            .attachment,
        "isolated_embedding"
    );
    assert!(
        snapshot
            .members
            .iter()
            .find(|member| member.node_id == "isolated-none")
            .is_some_and(|member| member.attachment == "isolated_ungrounded"),
        "the late ungrounded write must queue and publish the isolated attachment"
    );
    assert_eq!(
        snapshot
            .members
            .iter()
            .find(|member| member.node_id == "z-chain-a")
            .expect("multi-hop isolated node must attach")
            .community_id,
        core.community_id,
        "multi-hop strongest-neighbor chains must resolve to the core community"
    );
    assert!(
        snapshot
            .members
            .iter()
            .all(|member| !member.node_id.starts_with("z-cycle-")),
        "an ungrounded cycle with no core or embedding path must stay unassigned"
    );
    let core_before = snapshot
        .members
        .iter()
        .filter(|member| member.attachment == "core")
        .map(|member| (member.node_id.clone(), member.community_id.clone()))
        .collect::<Vec<_>>();

    let mut version_rows = observer
        .query(
            "SELECT DISTINCT algo_version, projection_version \
             FROM communities WHERE space = ?1 AND retired_at IS NULL",
            libsql::params![SPACE],
        )
        .await
        .expect("query persisted community versions");
    while let Some(row) = version_rows
        .next()
        .await
        .expect("read persisted community versions")
    {
        assert_eq!(row.get::<String>(0).expect("algo version"), "leiden-m4-v1");
        assert_eq!(
            row.get::<String>(1).expect("projection version"),
            "grounded-relates-v1"
        );
    }

    db.supersede_relation(&late_relation, "unused-review-winner")
        .await
        .expect("soft-invalidate the ungrounded attachment");
    assert_eq!(read_m4_generation(&observer, SPACE).await, 10);
    assert_eq!(
        read_m4_grouping_generation(&observer, SPACE).await,
        leased_generation + 2
    );
    let deleted = db
        .run_next_community_grouping_cycle()
        .await
        .expect("run attachment deletion regroup")
        .expect("ungrounded deletion must queue a grouping cycle");
    match deleted {
        CommunityGroupingOutcome::Published(receipt) => {
            assert_eq!(receipt.published_generation, leased_generation + 2);
            assert_eq!(
                receipt.compute_mode,
                CommunityGroupingComputeMode::CoreReused,
                "ungrounded-only deletion must preserve the published core partition"
            );
        }
        other => panic!("ungrounded deletion must publish current input: {other:?}"),
    }
    let deleted_snapshot = read_m4_persisted_snapshot(&observer, SPACE).await;
    assert!(
        deleted_snapshot
            .members
            .iter()
            .all(|member| member.node_id != "isolated-none"),
        "soft-invalidating the only ungrounded attachment must remove the stale assignment"
    );

    observer
        .execute(
            "UPDATE entities SET embedding = vector32(?1) WHERE id = 'isolated-embedding'",
            libsql::params![m4_unit_embedding(1)],
        )
        .await
        .expect("mutate an attachment embedding after clean publication");
    assert_eq!(
        read_m4_grouping_generation(&observer, SPACE).await,
        leased_generation + 3
    );
    match db
        .run_next_community_grouping_cycle()
        .await
        .expect("run embedding-only generation")
        .expect("embedding mutation must queue grouping")
    {
        CommunityGroupingOutcome::Published(receipt) => {
            assert_eq!(receipt.published_generation, leased_generation + 3);
            assert_eq!(
                receipt.compute_mode,
                CommunityGroupingComputeMode::CoreReused,
                "embedding-only publication must preserve the core partition"
            );
        }
        other => panic!("embedding-only generation must publish: {other:?}"),
    }
    let embedding_snapshot = read_m4_persisted_snapshot(&observer, SPACE).await;
    let core_after = embedding_snapshot
        .members
        .iter()
        .filter(|member| member.attachment == "core")
        .map(|member| (member.node_id.clone(), member.community_id.clone()))
        .collect::<Vec<_>>();
    assert_eq!(
        core_after, core_before,
        "attachment-only cycles must preserve core durable membership byte-for-byte"
    );

    let mut promotions = Vec::new();
    for (left, right) in [
        ("core-00", "core-05"),
        ("core-01", "core-06"),
        ("core-02", "core-07"),
    ] {
        promotions
            .push(stage_m4_promotion(&db, left, right, SOURCE_ID, SOURCE_CONTENT, ROOT).await);
    }
    assert_eq!(
        db.promote_edges_grounded(&promotions)
            .await
            .expect("promote wide dirty frontier"),
        promotions.len(),
    );
    let threshold = db
        .run_next_community_grouping_cycle()
        .await
        .expect("run threshold full-mode control")
        .expect("wide frontier must queue grouping");
    assert!(matches!(
        threshold,
        CommunityGroupingOutcome::Published(ref receipt)
            if receipt.compute_mode == CommunityGroupingComputeMode::Full {
                reason: CommunityGroupingFullReason::ThresholdExceeded,
            }
    ));

    observer
        .execute(
            "UPDATE space_graph_state \
             SET graph_generation = graph_generation + 1, \
                 grouping_generation = grouping_generation + 1, dirty = 1 \
             WHERE space = ?1",
            libsql::params![SPACE],
        )
        .await
        .expect("advance graph generation without a volatile frontier");
    let missing_frontier = db
        .run_next_community_grouping_cycle()
        .await
        .expect("run missing-frontier full-mode control")
        .expect("manual generation advance must queue grouping");
    assert!(matches!(
        missing_frontier,
        CommunityGroupingOutcome::Published(ref receipt)
            if receipt.compute_mode == CommunityGroupingComputeMode::Full {
                reason: CommunityGroupingFullReason::DirtyFrontierMissing,
            }
    ));

    observer
        .execute(
            "UPDATE communities SET algo_version = 'old-m4-version' \
             WHERE space = ?1 AND retired_at IS NULL",
            libsql::params![SPACE],
        )
        .await
        .expect("inject published algorithm-version mismatch");
    observer
        .execute(
            "UPDATE entities SET embedding = vector32(?1) WHERE id = 'isolated-embedding'",
            libsql::params![m4_unit_embedding(2)],
        )
        .await
        .expect("queue grouping after version mismatch");
    let version_changed = db
        .run_next_community_grouping_cycle()
        .await
        .expect("run version full-mode control")
        .expect("version mismatch must queue grouping");
    assert!(matches!(
        version_changed,
        CommunityGroupingOutcome::Published(ref receipt)
            if receipt.compute_mode == CommunityGroupingComputeMode::Full {
                reason: CommunityGroupingFullReason::VersionChanged,
            }
    ));

    observer
        .execute(
            "UPDATE entities SET embedding = vector32(?1) WHERE id = 'isolated-embedding'",
            libsql::params![m4_unit_embedding(3)],
        )
        .await
        .expect("queue grouping before runtime restart");
    drop(observer);
    drop(observer_db);
    drop(db);
    let reopened = MemoryDB::new(dir.path(), Arc::new(NoopEmitter))
        .await
        .expect("reopen M4 database without volatile runtime state");
    let restarted = reopened
        .run_next_community_grouping_cycle()
        .await
        .expect("run restart full-mode control")
        .expect("queued restart generation must be selected");
    assert!(matches!(
        restarted,
        CommunityGroupingOutcome::Published(ref receipt)
            if receipt.compute_mode == CommunityGroupingComputeMode::Full {
                reason: CommunityGroupingFullReason::RuntimeStateMissing,
            }
    ));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn m4_raw_grounded_writes_force_full_recovery_instead_of_reusing_stale_core() {
    const SPACE: &str = "m4-raw-trigger-space";
    const ROOT: &str = "m4-raw-trigger-root";

    let dir = tempfile::tempdir().expect("M4 raw-trigger tempdir");
    let db = Arc::new(
        MemoryDB::new(dir.path(), Arc::new(NoopEmitter))
            .await
            .expect("open M4 raw-trigger database"),
    );
    let observer_db = libsql::Builder::new_local(dir.path().join("origin_memory.db"))
        .build()
        .await
        .expect("open M4 raw-trigger observer");
    let observer = observer_db
        .connect()
        .expect("connect M4 raw-trigger observer");

    observer
        .execute(
            "INSERT INTO provenance_roots
                (root_id, identity_version, identity_digest, root_kind,
                 independence_group_id, status, created_at)
             VALUES (?1, 1, 'm4-raw-trigger-digest', 'generated',
                     'm4-raw-trigger-group', 'active', 1712707200)",
            libsql::params![ROOT],
        )
        .await
        .expect("seed raw-trigger provenance root");
    for node in 0..10 {
        observer
            .execute(
                "INSERT INTO entities
                    (id, name, entity_type, space, created_at, updated_at)
                 VALUES (?1, ?1, 'concept', ?2, 1712707200, 1712707200)",
                libsql::params![format!("raw-node-{node:02}"), SPACE],
            )
            .await
            .expect("seed raw-trigger entity");
    }
    for node in 0..10 {
        observer
            .execute(
                "INSERT INTO edges
                    (edge_id, src_id, src_kind, dst_id, dst_kind, edge_type, lineage,
                     grounded, root_id, space, created_at)
                 VALUES (?1, ?2, 'entity', ?3, 'entity', 'relates', 'assertion',
                         1, ?4, ?5, 1712707200)",
                libsql::params![
                    format!("raw-base-edge-{node:02}"),
                    format!("raw-node-{node:02}"),
                    format!("raw-node-{:02}", (node + 1) % 10),
                    ROOT,
                    SPACE
                ],
            )
            .await
            .expect("seed raw-trigger base edge");
    }
    let mut runtime = CommunityGroupingRuntime::default();
    let first = run_community_grouping_cycle(&db, &mut runtime, SPACE)
        .await
        .expect("publish raw-trigger baseline");
    assert!(matches!(
        first,
        CommunityGroupingOutcome::Published(ref receipt)
            if receipt.compute_mode
                == (CommunityGroupingComputeMode::Full {
                    reason: CommunityGroupingFullReason::FirstPublication,
                })
    ));

    let baseline_graph_generation = read_m4_generation(&observer, SPACE).await;
    observer
        .execute(
            "INSERT INTO edges
                (edge_id, src_id, src_kind, dst_id, dst_kind, edge_type, lineage,
                 grounded, root_id, space, created_at)
             VALUES ('raw-direct-edge', 'raw-node-00', 'entity', 'raw-node-03', 'entity',
                     'relates', 'assertion', 1, ?1, ?2, 1712707200)",
            libsql::params![ROOT, SPACE],
        )
        .await
        .expect("raw direct grounded insert");
    assert_eq!(
        read_m4_generation(&observer, SPACE).await,
        baseline_graph_generation + 1,
        "raw grounded insert must advance the core graph generation"
    );
    let inserted = run_community_grouping_cycle(&db, &mut runtime, SPACE)
        .await
        .expect("recover raw grounded insert");
    assert!(matches!(
        inserted,
        CommunityGroupingOutcome::Published(ref receipt)
            if receipt.compute_mode
                == (CommunityGroupingComputeMode::Full {
                    reason: CommunityGroupingFullReason::DirtyFrontierMissing,
                })
    ));

    observer
        .execute("DELETE FROM edges WHERE edge_id = 'raw-direct-edge'", ())
        .await
        .expect("raw direct grounded delete");
    assert_eq!(
        read_m4_generation(&observer, SPACE).await,
        baseline_graph_generation + 2,
        "raw grounded delete must advance the core graph generation"
    );
    let deleted = run_community_grouping_cycle(&db, &mut runtime, SPACE)
        .await
        .expect("recover raw grounded delete");
    assert!(matches!(
        deleted,
        CommunityGroupingOutcome::Published(ref receipt)
            if receipt.compute_mode
                == (CommunityGroupingComputeMode::Full {
                    reason: CommunityGroupingFullReason::DirtyFrontierMissing,
                })
    ));

    observer
        .execute(
            "UPDATE edges SET dst_id = 'raw-node-04'
             WHERE edge_id = 'raw-base-edge-00'",
            (),
        )
        .await
        .expect("raw direct grounded topology update");
    assert_eq!(
        read_m4_generation(&observer, SPACE).await,
        baseline_graph_generation + 3,
        "raw grounded topology update must advance the core graph generation"
    );
    let updated = run_community_grouping_cycle(&db, &mut runtime, SPACE)
        .await
        .expect("recover raw grounded topology update");
    assert!(matches!(
        updated,
        CommunityGroupingOutcome::Published(ref receipt)
            if receipt.compute_mode
                == (CommunityGroupingComputeMode::Full {
                    reason: CommunityGroupingFullReason::DirtyFrontierMissing,
                })
    ));

    observer
        .execute(
            "UPDATE edges SET dst_id = 'raw-node-05'
             WHERE edge_id = 'raw-base-edge-00'",
            (),
        )
        .await
        .expect("queue shared-runtime baseline");
    let shared_baseline = db
        .run_next_community_grouping_cycle()
        .await
        .expect("publish shared-runtime baseline")
        .expect("shared-runtime baseline work");
    assert!(matches!(
        shared_baseline,
        CommunityGroupingOutcome::Published(ref receipt)
            if receipt.compute_mode
                == (CommunityGroupingComputeMode::Full {
                    reason: CommunityGroupingFullReason::RuntimeStateMissing,
                })
    ));

    observer
        .execute(
            "UPDATE edges SET dst_id = 'raw-node-06'
             WHERE edge_id = 'raw-base-edge-00'",
            (),
        )
        .await
        .expect("queue concurrent shared-runtime correction");
    let (left, right) = tokio::join!(
        db.run_next_community_grouping_cycle(),
        db.run_next_community_grouping_cycle()
    );
    let outcomes = [
        left.expect("left same-space phase contender"),
        right.expect("right same-space phase contender"),
    ];
    assert_eq!(
        outcomes.iter().filter(|outcome| outcome.is_some()).count(),
        1,
        "exactly one same-space phase contender must publish"
    );
    assert_eq!(
        outcomes.iter().filter(|outcome| outcome.is_none()).count(),
        1,
        "the losing same-space phase contender must skip cleanly"
    );
    let winner = outcomes
        .into_iter()
        .flatten()
        .next()
        .expect("same-space winner");
    assert!(
        matches!(
            winner,
            CommunityGroupingOutcome::Published(ref receipt)
                if receipt.compute_mode
                    == (CommunityGroupingComputeMode::Full {
                        reason: CommunityGroupingFullReason::DirtyFrontierMissing,
                    })
        ),
        "the lease winner must receive the prior shared runtime, got {winner:?}"
    );
}

#[tokio::test]
async fn m4_below_floor_holds_prior_publication_and_stays_queued() {
    const SPACE: &str = "m4-below-floor-space";
    const PARTICIPANTS: usize = 10;

    let dir = tempfile::tempdir().expect("M4 below-floor tempdir");
    let db = MemoryDB::new(dir.path(), Arc::new(NoopEmitter))
        .await
        .expect("open M4 below-floor database");
    let observer_db = libsql::Builder::new_local(dir.path().join("origin_memory.db"))
        .build()
        .await
        .expect("open below-floor observer");
    let observer = observer_db.connect().expect("connect below-floor observer");

    observer
        .execute(
            "INSERT INTO provenance_roots
             (root_id, identity_version, identity_digest, root_kind,
              independence_group_id, status, created_at)
             VALUES ('m4-below-floor-root', 1, 'm4-below-floor-digest',
                     'generated', 'm4-below-floor-group', 'active', 1712707200)",
            (),
        )
        .await
        .expect("seed below-floor provenance root");
    for node in 0..PARTICIPANTS {
        let node_id = format!("below-floor-node-{node:02}");
        observer
            .execute(
                "INSERT INTO entities
                 (id, name, entity_type, space, created_at, updated_at)
                 VALUES (?1, ?1, 'concept', ?2, 1712707200, 1712707200)",
                libsql::params![node_id, SPACE],
            )
            .await
            .expect("seed below-floor entity");
    }
    for node in 0..PARTICIPANTS {
        observer
            .execute(
                "INSERT INTO edges
                 (edge_id, src_id, src_kind, dst_id, dst_kind, edge_type, lineage,
                  grounded, root_id, space, created_at)
                 VALUES (?1, ?2, 'entity', ?3, 'entity', 'relates', 'assertion',
                         1, 'm4-below-floor-root', ?4, 1712707200)",
                libsql::params![
                    format!("below-floor-edge-{node:02}"),
                    format!("below-floor-node-{node:02}"),
                    format!("below-floor-node-{:02}", (node + 1) % PARTICIPANTS),
                    SPACE,
                ],
            )
            .await
            .expect("seed below-floor grounded edge");
    }

    let mut runtime = CommunityGroupingRuntime::default();
    let first = run_community_grouping_cycle(&db, &mut runtime, SPACE)
        .await
        .expect("publish viable baseline");
    assert!(
        matches!(first, CommunityGroupingOutcome::Published(_)),
        "ten participants are the positive floor control: {first:?}"
    );
    let before = read_m4_persisted_snapshot(&observer, SPACE).await;
    let published_generation = before
        .published_generation
        .expect("positive floor control publishes");
    assert_eq!(before.members.len(), PARTICIPANTS);

    observer
        .execute(
            "UPDATE edges SET grounded = 0
             WHERE space = ?1 AND edge_id <> 'below-floor-edge-00'",
            libsql::params![SPACE],
        )
        .await
        .expect("drop grounded graph below the viability floor");
    let held_generation = read_m4_grouping_generation(&observer, SPACE).await;

    let below_floor = run_community_grouping_cycle(&db, &mut runtime, SPACE)
        .await
        .expect("run below-floor generation");
    assert!(
        matches!(below_floor, CommunityGroupingOutcome::Held(_)),
        "below-floor input must return Held instead of publishing empty: {below_floor:?}"
    );
    let after = read_m4_persisted_snapshot(&observer, SPACE).await;
    assert_eq!(
        after.published_generation,
        Some(published_generation),
        "a hold must not advance the publication generation"
    );
    assert_eq!(
        after.members, before.members,
        "a hold must preserve the prior durable publication"
    );
    assert!(after.dirty, "a held generation must stay queued");
    assert_eq!(
        read_m4_grouping_generation(&observer, SPACE).await,
        held_generation,
        "a hold must preserve the queued grouping generation"
    );
    assert_eq!(
        runtime.published_generation(SPACE),
        Some(published_generation),
        "a hold must restore the prior runtime partition"
    );
    assert!(
        read_m4_lease_generations(&observer, SPACE).await.is_empty(),
        "a hold must consume its durable lease"
    );

    let repeated = run_community_grouping_cycle(&db, &mut runtime, SPACE)
        .await
        .expect("repeat held generation");
    assert!(
        matches!(repeated, CommunityGroupingOutcome::Held(_)),
        "repeated phase firing must hold again, never publish empty: {repeated:?}"
    );
    let repeated_snapshot = read_m4_persisted_snapshot(&observer, SPACE).await;
    assert_eq!(repeated_snapshot, after);
    assert!(
        read_m4_lease_generations(&observer, SPACE).await.is_empty(),
        "a repeated hold must also consume its durable lease"
    );
}

#[tokio::test]
async fn m4_phase_round_robin_reaches_later_dirty_space_after_held_restart() {
    const HELD_SPACE: &str = "a-m4-held-space";
    const VIABLE_SPACE: &str = "z-m4-viable-space";
    const ROOT: &str = "m4-round-robin-root";
    const VIABLE_PARTICIPANTS: usize = 10;

    let dir = tempfile::tempdir().expect("M4 round-robin tempdir");
    let db = MemoryDB::new(dir.path(), Arc::new(NoopEmitter))
        .await
        .expect("open M4 round-robin database");
    let observer_db = libsql::Builder::new_local(dir.path().join("origin_memory.db"))
        .build()
        .await
        .expect("open M4 round-robin observer");
    let observer = observer_db
        .connect()
        .expect("connect M4 round-robin observer");

    observer
        .execute(
            "INSERT INTO provenance_roots
             (root_id, identity_version, identity_digest, root_kind,
              independence_group_id, status, created_at)
             VALUES (?1, 1, 'm4-round-robin-digest',
                     'generated', 'm4-round-robin-group', 'active', 1712707200)",
            libsql::params![ROOT],
        )
        .await
        .expect("seed round-robin provenance root");

    for (space, participants, prefix) in [
        (HELD_SPACE, 2usize, "held"),
        (VIABLE_SPACE, VIABLE_PARTICIPANTS, "viable"),
    ] {
        for node in 0..participants {
            observer
                .execute(
                    "INSERT INTO entities
                     (id, name, entity_type, space, created_at, updated_at)
                     VALUES (?1, ?1, 'concept', ?2, 1712707200, 1712707200)",
                    libsql::params![format!("{prefix}-node-{node:02}"), space],
                )
                .await
                .expect("seed round-robin entity");
        }
        for node in 0..participants {
            observer
                .execute(
                    "INSERT INTO edges
                     (edge_id, src_id, src_kind, dst_id, dst_kind, edge_type, lineage,
                      grounded, root_id, space, created_at)
                     VALUES (?1, ?2, 'entity', ?3, 'entity', 'relates', 'assertion',
                             1, ?4, ?5, 1712707200)",
                    libsql::params![
                        format!("{prefix}-edge-{node:02}"),
                        format!("{prefix}-node-{node:02}"),
                        format!("{prefix}-node-{:02}", (node + 1) % participants),
                        ROOT,
                        space,
                    ],
                )
                .await
                .expect("seed round-robin grounded edge");
        }
    }

    let held_generation = read_m4_grouping_generation(&observer, HELD_SPACE).await;
    let first = db
        .run_next_community_grouping_cycle()
        .await
        .expect("run first round-robin phase")
        .expect("held space must be selected");
    assert!(
        matches!(first, CommunityGroupingOutcome::Held(_)),
        "lexicographically early below-floor space must hold first: {first:?}"
    );
    let held_after_first = read_m4_persisted_snapshot(&observer, HELD_SPACE).await;
    assert_eq!(held_after_first.published_generation, None);
    assert!(held_after_first.dirty);
    assert!(held_after_first.members.is_empty());
    assert_eq!(
        read_m4_grouping_generation(&observer, HELD_SPACE).await,
        held_generation,
        "held queue entry must preserve its grouping generation"
    );
    assert!(
        read_m4_lease_generations(&observer, HELD_SPACE)
            .await
            .is_empty(),
        "held attempt must consume its durable lease"
    );

    drop(observer);
    drop(observer_db);
    drop(db);

    let reopened = MemoryDB::new(dir.path(), Arc::new(NoopEmitter))
        .await
        .expect("reopen M4 round-robin database");
    let observer_db = libsql::Builder::new_local(dir.path().join("origin_memory.db"))
        .build()
        .await
        .expect("reopen M4 round-robin observer");
    let observer = observer_db
        .connect()
        .expect("reconnect M4 round-robin observer");
    let held_generation_after_restart = read_m4_grouping_generation(&observer, HELD_SPACE).await;
    let second = reopened
        .run_next_community_grouping_cycle()
        .await
        .expect("run second round-robin phase after restart")
        .expect("later viable space must remain queued");
    assert!(
        matches!(
            second,
            CommunityGroupingOutcome::Published(ref receipt)
                if receipt.member_count == VIABLE_PARTICIPANTS
        ),
        "durable cursor must advance past the held space after restart: {second:?}"
    );

    let held_after_second = read_m4_persisted_snapshot(&observer, HELD_SPACE).await;
    assert_eq!(held_after_second, held_after_first);
    assert_eq!(
        read_m4_grouping_generation(&observer, HELD_SPACE).await,
        held_generation_after_restart,
        "later publication must not consume or rewrite the held generation"
    );
    let viable_after_second = read_m4_persisted_snapshot(&observer, VIABLE_SPACE).await;
    assert_eq!(
        viable_after_second.members.len(),
        VIABLE_PARTICIPANTS,
        "later viable space must publish its complete membership"
    );
    assert!(viable_after_second.published_generation.is_some());
    assert!(!viable_after_second.dirty);
    assert!(
        read_m4_lease_generations(&observer, HELD_SPACE)
            .await
            .is_empty()
            && read_m4_lease_generations(&observer, VIABLE_SPACE)
                .await
                .is_empty(),
        "neither held nor published attempt may leak a lease"
    );
}

#[tokio::test]
async fn m4_generation_tracks_grounded_retraction_reactivation_and_entity_merge() {
    const SPACE: &str = "m4-correction-space";
    const SOURCE_ID: &str = "m4-correction-source";
    const SOURCE_CONTENT: &str =
        "The correction fixture grounds relations before retracting or merging them.";

    let dir = tempfile::tempdir().expect("M4 correction tempdir");
    let db = MemoryDB::new(dir.path(), Arc::new(NoopEmitter))
        .await
        .expect("open M4 correction database");
    db.upsert_documents(vec![RawDocument {
        source: "memory".to_owned(),
        source_id: SOURCE_ID.to_owned(),
        title: "M4 corrections".to_owned(),
        content: SOURCE_CONTENT.to_owned(),
        last_modified: 1_722_000_000,
        memory_type: Some("fact".to_owned()),
        space: Some(SPACE.to_owned()),
        source_agent: Some("folder".to_owned()),
        confirmed: Some(true),
        ..Default::default()
    }])
    .await
    .expect("seed M4 correction source");
    let root_id = db
        .acquire_provenance_root(
            "document_ingest",
            SOURCE_CONTENT,
            &IndependenceSignals {
                source_identity: Some("file:///m4-corrections.md"),
                agent_turn: None,
                import_batch: None,
            },
        )
        .await
        .expect("acquire correction provenance root");
    let observer_db = libsql::Builder::new_local(dir.path().join("origin_memory.db"))
        .build()
        .await
        .expect("open correction observer");
    let observer = observer_db.connect().expect("connect correction observer");

    let left = db
        .store_entity("Correction Left", "concept", Some(SPACE), None, None)
        .await
        .expect("create left entity");
    let right = db
        .store_entity("Correction Right", "concept", Some(SPACE), None, None)
        .await
        .expect("create right entity");
    let relation_id = db
        .create_relation(
            &left,
            &right,
            "related_to",
            Some("m4-gate"),
            Some(1.0),
            Some("ground before retract"),
            Some(SOURCE_ID),
        )
        .await
        .expect("create relation to retract");
    let edge_id = compute_edge_id("relates", "entity", &left, "entity", &right, "related_to");
    assert_eq!(
        db.promote_edges_grounded(&[EdgePromotion {
            edge_id: edge_id.clone(),
            root_id: root_id.clone(),
            payload: r#"{"grounded":true,"source":"m4-correction"}"#.to_owned(),
            source_memory_id: SOURCE_ID.to_owned(),
            judged_content: SOURCE_CONTENT.to_owned(),
        }])
        .await
        .expect("ground relation before retract"),
        1
    );
    assert_eq!(read_m4_generation(&observer, SPACE).await, 1);

    db.supersede_relation(&relation_id, "replacement-relation")
        .await
        .expect("retract grounded relation");
    assert_eq!(
        read_m4_generation(&observer, SPACE).await,
        2,
        "grounded retraction must advance the space exactly once"
    );

    db.create_relation(
        &left,
        &right,
        "related_to",
        Some("m4-gate"),
        Some(1.0),
        Some("reactivate grounded relation"),
        Some(SOURCE_ID),
    )
    .await
    .expect("reactivate relation");
    assert_eq!(
        read_m4_generation(&observer, SPACE).await,
        3,
        "reactivating an already-grounded edge must advance the space exactly once"
    );

    let canonical = db
        .store_entity("Canonical Entity", "concept", Some(SPACE), None, None)
        .await
        .expect("create canonical entity");
    let alias = db
        .store_entity("Alias Entity", "concept", Some(SPACE), None, None)
        .await
        .expect("create alias entity");
    let neighbor = db
        .store_entity("Merge Neighbor", "concept", Some(SPACE), None, None)
        .await
        .expect("create merge neighbor");
    let merge_relation = db
        .create_relation(
            &alias,
            &neighbor,
            "related_to",
            Some("m4-gate"),
            Some(1.0),
            Some("ground before entity merge"),
            Some(SOURCE_ID),
        )
        .await
        .expect("create merge relation");
    let old_merge_edge = compute_edge_id(
        "relates",
        "entity",
        &alias,
        "entity",
        &neighbor,
        "related_to",
    );
    assert_eq!(
        db.promote_edges_grounded(&[EdgePromotion {
            edge_id: old_merge_edge.clone(),
            root_id,
            payload: r#"{"grounded":true,"source":"m4-merge"}"#.to_owned(),
            source_memory_id: SOURCE_ID.to_owned(),
            judged_content: SOURCE_CONTENT.to_owned(),
        }])
        .await
        .expect("ground relation before merge"),
        1
    );
    assert_eq!(read_m4_generation(&observer, SPACE).await, 4);

    db.merge_entities(&canonical, &alias)
        .await
        .expect("merge grounded relation endpoint");
    assert_eq!(
        read_m4_generation(&observer, SPACE).await,
        5,
        "one entity-merge transaction must advance each affected space exactly once"
    );
    let new_merge_edge = compute_edge_id(
        "relates",
        "entity",
        &canonical,
        "entity",
        &neighbor,
        "related_to",
    );
    assert!(
        read_edge_active_grounded(&observer, &old_merge_edge)
            .await
            .is_none(),
        "the pre-merge grounded edge must leave the active graph"
    );
    assert_eq!(
        read_edge_active_grounded(&observer, &new_merge_edge).await,
        Some((canonical, neighbor)),
        "the corrected endpoint edge must remain active and grounded"
    );
    let merged_grouping_generation = read_m4_grouping_generation(&observer, SPACE).await;

    let below_floor = db
        .run_next_community_grouping_cycle()
        .await
        .expect("run the phase-slot M4 wrapper")
        .expect("the corrected space is dirty");
    assert!(
        matches!(below_floor, CommunityGroupingOutcome::Held(_)),
        "five participants must hold below the authored viability floor: {below_floor:?}"
    );
    let held_snapshot = read_m4_persisted_snapshot(&observer, SPACE).await;
    assert_eq!(held_snapshot.published_generation, None);
    assert!(held_snapshot.dirty);
    assert!(held_snapshot.members.is_empty());
    assert_eq!(
        read_m4_grouping_generation(&observer, SPACE).await,
        merged_grouping_generation
    );
    assert!(
        read_m4_lease_generations(&observer, SPACE).await.is_empty(),
        "a hold must consume its durable lease"
    );

    let restart_left = db
        .store_entity("Restart Left", "concept", Some(SPACE), None, None)
        .await
        .expect("create restart left");
    let restart_right = db
        .store_entity("Restart Right", "concept", Some(SPACE), None, None)
        .await
        .expect("create restart right");
    let restart_relation = db
        .create_relation(
            &restart_left,
            &restart_right,
            "related_to",
            Some("m4-gate"),
            Some(1.0),
            Some("ground before process restart"),
            Some(SOURCE_ID),
        )
        .await
        .expect("create restart relation");
    let restart_edge = compute_edge_id(
        "relates",
        "entity",
        &restart_left,
        "entity",
        &restart_right,
        "related_to",
    );
    assert_eq!(
        db.promote_edges_grounded(&[EdgePromotion {
            edge_id: restart_edge,
            root_id: db
                .acquire_provenance_root(
                    "document_ingest",
                    SOURCE_CONTENT,
                    &IndependenceSignals {
                        source_identity: Some("file:///m4-corrections.md"),
                        agent_turn: None,
                        import_batch: None,
                    },
                )
                .await
                .expect("reacquire restart root"),
            payload: r#"{"grounded":true,"source":"m4-restart"}"#.to_owned(),
            source_memory_id: SOURCE_ID.to_owned(),
            judged_content: SOURCE_CONTENT.to_owned(),
        }])
        .await
        .expect("ground restart correction"),
        1
    );
    assert_eq!(read_m4_generation(&observer, SPACE).await, 6);
    let restart_grouping_generation = read_m4_grouping_generation(&observer, SPACE).await;
    drop(observer);
    drop(observer_db);
    drop(db);

    let reopened = MemoryDB::new(dir.path(), Arc::new(NoopEmitter))
        .await
        .expect("reopen after simulated process restart");
    let restarted = reopened
        .run_next_community_grouping_cycle()
        .await
        .expect("restart routes missing volatile state to full partition")
        .expect("durable dirty generation survives restart");
    let restarted_generation = match restarted {
        CommunityGroupingOutcome::Held(receipt) => {
            assert!(
                receipt.input_generation >= restart_grouping_generation,
                "startup embedding recovery may advance the broader grouping input generation"
            );
            assert_eq!(receipt.projected_edge_count, 3);
            receipt.input_generation
        }
        other => panic!("restart must preserve the below-floor hold: {other:?}"),
    };

    let observer_db = libsql::Builder::new_local(dir.path().join("origin_memory.db"))
        .build()
        .await
        .expect("reopen correction observer");
    let observer = observer_db
        .connect()
        .expect("reconnect correction observer");
    assert_eq!(
        read_m4_grouping_generation(&observer, SPACE).await,
        restarted_generation,
        "restart must keep the exact durable grouping input queued"
    );
    let restarted_snapshot = read_m4_persisted_snapshot(&observer, SPACE).await;
    assert_eq!(restarted_snapshot.published_generation, None);
    assert!(restarted_snapshot.dirty);
    assert!(restarted_snapshot.members.is_empty());
    assert!(
        read_m4_lease_generations(&observer, SPACE).await.is_empty(),
        "a restart hold must consume its durable lease"
    );

    const DELETE_SOURCE: &str = "m4-delete-cascade-source";
    const DELETE_CONTENT: &str =
        "Deleting this source must retract its grounded relation edge atomically.";
    reopened
        .upsert_documents(vec![RawDocument {
            source: "memory".to_owned(),
            source_id: DELETE_SOURCE.to_owned(),
            title: "M4 delete cascade".to_owned(),
            content: DELETE_CONTENT.to_owned(),
            last_modified: 1_722_000_001,
            memory_type: Some("fact".to_owned()),
            space: Some(SPACE.to_owned()),
            source_agent: Some("folder".to_owned()),
            confirmed: Some(true),
            ..Default::default()
        }])
        .await
        .expect("seed deletion-cascade source");
    let delete_left = reopened
        .store_entity("Delete Cascade Left", "concept", Some(SPACE), None, None)
        .await
        .expect("create delete-cascade left");
    let delete_right = reopened
        .store_entity("Delete Cascade Right", "concept", Some(SPACE), None, None)
        .await
        .expect("create delete-cascade right");
    let delete_relation = reopened
        .create_relation(
            &delete_left,
            &delete_right,
            "related_to",
            Some("m4-gate"),
            Some(1.0),
            Some("ground before deleting source evidence"),
            Some(DELETE_SOURCE),
        )
        .await
        .expect("create delete-cascade relation");
    let delete_edge = compute_edge_id(
        "relates",
        "entity",
        &delete_left,
        "entity",
        &delete_right,
        "related_to",
    );
    let delete_root = reopened
        .acquire_provenance_root(
            "document_ingest",
            DELETE_CONTENT,
            &IndependenceSignals {
                source_identity: Some("file:///m4-delete-cascade.md"),
                agent_turn: None,
                import_batch: None,
            },
        )
        .await
        .expect("acquire delete-cascade root");
    assert_eq!(
        reopened
            .promote_edges_grounded(&[EdgePromotion {
                edge_id: delete_edge.clone(),
                root_id: delete_root,
                payload: r#"{"grounded":true,"source":"m4-delete"}"#.to_owned(),
                source_memory_id: DELETE_SOURCE.to_owned(),
                judged_content: DELETE_CONTENT.to_owned(),
            }])
            .await
            .expect("ground delete-cascade relation"),
        1
    );
    assert_eq!(read_m4_generation(&observer, SPACE).await, 7);
    reopened
        .delete_by_source_id("memory", DELETE_SOURCE)
        .await
        .expect("delete grounded relation source");
    assert_eq!(
        read_m4_generation(&observer, SPACE).await,
        8,
        "source deletion must advance the grounded graph generation exactly once"
    );
    assert!(
        matches!(
            read_edge_valid_until(&observer, &delete_edge).await,
            Some(Some(_))
        ),
        "source deletion must soft-invalidate the grounded edge, never leave or hard-delete it"
    );

    const REFRESH_SOURCE: &str = "m4-refresh-cascade-source";
    const REFRESH_CONTENT: &str =
        "Refreshing this source must retract the grounded projection built from old evidence.";
    reopened
        .upsert_documents(vec![RawDocument {
            source: "memory".to_owned(),
            source_id: REFRESH_SOURCE.to_owned(),
            title: "M4 refresh cascade".to_owned(),
            content: REFRESH_CONTENT.to_owned(),
            last_modified: 1_722_000_002,
            memory_type: Some("fact".to_owned()),
            space: Some(SPACE.to_owned()),
            source_agent: Some("folder".to_owned()),
            confirmed: Some(true),
            ..Default::default()
        }])
        .await
        .expect("seed refresh-cascade source");
    let refresh_left = reopened
        .store_entity("Refresh Cascade Left", "concept", Some(SPACE), None, None)
        .await
        .expect("create refresh-cascade left");
    let refresh_right = reopened
        .store_entity("Refresh Cascade Right", "concept", Some(SPACE), None, None)
        .await
        .expect("create refresh-cascade right");
    let refresh_relation = reopened
        .create_relation(
            &refresh_left,
            &refresh_right,
            "related_to",
            Some("m4-gate"),
            Some(1.0),
            Some("ground before refreshing source evidence"),
            Some(REFRESH_SOURCE),
        )
        .await
        .expect("create refresh-cascade relation");
    let refresh_edge = compute_edge_id(
        "relates",
        "entity",
        &refresh_left,
        "entity",
        &refresh_right,
        "related_to",
    );
    let refresh_root = reopened
        .acquire_provenance_root(
            "document_ingest",
            REFRESH_CONTENT,
            &IndependenceSignals {
                source_identity: Some("file:///m4-refresh-cascade.md"),
                agent_turn: None,
                import_batch: None,
            },
        )
        .await
        .expect("acquire refresh-cascade root");
    assert_eq!(
        reopened
            .promote_edges_grounded(&[EdgePromotion {
                edge_id: refresh_edge.clone(),
                root_id: refresh_root,
                payload: r#"{"grounded":true,"source":"m4-refresh"}"#.to_owned(),
                source_memory_id: REFRESH_SOURCE.to_owned(),
                judged_content: REFRESH_CONTENT.to_owned(),
            }])
            .await
            .expect("ground refresh-cascade relation"),
        1
    );
    assert_eq!(read_m4_generation(&observer, SPACE).await, 9);
    reopened
        .update_memory(
            REFRESH_SOURCE,
            "The evidence changed, so the old grounded projection is no longer valid.",
        )
        .await
        .expect("refresh grounded relation source");
    assert_eq!(
        read_m4_generation(&observer, SPACE).await,
        10,
        "source refresh must advance the grounded graph generation exactly once"
    );
    assert!(
        matches!(
            read_edge_valid_until(&observer, &refresh_edge).await,
            Some(Some(_))
        ),
        "source refresh must soft-invalidate the grounded edge"
    );
}

#[tokio::test]
async fn m4_merge_collision_advances_graph_and_rebuilds_changed_node_order() {
    const SPACE: &str = "m4-merge-collision-space";
    const SOURCE_ID: &str = "m4-merge-collision-source";
    const SOURCE_CONTENT: &str =
        "A grounded alias edge collapses onto an already-grounded canonical edge.";
    const CORE_NODES: usize = 10;

    let dir = tempfile::tempdir().expect("M4 merge-collision tempdir");
    let db = MemoryDB::new(dir.path(), Arc::new(NoopEmitter))
        .await
        .expect("open M4 merge-collision database");
    db.upsert_documents(vec![RawDocument {
        source: "memory".to_owned(),
        source_id: SOURCE_ID.to_owned(),
        title: "M4 merge collision".to_owned(),
        content: SOURCE_CONTENT.to_owned(),
        last_modified: 1_722_000_000,
        memory_type: Some("fact".to_owned()),
        space: Some(SPACE.to_owned()),
        source_agent: Some("folder".to_owned()),
        confirmed: Some(true),
        ..Default::default()
    }])
    .await
    .expect("seed M4 merge-collision source");
    let root_id = db
        .acquire_provenance_root(
            "document_ingest",
            SOURCE_CONTENT,
            &IndependenceSignals {
                source_identity: Some("file:///m4-merge-collision.md"),
                agent_turn: None,
                import_batch: None,
            },
        )
        .await
        .expect("acquire M4 merge-collision root");
    let observer_db = libsql::Builder::new_local(dir.path().join("origin_memory.db"))
        .build()
        .await
        .expect("open M4 merge-collision observer");
    let observer = observer_db
        .connect()
        .expect("connect M4 merge-collision observer");

    for node in 0..CORE_NODES {
        observer
            .execute(
                "INSERT INTO entities
                    (id, name, entity_type, space, created_at, updated_at)
                 VALUES (?1, ?1, 'concept', ?2, 1712707200, 1712707200)",
                libsql::params![format!("merge-core-{node:02}"), SPACE],
            )
            .await
            .expect("seed merge-collision core entity");
    }
    observer
        .execute(
            "INSERT INTO entities
                (id, name, entity_type, space, created_at, updated_at)
             VALUES ('merge-alias', 'merge-alias', 'concept', ?1, 1712707200, 1712707200)",
            libsql::params![SPACE],
        )
        .await
        .expect("seed merge-collision alias");
    for node in 0..CORE_NODES {
        observer
            .execute(
                "INSERT INTO edges
                    (edge_id, src_id, src_kind, dst_id, dst_kind, edge_type, lineage,
                     grounded, root_id, space, created_at)
                 VALUES (?1, ?2, 'entity', ?3, 'entity', 'relates', 'assertion',
                         1, ?4, ?5, 1712707200)",
                libsql::params![
                    format!("merge-ring-{node:02}"),
                    format!("merge-core-{node:02}"),
                    format!("merge-core-{:02}", (node + 1) % CORE_NODES),
                    root_id.as_str(),
                    SPACE
                ],
            )
            .await
            .expect("seed merge-collision core ring");
    }

    let canonical_promotion = stage_m4_promotion(
        &db,
        "merge-core-00",
        "merge-core-01",
        SOURCE_ID,
        SOURCE_CONTENT,
        &root_id,
    )
    .await;
    let canonical_relation_id = canonical_promotion.edge_id.clone();
    assert_eq!(
        db.promote_edges_grounded(&[canonical_promotion])
            .await
            .expect("ground canonical collision edge before superseding"),
        1
    );
    db.supersede_relation(&canonical_relation_id, "merge-canonical-superseder")
        .await
        .expect("supersede canonical collision relation");

    let alias_promotion = stage_m4_promotion(
        &db,
        "merge-alias",
        "merge-core-01",
        SOURCE_ID,
        SOURCE_CONTENT,
        &root_id,
    )
    .await;
    assert_eq!(
        db.promote_edges_grounded(&[alias_promotion])
            .await
            .expect("ground alias collision edge"),
        1
    );

    let first = db
        .run_next_community_grouping_cycle()
        .await
        .expect("publish merge-collision baseline")
        .expect("merge-collision baseline work");
    assert!(matches!(
        first,
        CommunityGroupingOutcome::Published(ref receipt)
            if receipt.member_count == CORE_NODES + 1
    ));
    let graph_before = read_m4_generation(&observer, SPACE).await;

    db.merge_entities("merge-core-00", "merge-alias")
        .await
        .expect("merge alias onto existing grounded canonical edge");
    assert_eq!(
        read_m4_generation(&observer, SPACE).await,
        graph_before + 1,
        "colliding merge must retain the real old-edge retraction bump"
    );

    let merged = db
        .run_next_community_grouping_cycle()
        .await
        .expect("publish merge-collision correction")
        .expect("merge-collision correction work");
    assert!(
        matches!(
            merged,
            CommunityGroupingOutcome::Published(ref receipt)
                if receipt.member_count == CORE_NODES
                    && receipt.compute_mode
                        == (CommunityGroupingComputeMode::Full {
                            reason: CommunityGroupingFullReason::NodeOrderChanged,
                        })
        ),
        "node removal must rebuild instead of reusing stale membership: {merged:?}"
    );
    let snapshot = read_m4_persisted_snapshot(&observer, SPACE).await;
    assert!(
        snapshot
            .members
            .iter()
            .all(|member| member.node_id != "merge-alias"),
        "published members must not retain the merged-away alias"
    );
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PersistedM4Member {
    node_id: String,
    community_id: String,
    published_generation: i64,
    attachment: String,
}

#[derive(Debug, PartialEq, Eq)]
struct PersistedM4Snapshot {
    graph_generation: i64,
    published_generation: Option<i64>,
    dirty: bool,
    members: Vec<PersistedM4Member>,
}

async fn stage_m4_promotion(
    db: &MemoryDB,
    from: &str,
    to: &str,
    source_memory_id: &str,
    judged_content: &str,
    root_id: &str,
) -> EdgePromotion {
    db.create_relation(
        from,
        to,
        "related_to",
        Some("m4-gate"),
        Some(1.0),
        Some("M4 real-job acceptance edge"),
        Some(source_memory_id),
    )
    .await
    .expect("stage grounded=0 relation");
    EdgePromotion {
        edge_id: compute_edge_id("relates", "entity", from, "entity", to, "related_to"),
        root_id: root_id.to_owned(),
        payload: r#"{"grounded":true,"score":1.0,"source":"m4-gate"}"#.to_owned(),
        source_memory_id: source_memory_id.to_owned(),
        judged_content: judged_content.to_owned(),
    }
}

fn m4_scale_node(space: usize, node: usize) -> String {
    format!("space-{space}-cluster-00-node-{node:03}")
}

fn m4_scale_chord_pairs(count: usize) -> Vec<(String, String)> {
    const NODES_PER_CLUSTER: usize = 256;
    let mut pairs = Vec::with_capacity(count);
    for left in 0..NODES_PER_CLUSTER {
        for right in (left + 1)..NODES_PER_CLUSTER {
            let distance = (right - left).min(NODES_PER_CLUSTER - (right - left));
            if distance > 3 {
                pairs.push((m4_scale_node(0, left), m4_scale_node(0, right)));
                if pairs.len() == count {
                    return pairs;
                }
            }
        }
    }
    panic!("requested {count} non-base chords");
}

async fn seed_m4_graded_corpus(observer: &libsql::Connection) {
    const MEMORIES: usize = 100_000;
    const PAGES: usize = 5_000;
    const SPACES: usize = 3;
    const CLUSTERS_PER_SPACE: usize = 8;
    const NODES_PER_CLUSTER: usize = 256;

    observer
        .execute("BEGIN", ())
        .await
        .expect("begin M4 corpus");
    for index in 0..MEMORIES {
        observer
            .execute(
                "INSERT INTO memories (id, content, source, source_id, title, chunk_index, \
                    last_modified, chunk_type, source_agent, space, confidence, confirmed, \
                    memory_type, pending_revision) \
                 VALUES (?1, 'c', 'memory', ?1, 't', 0, 1712707200, 'text', NULL, ?2, \
                    1.0, 0, 'fact', 0)",
                libsql::params![
                    format!("m4-scale-memory-{index}"),
                    format!("space-{}", index % SPACES)
                ],
            )
            .await
            .expect("seed M4 scale memory");
    }
    for page in 0..PAGES {
        let space = format!("space-{}", page % SPACES);
        observer
            .execute(
                "INSERT INTO pages (id, title, content, created_at, last_compiled, \
                    last_modified, space, workspace) \
                 VALUES (?1, 't', 'c', '2024-01-01T00:00:00Z', \
                    '2024-01-01T00:00:00Z', '2024-01-01T00:00:00Z', ?2, ?2)",
                libsql::params![format!("m4-scale-page-{page}"), space],
            )
            .await
            .expect("seed M4 scale page");
    }
    for space in 0..SPACES {
        observer
            .execute(
                "INSERT INTO provenance_roots (root_id, identity_version, identity_digest, \
                    root_kind, independence_group_id, status, created_at) \
                 VALUES (?1, 1, ?2, 'generated', ?3, 'active', 1712707200)",
                libsql::params![
                    format!("m4-scale-root-{space}"),
                    format!("m4-scale-root-digest-{space}"),
                    format!("m4-scale-space-{space}")
                ],
            )
            .await
            .expect("seed M4 scale root");
        for cluster in 0..CLUSTERS_PER_SPACE {
            for node in 0..NODES_PER_CLUSTER {
                let entity_id = format!("space-{space}-cluster-{cluster:02}-node-{node:03}");
                observer
                    .execute(
                        "INSERT INTO entities \
                         (id, name, entity_type, space, created_at, updated_at) \
                         VALUES (?1, ?1, 'concept', ?2, 1712707200, 1712707200)",
                        libsql::params![entity_id, format!("space-{space}")],
                    )
                    .await
                    .expect("seed M4 scale entity");
            }
        }

        let mut edge_index = 0usize;
        for cluster in 0..CLUSTERS_PER_SPACE {
            for node in 0..NODES_PER_CLUSTER {
                for offset in 1..=3 {
                    observer
                        .execute(
                            "INSERT INTO edges \
                             (edge_id, src_id, src_kind, dst_id, dst_kind, edge_type, lineage, \
                                grounded, root_id, space, created_at) \
                             VALUES (?1, ?2, 'entity', ?3, 'entity', 'relates', 'assertion', \
                                1, ?4, ?5, 1712707200)",
                            libsql::params![
                                format!("m4-scale-space-{space}-edge-{edge_index:06}"),
                                format!("space-{space}-cluster-{cluster:02}-node-{node:03}"),
                                format!(
                                    "space-{space}-cluster-{cluster:02}-node-{:03}",
                                    (node + offset) % NODES_PER_CLUSTER
                                ),
                                format!("m4-scale-root-{space}"),
                                format!("space-{space}")
                            ],
                        )
                        .await
                        .expect("seed M4 scale edge");
                    edge_index += 1;
                }
            }
            if cluster + 1 < CLUSTERS_PER_SPACE {
                observer
                    .execute(
                        "INSERT INTO edges \
                         (edge_id, src_id, src_kind, dst_id, dst_kind, edge_type, lineage, \
                            grounded, root_id, space, created_at) \
                         VALUES (?1, ?2, 'entity', ?3, 'entity', 'relates', 'assertion', \
                            1, ?4, ?5, 1712707200)",
                        libsql::params![
                            format!("m4-scale-space-{space}-edge-{edge_index:06}"),
                            format!("space-{space}-cluster-{cluster:02}-node-000"),
                            format!("space-{space}-cluster-{:02}-node-000", cluster + 1),
                            format!("m4-scale-root-{space}"),
                            format!("space-{space}")
                        ],
                    )
                    .await
                    .expect("seed M4 scale bridge edge");
                edge_index += 1;
            }
        }
        assert_eq!(edge_index, 6_151);

        for page in (space..PAGES).step_by(SPACES) {
            let cluster = (page / SPACES) % CLUSTERS_PER_SPACE;
            let node = (page / (SPACES * CLUSTERS_PER_SPACE)) % NODES_PER_CLUSTER;
            let page_entity_id = format!("space-{space}-cluster-{cluster:02}-node-{node:03}");
            let memory_entity_id = format!(
                "space-{space}-cluster-{cluster:02}-node-{:03}",
                (node + 1) % NODES_PER_CLUSTER
            );
            let page_id = format!("m4-scale-page-{page}");
            let memory_id = format!("m4-scale-memory-{page}");
            observer
                .execute(
                    "UPDATE pages SET entity_id=?1 WHERE id=?2",
                    libsql::params![page_entity_id, page_id.clone()],
                )
                .await
                .expect("attach M4 scale page entity");
            observer
                .execute(
                    "UPDATE memories SET entity_id=?1 WHERE id=?2",
                    libsql::params![memory_entity_id, memory_id.clone()],
                )
                .await
                .expect("attach M4 scale memory entity");
            observer
                .execute(
                    "INSERT INTO edges
                        (edge_id, src_id, src_kind, dst_id, dst_kind, edge_type,
                         lineage, grounded, root_id, space, created_at)
                     VALUES (?1, ?2, 'page', ?3, 'memory', 'cites',
                             'evidence', 0, NULL, ?4, 1712707200)",
                    libsql::params![
                        format!("m4-scale-cite-{page}"),
                        page_id,
                        memory_id,
                        format!("space-{space}")
                    ],
                )
                .await
                .expect("seed M4 native page citation");
        }
    }
    observer
        .execute("COMMIT", ())
        .await
        .expect("commit M4 graded corpus");
    observer
        .execute("ANALYZE edges", ())
        .await
        .expect("analyze M4 edge corpus");
}

async fn run_m4_cycle_with_foreground_probe(
    db: &Arc<MemoryDB>,
    runtime: &mut CommunityGroupingRuntime,
    space: &str,
) -> (
    Result<CommunityGroupingOutcome, CommunityGroupingError>,
    Duration,
) {
    let probe_db = Arc::clone(db);
    let foreground = tokio::spawn(async move {
        tokio::task::yield_now().await;
        let started = Instant::now();
        probe_db
            .get_memory_count()
            .await
            .expect("foreground memory-count query");
        started.elapsed()
    });
    let grouping = run_community_grouping_cycle(db, runtime, space).await;
    let foreground = foreground.await.expect("foreground probe task");
    (grouping, foreground)
}

fn assert_m4_job_phase_structure() {
    let db_source = include_str!("../src/db.rs");
    for signature in [
        "pub async fn prepare_community_grouping",
        "pub async fn finalize_community_grouping",
    ] {
        let body = rust_function_body(db_source, signature);
        for forbidden in [
            "project_grounded_relates(",
            "full_partition(",
            "incremental_partition(",
            "rebind_durable_ids(",
            "compute_community_grouping(",
            "get_or_compute_embedding(",
        ] {
            assert!(
                !body.contains(forbidden),
                "{signature} must not run compute inside its DB phase: found {forbidden}"
            );
        }
    }
    for signature in [
        "async fn acquire_community_grouping_lease",
        "pub async fn finalize_community_grouping",
    ] {
        let body = rust_function_body(db_source, signature);
        assert!(
            body.contains("transaction_with_behavior(libsql::TransactionBehavior::Immediate)"),
            "{signature} must use the pinned libSQL RAII transaction API"
        );
        assert!(
            !body.contains("execute(\"BEGIN\"")
                && !body.contains("execute(\"COMMIT\"")
                && !body.contains("execute(\"ROLLBACK\""),
            "{signature} must not leave a manual transaction open when its future is dropped"
        );
    }

    let lease_source = include_str!("../src/db/community_grouping_state.rs");
    assert!(
        lease_source.contains("tokio::runtime::Handle::try_current()"),
        "attempt-drop lease cleanup must fall back to expiry instead of panicking without a runtime"
    );
    let job_source = include_str!("../src/community_grouping.rs");
    let composer = rust_function_body(job_source, "pub async fn run_community_grouping_cycle");
    let prepare = composer
        .find("prepare_community_grouping(")
        .expect("composer prepare phase");
    let compute = composer
        .find("compute_community_grouping(")
        .expect("composer pure compute phase");
    let finalize = composer
        .find("work.finalize(")
        .expect("composer work finalization phase");
    assert!(
        prepare < compute && compute < finalize,
        "production composer must preserve prepare -> pure compute -> finalize ordering"
    );
    let phase_slot =
        rust_function_body(job_source, "pub async fn run_next_community_grouping_cycle");
    assert!(
        !phase_slot.contains(".clone()"),
        "the production phase slot must move only the selected space, not clone graph-wide runtime"
    );
    assert_eq!(
        phase_slot.matches("prepare_community_grouping(").count(),
        1,
        "the phase slot must win the durable lease before touching volatile runtime state"
    );
    let prepare = phase_slot
        .find("prepare_community_grouping(")
        .expect("phase-slot lease acquisition");
    let runtime_claim = phase_slot
        .find("SharedRuntimeSpaceGuard::take(")
        .expect("phase-slot runtime claim");
    assert!(
        prepare < runtime_claim,
        "same-space contenders must acquire the lease before one removes runtime state"
    );
    assert!(
        !job_source.contains("community_degrees: self.community_degrees.clone()"),
        "the rollback journal must not clone graph-wide community statistics"
    );
    let partition_source = include_str!("../src/community_partition.rs");
    let recoverable = rust_function_body(
        partition_source,
        "pub(crate) fn incremental_partition_recoverable",
    );
    let threshold = recoverable
        .find("incremental_frontier(")
        .expect("incremental threshold/frontier validation");
    let journal = recoverable
        .find("rollback_snapshot(")
        .expect("bounded rollback journal allocation");
    assert!(
        threshold < journal,
        "dirty/frontier threshold rejection must happen before rollback-journal allocation"
    );
    let snapshot = rust_function_body(partition_source, "fn rollback_snapshot");
    assert!(
        snapshot.contains("self.adjacency[node]") && snapshot.contains("graph.adjacency[node]"),
        "the bounded journal must cover every old/new adjacency community candidate"
    );

    let refinery_source = include_str!("../src/refinery/mod.rs");
    let legacy = refinery_source
        .find("let legacy_count = db_ref.detect_communities().await?")
        .expect("legacy rollback producer stays live");
    let flag = refinery_source
        .find("crate::db::community_leiden_enabled()")
        .expect("default-off M4 phase gate");
    let shadow = refinery_source
        .find(".run_next_community_grouping_cycle()")
        .expect("M4 job uses the existing CommunityDetection phase");
    assert!(
        legacy < flag && flag < shadow,
        "PR-1 phase wiring must keep legacy live, then gate the write-only M4 shadow"
    );
}

#[test]
fn m4_prepare_reads_are_keyset_paged_and_lock_scoped_per_page() {
    let source = include_str!("../src/db.rs");
    let prepare = rust_function_body(source, "pub async fn prepare_community_grouping");
    for helper in [
        "load_community_grounded_edges_paged(",
        "load_community_ungrounded_edges_paged(",
        "load_community_entities_paged(",
        "load_previous_community_members_paged(",
    ] {
        assert!(
            prepare.contains(helper),
            "prepare must route its potentially large {helper} input through a paged loader"
        );
    }
    assert!(
        !prepare.contains("while let Some(row)"),
        "prepare must not hold one connection guard while draining an unbounded row stream"
    );

    for signature in [
        "async fn load_community_grounded_edges_paged",
        "async fn load_community_ungrounded_edges_paged",
        "async fn load_community_entities_paged",
        "async fn load_previous_community_members_paged",
    ] {
        let body = rust_function_body(source, signature);
        assert!(
            body.contains("COMMUNITY_READ_PAGE_SIZE"),
            "{signature} must use the authored fixed page bound"
        );
        assert!(
            body.contains("ORDER BY") && body.contains("LIMIT"),
            "{signature} must use deterministic keyset pages"
        );
        assert!(
            body.contains("self.conn.lock().await"),
            "{signature} must acquire and release the DB mutex one page at a time"
        );
    }
}

fn rust_function_body<'a>(source: &'a str, signature: &str) -> &'a str {
    let signature_start = source
        .find(signature)
        .unwrap_or_else(|| panic!("missing structural function {signature}"));
    let body_start = source[signature_start..]
        .find('{')
        .map(|offset| signature_start + offset)
        .unwrap_or_else(|| panic!("missing body for {signature}"));
    let mut depth = 0usize;
    for (offset, byte) in source.as_bytes()[body_start..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[body_start..=body_start + offset];
                }
            }
            _ => {}
        }
    }
    panic!("unterminated body for {signature}");
}

async fn read_m4_persisted_snapshot(
    observer: &libsql::Connection,
    space: &str,
) -> PersistedM4Snapshot {
    let mut state_rows = observer
        .query(
            "SELECT graph_generation, published_generation, dirty \
             FROM space_graph_state WHERE space = ?1",
            libsql::params![space.to_owned()],
        )
        .await
        .expect("query M4 graph state");
    let state = state_rows
        .next()
        .await
        .expect("read M4 graph state")
        .expect("M4 graph state row");
    let graph_generation = state.get::<i64>(0).expect("graph generation");
    let published_generation = state.get::<Option<i64>>(1).expect("published generation");
    let dirty = state.get::<i64>(2).expect("dirty") != 0;
    drop(state_rows);

    let mut rows = observer
        .query(
            "SELECT node_id, community_id, published_generation, attachment \
             FROM community_members WHERE space = ?1 ORDER BY node_id",
            libsql::params![space.to_owned()],
        )
        .await
        .expect("query durable community members");
    let mut members = Vec::new();
    while let Some(row) = rows.next().await.expect("read durable community member") {
        members.push(PersistedM4Member {
            node_id: row.get(0).expect("member node id"),
            community_id: row.get(1).expect("member community id"),
            published_generation: row.get(2).expect("member generation"),
            attachment: row.get(3).expect("member attachment"),
        });
    }
    PersistedM4Snapshot {
        graph_generation,
        published_generation,
        dirty,
        members,
    }
}

async fn assert_m4_schema(observer: &libsql::Connection) {
    let mut version_rows = observer
        .query("PRAGMA user_version", ())
        .await
        .expect("query M4 schema version");
    assert_eq!(
        version_rows
            .next()
            .await
            .expect("read M4 schema version")
            .expect("M4 schema version row")
            .get::<i64>(0)
            .expect("decode M4 schema version"),
        i64::from(wenlan_core::db::SCHEMA_VERSION)
    );

    for (table, required_columns) in [
        (
            "communities",
            &[
                "community_id",
                "space",
                "algo_version",
                "projection_version",
                "retired_at",
            ][..],
        ),
        (
            "community_members",
            &[
                "space",
                "node_id",
                "community_id",
                "published_generation",
                "attachment",
            ][..],
        ),
        (
            "space_graph_state",
            &[
                "space",
                "graph_generation",
                "grouping_generation",
                "published_generation",
                "dirty",
            ][..],
        ),
        (
            "grouping_leases",
            &[
                "phase",
                "space",
                "input_generation",
                "token",
                "expires_at",
                "attempt",
            ][..],
        ),
    ] {
        let mut rows = observer
            .query(&format!("PRAGMA table_info({table})"), ())
            .await
            .unwrap_or_else(|error| panic!("query {table} schema: {error}"));
        let mut actual = BTreeSet::new();
        while let Some(row) = rows
            .next()
            .await
            .unwrap_or_else(|error| panic!("read {table} schema: {error}"))
        {
            actual.insert(row.get::<String>(1).expect("decode M4 column"));
        }
        for column in required_columns {
            assert!(
                actual.contains(*column),
                "migration 95 {table} is missing {column}: {actual:?}"
            );
        }
    }
}

async fn read_legacy_community_shadow(
    observer: &libsql::Connection,
    space: &str,
) -> Vec<(String, Option<i64>)> {
    let mut rows = observer
        .query(
            "SELECT id, community_id FROM entities WHERE space = ?1 ORDER BY id",
            libsql::params![space.to_owned()],
        )
        .await
        .expect("query legacy community shadow");
    let mut shadow = Vec::new();
    while let Some(row) = rows.next().await.expect("read legacy community shadow") {
        shadow.push((
            row.get(0).expect("legacy entity id"),
            row.get(1).expect("legacy community id"),
        ));
    }
    shadow
}

async fn seed_obsolete_m4_leases(
    observer: &libsql::Connection,
    space: &str,
    current_generation: i64,
) {
    observer
        .execute(
            "INSERT INTO grouping_leases \
             (phase, space, input_generation, token, expires_at, attempt) VALUES \
             ('community', ?1, 1, 'old-generation', unixepoch() + 3600, 1), \
             ('community', ?1, ?2, 'expired-current', unixepoch() - 1, 1)",
            libsql::params![space.to_owned(), current_generation],
        )
        .await
        .expect("seed expired and old-generation phase leases");
}

async fn read_m4_lease_generations(observer: &libsql::Connection, space: &str) -> Vec<i64> {
    let mut rows = observer
        .query(
            "SELECT input_generation FROM grouping_leases \
             WHERE phase = 'community' AND space = ?1 ORDER BY input_generation",
            libsql::params![space.to_owned()],
        )
        .await
        .expect("query M4 phase leases");
    let mut generations = Vec::new();
    while let Some(row) = rows.next().await.expect("read M4 phase lease") {
        generations.push(row.get(0).expect("lease generation"));
    }
    generations
}

async fn read_m4_generation(observer: &libsql::Connection, space: &str) -> i64 {
    let mut rows = observer
        .query(
            "SELECT graph_generation FROM space_graph_state WHERE space = ?1",
            libsql::params![space.to_owned()],
        )
        .await
        .expect("query M4 generation");
    rows.next()
        .await
        .expect("read M4 generation")
        .expect("M4 generation row")
        .get(0)
        .expect("decode M4 generation")
}

async fn read_m4_grouping_generation(observer: &libsql::Connection, space: &str) -> i64 {
    let mut rows = observer
        .query(
            "SELECT grouping_generation FROM space_graph_state WHERE space = ?1",
            libsql::params![space.to_owned()],
        )
        .await
        .expect("query M4 grouping generation");
    rows.next()
        .await
        .expect("read M4 grouping generation")
        .expect("M4 grouping generation row")
        .get(0)
        .expect("decode M4 grouping generation")
}

async fn read_edge_active_grounded(
    observer: &libsql::Connection,
    edge_id: &str,
) -> Option<(String, String)> {
    let mut rows = observer
        .query(
            "SELECT src_id, dst_id FROM edges \
             WHERE edge_id = ?1 AND grounded = 1 AND valid_until IS NULL",
            libsql::params![edge_id.to_owned()],
        )
        .await
        .expect("query active grounded edge");
    rows.next()
        .await
        .expect("read active grounded edge")
        .map(|row| {
            (
                row.get(0).expect("decode grounded source"),
                row.get(1).expect("decode grounded destination"),
            )
        })
}

async fn read_edge_valid_until(
    observer: &libsql::Connection,
    edge_id: &str,
) -> Option<Option<i64>> {
    let mut rows = observer
        .query(
            "SELECT valid_until FROM edges WHERE edge_id = ?1",
            libsql::params![edge_id.to_owned()],
        )
        .await
        .expect("query edge validity");
    rows.next()
        .await
        .expect("read edge validity")
        .map(|row| row.get(0).expect("decode edge validity"))
}

fn m4_unit_embedding(axis: usize) -> String {
    let mut values = vec!["0"; 768];
    values[axis] = "1";
    format!("[{}]", values.join(","))
}

struct ChurnSeries {
    durations: Vec<Duration>,
    churn: Vec<f64>,
    poison_fraction: f64,
    min_modularity_delta: f64,
    max_frontier_fraction: f64,
}

fn incremental_churn_series(
    mut edges: Vec<ProjectionInputEdge>,
    prefix: &str,
    cycles: usize,
) -> ChurnSeries {
    let base_edge_count = edges.len();
    let initial_graph = project_grounded_relates(&edges, ProjectionConfig::default());
    let initial =
        full_partition(&initial_graph, PartitionConfig::default()).expect("initial partition");
    let mut membership = initial.membership().to_vec();
    let mut state = IncrementalPartitionState::new(&initial_graph, &membership)
        .expect("initial incremental state");
    let mut durable = membership
        .iter()
        .map(|community| format!("community-{community}"))
        .collect::<Vec<_>>();
    let mut durations = Vec::with_capacity(cycles);
    let mut churn = Vec::with_capacity(cycles);
    let mut min_modularity_delta = f64::INFINITY;
    let mut max_frontier_fraction = 0.0f64;
    let mut basin_diagnosed = false;

    for cycle in 0..cycles {
        let cluster = cycle % 8;
        let src_node = (cycle * 5) % 256;
        let dst_node = (src_node + 11 + cycle / 8) % 256;
        let src_id = format!("space-0-cluster-{cluster:02}-node-{src_node:03}");
        let dst_id = format!("space-0-cluster-{cluster:02}-node-{dst_node:03}");
        edges.push(ProjectionInputEdge::new(
            format!("{prefix}-incremental-{cycle:03}"),
            &src_id,
            &dst_id,
        ));
        let changed = project_grounded_relates(&edges, ProjectionConfig::default());
        assert_eq!(
            changed.node_ids(),
            initial_graph.node_ids(),
            "incremental projection must preserve stable node indices"
        );
        let dirty = [
            changed
                .node_ids()
                .binary_search(&src_id)
                .expect("src index"),
            changed
                .node_ids()
                .binary_search(&dst_id)
                .expect("dst index"),
        ];
        let carry_forward_q = modularity(&changed, &membership);

        let started = Instant::now();
        let incremental =
            incremental_partition(&changed, state, &dirty, IncrementalConfig::default())
                .expect("frontier partition");
        durations.push(started.elapsed());
        max_frontier_fraction = max_frontier_fraction
            .max(incremental.optimized_nodes().len() as f64 / changed.node_ids().len() as f64);

        let fresh =
            full_partition(&changed, PartitionConfig::default()).expect("fresh differential run");
        let incremental_q = modularity(&changed, incremental.partition().membership());
        let fresh_q = modularity(&changed, fresh.membership());
        assert!(
            (incremental.partition().modularity() - incremental_q).abs() <= 1e-12,
            "stateful incremental modularity must match the local oracle at cycle {cycle}"
        );
        assert!(
            (fresh.modularity() - fresh_q).abs() <= 1e-12,
            "fresh crate-reported modularity must match the local oracle at cycle {cycle}"
        );
        let signed_delta = incremental_q - fresh_q;
        min_modularity_delta = min_modularity_delta.min(signed_delta);
        let carry_forward_delta = incremental_q - carry_forward_q;
        assert!(
            carry_forward_delta >= -1e-9,
            "incremental {incremental_q:.12} must not degrade below carried prior \
             {carry_forward_q:.12} at cycle {cycle}; signed delta {carry_forward_delta:.3e}, \
             floor -1e-9"
        );
        if signed_delta < -1e-9 && !basin_diagnosed {
            let (inside_frontier, outside_frontier) = improving_single_node_moves(
                &changed,
                incremental.partition().membership(),
                incremental.optimized_nodes(),
            );
            assert_eq!(
                inside_frontier, 0,
                "frontier optimizer left an improving single-node move at cycle {cycle}"
            );
            eprintln!(
                "[m4_basin_diagnostic] cycle={cycle} incremental_q={incremental_q:.12} \
                 fresh_q={fresh_q:.12} signed_delta={signed_delta:.3e} \
                 improving_inside_frontier={inside_frontier} \
                 improving_outside_frontier={outside_frontier}"
            );
            basin_diagnosed = true;
        }
        assert_eq!(
            disconnected_community_count(&changed, incremental.partition().membership()),
            0,
            "incremental connectedness at cycle {cycle}"
        );

        let rebound = rebind_durable_ids(&durable, incremental.partition().membership());
        let changed_ids = rebound
            .iter()
            .zip(&durable)
            .filter(|(next, previous)| next != previous)
            .count();
        churn.push(changed_ids as f64 / rebound.len() as f64);
        membership = incremental.partition().membership().to_vec();
        durable = rebound;
        state = incremental.into_state();
    }

    ChurnSeries {
        durations,
        churn,
        poison_fraction: (base_edge_count.saturating_sub(6_151)) as f64 / base_edge_count as f64,
        min_modularity_delta,
        max_frontier_fraction,
    }
}

fn large_space_edges(space: usize) -> Vec<ProjectionInputEdge> {
    let mut edges = Vec::new();
    let mut edge_index = 0usize;
    for cluster in 0..8 {
        for node in 0..256 {
            for offset in 1..=3 {
                edges.push(ProjectionInputEdge::new(
                    format!("space-{space}-edge-{edge_index:06}"),
                    format!("space-{space}-cluster-{cluster:02}-node-{node:03}"),
                    format!(
                        "space-{space}-cluster-{cluster:02}-node-{:03}",
                        (node + offset) % 256
                    ),
                ));
                edge_index += 1;
            }
        }
        if cluster + 1 < 8 {
            edges.push(ProjectionInputEdge::new(
                format!("space-{space}-edge-{edge_index:06}"),
                format!("space-{space}-cluster-{cluster:02}-node-000"),
                format!("space-{space}-cluster-{:02}-node-000", cluster + 1),
            ));
            edge_index += 1;
        }
    }
    edges
}

fn add_five_percent_poison(mut edges: Vec<ProjectionInputEdge>) -> Vec<ProjectionInputEdge> {
    let poison_count = edges.len().div_ceil(19);
    for index in 0..poison_count {
        let (src_id, dst_id) = if index % 2 == 0 {
            let dst_cluster = 1 + (index / 2) % 7;
            let dst_node = (index * 17) % 256;
            (
                "space-0-cluster-00-node-000".to_owned(),
                format!("space-0-cluster-{dst_cluster:02}-node-{dst_node:03}"),
            )
        } else {
            (
                "space-0-cluster-00-node-001".to_owned(),
                "space-0-cluster-01-node-001".to_owned(),
            )
        };
        edges.push(ProjectionInputEdge::new(
            format!("poison-edge-{index:06}"),
            src_id,
            dst_id,
        ));
    }
    edges
}

fn community_count(membership: &[usize]) -> usize {
    membership.iter().copied().collect::<BTreeSet<_>>().len()
}

fn duration_p95(values: &[Duration]) -> Duration {
    duration_percentile(values, 95)
}

fn duration_percentile(values: &[Duration], percentile: usize) -> Duration {
    assert!(!values.is_empty());
    assert!((1..=100).contains(&percentile));
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted[(sorted.len() * percentile).div_ceil(100) - 1]
}

fn f64_p95(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[(sorted.len() * 95).div_ceil(100) - 1]
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn duration_ratio_percent(numerator: Duration, denominator: Duration) -> f64 {
    numerator.as_secs_f64() / denominator.as_secs_f64() * 100.0
}

fn ring_edges(node_count: usize) -> Vec<ProjectionInputEdge> {
    (0..node_count)
        .map(|node| {
            ProjectionInputEdge::new(
                format!("ring-edge-{node:06}"),
                format!("ring-node-{node:06}"),
                format!("ring-node-{:06}", (node + 1) % node_count),
            )
        })
        .collect()
}

fn incremental_p95(
    graph: &ProjectedGraph,
    membership: &[usize],
    dirty: &[usize],
    config: IncrementalConfig,
    runs: usize,
) -> Duration {
    let mut elapsed = Vec::with_capacity(runs);
    let mut state =
        IncrementalPartitionState::new(graph, membership).expect("incremental benchmark state");
    for _ in 0..runs {
        let started = Instant::now();
        let output =
            incremental_partition(graph, state, dirty, config).expect("incremental partition");
        elapsed.push(started.elapsed());
        state = output.into_state();
    }
    duration_p95(&elapsed)
}

fn improving_single_node_moves(
    graph: &ProjectedGraph,
    membership: &[usize],
    optimized_nodes: &[usize],
) -> (usize, usize) {
    let mut candidates = vec![BTreeSet::new(); graph.node_ids().len()];
    for edge in graph.edges() {
        candidates[edge.src].insert(membership[edge.dst]);
        candidates[edge.dst].insert(membership[edge.src]);
    }
    let optimized = optimized_nodes.iter().copied().collect::<BTreeSet<_>>();
    let baseline = modularity(graph, membership);
    let mut inside = 0usize;
    let mut outside = 0usize;
    for (node, node_candidates) in candidates.into_iter().enumerate() {
        let improves = node_candidates.into_iter().any(|candidate| {
            if candidate == membership[node] {
                return false;
            }
            let mut moved = membership.to_vec();
            moved[node] = candidate;
            modularity(graph, &moved) > baseline + 1e-12
        });
        if improves {
            if optimized.contains(&node) {
                inside += 1;
            } else {
                outside += 1;
            }
        }
    }
    (inside, outside)
}

#[cfg(target_os = "macos")]
fn fresh_child_partition_peak_rss_delta() -> u64 {
    let baseline = fresh_child_max_rss("m4_partition_rss_baseline_child", "baseline");
    let partition = fresh_child_max_rss("m4_partition_rss_partition_child", "partition");
    partition.saturating_sub(baseline)
}

#[cfg(target_os = "macos")]
fn fresh_child_max_rss(test_name: &str, child_kind: &str) -> u64 {
    let output = Command::new("/usr/bin/time")
        .arg("-l")
        .arg(std::env::current_exe().expect("current test executable"))
        .arg("--ignored")
        .arg("--exact")
        .arg(test_name)
        .arg("--nocapture")
        .env("WENLAN_M4_RSS_CHILD", child_kind)
        .output()
        .expect("run fresh RSS child through /usr/bin/time");
    assert!(
        output.status.success(),
        "fresh RSS child {test_name} failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    let stderr = String::from_utf8(output.stderr).expect("time stderr is UTF-8");
    stderr
        .lines()
        .find_map(|line| {
            line.trim()
                .strip_suffix("  maximum resident set size")
                .and_then(|value| value.trim().parse::<u64>().ok())
        })
        .unwrap_or_else(|| panic!("missing maximum resident set size in:\n{stderr}"))
}

#[cfg(not(target_os = "macos"))]
fn peak_rss_delta_for_full_partition(graph: &ProjectedGraph, runs: usize) -> u64 {
    let pid = sysinfo::get_current_pid().expect("current process id");
    let mut baseline_system = sysinfo::System::new_all();
    baseline_system.refresh_processes_specifics(
        sysinfo::ProcessesToUpdate::Some(&[pid]),
        false,
        sysinfo::ProcessRefreshKind::nothing().with_memory(),
    );
    let baseline = baseline_system
        .process(pid)
        .map_or(0, sysinfo::Process::memory);
    let stop = Arc::new(AtomicBool::new(false));
    let barrier = Arc::new(Barrier::new(2));
    let sample_stop = Arc::clone(&stop);
    let sample_barrier = Arc::clone(&barrier);
    let sampler = thread::spawn(move || {
        let mut system = sysinfo::System::new();
        let mut peak = baseline;
        sample_barrier.wait();
        loop {
            system.refresh_processes_specifics(
                sysinfo::ProcessesToUpdate::Some(&[pid]),
                false,
                sysinfo::ProcessRefreshKind::nothing().with_memory(),
            );
            if let Some(process) = system.process(pid) {
                peak = peak.max(process.memory());
            }
            if sample_stop.load(Ordering::Relaxed) {
                return peak;
            }
            thread::sleep(Duration::from_millis(1));
        }
    });
    barrier.wait();
    for _ in 0..runs {
        full_partition(graph, PartitionConfig::default()).expect("RSS full partition");
    }
    stop.store(true, Ordering::Relaxed);
    sampler
        .join()
        .expect("RSS sampler")
        .saturating_sub(baseline)
}

fn planted_cluster_edges(
    cluster_count: usize,
    nodes_per_cluster: usize,
) -> Vec<ProjectionInputEdge> {
    let mut edges = Vec::new();
    let mut edge_index = 0usize;
    for cluster in 0..cluster_count {
        for left in 0..nodes_per_cluster {
            for right in (left + 1)..nodes_per_cluster {
                edges.push(ProjectionInputEdge::new(
                    format!("edge-{edge_index:05}"),
                    format!("cluster-{cluster}-node-{left}"),
                    format!("cluster-{cluster}-node-{right}"),
                ));
                edge_index += 1;
            }
        }
        if cluster + 1 < cluster_count {
            edges.push(ProjectionInputEdge::new(
                format!("edge-{edge_index:05}"),
                format!("cluster-{cluster}-node-0"),
                format!("cluster-{}-node-0", cluster + 1),
            ));
            edge_index += 1;
        }
    }
    edges
}
