// SPDX-License-Identifier: Apache-2.0

#[cfg(target_os = "macos")]
use std::process::Command;
use std::{
    collections::BTreeSet,
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
    compute_community_grouping, run_community_grouping_cycle, CommunityGroupingError,
    CommunityGroupingOutcome, CommunityGroupingRuntime,
};
use wenlan_core::community_partition::{
    disconnected_community_count, full_partition, incremental_partition,
    label_propagation_partition, modularity, project_grounded_relates, rebind_durable_ids,
    IncrementalConfig, IncrementalPartitionError, IncrementalPartitionState, PartitionConfig,
    ProjectedEdge, ProjectedGraph, ProjectionConfig, ProjectionInputEdge,
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

    let chords = m4_scale_chord_pairs(22);
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

    db.set_reader_cutover("communities", true)
        .await
        .expect("route legacy detector through the graded edge corpus");
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
    assert_eq!(initial_state.graph_generation, 1);
    assert_eq!(initial_state.published_generation, None);
    assert!(initial_state.dirty);
    assert!(initial_state.members.is_empty());

    let mut runtime = CommunityGroupingRuntime::default();
    let mut mutex_holds = Vec::new();
    let mut foreground_latencies = Vec::new();
    let (first_result, first_foreground_latency) =
        run_m4_cycle_with_foreground_probe(&db, &mut runtime, SPACE).await;
    foreground_latencies.push(first_foreground_latency);
    let first = match first_result.expect("first production grouping cycle") {
        CommunityGroupingOutcome::Published(receipt) => receipt,
        other => panic!("first dirty cycle must publish, got {other:?}"),
    };
    assert_eq!(first.input_generation, 1);
    assert_eq!(first.published_generation, 1);
    assert_eq!(
        first.projected_edge_count,
        BASE_EDGES_PER_SPACE + 1,
        "the real SELECT must exclude the grounded=0 relation and the promoted second-space edge"
    );
    assert_eq!(first.member_count, PARTICIPANTS_PER_SPACE);
    mutex_holds.push(first.db_mutex_hold);

    let first_snapshot = read_m4_persisted_snapshot(&observer, SPACE).await;
    assert_eq!(first_snapshot.graph_generation, 1);
    assert_eq!(first_snapshot.published_generation, Some(1));
    assert!(!first_snapshot.dirty);
    assert_eq!(first_snapshot.members.len(), PARTICIPANTS_PER_SPACE);
    assert!(first_snapshot
        .members
        .iter()
        .all(|member| member.published_generation == 1));

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
        let expected_generation = cycle as i64 + 2;
        assert_eq!(published.input_generation, expected_generation);
        assert_eq!(published.published_generation, expected_generation);
        assert_eq!(
            published.projected_edge_count,
            BASE_EDGES_PER_SPACE + cycle + 2,
            "the published receipt must include that cycle's corrected edge"
        );
        mutex_holds.push(published.db_mutex_hold);

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
    assert_eq!(before_stale.graph_generation, 20);
    assert_eq!(before_stale.published_generation, Some(19));
    assert!(before_stale.dirty);

    seed_obsolete_m4_leases(&observer, SPACE, before_stale.graph_generation).await;
    let attempt = db
        .prepare_community_grouping(SPACE)
        .await
        .expect("prepare real lease + grounded SELECT");
    assert_eq!(attempt.input_generation(), 20);
    assert_eq!(
        read_m4_lease_generations(&observer, SPACE).await,
        vec![20],
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
                input_generation: 20,
            } if space == SPACE
        ),
        "same-key exclusion must be typed LeaseHeld, got {held:?}"
    );

    let computed = compute_community_grouping(&attempt, &runtime)
        .expect("pure compute between DB phases, with no DB handle");
    assert_eq!(computed.projected_edge_count(), BASE_EDGES_PER_SPACE + 20);

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

    let stale = match db
        .finalize_community_grouping(attempt, computed)
        .await
        .expect("generation-CAS stale finalize")
    {
        CommunityGroupingOutcome::Stale(receipt) => receipt,
        other => panic!("mid-run promotion must reject stale publication, got {other:?}"),
    };
    assert_eq!(stale.input_generation, 20);
    mutex_holds.push(stale.db_mutex_hold);
    assert_eq!(
        runtime.published_generation(SPACE),
        Some(19),
        "a stale finalize must not install its computed in-memory state"
    );

    let after_stale = read_m4_persisted_snapshot(&observer, SPACE).await;
    assert_eq!(after_stale.graph_generation, 21);
    assert_eq!(after_stale.published_generation, Some(19));
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
    assert_eq!(recovered.input_generation, 21);
    assert_eq!(recovered.published_generation, 21);
    assert_eq!(recovered.projected_edge_count, BASE_EDGES_PER_SPACE + 21);
    mutex_holds.push(recovered.db_mutex_hold);
    assert_eq!(runtime.published_generation(SPACE), Some(21));

    let recovered_snapshot = read_m4_persisted_snapshot(&observer, SPACE).await;
    assert_eq!(recovered_snapshot.graph_generation, 21);
    assert_eq!(recovered_snapshot.published_generation, Some(21));
    assert!(!recovered_snapshot.dirty);
    assert!(recovered_snapshot
        .members
        .iter()
        .all(|member| member.published_generation == 21));
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
    assert_m4_job_phase_structure();
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
    let relation_id = db
        .create_relation(
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
        relation_id,
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

    let job_source = include_str!("../src/community_grouping.rs");
    let composer = rust_function_body(job_source, "pub async fn run_community_grouping_cycle");
    let prepare = composer
        .find("prepare_community_grouping(")
        .expect("composer prepare phase");
    let compute = composer
        .find("compute_community_grouping(")
        .expect("composer pure compute phase");
    let finalize = composer
        .find("finalize_community_grouping(")
        .expect("composer finalize phase");
    assert!(
        prepare < compute && compute < finalize,
        "production composer must preserve prepare -> pure compute -> finalize ordering"
    );
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
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted[(sorted.len() * 95).div_ceil(100) - 1]
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
