// SPDX-License-Identifier: Apache-2.0
//! Pure projection and partitioning primitives for persisted communities.

use std::collections::{BTreeMap, BTreeSet};

use leiden_rs::{GraphDataBuilder, Leiden, LeidenConfig, LeidenError, QualityType};
use thiserror::Error;

/// Fixed M4 projection parameters authored before the gate benchmark.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectionConfig {
    pub relates_weight: f64,
    pub parallel_edge_cap_multiplier: f64,
    pub hub_degree_cap: usize,
}

impl Default for ProjectionConfig {
    fn default() -> Self {
        Self {
            relates_weight: 1.0,
            parallel_edge_cap_multiplier: 3.0,
            hub_degree_cap: 50,
        }
    }
}

/// One active, grounded, same-space `relates` edge loaded from the database.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectionInputEdge {
    pub edge_id: String,
    pub src_id: String,
    pub dst_id: String,
}

impl ProjectionInputEdge {
    pub fn new(
        edge_id: impl Into<String>,
        src_id: impl Into<String>,
        dst_id: impl Into<String>,
    ) -> Self {
        Self {
            edge_id: edge_id.into(),
            src_id: src_id.into(),
            dst_id: dst_id.into(),
        }
    }
}

/// One weighted undirected edge over stable sorted node indices.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectedEdge {
    pub src: usize,
    pub dst: usize,
    pub weight: f64,
}

impl ProjectedEdge {
    pub const fn new(src: usize, dst: usize, weight: f64) -> Self {
        Self { src, dst, weight }
    }
}

/// Deterministic weighted graph consumed by both Leiden and label propagation.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectedGraph {
    node_ids: Vec<String>,
    node_order_fingerprint: u64,
    edges: Vec<ProjectedEdge>,
    adjacency: Vec<Vec<(usize, f64)>>,
    weighted_degrees: Vec<f64>,
    total_edge_weight: f64,
}

impl ProjectedGraph {
    pub fn node_ids(&self) -> &[String] {
        &self.node_ids
    }

    pub fn edges(&self) -> &[ProjectedEdge] {
        &self.edges
    }
}

/// Fixed M4 Leiden parameters authored before the gate benchmark.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PartitionConfig {
    pub resolution: f64,
    pub seed: u64,
    pub max_iterations: usize,
}

impl Default for PartitionConfig {
    fn default() -> Self {
        Self {
            resolution: 1.0,
            seed: 0x4D34,
            max_iterations: 100,
        }
    }
}

/// One full-partition result over the stable projected node order.
#[derive(Debug, Clone, PartialEq)]
pub struct PartitionOutput {
    membership: Vec<usize>,
    modularity: f64,
}

impl PartitionOutput {
    pub fn membership(&self) -> &[usize] {
        &self.membership
    }

    pub fn modularity(&self) -> f64 {
        self.modularity
    }
}

/// Weighted Newman-Girvan modularity for one hard partition.
pub fn modularity(graph: &ProjectedGraph, membership: &[usize]) -> f64 {
    assert_eq!(
        graph.node_ids.len(),
        membership.len(),
        "membership must cover every projected node"
    );
    let total_edge_weight = graph.total_edge_weight;
    if total_edge_weight <= 0.0 {
        return 0.0;
    }

    let mut internal_weight = BTreeMap::<usize, f64>::new();
    let mut degree_weight = BTreeMap::<usize, f64>::new();
    for edge in &graph.edges {
        *degree_weight.entry(membership[edge.src]).or_default() += edge.weight;
        *degree_weight.entry(membership[edge.dst]).or_default() += edge.weight;
        if membership[edge.src] == membership[edge.dst] {
            *internal_weight.entry(membership[edge.src]).or_default() += edge.weight;
        }
    }

    degree_weight
        .into_iter()
        .map(|(community, degree)| {
            let internal = internal_weight.get(&community).copied().unwrap_or_default();
            internal / total_edge_weight - (degree / (2.0 * total_edge_weight)).powi(2)
        })
        .sum()
}

/// Deterministic weighted label propagation matching the legacy baseline.
///
/// Gate-only: no production caller. Kept for the M4 Leiden-beats-label-propagation
/// baseline in `tests/m4_community_gates.rs`; production partitioning goes through
/// `full_partition`.
pub fn label_propagation_partition(graph: &ProjectedGraph) -> PartitionOutput {
    let mut membership = (0..graph.node_ids.len()).collect::<Vec<_>>();

    for _ in 0..100 {
        let mut changed = false;
        for node in 0..membership.len() {
            let mut weights = BTreeMap::<usize, f64>::new();
            for &(neighbor, weight) in &graph.adjacency[node] {
                *weights.entry(membership[neighbor]).or_default() += weight;
            }
            let mut best = None::<(usize, f64)>;
            for (community, weight) in weights {
                if best.is_none_or(|(_, best_weight)| weight > best_weight) {
                    best = Some((community, weight));
                }
            }
            if let Some((community, _)) = best {
                if membership[node] != community {
                    membership[node] = community;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    PartitionOutput {
        modularity: modularity(graph, &membership),
        membership,
    }
}

/// Count communities whose induced subgraph is disconnected.
///
/// Gate-only: no production caller. Kept as the connectedness metric for
/// `tests/m4_community_gates.rs`.
pub fn disconnected_community_count(graph: &ProjectedGraph, membership: &[usize]) -> usize {
    assert_eq!(
        graph.node_ids.len(),
        membership.len(),
        "membership must cover every projected node"
    );
    let mut communities = BTreeMap::<usize, Vec<usize>>::new();
    for (node, &community) in membership.iter().enumerate() {
        communities.entry(community).or_default().push(node);
    }

    communities
        .into_iter()
        .filter(|(community, _)| !community_is_connected(graph, membership, *community))
        .count()
}

fn community_is_connected(graph: &ProjectedGraph, membership: &[usize], community: usize) -> bool {
    let Some(first) = membership
        .iter()
        .position(|candidate| *candidate == community)
    else {
        return true;
    };
    let expected = membership
        .iter()
        .filter(|candidate| **candidate == community)
        .count();
    let mut visited = BTreeSet::new();
    let mut stack = vec![first];
    while let Some(node) = stack.pop() {
        if !visited.insert(node) {
            continue;
        }
        for &(neighbor, _) in &graph.adjacency[node] {
            if membership[neighbor] == community {
                stack.push(neighbor);
            }
        }
    }
    visited.len() == expected
}

/// Fixed bounds for frontier-restricted modularity optimization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IncrementalConfig {
    pub max_dirty_fraction: f64,
    pub max_frontier_fraction: f64,
    pub max_passes: usize,
    pub epsilon: f64,
}

impl Default for IncrementalConfig {
    fn default() -> Self {
        Self {
            max_dirty_fraction: 0.25,
            max_frontier_fraction: 0.25,
            max_passes: 100,
            epsilon: 1e-12,
        }
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum IncrementalPartitionError {
    #[error("prior membership has {actual} nodes, expected {expected}")]
    MembershipSize { expected: usize, actual: usize },
    #[error("dirty node index {node} is outside graph with {node_count} nodes")]
    DirtyNode { node: usize, node_count: usize },
    #[error("dirty fraction {actual:.6} exceeds full-repartition threshold {maximum:.6}")]
    DirtyFraction { actual: f64, maximum: f64 },
    #[error(
        "expanded frontier fraction {actual:.6} exceeds full-repartition threshold {maximum:.6}"
    )]
    FrontierFraction { actual: f64, maximum: f64 },
    #[error("projected node order differs from the incremental state")]
    NodeOrder,
    #[error("frontier optimization produced a disconnected community")]
    DisconnectedCommunity,
}

/// Reusable statistics for frontier-local updates.
///
/// Building this state is graph-wide and belongs to first publication, process
/// restart, an algorithm/projection version change, or an explicit full
/// repartition. Steady-state edge changes update it only through the supplied
/// dirty endpoints and their old/new incident edges.
#[derive(Debug, PartialEq)]
pub struct IncrementalPartitionState {
    partition: PartitionOutput,
    node_order_fingerprint: u64,
    adjacency: Vec<BTreeMap<usize, f64>>,
    weighted_degrees: Vec<f64>,
    community_degrees: BTreeMap<usize, f64>,
    total_internal_weight: f64,
    sum_community_degree_squares: f64,
    total_edge_weight: f64,
    next_community: usize,
}

#[derive(Debug)]
pub(crate) struct IncrementalPartitionRollback {
    membership: Vec<(usize, usize)>,
    adjacency: Vec<(usize, BTreeMap<usize, f64>)>,
    weighted_degrees: Vec<(usize, f64)>,
    community_degrees: Vec<(usize, f64)>,
    partition_modularity: f64,
    total_internal_weight: f64,
    sum_community_degree_squares: f64,
    total_edge_weight: f64,
    next_community: usize,
}

impl IncrementalPartitionState {
    pub fn new(
        graph: &ProjectedGraph,
        membership: &[usize],
    ) -> Result<Self, IncrementalPartitionError> {
        if membership.len() != graph.node_ids.len() {
            return Err(IncrementalPartitionError::MembershipSize {
                expected: graph.node_ids.len(),
                actual: membership.len(),
            });
        }

        let mut community_degrees = BTreeMap::<usize, f64>::new();
        for (node, &community) in membership.iter().enumerate() {
            *community_degrees.entry(community).or_default() += graph.weighted_degrees[node];
        }
        let sum_community_degree_squares = community_degrees
            .values()
            .map(|degree| degree.powi(2))
            .sum();
        let total_internal_weight = graph
            .edges
            .iter()
            .filter(|edge| membership[edge.src] == membership[edge.dst])
            .map(|edge| edge.weight)
            .sum();
        let modularity = modularity_from_totals(
            graph.total_edge_weight,
            total_internal_weight,
            sum_community_degree_squares,
        );
        let adjacency = graph
            .adjacency
            .iter()
            .map(|neighbors| neighbors.iter().copied().collect())
            .collect();

        Ok(Self {
            partition: PartitionOutput {
                membership: membership.to_vec(),
                modularity,
            },
            node_order_fingerprint: graph.node_order_fingerprint,
            adjacency,
            weighted_degrees: graph.weighted_degrees.clone(),
            community_degrees,
            total_internal_weight,
            sum_community_degree_squares,
            total_edge_weight: graph.total_edge_weight,
            next_community: membership.iter().copied().max().unwrap_or_default() + 1,
        })
    }

    pub fn partition(&self) -> &PartitionOutput {
        &self.partition
    }

    fn incremental_frontier(
        &self,
        graph: &ProjectedGraph,
        dirty_nodes: &[usize],
        config: IncrementalConfig,
    ) -> Result<(BTreeSet<usize>, BTreeSet<usize>), IncrementalPartitionError> {
        let node_count = graph.node_ids.len();
        if self.partition.membership.len() != node_count {
            return Err(IncrementalPartitionError::MembershipSize {
                expected: node_count,
                actual: self.partition.membership.len(),
            });
        }
        if self.node_order_fingerprint != graph.node_order_fingerprint {
            return Err(IncrementalPartitionError::NodeOrder);
        }
        let dirty = dirty_nodes.iter().copied().collect::<BTreeSet<_>>();
        if let Some(&node) = dirty.iter().find(|&&node| node >= node_count) {
            return Err(IncrementalPartitionError::DirtyNode { node, node_count });
        }
        let dirty_fraction = if node_count == 0 {
            0.0
        } else {
            dirty.len() as f64 / node_count as f64
        };
        if dirty_fraction > config.max_dirty_fraction {
            return Err(IncrementalPartitionError::DirtyFraction {
                actual: dirty_fraction,
                maximum: config.max_dirty_fraction,
            });
        }

        let mut frontier = dirty.clone();
        for &node in &dirty {
            frontier.extend(self.adjacency[node].keys().copied());
            frontier.extend(graph.adjacency[node].iter().map(|(neighbor, _)| *neighbor));
        }
        let frontier_fraction = if node_count == 0 {
            0.0
        } else {
            frontier.len() as f64 / node_count as f64
        };
        if frontier_fraction > config.max_frontier_fraction {
            return Err(IncrementalPartitionError::FrontierFraction {
                actual: frontier_fraction,
                maximum: config.max_frontier_fraction,
            });
        }
        Ok((dirty, frontier))
    }

    fn rollback_snapshot(
        &self,
        graph: &ProjectedGraph,
        dirty: &BTreeSet<usize>,
        frontier: &BTreeSet<usize>,
    ) -> IncrementalPartitionRollback {
        let mut touched_communities = BTreeSet::new();
        for &node in frontier {
            touched_communities.insert(self.partition.membership[node]);
            touched_communities.extend(
                self.adjacency[node]
                    .keys()
                    .map(|&neighbor| self.partition.membership[neighbor]),
            );
            touched_communities.extend(
                graph.adjacency[node]
                    .iter()
                    .map(|(neighbor, _)| self.partition.membership[*neighbor]),
            );
        }
        for &node in dirty {
            touched_communities.insert(self.partition.membership[node]);
        }

        IncrementalPartitionRollback {
            membership: frontier
                .iter()
                .map(|&node| (node, self.partition.membership[node]))
                .collect(),
            adjacency: frontier
                .iter()
                .map(|&node| (node, self.adjacency[node].clone()))
                .collect(),
            weighted_degrees: frontier
                .iter()
                .map(|&node| (node, self.weighted_degrees[node]))
                .collect(),
            community_degrees: touched_communities
                .into_iter()
                .filter_map(|community| {
                    self.community_degrees
                        .get(&community)
                        .copied()
                        .map(|degree| (community, degree))
                })
                .collect(),
            partition_modularity: self.partition.modularity,
            total_internal_weight: self.total_internal_weight,
            sum_community_degree_squares: self.sum_community_degree_squares,
            total_edge_weight: self.total_edge_weight,
            next_community: self.next_community,
        }
    }

    pub(crate) fn restore(&mut self, rollback: IncrementalPartitionRollback) {
        for (node, membership) in rollback.membership {
            self.partition.membership[node] = membership;
        }
        for (node, adjacency) in rollback.adjacency {
            self.adjacency[node] = adjacency;
        }
        for (node, weighted_degree) in rollback.weighted_degrees {
            self.weighted_degrees[node] = weighted_degree;
        }
        self.community_degrees
            .retain(|community, _| *community < rollback.next_community);
        for (community, degree) in rollback.community_degrees {
            self.community_degrees.insert(community, degree);
        }
        self.partition.modularity = rollback.partition_modularity;
        self.total_internal_weight = rollback.total_internal_weight;
        self.sum_community_degree_squares = rollback.sum_community_degree_squares;
        self.total_edge_weight = rollback.total_edge_weight;
        self.next_community = rollback.next_community;
    }

    fn adjust_community_degree(&mut self, community: usize, delta: f64) {
        let degree = self.community_degrees.entry(community).or_default();
        self.sum_community_degree_squares -= degree.powi(2);
        *degree += delta;
        self.sum_community_degree_squares += degree.powi(2);
    }

    fn refresh_modularity(&mut self) {
        self.partition.modularity = modularity_from_totals(
            self.total_edge_weight,
            self.total_internal_weight,
            self.sum_community_degree_squares,
        );
    }
}

/// Result of a frontier-restricted update. Nodes outside `optimized_nodes` are
/// carried forward byte-for-byte from the prior partition.
#[derive(Debug, PartialEq)]
pub struct IncrementalOutput {
    state: IncrementalPartitionState,
    optimized_nodes: Vec<usize>,
}

impl IncrementalOutput {
    pub fn partition(&self) -> &PartitionOutput {
        self.state.partition()
    }

    pub fn optimized_nodes(&self) -> &[usize] {
        &self.optimized_nodes
    }

    pub fn into_state(self) -> IncrementalPartitionState {
        self.state
    }
}

/// Re-optimize dirty nodes plus their one-hop neighborhood against global
/// modularity statistics while holding every other assignment fixed.
///
/// The previous partition must already satisfy the connected-community
/// invariant. Only communities touched by accepted frontier moves can change
/// connectedness, so the postcondition checks exactly that bounded set.
///
/// Gate-only: no production caller. Kept for `tests/m4_community_gates.rs`;
/// production calls `incremental_partition_recoverable` directly.
pub fn incremental_partition(
    graph: &ProjectedGraph,
    state: IncrementalPartitionState,
    dirty_nodes: &[usize],
    config: IncrementalConfig,
) -> Result<IncrementalOutput, IncrementalPartitionError> {
    incremental_partition_recoverable(graph, state, dirty_nodes, config)
        .map(|(output, _rollback)| output)
        .map_err(|failure| {
            let (error, _state) = *failure;
            error
        })
}

pub(crate) fn incremental_partition_recoverable(
    graph: &ProjectedGraph,
    mut state: IncrementalPartitionState,
    dirty_nodes: &[usize],
    config: IncrementalConfig,
) -> Result<
    (IncrementalOutput, IncrementalPartitionRollback),
    Box<(IncrementalPartitionError, IncrementalPartitionState)>,
> {
    let (dirty, frontier) = match state.incremental_frontier(graph, dirty_nodes, config) {
        Ok(frontier) => frontier,
        Err(error) => return Err(Box::new((error, state))),
    };
    let rollback = state.rollback_snapshot(graph, &dirty, &frontier);

    let mut touched_communities = apply_projection_deltas(graph, &mut state, &dirty);
    let optimized_nodes = frontier.iter().copied().collect::<Vec<_>>();

    if state.total_edge_weight > 0.0 {
        for _ in 0..config.max_passes {
            let mut changed = false;
            for &node in &optimized_nodes {
                let current = state.partition.membership[node];
                let node_degree = state.weighted_degrees[node];
                let mut incident_by_community = BTreeMap::<usize, f64>::new();
                for &(neighbor, weight) in &graph.adjacency[node] {
                    *incident_by_community
                        .entry(state.partition.membership[neighbor])
                        .or_default() += weight;
                }
                incident_by_community
                    .entry(state.next_community)
                    .or_default();

                let from_incident = incident_by_community
                    .get(&current)
                    .copied()
                    .unwrap_or_default();
                let from_degree = state
                    .community_degrees
                    .get(&current)
                    .copied()
                    .unwrap_or_default();
                let mut best = None::<(usize, f64)>;
                for (&candidate, &to_incident) in &incident_by_community {
                    if candidate == current {
                        continue;
                    }
                    let to_degree = state
                        .community_degrees
                        .get(&candidate)
                        .copied()
                        .unwrap_or_default();
                    let delta = modularity_move_delta(
                        state.total_edge_weight,
                        node_degree,
                        from_incident,
                        to_incident,
                        from_degree,
                        to_degree,
                    );
                    if delta > config.epsilon
                        && best.is_none_or(|(best_community, best_delta)| {
                            delta > best_delta + config.epsilon
                                || ((delta - best_delta).abs() <= config.epsilon
                                    && candidate < best_community)
                        })
                    {
                        best = Some((candidate, delta));
                    }
                }

                if let Some((candidate, _)) = best {
                    let to_incident = incident_by_community
                        .get(&candidate)
                        .copied()
                        .unwrap_or_default();
                    state.partition.membership[node] = candidate;
                    state.adjust_community_degree(current, -node_degree);
                    state.adjust_community_degree(candidate, node_degree);
                    state.total_internal_weight += to_incident - from_incident;
                    state.refresh_modularity();
                    touched_communities.insert(current);
                    touched_communities.insert(candidate);
                    if candidate == state.next_community {
                        state.next_community += 1;
                    }
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
    }

    if touched_communities
        .into_iter()
        .any(|community| !community_is_connected(graph, &state.partition.membership, community))
    {
        state.restore(rollback);
        return Err(Box::new((
            IncrementalPartitionError::DisconnectedCommunity,
            state,
        )));
    }

    Ok((
        IncrementalOutput {
            state,
            optimized_nodes,
        },
        rollback,
    ))
}

fn apply_projection_deltas(
    graph: &ProjectedGraph,
    state: &mut IncrementalPartitionState,
    dirty: &BTreeSet<usize>,
) -> BTreeSet<usize> {
    let mut changed_pairs = BTreeSet::new();
    let mut connectivity_touched = BTreeSet::new();
    for &node in dirty {
        changed_pairs.extend(
            state.adjacency[node]
                .keys()
                .map(|&neighbor| ordered_pair(node, neighbor)),
        );
        changed_pairs.extend(
            graph.adjacency[node]
                .iter()
                .map(|(neighbor, _)| ordered_pair(node, *neighbor)),
        );
    }

    for (src, dst) in changed_pairs {
        let old_weight = state.adjacency[src].get(&dst).copied().unwrap_or_default();
        let new_weight = graph.adjacency[src]
            .binary_search_by_key(&dst, |(neighbor, _)| *neighbor)
            .ok()
            .map(|index| graph.adjacency[src][index].1)
            .unwrap_or_default();
        let delta = new_weight - old_weight;
        if delta == 0.0 {
            continue;
        }

        let src_community = state.partition.membership[src];
        let dst_community = state.partition.membership[dst];
        if old_weight > 0.0 && new_weight == 0.0 && src_community == dst_community {
            connectivity_touched.insert(src_community);
        }
        state.weighted_degrees[src] += delta;
        state.weighted_degrees[dst] += delta;
        state.adjust_community_degree(src_community, delta);
        state.adjust_community_degree(dst_community, delta);
        state.total_edge_weight += delta;
        if src_community == dst_community {
            state.total_internal_weight += delta;
        }

        if new_weight == 0.0 {
            state.adjacency[src].remove(&dst);
            state.adjacency[dst].remove(&src);
        } else {
            state.adjacency[src].insert(dst, new_weight);
            state.adjacency[dst].insert(src, new_weight);
        }
    }
    state.refresh_modularity();
    connectivity_touched
}

fn ordered_pair(left: usize, right: usize) -> (usize, usize) {
    if left < right {
        (left, right)
    } else {
        (right, left)
    }
}

fn modularity_from_totals(
    total_edge_weight: f64,
    total_internal_weight: f64,
    sum_community_degree_squares: f64,
) -> f64 {
    if total_edge_weight <= 0.0 {
        return 0.0;
    }
    total_internal_weight / total_edge_weight
        - sum_community_degree_squares / (4.0 * total_edge_weight.powi(2))
}

fn modularity_move_delta(
    total_edge_weight: f64,
    node_degree: f64,
    from_incident: f64,
    to_incident: f64,
    from_degree: f64,
    to_degree: f64,
) -> f64 {
    let internal_delta = (to_incident - from_incident) / total_edge_weight;
    let degree_delta = ((to_degree + node_degree).powi(2) - to_degree.powi(2)
        + (from_degree - node_degree).powi(2)
        - from_degree.powi(2))
        / (4.0 * total_edge_weight.powi(2));
    internal_delta - degree_delta
}

/// Rebind partitioner groups to prior durable community IDs by maximum member
/// overlap. Partitioner labels never become durable identity.
///
/// Gate-only: no production caller. Pins the unweighted stability baseline in
/// `tests/m4_community_gates.rs`; production uses `rebind_durable_ids_weighted`.
pub fn rebind_durable_ids(previous_ids: &[String], new_membership: &[usize]) -> Vec<String> {
    assert_eq!(
        previous_ids.len(),
        new_membership.len(),
        "prior IDs and new membership must cover the same nodes"
    );
    let mut overlaps = BTreeMap::<(usize, String), f64>::new();
    for (node, &group) in new_membership.iter().enumerate() {
        *overlaps
            .entry((group, previous_ids[node].clone()))
            .or_default() += 1.0;
    }
    assign_rebound_ids(previous_ids, new_membership, overlaps)
}

/// Rebind partitioner groups by weighted-fractional overlap with prior
/// communities. Each node contributes one unit split across the prior
/// communities reached by its incident grounded weight. A node with no
/// grounded adjacency falls back to its own prior identity.
pub fn rebind_durable_ids_weighted(
    previous_ids: &[String],
    new_membership: &[usize],
    graph: &ProjectedGraph,
) -> Vec<String> {
    assert_eq!(
        previous_ids.len(),
        new_membership.len(),
        "prior IDs and new membership must cover the same nodes"
    );
    assert_eq!(
        previous_ids.len(),
        graph.node_ids.len(),
        "weighted rebinding inputs must cover the projected graph"
    );

    let mut overlaps = BTreeMap::<(usize, String), f64>::new();
    for (node, &group) in new_membership.iter().enumerate() {
        if is_new_community_marker(&previous_ids[node]) {
            continue;
        }
        let mut incident_by_old = BTreeMap::<String, f64>::new();
        let mut total = 0.0;
        for &(neighbor, weight) in &graph.adjacency[node] {
            let prior_id = &previous_ids[neighbor];
            if weight <= 0.0 || is_new_community_marker(prior_id) {
                continue;
            }
            *incident_by_old.entry(prior_id.clone()).or_default() += weight;
            total += weight;
        }
        if total > 0.0 {
            for (old_id, weight) in incident_by_old {
                *overlaps.entry((group, old_id)).or_default() += weight / total;
            }
        } else if !is_new_community_marker(&previous_ids[node]) {
            *overlaps
                .entry((group, previous_ids[node].clone()))
                .or_default() += 1.0;
        }
    }
    assign_rebound_ids(previous_ids, new_membership, overlaps)
}

fn is_new_community_marker(community_id: &str) -> bool {
    community_id.starts_with("__m4-new-node-") || community_id.starts_with("community-m4-new-")
}

fn assign_rebound_ids(
    previous_ids: &[String],
    new_membership: &[usize],
    overlaps: BTreeMap<(usize, String), f64>,
) -> Vec<String> {
    let groups = new_membership.iter().copied().collect::<BTreeSet<_>>();
    let mut candidates = overlaps
        .into_iter()
        .map(|((group, old_id), overlap)| (overlap, old_id, group))
        .collect::<Vec<_>>();
    candidates.sort_by(
        |(left_overlap, left_old, left_group), (right_overlap, right_old, right_group)| {
            right_overlap
                .total_cmp(left_overlap)
                .then(left_old.cmp(right_old))
                .then(left_group.cmp(right_group))
        },
    );

    let mut claimed_old = BTreeSet::new();
    let mut rebound = BTreeMap::<usize, String>::new();
    for (_, old_id, group) in candidates {
        if !rebound.contains_key(&group) && claimed_old.insert(old_id.clone()) {
            rebound.insert(group, old_id);
        }
    }

    let mut used_ids = previous_ids.iter().cloned().collect::<BTreeSet<_>>();
    let mut fresh_ordinal = 0usize;
    for &group in &groups {
        if rebound.contains_key(&group) {
            continue;
        }
        let fresh_id = loop {
            let candidate = format!("community-m4-new-{fresh_ordinal}");
            fresh_ordinal += 1;
            if used_ids.insert(candidate.clone()) {
                break candidate;
            }
        };
        rebound.insert(group, fresh_id);
    }

    new_membership
        .iter()
        .map(|group| rebound[group].clone())
        .collect()
}

/// Run sequential seeded Leiden over the complete projected graph.
pub fn full_partition(
    graph: &ProjectedGraph,
    config: PartitionConfig,
) -> Result<PartitionOutput, LeidenError> {
    if graph.node_ids.is_empty() {
        return Ok(PartitionOutput {
            membership: Vec::new(),
            modularity: 0.0,
        });
    }

    let mut builder = GraphDataBuilder::new(graph.node_ids.len());
    for edge in &graph.edges {
        builder.add_edge(edge.src, edge.dst, edge.weight)?;
    }
    let data = builder.build()?;
    let leiden = Leiden::new(LeidenConfig {
        max_iterations: config.max_iterations,
        resolution: config.resolution,
        seed: Some(config.seed),
        quality: QualityType::Modularity,
        ..LeidenConfig::default()
    });
    let result = leiden.run(&data)?;

    Ok(PartitionOutput {
        membership: result.partition.as_slice().to_vec(),
        modularity: result.quality,
    })
}

/// Project active grounded `relates` rows into the exact M4 undirected graph.
pub fn project_grounded_relates(
    input: &[ProjectionInputEdge],
    config: ProjectionConfig,
) -> ProjectedGraph {
    let mut node_ids = BTreeSet::new();
    let mut folded_weights: BTreeMap<(String, String), f64> = BTreeMap::new();
    let cap = config.parallel_edge_cap_multiplier * config.relates_weight;

    let mut ordered = input.iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| left.edge_id.cmp(&right.edge_id));

    for edge in ordered {
        if edge.src_id == edge.dst_id {
            continue;
        }
        node_ids.insert(edge.src_id.clone());
        node_ids.insert(edge.dst_id.clone());

        let pair = if edge.src_id < edge.dst_id {
            (edge.src_id.clone(), edge.dst_id.clone())
        } else {
            (edge.dst_id.clone(), edge.src_id.clone())
        };
        let weight = folded_weights.entry(pair).or_default();
        *weight = (*weight + config.relates_weight).min(cap);
    }

    let node_ids = node_ids.into_iter().collect::<Vec<_>>();
    let node_order_fingerprint = stable_node_order_fingerprint(&node_ids);
    let node_indices = node_ids
        .iter()
        .enumerate()
        .map(|(index, id)| (id.as_str(), index))
        .collect::<BTreeMap<_, _>>();
    let mut folded_degrees = vec![0usize; node_ids.len()];
    for (src_id, dst_id) in folded_weights.keys() {
        folded_degrees[node_indices[&src_id.as_str()]] += 1;
        folded_degrees[node_indices[&dst_id.as_str()]] += 1;
    }
    let edges = folded_weights
        .into_iter()
        .map(|((src_id, dst_id), weight)| {
            let src = node_indices[&src_id.as_str()];
            let dst = node_indices[&dst_id.as_str()];
            let src_factor = (config.hub_degree_cap as f64 / folded_degrees[src] as f64).min(1.0);
            let dst_factor = (config.hub_degree_cap as f64 / folded_degrees[dst] as f64).min(1.0);
            ProjectedEdge::new(src, dst, weight * src_factor * dst_factor)
        })
        .collect::<Vec<_>>();
    let mut adjacency = vec![Vec::new(); node_ids.len()];
    let mut weighted_degrees = vec![0.0; node_ids.len()];
    let mut total_edge_weight = 0.0;
    for edge in &edges {
        adjacency[edge.src].push((edge.dst, edge.weight));
        adjacency[edge.dst].push((edge.src, edge.weight));
        weighted_degrees[edge.src] += edge.weight;
        weighted_degrees[edge.dst] += edge.weight;
        total_edge_weight += edge.weight;
    }
    for neighbors in &mut adjacency {
        neighbors.sort_by_key(|(node, _)| *node);
    }

    ProjectedGraph {
        node_ids,
        node_order_fingerprint,
        edges,
        adjacency,
        weighted_degrees,
        total_edge_weight,
    }
}

fn stable_node_order_fingerprint(node_ids: &[String]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hash = FNV_OFFSET;
    for node_id in node_ids {
        for byte in (node_id.len() as u64)
            .to_le_bytes()
            .into_iter()
            .chain(node_id.bytes())
        {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}
