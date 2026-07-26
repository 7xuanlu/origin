// SPDX-License-Identifier: Apache-2.0
//! M4 community grouping job: short DB phases around pure graph computation.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::Duration,
};

use thiserror::Error;

use crate::{
    community_partition::{
        full_partition, incremental_partition_recoverable, project_grounded_relates,
        rebind_durable_ids_weighted, IncrementalConfig, IncrementalPartitionError,
        IncrementalPartitionRollback, IncrementalPartitionState, PartitionConfig, ProjectionConfig,
        ProjectionInputEdge,
    },
    db::MemoryDB,
};

pub(crate) const COMMUNITY_ALGO_VERSION: &str = "leiden-m4-v1";
pub(crate) const COMMUNITY_PROJECTION_VERSION: &str = "grounded-relates-v1";
const MIN_COMMUNITY_PARTICIPANTS: usize = 10;

#[derive(Debug, Error)]
pub enum CommunityGroupingError {
    #[error(
        "community grouping lease already held for space {space} at generation {input_generation}"
    )]
    LeaseHeld {
        space: String,
        input_generation: i64,
    },
    #[error("space {space} has no dirty community graph generation")]
    NotDirty { space: String },
    #[error("community grouping database error: {0}")]
    Database(String),
    #[error("community grouping computation failed: {0}")]
    Compute(String),
}

#[derive(Debug)]
pub struct CommunityGroupingAttempt {
    pub(crate) space: String,
    pub(crate) composer_started: std::time::Instant,
    pub(crate) graph_generation: i64,
    pub(crate) input_generation: i64,
    pub(crate) published_generation: Option<i64>,
    pub(crate) published_versions: Option<(String, String)>,
    pub(crate) token: String,
    pub(crate) lease_cleanup: CommunityGroupingLeaseCleanup,
    pub(crate) edges: Vec<ProjectionInputEdge>,
    pub(crate) ungrounded_edges: Vec<ProjectionInputEdge>,
    pub(crate) entity_embeddings: BTreeMap<String, Vec<f32>>,
    pub(crate) previous_ids: BTreeMap<String, String>,
    pub(crate) dirty_node_ids: BTreeSet<String>,
    pub(crate) db_mutex_hold: Duration,
}

impl CommunityGroupingAttempt {
    pub fn input_generation(&self) -> i64 {
        self.input_generation
    }

    pub(crate) fn disarm_lease_cleanup(&mut self) {
        self.lease_cleanup.disarm();
    }
}

pub(crate) struct CommunityGroupingLeaseCleanup {
    conn: Arc<tokio::sync::Mutex<libsql::Connection>>,
    space: String,
    input_generation: i64,
    token: String,
    armed: bool,
}

impl std::fmt::Debug for CommunityGroupingLeaseCleanup {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CommunityGroupingLeaseCleanup")
            .field("space", &self.space)
            .field("input_generation", &self.input_generation)
            .field("armed", &self.armed)
            .finish_non_exhaustive()
    }
}

impl CommunityGroupingLeaseCleanup {
    pub(crate) fn new(
        conn: Arc<tokio::sync::Mutex<libsql::Connection>>,
        space: String,
        input_generation: i64,
        token: String,
    ) -> Self {
        Self {
            conn,
            space,
            input_generation,
            token,
            armed: true,
        }
    }

    pub(crate) fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for CommunityGroupingLeaseCleanup {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let conn = Arc::clone(&self.conn);
        let space = self.space.clone();
        let input_generation = self.input_generation;
        let token = self.token.clone();
        runtime.spawn(async move {
            let conn = conn.lock().await;
            let _ = conn
                .execute(
                    "DELETE FROM grouping_leases
                     WHERE phase = 'community' AND space = ?1
                       AND input_generation = ?2 AND token = ?3",
                    libsql::params![space, input_generation, token],
                )
                .await;
        });
    }
}

#[derive(Debug)]
pub struct CommunityGroupingComputed {
    pub(crate) members: Vec<ComputedCommunityMember>,
    pub(crate) projected_edge_count: usize,
    pub(crate) compute_mode: CommunityGroupingComputeMode,
}

impl CommunityGroupingComputed {
    pub fn projected_edge_count(&self) -> usize {
        self.projected_edge_count
    }
}

#[derive(Debug)]
pub(crate) struct ComputedCommunityMember {
    pub(crate) node_id: String,
    pub(crate) community_id: String,
    pub(crate) attachment: &'static str,
}

#[derive(Debug)]
pub struct CommunityGroupingReceipt {
    pub input_generation: i64,
    pub published_generation: i64,
    pub projected_edge_count: usize,
    pub member_count: usize,
    pub db_mutex_hold: Duration,
    pub composer_elapsed: Duration,
    pub input_rows_loaded: usize,
    pub member_rows_written: usize,
    pub compute_mode: CommunityGroupingComputeMode,
}

#[derive(Debug)]
pub enum CommunityGroupingOutcome {
    Published(CommunityGroupingReceipt),
    Stale(CommunityGroupingReceipt),
}

#[derive(Debug)]
struct RuntimeCommunityState {
    published_generation: i64,
    graph_generation: i64,
    algo_version: String,
    projection_version: String,
    partition: IncrementalPartitionState,
}

#[derive(Debug, Default)]
pub struct CommunityGroupingRuntime {
    spaces: BTreeMap<String, RuntimeCommunityState>,
}

impl CommunityGroupingRuntime {
    pub fn published_generation(&self, space: &str) -> Option<i64> {
        self.spaces
            .get(space)
            .map(|state| state.published_generation)
    }

    fn take(&mut self, space: &str) -> Option<RuntimeCommunityState> {
        self.spaces.remove(space)
    }

    fn install(&mut self, space: String, state: RuntimeCommunityState) {
        self.spaces.insert(space, state);
    }

    pub(crate) fn take_space_from(&mut self, other: &mut Self, space: &str) {
        if let Some(state) = other.take(space) {
            self.install(space.to_owned(), state);
        }
    }

    fn take_space_from_if_not_older(&mut self, other: &mut Self, space: &str) {
        let Some(candidate) = other.take(space) else {
            return;
        };
        let should_install = self
            .spaces
            .get(space)
            .is_none_or(|current| candidate.published_generation >= current.published_generation);
        if should_install {
            self.install(space.to_owned(), candidate);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommunityGroupingComputeMode {
    CoreReused,
    Incremental { optimized_nodes: usize },
    Full { reason: CommunityGroupingFullReason },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommunityGroupingFullReason {
    FirstPublication,
    RuntimeStateMissing,
    VersionChanged,
    PublishedGenerationMismatch,
    GraphGenerationRegressed,
    DirtyFrontierMissing,
    ThresholdExceeded,
    NodeOrderChanged,
    IncrementalRejected,
}

#[derive(Debug)]
enum RuntimeTransition {
    Reused {
        state: RuntimeCommunityState,
    },
    Incremental {
        state: RuntimeCommunityState,
        rollback: IncrementalPartitionRollback,
    },
    Full {
        prior: Option<RuntimeCommunityState>,
        next: RuntimeCommunityState,
    },
}

impl RuntimeTransition {
    fn publish(self, published_generation: i64, graph_generation: i64) -> RuntimeCommunityState {
        let mut state = match self {
            Self::Reused { state } | Self::Incremental { state, .. } => state,
            Self::Full { next, .. } => next,
        };
        state.published_generation = published_generation;
        state.graph_generation = graph_generation;
        state.algo_version = COMMUNITY_ALGO_VERSION.to_owned();
        state.projection_version = COMMUNITY_PROJECTION_VERSION.to_owned();
        state
    }

    fn restore(self) -> Option<RuntimeCommunityState> {
        match self {
            Self::Reused { state } => Some(state),
            Self::Incremental {
                mut state,
                rollback,
            } => {
                state.partition.restore(rollback);
                Some(state)
            }
            Self::Full { prior, .. } => prior,
        }
    }
}

struct RuntimeTransitionGuard<'a> {
    runtime: &'a mut CommunityGroupingRuntime,
    space: String,
    transition: Option<RuntimeTransition>,
}

impl<'a> RuntimeTransitionGuard<'a> {
    fn new(
        runtime: &'a mut CommunityGroupingRuntime,
        space: String,
        transition: RuntimeTransition,
    ) -> Self {
        Self {
            runtime,
            space,
            transition: Some(transition),
        }
    }

    fn publish(&mut self, published_generation: i64, graph_generation: i64) {
        if let Some(transition) = self.transition.take() {
            self.runtime.install(
                self.space.clone(),
                transition.publish(published_generation, graph_generation),
            );
        }
    }

    fn restore(&mut self) {
        if let Some(transition) = self.transition.take() {
            if let Some(state) = transition.restore() {
                self.runtime.install(self.space.clone(), state);
            }
        }
    }
}

impl Drop for RuntimeTransitionGuard<'_> {
    fn drop(&mut self) {
        self.restore();
    }
}

struct SharedRuntimeSpaceGuard<'a> {
    shared: &'a std::sync::Mutex<CommunityGroupingRuntime>,
    space: String,
    local: CommunityGroupingRuntime,
}

impl<'a> SharedRuntimeSpaceGuard<'a> {
    fn take(shared: &'a std::sync::Mutex<CommunityGroupingRuntime>, space: &str) -> Self {
        let mut local = CommunityGroupingRuntime::default();
        {
            let mut shared_runtime = shared
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            local.take_space_from(&mut shared_runtime, space);
        }
        Self {
            shared,
            space: space.to_owned(),
            local,
        }
    }

    fn runtime_mut(&mut self) -> &mut CommunityGroupingRuntime {
        &mut self.local
    }
}

impl Drop for SharedRuntimeSpaceGuard<'_> {
    fn drop(&mut self) {
        let mut shared = self
            .shared
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        shared.take_space_from_if_not_older(&mut self.local, &self.space);
    }
}

#[derive(Debug)]
pub struct CommunityGroupingWork {
    computed: CommunityGroupingComputed,
    transition: RuntimeTransition,
}

type GroupingSelection = (Vec<usize>, RuntimeTransition, CommunityGroupingComputeMode);
type GroupingSelectionFailure = Box<(CommunityGroupingError, Option<RuntimeCommunityState>)>;

impl CommunityGroupingWork {
    pub fn projected_edge_count(&self) -> usize {
        self.computed.projected_edge_count()
    }

    pub async fn finalize(
        self,
        db: &MemoryDB,
        attempt: CommunityGroupingAttempt,
        runtime: &mut CommunityGroupingRuntime,
    ) -> Result<CommunityGroupingOutcome, CommunityGroupingError> {
        let space = attempt.space.clone();
        let graph_generation = attempt.graph_generation;
        let input_generation = attempt.input_generation;
        let mut transition = RuntimeTransitionGuard::new(runtime, space, self.transition);
        match db.finalize_community_grouping(attempt, self.computed).await {
            Ok(outcome) => {
                if matches!(outcome, CommunityGroupingOutcome::Published(_)) {
                    transition.publish(input_generation, graph_generation);
                } else {
                    transition.restore();
                }
                Ok(outcome)
            }
            Err(error) => Err(error),
        }
    }
}

/// Pure graph computation. This function deliberately has no database handle.
pub fn compute_community_grouping(
    attempt: &CommunityGroupingAttempt,
    runtime: &mut CommunityGroupingRuntime,
) -> Result<CommunityGroupingWork, CommunityGroupingError> {
    let graph = project_grounded_relates(&attempt.edges, ProjectionConfig::default());
    let dirty_nodes = attempt
        .dirty_node_ids
        .iter()
        .filter_map(|node_id| graph.node_ids().binary_search(node_id).ok())
        .collect::<Vec<_>>();

    let prior = runtime.take(&attempt.space);
    let full = |prior: Option<RuntimeCommunityState>,
                reason: CommunityGroupingFullReason|
     -> Result<GroupingSelection, GroupingSelectionFailure> {
        let (membership, partition) = match full_partition_state(&graph) {
            Ok(full) => full,
            Err(error) => return Err(Box::new((error, prior))),
        };
        let next = RuntimeCommunityState {
            published_generation: attempt.published_generation.unwrap_or_default(),
            graph_generation: attempt.graph_generation,
            algo_version: COMMUNITY_ALGO_VERSION.to_owned(),
            projection_version: COMMUNITY_PROJECTION_VERSION.to_owned(),
            partition,
        };
        Ok((
            membership,
            RuntimeTransition::Full { prior, next },
            CommunityGroupingComputeMode::Full { reason },
        ))
    };

    let selection = match prior {
        None => full(
            None,
            if attempt.published_generation.is_none() {
                CommunityGroupingFullReason::FirstPublication
            } else {
                CommunityGroupingFullReason::RuntimeStateMissing
            },
        ),
        Some(prior) => {
            let published_versions_match =
                attempt
                    .published_versions
                    .as_ref()
                    .is_none_or(|(algo, projection)| {
                        algo == COMMUNITY_ALGO_VERSION && projection == COMMUNITY_PROJECTION_VERSION
                    });
            if prior.algo_version != COMMUNITY_ALGO_VERSION
                || prior.projection_version != COMMUNITY_PROJECTION_VERSION
                || !published_versions_match
            {
                full(Some(prior), CommunityGroupingFullReason::VersionChanged)
            } else if Some(prior.published_generation) != attempt.published_generation {
                full(
                    Some(prior),
                    CommunityGroupingFullReason::PublishedGenerationMismatch,
                )
            } else if attempt.graph_generation < prior.graph_generation {
                full(
                    Some(prior),
                    CommunityGroupingFullReason::GraphGenerationRegressed,
                )
            } else if attempt.graph_generation == prior.graph_generation {
                let membership = prior.partition.partition().membership().to_vec();
                Ok((
                    membership,
                    RuntimeTransition::Reused { state: prior },
                    CommunityGroupingComputeMode::CoreReused,
                ))
            } else if dirty_nodes.is_empty() {
                full(
                    Some(prior),
                    if attempt.dirty_node_ids.is_empty() {
                        CommunityGroupingFullReason::DirtyFrontierMissing
                    } else {
                        CommunityGroupingFullReason::NodeOrderChanged
                    },
                )
            } else {
                let RuntimeCommunityState {
                    published_generation,
                    graph_generation,
                    algo_version,
                    projection_version,
                    partition,
                } = prior;
                match incremental_partition_recoverable(
                    &graph,
                    partition,
                    &dirty_nodes,
                    IncrementalConfig::default(),
                ) {
                    Ok((output, rollback)) => {
                        let optimized_nodes = output.optimized_nodes().len();
                        let membership = output.partition().membership().to_vec();
                        let state = RuntimeCommunityState {
                            published_generation,
                            graph_generation,
                            algo_version,
                            projection_version,
                            partition: output.into_state(),
                        };
                        Ok((
                            membership,
                            RuntimeTransition::Incremental { state, rollback },
                            CommunityGroupingComputeMode::Incremental { optimized_nodes },
                        ))
                    }
                    Err(failure) => {
                        let (error, partition) = *failure;
                        full(
                            Some(RuntimeCommunityState {
                                published_generation,
                                graph_generation,
                                algo_version,
                                projection_version,
                                partition,
                            }),
                            full_reason_for_incremental_error(&error),
                        )
                    }
                }
            }
        }
    };
    let (membership, transition, compute_mode) = match selection {
        Ok(selection) => selection,
        Err(failure) => {
            let (error, prior) = *failure;
            if let Some(prior) = prior {
                runtime.install(attempt.space.clone(), prior);
            }
            return Err(error);
        }
    };

    let members = if graph.node_ids().len() < MIN_COMMUNITY_PARTICIPANTS {
        Vec::new()
    } else {
        let previous_ids = graph
            .node_ids()
            .iter()
            .map(|node_id| {
                attempt
                    .previous_ids
                    .get(node_id)
                    .cloned()
                    .unwrap_or_else(|| format!("__m4-new-node-{node_id}"))
            })
            .collect::<Vec<_>>();
        let rebound = rebind_durable_ids_weighted(&previous_ids, &membership, &graph);
        let mut minted = BTreeMap::<String, String>::new();
        let mut core_members = graph
            .node_ids()
            .iter()
            .zip(rebound)
            .map(|(node_id, rebound_id)| {
                let community_id = if rebound_id.starts_with("__m4-new-node-")
                    || rebound_id.starts_with("community-m4-new-")
                {
                    minted
                        .entry(rebound_id)
                        .or_insert_with(|| uuid::Uuid::new_v4().to_string())
                        .clone()
                } else {
                    rebound_id
                };
                ComputedCommunityMember {
                    node_id: node_id.clone(),
                    community_id,
                    attachment: "core",
                }
            })
            .collect::<Vec<_>>();
        let isolated = posthoc_isolated_attachments(
            &core_members,
            &attempt.ungrounded_edges,
            &attempt.entity_embeddings,
        );
        core_members.extend(isolated);
        core_members
    };

    Ok(CommunityGroupingWork {
        computed: CommunityGroupingComputed {
            members,
            projected_edge_count: attempt.edges.len(),
            compute_mode,
        },
        transition,
    })
}

fn full_reason_for_incremental_error(
    error: &IncrementalPartitionError,
) -> CommunityGroupingFullReason {
    match error {
        IncrementalPartitionError::DirtyFraction { .. }
        | IncrementalPartitionError::FrontierFraction { .. } => {
            CommunityGroupingFullReason::ThresholdExceeded
        }
        IncrementalPartitionError::MembershipSize { .. }
        | IncrementalPartitionError::DirtyNode { .. }
        | IncrementalPartitionError::NodeOrder => CommunityGroupingFullReason::NodeOrderChanged,
        IncrementalPartitionError::DisconnectedCommunity => {
            CommunityGroupingFullReason::IncrementalRejected
        }
    }
}

fn posthoc_isolated_attachments(
    core_members: &[ComputedCommunityMember],
    ungrounded_edges: &[ProjectionInputEdge],
    entity_embeddings: &BTreeMap<String, Vec<f32>>,
) -> Vec<ComputedCommunityMember> {
    let core_community = core_members
        .iter()
        .map(|member| (member.node_id.clone(), member.community_id.clone()))
        .collect::<BTreeMap<_, _>>();
    let ungrounded = project_grounded_relates(ungrounded_edges, ProjectionConfig::default());
    let mut strongest_neighbor = BTreeMap::<String, (f64, String)>::new();
    for edge in ungrounded.edges() {
        let src = &ungrounded.node_ids()[edge.src];
        let dst = &ungrounded.node_ids()[edge.dst];
        update_strongest_neighbor(&mut strongest_neighbor, src, dst, edge.weight);
        update_strongest_neighbor(&mut strongest_neighbor, dst, src, edge.weight);
    }

    let centroids = community_centroids(&core_community, entity_embeddings);
    let ungrounded_communities = resolve_ungrounded_communities(
        entity_embeddings.keys().map(String::as_str),
        &strongest_neighbor,
        &core_community,
    );
    let mut attached = Vec::new();
    for (node_id, embedding) in entity_embeddings {
        if core_community.contains_key(node_id) {
            continue;
        }
        if let Some(community_id) = ungrounded_communities.get(node_id) {
            attached.push(ComputedCommunityMember {
                node_id: node_id.clone(),
                community_id: community_id.clone(),
                attachment: "isolated_ungrounded",
            });
            continue;
        }
        if let Some(community_id) = nearest_centroid(embedding, &centroids) {
            attached.push(ComputedCommunityMember {
                node_id: node_id.clone(),
                community_id,
                attachment: "isolated_embedding",
            });
        }
    }
    attached
}

fn update_strongest_neighbor(
    strongest: &mut BTreeMap<String, (f64, String)>,
    node_id: &str,
    neighbor_id: &str,
    weight: f64,
) {
    let candidate = (weight, neighbor_id.to_owned());
    let replace = strongest.get(node_id).is_none_or(|current| {
        candidate.0.total_cmp(&current.0).is_gt()
            || (candidate.0.total_cmp(&current.0).is_eq() && candidate.1 < current.1)
    });
    if replace {
        strongest.insert(node_id.to_owned(), candidate);
    }
}

fn resolve_ungrounded_communities<'a>(
    node_ids: impl IntoIterator<Item = &'a str>,
    strongest_neighbor: &BTreeMap<String, (f64, String)>,
    core_community: &BTreeMap<String, String>,
) -> BTreeMap<String, String> {
    let mut resolved = BTreeMap::<String, Option<String>>::new();
    for node_id in node_ids {
        if core_community.contains_key(node_id) || resolved.contains_key(node_id) {
            continue;
        }

        let mut path = Vec::new();
        let mut path_index = BTreeMap::<String, usize>::new();
        let mut cursor = node_id.to_owned();
        let result = loop {
            if let Some(community_id) = core_community.get(&cursor) {
                break Some(community_id.clone());
            }
            if let Some(cached) = resolved.get(&cursor) {
                break cached.clone();
            }
            if path_index.insert(cursor.clone(), path.len()).is_some() {
                break None;
            }
            path.push(cursor.clone());
            let Some((_, neighbor)) = strongest_neighbor.get(&cursor) else {
                break None;
            };
            cursor.clone_from(neighbor);
        };

        for visited in path {
            resolved.insert(visited, result.clone());
        }
    }

    resolved
        .into_iter()
        .filter_map(|(node_id, community_id)| community_id.map(|value| (node_id, value)))
        .collect()
}

fn community_centroids(
    core_community: &BTreeMap<String, String>,
    entity_embeddings: &BTreeMap<String, Vec<f32>>,
) -> BTreeMap<String, Vec<f64>> {
    let mut sums = BTreeMap::<String, (Vec<f64>, usize)>::new();
    for (node_id, community_id) in core_community {
        let Some(embedding) = entity_embeddings
            .get(node_id)
            .filter(|value| !value.is_empty())
        else {
            continue;
        };
        let entry = sums
            .entry(community_id.clone())
            .or_insert_with(|| (vec![0.0; embedding.len()], 0));
        if entry.0.len() != embedding.len() {
            continue;
        }
        for (sum, value) in entry.0.iter_mut().zip(embedding) {
            *sum += f64::from(*value);
        }
        entry.1 += 1;
    }
    sums.into_iter()
        .filter_map(|(community_id, (mut sum, count))| {
            if count == 0 {
                return None;
            }
            for value in &mut sum {
                *value /= count as f64;
            }
            Some((community_id, sum))
        })
        .collect()
}

fn nearest_centroid(embedding: &[f32], centroids: &BTreeMap<String, Vec<f64>>) -> Option<String> {
    if embedding.is_empty() {
        return None;
    }
    let mut best: Option<(f64, String)> = None;
    for (community_id, centroid) in centroids {
        if centroid.len() != embedding.len() {
            continue;
        }
        let dot = centroid
            .iter()
            .zip(embedding)
            .map(|(left, right)| left * f64::from(*right))
            .sum::<f64>();
        let left_norm = centroid.iter().map(|value| value * value).sum::<f64>();
        let right_norm = embedding
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>();
        if left_norm <= 0.0 || right_norm <= 0.0 {
            continue;
        }
        let cosine = dot / (left_norm.sqrt() * right_norm.sqrt());
        let replace = best.as_ref().is_none_or(|current| {
            cosine.total_cmp(&current.0).is_gt()
                || (cosine.total_cmp(&current.0).is_eq() && community_id < &current.1)
        });
        if replace {
            best = Some((cosine, community_id.clone()));
        }
    }
    best.map(|(_, community_id)| community_id)
}

fn full_partition_state(
    graph: &crate::community_partition::ProjectedGraph,
) -> Result<(Vec<usize>, IncrementalPartitionState), CommunityGroupingError> {
    let output = full_partition(graph, PartitionConfig::default())
        .map_err(|error| CommunityGroupingError::Compute(error.to_string()))?;
    let membership = output.membership().to_vec();
    let state = IncrementalPartitionState::new(graph, &membership)
        .map_err(|error| CommunityGroupingError::Compute(error.to_string()))?;
    Ok((membership, state))
}

/// Production composer. Runtime state is installed only after a successful
/// generation-CAS publication.
pub async fn run_community_grouping_cycle(
    db: &MemoryDB,
    runtime: &mut CommunityGroupingRuntime,
    space: &str,
) -> Result<CommunityGroupingOutcome, CommunityGroupingError> {
    let attempt = db.prepare_community_grouping(space).await?;
    let work = compute_community_grouping(&attempt, runtime)?;
    work.finalize(db, attempt, runtime).await
}

impl MemoryDB {
    /// Run at most one dirty space in the existing CommunityDetection phase
    /// slot. Runtime statistics are snapshotted before the async job and only
    /// the successfully published space is merged back afterward.
    pub async fn run_next_community_grouping_cycle(
        &self,
    ) -> Result<Option<CommunityGroupingOutcome>, CommunityGroupingError> {
        let space = {
            let conn = self.conn.lock().await;
            let mut rows = conn
                .query(
                    "SELECT space FROM space_graph_state \
                     WHERE dirty = 1 ORDER BY space LIMIT 1",
                    (),
                )
                .await
                .map_err(|error| {
                    CommunityGroupingError::Database(format!(
                        "select next dirty community space: {error}"
                    ))
                })?;
            rows.next()
                .await
                .map_err(|error| {
                    CommunityGroupingError::Database(format!(
                        "read next dirty community space: {error}"
                    ))
                })?
                .map(|row| {
                    row.get::<String>(0).map_err(|error| {
                        CommunityGroupingError::Database(format!(
                            "decode next dirty community space: {error}"
                        ))
                    })
                })
                .transpose()?
        };
        let Some(space) = space else {
            return Ok(None);
        };

        let attempt = match self.prepare_community_grouping(&space).await {
            Ok(attempt) => attempt,
            Err(
                CommunityGroupingError::LeaseHeld { .. } | CommunityGroupingError::NotDirty { .. },
            ) => return Ok(None),
            Err(error) => return Err(error),
        };
        let mut runtime = SharedRuntimeSpaceGuard::take(&self.community_grouping_runtime, &space);
        #[cfg(test)]
        crate::db::tests::community_grouping_test_hooks::after_runtime_take(&space).await;
        let work = compute_community_grouping(&attempt, runtime.runtime_mut())?;
        let result = work.finalize(self, attempt, runtime.runtime_mut()).await;
        match result {
            Ok(outcome) => Ok(Some(outcome)),
            Err(error) => Err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lease_cleanup_drop_without_tokio_runtime_falls_back_without_panicking() {
        let directory = tempfile::tempdir().expect("lease cleanup tempdir");
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("lease cleanup setup runtime");
        let (_db, connection) = runtime.block_on(async {
            let db = libsql::Builder::new_local(directory.path().join("lease-cleanup.db"))
                .build()
                .await
                .expect("lease cleanup database");
            let connection = db.connect().expect("lease cleanup connection");
            (db, connection)
        });
        drop(runtime);

        let cleanup = CommunityGroupingLeaseCleanup::new(
            Arc::new(tokio::sync::Mutex::new(connection)),
            "no-runtime-space".to_owned(),
            1,
            "no-runtime-token".to_owned(),
        );
        drop(cleanup);
    }
}
