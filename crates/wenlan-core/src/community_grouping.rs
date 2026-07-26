// SPDX-License-Identifier: Apache-2.0
//! M4 community grouping job: short DB phases around pure graph computation.

use std::{
    collections::{BTreeMap, BTreeSet},
    time::Duration,
};

use thiserror::Error;

use crate::{
    community_partition::{
        full_partition, incremental_partition, project_grounded_relates,
        rebind_durable_ids_weighted, IncrementalConfig, IncrementalPartitionState, PartitionConfig,
        ProjectionConfig, ProjectionInputEdge,
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
    #[error("community grouping database error: {0}")]
    Database(String),
    #[error("community grouping computation failed: {0}")]
    Compute(String),
}

#[derive(Debug)]
pub struct CommunityGroupingAttempt {
    pub(crate) space: String,
    pub(crate) input_generation: i64,
    pub(crate) published_generation: Option<i64>,
    pub(crate) token: String,
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
}

#[derive(Debug)]
pub struct CommunityGroupingComputed {
    pub(crate) members: Vec<ComputedCommunityMember>,
    pub(crate) projected_edge_count: usize,
    pub(crate) next_state: IncrementalPartitionState,
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
    pub(crate) next_state: Option<IncrementalPartitionState>,
}

#[derive(Debug)]
pub enum CommunityGroupingOutcome {
    Published(CommunityGroupingReceipt),
    Stale(CommunityGroupingReceipt),
}

#[derive(Debug, Clone)]
struct RuntimeCommunityState {
    published_generation: i64,
    partition: IncrementalPartitionState,
}

#[derive(Debug, Default, Clone)]
pub struct CommunityGroupingRuntime {
    spaces: BTreeMap<String, RuntimeCommunityState>,
}

impl CommunityGroupingRuntime {
    pub fn published_generation(&self, space: &str) -> Option<i64> {
        self.spaces
            .get(space)
            .map(|state| state.published_generation)
    }

    fn matching_partition(
        &self,
        space: &str,
        published_generation: Option<i64>,
    ) -> Option<IncrementalPartitionState> {
        let published_generation = published_generation?;
        self.spaces
            .get(space)
            .filter(|state| state.published_generation == published_generation)
            .map(|state| state.partition.clone())
    }

    fn install(
        &mut self,
        space: String,
        published_generation: i64,
        partition: IncrementalPartitionState,
    ) {
        self.spaces.insert(
            space,
            RuntimeCommunityState {
                published_generation,
                partition,
            },
        );
    }

    pub(crate) fn adopt_space_from(&mut self, other: &Self, space: &str) {
        if let Some(state) = other.spaces.get(space) {
            self.spaces.insert(space.to_owned(), state.clone());
        }
    }
}

/// Pure graph computation. This function deliberately has no database handle.
pub fn compute_community_grouping(
    attempt: &CommunityGroupingAttempt,
    runtime: &CommunityGroupingRuntime,
) -> Result<CommunityGroupingComputed, CommunityGroupingError> {
    let graph = project_grounded_relates(&attempt.edges, ProjectionConfig::default());
    let dirty_nodes = attempt
        .dirty_node_ids
        .iter()
        .filter_map(|node_id| graph.node_ids().binary_search(node_id).ok())
        .collect::<Vec<_>>();

    let (membership, next_state) = if let Some(prior) =
        runtime.matching_partition(&attempt.space, attempt.published_generation)
    {
        if dirty_nodes.is_empty() {
            full_partition_state(&graph)?
        } else {
            match incremental_partition(&graph, prior, &dirty_nodes, IncrementalConfig::default()) {
                Ok(output) => {
                    let membership = output.partition().membership().to_vec();
                    (membership, output.into_state())
                }
                Err(_) => full_partition_state(&graph)?,
            }
        }
    } else {
        full_partition_state(&graph)?
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

    Ok(CommunityGroupingComputed {
        members,
        projected_edge_count: attempt.edges.len(),
        next_state,
    })
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
    let mut attached = Vec::new();
    for (node_id, embedding) in entity_embeddings {
        if core_community.contains_key(node_id) {
            continue;
        }
        let ungrounded_community =
            resolve_ungrounded_community(node_id, &strongest_neighbor, &core_community);
        if let Some(community_id) = ungrounded_community {
            attached.push(ComputedCommunityMember {
                node_id: node_id.clone(),
                community_id,
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

fn resolve_ungrounded_community(
    node_id: &str,
    strongest_neighbor: &BTreeMap<String, (f64, String)>,
    core_community: &BTreeMap<String, String>,
) -> Option<String> {
    let mut cursor = node_id;
    let mut visited = BTreeSet::new();
    while visited.insert(cursor.to_owned()) {
        let (_, neighbor) = strongest_neighbor.get(cursor)?;
        if let Some(community_id) = core_community.get(neighbor) {
            return Some(community_id.clone());
        }
        cursor = neighbor;
    }
    None
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
    let computed = compute_community_grouping(&attempt, runtime)?;
    let outcome = db.finalize_community_grouping(attempt, computed).await?;
    if let CommunityGroupingOutcome::Published(receipt) = outcome {
        let mut receipt = receipt;
        if let Some(next_state) = receipt.next_state.take() {
            runtime.install(space.to_owned(), receipt.published_generation, next_state);
        }
        Ok(CommunityGroupingOutcome::Published(receipt))
    } else {
        Ok(outcome)
    }
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

        let mut runtime = self
            .community_grouping_runtime
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        let outcome = match run_community_grouping_cycle(self, &mut runtime, &space).await {
            Ok(outcome) => outcome,
            Err(CommunityGroupingError::LeaseHeld { .. }) => return Ok(None),
            Err(error) => return Err(error),
        };
        if matches!(outcome, CommunityGroupingOutcome::Published(_)) {
            self.community_grouping_runtime
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .adopt_space_from(&runtime, &space);
        }
        Ok(Some(outcome))
    }
}
